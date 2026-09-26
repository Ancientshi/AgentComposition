#!/usr/bin/env python3
"""Flask inference service for the retrieval-grounded EasyRec bundle critic.

Endpoints
---------
GET  /health
POST /v1/score
POST /v1/compare
POST /v1/rerank
POST /v1/analyze

The service intentionally reproduces the training-time candidate serialization.
Counterfactual metrics are computed on raw critic logits:
  marginal(tool) = s(q, A) - s(q, A \\ {tool})
  interaction(i,j) = s(q, A) - s(q, A \\ {i}) - s(q, A \\ {j})
                     + s(q, A \\ {i,j})
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from transformers import AutoConfig, AutoTokenizer

DEFAULT_CHECKPOINT_DIR = str(AC_ROOT / 'checkpoints/bundle_critic_easyrec_v2')
DEFAULT_EASYREC_CODE_DIR = str(AC_ROOT / 'models/easyrec')
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8012

TOKEN_RE = re.compile(r"(<LLM_[^>]+>|<<.*?>>)")
REF_RE = re.compile(r"\[(\d+)\]")
EVIDENCE_ITEM_RE = re.compile(
    r"(?:^|\n)\s*\[(\d+)\]\s+(.*?)(?=(?:\n\s*\[\d+\]\s)|\Z)", re.S
)
SECTION_NAMES = (
    "cf_retrieved_llm",
    "cf_retrieved_tool_bundle",
    "semantic_retrieved_llm",
    "semantic_retrieved_tool",
)


def compact_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    first = max_chars * 2 // 3
    last = max_chars - first
    return text[:first] + " ...[TRUNCATED]... " + text[-last:]


@dataclass
class RetrievalEntry:
    section: str
    ref: str
    rank: int
    raw: str
    tokens: Tuple[str, ...]


@dataclass
class ParsedEvidence:
    by_token: Dict[str, List[str]]
    by_ref: Dict[str, str]
    retrieval_by_token: Dict[str, List[RetrievalEntry]]
    bundle_entries: List[RetrievalEntry]


def _extract_section(text: str, section: str) -> str:
    start_match = re.search(rf"(?:^|\n){re.escape(section)}:\s*\n", text)
    if not start_match:
        return ""
    start = start_match.end()
    end = len(text)
    for next_name in (*SECTION_NAMES, "evidence"):
        if next_name == section:
            continue
        match = re.search(rf"\n{re.escape(next_name)}:\s*\n", text[start:])
        if match:
            end = min(end, start + match.start())
    return text[start:end].strip()


def parse_evidence_context(text: str) -> ParsedEvidence:
    text = text or ""
    by_ref: Dict[str, str] = {}
    by_token: Dict[str, List[str]] = {}

    evidence_start = re.search(r"(?:^|\n)evidence:\s*\n", text)
    evidence_text = text[evidence_start.end():] if evidence_start else ""
    for ref, body in EVIDENCE_ITEM_RE.findall(evidence_text):
        clean = " ".join(body.strip().split())
        by_ref[ref] = clean
        for token in TOKEN_RE.findall(clean):
            by_token.setdefault(token, []).append(clean)

    retrieval_by_token: Dict[str, List[RetrievalEntry]] = {}
    bundle_entries: List[RetrievalEntry] = []
    for section in SECTION_NAMES:
        body = _extract_section(text, section)
        if not body:
            continue
        matches = list(REF_RE.finditer(body))
        prev_end = 0
        for idx, match in enumerate(matches):
            raw = body[prev_end:match.start()].strip(" ,\n")
            ref = match.group(1)
            tokens = tuple(TOKEN_RE.findall(raw))
            entry = RetrievalEntry(section, ref, idx + 1, raw, tokens)
            for token in tokens:
                retrieval_by_token.setdefault(token, []).append(entry)
            if section == "cf_retrieved_tool_bundle":
                bundle_entries.append(entry)
            prev_end = match.end()

    for token, entries in retrieval_by_token.items():
        for entry in entries:
            if entry.ref in by_ref:
                by_token.setdefault(token, []).append(by_ref[entry.ref])
    for token, items in list(by_token.items()):
        by_token[token] = list(dict.fromkeys(items))

    return ParsedEvidence(by_token, by_ref, retrieval_by_token, bundle_entries)


def component_context(token: str, parsed: ParsedEvidence, max_chars: int) -> str:
    exact = parsed.by_token.get(token, [])
    retrieval = parsed.retrieval_by_token.get(token, [])
    parts: List[str] = []
    if exact:
        parts.append("Evidence: " + compact_text(exact[0], max_chars))
    else:
        parts.append("Evidence: <EXACT_COMPONENT_EVIDENCE_UNAVAILABLE>")
    if retrieval:
        metadata: List[str] = []
        seen = set()
        for entry in retrieval:
            key = (entry.section, entry.ref, entry.rank)
            if key in seen:
                continue
            seen.add(key)
            metadata.append(f"{entry.section}: rank={entry.rank}, ref=[{entry.ref}]")
        parts.append("Retrieval metadata: " + "; ".join(metadata[:4]))
    else:
        parts.append("Retrieval metadata: <NOT_FOUND_IN_RETRIEVAL_CONTEXT>")
    return "\n".join(parts)


def historical_bundle_context(
    tools: Sequence[str], parsed: ParsedEvidence, max_entries: int
) -> List[str]:
    selected = set(tools)
    scored: List[Tuple[int, int, RetrievalEntry]] = []
    for entry in parsed.bundle_entries:
        overlap = len(selected.intersection(entry.tokens))
        if overlap:
            scored.append((overlap, -entry.rank, entry))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    output = []
    for overlap, _, entry in scored[:max_entries]:
        output.append(
            f"overlap={overlap}/{max(len(selected), 1)}, retrieval_rank={entry.rank}, "
            f"ref=[{entry.ref}], bundle={entry.raw}"
        )
    return output


def serialize_candidate(
    query: str,
    candidate: Mapping[str, Any],
    parsed: ParsedEvidence,
    max_component_evidence_chars: int,
    max_bundle_evidence: int,
) -> str:
    llm = str(candidate.get("llm") or "<NO_LLM>")
    tools = [str(tool) for tool in (candidate.get("tools") or [])]
    parts = [
        "[QUERY]",
        query.strip(),
        "",
        "[CANDIDATE AGENT BUNDLE]",
        f"Selected LLM: {llm}",
        f"Number of tools: {len(tools)}",
        "Selected Tools:",
        *(f"- {tool}" for tool in tools),
        "",
        "[COMPONENT-ALIGNED EVIDENCE]",
        "[LLM EVIDENCE]",
        component_context(llm, parsed, max_component_evidence_chars),
    ]
    for idx, tool in enumerate(tools, 1):
        parts.extend(
            [
                "",
                f"[TOOL {idx} EVIDENCE]",
                f"Component: {tool}",
                component_context(tool, parsed, max_component_evidence_chars),
            ]
        )
    bundle_contexts = historical_bundle_context(tools, parsed, max_bundle_evidence)
    parts.extend(["", "[HISTORICAL BUNDLE EVIDENCE]"])
    if bundle_contexts:
        parts.extend(f"- {item}" for item in bundle_contexts)
    else:
        parts.append("- <NO_OVERLAPPING_RETRIEVED_BUNDLE>")
    return "\n".join(parts)


class EasyRecBundleCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        head_hidden: int,
        dropout: float,
        normalize_embedding: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.normalize_embedding = normalize_embedding
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder.encode(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        embedding = output.pooler_output.float()
        if self.normalize_embedding:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.scorer(self.encode(batch)).squeeze(-1)


class BundleCriticService:
    def __init__(
        self,
        checkpoint_dir: str,
        easyrec_code_dir: str,
        device_name: str,
        batch_size: int,
    ) -> None:
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir_path / "best_critic.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not (Path(easyrec_code_dir) / "model.py").is_file():
            raise FileNotFoundError(f"EasyRec model.py not found: {easyrec_code_dir}")

        sys.path.insert(0, easyrec_code_dir)
        from model import Easyrec  # type: ignore

        self.device = torch.device(
            device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.batch_size = max(1, batch_size)
        self.lock = threading.Lock()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        train_args = checkpoint.get("args", {})
        model_dir = str(train_args.get("model_dir") or "")
        if not model_dir or not Path(model_dir).is_dir():
            raise FileNotFoundError(
                "Base EasyRec model directory stored in checkpoint is unavailable: " + model_dir
            )

        tokenizer_dir = checkpoint_dir_path / "tokenizer"
        tokenizer_source = tokenizer_dir if tokenizer_dir.is_dir() else Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, use_fast=True, local_files_only=True
        )
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        encoder = Easyrec.from_pretrained(model_dir, config=config, local_files_only=True)
        hidden_size = int(
            checkpoint.get("hidden_size")
            or getattr(config, "hidden_size", 0)
            or getattr(config, "d_model", 0)
        )
        if hidden_size <= 0:
            raise ValueError("Cannot infer EasyRec hidden size")

        self.max_length = int(train_args.get("max_length", 768))
        self.max_component_evidence_chars = int(
            train_args.get("max_component_evidence_chars", 1500)
        )
        self.max_bundle_evidence = int(train_args.get("max_bundle_evidence", 3))
        self.model = EasyRecBundleCritic(
            encoder=encoder,
            hidden_size=hidden_size,
            head_hidden=int(train_args.get("head_hidden", 512)),
            dropout=float(train_args.get("dropout", 0.10)),
            normalize_embedding=bool(train_args.get("normalize_embedding", True)),
        )
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint state mismatch; missing={missing}, unexpected={unexpected}"
            )
        self.model.to(self.device)
        self.model.eval()
        self.metadata = {
            "checkpoint_dir": str(checkpoint_dir_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_list_ndcg": checkpoint.get("best_list_ndcg"),
            "base_model_dir": model_dir,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    @torch.inference_mode()
    def score_texts(self, texts: Sequence[str]) -> List[float]:
        scores: List[float] = []
        with self.lock:
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(texts[start:start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                batch_scores = self.model(encoded).detach().float().cpu().tolist()
                scores.extend(float(score) for score in batch_scores)
        return scores

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        evidence_context: str,
    ) -> Tuple[List[float], List[str]]:
        parsed = parse_evidence_context(evidence_context)
        texts = [
            serialize_candidate(
                query,
                candidate,
                parsed,
                self.max_component_evidence_chars,
                self.max_bundle_evidence,
            )
            for candidate in candidates
        ]
        return self.score_texts(texts), texts


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def require_json_object() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def require_query(payload: Mapping[str, Any]) -> str:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("Field 'query' is required")
    return query


def normalize_candidate(candidate: Mapping[str, Any], index: int = 0) -> Dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError("Each candidate must be a JSON object")
    llm = str(candidate.get("llm") or "").strip()
    if not llm:
        raise ValueError("Each candidate requires a non-empty 'llm'")
    raw_tools = candidate.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ValueError("Candidate field 'tools' must be a list")
    tools = list(dict.fromkeys(str(tool).strip() for tool in raw_tools if str(tool).strip()))
    return {
        "id": str(candidate.get("id") or f"candidate_{index}"),
        "llm": llm,
        "tools": tools,
    }


def create_app(service: BundleCriticService) -> Flask:
    app = Flask(__name__)

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        return jsonify({"error": "bad_request", "message": str(error)}), 400

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        app.logger.exception("Unhandled error")
        return jsonify({"error": "internal_error", "message": str(error)}), 500

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": service.metadata})

    @app.post("/v1/score")
    def score():
        payload = require_json_object()
        query = require_query(payload)
        candidate = normalize_candidate(payload, 0)
        evidence_context = str(payload.get("evidence_context") or "")
        scores, texts = service.score_candidates(query, [candidate], evidence_context)
        raw_score = scores[0]
        response: Dict[str, Any] = {
            "candidate": candidate,
            "raw_score": raw_score,
            "sigmoid_score": sigmoid(raw_score),
        }
        if bool(payload.get("return_serialized_text", False)):
            response["serialized_text"] = texts[0]
        return jsonify(response)

    @app.post("/v1/compare")
    def compare():
        payload = require_json_object()
        query = require_query(payload)
        left = normalize_candidate(payload.get("left") or {}, 0)
        right = normalize_candidate(payload.get("right") or {}, 1)
        evidence_context = str(payload.get("evidence_context") or "")
        scores, _ = service.score_candidates(query, [left, right], evidence_context)
        margin = scores[0] - scores[1]
        return jsonify(
            {
                "left": {**left, "raw_score": scores[0], "sigmoid_score": sigmoid(scores[0])},
                "right": {**right, "raw_score": scores[1], "sigmoid_score": sigmoid(scores[1])},
                "margin": margin,
                "preferred": left["id"] if margin > 0 else right["id"] if margin < 0 else "tie",
            }
        )

    @app.post("/v1/rerank")
    def rerank():
        payload = require_json_object()
        query = require_query(payload)
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("Field 'candidates' must be a non-empty list")
        candidates = [normalize_candidate(candidate, idx) for idx, candidate in enumerate(raw_candidates)]
        evidence_context = str(payload.get("evidence_context") or "")
        scores, _ = service.score_candidates(query, candidates, evidence_context)
        ranked = [
            {**candidate, "raw_score": score_value, "sigmoid_score": sigmoid(score_value)}
            for candidate, score_value in zip(candidates, scores)
        ]
        ranked.sort(key=lambda item: item["raw_score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        top_k = int(payload.get("top_k") or len(ranked))
        return jsonify({"query": query, "count": len(ranked), "ranking": ranked[:max(1, top_k)]})

    @app.post("/v1/analyze")
    def analyze():
        payload = require_json_object()
        query = require_query(payload)
        candidate = normalize_candidate(payload, 0)
        tools = candidate["tools"]
        evidence_context = str(payload.get("evidence_context") or "")
        include_pairwise = bool(payload.get("include_pairwise", True))
        max_pairs = max(0, int(payload.get("max_pairs", 100)))

        variants: List[Dict[str, Any]] = [candidate]
        keys: List[Tuple[str, Any]] = [("full", None)]
        for idx, tool in enumerate(tools):
            variants.append({"id": f"remove_{idx}", "llm": candidate["llm"], "tools": tools[:idx] + tools[idx + 1:]})
            keys.append(("remove_one", idx))

        pair_indices: List[Tuple[int, int]] = []
        if include_pairwise:
            for i in range(len(tools)):
                for j in range(i + 1, len(tools)):
                    if len(pair_indices) >= max_pairs:
                        break
                    pair_indices.append((i, j))
                if len(pair_indices) >= max_pairs:
                    break
            for i, j in pair_indices:
                pair_tools = [tool for idx, tool in enumerate(tools) if idx not in (i, j)]
                variants.append({"id": f"remove_{i}_{j}", "llm": candidate["llm"], "tools": pair_tools})
                keys.append(("remove_two", (i, j)))

        scores, _ = service.score_candidates(query, variants, evidence_context)
        score_map = {key: score_value for key, score_value in zip(keys, scores)}
        full_score = score_map[("full", None)]

        marginals = []
        for idx, tool in enumerate(tools):
            without_score = score_map[("remove_one", idx)]
            delta = full_score - without_score
            marginals.append(
                {
                    "component": tool,
                    "full_score": full_score,
                    "without_component_score": without_score,
                    "marginal_raw": delta,
                    "marginal_sigmoid": sigmoid(full_score) - sigmoid(without_score),
                    "effect": "positive" if delta > 0 else "negative" if delta < 0 else "neutral",
                }
            )
        marginals.sort(key=lambda item: item["marginal_raw"], reverse=True)

        interactions = []
        for i, j in pair_indices:
            remove_i = score_map[("remove_one", i)]
            remove_j = score_map[("remove_one", j)]
            remove_both = score_map[("remove_two", (i, j))]
            interaction = full_score - remove_i - remove_j + remove_both
            interactions.append(
                {
                    "component_i": tools[i],
                    "component_j": tools[j],
                    "score_full": full_score,
                    "score_without_i": remove_i,
                    "score_without_j": remove_j,
                    "score_without_both": remove_both,
                    "interaction_raw": interaction,
                    "effect": "synergy" if interaction > 0 else "redundancy_or_conflict" if interaction < 0 else "additive",
                }
            )
        interactions.sort(key=lambda item: item["interaction_raw"], reverse=True)

        positive_marginals = sum(item["marginal_raw"] > 0 for item in marginals)
        positive_interactions = sum(item["interaction_raw"] > 0 for item in interactions)
        summary = {
            "tool_count": len(tools),
            "full_raw_score": full_score,
            "full_sigmoid_score": sigmoid(full_score),
            "positive_component_rate": positive_marginals / max(len(marginals), 1),
            "mean_marginal_raw": sum(item["marginal_raw"] for item in marginals) / max(len(marginals), 1),
            "pair_count": len(interactions),
            "positive_synergy_rate": positive_interactions / max(len(interactions), 1) if interactions else None,
            "mean_interaction_raw": sum(item["interaction_raw"] for item in interactions) / len(interactions) if interactions else None,
        }
        return jsonify(
            {
                "query": query,
                "candidate": candidate,
                "summary": summary,
                "component_marginals": marginals,
                "pairwise_interactions": interactions,
                "metric_note": "Counterfactual metrics are critic-estimated associations, not causal effects.",
            }
        )

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--easyrec_code_dir", default=DEFAULT_EASYREC_CODE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.time()
    service = BundleCriticService(
        checkpoint_dir=args.checkpoint_dir,
        easyrec_code_dir=args.easyrec_code_dir,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print("[loaded]", json.dumps(service.metadata, ensure_ascii=False, indent=2))
    print(f"[loaded] seconds={time.time() - started:.2f}")
    app = create_app(service)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
