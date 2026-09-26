#!/usr/bin/env python3
"""Train a retrieval-grounded bundle critic with candidate-aligned evidence.

Expected JSONL fields:
  qid, query, evidence_context, candidates, ranking, optional scores

Key design choices:
  * Parse component-specific evidence from evidence_context.
  * Put query and candidate bundle before evidence to avoid truncating the bundle.
  * Exclude teacher candidate reasons/perturbation labels from model input.
  * Use score-gap-weighted pairwise ranking plus optional score calibration.
  * Evaluate both pairwise accuracy and query-level ranking metrics.
  * Save the complete model state, including an unfrozen encoder.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

# -------------------------------
# Project defaults: edit here once
# -------------------------------
DEFAULT_DATA_PATH = str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/bundle_preference_data.gpt54.jsonl')
DEFAULT_MODEL_DIR = str(AC_EASYREC_MODEL)
DEFAULT_OUTPUT_DIR = str(AC_ROOT / 'checkpoints/bundle_critic_easyrec_v2')
DEFAULT_EASYREC_CODE_DIR = str(AC_ROOT / 'models/easyrec')

DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.10
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_WORKERS = 4
DEFAULT_HEAD_HIDDEN = 512
DEFAULT_DROPOUT = 0.10
DEFAULT_HEAD_LR = 1e-4
DEFAULT_ENCODER_LR = 5e-6
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_PAIRS_PER_RECORD = 32
DEFAULT_AUX_SCORE_LOSS_WEIGHT = 0.10
DEFAULT_MAX_COMPONENT_EVIDENCE_CHARS = 1500
DEFAULT_MAX_BUNDLE_EVIDENCE = 3
DEFAULT_FREEZE_ENCODER = False
DEFAULT_FP16 = True
DEFAULT_PATIENCE = 2

TOKEN_RE = re.compile(r"(<LLM_[^>]+>|<<.*?>>)")
REF_RE = re.compile(r"\[(\d+)\]")
EVIDENCE_ITEM_RE = re.compile(r"(?:^|\n)\s*\[(\d+)\]\s+(.*?)(?=(?:\n\s*\[\d+\]\s)|\Z)", re.S)
SECTION_NAMES = (
    "cf_retrieved_llm",
    "cf_retrieved_tool_bundle",
    "semantic_retrieved_llm",
    "semantic_retrieved_tool",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Bad JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Line {line_no} is not a JSON object")
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"No valid records found in {path}")
    return rows


def compact_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # Preserve both metadata at the beginning and description tail.
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
        m = re.search(rf"\n{re.escape(next_name)}:\s*\n", text[start:])
        if m:
            end = min(end, start + m.start())
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
        tokens = TOKEN_RE.findall(clean)
        for token in tokens:
            by_token.setdefault(token, []).append(clean)

    retrieval_by_token: Dict[str, List[RetrievalEntry]] = {}
    bundle_entries: List[RetrievalEntry] = []
    for section in SECTION_NAMES:
        body = _extract_section(text, section)
        if not body:
            continue
        # Entries are comma-separated at top level in current data. Extract each
        # reference and use the text since the previous reference boundary.
        matches = list(REF_RE.finditer(body))
        prev_end = 0
        for idx, match in enumerate(matches):
            # Each retrieval item ends at its own [reference]. The text between
            # the previous reference and the current reference is this item.
            raw = body[prev_end:match.start()].strip(" ,\n")
            ref = match.group(1)
            tokens = tuple(TOKEN_RE.findall(raw))
            entry = RetrievalEntry(section=section, ref=ref, rank=idx + 1, raw=raw, tokens=tokens)
            for token in tokens:
                retrieval_by_token.setdefault(token, []).append(entry)
            if section == "cf_retrieved_tool_bundle":
                bundle_entries.append(entry)
            prev_end = match.end()

    # Link exact evidence by retrieval reference when token parsing is impossible.
    for token, entries in retrieval_by_token.items():
        for entry in entries:
            if entry.ref in by_ref:
                by_token.setdefault(token, []).append(by_ref[entry.ref])

    # De-duplicate while preserving order.
    for token, items in list(by_token.items()):
        by_token[token] = list(dict.fromkeys(items))

    return ParsedEvidence(
        by_token=by_token,
        by_ref=by_ref,
        retrieval_by_token=retrieval_by_token,
        bundle_entries=bundle_entries,
    )


def component_context(token: str, parsed: ParsedEvidence, max_chars: int) -> str:
    exact = parsed.by_token.get(token, [])
    retrieval = parsed.retrieval_by_token.get(token, [])
    parts: List[str] = []

    if exact:
        parts.append("Evidence: " + compact_text(exact[0], max_chars))
    else:
        parts.append("Evidence: <EXACT_COMPONENT_EVIDENCE_UNAVAILABLE>")

    if retrieval:
        meta = []
        seen = set()
        for entry in retrieval:
            key = (entry.section, entry.ref, entry.rank)
            if key in seen:
                continue
            seen.add(key)
            meta.append(f"{entry.section}: rank={entry.rank}, ref=[{entry.ref}]")
        parts.append("Retrieval metadata: " + "; ".join(meta[:4]))
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
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    output = []
    for overlap, _, entry in scored[:max_entries]:
        output.append(
            f"overlap={overlap}/{max(len(selected), 1)}, retrieval_rank={entry.rank}, "
            f"ref=[{entry.ref}], bundle={entry.raw}"
        )
    return output


def serialize_candidate(
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    parsed: ParsedEvidence,
    max_component_evidence_chars: int,
    max_bundle_evidence: int,
) -> str:
    query = str(record.get("query") or record.get("original_query") or "").strip()
    llm = str(candidate.get("llm") or "<NO_LLM>")
    tools = [str(x) for x in (candidate.get("tools") or [])]

    # Candidate comes before evidence so right-side tokenizer truncation cannot
    # erase the positive/negative distinction.
    parts = [
        "[QUERY]",
        query,
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
        parts.extend(f"- {x}" for x in bundle_contexts)
    else:
        parts.append("- <NO_OVERLAPPING_RETRIEVED_BUNDLE>")

    # Deliberately exclude candidate['reason'], perturbation, ranking and the
    # global teacher rationale: these are labels and would leak the answer.
    return "\n".join(parts)


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    valid = 0
    duplicate_candidate_records = 0
    score_coverage = 0
    missing_ranked_ids = 0
    exact_gold_present = 0
    component_count = 0
    component_with_exact_evidence = 0

    for row in rows:
        candidates = row.get("candidates") or []
        ranking = [str(x) for x in (row.get("ranking") or [])]
        cmap = {str(c.get("id")): c for c in candidates if c.get("id") is not None}
        if len(ranking) >= 2 and all(cid in cmap for cid in ranking):
            valid += 1
        missing_ranked_ids += sum(cid not in cmap for cid in ranking)
        if row.get("scores") and all(cid in row["scores"] for cid in ranking):
            score_coverage += 1

        signatures = []
        for cand in candidates:
            signatures.append((str(cand.get("llm")), tuple(sorted(map(str, cand.get("tools") or [])))))
        if len(signatures) != len(set(signatures)):
            duplicate_candidate_records += 1

        gold = row.get("gold") or {}
        gold_sig = (str(gold.get("llm")), tuple(sorted(map(str, gold.get("tools") or []))))
        if gold and gold_sig in signatures:
            exact_gold_present += 1

        parsed = parse_evidence_context(str(row.get("evidence_context") or ""))
        for cand in candidates:
            comps = [str(cand.get("llm"))] + [str(x) for x in (cand.get("tools") or [])]
            for comp in comps:
                component_count += 1
                if parsed.by_token.get(comp):
                    component_with_exact_evidence += 1

    n = max(len(rows), 1)
    return {
        "records": float(len(rows)),
        "valid_ranking_rate": valid / n,
        "score_coverage_rate": score_coverage / n,
        "duplicate_candidate_record_rate": duplicate_candidate_records / n,
        "exact_gold_present_rate": exact_gold_present / n,
        "missing_ranked_ids": float(missing_ranked_ids),
        "component_exact_evidence_rate": component_with_exact_evidence / max(component_count, 1),
    }


def split_by_qid(
    rows: Sequence[Dict[str, Any]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("qid", f"__row_{idx}"))
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = min(max(1, round(len(keys) * val_ratio)), max(len(keys) - 1, 1))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for key, group in groups.items():
        (val if key in val_keys else train).extend(group)
    return train, val


@dataclass
class PairExample:
    record: Dict[str, Any]
    positive: Dict[str, Any]
    negative: Dict[str, Any]
    positive_target: float
    negative_target: float
    weight: float


def _candidate_score(record: Mapping[str, Any], cid: str, rank_index: int, n: int) -> float:
    scores = record.get("scores") or {}
    if cid in scores:
        try:
            return float(scores[cid])
        except (TypeError, ValueError):
            pass
    return 1.0 - rank_index / max(n - 1, 1)


def build_pairs(
    rows: Sequence[Dict[str, Any]], max_pairs_per_record: int, seed: int
) -> List[PairExample]:
    all_pairs: List[PairExample] = []
    for row_idx, record in enumerate(rows):
        cmap = {str(c.get("id")): c for c in (record.get("candidates") or [])}
        ranking = [str(x) for x in (record.get("ranking") or []) if str(x) in cmap]
        n = len(ranking)
        if n < 2:
            continue

        scores = {cid: _candidate_score(record, cid, i, n) for i, cid in enumerate(ranking)}
        pair_indices: List[Tuple[int, int]] = []

        # Always keep adjacent hard pairs and top-vs-rest pairs.
        pair_indices.extend((i, i + 1) for i in range(n - 1))
        pair_indices.extend((0, j) for j in range(1, n))

        # Fill remaining budget with deterministic random pairs for broad coverage.
        remaining = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_indices = list(dict.fromkeys(pair_indices))
        used = set(pair_indices)
        remaining = [p for p in remaining if p not in used]
        rng = random.Random(seed * 1000003 + row_idx)
        rng.shuffle(remaining)
        pair_indices.extend(remaining)
        if max_pairs_per_record > 0:
            pair_indices = pair_indices[:max_pairs_per_record]

        for i, j in pair_indices:
            pos_id, neg_id = ranking[i], ranking[j]
            pos_target, neg_target = scores[pos_id], scores[neg_id]
            score_gap = max(pos_target - neg_target, 1e-3)
            # sqrt prevents a few easy pairs from dominating while still using
            # teacher confidence/rank distance.
            weight = math.sqrt(score_gap)
            all_pairs.append(
                PairExample(
                    record=record,
                    positive=cmap[pos_id],
                    negative=cmap[neg_id],
                    positive_target=pos_target,
                    negative_target=neg_target,
                    weight=weight,
                )
            )
    return all_pairs


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairExample],
        max_component_evidence_chars: int,
        max_bundle_evidence: int,
    ) -> None:
        self.pairs = list(pairs)
        self.max_component_evidence_chars = max_component_evidence_chars
        self.max_bundle_evidence = max_bundle_evidence
        self._parsed_cache: Dict[int, ParsedEvidence] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ex = self.pairs[idx]
        cache_key = id(ex.record)
        parsed = self._parsed_cache.get(cache_key)
        if parsed is None:
            parsed = parse_evidence_context(str(ex.record.get("evidence_context") or ""))
            self._parsed_cache[cache_key] = parsed
        positive_text = serialize_candidate(
            ex.record,
            ex.positive,
            parsed,
            self.max_component_evidence_chars,
            self.max_bundle_evidence,
        )
        negative_text = serialize_candidate(
            ex.record,
            ex.negative,
            parsed,
            self.max_component_evidence_chars,
            self.max_bundle_evidence,
        )
        return positive_text, negative_text, ex.positive_target, ex.negative_target, ex.weight


class PairCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pos_texts, neg_texts, pos_targets, neg_targets, weights = zip(*batch)
        common = dict(
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        pos_batch = self.tokenizer(list(pos_texts), **common)
        neg_batch = self.tokenizer(list(neg_texts), **common)
        return (
            pos_batch,
            neg_batch,
            torch.tensor(pos_targets, dtype=torch.float32),
            torch.tensor(neg_targets, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )


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
        out = self.encoder.encode(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        emb = out.pooler_output.float()
        if self.normalize_embedding:
            emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.scorer(self.encode(batch)).squeeze(-1)


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def ndcg_from_order(pred_order: Sequence[str], true_order: Sequence[str]) -> float:
    n = len(true_order)
    relevance = {cid: n - idx for idx, cid in enumerate(true_order)}
    dcg = sum((2 ** relevance[cid] - 1) / math.log2(i + 2) for i, cid in enumerate(pred_order))
    idcg = sum(
        (2 ** (n - i) - 1) / math.log2(i + 2)
        for i in range(n)
    )
    return dcg / max(idcg, 1e-12)


@torch.no_grad()
def evaluate_pairs(model, loader, device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    weighted_correct = 0.0
    total_weight = 0.0
    margins: List[float] = []
    for pos, neg, _, _, weights in tqdm(loader, desc="eval-pairs", leave=False):
        pos = move_batch(pos, device)
        neg = move_batch(neg, device)
        weights = weights.to(device)
        diff = model(pos) - model(neg)
        wins = (diff > 0).float()
        correct += int(wins.sum().item())
        total += diff.numel()
        weighted_correct += float((wins * weights).sum().item())
        total_weight += float(weights.sum().item())
        margins.extend(diff.detach().cpu().tolist())
    return {
        "pair_accuracy": correct / max(total, 1),
        "weighted_pair_accuracy": weighted_correct / max(total_weight, 1e-12),
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "pairs": float(total),
    }


@torch.no_grad()
def evaluate_lists(
    model,
    rows: Sequence[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    max_length: int,
    max_component_evidence_chars: int,
    max_bundle_evidence: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    top1 = 0
    mrr_values: List[float] = []
    ndcg_values: List[float] = []
    spearman_values: List[float] = []
    evaluated = 0

    for record in tqdm(rows, desc="eval-lists", leave=False):
        cmap = {str(c.get("id")): c for c in (record.get("candidates") or [])}
        ranking = [str(x) for x in (record.get("ranking") or []) if str(x) in cmap]
        if len(ranking) < 2:
            continue
        parsed = parse_evidence_context(str(record.get("evidence_context") or ""))
        texts = [
            serialize_candidate(
                record,
                cmap[cid],
                parsed,
                max_component_evidence_chars,
                max_bundle_evidence,
            )
            for cid in ranking
        ]
        scores: List[float] = []
        for start in range(0, len(texts), batch_size):
            tokenized = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = move_batch(tokenized, device)
            scores.extend(model(tokenized).detach().cpu().tolist())

        pred_indices = sorted(range(len(ranking)), key=lambda i: scores[i], reverse=True)
        pred_order = [ranking[i] for i in pred_indices]
        gold_id = ranking[0]
        top1 += int(pred_order[0] == gold_id)
        gold_rank = pred_order.index(gold_id) + 1
        mrr_values.append(1.0 / gold_rank)
        ndcg_values.append(ndcg_from_order(pred_order, ranking))

        # Spearman over rank positions, implemented without scipy.
        pred_pos = {cid: i for i, cid in enumerate(pred_order)}
        d2 = sum((i - pred_pos[cid]) ** 2 for i, cid in enumerate(ranking))
        n = len(ranking)
        spearman = 1.0 - 6.0 * d2 / max(n * (n * n - 1), 1)
        spearman_values.append(spearman)
        evaluated += 1

    return {
        "list_top1_accuracy": top1 / max(evaluated, 1),
        "list_mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        "list_ndcg": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "list_spearman": float(np.mean(spearman_values)) if spearman_values else 0.0,
        "list_records": float(evaluated),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--easyrec_code_dir", default=DEFAULT_EASYREC_CODE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum_steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max_component_evidence_chars", type=int, default=DEFAULT_MAX_COMPONENT_EVIDENCE_CHARS)
    parser.add_argument("--max_bundle_evidence", type=int, default=DEFAULT_MAX_BUNDLE_EVIDENCE)
    parser.add_argument("--head_hidden", type=int, default=DEFAULT_HEAD_HIDDEN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--head_lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--encoder_lr", type=float, default=DEFAULT_ENCODER_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max_pairs_per_record", type=int, default=DEFAULT_MAX_PAIRS_PER_RECORD)
    parser.add_argument("--aux_score_loss_weight", type=float, default=DEFAULT_AUX_SCORE_LOSS_WEIGHT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=DEFAULT_FREEZE_ENCODER)
    parser.add_argument("--normalize_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=DEFAULT_FP16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"EasyRec checkpoint not found: {args.model_dir}")
    if not (Path(args.easyrec_code_dir) / "model.py").is_file():
        raise FileNotFoundError(f"model.py not found under: {args.easyrec_code_dir}")

    sys.path.insert(0, args.easyrec_code_dir)
    try:
        from model import Easyrec
    except Exception as exc:
        raise RuntimeError(f"Cannot import Easyrec from {args.easyrec_code_dir}: {exc}") from exc

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    print("[config]", json.dumps(vars(args), ensure_ascii=False, indent=2))

    rows = load_jsonl(args.data_path)
    validation_stats = validate_rows(rows)
    print("[data-validation]", json.dumps(validation_stats, ensure_ascii=False, indent=2))
    with open(output_dir / "data_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_stats, f, ensure_ascii=False, indent=2)

    train_rows, val_rows = split_by_qid(rows, args.val_ratio, args.seed)
    train_pairs = build_pairs(train_rows, args.max_pairs_per_record, args.seed)
    val_pairs = build_pairs(val_rows, args.max_pairs_per_record, args.seed + 1)
    if not train_pairs or not val_pairs:
        raise RuntimeError("No train/validation pairs were constructed; check ranking and candidate IDs")
    print(
        f"[data] records={len(rows)} train_records={len(train_rows)} val_records={len(val_rows)} "
        f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}"
    )

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=True, local_files_only=True
    )
    encoder = Easyrec.from_pretrained(
        args.model_dir, config=config, local_files_only=True
    )
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    if hidden_size is None:
        raise ValueError("Cannot infer hidden_size from EasyRec config")

    model = EasyRecBundleCritic(
        encoder=encoder,
        hidden_size=int(hidden_size),
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        normalize_embedding=args.normalize_embedding,
    )
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        model.encoder.eval()
    model.to(device)

    collator = PairCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        PairDataset(train_pairs, args.max_component_evidence_chars, args.max_bundle_evidence),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, args.max_component_evidence_chars, args.max_bundle_evidence),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    parameter_groups = [{"params": model.scorer.parameters(), "lr": args.head_lr}]
    if not args.freeze_encoder:
        parameter_groups.insert(0, {"params": model.encoder.parameters(), "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = -float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    global_update = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_examples = 0
        progress = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")

        for step, (pos, neg, pos_targets, neg_targets, weights) in enumerate(progress, 1):
            pos = move_batch(pos, device)
            neg = move_batch(neg, device)
            pos_targets = pos_targets.to(device)
            neg_targets = neg_targets.to(device)
            weights = weights.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pos_scores = model(pos)
                neg_scores = model(neg)
                rank_loss_each = -F.logsigmoid(pos_scores - neg_scores)
                rank_loss = (rank_loss_each * weights).sum() / weights.sum().clamp_min(1e-8)
                calibration_loss = 0.5 * (
                    F.mse_loss(torch.sigmoid(pos_scores), pos_targets)
                    + F.mse_loss(torch.sigmoid(neg_scores), neg_targets)
                )
                loss = rank_loss + args.aux_score_loss_weight * calibration_loss
                scaled_loss = loss / max(args.grad_accum_steps, 1)

            scaler.scale(scaled_loss).backward()
            batch_n = pos_scores.numel()
            running_loss += float(loss.item()) * batch_n
            running_examples += batch_n

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

            progress.set_postfix(loss=f"{running_loss / max(running_examples, 1):.4f}")

        metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(running_examples, 1),
            "global_update": float(global_update),
        }
        metrics.update(evaluate_pairs(model, val_loader, device))
        metrics.update(
            evaluate_lists(
                model,
                val_rows,
                tokenizer,
                device,
                args.max_length,
                args.max_component_evidence_chars,
                args.max_bundle_evidence,
                args.batch_size,
            )
        )
        history.append(metrics)
        print("[val]", json.dumps(metrics, ensure_ascii=False, indent=2))

        # Query-level nDCG is the primary model-selection metric.
        selection_metric = metrics["list_ndcg"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "hidden_size": int(hidden_size),
                "best_list_ndcg": best_metric,
                "epoch": epoch,
            }
            torch.save(checkpoint, output_dir / "best_critic.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")
            with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[early-stop] no list_ndcg improvement for {args.patience} epochs")
                break

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_list_ndcg": best_metric,
                "history": history,
                "args": vars(args),
                "data_validation": validation_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] best_list_ndcg={best_metric:.6f}")
    print(f"[done] saved to {output_dir}")


if __name__ == "__main__":
    main()
