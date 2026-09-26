#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'Train an adapted Text2Bundle baseline for Agent Generative Recommendation.\n\nThis implementation preserves the core mechanics of Text2Bundle while adapting\nits user-item bundle generation setting to query -> tool-bundle generation:\n\n  1) EasyRec provides frozen text representations for queries and tools.\n  2) A Unified State Encoder (Transformer) represents query + selected tools.\n  3) Action Relation Modeling (Transformer) contextualizes candidate tools.\n  4) A learnable END action enables adaptive bundle length.\n  5) Recommendation pre-training uses pairwise BPR bundle completion.\n  6) Bundle generation is trained with Double-DQN and experience replay.\n\nImportant adaptation / fairness choices\n---------------------------------------\n* There is no persistent user ID/history in this task, so no user-ID embedding\n  or personalization reward is used.\n* The candidate pool mirrors the final AgentRec retrieval exposure: tools from\n  TOP-5 CF-retrieved historical bundles are concatenated with TOP-25 semantic\n  tool candidates, then de-duplicated in CF-first order. The stored training\n  context is used to avoid extra online retrieval calls during training.\n* Training may inject a missing gold tool into the TRAIN action pool so the\n  supervised/RL objective remains learnable. Validation NEVER injects gold;\n  therefore validation retains the real retrieval ceiling.\n* The target is used only as the training/validation label. No explanation,\n  prompt completion, or target_context_explanation is fed to the model.\n* EasyRec is used as a representation backbone only; no bundle-critic checkpoint\n  or bundle-preference supervision is loaded.\n\nExpected input JSONL fields (current generative_v9_sft schema):\n  qid, query, context, target, ...\n\nDefault data:\n  /root/yunxshi/NIPS2026/datasets/generative_v9_sft/sft_train_PartII.jsonl\n\nOutputs:\n  <output_dir>/\n    data_stats.json\n    split_summary.json\n    train_qids.txt\n    val_qids.txt\n    config.json\n    pretrain_history.json\n    rl_history.json\n    best_model.pt\n    final_model.pt\n    best_val_metrics.json\n    best_val_predictions.jsonl\n    final_val_metrics.json\n    final_val_predictions.jsonl\n\nThe validation split is grouped by qid and defaults to 10%. Validation also\nreports target-only, single-result ranked-recall metrics compatible with the\nmain RDCR protocol (missing ranks are zero).\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_DATA_PATH = str(AC_ROOT / 'datasets/generative_v9_sft/sft_train_PartII.jsonl')
DEFAULT_EASYREC_MODEL_DIR = str(AC_EASYREC_MODEL)
DEFAULT_EASYREC_CODE_DIR = str(AC_ROOT / 'models/easyrec')
DEFAULT_OUTPUT_DIR = str(AC_ROOT / 'outputs/text2bundle_adapted_train_seed42')

TOOL_EMPTY_TOKEN = "<TOOL_EMPTY>"
TOOL_SEP_TOKEN = "<TOOL_SEP>"
END_TOKEN = "<SPECIAL_END>"
END_ACTION = -1

# Complete double-bracket tool tokens or serialized <TOOL_...> tokens.
TOOL_TOKEN_RE = re.compile(r"<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>")
LLM_TOKEN_RE = re.compile(r"<LLM_[^<>\n\r]+>")

SECTION_NAMES = (
    "cf_retrieved_llm",
    "cf_retrieved_tool_bundle",
    "semantic_retrieved_llm",
    "semantic_retrieved_tool",
    "evidence",
)


# -----------------------------------------------------------------------------
# Reproducibility / IO
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception as exc:
                raise RuntimeError(f"Bad JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            if not str(obj.get("query") or "").strip():
                continue
            if "target" not in obj:
                continue
            obj = dict(obj)
            obj["_line_no"] = line_no
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"No usable rows found in {path}")
    return rows


# -----------------------------------------------------------------------------
# Tool / context parsing
# -----------------------------------------------------------------------------


def canonical_tool(token: str) -> str:
    """Map current tool serializations to one comparison identity.

    Examples:
      <<google-serper>> -> google-serper
      <TOOL_google-serper> -> google-serper
      <TOOL_EMPTY> -> ""
    """
    s = str(token or "").strip()
    if not s or s in {TOOL_EMPTY_TOKEN, TOOL_SEP_TOKEN, END_TOKEN}:
        return ""
    if s.startswith("<<") and s.endswith(">>"):
        s = s[2:-2]
    elif s.startswith("<TOOL_") and s.endswith(">"):
        s = s[len("<TOOL_"):-1]
    return " ".join(s.strip().split())


def parse_gold_agent(target: Any) -> Tuple[str, List[str], List[str]]:
    """Parse gold ONLY from target; return LLM, exact tool tokens, canonical tools."""
    text = str(target or "")
    llms = LLM_TOKEN_RE.findall(text)
    llm = llms[0] if llms else ""
    exact_tokens: List[str] = []
    tools: List[str] = []
    seen_tok = set()
    seen_tool = set()
    for tok in TOOL_TOKEN_RE.findall(text):
        if tok in {TOOL_EMPTY_TOKEN, TOOL_SEP_TOKEN, END_TOKEN}:
            continue
        if tok not in seen_tok:
            seen_tok.add(tok)
            exact_tokens.append(tok)
        tool = canonical_tool(tok)
        if tool and tool not in seen_tool:
            seen_tool.add(tool)
            tools.append(tool)
    return llm, exact_tokens, tools


def parse_gold_tools(target: Any) -> List[str]:
    return parse_gold_agent(target)[2]


def parse_cf_llms(context: str, depth: int) -> List[str]:
    section = extract_section(context, "cf_retrieved_llm")
    if not section or depth <= 0:
        return []
    out: List[str] = []
    seen = set()
    for tok in LLM_TOKEN_RE.findall(section):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= depth:
            break
    return out


def parse_cf_bundle_tool_tokens(context: str, depth: int) -> Tuple[List[str], int]:
    section = extract_section(context, "cf_retrieved_tool_bundle")
    if not section:
        return [], 0
    matches = list(re.finditer(r"\{(.*?)\}\s*\[\d+\]", section, flags=re.S))
    out: List[str] = []
    seen = set()
    for match in matches[:max(depth, 0)]:
        for tok in TOOL_TOKEN_RE.findall(match.group(1)):
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out, len(matches)


def parse_semantic_tool_tokens(context: str, depth: int) -> Tuple[List[str], int]:
    section = extract_section(context, "semantic_retrieved_tool")
    if not section:
        return [], 0
    all_tokens: List[str] = []
    seen_all = set()
    for tok in TOOL_TOKEN_RE.findall(section):
        if tok not in seen_all:
            seen_all.add(tok)
            all_tokens.append(tok)
    return all_tokens[:max(depth, 0)], len(all_tokens)


def extract_section(context: str, name: str) -> str:
    context = str(context or "")
    m = re.search(rf"(?:^|\n){re.escape(name)}:\s*\n", context)
    if not m:
        return ""
    start = m.end()
    end = len(context)
    for other in SECTION_NAMES:
        if other == name:
            continue
        m2 = re.search(rf"\n{re.escape(other)}:\s*\n", context[start:])
        if m2:
            end = min(end, start + m2.start())
    return context[start:end].strip()


def parse_cf_bundle_tools(context: str, depth: int) -> List[str]:
    tokens, _ = parse_cf_bundle_tool_tokens(context, depth)
    out: List[str] = []
    seen = set()
    for tok in tokens:
        tool = canonical_tool(tok)
        if tool and tool not in seen:
            seen.add(tool)
            out.append(tool)
    return out


def parse_semantic_tools(context: str, depth: int) -> List[str]:
    tokens, _ = parse_semantic_tool_tokens(context, depth)
    out: List[str] = []
    seen = set()
    for tok in tokens:
        tool = canonical_tool(tok)
        if tool and tool not in seen:
            seen.add(tool)
            out.append(tool)
    return out

def _clean_desc(text: str, max_chars: int = 600) -> str:
    text = " ".join(str(text or "").strip().split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def parse_intrinsic_tool_descriptions(context: str) -> Dict[str, str]:
    """Extract tool descriptions without query-specific rank/support metadata."""
    evidence = extract_section(context, "evidence")
    if not evidence:
        return {}
    out: Dict[str, str] = {}

    for line in evidence.splitlines():
        line = line.strip()
        if not line:
            continue

        # SEMANTIC-TOOL entries have token=... | ... | desc=...
        if "SEMANTIC-TOOL" in line and "token=" in line:
            tok_m = re.search(r"token=(<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>)", line)
            desc_m = re.search(r"(?:^|\|)\s*desc=(.*)$", line)
            if tok_m:
                tool = canonical_tool(tok_m.group(1))
                desc = _clean_desc(desc_m.group(1) if desc_m else "")
                if tool and desc and len(desc) > len(out.get(tool, "")):
                    out[tool] = desc

        # CF-TOOL-BUNDLE entries end with tools=TOKEN: desc; TOKEN: desc
        if "CF-TOOL-BUNDLE" in line and "tools=" in line:
            tool_part = line.split("tools=", 1)[1]
            token_matches = list(TOOL_TOKEN_RE.finditer(tool_part))
            for i, tm in enumerate(token_matches):
                tool = canonical_tool(tm.group(0))
                seg_start = tm.end()
                seg_end = token_matches[i + 1].start() if i + 1 < len(token_matches) else len(tool_part)
                seg = tool_part[seg_start:seg_end].strip(" :;|")
                desc = _clean_desc(seg)
                if tool and desc and len(desc) > len(out.get(tool, "")):
                    out[tool] = desc
    return out




def default_tool_token(tool: str) -> str:
    """Deterministic fallback serialization for the current agent grammar.

    Plain tool identities use <TOOL_...>. API/function phrases containing ``&&``
    use the native <<...>> form. A train-gold surface registry overrides this
    fallback whenever the tool appeared in training targets.
    """
    tool = str(tool or "").strip()
    if not tool:
        return ""
    return f"<<{tool}>>" if "&&" in tool else f"<TOOL_{tool}>"


def normalize_retrieval_surface_for_output(token: str) -> str:
    """Convert a retrieval-context token into the target/output grammar.

    The CF-bundle serializer may display plain tool ids as ``<<plain-tool>>``
    even when SFT targets serialize that same plain id as ``<TOOL_plain-tool>``.
    To avoid a format-only false miss under the target-only evaluator:
      * <TOOL_...> is preserved;
      * <<API&&Endpoint>> is preserved;
      * <<plain-tool>> becomes <TOOL_plain-tool>.
    Training-target surfaces, when available, take precedence elsewhere.
    """
    token = str(token or "").strip()
    if not token:
        return ""
    if token.startswith("<TOOL_") and token.endswith(">"):
        return token
    if token.startswith("<<") and token.endswith(">>"):
        inner = token[2:-2].strip()
        return token if "&&" in inner else f"<TOOL_{inner}>"
    return default_tool_token(canonical_tool(token))


def build_strict_agent_text(llm_token: str, tool_tokens: Sequence[str]) -> str:
    llm = str(llm_token or "").strip()
    tools = [str(x).strip() for x in tool_tokens if str(x).strip()]
    if not tools:
        tools = [TOOL_EMPTY_TOKEN]
    return " ".join([llm, TOOL_SEP_TOKEN] + tools + [END_TOKEN]).strip()

def tool_to_text(tool: str, desc: str = "") -> str:
    readable = tool.replace("&&", " | ").replace("_", " ")
    if desc:
        return f"Tool: {readable}. Description: {desc}"
    return f"Tool: {readable}."


@dataclass
class ParsedRow:
    row_index: int
    qid: str
    query: str
    gold_llm: str
    gold_tool_tokens: List[str]
    gold_tools: List[str]
    retrieved_llms: List[str]
    retrieved_tools: List[str]
    retrieved_tool_surfaces: Dict[str, str]
    cf_bundle_count_available: int
    semantic_tool_count_available: int
    descriptions: Dict[str, str]
    source_line: int



def parse_rows(
    rows: Sequence[Mapping[str, Any]],
    cf_bundle_depth: int,
    semantic_tool_depth: int,
    cf_llm_depth: int = 10,
) -> List[ParsedRow]:
    parsed: List[ParsedRow] = []
    for idx, row in enumerate(rows):
        query = str(row.get("query") or "").strip()
        qid = str(row.get("qid") or f"__row_{idx}")
        target = row.get("target")
        context = str(row.get("context") or "")
        gold_llm, gold_tool_tokens, gold = parse_gold_agent(target)
        llms = parse_cf_llms(context, cf_llm_depth)

        cf_tokens, cf_available = parse_cf_bundle_tool_tokens(context, cf_bundle_depth)
        sem_tokens, sem_available = parse_semantic_tool_tokens(context, semantic_tool_depth)
        merged_tokens = list(dict.fromkeys(cf_tokens + sem_tokens))
        retrieved: List[str] = []
        surfaces: Dict[str, str] = {}
        seen = set()
        for tok in merged_tokens:
            tool = canonical_tool(tok)
            if not tool:
                continue
            surfaces.setdefault(tool, normalize_retrieval_surface_for_output(tok))
            if tool not in seen:
                seen.add(tool)
                retrieved.append(tool)

        parsed.append(
            ParsedRow(
                row_index=idx,
                qid=qid,
                query=query,
                gold_llm=gold_llm,
                gold_tool_tokens=gold_tool_tokens,
                gold_tools=gold,
                retrieved_llms=llms,
                retrieved_tools=retrieved,
                retrieved_tool_surfaces=surfaces,
                cf_bundle_count_available=cf_available,
                semantic_tool_count_available=sem_available,
                descriptions=parse_intrinsic_tool_descriptions(context),
                source_line=int(row.get("_line_no") or -1),
            )
        )
    return parsed


# -----------------------------------------------------------------------------
# Split / prepared records
# -----------------------------------------------------------------------------


def grouped_split(
    rows: Sequence[ParsedRow], val_ratio: float, seed: int
) -> Tuple[List[ParsedRow], List[ParsedRow], List[str], List[str]]:
    by_qid: Dict[str, List[ParsedRow]] = {}
    for row in rows:
        by_qid.setdefault(row.qid, []).append(row)
    qids = list(by_qid)
    rng = random.Random(seed)
    rng.shuffle(qids)
    if len(qids) <= 1:
        n_val = 1 if qids else 0
    else:
        n_val = max(1, min(len(qids) - 1, int(round(len(qids) * val_ratio))))
    val_qids = set(qids[:n_val])
    train_rows: List[ParsedRow] = []
    val_rows: List[ParsedRow] = []
    for row in rows:
        (val_rows if row.qid in val_qids else train_rows).append(row)
    return train_rows, val_rows, [q for q in qids if q not in val_qids], list(qids[:n_val])


@dataclass
class PreparedRecord:
    row_index: int
    qid: str
    query_index: int
    candidate_tool_ids: List[int]
    gold_tool_ids: List[int]
    retrieved_tool_ids: List[int]
    gold_llm: str
    selected_llm: str
    gold_tool_tokens: List[str]
    candidate_surface_by_tool_id: Dict[int, str]
    injected_gold_count: int

    def local_gold_indices(self) -> List[int]:
        pos = {tid: i for i, tid in enumerate(self.candidate_tool_ids)}
        return [pos[tid] for tid in self.gold_tool_ids if tid in pos]

    def retrieved_gold_recall(self) -> float:
        gold = set(self.gold_tool_ids)
        if not gold:
            return 1.0
        return len(gold.intersection(self.retrieved_tool_ids)) / len(gold)


# -----------------------------------------------------------------------------
# EasyRec frozen embedding backbone
# -----------------------------------------------------------------------------


class EasyRecTextEncoder:
    def __init__(
        self,
        model_dir: str,
        easyrec_code_dir: str,
        device: torch.device,
        max_length: int,
        batch_size: int,
        fp16: bool,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_amp = bool(fp16 and device.type == "cuda")

        code_dir = Path(easyrec_code_dir)
        if not (code_dir / "model.py").is_file():
            raise FileNotFoundError(f"EasyRec model.py not found: {code_dir / 'model.py'}")
        if easyrec_code_dir not in sys.path:
            sys.path.insert(0, easyrec_code_dir)
        try:
            from model import Easyrec  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Cannot import Easyrec from {easyrec_code_dir}: {exc}") from exc

        try:
            from transformers import AutoConfig, AutoTokenizer  # lazy import: data dry-run does not require transformers
        except Exception as exc:
            raise RuntimeError("The transformers package is required for EasyRec encoding") from exc

        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        self.model = Easyrec.from_pretrained(model_dir, config=self.config, local_files_only=True)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        hidden = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        if hidden is None:
            raise RuntimeError("Cannot infer EasyRec hidden size")
        self.hidden_size = int(hidden)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], desc: str = "EasyRec") -> torch.Tensor:
        all_embs: List[torch.Tensor] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch_texts = list(texts[start:start + self.batch_size])
            tok = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model.encode(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
                emb = out.pooler_output.float()
                emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        if not all_embs:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        return torch.cat(all_embs, dim=0)


# -----------------------------------------------------------------------------
# Adapted Text2Bundle network
# -----------------------------------------------------------------------------


class Text2BundlePolicy(nn.Module):
    def __init__(
        self,
        text_dim: int,
        model_dim: int = 320,
        nhead: int = 2,
        state_layers: int = 1,
        action_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % nhead != 0:
            raise ValueError("model_dim must be divisible by nhead")
        self.model_dim = model_dim

        # Text embedding -> shared latent state/action space.
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.query_type = nn.Parameter(torch.zeros(model_dim))
        self.selected_type = nn.Parameter(torch.zeros(model_dim))

        state_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(state_layer, num_layers=state_layers)

        action_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(action_layer, num_layers=action_layers)
        self.end_action = nn.Parameter(torch.empty(model_dim))
        self.state_norm = nn.LayerNorm(model_dim)
        self.action_norm = nn.LayerNorm(model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_type, std=0.02)
        nn.init.normal_(self.selected_type, std=0.02)
        nn.init.normal_(self.end_action, std=0.02)

    def forward(
        self,
        query_emb: torch.Tensor,          # [B, H]
        candidate_emb: torch.Tensor,      # [B, C, H]
        candidate_mask: torch.Tensor,     # [B, C] True=real
        selected_mask: torch.Tensor,      # [B, C] True=already selected
    ) -> torch.Tensor:
        bsz, csz, _ = candidate_emb.shape
        q = self.text_proj(query_emb)  # [B,D]
        cand = self.text_proj(candidate_emb)  # [B,C,D]

        # Unified state = query + current selected bundle. Unselected candidate
        # positions are padding in the state Transformer.
        q_state = q + self.query_type
        selected_state = cand + self.selected_type.view(1, 1, -1)
        state_seq = torch.cat([q_state.unsqueeze(1), selected_state], dim=1)
        state_valid = torch.cat(
            [torch.ones((bsz, 1), dtype=torch.bool, device=query_emb.device), selected_mask],
            dim=1,
        )
        state_out = self.state_encoder(state_seq, src_key_padding_mask=~state_valid)
        weights = state_valid.float().unsqueeze(-1)
        state = (state_out * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        state = self.state_norm(state)

        # Action relation modeling over the candidate set.
        action_out = self.action_encoder(cand, src_key_padding_mask=~candidate_mask)
        action_out = self.action_norm(action_out)
        q_tools = torch.einsum("bd,bcd->bc", state, action_out) / math.sqrt(self.model_dim)
        q_end = torch.einsum("bd,d->b", state, self.end_action) / math.sqrt(self.model_dim)
        return torch.cat([q_tools, q_end.unsqueeze(1)], dim=1)  # [B,C+1]


# -----------------------------------------------------------------------------
# Record batching utilities
# -----------------------------------------------------------------------------


class RuntimeData:
    def __init__(
        self,
        query_embeddings: torch.Tensor,
        tool_embeddings: torch.Tensor,
        train_records: Sequence[PreparedRecord],
        val_records: Sequence[PreparedRecord],
    ) -> None:
        self.query_embeddings = query_embeddings.float().contiguous()
        self.tool_embeddings = tool_embeddings.float().contiguous()
        self.train_records = list(train_records)
        self.val_records = list(val_records)

    def record_batch(
        self,
        records: Sequence[PreparedRecord],
        record_indices: Sequence[int],
        selected_local: Sequence[Sequence[int]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        batch_records = [records[i] for i in record_indices]
        max_c = max((len(r.candidate_tool_ids) for r in batch_records), default=0)
        max_c = max(max_c, 1)  # Transformer cannot process zero-length candidate seq.

        q_emb = torch.stack([self.query_embeddings[r.query_index] for r in batch_records], dim=0)
        cand_emb = torch.zeros(
            (len(batch_records), max_c, self.tool_embeddings.shape[1]), dtype=torch.float32
        )
        cand_mask = torch.zeros((len(batch_records), max_c), dtype=torch.bool)
        sel_mask = torch.zeros((len(batch_records), max_c), dtype=torch.bool)
        lengths: List[int] = []

        for bi, (record, selected) in enumerate(zip(batch_records, selected_local)):
            c = len(record.candidate_tool_ids)
            lengths.append(c)
            if c:
                ids = torch.tensor(record.candidate_tool_ids, dtype=torch.long)
                cand_emb[bi, :c] = self.tool_embeddings.index_select(0, ids)
                cand_mask[bi, :c] = True
            else:
                # Keep one zero dummy token unmasked so Transformer attention never
                # receives an all-padding sequence. Local candidate length remains 0,
                # so this dummy action is never selectable.
                cand_mask[bi, 0] = True
            for local_idx in selected:
                if 0 <= int(local_idx) < c:
                    sel_mask[bi, int(local_idx)] = True

        return (
            q_emb.to(device, non_blocking=True),
            cand_emb.to(device, non_blocking=True),
            cand_mask.to(device, non_blocking=True),
            sel_mask.to(device, non_blocking=True),
            lengths,
        )


# -----------------------------------------------------------------------------
# Metrics / rollout
# -----------------------------------------------------------------------------


def set_metrics(pred: Sequence[int], gold: Sequence[int]) -> Tuple[float, float, float, bool, bool, bool]:
    p = set(pred)
    g = set(gold)
    inter = len(p & g)
    if not p:
        precision = 1.0 if not g else 0.0
    else:
        precision = inter / len(p)
    if not g:
        recall = 1.0 if not p else 0.0
    else:
        recall = inter / len(g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = p == g
    all_covered = g.issubset(p)
    any_hit = bool(p & g) or (not p and not g)
    return precision, recall, f1, exact, all_covered, any_hit


def mask_invalid_actions(
    q_values: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
    min_bundle_size: int = 0,
) -> torch.Tensor:
    # Tool actions invalid if padding or already selected. END is masked until
    # the minimum bundle size is reached, matching the main system's MIN_TOOLS.
    tool_q = q_values[:, :-1].masked_fill(~candidate_mask | selected_mask, -1e9)
    end_q = q_values[:, -1:].clone()
    if min_bundle_size > 0:
        too_short = selected_mask.sum(dim=1) < int(min_bundle_size)
        end_q = end_q.masked_fill(too_short.unsqueeze(1), -1e9)
    return torch.cat([tool_q, end_q], dim=1)


@torch.no_grad()
def evaluate_policy(
    model: Text2BundlePolicy,
    runtime: RuntimeData,
    records: Sequence[PreparedRecord],
    device: torch.device,
    max_bundle_size: int,
    tool_names: Sequence[str],
    min_bundle_size: int = 1,
    batch_size: int = 128,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Greedy validation with both tool diagnostics and main-protocol metrics.

    The main protocol treats Text2Bundle as a single-result baseline: one tool
    bundle is generated and paired with the Top-1 CF-retrieved LLM. Gold comes
    only from the row target. For RDCR@K, ranks 2..K are missing and therefore
    receive zero recall, exactly matching the existing baseline convention.
    """
    model.eval()
    ps: List[float] = []
    rs: List[float] = []
    f1s: List[float] = []
    exacts: List[float] = []
    covers: List[float] = []
    hits: List[float] = []
    pred_sizes: List[int] = []
    candidate_recalls: List[float] = []
    candidate_all: List[float] = []

    strict_tool_recalls: List[float] = []
    component_recalls: List[float] = []
    llm_hits: List[float] = []
    agent_complete_hits: List[float] = []
    predictions: List[Dict[str, Any]] = []

    def rd_single(value: float, k: int) -> float:
        denom = sum(1.0 / math.log2(r + 1) for r in range(1, k + 1))
        return value / denom if denom > 0 else 0.0

    for ridx, record in enumerate(tqdm(records, desc="validate", leave=False)):
        selected: List[int] = []
        for _ in range(max_bundle_size + 1):
            q, cand, cand_mask, sel_mask, lengths = runtime.record_batch(
                records, [ridx], [selected], device
            )
            qv = mask_invalid_actions(
                model(q, cand, cand_mask, sel_mask), cand_mask, sel_mask,
                min_bundle_size=min_bundle_size,
            )[0]
            c = lengths[0]
            padded_end = qv.shape[0] - 1
            valid_actions = [i for i in range(c) if i not in selected] + [padded_end]
            vals = torch.stack([qv[i] for i in valid_actions])
            chosen_col = valid_actions[int(torch.argmax(vals).item())]
            if chosen_col == padded_end:
                break
            if chosen_col in selected or chosen_col >= c:
                break
            selected.append(chosen_col)
            if len(selected) >= max_bundle_size:
                break

        pred_tool_ids = [record.candidate_tool_ids[i] for i in selected]
        gold_tool_ids = list(record.gold_tool_ids)
        p, r, f1, exact, all_cov, any_hit = set_metrics(pred_tool_ids, gold_tool_ids)
        ps.append(p); rs.append(r); f1s.append(f1)
        exacts.append(float(exact)); covers.append(float(all_cov)); hits.append(float(any_hit))
        pred_sizes.append(len(pred_tool_ids))

        retrieved = set(record.retrieved_tool_ids)
        gold = set(gold_tool_ids)
        cand_recall = 1.0 if not gold else len(retrieved & gold) / len(gold)
        candidate_recalls.append(cand_recall)
        candidate_all.append(float(gold.issubset(retrieved)))

        pred_tool_tokens = [
            record.candidate_surface_by_tool_id.get(tid, default_tool_token(tool_names[tid]))
            for tid in pred_tool_ids
        ]
        gold_tool_tokens = list(record.gold_tool_tokens)
        pred_tok_set = set(pred_tool_tokens)
        gold_tok_set = set(gold_tool_tokens)
        strict_tool_recall = (
            1.0 if not gold_tok_set else len(pred_tok_set & gold_tok_set) / len(gold_tok_set)
        )
        strict_tool_complete = gold_tok_set.issubset(pred_tok_set)
        llm_hit = bool(record.selected_llm and record.selected_llm == record.gold_llm)
        denom_comp = 1 + len(gold_tok_set)
        component_recall = (float(llm_hit) + len(pred_tok_set & gold_tok_set)) / max(denom_comp, 1)
        agent_complete = bool(llm_hit and strict_tool_complete)

        strict_tool_recalls.append(strict_tool_recall)
        component_recalls.append(component_recall)
        llm_hits.append(float(llm_hit))
        agent_complete_hits.append(float(agent_complete))

        strict_text = build_strict_agent_text(record.selected_llm, pred_tool_tokens)
        predictions.append({
            "qid": record.qid,
            "selected_llm": record.selected_llm,
            "gold_llm": record.gold_llm,
            "predicted_tools": [tool_names[i] for i in pred_tool_ids],
            "gold_tools": [tool_names[i] for i in gold_tool_ids],
            "predicted_tool_tokens": pred_tool_tokens,
            "gold_tool_tokens": gold_tool_tokens,
            "strict_text": strict_text,
            "precision_identity": p,
            "recall_identity": r,
            "f1_identity": f1,
            "exact_match_identity": exact,
            "all_gold_covered_identity": all_cov,
            "strict_tool_recall": strict_tool_recall,
            "llm_hit": llm_hit,
            "component_recall": component_recall,
            "agent_complete_hit": agent_complete,
            "candidate_gold_recall_identity": cand_recall,
        })

    mean_component = float(np.mean(component_recalls)) if component_recalls else 0.0
    mean_strict_tool = float(np.mean(strict_tool_recalls)) if strict_tool_recalls else 0.0
    mean_complete = float(np.mean(agent_complete_hits)) if agent_complete_hits else 0.0
    metrics = {
        "samples": float(len(records)),
        # Text2Bundle/tool-level diagnostics on canonical tool identities.
        "precision": float(np.mean(ps)) if ps else 0.0,
        "recall": float(np.mean(rs)) if rs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "exact_match": float(np.mean(exacts)) if exacts else 0.0,
        "all_gold_covered": float(np.mean(covers)) if covers else 0.0,
        "any_hit": float(np.mean(hits)) if hits else 0.0,
        "avg_predicted_bundle_size": float(np.mean(pred_sizes)) if pred_sizes else 0.0,
        "retrieval_candidate_recall": float(np.mean(candidate_recalls)) if candidate_recalls else 0.0,
        "retrieval_all_gold_available": float(np.mean(candidate_all)) if candidate_all else 0.0,
        # Main target-only protocol diagnostics using exact serialized tokens.
        "top1_llm_accuracy": float(np.mean(llm_hits)) if llm_hits else 0.0,
        "top1_tool_recall": mean_strict_tool,
        "top1_component_recall": mean_component,
        "agent_complete_hit": mean_complete,
        "cr_hit@1": mean_complete,
        "cr_mrr@1": mean_complete,
        "rdcr@1": mean_component,
        "cr_hit@5": mean_complete,
        "cr_mrr@5": mean_complete,
        "rdcr@5": rd_single(mean_component, 5),
        "cr_hit@10": mean_complete,
        "cr_mrr@10": mean_complete,
        "rdcr@10": rd_single(mean_component, 10),
        "tool_hit@1": float(np.mean([float(x >= 1.0 - 1e-12) for x in strict_tool_recalls])) if strict_tool_recalls else 0.0,
        "rdtr@1": mean_strict_tool,
        "rdtr@5": rd_single(mean_strict_tool, 5),
        "rdtr@10": rd_single(mean_strict_tool, 10),
        "ranked_results_per_query": 1.0,
        "missing_rank_policy": "zero",
        "gold_source": "target_only",
    }
    return metrics, predictions


# -----------------------------------------------------------------------------
# BPR recommendation pre-training
# -----------------------------------------------------------------------------


@dataclass
class BPRExample:
    record_idx: int
    selected: Tuple[int, ...]
    positive_idx: int
    negative_idx: int



def make_bpr_examples(
    records: Sequence[PreparedRecord], max_bundle_size: int, seed: int, epoch: int
) -> List[BPRExample]:
    rng = random.Random(seed * 1000003 + epoch)
    examples: List[BPRExample] = []
    for ridx, record in enumerate(records):
        gold_local = record.local_gold_indices()
        if not gold_local:
            continue
        gold_set = set(gold_local)
        negatives = [i for i in range(len(record.candidate_tool_ids)) if i not in gold_set]
        if not negatives:
            continue
        order = list(gold_local)
        rng.shuffle(order)
        order = order[:max_bundle_size]
        selected: List[int] = []
        for pos in order:
            neg = rng.choice(negatives)
            examples.append(
                BPRExample(
                    record_idx=ridx,
                    selected=tuple(selected),
                    positive_idx=pos,
                    negative_idx=neg,
                )
            )
            selected.append(pos)
    rng.shuffle(examples)
    return examples


def train_bpr_epoch(
    model: Text2BundlePolicy,
    runtime: RuntimeData,
    records: Sequence[PreparedRecord],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    examples: Sequence[BPRExample],
    batch_size: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for start in tqdm(range(0, len(examples), batch_size), desc="BPR-pretrain", leave=False):
        batch = examples[start:start + batch_size]
        rec_ids = [x.record_idx for x in batch]
        selected = [x.selected for x in batch]
        q, cand, cmask, smask, _ = runtime.record_batch(records, rec_ids, selected, device)
        qv = model(q, cand, cmask, smask)
        pos = torch.tensor([x.positive_idx for x in batch], dtype=torch.long, device=device)
        neg = torch.tensor([x.negative_idx for x in batch], dtype=torch.long, device=device)
        rows = torch.arange(len(batch), device=device)
        pos_scores = qv[rows, pos]
        neg_scores = qv[rows, neg]
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        total_n += len(batch)
    return total_loss / max(total_n, 1)


# -----------------------------------------------------------------------------
# Double-DQN
# -----------------------------------------------------------------------------


@dataclass
class Transition:
    record_idx: int
    selected: Tuple[int, ...]
    action_idx: int       # local tool index, or candidate_count for END
    reward: float
    next_selected: Tuple[int, ...]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, x: Transition) -> None:
        self.data.append(x)

    def sample(self, n: int, rng: random.Random) -> List[Transition]:
        n = min(n, len(self.data))
        return rng.sample(list(self.data), n)


def precision_reward(selected_tool_ids: Sequence[int], gold_tool_ids: Sequence[int]) -> float:
    p = set(selected_tool_ids)
    g = set(gold_tool_ids)
    if not p:
        return 1.0 if not g else 0.0
    return len(p & g) / len(p)


def dense_action_reward(
    query_emb: torch.Tensor,
    action_emb: torch.Tensor,
    selected_embs: Optional[torch.Tensor],
    rel_weight: float,
    comp_weight: float,
) -> float:
    # EasyRec embeddings are L2-normalized, so dot products are cosine scores.
    # Keep the signed score rather than shifting it positive: this is closer to
    # Text2Bundle's dot-product personalization/complementarity shaping and
    # avoids rewarding every additional tool merely for increasing bundle size.
    rel = float(torch.dot(query_emb, action_emb).clamp(-1, 1).item())
    comp = 0.0
    if selected_embs is not None and selected_embs.numel() > 0:
        sims = torch.mv(selected_embs, action_emb).clamp(-1, 1)
        comp = float(sims.mean().item())
    return rel_weight * rel + comp_weight * comp


def dqn_update(
    online: Text2BundlePolicy,
    target: Text2BundlePolicy,
    runtime: RuntimeData,
    records: Sequence[PreparedRecord],
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    gamma: float,
    min_bundle_size: int,
    grad_clip: float,
) -> float:
    online.train()
    target.eval()

    rec_ids = [x.record_idx for x in transitions]
    selected = [x.selected for x in transitions]
    q, cand, cmask, smask, lengths = runtime.record_batch(records, rec_ids, selected, device)
    q_all = online(q, cand, cmask, smask)

    # END is stored as local index == candidate_count. In padded q_all END is at
    # final column max_c, so remap per example.
    max_c = q_all.shape[1] - 1
    action_cols = []
    for tr, c in zip(transitions, lengths):
        action_cols.append(max_c if tr.action_idx == c else tr.action_idx)
    action_cols_t = torch.tensor(action_cols, dtype=torch.long, device=device)
    rows_t = torch.arange(len(transitions), device=device)
    q_sa = q_all[rows_t, action_cols_t]

    rewards = torch.tensor([x.reward for x in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([x.done for x in transitions], dtype=torch.float32, device=device)

    with torch.no_grad():
        next_selected = [x.next_selected for x in transitions]
        nq, ncand, ncmask, nsmask, nlengths = runtime.record_batch(
            records, rec_ids, next_selected, device
        )
        next_online = mask_invalid_actions(
            online(nq, ncand, ncmask, nsmask), ncmask, nsmask, min_bundle_size=min_bundle_size
        )
        next_target = mask_invalid_actions(
            target(nq, ncand, ncmask, nsmask), ncmask, nsmask, min_bundle_size=min_bundle_size
        )
        padded_end = next_online.shape[1] - 1

        next_vals: List[torch.Tensor] = []
        for bi, c in enumerate(nlengths):
            # Valid tool columns [0,c), plus END at padded_end.
            tool_vals = next_online[bi, :c]
            end_val = next_online[bi, padded_end].view(1)
            valid_online = torch.cat([tool_vals, end_val], dim=0)
            best_local = int(torch.argmax(valid_online).item())
            if best_local == c:
                val = next_target[bi, padded_end]
            else:
                val = next_target[bi, best_local]
            next_vals.append(val)
        next_v = torch.stack(next_vals)
        td_target = rewards + gamma * (1.0 - dones) * next_v

    loss = F.smooth_l1_loss(q_sa, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def epsilon_by_step(step: int, total_steps: int, eps_start: float, eps_end: float, frac: float) -> float:
    decay_steps = max(1, int(total_steps * frac))
    t = min(step / decay_steps, 1.0)
    return eps_start + t * (eps_end - eps_start)


def run_training_episode(
    model: Text2BundlePolicy,
    runtime: RuntimeData,
    records: Sequence[PreparedRecord],
    ridx: int,
    device: torch.device,
    max_bundle_size: int,
    min_bundle_size: int,
    epsilon: float,
    main_reward_weight: float,
    rel_reward_weight: float,
    comp_reward_weight: float,
    rng: random.Random,
) -> Tuple[List[Transition], List[int]]:
    model.eval()
    record = records[ridx]
    selected: List[int] = []
    transitions: List[Transition] = []
    c = len(record.candidate_tool_ids)

    for _ in range(max_bundle_size + 1):
        q, cand, cmask, smask, lengths = runtime.record_batch(records, [ridx], [selected], device)
        with torch.no_grad():
            qv = mask_invalid_actions(
                model(q, cand, cmask, smask), cmask, smask, min_bundle_size=min_bundle_size
            )[0]
        c = lengths[0]
        valid_tool_actions = [i for i in range(c) if i not in selected]
        # END is a legal exploratory action only after MIN_TOOLS has been met.
        # If the retrieval pool is empty/exhausted, allow END as the only safe
        # fallback rather than crashing the episode.
        end_allowed = len(selected) >= min_bundle_size or not valid_tool_actions
        valid_actions = valid_tool_actions + ([c] if end_allowed else [])  # c denotes END.

        if rng.random() < epsilon:
            action = rng.choice(valid_actions)
        else:
            padded_end = qv.shape[0] - 1
            vals = [float(qv[i].item()) for i in valid_tool_actions]
            vals.append(float(qv[padded_end].item()))
            action = valid_actions[int(np.argmax(vals))]

        before = tuple(selected)
        reward = 0.0
        done = False

        if action == c:
            selected_tool_ids = [record.candidate_tool_ids[i] for i in selected]
            reward += main_reward_weight * precision_reward(selected_tool_ids, record.gold_tool_ids)
            done = True
        else:
            action_tid = record.candidate_tool_ids[action]
            q_emb_cpu = runtime.query_embeddings[record.query_index]
            a_emb_cpu = runtime.tool_embeddings[action_tid]
            selected_ids = [record.candidate_tool_ids[i] for i in selected]
            selected_embs = (
                runtime.tool_embeddings.index_select(0, torch.tensor(selected_ids, dtype=torch.long))
                if selected_ids else None
            )
            reward += dense_action_reward(
                q_emb_cpu,
                a_emb_cpu,
                selected_embs,
                rel_reward_weight,
                comp_reward_weight,
            )
            selected.append(action)
            if len(selected) >= max_bundle_size or len(selected) >= c:
                selected_tool_ids = [record.candidate_tool_ids[i] for i in selected]
                reward += main_reward_weight * precision_reward(selected_tool_ids, record.gold_tool_ids)
                done = True

        transitions.append(
            Transition(
                record_idx=ridx,
                selected=before,
                action_idx=action,
                reward=float(reward),
                next_selected=tuple(selected),
                done=done,
            )
        )
        if done:
            break
    return transitions, selected


# -----------------------------------------------------------------------------
# Dataset preparation and cache
# -----------------------------------------------------------------------------


def data_fingerprint(args: argparse.Namespace, parsed_rows: Sequence[ParsedRow]) -> str:
    p = Path(args.data_path)
    h = hashlib.sha256()
    h.update(str(p.resolve()).encode())
    if p.exists():
        st = p.stat()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    h.update(str(args.cf_llm_depth).encode())
    h.update(str(args.cf_bundle_depth).encode())
    h.update(str(args.semantic_tool_depth).encode())
    h.update(str(args.seed).encode())
    h.update(str(args.val_ratio).encode())
    h.update(str(len(parsed_rows)).encode())
    return h.hexdigest()[:20]


def build_vocab_and_records(
    train_rows: Sequence[ParsedRow],
    val_rows: Sequence[ParsedRow],
    inject_missing_gold_train: bool,
) -> Tuple[List[str], Dict[str, int], Dict[str, str], Dict[str, str], List[PreparedRecord], List[PreparedRecord], List[str]]:
    # Avoid validation-label leakage: intrinsic description representations are
    # built from TRAIN rows only. Validation-only tools fall back to their names.
    registry: Dict[str, str] = {}
    for row in train_rows:
        for tool, desc in row.descriptions.items():
            if len(desc) > len(registry.get(tool, "")):
                registry[tool] = desc

    # Learn the preferred serialization from TRAIN gold labels only. This keeps
    # <TOOL_x> vs <<x>> decisions out of the validation labels while matching the
    # dataset grammar for tools observed during training.
    surface_counts: Dict[str, Counter] = {}
    for row in train_rows:
        for tok in row.gold_tool_tokens:
            tool = canonical_tool(tok)
            if tool:
                surface_counts.setdefault(tool, Counter())[tok] += 1
    surface_registry: Dict[str, str] = {
        tool: counts.most_common(1)[0][0] for tool, counts in surface_counts.items()
    }

    tools: List[str] = []
    seen = set()
    for row in list(train_rows) + list(val_rows):
        for tool in row.retrieved_tools + row.gold_tools:
            if tool and tool not in seen:
                seen.add(tool)
                tools.append(tool)
    tool_to_id = {t: i for i, t in enumerate(tools)}

    query_texts: List[str] = []
    query_to_index: Dict[str, int] = {}
    all_rows = list(train_rows) + list(val_rows)
    for row in all_rows:
        key = row.query
        if key not in query_to_index:
            query_to_index[key] = len(query_texts)
            query_texts.append(key)

    def prep(row: ParsedRow, train: bool) -> PreparedRecord:
        retrieved_ids = [tool_to_id[t] for t in row.retrieved_tools if t in tool_to_id]
        candidate_ids = list(retrieved_ids)
        gold_ids = [tool_to_id[t] for t in row.gold_tools if t in tool_to_id]
        injected = 0
        if train and inject_missing_gold_train:
            present = set(candidate_ids)
            for tid in gold_ids:
                if tid not in present:
                    candidate_ids.append(tid)
                    present.add(tid)
                    injected += 1

        candidate_surfaces: Dict[int, str] = {}
        for tool, tid in ((t, tool_to_id[t]) for t in row.retrieved_tools if t in tool_to_id):
            candidate_surfaces[tid] = surface_registry.get(
                tool, row.retrieved_tool_surfaces.get(tool, default_tool_token(tool))
            )
        # Injected TRAIN gold actions use their gold token when available.
        if train:
            gold_surface_map = {canonical_tool(tok): tok for tok in row.gold_tool_tokens}
            for tool, tid in ((t, tool_to_id[t]) for t in row.gold_tools if t in tool_to_id):
                candidate_surfaces.setdefault(tid, surface_registry.get(tool, gold_surface_map.get(tool, default_tool_token(tool))))

        return PreparedRecord(
            row_index=row.row_index,
            qid=row.qid,
            query_index=query_to_index[row.query],
            candidate_tool_ids=candidate_ids,
            gold_tool_ids=gold_ids,
            retrieved_tool_ids=retrieved_ids,
            gold_llm=row.gold_llm,
            selected_llm=(row.retrieved_llms[0] if row.retrieved_llms else ""),
            gold_tool_tokens=list(row.gold_tool_tokens),
            candidate_surface_by_tool_id=candidate_surfaces,
            injected_gold_count=injected,
        )

    train_records = [prep(r, True) for r in train_rows]
    val_records = [prep(r, False) for r in val_rows]
    return tools, tool_to_id, registry, surface_registry, train_records, val_records, query_texts


def summarize_data(
    parsed_rows: Sequence[ParsedRow],
    train_records: Sequence[PreparedRecord],
    val_records: Sequence[PreparedRecord],
    *,
    requested_cf_bundle_depth: int,
    requested_semantic_tool_depth: int,
    requested_cf_llm_depth: int,
) -> Dict[str, Any]:
    def rec_stats(records: Sequence[PreparedRecord]) -> Dict[str, float]:
        if not records:
            return {}
        recalls = [r.retrieved_gold_recall() for r in records]
        return {
            "records": float(len(records)),
            "avg_candidate_size": float(np.mean([len(r.retrieved_tool_ids) for r in records])),
            "avg_gold_size": float(np.mean([len(r.gold_tool_ids) for r in records])),
            "retrieval_gold_recall": float(np.mean(recalls)),
            "retrieval_all_gold_available": float(np.mean([x >= 1.0 - 1e-12 for x in recalls])),
            "empty_gold_rate": float(np.mean([len(r.gold_tool_ids) == 0 for r in records])),
            "empty_candidate_rate": float(np.mean([len(r.retrieved_tool_ids) == 0 for r in records])),
            "avg_train_gold_injected": float(np.mean([r.injected_gold_count for r in records])),
        }
    def depth_stats(values: Sequence[int], requested: int) -> Dict[str, float]:
        vals = list(map(int, values))
        if not vals:
            return {
                "requested": float(requested),
                "mean_available": 0.0,
                "min_available": 0.0,
                "max_available": 0.0,
                "fraction_meeting_requested_depth": 0.0,
            }
        return {
            "requested": float(requested),
            "mean_available": float(np.mean(vals)),
            "min_available": float(min(vals)),
            "max_available": float(max(vals)),
            "fraction_meeting_requested_depth": float(
                np.mean([v >= requested for v in vals]) if requested > 0 else 1.0
            ),
        }

    cf_depth = depth_stats(
        [r.cf_bundle_count_available for r in parsed_rows], requested_cf_bundle_depth
    )
    sem_depth = depth_stats(
        [r.semantic_tool_count_available for r in parsed_rows], requested_semantic_tool_depth
    )
    llm_depth = depth_stats(
        [len(r.retrieved_llms) for r in parsed_rows], requested_cf_llm_depth
    )

    return {
        "parsed_records": len(parsed_rows),
        "train": rec_stats(train_records),
        "validation": rec_stats(val_records),
        "stored_retrieval_depth": {
            "cf_llm": llm_depth,
            "cf_tool_bundle": cf_depth,
            "semantic_tool": sem_depth,
        },
        "retrieval_alignment_warning": bool(
            cf_depth["fraction_meeting_requested_depth"] < 0.999999
            or sem_depth["fraction_meeting_requested_depth"] < 0.999999
            or llm_depth["fraction_meeting_requested_depth"] < 0.999999
        ),
    }


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: Text2BundlePolicy,
    args: argparse.Namespace,
    text_dim: int,
    metrics: Mapping[str, Any],
    stage: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "text_dim": text_dim,
            "model_dim": args.model_dim,
            "nhead": args.nhead,
            "state_layers": args.state_layers,
            "action_layers": args.action_layers,
            "dropout": args.dropout,
            "metrics": dict(metrics),
            "stage": stage,
        },
        path,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train adapted Text2Bundle for query -> tool-bundle generation")
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--easyrec_model_dir", default=DEFAULT_EASYREC_MODEL_DIR)
    ap.add_argument("--easyrec_code_dir", default=DEFAULT_EASYREC_CODE_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.10)

    # Retrieval exposure stored in the training JSONL context.
    ap.add_argument("--cf_llm_depth", type=int, default=10)
    ap.add_argument("--cf_bundle_depth", type=int, default=5)
    ap.add_argument("--semantic_tool_depth", type=int, default=25)
    ap.add_argument("--inject_missing_gold_train", action=argparse.BooleanOptionalAction, default=True)

    # EasyRec representation.
    ap.add_argument("--easyrec_max_length", type=int, default=256)
    ap.add_argument("--easyrec_batch_size", type=int, default=128)
    ap.add_argument("--fp16_easyrec", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse_embedding_cache", action=argparse.BooleanOptionalAction, default=True)

    # Text2Bundle architecture; paper uses 1 layer / 2 heads.
    ap.add_argument("--model_dim", type=int, default=320)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--state_layers", type=int, default=1)
    ap.add_argument("--action_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--max_bundle_size", type=int, default=6)
    ap.add_argument("--min_bundle_size", type=int, default=1)

    # Recommendation pre-training.
    ap.add_argument("--pretrain_epochs", type=int, default=5)
    ap.add_argument("--pretrain_batch_size", type=int, default=256)
    ap.add_argument("--pretrain_lr", type=float, default=5e-5)

    # Double-DQN; paper uses gamma=.99, replay=20k, batch=256.
    ap.add_argument("--rl_epochs", type=int, default=20)
    ap.add_argument("--rl_lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--replay_size", type=int, default=20000)
    ap.add_argument("--rl_batch_size", type=int, default=256)
    ap.add_argument("--learning_starts", type=int, default=512)
    ap.add_argument("--update_every", type=int, default=4)
    ap.add_argument("--target_update_steps", type=int, default=500)
    ap.add_argument("--epsilon_start", type=float, default=1.0)
    ap.add_argument("--epsilon_end", type=float, default=0.05)
    ap.add_argument("--epsilon_decay_fraction", type=float, default=0.70)

    # Adapted reward: precision main reward + query relevance + tool relation.
    ap.add_argument("--main_reward_weight", type=float, default=3.0)
    ap.add_argument("--rel_reward_weight", type=float, default=0.20)
    ap.add_argument("--comp_reward_weight", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dry_run_data", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val_ratio must be in (0,1)")
    if args.cf_llm_depth < 0 or args.cf_bundle_depth < 0 or args.semantic_tool_depth < 0:
        raise SystemExit("retrieval depths must be >= 0")
    if args.max_bundle_size < 1:
        raise SystemExit("--max_bundle_size must be >= 1")
    if args.min_bundle_size < 0 or args.min_bundle_size > args.max_bundle_size:
        raise SystemExit("--min_bundle_size must satisfy 0 <= min <= max")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", vars(args))

    print(f"[data] loading {args.data_path}")
    raw_rows = load_jsonl(args.data_path)
    parsed_rows = parse_rows(raw_rows, args.cf_bundle_depth, args.semantic_tool_depth, args.cf_llm_depth)
    train_rows, val_rows, train_qids, val_qids = grouped_split(parsed_rows, args.val_ratio, args.seed)

    tools, tool_to_id, desc_registry, surface_registry, train_records, val_records, query_texts = build_vocab_and_records(
        train_rows, val_rows, args.inject_missing_gold_train
    )
    write_json(output_dir / "tool_surface_registry.json", surface_registry)
    stats = summarize_data(
        parsed_rows, train_records, val_records,
        requested_cf_bundle_depth=args.cf_bundle_depth,
        requested_semantic_tool_depth=args.semantic_tool_depth,
        requested_cf_llm_depth=args.cf_llm_depth,
    )
    stats.update({
        "unique_tools": len(tools),
        "unique_query_texts": len(query_texts),
        "train_qids": len(set(train_qids)),
        "val_qids": len(set(val_qids)),
        "val_ratio_requested": args.val_ratio,
        "cf_llm_depth": args.cf_llm_depth,
        "cf_bundle_depth": args.cf_bundle_depth,
        "semantic_tool_depth": args.semantic_tool_depth,
        "train_gold_injection_enabled": bool(args.inject_missing_gold_train),
    })
    write_json(output_dir / "data_stats.json", stats)
    write_json(output_dir / "split_summary.json", {
        "seed": args.seed,
        "group_key": "qid",
        "train_records": len(train_rows),
        "validation_records": len(val_rows),
        "train_unique_qids": len(set(train_qids)),
        "validation_unique_qids": len(set(val_qids)),
        "validation_ratio_requested": args.val_ratio,
    })
    (output_dir / "train_qids.txt").write_text("\n".join(train_qids) + "\n", encoding="utf-8")
    (output_dir / "val_qids.txt").write_text("\n".join(val_qids) + "\n", encoding="utf-8")

    print("[data-stats]", json.dumps(stats, ensure_ascii=False, indent=2))
    if stats.get("retrieval_alignment_warning"):
        print(
            "[WARNING] Stored retrieval context does not meet one or more requested "
            "SOTA depths on every record. See data_stats.json -> stored_retrieval_depth. "
            "The parser never invents missing candidates; exact alignment requires "
            "refreshing those stored retrieval contexts.",
            flush=True,
        )
    if args.dry_run_data:
        print("[done] dry_run_data: parsing/splitting completed; model was not loaded")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    # ------------------------------------------------------------------
    # Frozen EasyRec embeddings, cached because they are deterministic.
    # ------------------------------------------------------------------
    fingerprint = data_fingerprint(args, parsed_rows)
    cache_path = output_dir / f"easyrec_embeddings_{fingerprint}.pt"
    cache = None
    if args.reuse_embedding_cache and cache_path.is_file():
        try:
            loaded = torch.load(cache_path, map_location="cpu")
            if loaded.get("fingerprint") == fingerprint and loaded.get("tools") == tools and loaded.get("query_texts") == query_texts:
                cache = loaded
                print(f"[cache] loaded {cache_path}")
        except Exception as exc:
            print(f"[cache] ignored invalid cache: {exc!r}")

    if cache is None:
        easyrec = EasyRecTextEncoder(
            model_dir=args.easyrec_model_dir,
            easyrec_code_dir=args.easyrec_code_dir,
            device=device,
            max_length=args.easyrec_max_length,
            batch_size=args.easyrec_batch_size,
            fp16=args.fp16_easyrec,
        )
        query_embeddings = easyrec.encode(query_texts, desc="EasyRec-query")
        tool_texts = [tool_to_text(t, desc_registry.get(t, "")) for t in tools]
        tool_embeddings = easyrec.encode(tool_texts, desc="EasyRec-tool")
        text_dim = easyrec.hidden_size
        cache = {
            "fingerprint": fingerprint,
            "tools": tools,
            "query_texts": query_texts,
            "query_embeddings": query_embeddings.half(),
            "tool_embeddings": tool_embeddings.half(),
            "text_dim": text_dim,
        }
        torch.save(cache, cache_path)
        print(f"[cache] saved {cache_path}")
        # Release EasyRec GPU memory before policy training.
        del easyrec
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        query_embeddings = cache["query_embeddings"].float()
        tool_embeddings = cache["tool_embeddings"].float()
        text_dim = int(cache["text_dim"])

    runtime = RuntimeData(query_embeddings, tool_embeddings, train_records, val_records)

    model = Text2BundlePolicy(
        text_dim=text_dim,
        model_dim=args.model_dim,
        nhead=args.nhead,
        state_layers=args.state_layers,
        action_layers=args.action_layers,
        dropout=args.dropout,
    ).to(device)

    # ------------------------------------------------------------------
    # Stage 1: BPR recommendation / bundle-completion pre-training.
    # ------------------------------------------------------------------
    pre_opt = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    pre_history: List[Dict[str, float]] = []
    best_pre_rdcr10 = -1.0
    best_pre_state = None

    for epoch in range(1, args.pretrain_epochs + 1):
        examples = make_bpr_examples(train_records, args.max_bundle_size, args.seed, epoch)
        loss = train_bpr_epoch(
            model,
            runtime,
            train_records,
            pre_opt,
            device,
            examples,
            args.pretrain_batch_size,
            args.grad_clip,
        )
        val_metrics, _ = evaluate_policy(
            model, runtime, val_records, device, args.max_bundle_size, tools,
            min_bundle_size=args.min_bundle_size,
        )
        row = {"epoch": epoch, "bpr_loss": loss, **val_metrics}
        pre_history.append(row)
        print("[pretrain]", json.dumps(row, ensure_ascii=False))
        if val_metrics["rdcr@10"] > best_pre_rdcr10:
            best_pre_rdcr10 = val_metrics["rdcr@10"]
            best_pre_state = copy.deepcopy(model.state_dict())

    if best_pre_state is not None:
        model.load_state_dict(best_pre_state)
    write_json(output_dir / "pretrain_history.json", pre_history)
    save_checkpoint(
        output_dir / "pretrained_model.pt",
        model,
        args,
        text_dim,
        {"best_pretrain_rdcr@10": best_pre_rdcr10},
        stage="bpr_pretrained",
    )

    # ------------------------------------------------------------------
    # Stage 2: Double-DQN sequential bundle generation.
    # ------------------------------------------------------------------
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    rl_opt = torch.optim.Adam(model.parameters(), lr=args.rl_lr)
    replay = ReplayBuffer(args.replay_size)
    rng = random.Random(args.seed + 2026)
    global_env_step = 0
    global_update = 0
    estimated_total_steps = max(1, args.rl_epochs * len(train_records) * args.max_bundle_size)
    best_rdcr10 = best_pre_rdcr10
    best_epoch = 0
    patience_left = args.patience
    rl_history: List[Dict[str, Any]] = []

    # Treat pre-trained policy as a candidate best checkpoint before RL.
    initial_metrics, initial_preds = evaluate_policy(
        model, runtime, val_records, device, args.max_bundle_size, tools,
        min_bundle_size=args.min_bundle_size,
    )
    save_checkpoint(output_dir / "best_model.pt", model, args, text_dim, initial_metrics, stage="pretrain")
    write_json(output_dir / "best_val_metrics.json", initial_metrics)
    write_jsonl(output_dir / "best_val_predictions.jsonl", initial_preds)

    for epoch in range(1, args.rl_epochs + 1):
        order = list(range(len(train_records)))
        rng.shuffle(order)
        epoch_rewards: List[float] = []
        epoch_losses: List[float] = []
        epoch_lengths: List[int] = []

        progress = tqdm(order, desc=f"RL {epoch}/{args.rl_epochs}")
        for ridx in progress:
            epsilon = epsilon_by_step(
                global_env_step,
                estimated_total_steps,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_fraction,
            )
            transitions, selected = run_training_episode(
                model,
                runtime,
                train_records,
                ridx,
                device,
                args.max_bundle_size,
                args.min_bundle_size,
                epsilon,
                args.main_reward_weight,
                args.rel_reward_weight,
                args.comp_reward_weight,
                rng,
            )
            episode_reward = sum(x.reward for x in transitions)
            epoch_rewards.append(episode_reward)
            epoch_lengths.append(len(selected))

            for tr in transitions:
                replay.add(tr)
                global_env_step += 1
                if len(replay) >= args.learning_starts and global_env_step % args.update_every == 0:
                    batch = replay.sample(args.rl_batch_size, rng)
                    loss = dqn_update(
                        model,
                        target_model,
                        runtime,
                        train_records,
                        rl_opt,
                        batch,
                        device,
                        args.gamma,
                        args.min_bundle_size,
                        args.grad_clip,
                    )
                    epoch_losses.append(loss)
                    global_update += 1
                    if global_update % args.target_update_steps == 0:
                        target_model.load_state_dict(model.state_dict())

            progress.set_postfix(
                eps=f"{epsilon:.3f}",
                replay=len(replay),
                loss=f"{np.mean(epoch_losses[-20:]):.4f}" if epoch_losses else "-",
            )

        val_metrics, val_preds = evaluate_policy(
            model,
            runtime,
            val_records,
            device,
            args.max_bundle_size,
            tools,
            min_bundle_size=args.min_bundle_size,
        )
        epoch_row: Dict[str, Any] = {
            "epoch": epoch,
            "global_env_step": global_env_step,
            "global_update": global_update,
            "epsilon": epsilon_by_step(
                global_env_step,
                estimated_total_steps,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_fraction,
            ),
            "replay_size": len(replay),
            "mean_episode_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "mean_td_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "mean_train_bundle_size": float(np.mean(epoch_lengths)) if epoch_lengths else 0.0,
            **val_metrics,
        }
        rl_history.append(epoch_row)
        write_json(output_dir / "rl_history.json", rl_history)
        print("[rl-val]", json.dumps(epoch_row, ensure_ascii=False, indent=2))

        if val_metrics["rdcr@10"] > best_rdcr10 + 1e-8:
            best_rdcr10 = val_metrics["rdcr@10"]
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(output_dir / "best_model.pt", model, args, text_dim, val_metrics, stage="double_dqn")
            write_json(output_dir / "best_val_metrics.json", {**val_metrics, "epoch": epoch})
            write_jsonl(output_dir / "best_val_predictions.jsonl", val_preds)
            print(f"[best] epoch={epoch} RDCR@10={best_rdcr10:.6f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] no validation RDCR@10 improvement for {args.patience} epochs")
                break

    final_metrics, final_preds = evaluate_policy(
        model, runtime, val_records, device, args.max_bundle_size, tools,
        min_bundle_size=args.min_bundle_size,
    )
    save_checkpoint(output_dir / "final_model.pt", model, args, text_dim, final_metrics, stage="final")
    write_json(output_dir / "final_val_metrics.json", final_metrics)
    write_jsonl(output_dir / "final_val_predictions.jsonl", final_preds)
    write_json(output_dir / "training_summary.json", {
        "best_val_rdcr@10": best_rdcr10,
        "best_rl_epoch": best_epoch,
        "final_val_metrics": final_metrics,
        "output_dir": str(output_dir.resolve()),
        "method": "Text2Bundle-adapted",
        "easyrec_role": "frozen query/tool text representation only",
        "validation_split": "10% grouped by qid" if abs(args.val_ratio - 0.10) < 1e-9 else f"{args.val_ratio:.3f} grouped by qid",
    })
    print(f"[done] best_val_RDCR@10={best_rdcr10:.6f}")
    print(f"[done] output={output_dir}")


if __name__ == "__main__":
    main()
