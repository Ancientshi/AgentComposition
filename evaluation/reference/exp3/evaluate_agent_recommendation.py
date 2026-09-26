#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-argument evaluation for Exp2 v13 bundle-aware agent recommendation.

Run:
    python -u evaluate_exp2_agent_recommendation.py

All paths and evaluation settings are intentionally fixed below.  The script
does not evaluate natural-language explanations.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import csv
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# Fixed experiment configuration -- no shell and no command-line arguments
# =============================================================================

PER_SAMPLE_DIR = Path(
    str(AC_ROOT / 'outputs/exp2_v13_valid_seed42_n1000/per_sample')
)
FULL_VALID_JSONL = Path(
    str(AC_ROOT / 'datasets/generative_v9_sft/sft_valid.jsonl')
)
OUTPUT_DIR = Path(
    str(AC_ROOT / 'outputs/exp2_v13_valid_seed42_n1000/') +
    "agent_recommendation_evaluation"
)

# Top-K recommendation metrics are computed from the complete, score-sorted
# final reranking pool. The user-facing ``results`` list is capped at Top-5,
# so it cannot by itself support correct @10/@20 evaluation.
TOP_KS = (1, 3, 5, 10, 20)
PRIMARY_K = 20
EXPECTED_RESULTS_PER_QUERY = 5
CASE_INSENSITIVE = False

LLM_RE = re.compile(r"<LLM_[^<>]+>")
DOUBLE_TOOL_RE = re.compile(r"<<[^<>]+>>")
SINGLE_TOOL_RE = re.compile(r"<TOOL_(?!SEP\b)[^<>]+>")
EMPTY_TOOL_NAMES = {"<TOOL_EMPTY>", "<TOOL_NONE>", "<TOOL_NULL>"}


# =============================================================================
# Generic helpers
# =============================================================================

def mean(xs: Iterable[float]) -> float:
    values = list(xs)
    return sum(values) / len(values) if values else 0.0


def safe_div(num: float, den: float, both_empty: float = 0.0) -> float:
    if den:
        return num / den
    return both_empty


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    values = sorted(float(x) for x in xs)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split())
    return text.casefold() if CASE_INSENSITIVE else text


def canonical_tools(tools: Iterable[Any]) -> Tuple[str, ...]:
    cleaned = {
        norm(tool)
        for tool in tools
        if norm(tool) and norm(tool) not in EMPTY_TOOL_NAMES
    }
    return tuple(sorted(cleaned))


def config_key(llm: Any, tools: Iterable[Any]) -> Tuple[str, Tuple[str, ...]]:
    return norm(llm), canonical_tools(tools)


def typed_components(agent: Tuple[str, Tuple[str, ...]]) -> set:
    llm, tools = agent
    result = {("llm", llm)} if llm else set()
    result.update(("tool", tool) for tool in tools)
    return result


def jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return safe_div(len(a & b), len(a | b))


def agent_jaccard(
    pred: Tuple[str, Tuple[str, ...]],
    gold: Tuple[str, Tuple[str, ...]],
) -> float:
    return jaccard_sets(typed_components(pred), typed_components(gold))


def tool_counts(
    pred: Tuple[str, Tuple[str, ...]],
    gold: Tuple[str, Tuple[str, ...]],
) -> Tuple[int, int, int]:
    p, g = set(pred[1]), set(gold[1])
    return len(p & g), len(p - g), len(g - p)


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # Same convention as Exp1: two empty sets receive P=R=F1=1.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def best_gold_alignment(
    pred: Tuple[str, Tuple[str, ...]],
    golds: Sequence[Tuple[str, Tuple[str, ...]]],
) -> Tuple[str, Tuple[str, ...]]:
    """Deterministic alignment used for micro counts under multiple positives."""
    if not golds:
        return ("", tuple())

    def score(gold: Tuple[str, Tuple[str, ...]]) -> Tuple:
        tp, fp, fn = tool_counts(pred, gold)
        _, _, tool_f1 = prf(tp, fp, fn)
        return (
            int(pred == gold),
            agent_jaccard(pred, gold),
            tool_f1,
            int(pred[0] == gold[0]),
            -abs(len(pred[1]) - len(gold[1])),
            gold,
        )

    return max(golds, key=score)


def seconds_between(start: Any, end: Any) -> Optional[float]:
    try:
        a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds())
    except Exception:
        return None


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")


def get_path(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


# =============================================================================
# Structure inspection and parsing
# =============================================================================

def structure_lines(
    obj: Any,
    name: str = "root",
    indent: str = "",
    depth: int = 0,
    max_depth: int = 5,
) -> List[str]:
    if isinstance(obj, Mapping):
        lines = [f"{indent}{name}: object[{len(obj)}]"]
        if depth < max_depth:
            for key, value in obj.items():
                lines.extend(
                    structure_lines(
                        value, str(key), indent + "  ", depth + 1, max_depth
                    )
                )
        return lines
    if isinstance(obj, list):
        lines = [f"{indent}{name}: array[{len(obj)}]"]
        if obj and depth < max_depth:
            lines.extend(
                structure_lines(
                    obj[0], "[0] representative", indent + "  ", depth + 1, max_depth
                )
            )
        return lines
    return [f"{indent}{name}: {type(obj).__name__}"]


def parse_agent_text(text: Any) -> Tuple[str, Tuple[str, ...]]:
    value = norm(text)
    llm_match = LLM_RE.search(value)
    llm = llm_match.group(0) if llm_match else ""
    tools = DOUBLE_TOOL_RE.findall(value) + SINGLE_TOOL_RE.findall(value)
    return config_key(llm, tools)


def extract_predictions(log: Mapping[str, Any]) -> List[Tuple[str, Tuple[str, ...]]]:
    # ``final_rerank_pool_before_cap`` is the complete final ranking, ordered by
    # the score used to select the reported Top-5. Prefer it so that @10/@20
    # are genuine ranking metrics. Fall back to the capped results for older
    # logs that do not contain the full pool.
    rows = get_path(
        log,
        ("generation", "search_trace", "final_rerank_pool_before_cap"),
        [],
    )
    if not isinstance(rows, list) or not rows:
        rows = log.get("results")
    if not isinstance(rows, list):
        rows = get_path(log, ("generation", "results"), [])
    parsed = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        llm = row.get("llm_token", row.get("llm", ""))
        tools = row.get("tool_tokens", row.get("tools", []))
        if isinstance(tools, str):
            tools = parse_agent_text(tools)[1]
        agent = config_key(llm, tools or [])
        if not agent[0]:
            agent = parse_agent_text(row.get("strict_text", row.get("gen_text", "")))
        parsed.append(agent)
    return parsed


def extract_stage_agents(
    log: Mapping[str, Any], field: str
) -> List[Tuple[str, Tuple[str, ...]]]:
    rows = get_path(log, ("generation", "search_trace", field), [])
    result = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        agent = config_key(
            row.get("llm", row.get("llm_token", "")),
            row.get("tools", row.get("tool_tokens", [])) or [],
        )
        if agent[0]:
            result.append(agent)
    return result


def candidate_token_list(rows: Any) -> List[str]:
    result = []
    for row in rows or []:
        value = row.get("token", row.get("component", "")) if isinstance(row, Mapping) else row
        if norm(value):
            result.append(norm(value))
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON must be an object")
    return obj


def input_paths() -> List[Path]:
    paths = sorted(PER_SAMPLE_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No *.json files found in fixed path: {PER_SAMPLE_DIR}")
    return paths


def load_full_gold() -> Tuple[Dict[str, List[Tuple[str, Tuple[str, ...]]]], Dict[str, Any]]:
    by_qid: Dict[str, set] = defaultdict(set)
    stats = {
        "path": str(FULL_VALID_JSONL),
        "available": FULL_VALID_JSONL.is_file(),
        "rows": 0,
        "valid_gold_rows": 0,
        "malformed_rows": 0,
    }
    if not FULL_VALID_JSONL.is_file():
        return {}, stats
    with FULL_VALID_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            stats["rows"] += 1
            try:
                row = json.loads(line)
                qid = norm(row.get("qid"))
                agent = parse_agent_text(row.get("target", ""))
                if qid and agent[0]:
                    by_qid[qid].add(agent)
                    stats["valid_gold_rows"] += 1
                else:
                    stats["malformed_rows"] += 1
            except Exception:
                stats["malformed_rows"] += 1
    return {qid: sorted(golds) for qid, golds in by_qid.items()}, stats


# =============================================================================
# Per-record evaluation
# =============================================================================

def ranked_metrics(
    preds: Sequence[Tuple[str, Tuple[str, ...]]],
    golds: Sequence[Tuple[str, Tuple[str, ...]]],
    k: int,
) -> Dict[str, float]:
    ranked = list(preds[:k])
    gold_set = set(golds)
    gold_bundles = {g[1] for g in golds}
    gold_llms = {g[0] for g in golds}

    exact_ranks = [i for i, p in enumerate(ranked, 1) if p in gold_set]
    bundle_ranks = [i for i, p in enumerate(ranked, 1) if p[1] in gold_bundles]
    llm_ranks = [i for i, p in enumerate(ranked, 1) if p[0] in gold_llms]

    best_tool_f1, best_agent_j = 0.0, 0.0
    graded = []
    for pred in ranked:
        similarities = [agent_jaccard(pred, gold) for gold in golds] or [0.0]
        graded.append(max(similarities))
        for gold in golds:
            tp, fp, fn = tool_counts(pred, gold)
            best_tool_f1 = max(best_tool_f1, prf(tp, fp, fn)[2])
            best_agent_j = max(best_agent_j, agent_jaccard(pred, gold))

    # Exact binary nDCG/AP with one-to-one matching of distinct gold configs.
    unmatched = set(golds)
    binary_rels = []
    for pred in ranked:
        if pred in unmatched:
            binary_rels.append(1)
            unmatched.remove(pred)
        else:
            binary_rels.append(0)
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(binary_rels, 1))
    ideal_hits = min(len(set(golds)), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    binary_ndcg = safe_div(dcg, idcg, both_empty=1.0 if not golds else 0.0)
    ap_num = sum(
        sum(binary_rels[:rank]) / rank
        for rank, rel in enumerate(binary_rels, 1)
        if rel
    )
    ap_den = min(len(set(golds)), k)
    average_precision = safe_div(ap_num, ap_den, both_empty=1.0 if not golds else 0.0)

    # Diagnostic only: ordering quality within the returned list.
    graded_dcg = sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(graded, 1)
    )
    graded_ideal = sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(sorted(graded, reverse=True), 1)
    )

    return {
        f"agent_hit@{k}": float(bool(exact_ranks)),
        f"agent_mrr@{k}": 1.0 / exact_ranks[0] if exact_ranks else 0.0,
        f"bundle_hit@{k}": float(bool(bundle_ranks)),
        f"bundle_mrr@{k}": 1.0 / bundle_ranks[0] if bundle_ranks else 0.0,
        f"llm_hit@{k}": float(bool(llm_ranks)),
        f"llm_mrr@{k}": 1.0 / llm_ranks[0] if llm_ranks else 0.0,
        f"best_tool_f1@{k}": best_tool_f1,
        f"best_agent_jaccard@{k}": best_agent_j,
        f"binary_ndcg@{k}": binary_ndcg,
        f"average_precision@{k}": average_precision,
        f"local_graded_ndcg@{k}": safe_div(
            graded_dcg,
            graded_ideal,
            both_empty=1.0 if ranked else 0.0,
        ),
    }


def evaluate_record(
    log: Mapping[str, Any],
    golds: Sequence[Tuple[str, Tuple[str, ...]]],
    unit_id: str,
) -> Dict[str, Any]:
    preds = extract_predictions(log)
    top1 = preds[0] if preds else ("", tuple())
    aligned = best_gold_alignment(top1, golds)
    tp, fp, fn = tool_counts(top1, aligned)
    p, r, f1 = prf(tp, fp, fn)

    meta = get_path(log, ("context", "meta"), {}) or {}
    llm_candidates = set(candidate_token_list(meta.get("llm_candidates", [])))
    tool_candidates = set(candidate_token_list(meta.get("tool_candidates", [])))
    gold_llm_covered = any(g[0] in llm_candidates for g in golds)
    candidate_tool_recalls = [
        safe_div(
            len(set(g[1]) & tool_candidates),
            len(g[1]),
            both_empty=1.0,
        )
        for g in golds
    ]
    all_tools_covered = any(set(g[1]).issubset(tool_candidates) for g in golds)
    recoverable = any(
        g[0] in llm_candidates and set(g[1]).issubset(tool_candidates)
        for g in golds
    )

    completed = extract_stage_agents(log, "completed_pool")
    rerank_pool = extract_stage_agents(log, "final_rerank_pool_before_cap")
    generated = any(agent in set(golds) for agent in completed)
    in_rerank_pool = any(agent in set(golds) for agent in rerank_pool)

    strict_rows = log.get("results")
    if not isinstance(strict_rows, list):
        strict_rows = get_path(log, ("generation", "results"), []) or []
    strict_texts = [
        norm(row.get("strict_text", row.get("gen_text", "")))
        for row in strict_rows
        if isinstance(row, Mapping)
    ]
    raw_tool_lists = [
        row.get("tool_tokens", row.get("tools", [])) or []
        for row in strict_rows
        if isinstance(row, Mapping)
    ]
    duplicate_tool_outputs = sum(
        len([norm(x) for x in xs]) != len(set(norm(x) for x in xs))
        for xs in raw_tool_lists
    )

    max_tools = int(get_path(log, ("config", "max_tools"), 0) or 0)
    pair_distances = []
    for i in range(len(preds[:PRIMARY_K])):
        for j in range(i + 1, len(preds[:PRIMARY_K])):
            pair_distances.append(
                1.0
                - jaccard_sets(
                    set(preds[i][1]),
                    set(preds[j][1]),
                )
            )

    trace = get_path(log, ("generation", "search_trace"), {}) or {}
    critic_events = get_path(log, ("generation", "critic_api_events"), []) or []
    critic_requested = sum(
        int(x.get("requested_count", 0) or 0)
        for x in critic_events
        if isinstance(x, Mapping)
    )
    critic_hits = sum(
        int(x.get("cache_hits", 0) or 0)
        for x in critic_events
        if isinstance(x, Mapping)
    )
    critic_latency = sum(
        float(x.get("latency_sec", 0) or 0)
        for x in critic_events
        if isinstance(x, Mapping)
    )
    generation_latency = seconds_between(
        get_path(log, ("generation", "started_at")),
        get_path(log, ("generation", "finished_at")),
    )

    def oracle(stage: Sequence[Tuple[str, Tuple[str, ...]]]) -> float:
        return max(
            (agent_jaccard(pred, gold) for pred in stage for gold in golds),
            default=0.0,
        )

    row: Dict[str, Any] = {
        "unit_id": unit_id,
        "file": log.get("__file__", ""),
        "qid": norm(get_path(log, ("dataset_example", "qid"), "")),
        "sample_number": get_path(log, ("dataset_example", "sample_number")),
        "part": get_path(log, ("dataset_example", "part")),
        "ok": bool(log.get("ok", False)),
        "gold_count": len(golds),
        "returned_count": len(preds),
        "reported_top5_count": len(strict_rows),
        "top1_llm_accuracy": float(any(top1[0] == g[0] for g in golds)),
        "top1_tool_precision": p,
        "top1_tool_recall": r,
        "top1_tool_f1": f1,
        "top1_tool_jaccard": jaccard_sets(set(top1[1]), set(aligned[1])),
        "top1_bundle_em": float(any(top1[1] == g[1] for g in golds)),
        "top1_agent_em": float(top1 in set(golds)),
        "top1_agent_jaccard": max(
            (agent_jaccard(top1, g) for g in golds), default=0.0
        ),
        "tool_tp": tp,
        "tool_fp": fp,
        "tool_fn": fn,
        "sample_with_tool_over_generation": float(fp > 0),
        "sample_with_tool_omission": float(fn > 0),
        "predicted_tool_count_top1": len(top1[1]),
        "aligned_gold_tool_count": len(aligned[1]),
        "tool_count_error_top1": len(top1[1]) - len(aligned[1]),
        "tool_count_abs_error_top1": abs(len(top1[1]) - len(aligned[1])),
        "llm_candidate_count": len(llm_candidates),
        "tool_candidate_count": len(tool_candidates),
        "gold_llm_candidate_coverage": float(gold_llm_covered),
        "gold_tool_candidate_recall": max(candidate_tool_recalls, default=0.0),
        "all_gold_tools_covered": float(all_tools_covered),
        "complete_configuration_recoverable": float(recoverable),
        "gold_generated_completed_pool": float(generated),
        "gold_in_final_rerank_pool": float(in_rerank_pool),
        "completed_pool_count": len(completed),
        "final_rerank_pool_count": len(rerank_pool),
        "oracle_agent_jaccard_completed_pool": oracle(completed),
        "oracle_agent_jaccard_rerank_pool": oracle(rerank_pool),
        "oracle_agent_jaccard_topk": oracle(preds[:PRIMARY_K]),
        "rerank_gap_pool_to_top1": oracle(completed)
        - max((agent_jaccard(top1, g) for g in golds), default=0.0),
        "topk_gap_pool_to_topk": oracle(completed) - oracle(preds[:PRIMARY_K]),
        "valid_output": float(bool(preds and top1[0])),
        "valid_topk_fraction": safe_div(
            sum(bool(p[0]) for p in preds[:PRIMARY_K]),
            len(preds[:PRIMARY_K]),
        ),
        "full_topk": float(len(preds) >= EXPECTED_RESULTS_PER_QUERY),
        "separator_fraction": safe_div(
            sum("<TOOL_SEP>" in text for text in strict_texts),
            len(strict_texts),
        ),
        "termination_fraction": safe_div(
            sum("<SPECIAL_END>" in text for text in strict_texts),
            len(strict_texts),
        ),
        "duplicate_tool_output_fraction": safe_div(
            duplicate_tool_outputs, len(raw_tool_lists)
        ),
        "candidate_legal_llm_fraction": safe_div(
            sum(p[0] in llm_candidates for p in preds[:PRIMARY_K]),
            len(preds[:PRIMARY_K]),
        ),
        "candidate_legal_tool_fraction": safe_div(
            sum(set(p[1]).issubset(tool_candidates) for p in preds[:PRIMARY_K]),
            len(preds[:PRIMARY_K]),
        ),
        "unique_configuration_ratio@5": safe_div(
            len(set(preds[:PRIMARY_K])), len(preds[:PRIMARY_K])
        ),
        "unique_llm_ratio@5": safe_div(
            len({p[0] for p in preds[:PRIMARY_K]}), len(preds[:PRIMARY_K])
        ),
        "unique_bundle_ratio@5": safe_div(
            len({p[1] for p in preds[:PRIMARY_K]}), len(preds[:PRIMARY_K])
        ),
        "intra_list_tool_diversity@5": mean(pair_distances),
        "max_tool_saturation_fraction@5": safe_div(
            sum(max_tools > 0 and len(p[1]) == max_tools for p in preds[:PRIMARY_K]),
            len(preds[:PRIMARY_K]),
        ),
        "used_fallback_free_generate": float(
            bool(get_path(log, ("generation", "used_fallback_free_generate"), False))
        ),
        "sample_latency_sec": float(
            get_path(log, ("batch", "sample_latency_sec"), 0) or 0
        ),
        "retrieval_latency_sec": float(
            get_path(log, ("retrieval", "latency_sec"), 0) or 0
        ),
        "generation_latency_sec": float(generation_latency or 0),
        "critic_latency_sec": critic_latency,
        "critic_requested_count": critic_requested,
        "critic_cache_hit_count": critic_hits,
        "generated_node_count": int(trace.get("generated_node_count", 0) or 0),
    }
    for k in TOP_KS:
        row.update(ranked_metrics(preds, golds, k))
        row[f"full_ranked_list@{k}"] = float(len(preds) >= k)
    return row


# =============================================================================
# Aggregation
# =============================================================================

MEAN_FIELDS = [
    "ok",
    "top1_llm_accuracy",
    "top1_tool_precision",
    "top1_tool_recall",
    "top1_tool_f1",
    "top1_tool_jaccard",
    "top1_bundle_em",
    "top1_agent_em",
    "top1_agent_jaccard",
    "sample_with_tool_over_generation",
    "sample_with_tool_omission",
    "predicted_tool_count_top1",
    "aligned_gold_tool_count",
    "tool_count_error_top1",
    "tool_count_abs_error_top1",
    "llm_candidate_count",
    "tool_candidate_count",
    "gold_llm_candidate_coverage",
    "gold_tool_candidate_recall",
    "all_gold_tools_covered",
    "complete_configuration_recoverable",
    "gold_generated_completed_pool",
    "gold_in_final_rerank_pool",
    "completed_pool_count",
    "final_rerank_pool_count",
    "oracle_agent_jaccard_completed_pool",
    "oracle_agent_jaccard_rerank_pool",
    "oracle_agent_jaccard_topk",
    "rerank_gap_pool_to_top1",
    "topk_gap_pool_to_topk",
    "valid_output",
    "valid_topk_fraction",
    "full_topk",
    "separator_fraction",
    "termination_fraction",
    "duplicate_tool_output_fraction",
    "candidate_legal_llm_fraction",
    "candidate_legal_tool_fraction",
    "unique_configuration_ratio@5",
    "unique_llm_ratio@5",
    "unique_bundle_ratio@5",
    "intra_list_tool_diversity@5",
    "max_tool_saturation_fraction@5",
    "used_fallback_free_generate",
    "sample_latency_sec",
    "retrieval_latency_sec",
    "generation_latency_sec",
    "critic_latency_sec",
    "critic_requested_count",
    "critic_cache_hit_count",
    "generated_node_count",
]
for _k in TOP_KS:
    MEAN_FIELDS.extend(
        [
            f"full_ranked_list@{_k}",
            f"agent_hit@{_k}",
            f"agent_mrr@{_k}",
            f"bundle_hit@{_k}",
            f"bundle_mrr@{_k}",
            f"llm_hit@{_k}",
            f"llm_mrr@{_k}",
            f"best_tool_f1@{_k}",
            f"best_agent_jaccard@{_k}",
            f"binary_ndcg@{_k}",
            f"average_precision@{_k}",
            f"local_graded_ndcg@{_k}",
        ]
    )


def aggregate(rows: Sequence[Mapping[str, Any]], level: str) -> Dict[str, Any]:
    if not rows:
        return {"evaluation_level": level, "sample_count": 0, "metrics": {}}
    metrics = {field: mean(float(row.get(field, 0) or 0) for row in rows) for field in MEAN_FIELDS}

    tp = sum(int(row.get("tool_tp", 0)) for row in rows)
    fp = sum(int(row.get("tool_fp", 0)) for row in rows)
    fn = sum(int(row.get("tool_fn", 0)) for row in rows)
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)
    metrics.update(
        {
            "top1_tool_micro_precision": micro_p,
            "top1_tool_micro_recall": micro_r,
            "top1_tool_micro_f1": micro_f1,
            "tool_label_relative_over_generation_rate_micro": safe_div(fp, tp + fp),
            "tool_omission_rate_micro": safe_div(fn, tp + fn),
            "tool_true_positive_total": tp,
            "tool_false_positive_total": fp,
            "tool_false_negative_total": fn,
        }
    )

    recoverable_rows = [
        row for row in rows if row.get("complete_configuration_recoverable") == 1.0
    ]
    generated_rows = [
        row for row in rows if row.get("gold_generated_completed_pool") == 1.0
    ]
    rerank_rows = [
        row for row in rows if row.get("gold_in_final_rerank_pool") == 1.0
    ]
    for k in TOP_KS:
        metrics[f"conditional_agent_hit@{k}_given_recoverable"] = mean(
            float(row.get(f"agent_hit@{k}", 0)) for row in recoverable_rows
        )
        metrics[f"rerank_retention@{k}_given_generated"] = mean(
            float(row.get(f"agent_hit@{k}", 0)) for row in generated_rows
        )
        metrics[f"topk_retention@{k}_given_rerank_pool"] = mean(
            float(row.get(f"agent_hit@{k}", 0)) for row in rerank_rows
        )
    metrics["gold_generated_given_recoverable"] = mean(
        float(row.get("gold_generated_completed_pool", 0)) for row in recoverable_rows
    )
    metrics["gold_rerank_pool_given_generated"] = mean(
        float(row.get("gold_in_final_rerank_pool", 0)) for row in generated_rows
    )

    for field in (
        "sample_latency_sec",
        "retrieval_latency_sec",
        "generation_latency_sec",
        "critic_latency_sec",
        "generated_node_count",
        "completed_pool_count",
    ):
        values = [float(row.get(field, 0) or 0) for row in rows]
        metrics[f"{field}_median"] = statistics.median(values)
        metrics[f"{field}_p95"] = percentile(values, 0.95)
    requested = sum(int(row.get("critic_requested_count", 0)) for row in rows)
    hits = sum(int(row.get("critic_cache_hit_count", 0)) for row in rows)
    metrics["critic_cache_hit_rate"] = safe_div(hits, requested + hits)

    return {
        "evaluation_level": level,
        "sample_count": len(rows),
        "recoverable_sample_count": len(recoverable_rows),
        "gold_generated_sample_count": len(generated_rows),
        "gold_in_rerank_pool_sample_count": len(rerank_rows),
        "metrics": metrics,
    }


METRIC_DEFINITIONS = {
    "configuration_level": (
        "Each per-sample log is compared only with its own dataset_example.target."
    ),
    "query_multi_positive": (
        "Logs are deduplicated by qid and compared with every distinct target for "
        "that qid in the complete validation JSONL. Gold tool bundles are never merged."
    ),
    "multi_positive_micro_alignment": (
        "For top-1 micro tool counts, select one gold configuration per query by "
        "lexicographically maximizing exact-agent match, typed-component Jaccard, "
        "tool F1, LLM match, and negative tool-count error."
    ),
    "empty_tool_convention": (
        "For tool precision, recall, F1, and Jaccard, two empty sets receive 1; "
        "an empty/non-empty mismatch receives 0."
    ),
    "top1_agent_em": "Exact LLM and exact unordered tool-set match at rank 1.",
    "agent_hit@K": "At least one exact gold agent configuration appears in Top-K.",
    "agent_mrr@K": "Reciprocal rank of the first exact gold agent within Top-K.",
    "top_k_ranking_source": (
        "Top-K metrics use the score-sorted "
        "generation.search_trace.final_rerank_pool_before_cap. Logs without "
        "that field fall back to the capped results list."
    ),
    "full_ranked_list@K": (
        "Fraction of evaluation units whose final ranking contains at least K candidates."
    ),
    "best_agent_jaccard@K": (
        "Maximum typed-component Jaccard between a Top-K prediction and any valid gold."
    ),
    "binary_ndcg@K": (
        "Exact-agent binary nDCG with one-to-one matching of distinct gold configurations."
    ),
    "local_graded_ndcg@K": (
        "Diagnostic ordering consistency among the returned K items using typed-component "
        "Jaccard as gain. It does not measure candidate-set recall."
    ),
    "complete_configuration_recoverable": (
        "At least one gold has its LLM in llm_candidates and all its tools in "
        "tool_candidates."
    ),
    "gold_generated_completed_pool": (
        "At least one exact gold appears in search_trace.completed_pool."
    ),
    "rerank_retention@K_given_generated": (
        "Exact Agent Hit@K conditional on an exact gold appearing in completed_pool."
    ),
    "tool_label_relative_over_generation_rate_micro": (
        "sum |predicted tools minus aligned gold tools| / sum |predicted tools|; "
        "this is label-relative over-generation, not candidate-space hallucination."
    ),
    "sample_latency_sec": (
        "End-to-end saved batch latency. The experiment generated explanations, so this "
        "timing can include explanation work even though explanation quality is not evaluated."
    ),
}


def write_flat_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        level = summary["evaluation_level"]
        for key, value in summary.get("metrics", {}).items():
            rows.append(
                {
                    "evaluation_level": level,
                    "metric": key,
                    "value": value,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["evaluation_level", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    return f"{100 * float(value or 0):.2f}%"


def write_report(
    path: Path,
    config_summary: Mapping[str, Any],
    query_summary: Mapping[str, Any],
    input_info: Mapping[str, Any],
) -> None:
    c = config_summary["metrics"]
    q = query_summary["metrics"]
    lines = [
        "# Exp2 v13 Agent Recommendation Evaluation",
        "",
        "Natural-language explanations are not evaluated.",
        "",
        "## Evaluation scope",
        "",
        f"- Loaded per-sample logs: {input_info['loaded_logs']}",
        f"- Configuration-level instances: {config_summary['sample_count']}",
        f"- Query-level unique qids: {query_summary['sample_count']}",
        f"- Malformed JSON logs: {input_info['malformed_log_count']}",
        "",
        "## Main recommendation results",
        "",
        "| Level | LLM Acc@1 | Tool Macro-F1@1 | Agent EM@1 |",
        "|---|---:|---:|---:|",
        (
            f"| Configuration | {pct(c['top1_llm_accuracy'])} | "
            f"{pct(c['top1_tool_f1'])} | {pct(c['top1_agent_em'])} |"
        ),
        (
            f"| Query multi-positive | {pct(q['top1_llm_accuracy'])} | "
            f"{pct(q['top1_tool_f1'])} | {pct(q['top1_agent_em'])} |"
        ),
        "",
        "## Top-K recommendation results",
        "",
        "| Level | K | Full ranked list | Agent Hit | Agent MRR | Best Agent J | Binary nDCG |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {level_name} | {k} | "
                f"{pct(metrics[f'full_ranked_list@{k}'])} | "
                f"{pct(metrics[f'agent_hit@{k}'])} | "
                f"{metrics[f'agent_mrr@{k}']:.4f} | "
                f"{metrics[f'best_agent_jaccard@{k}']:.4f} | "
                f"{metrics[f'binary_ndcg@{k}']:.4f} |"
            )
            for level_name, metrics in (
                ("Configuration", c),
                ("Query multi-positive", q),
            )
            for k in (5, 10, 20)
        ],
        "",
        "## Retrieval, search, and reranking decomposition",
        "",
        "| Level | Recoverable | Gold generated | Generated given recoverable | Hit@5 given recoverable | Hit@5 given generated |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Configuration | {pct(c['complete_configuration_recoverable'])} | "
            f"{pct(c['gold_generated_completed_pool'])} | "
            f"{pct(c['gold_generated_given_recoverable'])} | "
            f"{pct(c['conditional_agent_hit@5_given_recoverable'])} | "
            f"{pct(c['rerank_retention@5_given_generated'])} |"
        ),
        (
            f"| Query multi-positive | {pct(q['complete_configuration_recoverable'])} | "
            f"{pct(q['gold_generated_completed_pool'])} | "
            f"{pct(q['gold_generated_given_recoverable'])} | "
            f"{pct(q['conditional_agent_hit@5_given_recoverable'])} | "
            f"{pct(q['rerank_retention@5_given_generated'])} |"
        ),
        "",
        "## Bundle behavior (query multi-positive)",
        "",
        f"- Mean predicted tools at Top-1: {q['predicted_tool_count_top1']:.3f}",
        f"- Mean aligned gold tools: {q['aligned_gold_tool_count']:.3f}",
        f"- Tool-count bias: {q['tool_count_error_top1']:.3f}",
        f"- Tool-count MAE: {q['tool_count_abs_error_top1']:.3f}",
        f"- Max-tool saturation across Top-5: {pct(q['max_tool_saturation_fraction@5'])}",
        f"- Label-relative over-generation (micro): {pct(q['tool_label_relative_over_generation_rate_micro'])}",
        f"- Tool omission (micro): {pct(q['tool_omission_rate_micro'])}",
        "",
        "## Validity and efficiency (query multi-positive)",
        "",
        f"- Successful run rate: {pct(q['ok'])}",
        f"- Full Top-5 rate: {pct(q['full_topk'])}",
        f"- Candidate-legal LLM rate: {pct(q['candidate_legal_llm_fraction'])}",
        f"- Candidate-legal tool-bundle rate: {pct(q['candidate_legal_tool_fraction'])}",
        f"- Mean end-to-end latency: {q['sample_latency_sec']:.2f} s",
        f"- P95 end-to-end latency: {q['sample_latency_sec_p95']:.2f} s",
        f"- Mean generated nodes: {q['generated_node_count']:.2f}",
        "",
        "See `evaluation_summary.json` for all metrics and exact definitions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print(f"[INFO] Fixed per-sample directory: {PER_SAMPLE_DIR}")
    print(f"[INFO] Fixed full validation JSONL: {FULL_VALID_JSONL}")
    print(f"[INFO] Fixed output directory: {OUTPUT_DIR}")

    paths = input_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_gold, gold_stats = load_full_gold()
    if not full_gold:
        print(
            "[WARNING] Full validation JSONL was unavailable or yielded no gold. "
            "Query-level evaluation will fall back to gold targets found in the "
            "per-sample logs; this can omit un-sampled positives.",
            file=sys.stderr,
        )

    load_errors: List[Dict[str, str]] = []
    log_gold_by_qid: Dict[str, set] = defaultdict(set)
    representative_info: Dict[str, Dict[str, Any]] = {}
    config_rows = []
    skipped_no_gold = []
    first_structure: Optional[str] = None
    loaded_logs = 0
    for path in paths:
        try:
            log = load_json(path)
            log["__file__"] = str(path)
        except Exception as exc:
            load_errors.append(
                {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        loaded_logs += 1
        if first_structure is None:
            first_structure = "\n".join(structure_lines(log, max_depth=5)) + "\n"
            (OUTPUT_DIR / "input_structure.txt").write_text(
                first_structure, encoding="utf-8"
            )
            print("\n[STRUCTURE] First successfully loaded per-sample JSON")
            print(first_structure)

        qid = norm(get_path(log, ("dataset_example", "qid"), ""))
        gold = parse_agent_text(get_path(log, ("dataset_example", "target"), ""))
        if qid and gold[0]:
            log_gold_by_qid[qid].add(gold)
            config_rows.append(evaluate_record(log, [gold], f"sample::{log['__file__']}"))
        else:
            skipped_no_gold.append(
                {
                    "file": log.get("__file__", ""),
                    "qid": qid,
                    "reason": "missing qid or unparsable dataset_example.target",
                }
            )
        if qid:
            source_index = get_path(
                log, ("dataset_example", "source_index_zero_based")
            )
            order_key = (
                source_index if source_index is not None else 10**18,
                str(path),
            )
            signature = tuple(extract_predictions(log))
            info = representative_info.get(qid)
            if info is None:
                representative_info[qid] = {
                    "path": str(path),
                    "order_key": order_key,
                    "signatures": {signature},
                }
            else:
                info["signatures"].add(signature)
                if order_key < info["order_key"]:
                    info["path"] = str(path)
                    info["order_key"] = order_key

    if loaded_logs == 0:
        raise RuntimeError("Every per-sample JSON failed to load.")
    dedup_stats = {
        "unique_qids": len(representative_info),
        "duplicate_log_count": loaded_logs - len(representative_info),
        "qids_with_inconsistent_prediction_lists": sum(
            len(info["signatures"]) > 1 for info in representative_info.values()
        ),
        "representative_rule": "lowest source_index_zero_based, then filename",
        "memory_strategy": (
            "streaming first pass plus a second pass over one representative file per qid"
        ),
    }

    query_rows = []
    missing_full_gold_qids = []
    for qid, info in sorted(representative_info.items()):
        log = load_json(Path(info["path"]))
        log["__file__"] = info["path"]
        golds = full_gold.get(qid) or sorted(log_gold_by_qid.get(qid, set()))
        if qid not in full_gold:
            missing_full_gold_qids.append(qid)
        if golds:
            query_rows.append(evaluate_record(log, golds, f"qid::{qid}"))

    config_summary = aggregate(config_rows, "configuration_level")
    query_summary = aggregate(query_rows, "query_multi_positive")
    input_info = {
        "per_sample_dir": str(PER_SAMPLE_DIR),
        "full_valid_jsonl": str(FULL_VALID_JSONL),
        "output_dir": str(OUTPUT_DIR),
        "discovered_json_files": len(paths),
        "loaded_logs": loaded_logs,
        "malformed_log_count": len(load_errors),
        "skipped_configuration_logs_without_gold": len(skipped_no_gold),
        "missing_full_gold_qid_count": len(set(missing_full_gold_qids)),
        "top_ks": list(TOP_KS),
        "primary_k": PRIMARY_K,
        "expected_results_per_query": EXPECTED_RESULTS_PER_QUERY,
        "top_k_ranking_source": (
            "generation.search_trace.final_rerank_pool_before_cap; "
            "fallback: results"
        ),
        "normalization": {
            "unicode": "NFKC",
            "whitespace": "collapsed",
            "case_insensitive": CASE_INSENSITIVE,
            "tool_bundle_semantics": "unordered set",
        },
        "full_gold_loading": gold_stats,
        "query_log_deduplication": dedup_stats,
    }
    summary = {
        "evaluation_name": "Exp2 v13 Bundle-Aware Agent Recommendation",
        "purpose": (
            "Evaluate Top-K LLM-tool agent recommendations, retrieval coverage, "
            "beam-search generation, final reranking, validity, bundle behavior, "
            "diversity, and efficiency. Explanation quality is excluded."
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "input": input_info,
        "configuration_level": config_summary,
        "query_multi_positive": query_summary,
        "metric_definitions": METRIC_DEFINITIONS,
    }

    json_dump(OUTPUT_DIR / "evaluation_summary.json", summary)
    write_flat_csv(
        OUTPUT_DIR / "metrics.csv", [config_summary, query_summary]
    )
    with (OUTPUT_DIR / "per_record_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for level, rows in (
            ("configuration_level", config_rows),
            ("query_multi_positive", query_rows),
        ):
            for row in rows:
                f.write(
                    json.dumps(
                        {"evaluation_level": level, **row},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    json_dump(
        OUTPUT_DIR / "evaluation_issues.json",
        {
            "malformed_logs": load_errors,
            "skipped_logs_without_gold": skipped_no_gold,
            "qids_missing_from_full_validation_gold": sorted(
                set(missing_full_gold_qids)
            ),
        },
    )
    write_report(
        OUTPUT_DIR / "evaluation_report.md",
        config_summary,
        query_summary,
        input_info,
    )

    q = query_summary["metrics"]
    print("[DONE] Agent recommendation evaluation completed.")
    print(f"[RESULT] Query-level Agent EM@1: {pct(q.get('top1_agent_em'))}")
    print(f"[RESULT] Query-level Agent Hit@5: {pct(q.get('agent_hit@5'))}")
    print(f"[RESULT] Query-level Agent Hit@10: {pct(q.get('agent_hit@10'))}")
    print(f"[RESULT] Query-level Agent Hit@20: {pct(q.get('agent_hit@20'))}")
    print(
        "[RESULT] Query-level complete configuration recoverable: "
        f"{pct(q.get('complete_configuration_recoverable'))}"
    )
    print(f"[OUTPUT] {OUTPUT_DIR / 'evaluation_summary.json'}")
    print(f"[OUTPUT] {OUTPUT_DIR / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
