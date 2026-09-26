#!/usr/bin/env python3
"""Evaluate beam/critic and structural ablations with grounded soft overlap.

The script is deliberately independent of the inference code.  With no
arguments it evaluates the fixed experiment used for the 5 x 5 beam/critic
grid and writes one plotting-friendly aggregate JSON file.

Typical use:
    python evaluate_beam_critic_experiments.py

Override the location when needed:
    python evaluate_beam_critic_experiments.py \
        --experiment-root /path/to/experiment \
        --output /path/to/evaluation_summary.json
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_EXPERIMENT_ROOT = Path(
    str(AC_ROOT / 'outputs/') +
    "exp2_v13_ablation_beam_critic_seed42_first100"
)
DEFAULT_OUTPUT_NAME = "beam_critic_evaluation_summary.json"
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20)
EPS = 1e-12

LLM_TOKEN_RE = re.compile(r"<LLM_([^<>]+)>")
TOOL_TOKEN_RE = re.compile(r"<TOOL_([^<>]+)>")
DOUBLE_TOOL_RE = re.compile(r"<<([^<>]+)>>")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_seconds(start: Any, finish: Any) -> Optional[float]:
    left, right = parse_time(start), parse_time(finish)
    if left is None or right is None:
        return None
    result = (right - left).total_seconds()
    return result if result >= 0 else None


def canonical_llm(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = LLM_TOKEN_RE.fullmatch(text)
    if match:
        text = match.group(1)
    return text.strip() or None


def canonical_tool(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = DOUBLE_TOOL_RE.fullmatch(text)
    if match:
        text = match.group(1)
    else:
        match = TOOL_TOKEN_RE.fullmatch(text)
        if match:
            text = match.group(1)
    if text in {"SEP", "TOOL_SEP", "SPECIAL_END", "SPECIAL_EMPTY"}:
        return None
    return text.strip() or None


def parse_gold_target(target: Any) -> Tuple[Optional[str], List[str], bool]:
    if not isinstance(target, str):
        return None, [], False
    llm_match = LLM_TOKEN_RE.search(target)
    llm = llm_match.group(1).strip() if llm_match else None
    tools: List[str] = []
    token_matches: List[Tuple[int, str]] = []
    for match in TOOL_TOKEN_RE.finditer(target):
        token_matches.append((match.start(), match.group(1)))
    for match in DOUBLE_TOOL_RE.finditer(target):
        token_matches.append((match.start(), match.group(1)))
    for _, raw in sorted(token_matches):
        tool = canonical_tool(raw)
        if tool is not None:
            tools.append(tool)
    parse_valid = llm is not None and "<TOOL_SEP>" in target
    return llm, tools, parse_valid


def parse_candidate(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {
            "llm": None,
            "tools": [],
            "parse_valid": False,
            "is_complete": False,
            "termination_reason": None,
            "raw": {},
        }
    # Beam-search candidates use ``llm``/``tools`` while the direct-free and
    # constrained-greedy branches save the same information as
    # ``llm_token``/``tool_tokens`` under generation.results.  Accept both
    # schemas so structural ablations are evaluated by one code path.
    raw_llm = candidate.get("llm")
    if raw_llm is None:
        raw_llm = candidate.get("llm_token")
    raw_tools = candidate.get("tools")
    if raw_tools is None:
        raw_tools = candidate.get("tool_tokens")

    candidate_text = candidate.get("strict_text") or candidate.get("gen_text")
    if raw_llm is None and isinstance(candidate_text, str):
        llm_match = LLM_TOKEN_RE.search(candidate_text)
        raw_llm = llm_match.group(0) if llm_match else None
    if raw_tools is None and isinstance(candidate_text, str):
        token_matches: List[Tuple[int, str]] = []
        for match in TOOL_TOKEN_RE.finditer(candidate_text):
            token_matches.append((match.start(), match.group(0)))
        for match in DOUBLE_TOOL_RE.finditer(candidate_text):
            token_matches.append((match.start(), match.group(0)))
        raw_tools = [value for _, value in sorted(token_matches)]
    llm = canonical_llm(raw_llm)
    tools: List[str] = []
    tools_valid = isinstance(raw_tools, list)
    if tools_valid:
        for value in raw_tools:
            tool = canonical_tool(value)
            if tool is None:
                tools_valid = False
            else:
                tools.append(tool)
    complete = bool(candidate.get("is_complete"))
    last_action = candidate.get("last_action")
    reason = candidate.get("termination_reason")
    if last_action == "<SPECIAL_END>" or reason in {
        "selected_end",
        "max_tools_then_forced_end",
        "empty_then_forced_end",
    }:
        complete = True
    if isinstance(candidate_text, str) and "<SPECIAL_END>" in candidate_text:
        complete = True
    return {
        "llm": llm,
        "tools": tools,
        "parse_valid": llm is not None and tools_valid,
        "is_complete": complete,
        "termination_reason": reason if isinstance(reason, str) else None,
        "generator_logprob": safe_float(candidate.get("generator_logprob")),
        "generator_token_count": safe_float(candidate.get("generator_token_count")),
        "generator_avg_logprob": safe_float(candidate.get("generator_avg_logprob")),
        "critic_raw": safe_float(candidate.get("critic_raw")),
        "critic_sigmoid": safe_float(candidate.get("critic_sigmoid")),
        "search_score": safe_float(candidate.get("search_score")),
        "raw": candidate,
    }


def set_overlap(predicted: Iterable[str], gold: Iterable[str]) -> Dict[str, float]:
    pred_set, gold_set = set(predicted), set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    if not pred_set and not gold_set:
        precision = recall = f1 = jaccard = dice = 1.0
    else:
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 1.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        union = len(pred_set | gold_set)
        jaccard = tp / union if union else 1.0
        denom = len(pred_set) + len(gold_set)
        dice = 2.0 * tp / denom if denom else 1.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "dice": dice,
    }


def compare_candidate(
    candidate: Mapping[str, Any], gold_llm: Optional[str], gold_tools: Sequence[str]
) -> Dict[str, Any]:
    pred_llm = candidate.get("llm")
    pred_tools = list(candidate.get("tools") or [])
    parse_valid = bool(candidate.get("parse_valid"))
    complete = bool(candidate.get("is_complete"))
    llm_em = pred_llm is not None and pred_llm == gold_llm
    tool_ordered_em = pred_tools == list(gold_tools)
    tool_unordered_em = set(pred_tools) == set(gold_tools)
    tool_overlap = set_overlap(pred_tools, gold_tools)
    # Recall-oriented success: a prediction may contain extra tools, but it
    # must not omit any gold tool.  This is deliberately distinct from EM.
    tool_complete_hit = parse_valid and tool_overlap["fn"] == 0
    pred_components = ([pred_llm] if pred_llm is not None else []) + pred_tools
    gold_components = ([gold_llm] if gold_llm is not None else []) + list(gold_tools)
    component_overlap = set_overlap(pred_components, gold_components)
    duplicate_count = len(pred_tools) - len(set(pred_tools))
    return {
        "llm_em": llm_em,
        "tool_ordered_em": tool_ordered_em,
        "tool_unordered_em": tool_unordered_em,
        "agent_ordered_em": llm_em and tool_ordered_em,
        "agent_unordered_em": llm_em and tool_unordered_em,
        "tool_complete_hit": tool_complete_hit,
        "agent_complete_hit": llm_em and tool_complete_hit,
        "valid_agent_complete_hit": (
            llm_em and tool_complete_hit and parse_valid and complete and duplicate_count == 0
        ),
        "valid_agent_unordered_em": (
            llm_em and tool_unordered_em and parse_valid and complete and duplicate_count == 0
        ),
        "tool": tool_overlap,
        "component": component_overlap,
        "predicted_tool_count": len(pred_tools),
        "gold_tool_count": len(gold_tools),
        "tool_count_error": len(pred_tools) - len(gold_tools),
        "tool_count_absolute_error": abs(len(pred_tools) - len(gold_tools)),
        "tool_count_exact": len(pred_tools) == len(gold_tools),
        "tool_over_generation": len(pred_tools) > len(gold_tools),
        "tool_under_generation": len(pred_tools) < len(gold_tools),
        "duplicate_tool_count": duplicate_count,
        "has_duplicate_tools": duplicate_count > 0,
        "any_gold_omitted": tool_overlap["fn"] > 0,
        "gold_omission_fraction": (
            tool_overlap["fn"] / len(set(gold_tools)) if gold_tools else 0.0
        ),
        "any_label_relative_hallucination": tool_overlap["fp"] > 0,
        "label_relative_hallucination_fraction": (
            tool_overlap["fp"] / len(set(pred_tools)) if pred_tools else 0.0
        ),
    }


def apply_retrieval_grounding(
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    retrieved_llms: set,
    retrieved_tools: set,
) -> Dict[str, Any]:
    """Attach retrieval-grounded overlap scores to one candidate comparison.

    The per-sample logs do not contain the complete global component registry,
    but they do contain the exact candidate universe supplied to generation.
    A prediction outside that universe is therefore treated as ungrounded.  Its
    paper-facing soft-overlap scores are zeroed as a whole rather than allowing
    a partially matching, hallucinated bundle to receive credit.  Raw overlap
    remains available under ``tool``/``component`` for diagnosis.
    """

    pred_llm = candidate.get("llm")
    pred_tools = list(candidate.get("tools") or [])
    llm_grounded = pred_llm is not None and pred_llm in retrieved_llms
    ungrounded_tools = sorted({tool for tool in pred_tools if tool not in retrieved_tools})
    tools_grounded = not ungrounded_tools
    retrieval_grounded = (
        bool(candidate.get("parse_valid")) and llm_grounded and tools_grounded
    )

    def grounded_overlap(family: str) -> Dict[str, float]:
        raw = dict(comparison[family])
        if retrieval_grounded:
            return raw
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            raw[metric] = 0.0
        return raw

    result = dict(comparison)
    result.update(
        {
            "retrieval_grounded": retrieval_grounded,
            "llm_retrieval_grounded": llm_grounded,
            "all_tools_retrieval_grounded": tools_grounded,
            "ungrounded_tool_count": len(ungrounded_tools),
            "ungrounded_tools": ungrounded_tools,
            "grounded_tool": grounded_overlap("tool"),
            "grounded_component": grounded_overlap("component"),
        }
    )
    return result


def recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from recursive_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_strings(child)


def retrieval_universe(sample: Mapping[str, Any]) -> Tuple[set, set]:
    # Restrict extraction to retrieval/context; the dataset gold target is excluded.
    sources = {"retrieval": sample.get("retrieval"), "context": sample.get("context")}
    llms, tools = set(), set()
    for text in recursive_strings(sources):
        llms.update(match.group(1).strip() for match in LLM_TOKEN_RE.finditer(text))
        tools.update(match.group(1).strip() for match in DOUBLE_TOOL_RE.finditer(text))
        tools.update(match.group(1).strip() for match in TOOL_TOKEN_RE.finditer(text))
    tools.discard("SEP")
    tools.discard("TOOL_SEP")
    return llms, tools


def choose_ranked_pool(
    generation: Mapping[str, Any], sample: Mapping[str, Any]
) -> Tuple[List[Any], str]:
    trace = generation.get("search_trace") or {}
    pool = trace.get("final_rerank_pool_before_cap")
    if isinstance(pool, list) and pool:
        return pool, "generation.search_trace.final_rerank_pool_before_cap"
    pool = trace.get("final_candidates")
    if isinstance(pool, list) and pool:
        return pool, "generation.search_trace.final_candidates"
    # completed_pool is not guaranteed to be reranked; sorting by search score is
    # a conservative final fallback for schema variants.
    pool = trace.get("completed_pool")
    if isinstance(pool, list) and pool:
        sortable = [item for item in pool if isinstance(item, Mapping)]
        sortable.sort(
            key=lambda item: safe_float(item.get("search_score"))
            if safe_float(item.get("search_score")) is not None
            else -math.inf,
            reverse=True,
        )
        return sortable, "generation.search_trace.completed_pool_sorted_by_search_score"

    # Non-beam structural-ablation variants do not create search_trace.  They
    # save one structured prediction in generation.results (also mirrored at
    # the sample top level in current inference outputs).
    pool = generation.get("results")
    if isinstance(pool, list) and pool:
        return pool, "generation.results"
    pool = sample.get("results")
    if isinstance(pool, list) and pool:
        return pool, "results"

    # Last-resort support for older direct-generation logs that retained only
    # raw strings.  parse_candidate() will recover tokens from gen_text.
    raw_outputs = generation.get("raw_outputs")
    if isinstance(raw_outputs, list) and raw_outputs:
        return [
            item if isinstance(item, Mapping) else {"gen_text": item}
            for item in raw_outputs
        ], "generation.raw_outputs"
    return [], "none"


def first_true_rank(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[int]:
    for rank, row in enumerate(rows, start=1):
        if row.get(key):
            return rank
    return None


def rr_at(rank: Optional[int], cutoff: Optional[int]) -> float:
    if rank is None or (cutoff is not None and rank > cutoff):
        return 0.0
    return 1.0 / rank


def ndcg_single_at(rank: Optional[int], cutoff: Optional[int]) -> float:
    if rank is None or (cutoff is not None and rank > cutoff):
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def pairwise_jaccard_distance(sets: Sequence[set]) -> float:
    if len(sets) < 2:
        return 0.0
    values: List[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            union = sets[left_index] | sets[right_index]
            similarity = len(sets[left_index] & sets[right_index]) / len(union) if union else 1.0
            values.append(1.0 - similarity)
    return statistics.fmean(values) if values else 0.0


def average_tied_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denom = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denom <= EPS:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denom


def spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(average_tied_ranks(left), average_tied_ranks(right))


def ranked_metrics(
    parsed_pool: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    cutoff: Optional[int],
) -> Dict[str, Any]:
    available = len(parsed_pool)
    limit = available if cutoff is None else min(cutoff, available)
    candidates = list(parsed_pool[:limit])
    rows = list(comparisons[:limit])
    result: Dict[str, Any] = {
        "available_candidate_count": available,
        "evaluated_candidate_count": limit,
        "has_at_least_cutoff_candidates": True if cutoff is None else available >= cutoff,
    }
    rank_keys = (
        "llm_em",
        "tool_complete_hit",
        "agent_complete_hit",
        "tool_ordered_em",
        "tool_unordered_em",
        "agent_ordered_em",
        "agent_unordered_em",
    )
    for key in rank_keys:
        rank = first_true_rank(rows, key)
        result[f"{key}_hit"] = rank is not None
        result[f"{key}_first_rank"] = rank
        result[f"{key}_mrr"] = rr_at(rank, None)
        result[f"{key}_ndcg"] = ndcg_single_at(rank, None)
    for family in ("tool", "component"):
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            values = [float(row[family][metric]) for row in rows]
            result[f"oracle_best_{family}_{metric}"] = max(values) if values else 0.0
            # Unlike oracle-best overlap, this treats every retained candidate
            # as a positive recommendation and measures the overall quality of
            # the Top-K candidate set.  It is intentionally unweighted by rank.
            result[f"mean_candidate_{family}_{metric}"] = (
                statistics.fmean(values) if values else 0.0
            )
    for family in ("grounded_tool", "grounded_component"):
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            values = [float(row[family][metric]) for row in rows]
            result[f"oracle_best_{family}_{metric}"] = max(values) if values else 0.0
            result[f"mean_candidate_{family}_{metric}"] = (
                statistics.fmean(values) if values else 0.0
            )
    grounded_count = sum(bool(row.get("retrieval_grounded")) for row in rows)
    result["retrieval_grounded_candidate_count"] = grounded_count
    result["retrieval_grounded_candidate_fraction"] = (
        grounded_count / limit if limit else 0.0
    )
    result["zeroed_ungrounded_candidate_count"] = limit - grounded_count
    result["all_evaluated_candidates_retrieval_grounded"] = (
        grounded_count == limit if limit else False
    )
    # Paper-facing aliases. Coverage@K is the best recall achieved by one
    # candidate within Top-K; gold components cannot be assembled across
    # different candidates.
    result["tool_coverage"] = result["oracle_best_tool_recall"]
    result["component_coverage"] = result["oracle_best_component_recall"]
    if rows:
        result["top1_to_oracle_component_f1_regret"] = (
            result["oracle_best_component_f1"] - float(rows[0]["component"]["f1"])
        )
        result["top1_to_oracle_tool_f1_regret"] = (
            result["oracle_best_tool_f1"] - float(rows[0]["tool"]["f1"])
        )
    else:
        result["top1_to_oracle_component_f1_regret"] = 0.0
        result["top1_to_oracle_tool_f1_regret"] = 0.0

    llms = [candidate.get("llm") for candidate in candidates if candidate.get("llm")]
    tool_lists = [list(candidate.get("tools") or []) for candidate in candidates]
    tool_sets = [set(tools) for tools in tool_lists]
    component_sets = [
        set(([candidate.get("llm")] if candidate.get("llm") else []) + list(candidate.get("tools") or []))
        for candidate in candidates
    ]
    full_configs = [
        (candidate.get("llm"), tuple(sorted(set(candidate.get("tools") or []))))
        for candidate in candidates
    ]
    result.update(
        {
            "distinct_llm_count": len(set(llms)),
            "distinct_llm_ratio": len(set(llms)) / limit if limit else 0.0,
            "distinct_tool_bundle_count": len({tuple(sorted(value)) for value in tool_sets}),
            "distinct_tool_bundle_ratio": (
                len({tuple(sorted(value)) for value in tool_sets}) / limit if limit else 0.0
            ),
            "distinct_agent_config_count": len(set(full_configs)),
            "distinct_agent_config_ratio": len(set(full_configs)) / limit if limit else 0.0,
            "union_tool_count": len(set().union(*tool_sets)) if tool_sets else 0,
            "tool_bundle_pairwise_jaccard_distance": pairwise_jaccard_distance(tool_sets),
            "component_pairwise_jaccard_distance": pairwise_jaccard_distance(component_sets),
        }
    )
    return result


def evaluate_sample(path: Path) -> Dict[str, Any]:
    sample = load_json(path)
    dataset = sample.get("dataset_example") or {}
    target = dataset.get("target")
    if target is None:
        source = dataset.get("source_example") or {}
        target = source.get("target") or source.get("completion")
        if isinstance(target, str):
            target = target.split("Explanation:", 1)[0].strip()
    gold_llm, gold_tools, gold_parse_valid = parse_gold_target(target)
    if not gold_parse_valid:
        raise ValueError("gold target cannot be parsed")

    generation = sample.get("generation") or {}
    trace = generation.get("search_trace") or {}
    raw_pool, pool_source = choose_ranked_pool(generation, sample)
    parsed_pool = [parse_candidate(candidate) for candidate in raw_pool]
    retrieved_llms, retrieved_tools = retrieval_universe(sample)
    comparisons = [
        apply_retrieval_grounding(
            candidate,
            compare_candidate(candidate, gold_llm, gold_tools),
            retrieved_llms,
            retrieved_tools,
        )
        for candidate in parsed_pool
    ]
    top1_candidate = parsed_pool[0] if parsed_pool else parse_candidate(None)
    top1 = comparisons[0] if comparisons else compare_candidate(top1_candidate, gold_llm, gold_tools)
    pred_llm = top1_candidate.get("llm")
    pred_tools = list(top1_candidate.get("tools") or [])

    score_vectors: Dict[str, List[float]] = {
        "critic_raw": [],
        "generator_avg_logprob": [],
        "search_score": [],
    }
    relevance_vectors: Dict[str, Dict[str, List[float]]] = {
        key: {"component_f1": [], "component_recall": [], "tool_recall": []}
        for key in score_vectors
    }
    for candidate, comparison in zip(parsed_pool, comparisons):
        relevance = {
            "component_f1": float(comparison["component"]["f1"]),
            "component_recall": float(comparison["component"]["recall"]),
            "tool_recall": float(comparison["tool"]["recall"]),
        }
        for key in score_vectors:
            value = safe_float(candidate.get(key))
            if value is not None:
                score_vectors[key].append(value)
                for metric, metric_value in relevance.items():
                    relevance_vectors[key][metric].append(metric_value)
    correlations = {}
    for key in score_vectors:
        for metric, values in relevance_vectors[key].items():
            correlations[f"{key}_pearson_{metric}"] = pearson(score_vectors[key], values)
            correlations[f"{key}_spearman_{metric}"] = spearman(score_vectors[key], values)
        correlations[f"{key}_candidate_count"] = len(score_vectors[key])

    levels = trace.get("levels") if isinstance(trace.get("levels"), list) else []
    level_input_total = 0
    level_retained_total = 0
    level_pruned_total = 0
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        prune = level.get("prune") or {}
        level_input_total += int(prune.get("input_count") or 0)
        level_retained_total += int(prune.get("retained_count") or 0)
        level_pruned_total += int(prune.get("pruned_count") or 0)

    result = {
        "path": str(path),
        "sample_id": dataset.get("sample_id") or path.stem,
        "sample_number": dataset.get("sample_number"),
        "qid": dataset.get("qid"),
        "part": dataset.get("part") or "unknown",
        "gold": {"llm": gold_llm, "tools": gold_tools},
        "pool_source": pool_source,
        "pool_size": len(parsed_pool),
        "top1": {
            **top1,
            "parse_valid": bool(top1_candidate.get("parse_valid")),
            "termination_valid": bool(top1_candidate.get("is_complete")),
            "separator_present": bool(top1_candidate.get("parse_valid")),
            "duplicate_free": top1["duplicate_tool_count"] == 0,
            "termination_reason": top1_candidate.get("termination_reason"),
            "llm_in_retrieved_candidates": pred_llm in retrieved_llms if pred_llm else False,
            "all_tools_in_retrieved_candidates": all(tool in retrieved_tools for tool in pred_tools),
            "invalid_retrieved_tool_count": sum(tool not in retrieved_tools for tool in set(pred_tools)),
            "retrieval_grounded": bool(top1.get("retrieval_grounded")),
            "grounded_tool": top1.get("grounded_tool"),
            "grounded_component": top1.get("grounded_component"),
            "generator_logprob": top1_candidate.get("generator_logprob"),
            "generator_token_count": top1_candidate.get("generator_token_count"),
            "generator_avg_logprob": top1_candidate.get("generator_avg_logprob"),
            "critic_raw": top1_candidate.get("critic_raw"),
            "critic_sigmoid": top1_candidate.get("critic_sigmoid"),
            "search_score": top1_candidate.get("search_score"),
        },
        "retrieval_upper_bound": {
            "gold_llm_retrieved": gold_llm in retrieved_llms if gold_llm else False,
            "gold_tool_recall": (
                len(set(gold_tools) & retrieved_tools) / len(set(gold_tools)) if gold_tools else 1.0
            ),
            "all_gold_tools_retrieved": all(tool in retrieved_tools for tool in gold_tools),
            "full_gold_config_retrievable": (
                (gold_llm in retrieved_llms if gold_llm else False)
                and all(tool in retrieved_tools for tool in gold_tools)
            ),
            "retrieved_llm_count": len(retrieved_llms),
            "retrieved_tool_count": len(retrieved_tools),
        },
        "ranked": {},
        "ranking_score_alignment": correlations,
        "search": {
            "generation_latency_sec": elapsed_seconds(trace.get("started_at"), trace.get("finished_at")),
            "total_pipeline_latency_sec": elapsed_seconds(
                (sample.get("retrieval") or {}).get("started_at"), trace.get("finished_at")
            ),
            "model_loading_latency_sec": safe_float((sample.get("model_loading") or {}).get("latency_sec")),
            "input_token_count": safe_float((sample.get("prompt") or {}).get("input_token_count")),
            "prompt_truncated": bool((sample.get("prompt") or {}).get("truncated")),
            "used_fallback_free_generate": bool(generation.get("used_fallback_free_generate")),
            "llm_candidate_count": safe_float(trace.get("llm_candidate_count")),
            "tool_candidate_count": safe_float(trace.get("tool_candidate_count")),
            "generated_node_count": safe_float(trace.get("generated_node_count")),
            "completed_pool_count": safe_float(trace.get("completed_pool_count")),
            "final_rerank_candidate_count": safe_float(trace.get("final_rerank_candidate_count")),
            "level_count": len(levels),
            "level_input_total": level_input_total,
            "level_retained_total": level_retained_total,
            "level_pruned_total": level_pruned_total,
        },
    }
    return result, parsed_pool, comparisons


def percentile(sorted_values: Sequence[float], quantile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def distribution(values: Iterable[Any]) -> Dict[str, Any]:
    cleaned = sorted(value for item in values if (value := safe_float(item)) is not None)
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "std": statistics.pstdev(cleaned),
        "min": cleaned[0],
        "p25": percentile(cleaned, 0.25),
        "median": percentile(cleaned, 0.50),
        "p75": percentile(cleaned, 0.75),
        "p95": percentile(cleaned, 0.95),
        "max": cleaned[-1],
    }


def rate(values: Iterable[Any]) -> Dict[str, Any]:
    cleaned = [bool(value) for value in values]
    positives = sum(cleaned)
    return {
        "count": len(cleaned),
        "positive_count": positives,
        "rate": positives / len(cleaned) if cleaned else None,
    }


def micro_overlap(top1_rows: Sequence[Mapping[str, Any]], family: str) -> Dict[str, Any]:
    tp = sum(float(row[family]["tp"]) for row in top1_rows)
    fp = sum(float(row[family]["fp"]) for row in top1_rows)
    fn = sum(float(row[family]["fn"]) for row in top1_rows)
    precision = tp / (tp + fp) if tp + fp > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "dice": dice,
    }


def aggregate_ranked(sample_rows: Sequence[Mapping[str, Any]], cutoff_label: str) -> Dict[str, Any]:
    rows = [sample["ranked"][cutoff_label] for sample in sample_rows]
    tool_retrievable_pairs = [
        (sample["ranked"][cutoff_label], sample["retrieval_upper_bound"])
        for sample in sample_rows
        if sample["retrieval_upper_bound"]["all_gold_tools_retrieved"]
    ]
    agent_retrievable_pairs = [
        (sample["ranked"][cutoff_label], sample["retrieval_upper_bound"])
        for sample in sample_rows
        if sample["retrieval_upper_bound"]["full_gold_config_retrievable"]
    ]
    result: Dict[str, Any] = {
        "sample_count": len(rows),
        "candidate_availability": {
            "available_candidate_count": distribution(row["available_candidate_count"] for row in rows),
            "evaluated_candidate_count": distribution(row["evaluated_candidate_count"] for row in rows),
            "samples_with_at_least_cutoff": rate(row["has_at_least_cutoff_candidates"] for row in rows),
        },
        "hits": {},
        "rank_quality": {},
        "oracle_best": {},
        "candidate_mean_overlap": {},
        "retrieval_grounded_candidate_mean_overlap": {},
        "grounding": {},
        "retrieval_conditioned": {},
        "diversity": {},
    }
    for key in (
        "llm_em",
        "tool_complete_hit",
        "agent_complete_hit",
        "tool_ordered_em",
        "tool_unordered_em",
        "agent_ordered_em",
        "agent_unordered_em",
    ):
        result["hits"][key] = rate(row[f"{key}_hit"] for row in rows)
        result["rank_quality"][f"{key}_mrr"] = distribution(row[f"{key}_mrr"] for row in rows)
        result["rank_quality"][f"{key}_ndcg"] = distribution(row[f"{key}_ndcg"] for row in rows)
        result["rank_quality"][f"{key}_first_rank_among_hits"] = distribution(
            row[f"{key}_first_rank"] for row in rows
        )
    for family in ("tool", "component"):
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            name = f"oracle_best_{family}_{metric}"
            result["oracle_best"][name] = distribution(row[name] for row in rows)
            mean_name = f"mean_candidate_{family}_{metric}"
            result["candidate_mean_overlap"][f"{family}_{metric}"] = distribution(
                row[mean_name] for row in rows
            )
    for family in ("grounded_tool", "grounded_component"):
        public_family = family.removeprefix("grounded_")
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            mean_name = f"mean_candidate_{family}_{metric}"
            result["retrieval_grounded_candidate_mean_overlap"][
                f"{public_family}_{metric}"
            ] = distribution(row[mean_name] for row in rows)
    result["grounding"] = {
        "retrieval_grounded_candidate_fraction": distribution(
            row["retrieval_grounded_candidate_fraction"] for row in rows
        ),
        "zeroed_ungrounded_candidate_count": distribution(
            row["zeroed_ungrounded_candidate_count"] for row in rows
        ),
        "all_evaluated_candidates_retrieval_grounded": rate(
            row["all_evaluated_candidates_retrieval_grounded"] for row in rows
        ),
    }
    result["coverage"] = {
        "tool_recall": distribution(row["tool_coverage"] for row in rows),
        "component_recall": distribution(row["component_coverage"] for row in rows),
    }
    result["retrieval_conditioned"] = {
        "tool_complete_hit_given_all_gold_tools_retrieved": rate(
            row["tool_complete_hit_hit"] for row, _ in tool_retrievable_pairs
        ),
        "agent_complete_hit_given_full_gold_config_retrievable": rate(
            row["agent_complete_hit_hit"] for row, _ in agent_retrievable_pairs
        ),
        "tool_coverage_given_all_gold_tools_retrieved": distribution(
            row["tool_coverage"] for row, _ in tool_retrievable_pairs
        ),
        "component_coverage_given_full_gold_config_retrievable": distribution(
            row["component_coverage"] for row, _ in agent_retrievable_pairs
        ),
    }
    for key in (
        "top1_to_oracle_component_f1_regret",
        "top1_to_oracle_tool_f1_regret",
    ):
        result["oracle_best"][key] = distribution(row[key] for row in rows)
    for key in (
        "distinct_llm_count",
        "distinct_llm_ratio",
        "distinct_tool_bundle_count",
        "distinct_tool_bundle_ratio",
        "distinct_agent_config_count",
        "distinct_agent_config_ratio",
        "union_tool_count",
        "tool_bundle_pairwise_jaccard_distance",
        "component_pairwise_jaccard_distance",
    ):
        result["diversity"][key] = distribution(row[key] for row in rows)
    return result


def aggregate_core(sample_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    top1_rows = [sample["top1"] for sample in sample_rows]
    tool_retrievable_rows = [
        sample["top1"]
        for sample in sample_rows
        if sample["retrieval_upper_bound"]["all_gold_tools_retrieved"]
    ]
    agent_retrievable_rows = [
        sample["top1"]
        for sample in sample_rows
        if sample["retrieval_upper_bound"]["full_gold_config_retrievable"]
    ]
    result: Dict[str, Any] = {
        "sample_count": len(sample_rows),
        "exact": {},
        "recall_oriented": {},
        "retrieval_conditioned": {},
        "tool_overlap": {"macro": {}, "micro": micro_overlap(top1_rows, "tool")},
        "component_overlap": {"macro": {}, "micro": micro_overlap(top1_rows, "component")},
        "omission_and_hallucination": {},
        "format_and_validity": {},
        "cardinality": {},
        "scores": {},
    }
    for key in (
        "llm_em",
        "tool_ordered_em",
        "tool_unordered_em",
        "agent_ordered_em",
        "agent_unordered_em",
        "valid_agent_unordered_em",
    ):
        result["exact"][key] = rate(row[key] for row in top1_rows)
    for key in ("tool_complete_hit", "agent_complete_hit", "valid_agent_complete_hit"):
        result["recall_oriented"][key] = rate(row[key] for row in top1_rows)
    result["retrieval_conditioned"] = {
        "tool_complete_hit_given_all_gold_tools_retrieved": rate(
            row["tool_complete_hit"] for row in tool_retrievable_rows
        ),
        "agent_complete_hit_given_full_gold_config_retrievable": rate(
            row["agent_complete_hit"] for row in agent_retrievable_rows
        ),
        "tool_recall_given_all_gold_tools_retrieved": distribution(
            row["tool"]["recall"] for row in tool_retrievable_rows
        ),
        "component_recall_given_full_gold_config_retrievable": distribution(
            row["component"]["recall"] for row in agent_retrievable_rows
        ),
    }
    for family, output_key in (("tool", "tool_overlap"), ("component", "component_overlap")):
        for metric in ("precision", "recall", "f1", "jaccard", "dice"):
            result[output_key]["macro"][metric] = distribution(
                row[family][metric] for row in top1_rows
            )
    for key in (
        "any_gold_omitted",
        "any_label_relative_hallucination",
    ):
        result["omission_and_hallucination"][key] = rate(row[key] for row in top1_rows)
    for key in (
        "gold_omission_fraction",
        "label_relative_hallucination_fraction",
    ):
        result["omission_and_hallucination"][key] = distribution(row[key] for row in top1_rows)
    result["omission_and_hallucination"]["micro_gold_omission_rate"] = (
        result["tool_overlap"]["micro"]["fn"]
        / (result["tool_overlap"]["micro"]["tp"] + result["tool_overlap"]["micro"]["fn"])
        if result["tool_overlap"]["micro"]["tp"] + result["tool_overlap"]["micro"]["fn"] > 0
        else 0.0
    )
    result["omission_and_hallucination"]["micro_label_relative_hallucination_rate"] = (
        result["tool_overlap"]["micro"]["fp"]
        / (result["tool_overlap"]["micro"]["tp"] + result["tool_overlap"]["micro"]["fp"])
        if result["tool_overlap"]["micro"]["tp"] + result["tool_overlap"]["micro"]["fp"] > 0
        else 0.0
    )
    for key in (
        "parse_valid",
        "termination_valid",
        "separator_present",
        "duplicate_free",
        "llm_in_retrieved_candidates",
        "all_tools_in_retrieved_candidates",
    ):
        result["format_and_validity"][key] = rate(row[key] for row in top1_rows)
    result["format_and_validity"]["termination_reason_counts"] = dict(
        Counter(row.get("termination_reason") or "missing" for row in top1_rows)
    )
    result["format_and_validity"]["invalid_retrieved_tool_count"] = distribution(
        row["invalid_retrieved_tool_count"] for row in top1_rows
    )
    for key in (
        "predicted_tool_count",
        "gold_tool_count",
        "tool_count_error",
        "tool_count_absolute_error",
        "duplicate_tool_count",
    ):
        result["cardinality"][key] = distribution(row[key] for row in top1_rows)
    for key in ("tool_count_exact", "tool_over_generation", "tool_under_generation"):
        result["cardinality"][key] = rate(row[key] for row in top1_rows)
    for key in (
        "generator_logprob",
        "generator_token_count",
        "generator_avg_logprob",
        "critic_raw",
        "critic_sigmoid",
        "search_score",
    ):
        result["scores"][key] = distribution(row.get(key) for row in top1_rows)
    return result


def flatten_numeric_metrics(value: Any, prefix: str = "") -> Dict[str, float]:
    """Expose scalar metric leaves under dot paths for easy pandas/plot use."""
    output: Dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numeric_metrics(child, path))
    else:
        number = safe_float(value)
        if number is not None:
            output[prefix] = number
    return output


def aggregate_run(
    run_dir: Path,
    cutoffs: Sequence[int],
    include_sample_details: bool,
) -> Tuple[Dict[str, Any], set]:
    per_sample = run_dir / "per_sample"
    files = sorted(per_sample.glob("sample_*.json"))
    sample_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    first_sample_config: Dict[str, Any] = {}
    for path in files:
        try:
            sample, parsed_pool, comparisons = evaluate_sample(path)
            for cutoff in cutoffs:
                sample["ranked"][f"top_{cutoff}"] = ranked_metrics(
                    parsed_pool, comparisons, cutoff
                )
            sample["ranked"]["all"] = ranked_metrics(parsed_pool, comparisons, None)
            sample_rows.append(sample)
            if not first_sample_config:
                first_sample_config = load_json(path).get("config") or {}
        except Exception as exc:  # keep a bad sample visible without losing a whole run
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    failure_path = run_dir / "failures.jsonl"
    failure_line_count = 0
    failure_parse_error_count = 0
    if failure_path.exists():
        with failure_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                failure_line_count += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    failure_parse_error_count += 1

    manifest_path = run_dir / "sample_manifest.jsonl"
    manifest_count = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_count = sum(1 for line in handle if line.strip())

    metadata = {}
    for name in ("grid_point.json", "experiment_metadata.json", "run_summary.json"):
        path = run_dir / name
        if path.exists():
            try:
                metadata[name.removesuffix(".json")] = load_json(path)
            except Exception as exc:
                metadata[name.removesuffix(".json")] = {"read_error": str(exc)}

    result: Dict[str, Any] = {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "hyperparameters": {
            key: first_sample_config.get(key)
            for key in (
                "beam_retain_ratio",
                "beam_min_size",
                "beam_max_size",
                "llm_branch_factor",
                "tool_branch_factor",
                "num_results",
                "complete_pool_size",
                "search_score_mode",
                "critic_score_weight",
                "generator_score_weight",
                "search_length_penalty",
                "critic_start_real_tools",
                "max_tools",
                "min_tools",
                "dedup_tool_sets",
                'retrieval_mode',
                "cf_llm_topk",
                "cf_tool_bundle_topk",
                "semantic_tool_topk",
            )
        },
        "metadata": metadata,
        "data_quality": {
            "discovered_per_sample_files": len(files),
            "evaluated_samples": len(sample_rows),
            "evaluation_error_count": len(errors),
            "evaluation_errors": errors,
            "manifest_sample_count": manifest_count or None,
            "inference_failure_line_count": failure_line_count,
            "failure_json_parse_error_count": failure_parse_error_count,
            "unique_sample_id_count": len({row["sample_id"] for row in sample_rows}),
            "duplicate_sample_id_count": len(sample_rows)
            - len({row["sample_id"] for row in sample_rows}),
            "pool_source_counts": dict(Counter(row["pool_source"] for row in sample_rows)),
        },
        "metrics": {},
    }
    if sample_rows:
        result["metrics"]["top1"] = aggregate_core(sample_rows)
        result["metrics"]["retrieval_upper_bound"] = {
            "gold_llm_retrieved": rate(
                row["retrieval_upper_bound"]["gold_llm_retrieved"] for row in sample_rows
            ),
            "gold_tool_recall": distribution(
                row["retrieval_upper_bound"]["gold_tool_recall"] for row in sample_rows
            ),
            "all_gold_tools_retrieved": rate(
                row["retrieval_upper_bound"]["all_gold_tools_retrieved"] for row in sample_rows
            ),
            "full_gold_config_retrievable": rate(
                row["retrieval_upper_bound"]["full_gold_config_retrievable"] for row in sample_rows
            ),
            "retrieved_llm_count": distribution(
                row["retrieval_upper_bound"]["retrieved_llm_count"] for row in sample_rows
            ),
            "retrieved_tool_count": distribution(
                row["retrieval_upper_bound"]["retrieved_tool_count"] for row in sample_rows
            ),
        }
        result["metrics"]["ranked"] = {
            f"top_{cutoff}": aggregate_ranked(sample_rows, f"top_{cutoff}")
            for cutoff in cutoffs
        }
        result["metrics"]["ranked"]["all"] = aggregate_ranked(sample_rows, "all")
        result["metrics"]["search_efficiency"] = {
            key: distribution(row["search"].get(key) for row in sample_rows)
            for key in (
                "generation_latency_sec",
                "total_pipeline_latency_sec",
                "model_loading_latency_sec",
                "input_token_count",
                "llm_candidate_count",
                "tool_candidate_count",
                "generated_node_count",
                "completed_pool_count",
                "final_rerank_candidate_count",
                "level_count",
                "level_input_total",
                "level_retained_total",
                "level_pruned_total",
            )
        }
        result["metrics"]["search_efficiency"]["prompt_truncated"] = rate(
            row["search"]["prompt_truncated"] for row in sample_rows
        )
        result["metrics"]["search_efficiency"]["used_fallback_free_generate"] = rate(
            row["search"]["used_fallback_free_generate"] for row in sample_rows
        )
        correlation_keys = sorted(
            key
            for key in sample_rows[0]["ranking_score_alignment"]
            if not key.endswith("candidate_count")
        )
        result["metrics"]["ranking_score_alignment"] = {
            key: distribution(row["ranking_score_alignment"].get(key) for row in sample_rows)
            for key in correlation_keys
        }
        parts: Dict[str, List[Mapping[str, Any]]] = {}
        for row in sample_rows:
            parts.setdefault(str(row["part"]), []).append(row)
        result["metrics"]["by_part"] = {
            part: aggregate_core(rows) for part, rows in sorted(parts.items())
        }
        flat = flatten_numeric_metrics(result["metrics"])
        result["plot_metrics"] = dict(
            sorted(
                (key, value)
                for key, value in flat.items()
                if ".by_part." not in f".{key}."
                and (
                    key.endswith(".mean")
                    or key.endswith(".rate")
                    or key.endswith("sample_count")
                    or ".micro." in key
                    or key.endswith("micro_gold_omission_rate")
                    or key.endswith("micro_label_relative_hallucination_rate")
                )
            )
        )
    if include_sample_details:
        result["per_sample"] = sample_rows
    sample_ids = {str(row["sample_id"]) for row in sample_rows}
    return result, sample_ids


def metric_value(run: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    value: Any = run
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return safe_float(value)


def make_leaderboard(
    runs: Mapping[str, Mapping[str, Any]], path: Sequence[str], descending: bool = True
) -> List[Dict[str, Any]]:
    rows = []
    for name, run in runs.items():
        value = metric_value(run, path)
        if value is not None:
            rows.append({"run_name": name, "value": value})
    rows.sort(key=lambda row: row["value"], reverse=descending)
    return rows


def metric_definitions(cutoffs: Sequence[int]) -> Dict[str, Any]:
    return {
        "evaluation_unit": "One per-sample inference JSON; each run is macro-aggregated over its files.",
        "top1_prediction": "The first candidate in the highest-priority available prediction pool. Beam runs use the final rerank pool; non-beam runs use generation.results.",
        "ranked_pool": "Priority: final_rerank_pool_before_cap, final_candidates, sorted completed_pool, generation.results, top-level results, then generation.raw_outputs.",
        "cutoffs": list(cutoffs) + ["all"],
        "exact": {
            "llm_em": "Predicted and gold canonical LLM tokens are identical.",
            "tool_ordered_em": "Predicted and gold tool lists are exactly equal in order and length.",
            "tool_unordered_em": "Predicted and gold tool sets are equal, ignoring order.",
            "agent_ordered_em": "LLM EM and ordered tool-bundle EM are both true.",
            "agent_unordered_em": "LLM EM and unordered tool-bundle EM are both true.",
            "valid_agent_unordered_em": "Agent unordered EM plus valid parse, termination and no duplicate tools.",
        },
        "recall_oriented": {
            "tool_complete_hit": "All gold tools occur in one predicted candidate; extra predicted tools are allowed.",
            "agent_complete_hit": "The predicted LLM is the gold LLM and the same candidate contains every gold tool; extra tools are allowed.",
            "valid_agent_complete_hit": "Agent complete hit plus valid parse, termination and no duplicate tools.",
            "primary_metric": "Mean Candidate Component Recall@5 is the primary beam/critic-grid measure; complete hit remains a stricter diagnostic.",
            "retrieval_conditioned": "The same hit/coverage measures computed only where all required gold components were available to constrained generation; counts must be reported because this subset can be small.",
        },
        "set_overlap": {
            "precision": "|prediction intersect gold| / |prediction|.",
            "recall": "|prediction intersect gold| / |gold|.",
            "f1": "Harmonic mean of set precision and recall.",
            "jaccard": "|intersection| / |union|.",
            "dice": "2|intersection| / (|prediction| + |gold|).",
            "tool": "Computed over tool sets only.",
            "component": "Computed over the set containing the LLM and every tool.",
            "macro": "Mean of per-sample metrics.",
            "micro": "Computed after summing TP, FP and FN across samples.",
            "retrieval_grounded_soft_overlap": "Paper-facing soft overlap. If the predicted LLM or any predicted tool is absent from that sample's retrieval/context candidate universe, all soft-overlap scores for the entire candidate are set to 0. Raw overlap is retained separately for diagnosis.",
        },
        "errors": {
            "gold_omission": "Gold tool absent from the predicted tool set (FN).",
            "label_relative_hallucination": "Predicted tool absent from the gold tool set (FP); this is not a global-catalog validity claim.",
        },
        "ranking": {
            "hit_at_k": "At least one matching candidate occurs within the first k reranked candidates; complete-hit variants allow extra tools.",
            "coverage_at_k": "Maximum tool/component recall achieved by any single candidate within Top-K; components are not merged across candidates.",
            "mrr_at_k": "1/rank of the first matching candidate, or 0 if absent by k.",
            "ndcg_at_k": "For a single gold configuration, 1/log2(rank+1), or 0 if absent by k.",
            "oracle_best_at_k": "Best set-overlap value among the first k candidates.",
            "mean_candidate_overlap_at_k": "Unweighted mean of the per-candidate set-overlap metric across all available candidates within Top-K, followed by macro averaging across queries. This raw diagnostic does not zero retrieval-ungrounded candidates.",
            "retrieval_grounded_mean_candidate_overlap_at_k": "Primary paper-facing variant of mean candidate overlap. Each retrieval-ungrounded candidate contributes 0 before candidates and queries are averaged.",
            "regret_at_k": "Oracle-best F1 at k minus top-1 F1.",
        },
        "diversity": {
            "distinct_ratio": "Number of distinct values divided by evaluated candidates at k.",
            "pairwise_jaccard_distance": "Mean of 1 minus Jaccard similarity over candidate pairs.",
        },
        "retrieval_upper_bound": "Whether the gold LLM/tools occur in the retrieval/context candidate universe before generation.",
        "candidate_grounding": "A candidate is retrieval-grounded only when its parsed LLM and every parsed tool occur in that sample's retrieval/context candidate universe. This is the reproducible hallucination criterion available from the saved logs; it is stricter than merely matching part of the gold bundle.",
        "score_alignment": "Within-sample Pearson/Spearman correlation of critic, generator or final search score with component F1, component recall and tool recall.",
        "explanation": "Not scored. This experiment JSON exposes structural candidates; earlier protocol defers explanation evaluation separately.",
        "empty_set_convention": "Both-empty sets score 1. If only prediction is empty, precision/recall/F1 are 0. If only gold is empty, precision=0, recall=1 and F1=0.",
    }


def discover_run_dirs(root: Path) -> List[Path]:
    runs_root = root / "runs"
    search_root = runs_root if runs_root.is_dir() else root
    run_dirs = sorted({path.parent for path in search_root.rglob("per_sample") if path.is_dir()})
    if not run_dirs and (root / "per_sample").is_dir():
        run_dirs = [root]
    return run_dirs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <experiment-root>/evaluation/<default filename>",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Rank cutoffs; default: 1 3 5 10 20",
    )
    parser.add_argument(
        "--include-sample-details",
        action="store_true",
        help="Also include per-sample rows; off by default to keep the JSON statistical.",
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    cutoffs = sorted({value for value in args.cutoffs if value > 0})
    if not root.exists():
        print(f"ERROR: experiment root does not exist: {root}", file=sys.stderr)
        return 2
    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"ERROR: no per_sample directories found under: {root}", file=sys.stderr)
        return 2

    runs: Dict[str, Any] = {}
    sample_sets: Dict[str, set] = {}
    for run_dir in run_dirs:
        print(f"Evaluating {run_dir.name} ...", flush=True)
        result, sample_ids = aggregate_run(
            run_dir, cutoffs, bool(args.include_sample_details)
        )
        runs[run_dir.name] = result
        sample_sets[run_dir.name] = sample_ids

    all_sets = list(sample_sets.values())
    union_ids = set().union(*all_sets) if all_sets else set()
    common_ids = set.intersection(*all_sets) if all_sets else set()
    alignment = {
        "run_count": len(runs),
        "union_sample_count": len(union_ids),
        "common_sample_count": len(common_ids),
        "all_runs_have_identical_sample_sets": all(ids == all_sets[0] for ids in all_sets[1:])
        if all_sets
        else True,
        "per_run": {
            name: {
                "sample_count": len(ids),
                "missing_from_union_count": len(union_ids - ids),
                "extra_beyond_common_count": len(ids - common_ids),
            }
            for name, ids in sample_sets.items()
        },
    }

    summary = {
        "schema_version": "beam_critic_eval_v1.3_grounded_soft_topk_overlap",
        "created_at_utc": utc_now(),
        "evaluation_name": "Exp2 Beam/Critic Grid Evaluation",
        "purpose": "Compare generation settings primarily by retrieval-grounded mean soft component overlap across all Top-K candidates. A candidate containing any component outside its per-sample retrieval universe receives zero paper-facing soft-overlap credit; raw overlap is retained for diagnosis.",
        "experiment_root": str(root),
        "evaluator": str(Path(__file__).resolve()),
        "metric_definitions": metric_definitions(cutoffs),
        "sample_alignment": alignment,
        "runs": runs,
        "leaderboards": {
            "top5_grounded_mean_candidate_component_recall_desc": make_leaderboard(
                runs,
                ("metrics", "ranked", "top_5", "retrieval_grounded_candidate_mean_overlap", "component_recall", "mean"),
            ),
            "top5_grounded_mean_candidate_component_f1_desc": make_leaderboard(
                runs,
                ("metrics", "ranked", "top_5", "retrieval_grounded_candidate_mean_overlap", "component_f1", "mean"),
            ),
            "top1_agent_complete_hit_desc": make_leaderboard(
                runs, ("metrics", "top1", "recall_oriented", "agent_complete_hit", "rate")
            ),
            "top1_tool_recall_desc": make_leaderboard(
                runs, ("metrics", "top1", "tool_overlap", "macro", "recall", "mean")
            ),
            "top5_agent_complete_hit_desc": make_leaderboard(
                runs, ("metrics", "ranked", "top_5", "hits", "agent_complete_hit", "rate")
            ),
            "top5_tool_coverage_desc": make_leaderboard(
                runs, ("metrics", "ranked", "top_5", "coverage", "tool_recall", "mean")
            ),
            "top1_agent_unordered_em_desc": make_leaderboard(
                runs, ("metrics", "top1", "exact", "agent_unordered_em", "rate")
            ),
            "top1_component_f1_desc": make_leaderboard(
                runs, ("metrics", "top1", "component_overlap", "macro", "f1", "mean")
            ),
            "top5_agent_unordered_hit_desc": make_leaderboard(
                runs, ("metrics", "ranked", "top_5", "hits", "agent_unordered_em", "rate")
            ),
            "generation_latency_sec_asc": make_leaderboard(
                runs,
                ("metrics", "search_efficiency", "generation_latency_sec", "mean"),
                descending=False,
            ),
        },
    }

    output = args.output
    if output is None:
        output = root / "evaluation" / DEFAULT_OUTPUT_NAME
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            allow_nan=False,
        )
        handle.write("\n")
    print(f"Saved: {output}")
    print(
        f"Runs={len(runs)} | union_samples={len(union_ids)} | "
        f"common_samples={len(common_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())