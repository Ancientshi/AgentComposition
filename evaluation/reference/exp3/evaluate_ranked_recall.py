#!/usr/bin/env python3
'Evaluate ranked complete recall and rank-discounted component recall.\n\nUsage:\n    python evaluate_ranked_recall_minimal_v3.py         --experiment-root /root/yunxshi/NIPS2026/outputs/exp2_v13_ablation_beam_critic_seed42_first100_new\n\nThe script reuses the gold/candidate parsing logic from\nevaluate_beam_critic_experiments_PartII.py, so the evaluation protocol stays\nconsistent with the existing PartII evaluator.\n\nOutputs:\n    <experiment-root>/evaluation/ranked_recall_metrics.md\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import importlib.util
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_ROOT = Path(
    str(AC_ROOT / 'outputs/') +
    "exp2_v13_ablation_beam_critic_seed42_first100"
)
CUTOFFS = (5, 10, 20)


def load_base_evaluator():
    """Load the existing PartII evaluator from the same directory."""
    here = Path(__file__).resolve().parent

    # Normal filename on the user's server.
    exact = here / "evaluate_beam_critic_experiments_PartII.py"
    candidates = [exact] if exact.exists() else []

    # Also tolerate downloaded copies such as "...PartII(1).py".
    candidates += sorted(here.glob("evaluate_beam_critic_experiments_PartII*.py"))

    seen = set()
    candidates = [p for p in candidates if not (str(p) in seen or seen.add(str(p)))]

    if not candidates:
        raise RuntimeError(
            "Cannot find evaluate_beam_critic_experiments_PartII.py beside this script."
        )

    path = candidates[0]
    spec = importlib.util.spec_from_file_location("_partii_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "discover_run_dirs") or not hasattr(module, "evaluate_sample"):
        raise RuntimeError(
            f"Base evaluator lacks discover_run_dirs/evaluate_sample: {path}"
        )
    return module, path


BASE, BASE_PATH = load_base_evaluator()
discover_run_dirs = BASE.discover_run_dirs
evaluate_sample = BASE.evaluate_sample


def discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def rdcr_at_k(component_recalls: List[float], k: int) -> float:
    """Rank-Discounted Component Recall@K.

    Missing positions up to K are assigned recall 0.
    """
    numerator = sum(
        discount(r) * component_recalls[r - 1]
        for r in range(1, min(k, len(component_recalls)) + 1)
    )
    denominator = sum(discount(r) for r in range(1, k + 1))
    return numerator / denominator if denominator else 0.0


def oracle_rdcr_at_k(component_recalls: List[float], k: int) -> float:
    """Oracle-Ranked Rank-Discounted Component Recall@K.

    Keep exactly the same Top-K generated candidate set as RDCR@K, but reorder
    those candidates by their gold component recall in descending order before
    applying the same logarithmic rank discount. Missing positions up to K are
    assigned recall 0.

    This isolates the ranking upper bound of the already generated candidate set.
    """
    oracle_recalls = sorted(component_recalls[:k], reverse=True)
    numerator = sum(
        discount(r) * oracle_recalls[r - 1]
        for r in range(1, len(oracle_recalls) + 1)
    )
    denominator = sum(discount(r) for r in range(1, k + 1))
    return numerator / denominator if denominator else 0.0


def first_true_rank(values: List[bool]) -> Optional[int]:
    for rank, value in enumerate(values, 1):
        if value:
            return rank
    return None


def percentile(values: List[int], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac



def search_union_ceiling(
    parsed_pool: List[Dict[str, Any]],
    gold_llm: Optional[str],
    gold_tools: Sequence[str],
    k: int,
) -> Dict[str, float]:
    """Idealized recall ceiling from the union of components available in Top-K.

    This is deliberately stronger than oracle-best single-candidate recall:
    components may be recombined ACROSS the first K generated candidates into one
    hypothetical ideal candidate, which is then assumed to be ranked first.

    Full component recall treats the gold LLM as one component plus each unique
    gold tool as one component. Extra predicted components do not hurt recall.
    """
    topk = parsed_pool[:k]

    available_llms = {
        str(candidate.get("llm"))
        for candidate in topk
        if candidate.get("llm") is not None
    }
    available_tools = set()
    for candidate in topk:
        available_tools.update(
            str(tool)
            for tool in (candidate.get("tools") or [])
            if tool is not None
        )

    gold_tool_set = {str(tool) for tool in gold_tools if tool is not None}
    gold_tool_hits = len(gold_tool_set & available_tools)
    tool_denominator = len(gold_tool_set)
    tool_recall = (
        gold_tool_hits / tool_denominator
        if tool_denominator > 0
        else 1.0
    )
    tool_complete = float(gold_tool_hits == tool_denominator)

    has_gold_llm = gold_llm is not None and str(gold_llm) != ""
    llm_hit = int(str(gold_llm) in available_llms) if has_gold_llm else 0
    component_denominator = tool_denominator + int(has_gold_llm)
    component_hits = gold_tool_hits + llm_hit
    component_recall = (
        component_hits / component_denominator
        if component_denominator > 0
        else 1.0
    )
    component_complete = float(component_hits == component_denominator)

    return {
        "component_recall": component_recall,
        "component_complete": component_complete,
        "tool_recall": tool_recall,
        "tool_complete": tool_complete,
    }

def evaluate_one(path: Path) -> Dict[str, Any]:
    sample, parsed_pool, comparisons = evaluate_sample(path)

    gold = sample.get("gold") or {}
    gold_llm = gold.get("llm")
    gold_tools = list(gold.get("tools") or [])

    component_recalls = [
        float(row["component"]["recall"])
        for row in comparisons
    ]
    tool_recalls = [
        float(row["tool"]["recall"])
        for row in comparisons
    ]

    # Agent-level complete recall:
    #   correct LLM + all gold tools included in the SAME candidate.
    # Extra predicted tools are allowed.
    complete_hits = [
        bool(row["agent_complete_hit"])
        for row in comparisons
    ]
    first_rank = first_true_rank(complete_hits)

    # Tool-only complete recall (relaxed):
    #   all gold tools included in the SAME candidate.
    # LLM match is deliberately ignored and extra tools are allowed.
    tool_complete_hits = [
        bool(row["tool_complete_hit"])
        for row in comparisons
    ]
    first_tool_rank = first_true_rank(tool_complete_hits)

    out: Dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "qid": sample.get("qid"),
        "pool_size": len(parsed_pool),
        "first_complete_rank": first_rank,
        "first_tool_complete_rank": first_tool_rank,
        "top1_component_recall": (
            component_recalls[0] if component_recalls else 0.0
        ),
        "top1_tool_recall": (
            tool_recalls[0] if tool_recalls else 0.0
        ),
    }

    for k in CUTOFFS:
        hit = first_rank is not None and first_rank <= k
        out[f"hit@{k}"] = float(hit)
        out[f"mrr@{k}"] = 1.0 / first_rank if hit else 0.0
        out[f"rdcr@{k}"] = rdcr_at_k(component_recalls, k)

        # Oracle ranking upper bound: use the SAME actually generated Top-K
        # candidates as RDCR@K, but sort them by gold component recall before
        # applying the identical logarithmic rank discount.
        out[f"oracle_rdcr@{k}"] = oracle_rdcr_at_k(component_recalls, k)

        tool_hit = first_tool_rank is not None and first_tool_rank <= k
        out[f"tool_hit@{k}"] = float(tool_hit)
        out[f"tool_mrr@{k}"] = 1.0 / first_tool_rank if tool_hit else 0.0
        out[f"rdtr@{k}"] = rdcr_at_k(tool_recalls, k)

        out[f"available@{k}"] = float(len(parsed_pool) >= k)

        # ------------------------------------------------------------------
        # NEW: search -> generation -> ranking ceiling decomposition.
        # Existing metrics above are intentionally unchanged.
        # ------------------------------------------------------------------
        ceiling = search_union_ceiling(
            parsed_pool=parsed_pool,
            gold_llm=gold_llm,
            gold_tools=gold_tools,
            k=k,
        )

        # Search-space / ideal-assembly ceiling: union gold components across
        # Top-K candidates, then hypothetically assemble them into ONE perfect
        # candidate and place it at rank 1.
        out[f"search_union_component_recall@{k}"] = ceiling["component_recall"]
        out[f"search_union_complete@{k}"] = ceiling["component_complete"]
        out[f"search_union_tool_recall@{k}"] = ceiling["tool_recall"]
        out[f"search_union_tool_complete@{k}"] = ceiling["tool_complete"]

        # Alias matching the interpretation used in the paper discussion:
        # ideal Rank-1 recall equals the search-union ceiling under perfect
        # assembly and perfect ranking.
        out[f"ideal_rank1_component_recall@{k}"] = ceiling["component_recall"]
        out[f"ideal_rank1_tool_recall@{k}"] = ceiling["tool_recall"]

        # Generation ceiling: the best recall among ACTUALLY generated Top-K
        # candidates. Unlike search_union_* above, components cannot be merged
        # across candidates. This is the recall analogue of oracle-best F1.
        out[f"oracle_best_component_recall@{k}"] = (
            max(component_recalls[:k]) if component_recalls[:k] else 0.0
        )
        out[f"oracle_best_tool_recall@{k}"] = (
            max(tool_recalls[:k]) if tool_recalls[:k] else 0.0
        )

    return out


def aggregate_run(run_dir: Path) -> Dict[str, Any]:
    files = sorted((run_dir / "per_sample").glob("sample_*.json"))
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for path in files:
        try:
            rows.append(evaluate_one(path))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    first_ranks = [
        row["first_complete_rank"]
        for row in rows
        if row["first_complete_rank"] is not None
    ]
    first_tool_ranks = [
        row["first_tool_complete_rank"]
        for row in rows
        if row["first_tool_complete_rank"] is not None
    ]

    def make_rank_buckets(key: str) -> Counter:
        buckets = Counter()
        for row in rows:
            rank = row[key]
            if rank is None:
                buckets["No hit"] += 1
            elif rank == 1:
                buckets["1"] += 1
            elif rank <= 5:
                buckets["2-5"] += 1
            elif rank <= 10:
                buckets["6-10"] += 1
            elif rank <= 20:
                buckets["11-20"] += 1
            else:
                buckets[">20"] += 1
        return buckets

    buckets = make_rank_buckets("first_complete_rank")
    tool_buckets = make_rank_buckets("first_tool_complete_rank")

    result: Dict[str, Any] = {
        "run": run_dir.name,
        "discovered_files": len(files),
        "n": len(rows),
        "errors": errors,
        "pool_mean": (
            statistics.fmean(row["pool_size"] for row in rows)
            if rows else 0.0
        ),
        "any_hit": len(first_ranks) / len(rows) if rows else 0.0,
        "first_mean": statistics.fmean(first_ranks) if first_ranks else None,
        "first_median": statistics.median(first_ranks) if first_ranks else None,
        "first_p25": percentile(first_ranks, 0.25),
        "first_p75": percentile(first_ranks, 0.75),
        "buckets": buckets,
        "tool_any_hit": len(first_tool_ranks) / len(rows) if rows else 0.0,
        "tool_first_mean": (
            statistics.fmean(first_tool_ranks) if first_tool_ranks else None
        ),
        "tool_first_median": (
            statistics.median(first_tool_ranks) if first_tool_ranks else None
        ),
        "tool_first_p25": percentile(first_tool_ranks, 0.25),
        "tool_first_p75": percentile(first_tool_ranks, 0.75),
        "tool_buckets": tool_buckets,
    }

    for k in CUTOFFS:
        for metric in (
            "hit", "mrr", "rdcr",
            "tool_hit", "tool_mrr", "rdtr",
            "available",
        ):
            values = [float(row[f"{metric}@{k}"]) for row in rows]
            result[f"{metric}@{k}"] = (
                statistics.fmean(values) if values else 0.0
            )
        rdcr_values = [float(row[f"rdcr@{k}"]) for row in rows]
        result[f"rdcr_std@{k}"] = (
            statistics.pstdev(rdcr_values) if rdcr_values else 0.0
        )

        oracle_rdcr_values = [float(row[f"oracle_rdcr@{k}"]) for row in rows]
        result[f"oracle_rdcr@{k}"] = (
            statistics.fmean(oracle_rdcr_values) if oracle_rdcr_values else 0.0
        )
        result[f"oracle_rdcr_std@{k}"] = (
            statistics.pstdev(oracle_rdcr_values)
            if oracle_rdcr_values else 0.0
        )
        # Pure ranking headroom under the same Top-K candidate set and the same
        # RDCR discount; only the order changes.
        result[f"oracle_rdcr_gap@{k}"] = (
            result[f"oracle_rdcr@{k}"] - result[f"rdcr@{k}"]
        )

        rdtr_values = [float(row[f"rdtr@{k}"]) for row in rows]
        result[f"rdtr_std@{k}"] = (
            statistics.pstdev(rdtr_values) if rdtr_values else 0.0
        )

        # NEW ceiling-decomposition metrics (macro averaged over queries).
        for metric in (
            "search_union_component_recall",
            "search_union_complete",
            "search_union_tool_recall",
            "search_union_tool_complete",
            "ideal_rank1_component_recall",
            "ideal_rank1_tool_recall",
            "oracle_best_component_recall",
            "oracle_best_tool_recall",
        ):
            values = [float(row[f"{metric}@{k}"]) for row in rows]
            result[f"{metric}@{k}"] = (
                statistics.fmean(values) if values else 0.0
            )

        # Decompose the remaining headroom at each K. Negative values are kept
        # rather than clipped so unusual/free-generation behavior is visible.
        result[f"assembly_gap_component@{k}"] = (
            result[f"search_union_component_recall@{k}"]
            - result[f"oracle_best_component_recall@{k}"]
        )
        result[f"ranking_gap_component@{k}"] = (
            result[f"oracle_best_component_recall@{k}"]
            - (statistics.fmean(row["top1_component_recall"] for row in rows) if rows else 0.0)
        )
        result[f"total_gap_component@{k}"] = (
            result[f"search_union_component_recall@{k}"]
            - (statistics.fmean(row["top1_component_recall"] for row in rows) if rows else 0.0)
        )

        result[f"assembly_gap_tool@{k}"] = (
            result[f"search_union_tool_recall@{k}"]
            - result[f"oracle_best_tool_recall@{k}"]
        )
        result[f"ranking_gap_tool@{k}"] = (
            result[f"oracle_best_tool_recall@{k}"]
            - (statistics.fmean(row["top1_tool_recall"] for row in rows) if rows else 0.0)
        )
        result[f"total_gap_tool@{k}"] = (
            result[f"search_union_tool_recall@{k}"]
            - (statistics.fmean(row["top1_tool_recall"] for row in rows) if rows else 0.0)
        )

    result["top1_component_recall"] = (
        statistics.fmean(row["top1_component_recall"] for row in rows) if rows else 0.0
    )
    result["top1_tool_recall"] = (
        statistics.fmean(row["top1_tool_recall"] for row in rows) if rows else 0.0
    )

    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def f2(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def build_markdown(
    results: List[Dict[str, Any]],
    experiment_root: Path,
) -> str:
    md = [
        "# Ranked Complete-Recall and Component-Recall Evaluation",
        "",
        "## Evaluation protocol",
        "",
        f"- Experiment root: `{experiment_root}`",
        f"- Base evaluator: `{BASE_PATH}`",
        "- Gold agent: PartIII target LLM + PartII backup-target tools.",
        "- Ranked pool: the same final rerank pool selected by the existing PartII evaluator.",
        "",
        "For gold agent $G=(m^*,T^*)$ and candidate $A_r=(m_r,T_r)$, "
        "a **complete-recall hit** requires $m_r=m^*$ and $T^*\\subseteq T_r$. "
        "Extra predicted tools are therefore allowed.",
        "",
        "**CR-Hit@K** reports whether at least one complete-recall agent occurs "
        "within Top-$K$. **CR-MRR@K** rewards a complete-recall agent for appearing "
        "earlier in the ranking.",
        "",
        "For each candidate, component recall is",
        "",
        "$$R_r=\\frac{|C_r\\cap C_G|}{|C_G|},$$",
        "",
        "where the component set contains the LLM and all tools. "
        "The list-level **Rank-Discounted Component Recall (RDCR@K)** is",
        "",
        "$$\\mathrm{RDCR@K}="
        "\\frac{\\sum_{r=1}^{K}R_r/\\log_2(r+1)}"
        "{\\sum_{r=1}^{K}1/\\log_2(r+1)}.$$",
        "",
        "Higher-ranked candidates receive greater weight. Missing ranks are assigned "
        "recall 0. All reported values are macro-averaged over queries.",
        "",
        "**Oracle-RDCR@K** uses exactly the same generated Top-K candidate set and "
        "the same logarithmic discount as RDCR@K, but first sorts candidates by their "
        "gold component recall in descending order. It therefore gives the ranking "
        "upper bound of the generated candidate set. The difference "
        "**Oracle-RDCR@K - RDCR@K** isolates ranking headroom.",
        "",
        "A relaxed **tool-only** variant is also reported. It ignores the LLM identity "
        "and considers a candidate successful whenever $T^*\\subseteq T_r$. "
        "Its rank-discounted graded metric, **RDTR@K**, applies the same logarithmic "
        "discount to per-candidate tool recall only.",
        "",
        "## Main results",
        "",
        "| Run | N | CR-Hit@5 | CR-Hit@10 | CR-Hit@20 | CR-MRR@5 | CR-MRR@10 | CR-MRR@20 | RDCR@5 | RDCR@10 | RDCR@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md.append(
            f"| {row['run']} | {row['n']} | "
            f"{pct(row['hit@5'])} | {pct(row['hit@10'])} | {pct(row['hit@20'])} | "
            f"{row['mrr@5']:.4f} | {row['mrr@10']:.4f} | {row['mrr@20']:.4f} | "
            f"{row['rdcr@5']:.4f} | {row['rdcr@10']:.4f} | {row['rdcr@20']:.4f} |"
        )

    md += [
        "",
        "### Oracle-ranked RDCR",
        "",
        "Oracle-RDCR keeps the actually generated Top-K candidate set fixed, "
        "sorts candidates by component recall in descending order, and then applies "
        "the same logarithmic rank discount as RDCR. Thus Oracle-RDCR and RDCR are "
        "directly comparable; their difference is the ranking headroom.",
        "",
        "| Run | Oracle-RDCR@5 | RDCR@5 | Gap@5 | Oracle-RDCR@10 | RDCR@10 | Gap@10 | Oracle-RDCR@20 | RDCR@20 | Gap@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md.append(
            f"| {row['run']} | "
            f"{row['oracle_rdcr@5']:.4f} | {row['rdcr@5']:.4f} | {row['oracle_rdcr_gap@5']:.4f} | "
            f"{row['oracle_rdcr@10']:.4f} | {row['rdcr@10']:.4f} | {row['oracle_rdcr_gap@10']:.4f} | "
            f"{row['oracle_rdcr@20']:.4f} | {row['rdcr@20']:.4f} | {row['oracle_rdcr_gap@20']:.4f} |"
        )

    md += [
        "",
        "### Relaxed tool-only results",
        "",
        "This table ignores LLM matching. A hit requires only that one candidate "
        "contains every gold tool; extra tools are allowed.",
        "",
        "| Run | N | Tool-Hit@5 | Tool-Hit@10 | Tool-Hit@20 | Tool-MRR@5 | Tool-MRR@10 | Tool-MRR@20 | RDTR@5 | RDTR@10 | RDTR@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md.append(
            f"| {row['run']} | {row['n']} | "
            f"{pct(row['tool_hit@5'])} | {pct(row['tool_hit@10'])} | "
            f"{pct(row['tool_hit@20'])} | "
            f"{row['tool_mrr@5']:.4f} | {row['tool_mrr@10']:.4f} | "
            f"{row['tool_mrr@20']:.4f} | "
            f"{row['rdtr@5']:.4f} | {row['rdtr@10']:.4f} | "
            f"{row['rdtr@20']:.4f} |"
        )

    md += [
        "",
        "## First complete-recall rank",
        "",
        "Rank statistics below are conditional on samples for which at least one "
        "complete-recall candidate exists in the evaluated rerank pool.",
        "",
        "| Run | Any hit | Mean rank | Median | P25 | P75 | Rank 1 | Rank 2-5 | Rank 6-10 | Rank 11-20 | >20 | No hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        b = row["buckets"]
        md.append(
            f"| {row['run']} | {pct(row['any_hit'])} | "
            f"{f2(row['first_mean'])} | {f2(row['first_median'])} | "
            f"{f2(row['first_p25'])} | {f2(row['first_p75'])} | "
            f"{b['1']} | {b['2-5']} | {b['6-10']} | "
            f"{b['11-20']} | {b['>20']} | {b['No hit']} |"
        )

    md += [
        "",
        "### Relaxed tool-only first complete-recall rank",
        "",
        "The LLM identity is ignored here. Rank is determined by the first candidate "
        "whose tool set covers all gold tools.",
        "",
        "| Run | Any tool hit | Mean rank | Median | P25 | P75 | Rank 1 | Rank 2-5 | Rank 6-10 | Rank 11-20 | >20 | No hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        b = row["tool_buckets"]
        md.append(
            f"| {row['run']} | {pct(row['tool_any_hit'])} | "
            f"{f2(row['tool_first_mean'])} | {f2(row['tool_first_median'])} | "
            f"{f2(row['tool_first_p25'])} | {f2(row['tool_first_p75'])} | "
            f"{b['1']} | {b['2-5']} | {b['6-10']} | "
            f"{b['11-20']} | {b['>20']} | {b['No hit']} |"
        )

    # ----------------------------------------------------------------------
    # NEW: capability decomposition requested for the paper's panel (c).
    # ----------------------------------------------------------------------
    md += [
        "",
        "## Search-to-generation upper bound (new)",
        "",
        "This section separates three different limits that can otherwise be confused:",
        "",
        "1. **Search-Union Ceiling Recall@K**: take the union of gold components that "
        "appear anywhere in the first K generated candidates, hypothetically recombine "
        "those available components into one ideal candidate, and place it at rank 1. "
        "This is the idealized search-space upper bound under perfect assembly and ranking.",
        "2. **Oracle-Best Generated Recall@K**: choose the single actually generated "
        "candidate with the highest recall within Top-K. Components are not merged across "
        "candidates. This is the recall analogue of Oracle-Best F1.",
        "3. **Actual Top-1 Recall**: recall of the candidate the system really ranks first.",
        "",
        "Therefore, **Assembly gap = Search-Union Ceiling - Oracle-Best Generated**, "
        "and **Ranking gap = Oracle-Best Generated - Actual Top-1**. "
        "A complete-recoverable sample is one for which the Top-K union already contains "
        "every gold component, so an ideal assembler could achieve 100% recall at rank 1.",
        "",
        "### Component-level ceiling decomposition",
        "",
        "| Run | K | Search-Union Ceiling Recall@K | Complete-Recoverable@K | Oracle-Best Generated Recall@K | Actual Top-1 Recall | Assembly gap | Ranking gap | Total gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        for k in CUTOFFS:
            md.append(
                f"| {row['run']} | {k} | "
                f"{pct(row[f'search_union_component_recall@{k}'])} | "
                f"{pct(row[f'search_union_complete@{k}'])} | "
                f"{pct(row[f'oracle_best_component_recall@{k}'])} | "
                f"{pct(row['top1_component_recall'])} | "
                f"{pct(row[f'assembly_gap_component@{k}'])} | "
                f"{pct(row[f'ranking_gap_component@{k}'])} | "
                f"{pct(row[f'total_gap_component@{k}'])} |"
            )

    md += [
        "",
        "### Panel (c) summary at K=10",
        "",
        "This compact table is intended for direct comparison with the existing "
        "Top-10 diagnostics. **Ideal Rank-1 Recall@10** is numerically identical to "
        "Search-Union Ceiling Recall@10 because the hypothetical candidate is assumed "
        "to assemble every available gold component and be placed first.",
        "",
        "| Run | Ideal Rank-1 Component Recall@10 | Complete-Recoverable@10 | Oracle-Best Generated Recall@10 | Oracle-RDCR@10 | RDCR@10 | Oracle-RDCR gap | Actual Top-1 Recall | Generation/Assembly loss | Ranking loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md.append(
            f"| {row['run']} | "
            f"{pct(row['ideal_rank1_component_recall@10'])} | "
            f"{pct(row['search_union_complete@10'])} | "
            f"{pct(row['oracle_best_component_recall@10'])} | "
            f"{pct(row['oracle_rdcr@10'])} | "
            f"{pct(row['rdcr@10'])} | "
            f"{pct(row['oracle_rdcr_gap@10'])} | "
            f"{pct(row['top1_component_recall'])} | "
            f"{pct(row['assembly_gap_component@10'])} | "
            f"{pct(row['ranking_gap_component@10'])} |"
        )

    md += [
        "",
        "### Relaxed tool-only ceiling decomposition",
        "",
        "The same analysis is repeated after ignoring LLM identity. "
        "Search-Union Tool Recall@K is the fraction of gold tools appearing anywhere "
        "in the first K candidates; Tool Complete-Recoverable@K requires all gold tools "
        "to be present in that union.",
        "",
        "| Run | K | Search-Union Tool Recall@K | Tool Complete-Recoverable@K | Oracle-Best Generated Tool Recall@K | Actual Top-1 Tool Recall | Assembly gap | Ranking gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        for k in CUTOFFS:
            md.append(
                f"| {row['run']} | {k} | "
                f"{pct(row[f'search_union_tool_recall@{k}'])} | "
                f"{pct(row[f'search_union_tool_complete@{k}'])} | "
                f"{pct(row[f'oracle_best_tool_recall@{k}'])} | "
                f"{pct(row['top1_tool_recall'])} | "
                f"{pct(row[f'assembly_gap_tool@{k}'])} | "
                f"{pct(row[f'ranking_gap_tool@{k}'])} |"
            )

    md += [
        "",
        "## Candidate availability and evaluation diagnostics",
        "",
        "| Run | Files found | Evaluated | Mean pool | >=5 | >=10 | >=20 | RDCR@5 mean±std | RDCR@10 mean±std | RDCR@20 mean±std | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results:
        md.append(
            f"| {row['run']} | {row['discovered_files']} | {row['n']} | "
            f"{row['pool_mean']:.2f} | "
            f"{pct(row['available@5'])} | {pct(row['available@10'])} | "
            f"{pct(row['available@20'])} | "
            f"{row['rdcr@5']:.4f} ± {row['rdcr_std@5']:.4f} | "
            f"{row['rdcr@10']:.4f} ± {row['rdcr_std@10']:.4f} | "
            f"{row['rdcr@20']:.4f} ± {row['rdcr_std@20']:.4f} | "
            f"{len(row['errors'])} |"
        )

    all_errors = [
        (row["run"], err)
        for row in results
        for err in row["errors"]
    ]
    if all_errors:
        md += ["", "## Evaluation errors", ""]
        for run, err in all_errors[:50]:
            md.append(f"- `{run}`: {err}")
        if len(all_errors) > 50:
            md.append(f"- ... plus {len(all_errors) - 50} additional errors.")

    md += [
        "",
        "## Interpretation",
        "",
        "CR-Hit@K measures whether search recovers at least one fully recall-complete "
        "agent. CR-MRR@K measures how early that agent is ranked. RDCR@K provides "
        "graded credit to the full Top-K list according to component recall while "
        "discounting lower-ranked candidates. Tool-Hit@K, Tool-MRR@K, and RDTR@K "
        "provide the relaxed tool-only counterparts without requiring the predicted "
        "LLM to match the gold LLM.",
        "",
        "The new ceiling decomposition can be read as **availability -> generation -> "
        "oracle ranking -> actual ranking**. Search-Union Ceiling Recall@K asks what would "
        "be possible if all gold components exposed anywhere in Top-K could be perfectly "
        "assembled into one rank-1 candidate. Oracle-Best Generated Recall@K asks how much "
        "of that ceiling is realized by the best single generated candidate. Oracle-RDCR@K "
        "then measures the best rank-discounted list quality obtainable from the same "
        "generated Top-K candidates under perfect recall-based ordering, while RDCR@K "
        "measures the system's actual ordering. Their difference therefore isolates the "
        "remaining ranking headroom.",
        "",
    ]
    return "\n".join(md)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Experiment root containing run/per_sample directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <experiment-root>/evaluation/ranked_recall_metrics.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "evaluation" / "ranked_recall_metrics.md"
    )

    print(f"[INFO] Experiment root: {root}")
    print(f"[INFO] Base evaluator: {BASE_PATH}")

    if not root.exists():
        print(f"ERROR: experiment root not found: {root}", file=sys.stderr)
        return 2

    run_dirs = discover_run_dirs(root)
    if not run_dirs:
        print(f"ERROR: no per_sample directories found under: {root}", file=sys.stderr)
        return 2

    print(f"[INFO] Discovered {len(run_dirs)} run(s).")

    results: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        print(f"Evaluating {run_dir.name} ...", flush=True)
        row = aggregate_run(run_dir)
        print(
            f"  files={row['discovered_files']} evaluated={row['n']} "
            f"errors={len(row['errors'])} mean_pool={row['pool_mean']:.2f} "
            f"CR-Hit@5={row['hit@5']:.4f} RDCR@5={row['rdcr@5']:.4f} "
            f"Tool-Hit@5={row['tool_hit@5']:.4f} RDTR@5={row['rdtr@5']:.4f}"
        )
        print(
            f"  ceiling@10={row['search_union_component_recall@10']:.4f} "
            f"complete_recoverable@10={row['search_union_complete@10']:.4f} "
            f"oracle_generated_recall@10={row['oracle_best_component_recall@10']:.4f} "
            f"oracle_rdcr@10={row['oracle_rdcr@10']:.4f} "
            f"rdcr@10={row['rdcr@10']:.4f} "
            f"top1_recall={row['top1_component_recall']:.4f}"
        )
        results.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(results, root), encoding="utf-8")

    total_n = sum(row["n"] for row in results)
    total_errors = sum(len(row["errors"]) for row in results)

    print(f"[INFO] Saved: {output}")
    print(f"[INFO] Evaluated samples: {total_n}; errors: {total_errors}")

    if total_n == 0:
        print(
            "WARNING: zero samples were evaluated. Check the Errors section "
            "in the generated Markdown.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())