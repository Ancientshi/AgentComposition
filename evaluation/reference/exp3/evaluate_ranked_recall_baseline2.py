#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate Bundle-Ret with the same target-only protocol as Baseline 1.

Gold source
-----------
ONLY ``dataset_example.target`` from each saved per-sample JSON is used.
No PartII lookup, agent mapping, backup target, or retrieval response is used to
construct the gold configuration.

Primary ranked metric
---------------------
RDCR@10 (Rank-Discounted Component Recall@10), with missing ranks assigned zero,
matching the main ranked-recall protocol.  Bundle-Ret emits one agent per query,
so ranks 2..K are intentionally missing/zero rather than duplicated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CUTOFFS = (5, 10, 20)
TOOL_SEP_TOKEN = "<TOOL_SEP>"
END_TOKEN = "<SPECIAL_END>"
TOOL_EMPTY_TOKEN = "<TOOL_EMPTY>"

LLM_RE = re.compile(r"<LLM_[^<>\n\r]+>")
DOUBLE_TOOL_RE = re.compile(r"<<[^<>\n\r]+>>")
SINGLE_TOOL_RE = re.compile(r"<TOOL_[^<>\n\r]+>")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON must be an object")
    return obj


def dedup(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_agent_text(text: Any) -> Tuple[str, List[str]]:
    raw = str(text or "").strip()
    if not raw:
        return "", []
    # Ignore explanation/free-form suffix if present.
    if "Explanation:" in raw:
        raw = raw.split("Explanation:", 1)[0]
    if END_TOKEN in raw:
        raw = raw.split(END_TOKEN, 1)[0]

    llms = LLM_RE.findall(raw)
    llm = llms[0] if llms else ""

    tools = DOUBLE_TOOL_RE.findall(raw) + SINGLE_TOOL_RE.findall(raw)
    tools = [t for t in tools if t not in {TOOL_SEP_TOKEN, TOOL_EMPTY_TOKEN, END_TOKEN}]
    return llm, dedup(tools)


def parse_result_candidate(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    llm = str(item.get("llm_token") or "").strip()
    tools_raw = item.get("tool_tokens")
    tools: List[str] = []
    if isinstance(tools_raw, list):
        tools = [str(x).strip() for x in tools_raw if str(x).strip()]
        tools = [t for t in tools if t != TOOL_EMPTY_TOKEN]

    if not llm or not tools:
        for key in ("strict_text", "gen_text", "text", "output"):
            if item.get(key):
                parsed_llm, parsed_tools = parse_agent_text(item.get(key))
                if not llm:
                    llm = parsed_llm
                if not tools:
                    tools = parsed_tools
                break
    return llm, dedup(tools)


def get_candidates(record: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    containers = []
    if isinstance(record.get("results"), list):
        containers.append(record["results"])
    generation = record.get("generation")
    if isinstance(generation, dict) and isinstance(generation.get("results"), list):
        containers.append(generation["results"])

    for items in containers:
        parsed: List[Tuple[str, List[str]]] = []
        for item in items:
            if isinstance(item, dict):
                llm, tools = parse_result_candidate(item)
                if llm or tools:
                    parsed.append((llm, tools))
        if parsed:
            return parsed
    return []


def get_gold(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    dataset_example = record.get("dataset_example")
    if not isinstance(dataset_example, dict):
        raise ValueError("missing dataset_example object")
    if "target" not in dataset_example:
        raise ValueError("dataset_example.target is missing")
    target = dataset_example.get("target")
    llm, tools = parse_agent_text(target)
    if not llm:
        raise ValueError(f"cannot parse gold LLM from dataset_example.target={target!r}")
    # Zero-tool targets are allowed in principle.
    return llm, tools


def discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def ranked_discounted(values: Sequence[float], k: int) -> float:
    numerator = sum(discount(r) * values[r - 1] for r in range(1, min(k, len(values)) + 1))
    denominator = sum(discount(r) for r in range(1, k + 1))
    return numerator / denominator if denominator else 0.0


def first_true_rank(values: Sequence[bool], k: int) -> Optional[int]:
    for rank, value in enumerate(values[:k], 1):
        if value:
            return rank
    return None


def mean(xs: Sequence[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def evaluate_one(record: Dict[str, Any]) -> Dict[str, Any]:
    gold_llm, gold_tools = get_gold(record)
    candidates = get_candidates(record)
    if not candidates:
        raise ValueError("no parseable candidates in results/generation.results")

    gold_tool_set = set(gold_tools)
    gold_components = {gold_llm} | gold_tool_set

    component_recalls: List[float] = []
    tool_recalls: List[float] = []
    complete_hits: List[bool] = []
    tool_complete_hits: List[bool] = []
    llm_matches: List[bool] = []
    tool_precisions: List[float] = []
    tool_f1s: List[float] = []

    candidate_details: List[Dict[str, Any]] = []
    for rank, (pred_llm, pred_tools) in enumerate(candidates, 1):
        pred_tool_set = set(pred_tools)
        pred_components = ({pred_llm} if pred_llm else set()) | pred_tool_set

        llm_match = pred_llm == gold_llm
        tool_intersection = len(pred_tool_set & gold_tool_set)
        tool_recall = tool_intersection / len(gold_tool_set) if gold_tool_set else 1.0
        tool_precision = tool_intersection / len(pred_tool_set) if pred_tool_set else (1.0 if not gold_tool_set else 0.0)
        tool_f1 = (
            2.0 * tool_precision * tool_recall / (tool_precision + tool_recall)
            if tool_precision + tool_recall > 0 else 0.0
        )
        component_recall = (
            len(pred_components & gold_components) / len(gold_components)
            if gold_components else 0.0
        )
        tool_complete = gold_tool_set.issubset(pred_tool_set)
        complete = llm_match and tool_complete

        component_recalls.append(component_recall)
        tool_recalls.append(tool_recall)
        complete_hits.append(complete)
        tool_complete_hits.append(tool_complete)
        llm_matches.append(llm_match)
        tool_precisions.append(tool_precision)
        tool_f1s.append(tool_f1)
        candidate_details.append({
            "rank": rank,
            "pred_llm": pred_llm,
            "pred_tools": pred_tools,
            "llm_match": llm_match,
            "tool_precision": tool_precision,
            "tool_recall": tool_recall,
            "tool_f1": tool_f1,
            "component_recall": component_recall,
            "complete_recall_hit": complete,
        })

    out: Dict[str, Any] = {
        "gold_llm": gold_llm,
        "gold_tools": gold_tools,
        "num_candidates": len(candidates),
        "top1_llm_accuracy": float(llm_matches[0]),
        "top1_tool_precision": tool_precisions[0],
        "top1_tool_recall": tool_recalls[0],
        "top1_tool_f1": tool_f1s[0],
        "top1_component_recall": component_recalls[0],
        "top1_complete_recall": float(complete_hits[0]),
        "candidate_details": candidate_details,
    }
    for k in CUTOFFS:
        hit_rank = first_true_rank(complete_hits, k)
        tool_hit_rank = first_true_rank(tool_complete_hits, k)
        out[f"cr_hit@{k}"] = float(hit_rank is not None)
        out[f"cr_mrr@{k}"] = 0.0 if hit_rank is None else 1.0 / hit_rank
        out[f"tool_hit@{k}"] = float(tool_hit_rank is not None)
        out[f"tool_mrr@{k}"] = 0.0 if tool_hit_rank is None else 1.0 / tool_hit_rank
        out[f"rdcr@{k}"] = ranked_discounted(component_recalls, k)
        out[f"rdtr@{k}"] = ranked_discounted(tool_recalls, k)
    return out


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def discover_files(root: Path) -> List[Path]:
    direct = root / "per_sample"
    if direct.is_dir():
        return sorted(direct.glob("*.json"))
    return sorted(root.glob("**/per_sample/*.json"))


def build_markdown(root: Path, rows: List[Dict[str, Any]], errors: List[str]) -> str:
    metrics = {
        "llm_acc": mean([r["top1_llm_accuracy"] for r in rows]),
        "tool_p": mean([r["top1_tool_precision"] for r in rows]),
        "tool_r": mean([r["top1_tool_recall"] for r in rows]),
        "tool_f1": mean([r["top1_tool_f1"] for r in rows]),
        "component_r": mean([r["top1_component_recall"] for r in rows]),
        "complete": mean([r["top1_complete_recall"] for r in rows]),
    }
    for k in CUTOFFS:
        for name in ("cr_hit", "cr_mrr", "tool_hit", "tool_mrr", "rdcr", "rdtr"):
            metrics[f"{name}@{k}"] = mean([r[f"{name}@{k}"] for r in rows])

    md = [
        "# Baseline 2 — Bundle-Ret Ranked Recall Evaluation",
        "",
        "## Protocol",
        "",
        f"- Experiment root: `{root}`",
        "- **Gold source: `dataset_example.target` only.**",
        "- No PartII lookup, agent mapping, backup target, or retrieval-derived gold is used.",
        "- Prediction: Top-1 CF-retrieved LLM + intact Top-1 CF-retrieved historical tool bundle.",
        "- Extra predicted tools are allowed for complete-recall success.",
        "- Missing ranks are assigned zero in RDCR/RDTR, exactly as in the ranked-recall protocol.",
        "",
        "## Main results",
        "",
        "| N | LLM Acc. | Tool P | Tool R | Tool F1 | Top-1 Component Recall | Complete Recall | **RDCR@10** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {pct(metrics['llm_acc'])} | {pct(metrics['tool_p'])} | {pct(metrics['tool_r'])} | {pct(metrics['tool_f1'])} | {pct(metrics['component_r'])} | {pct(metrics['complete'])} | **{metrics['rdcr@10']:.4f}** |",
        "",
        "## Ranked recall diagnostics",
        "",
        "| K | CR-Hit@K | CR-MRR@K | RDCR@K | Tool-Hit@K | Tool-MRR@K | RDTR@K |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in CUTOFFS:
        md.append(
            f"| {k} | {pct(metrics[f'cr_hit@{k}'])} | {metrics[f'cr_mrr@{k}']:.4f} | "
            f"{metrics[f'rdcr@{k}']:.4f} | {pct(metrics[f'tool_hit@{k}'])} | "
            f"{metrics[f'tool_mrr@{k}']:.4f} | {metrics[f'rdtr@{k}']:.4f} |"
        )

    md += [
        "",
        "## Interpretation",
        "",
        "Bundle-Ret is a single-agent retrieval baseline. Therefore CR-Hit@5/10/20 "
        "and CR-MRR@5/10/20 are identical whenever the sole rank-1 candidate is a hit. "
        "RDCR@K decreases with larger K because the standard protocol assigns zero to "
        "unreturned ranks rather than replicating the rank-1 agent. This preserves direct "
        "comparability with methods that return an actual ranked candidate list.",
    ]

    if errors:
        md += ["", "## Evaluation errors", ""]
        md.extend(f"- {e}" for e in errors[:50])
        if len(errors) > 50:
            md.append(f"- ... plus {len(errors) - 50} additional errors.")
    return "\n".join(md) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Experiment root not found: {root}")

    files = discover_files(root)
    if not files:
        raise SystemExit(f"No per-sample JSON files found under: {root}")

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    per_sample_eval: List[Dict[str, Any]] = []
    for path in files:
        try:
            record = load_json(path)
            ev = evaluate_one(record)
            ev["file"] = str(path)
            rows.append(ev)
            per_sample_eval.append(ev)
        except Exception as exc:
            errors.append(f"`{path.name}`: {type(exc).__name__}: {exc}")

    if not rows:
        raise SystemExit("Zero samples evaluated successfully")

    out_md = args.output.expanduser().resolve() if args.output else root / "evaluation" / "bundle_retrieve_ranked_recall_metrics.md"
    out_dir = out_md.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    md = build_markdown(root, rows, errors)
    out_md.write_text(md, encoding="utf-8")

    aggregate: Dict[str, Any] = {
        "evaluation_name": "Baseline 2 — Bundle-Ret Ranked Recall Evaluation",
        "experiment_root": str(root),
        "gold_source": "dataset_example.target only",
        "n": len(rows),
        "errors": errors,
        "mean": {
            "llm_accuracy": mean([r["top1_llm_accuracy"] for r in rows]),
            "tool_precision": mean([r["top1_tool_precision"] for r in rows]),
            "tool_recall": mean([r["top1_tool_recall"] for r in rows]),
            "tool_f1": mean([r["top1_tool_f1"] for r in rows]),
            "top1_component_recall": mean([r["top1_component_recall"] for r in rows]),
            "complete_recall": mean([r["top1_complete_recall"] for r in rows]),
        },
    }
    for k in CUTOFFS:
        for name in ("cr_hit", "cr_mrr", "tool_hit", "tool_mrr", "rdcr", "rdtr"):
            aggregate["mean"][f"{name}@{k}"] = mean([r[f"{name}@{k}"] for r in rows])

    (out_dir / "bundle_retrieve_ranked_recall_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "per_sample_evaluation.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample_eval:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Evaluated: {len(rows)} samples; errors={len(errors)}")
    print(f"[INFO] Saved: {out_md}")
    print(f"[INFO] RDCR@10={aggregate['mean']['rdcr@10']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
