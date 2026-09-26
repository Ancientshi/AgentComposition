#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate Component-Ret (Baseline 1) with gold read DIRECTLY from target.

Expected experiment layout:
    <experiment-root>/
        k1/per_sample/*.json
        k2/per_sample/*.json
        ...
        k5/per_sample/*.json

Gold protocol
-------------
The gold agent is parsed ONLY from:
    record["dataset_example"]["target"]

For example:
    <LLM_nvidia__AceInstruct-7B> <TOOL_SEP>
    <<GoogleBooks&&moveVolume>> <SPECIAL_END>

No PartII file, PartII agent mapping, backup target, or tool-name replacement
is used.

Ranked baseline pool
--------------------
For one Component-Ret run with tool count K, the baseline saves:
  - the CF-LLM ranked list (normally Top-10), and
  - the selected Top-K semantic tools.

To obtain a ranked agent list for RDCR@10, candidate r is:
    A_r = (m_r, T_{1:K}),
where m_r is the r-th CF-retrieved LLM and T_{1:K} is the SAME selected tool
set for every rank. This preserves the Component-Ret ranking signal without
introducing a generator, critic, beam search, or gold oracle.

Main metric
-----------
For gold G=(m*, T*) and ranked candidate A_r=(m_r, T_r),

    R_r = ( 1[m_r=m*] + |T_r ∩ T*| ) / (1 + |T*|)

and

    RDCR@K =
      sum_{r=1..K} R_r / log2(r+1)
      --------------------------------
      sum_{r=1..K} 1 / log2(r+1)

Missing ranks are assigned recall 0.

Outputs:
    <experiment-root>/evaluation/ranked_recall_baseline1.md
    <experiment-root>/evaluation/ranked_recall_baseline1.json
    <experiment-root>/evaluation/ranked_recall_baseline1_per_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LLM_RE = re.compile(r"<LLM_[^<>]+>")
# Two legal tool token forms in the dataset:
#   1) native PartII/API token: <<API&&Tool>>
#   2) prefixed token:          <TOOL_Name>
# Explicitly exclude the structural separator <TOOL_SEP>.
TOOL_NATIVE_RE = re.compile(r"<<[^<>]+>>")
TOOL_PREFIX_RE = re.compile(r"<TOOL_(?!SEP>)[^<>]+>")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def clean_llm_token(value: Any) -> str:
    """Keep canonical token identity; only add wrapper for raw LLM names."""
    if value is None:
        return ""
    text = str(value).strip()
    m = LLM_RE.fullmatch(text)
    if m:
        return m.group(0)
    if text.startswith("LLM_"):
        return f"<{text}>"
    if text:
        # Match the baseline writer's normalize_raw_name(): whitespace -> "_".
        raw = "".join("_" if ch.isspace() else ch for ch in text)
        return f"<LLM_{raw}>"
    return ""


def clean_tool_token(value: Any) -> str:
    """Normalize one tool token without treating structural tokens as tools.

    Legal tool forms are preserved exactly:
      - <<API&&Tool>>
      - <TOOL_Name>

    Raw tool names are wrapped according to the dataset grammar:
      - names containing ``&&`` -> <<API&&Tool>>
      - other names            -> <TOOL_Name>

    Historical malformed nesting such as <TOOL_<<API&&Tool>>> is repaired to
    <<API&&Tool>>.  <TOOL_SEP> and <SPECIAL_END> are never tools.
    """
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text in {"<TOOL_SEP>", "<SPECIAL_END>", "<TOOL_EMPTY>"}:
        return ""

    if TOOL_NATIVE_RE.fullmatch(text):
        return text
    if TOOL_PREFIX_RE.fullmatch(text):
        return text

    # Repair the malformed legacy form: <TOOL_<<API&&Tool>>>
    if text.startswith("<TOOL_<<") and text.endswith(">>>"):
        inner = text[len("<TOOL_"):-1]
        if TOOL_NATIVE_RE.fullmatch(inner):
            return inner

    if "&&" in text:
        return f"<<{text.strip('<>')}>>"
    return f"<TOOL_{text.strip('<>')}>"


def extract_tool_tokens(text: str) -> List[str]:
    """Extract both legal tool-token forms in their original left-to-right order."""
    matches: List[Tuple[int, str]] = []
    for regex in (TOOL_NATIVE_RE, TOOL_PREFIX_RE):
        matches.extend((m.start(), m.group(0)) for m in regex.finditer(text))
    matches.sort(key=lambda x: x[0])
    return stable_unique(token for _, token in matches)


def stable_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_target_direct(record: Mapping[str, Any]) -> Tuple[str, List[str], str]:
    """Parse gold ONLY from dataset_example.target."""
    dataset_example = record.get("dataset_example")
    if not isinstance(dataset_example, Mapping):
        raise ValueError("missing record['dataset_example']")

    target = dataset_example.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("missing/non-string record['dataset_example']['target']")

    llms = LLM_RE.findall(target)
    if len(llms) != 1:
        raise ValueError(
            f"expected exactly one <LLM_...> in dataset_example.target, found {len(llms)}"
        )

    tools = extract_tool_tokens(target)
    return llms[0], tools, target


def _llm_from_item(item: Any) -> str:
    if isinstance(item, str):
        return clean_llm_token(item)
    if not isinstance(item, Mapping):
        return ""

    # Preferred field written by baseline1_run_component_retrieve_baseline.py
    for key in ("token", "llm_token", "component"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            if "<LLM_" in value:
                m = LLM_RE.search(value)
                if m:
                    return m.group(0)

    llm = item.get("llm")
    if isinstance(llm, str) and llm.strip():
        return clean_llm_token(llm)
    if isinstance(llm, Mapping):
        for key in ("token", "name", "canonical_id", "id", "model"):
            if llm.get(key):
                return clean_llm_token(llm[key])

    for key in ("name", "canonical_id", "model"):
        if item.get(key):
            return clean_llm_token(item[key])

    evidence = item.get("evidence")
    if isinstance(evidence, Mapping):
        component = evidence.get("component")
        if isinstance(component, str) and component.startswith("<LLM_"):
            return clean_llm_token(component)

    return ""


def parse_ranked_llms(record: Mapping[str, Any], max_rank: int) -> List[str]:
    retrieval = record.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise ValueError("missing record['retrieval']")

    sources: List[Any] = []

    parsed = retrieval.get("parsed_cf_llm_candidates")
    if isinstance(parsed, list):
        sources = parsed

    if not sources:
        response = retrieval.get("cf_llm_response")
        if isinstance(response, Mapping):
            for key in ("topk", "results", "evidence"):
                value = response.get(key)
                if isinstance(value, list) and value:
                    sources = value
                    break

    llms = stable_unique(_llm_from_item(item) for item in sources)
    if not llms:
        raise ValueError("cannot parse any ranked CF-LLM candidates from retrieval")
    return llms[:max_rank]


def _tool_from_item(item: Any) -> str:
    if isinstance(item, str):
        return clean_tool_token(item)
    if not isinstance(item, Mapping):
        return ""
    for key in ("token", "component", "tool_token", "item_id", "key"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return clean_tool_token(value)
    tool = item.get("tool")
    if isinstance(tool, str):
        return clean_tool_token(tool)
    if isinstance(tool, Mapping):
        for key in ("token", "component", "key", "item_id", "name"):
            if tool.get(key):
                return clean_tool_token(tool[key])
    return ""


def parse_selected_tools(record: Mapping[str, Any]) -> List[str]:
    """Read the actual Top-K tool set selected for this Component-Ret run."""
    composition = record.get("composition")
    if isinstance(composition, Mapping):
        selected = composition.get("tool_selection")
        if isinstance(selected, list):
            tools = stable_unique(_tool_from_item(x) for x in selected)
            if tools:
                return tools

    results = record.get("results")
    if isinstance(results, list) and results and isinstance(results[0], Mapping):
        tokens = results[0].get("tool_tokens")
        if isinstance(tokens, list):
            tools = stable_unique(
                clean_tool_token(x)
                for x in tokens
                if str(x).strip() != "<TOOL_EMPTY>"
            )
            if tools:
                return tools

    generation = record.get("generation")
    if isinstance(generation, Mapping):
        gres = generation.get("results")
        if isinstance(gres, list) and gres and isinstance(gres[0], Mapping):
            tokens = gres[0].get("tool_tokens")
            if isinstance(tokens, list):
                tools = stable_unique(
                    clean_tool_token(x)
                    for x in tokens
                    if str(x).strip() != "<TOOL_EMPTY>"
                )
                if tools:
                    return tools

    # Empty-tool configurations are valid, so return [] rather than guessing
    # from the full retrieval list.
    return []


def discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def rd_at_k(recalls: Sequence[float], k: int) -> float:
    numerator = sum(
        discount(rank) * float(recalls[rank - 1])
        for rank in range(1, min(k, len(recalls)) + 1)
    )
    denominator = sum(discount(rank) for rank in range(1, k + 1))
    return numerator / denominator if denominator else 0.0


def first_true_rank(values: Sequence[bool]) -> Optional[int]:
    for i, value in enumerate(values, 1):
        if value:
            return i
    return None


def evaluate_sample(path: Path, cutoffs: Sequence[int], max_rank: int) -> Dict[str, Any]:
    record = read_json(path)

    gold_llm, gold_tools, target = parse_target_direct(record)
    ranked_llms = parse_ranked_llms(record, max_rank=max_rank)
    selected_tools = parse_selected_tools(record)

    gold_tool_set = set(gold_tools)
    pred_tool_set = set(selected_tools)

    gold_tool_count = len(gold_tool_set)
    tool_overlap = len(gold_tool_set & pred_tool_set)

    tool_recall = (
        tool_overlap / gold_tool_count
        if gold_tool_count > 0
        else 1.0
    )
    tool_complete = gold_tool_set.issubset(pred_tool_set)

    component_recalls: List[float] = []
    tool_recalls: List[float] = []
    complete_hits: List[bool] = []
    tool_complete_hits: List[bool] = []

    component_denominator = 1 + gold_tool_count

    candidates: List[Dict[str, Any]] = []
    for rank, llm in enumerate(ranked_llms, 1):
        llm_hit = llm == gold_llm
        component_recall = (int(llm_hit) + tool_overlap) / component_denominator

        component_recalls.append(component_recall)
        tool_recalls.append(tool_recall)
        complete_hits.append(bool(llm_hit and tool_complete))
        tool_complete_hits.append(bool(tool_complete))

        candidates.append({
            "rank": rank,
            "llm": llm,
            "tools": selected_tools,
            "llm_hit": llm_hit,
            "tool_recall": tool_recall,
            "component_recall": component_recall,
            "agent_complete_hit": bool(llm_hit and tool_complete),
        })

    first_complete = first_true_rank(complete_hits)
    first_tool_complete = first_true_rank(tool_complete_hits)

    dataset_example = record.get("dataset_example") or {}
    config = record.get("config") or {}

    out: Dict[str, Any] = {
        "file": path.name,
        "sample_id": dataset_example.get("sample_id"),
        "qid": dataset_example.get("qid"),
        "component_k": config.get("component_k"),
        "gold_source": "dataset_example.target",
        "target": target,
        "gold_llm": gold_llm,
        "gold_tools": gold_tools,
        "ranked_llm_count": len(ranked_llms),
        "selected_tools": selected_tools,
        "first_complete_rank": first_complete,
        "first_tool_complete_rank": first_tool_complete,
        "top1_component_recall": component_recalls[0] if component_recalls else 0.0,
        "top1_tool_recall": tool_recalls[0] if tool_recalls else 0.0,
        "candidates": candidates,
    }

    for k in cutoffs:
        hit = first_complete is not None and first_complete <= k
        out[f"cr_hit@{k}"] = float(hit)
        out[f"cr_mrr@{k}"] = (1.0 / first_complete) if hit else 0.0
        out[f"rdcr@{k}"] = rd_at_k(component_recalls, k)

        tool_hit = first_tool_complete is not None and first_tool_complete <= k
        out[f"tool_hit@{k}"] = float(tool_hit)
        out[f"tool_mrr@{k}"] = (
            (1.0 / first_tool_complete) if tool_hit else 0.0
        )
        out[f"rdtr@{k}"] = rd_at_k(tool_recalls, k)
        out[f"available@{k}"] = float(len(ranked_llms) >= k)

    return out


def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return statistics.fmean(values) if values else 0.0


def aggregate_run(
    run_dir: Path,
    cutoffs: Sequence[int],
    max_rank: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    files = sorted((run_dir / "per_sample").glob("sample_*.json"))
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for path in files:
        try:
            rows.append(evaluate_sample(path, cutoffs=cutoffs, max_rank=max_rank))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    first_ranks = [
        int(row["first_complete_rank"])
        for row in rows
        if row["first_complete_rank"] is not None
    ]

    result: Dict[str, Any] = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "files_found": len(files),
        "n": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "gold_source": "dataset_example.target",
        "mean_ranked_llm_count": (
            statistics.fmean(float(row["ranked_llm_count"]) for row in rows)
            if rows else 0.0
        ),
        "mean_selected_tool_count": (
            statistics.fmean(len(row["selected_tools"]) for row in rows)
            if rows else 0.0
        ),
        "top1_component_recall": mean(rows, "top1_component_recall"),
        "top1_tool_recall": mean(rows, "top1_tool_recall"),
        "any_complete_hit": len(first_ranks) / len(rows) if rows else 0.0,
        "first_complete_rank_mean": (
            statistics.fmean(first_ranks) if first_ranks else None
        ),
        "first_complete_rank_median": (
            statistics.median(first_ranks) if first_ranks else None
        ),
    }

    for k in cutoffs:
        for metric in (
            "cr_hit", "cr_mrr", "rdcr",
            "tool_hit", "tool_mrr", "rdtr",
            "available",
        ):
            result[f"{metric}@{k}"] = mean(rows, f"{metric}@{k}")

        values = [float(row[f"rdcr@{k}"]) for row in rows]
        result[f"rdcr_std@{k}"] = (
            statistics.pstdev(values) if values else 0.0
        )

    return result, rows


def discover_runs(root: Path) -> List[Path]:
    runs: List[Path] = []

    if (root / "per_sample").is_dir():
        runs.append(root)

    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and (child / "per_sample").is_dir():
            runs.append(child)

    def sort_key(path: Path) -> Tuple[int, int, str]:
        m = re.fullmatch(r"k(\d+)", path.name, flags=re.IGNORECASE)
        if m:
            return (0, int(m.group(1)), path.name)
        return (1, 0, path.name)

    return sorted(stable_unique_paths(runs), key=sort_key)


def stable_unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def build_markdown(
    summaries: Sequence[Mapping[str, Any]],
    root: Path,
    cutoffs: Sequence[int],
) -> str:
    primary_k = 10 if 10 in cutoffs else max(cutoffs)

    lines = [
        "# Baseline 1 — Component-Ret Ranked Recall Evaluation",
        "",
        "## Protocol",
        "",
        f"- Experiment root: `{root}`",
        "- **Gold source: `dataset_example.target` only.**",
        "- No PartII lookup, PartII agent mapping, backup target, or target rewriting is used.",
        "- Tool identifiers are compared as written in the target/candidate tokens; `_` is not converted to a space.",
        "- Ranked candidate $A_r=(m_r,T_{1:K})$: the CF-LLM rank varies with $r$, while the selected Top-$K$ tool set is fixed for the run.",
        "",
        "For candidate rank $r$, component recall is",
        "",
        "$$R_r=\\frac{\\mathbf{1}[m_r=m^*]+|T_r\\cap T^*|}{1+|T^*|},$$",
        "",
        "and",
        "",
        "$$\\mathrm{RDCR@K}=\\frac{\\sum_{r=1}^{K}R_r/\\log_2(r+1)}{\\sum_{r=1}^{K}1/\\log_2(r+1)}.$$",
        "",
        "Extra predicted tools do not reduce recall. Missing ranks are assigned recall 0.",
        "",
        f"## Main results (primary: RDCR@{primary_k})",
        "",
    ]

    header = ["Run", "N", "Mean #tools", "Top-1 Comp. Recall"]
    for k in cutoffs:
        header += [f"CR-Hit@{k}", f"CR-MRR@{k}", f"RDCR@{k}"]
    header += ["Errors"]

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")

    for row in summaries:
        vals = [
            str(row["run"]),
            str(row["n"]),
            f'{float(row["mean_selected_tool_count"]):.2f}',
            pct(float(row["top1_component_recall"])),
        ]
        for k in cutoffs:
            vals += [
                pct(float(row[f"cr_hit@{k}"])),
                f'{float(row[f"cr_mrr@{k}"]):.4f}',
                f'{float(row[f"rdcr@{k}"]):.4f}',
            ]
        vals.append(str(row["error_count"]))
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Relaxed tool-only results",
        "",
    ]

    tool_header = ["Run", "N", "Top-1 Tool Recall"]
    for k in cutoffs:
        tool_header += [f"Tool-Hit@{k}", f"Tool-MRR@{k}", f"RDTR@{k}"]

    lines.append("| " + " | ".join(tool_header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(tool_header) - 1)) + "|")

    for row in summaries:
        vals = [
            str(row["run"]),
            str(row["n"]),
            pct(float(row["top1_tool_recall"])),
        ]
        for k in cutoffs:
            vals += [
                pct(float(row[f"tool_hit@{k}"])),
                f'{float(row[f"tool_mrr@{k}"]):.4f}',
                f'{float(row[f"rdtr@{k}"]):.4f}',
            ]
        lines.append("| " + " | ".join(vals) + " |")

    if summaries:
        best = max(summaries, key=lambda x: float(x.get(f"rdcr@{primary_k}", 0.0)))
        lines += [
            "",
            "## Global K selection",
            "",
            f"- Best run by **RDCR@{primary_k}**: `{best['run']}` "
            f"({float(best[f'rdcr@{primary_k}']):.4f}).",
            "- This selects one global Component-Ret tool-count hyperparameter from the validation run; it does not use per-query gold tool count.",
        ]

    all_errors = [
        (str(row["run"]), err)
        for row in summaries
        for err in row.get("errors", [])
    ]
    if all_errors:
        lines += ["", "## Evaluation errors", ""]
        for run, err in all_errors[:100]:
            lines.append(f"- `{run}`: {err}")
        if len(all_errors) > 100:
            lines.append(f"- ... plus {len(all_errors) - 100} more errors.")

    return "\n".join(lines) + "\n"


def parse_cutoffs(text: str) -> List[int]:
    values = stable_unique(
        str(int(x.strip()))
        for x in text.split(",")
        if x.strip()
    )
    out = sorted(int(x) for x in values)
    if not out or any(k < 1 for k in out):
        raise ValueError("cutoffs must be positive integers")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Root containing k1/k2/.../per_sample.",
    )
    ap.add_argument(
        "--cutoffs",
        type=str,
        default="1,5,10",
        help="Comma-separated ranked cutoffs; default: 1,5,10.",
    )
    ap.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="Maximum number of CF-LLM ranks used to build agent candidates.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <experiment-root>/evaluation",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.experiment_root.expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: experiment root not found: {root}", file=sys.stderr)
        return 2

    cutoffs = parse_cutoffs(args.cutoffs)
    if args.max_rank < 1:
        print("ERROR: --max-rank must be >= 1", file=sys.stderr)
        return 2
    if max(cutoffs) > args.max_rank:
        print(
            f"ERROR: max cutoff {max(cutoffs)} exceeds --max-rank {args.max_rank}",
            file=sys.stderr,
        )
        return 2

    runs = discover_runs(root)
    if not runs:
        print(
            f"ERROR: no run directories with per_sample/ found under {root}",
            file=sys.stderr,
        )
        return 2

    print("[INFO] Gold source: dataset_example.target ONLY")
    print("[INFO] No PartII mapping/lookup is used")
    print("[INFO] Runs:", ", ".join(p.name for p in runs))

    summaries: List[Dict[str, Any]] = []
    all_per_sample: List[Dict[str, Any]] = []

    for run_dir in runs:
        summary, rows = aggregate_run(
            run_dir,
            cutoffs=cutoffs,
            max_rank=args.max_rank,
        )
        summaries.append(summary)
        for row in rows:
            all_per_sample.append({"run": run_dir.name, **row})

        primary = 10 if 10 in cutoffs else max(cutoffs)
        print(
            f"[RESULT] {run_dir.name}: N={summary['n']} "
            f"RDCR@{primary}={summary[f'rdcr@{primary}']:.4f} "
            f"errors={summary['error_count']}"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ranked_recall_baseline1.json"
    md_path = output_dir / "ranked_recall_baseline1.md"
    per_sample_path = output_dir / "ranked_recall_baseline1_per_sample.jsonl"

    payload = {
        "evaluation": "Baseline 1 Component-Ret ranked recall",
        "gold_source": "dataset_example.target",
        "partii_processing": False,
        "candidate_rule": "A_r = (CF-LLM rank r, fixed selected Top-K tools)",
        "cutoffs": cutoffs,
        "max_rank": args.max_rank,
        "experiment_root": str(root),
        "runs": summaries,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path.write_text(
        build_markdown(summaries, root=root, cutoffs=cutoffs),
        encoding="utf-8",
    )

    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in all_per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[SAVED] {md_path}")
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {per_sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())