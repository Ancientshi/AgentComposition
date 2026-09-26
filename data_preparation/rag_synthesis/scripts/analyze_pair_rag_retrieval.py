#!/usr/bin/env python3
"""Evaluate retrieval coverage in pair_rag.jsonl.

The script evaluates:
  1. Authoritative overall coverage recorded in ``rag_meta``.
  2. Four routes parsed independently from ``context``:
       - cf_retrieved_llm
       - cf_retrieved_tool_bundle
       - semantic_retrieved_llm
       - semantic_retrieved_tool

For ``cf_retrieved_tool_bundle``, top-k means the first k retrieved bundles.
The tools occurring in those bundles are then unioned and deduplicated before
precision/recall is computed. Citation suffixes such as ``[17]`` are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


LOGGER = logging.getLogger("pair_rag_retrieval")

CUTOFFS: tuple[int | None, ...] = (5, 10, 15, 20, 50, None)
ROUTE_HEADERS = (
    "cf_retrieved_llm",
    "cf_retrieved_tool_bundle",
    "semantic_retrieved_llm",
    "semantic_retrieved_tool",
    "evidence",
)
LLM_RE = re.compile(r"<LLM_[^<>\r\n]+>")
TOOL_RE = re.compile(r"<<[^<>\r\n]+>>")
CITATION_RE = re.compile(r"\s*\[\d+\]")


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_component(value: Any) -> str:
    """Normalize a component without changing its actual identifier."""
    if value is None:
        return ""
    text = CITATION_RE.sub("", str(value)).strip().rstrip(",").strip()
    return text


def normalize_component_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return dedupe_preserve_order(
        item for raw in values if (item := normalize_component(raw))
    )


def split_context_sections(context: Any) -> dict[str, str]:
    """Split context by exact route headers, regardless of indentation."""
    if not isinstance(context, str):
        return {header: "" for header in ROUTE_HEADERS}

    header_alt = "|".join(re.escape(header) for header in ROUTE_HEADERS)
    pattern = re.compile(
        rf"(?m)^\s*(?P<header>{header_alt})\s*:\s*(?:\r?\n)?"
    )
    matches = list(pattern.finditer(context))
    sections = {header: "" for header in ROUTE_HEADERS}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        sections[match.group("header")] = context[start:end].strip()
    return sections


def parse_llms(section: str) -> list[str]:
    return dedupe_preserve_order(normalize_component(x) for x in LLM_RE.findall(section))


def parse_tools(section: str) -> list[str]:
    return dedupe_preserve_order(normalize_component(x) for x in TOOL_RE.findall(section))


def parse_tool_bundles(section: str) -> list[list[str]]:
    """Return ranked bundles; duplicate bundles still consume retrieval ranks."""
    bundles: list[list[str]] = []
    for match in re.finditer(r"\{(?P<body>[^{}]*)\}", section, flags=re.DOTALL):
        tools = parse_tools(match.group("body"))
        if tools:
            bundles.append(tools)
    return bundles


def route_candidates(parsed: dict[str, Any], route: str, cutoff: int | None) -> list[str]:
    if route == "cf_retrieved_tool_bundle":
        bundles: list[list[str]] = parsed[route]
        selected_bundles = bundles if cutoff is None else bundles[:cutoff]
        return dedupe_preserve_order(tool for bundle in selected_bundles for tool in bundle)

    candidates: list[str] = parsed[route]
    selected = candidates if cutoff is None else candidates[:cutoff]
    # Deduplication occurs after slicing, so top-k retains the stored ranking.
    return dedupe_preserve_order(selected)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class Aggregate:
    examples: int = 0
    gold_total: int = 0
    retrieved_total: int = 0
    hit_total: int = 0
    macro_precision_sum: float = 0.0
    macro_recall_sum: float = 0.0
    macro_f1_sum: float = 0.0
    any_hit: int = 0
    all_gold_covered: int = 0
    empty_retrieval: int = 0
    retrieved_count_sum: int = 0

    def add(self, gold: set[str], retrieved: set[str]) -> None:
        hits = len(gold & retrieved)
        precision = safe_div(hits, len(retrieved))
        recall = safe_div(hits, len(gold))
        f1 = safe_div(2 * precision * recall, precision + recall)

        self.examples += 1
        self.gold_total += len(gold)
        self.retrieved_total += len(retrieved)
        self.hit_total += hits
        self.macro_precision_sum += precision
        self.macro_recall_sum += recall
        self.macro_f1_sum += f1
        self.any_hit += int(hits > 0)
        self.all_gold_covered += int(gold.issubset(retrieved))
        self.empty_retrieval += int(not retrieved)
        self.retrieved_count_sum += len(retrieved)

    def as_dict(self) -> dict[str, int | float]:
        micro_precision = safe_div(self.hit_total, self.retrieved_total)
        micro_recall = safe_div(self.hit_total, self.gold_total)
        return {
            "examples": self.examples,
            "gold_components": self.gold_total,
            "retrieved_unique_components": self.retrieved_total,
            "true_positive_components": self.hit_total,
            "mean_retrieved_count": safe_div(self.retrieved_count_sum, self.examples),
            "macro_precision": safe_div(self.macro_precision_sum, self.examples),
            "macro_recall": safe_div(self.macro_recall_sum, self.examples),
            "macro_f1": safe_div(self.macro_f1_sum, self.examples),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
            "any_gold_hit_rate": safe_div(self.any_hit, self.examples),
            "all_gold_covered_rate": safe_div(self.all_gold_covered, self.examples),
            "empty_retrieval_rate": safe_div(self.empty_retrieval, self.examples),
        }


@dataclass
class OverallCoverage:
    rows: int = 0
    llm_covered: int = 0
    tool_gold_total: int = 0
    tool_hit_total: int = 0
    tool_recall_sum: float = 0.0
    rows_with_tools: int = 0
    any_tool_covered: int = 0
    all_tools_covered: int = 0
    complete_configuration_recoverable: int = 0

    def add(self, covered_llm: bool, gold_tools: set[str], covered_tools: set[str]) -> None:
        tool_hits = len(gold_tools & covered_tools)
        all_tools = gold_tools.issubset(covered_tools)
        self.rows += 1
        self.llm_covered += int(covered_llm)
        self.tool_gold_total += len(gold_tools)
        self.tool_hit_total += tool_hits
        if gold_tools:
            self.rows_with_tools += 1
            self.tool_recall_sum += safe_div(tool_hits, len(gold_tools))
            self.any_tool_covered += int(tool_hits > 0)
            self.all_tools_covered += int(all_tools)
        self.complete_configuration_recoverable += int(covered_llm and all_tools)

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "rows": self.rows,
            "rows_with_gold_tools": self.rows_with_tools,
            "gold_llm_coverage": safe_div(self.llm_covered, self.rows),
            "gold_tool_recall_macro": safe_div(self.tool_recall_sum, self.rows_with_tools),
            "gold_tool_recall_micro": safe_div(self.tool_hit_total, self.tool_gold_total),
            "any_gold_tool_covered_rate": safe_div(self.any_tool_covered, self.rows_with_tools),
            "all_gold_tools_covered_rate": safe_div(self.all_tools_covered, self.rows_with_tools),
            "complete_configuration_recoverable_rate": safe_div(
                self.complete_configuration_recoverable, self.rows
            ),
            "note": (
                "Tool rates except complete_configuration_recoverable are conditional on "
                "rows containing at least one gold tool. Overall precision cannot be derived "
                "from rag_meta because it stores covered gold items, not all retrieved items."
            ),
        }


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSON at line %d: %s", line_number, exc)
                continue
            if not isinstance(value, dict):
                LOGGER.warning("Skipping non-object JSON at line %d", line_number)
                continue
            yield line_number, value


def cutoff_label(cutoff: int | None) -> str:
    return "all" if cutoff is None else str(cutoff)


def evaluate(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    routes = (
        "cf_retrieved_llm",
        "cf_retrieved_tool_bundle",
        "semantic_retrieved_llm",
        "semantic_retrieved_tool",
    )
    aggregates: dict[tuple[str, int | None], Aggregate] = {
        (route, cutoff): Aggregate() for route in routes for cutoff in CUTOFFS
    }
    overall = OverallCoverage()
    issues: Counter[str] = Counter()
    row_details: list[dict[str, Any]] = []
    parsed_rows = 0

    for line_number, row in iter_jsonl(input_path):
        parsed_rows += 1
        rag_meta = row.get("rag_meta")
        if not isinstance(rag_meta, dict):
            issues["missing_or_invalid_rag_meta"] += 1
            continue

        gold_llm = normalize_component(rag_meta.get("gold_llm"))
        gold_tools = set(normalize_component_list(rag_meta.get("gold_tools")))
        covered_tools = set(normalize_component_list(rag_meta.get("covered_tools")))
        covered_llm_raw = rag_meta.get("covered_llm")
        if not gold_llm:
            issues["missing_gold_llm"] += 1
        if not isinstance(covered_llm_raw, bool):
            issues["covered_llm_not_boolean"] += 1
        covered_llm = bool(covered_llm_raw)
        overall.add(covered_llm, gold_tools, covered_tools)

        sections = split_context_sections(row.get("context"))
        parsed: dict[str, Any] = {
            "cf_retrieved_llm": parse_llms(sections["cf_retrieved_llm"]),
            "cf_retrieved_tool_bundle": parse_tool_bundles(
                sections["cf_retrieved_tool_bundle"]
            ),
            "semantic_retrieved_llm": parse_llms(
                sections["semantic_retrieved_llm"]
            ),
            "semantic_retrieved_tool": parse_tools(
                sections["semantic_retrieved_tool"]
            ),
        }

        for route in routes:
            if not sections[route]:
                issues[f"empty_section:{route}"] += 1
            route_gold = {gold_llm} if route.endswith("_llm") and gold_llm else gold_tools
            if not route_gold:
                # Tool-route metrics are undefined for zero-tool configurations.
                if "tool" in route:
                    issues[f"zero_gold_tools_skipped:{route}"] += 1
                else:
                    issues[f"empty_gold_skipped:{route}"] += 1
                continue
            for cutoff in CUTOFFS:
                retrieved = set(route_candidates(parsed, route, cutoff))
                aggregates[(route, cutoff)].add(route_gold, retrieved)

        # Compact all-candidate row diagnostics make spot checks possible.
        row_details.append(
            {
                "line_number": line_number,
                "qid": row.get("qid", ""),
                "agent_id": row.get("agent_id", ""),
                "part": row.get("part", ""),
                "stored_topk": row.get("topk", ""),
                "gold_llm": gold_llm,
                "gold_tool_count": len(gold_tools),
                "covered_llm": covered_llm,
                "covered_gold_tool_count": len(gold_tools & covered_tools),
                "cf_llm_count": len(parsed["cf_retrieved_llm"]),
                "cf_bundle_count": len(parsed["cf_retrieved_tool_bundle"]),
                "cf_bundle_unique_tool_count": len(
                    route_candidates(parsed, "cf_retrieved_tool_bundle", None)
                ),
                "semantic_llm_count": len(parsed["semantic_retrieved_llm"]),
                "semantic_tool_count": len(parsed["semantic_retrieved_tool"]),
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for route in routes:
        for cutoff in CUTOFFS:
            metric_rows.append(
                {
                    "route": route,
                    "cutoff": cutoff_label(cutoff),
                    **aggregates[(route, cutoff)].as_dict(),
                }
            )

    summary = {
        "input": str(input_path),
        "valid_json_object_rows": parsed_rows,
        "evaluated_rag_meta_rows": overall.rows,
        "cutoffs": [cutoff_label(k) for k in CUTOFFS],
        "semantics": {
            "cf_tool_cutoff": "first k bundles, then union and deduplicate their tools",
            "all": "all candidates actually stored for that route in that row",
            "zero_tool_rows": "excluded from tool-route P/R/F1; included in overall complete-configuration recovery",
            "duplicate_ranking": "slicing precedes component deduplication; duplicate bundles consume ranks",
        },
        "overall_from_rag_meta": overall.as_dict(),
        "data_quality_issues": dict(sorted(issues.items())),
        "route_metrics": metric_rows,
    }
    return summary, metric_rows, row_details


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_compact_report(summary: dict[str, Any]) -> None:
    overall = summary["overall_from_rag_meta"]
    print("\nAuthoritative overall coverage from rag_meta")
    for key in (
        "gold_llm_coverage",
        "gold_tool_recall_macro",
        "gold_tool_recall_micro",
        "all_gold_tools_covered_rate",
        "complete_configuration_recoverable_rate",
    ):
        print(f"  {key}: {overall[key]:.4%}")

    print("\nFour-route metrics")
    print(
        f"  {'route':32s} {'k':>4s} {'P-macro':>10s} {'R-macro':>10s} "
        f"{'F1-macro':>10s} {'all-gold':>10s} {'mean #':>8s}"
    )
    for row in summary["route_metrics"]:
        print(
            f"  {row['route']:32s} {row['cutoff']:>4s} "
            f"{row['macro_precision']:10.4%} {row['macro_recall']:10.4%} "
            f"{row['macro_f1']:10.4%} {row['all_gold_covered_rate']:10.4%} "
            f"{row['mean_retrieved_count']:8.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to pair_rag.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pair_rag_retrieval_metrics"),
        help="Directory for JSON/CSV outputs",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )
    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, metric_rows, row_details = evaluate(args.input)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "route_metrics.csv", metric_rows)
    write_csv(args.output_dir / "row_diagnostics.csv", row_details)
    print_compact_report(summary)
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
