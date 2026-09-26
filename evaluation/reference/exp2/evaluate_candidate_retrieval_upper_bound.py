#!/usr/bin/env python3
'Evaluate the candidate-retrieval upper bound before gold injection.\n\nThis script is designed for ``pair_rag.jsonl`` records with the schema::\n\n    {\n      "qid": ...,\n      "part": ...,\n      "context": ...,\n      "rag_meta": {\n        "gold_llm": ...,\n        "gold_tools": [...],\n        "covered_llm": bool,\n        "covered_tools": [...],\n        "missing_llm": ...,\n        "missing_tools": [...],\n        "injected_llm": bool,\n        "injected_tools": [...]\n      }\n    }\n\nTwo complementary retrieval views are produced at both configuration level and\nquery-level multi-positive level:\n\n1. ``logged_full_context`` uses ``rag_meta.covered_*``.  These fields record\n   coverage before forced gold insertion and are therefore authoritative for\n   the exact context used to construct the training record.\n2. ``retrieval_sweep`` parses the four ranked retrieval channels from\n   ``context``, removes every logged injected component, and evaluates the\n   fixed cutoffs 5, 10, 15, 20, 50, and 100.  It reports each retrieval alone,\n   the two CF channels together, the two semantic channels together, and the\n   four-channel hybrid union.\n3. ``all_retrieved`` applies no top-k cutoff.  For each record and channel, all\n   pre-injection candidates actually stored in that record are used.  This is\n   deliberately different from a claimed Top-k result when stored lists have\n   heterogeneous or shorter lengths.\n\nFor CF tool retrieval, k counts ranked *tool bundles*, not individual tools.\nCoverage is computed after expanding the first k bundles and deduplicating all\nindividual tools contained in them.  Semantic-tool k directly counts tools.\n\nPrimary metric definitions for sample i, with gold LLM m_i, gold tool set G_i,\ncandidate LLM set M_i, and candidate tool set T_i, are:\n\n  LLM coverage_i       = 1[m_i in M_i]\n  tool recall_i        = |G_i intersect T_i| / |G_i|       (|G_i| > 0)\n  all-tools covered_i  = 1[G_i subseteq T_i]\n  config recoverable_i = 1[m_i in M_i and G_i subseteq T_i]\n\nWhen one query has multiple valid agent configurations, query-level metrics do\nnot merge their tool sets.  Instead, they use the following multi-positive\ndefinitions:\n\n  LLM coverage_q       = max_i LLM coverage_i\n  tool recall_q        = max_i tool recall_i\n  all-tools covered_q  = max_i all-tools covered_i\n  config recoverable_q = max_i config recoverable_i\n\nwhere i ranges over valid positive configurations for query q.  Thus, a query\nis recoverable when at least one complete positive configuration is available.\n\nZero-tool examples are valid.  They count as all-tools-covered by vacuous truth\nand require only the gold LLM for configuration recoverability.  The primary\nmacro tool recall and all-tools coverage are reported on tool-requiring samples;\nseparate ``*_all`` fields include zero-tool samples with recall/coverage 1.\n\nOutputs (under --output-dir):\n  metrics.json              full machine-readable results and run metadata\n  metrics_by_scope_and_k.csv both evaluation levels by scope/cutoff/split\n  metrics_by_depth.csv      compatibility copy of the same table\n  metrics_logged.csv        both evaluation levels at stored context/split\n  summary.md                concise human-readable report\n  validation_issues.jsonl   malformed/inconsistent records (if any)\n  per_example.jsonl         optional configuration-level rows\n  per_query.jsonl           optional query-level multi-positive rows\n\nThe implementation uses only the Python standard library.\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CHANNELS = ("cf_llm", "cf_bundle", "semantic_llm", "semantic_tool")
CHANNEL_DISPLAY = {
    "cf_llm": "CF-LLM",
    "cf_bundle": "CF-tool-bundle",
    "semantic_llm": "Semantic-LLM",
    "semantic_tool": "Semantic-tool",
}
SECTION_NAMES = {
    "cf_retrieved_llm": "cf_llm",
    "cf_retrieved_tool_bundle": "cf_bundle",
    "semantic_retrieved_llm": "semantic_llm",
    "semantic_retrieved_tool": "semantic_tool",
}
TOOL_TOKEN_RE = re.compile(r"<<[^<>]+>>")
BUNDLE_RE = re.compile(r"\{(.*?)\}\s*\[\d+\](?=\s*,|\s*$)", re.DOTALL)
INDEXED_LLM_RE = re.compile(r"(<LLM_[^<>]+>)\s*\[\d+\]")
INDEXED_TOOL_RE = re.compile(r"(<<[^<>]+>>)\s*\[\d+\]")


SCOPE_CHANNELS = {
    "cf_llm": ("cf_llm",),
    "cf_tool_bundle": ("cf_bundle",),
    "semantic_llm": ("semantic_llm",),
    "semantic_tool": ("semantic_tool",),
    "cf_combined": ("cf_llm", "cf_bundle"),
    "semantic_combined": ("semantic_llm", "semantic_tool"),
    "hybrid_merged": CHANNELS,
}
SCOPE_DISPLAY = {
    "cf_llm": "CF-LLM",
    "cf_tool_bundle": "CF-tool-bundle",
    "semantic_llm": "Semantic-LLM",
    "semantic_tool": "Semantic-tool",
    "cf_combined": "CF combined",
    "semantic_combined": "Semantic combined",
    "hybrid_merged": "Hybrid merged",
}


@dataclass(frozen=True)
class EvaluationPoint:
    retrieval_scope: str
    cutoff: int | None

    @property
    def label(self) -> str:
        return "all_retrieved" if self.cutoff is None else f"top_{self.cutoff}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_scope": self.retrieval_scope,
            "retrieval_scope_display": SCOPE_DISPLAY[self.retrieval_scope],
            "cutoff_label": self.label,
            "top_k": self.cutoff,
            "uses_all_stored_candidates": self.cutoff is None,
        }


@dataclass
class Example:
    line_no: int
    qid: str
    part: str
    gold_llm: str
    gold_tools: frozenset[str]
    logged_llm_covered: bool
    logged_tools_covered: frozenset[str]
    channels: dict[str, list[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute pre-injection candidate-retrieval upper-bound metrics.",
    )
    parser.add_argument("--input", type=Path, required=True, help="pair_rag.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("retrieval_upper_bound"))
    parser.add_argument(
        "--depths",
        default="5,10,15,20,50,100",
        help="Comma-separated shared depths applied to all four retrieval channels.",
    )
    parser.add_argument(
        "--scopes",
        default=",".join(SCOPE_CHANNELS),
        help=("Comma-separated retrieval scopes. Available: "
              + ",".join(SCOPE_CHANNELS)),
    )
    parser.add_argument(
        "--no-all-retrieved", action="store_true",
        help="Disable the additional no-cutoff all_retrieved evaluation point.",
    )
    parser.add_argument(
        "--group-by",
        choices=("none", "part"),
        default="part",
        help="Also report authoritative logged metrics by this field.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on any genuine validation issue (repeated qids are expected positives).",
    )
    parser.add_argument("--save-per-example", action="store_true")
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def parse_positive_ints(raw: str, option: str) -> list[int]:
    if not raw.strip():
        return []
    try:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{option} must contain integers: {raw!r}") from exc
    if any(x <= 0 for x in values):
        raise ValueError(f"{option} values must be positive")
    return values


def build_evaluation_points(
    depths: str, scopes_raw: str, include_all: bool,
) -> list[EvaluationPoint]:
    depths_parsed = list(dict.fromkeys(parse_positive_ints(depths, "--depths")))
    if not depths_parsed:
        raise ValueError("At least one top-k depth is required")
    scopes = list(dict.fromkeys(x.strip() for x in scopes_raw.split(",") if x.strip()))
    unknown = [scope for scope in scopes if scope not in SCOPE_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown --scopes value(s): {unknown!r}; available={list(SCOPE_CHANNELS)!r}"
        )
    if not scopes:
        raise ValueError("At least one retrieval scope is required")
    points: list[EvaluationPoint] = []
    for scope in scopes:
        points.extend(EvaluationPoint(scope, k) for k in depths_parsed)
        if include_all:
            points.append(EvaluationPoint(scope, None))
    return points


def context_sections(context: str) -> dict[str, str]:
    """Return text belonging to each compact candidate-list section."""
    positions: list[tuple[int, str, int]] = []
    for source_name, channel in SECTION_NAMES.items():
        match = re.search(rf"(?m)^{re.escape(source_name)}:\s*$", context)
        if match:
            positions.append((match.start(), channel, match.end()))
    evidence = re.search(r"(?m)^evidence:\s*$", context)
    boundary = evidence.start() if evidence else len(context)
    positions.sort()
    result: dict[str, str] = {}
    for idx, (_, channel, content_start) in enumerate(positions):
        content_end = positions[idx + 1][0] if idx + 1 < len(positions) else boundary
        result[channel] = context[content_start:content_end]
    return result


def parse_channels(context: str) -> dict[str, list[Any]]:
    sections = context_sections(context)
    channels: dict[str, list[Any]] = {name: [] for name in CHANNELS}
    channels["cf_llm"] = INDEXED_LLM_RE.findall(sections.get("cf_llm", ""))
    channels["semantic_llm"] = INDEXED_LLM_RE.findall(sections.get("semantic_llm", ""))
    channels["semantic_tool"] = INDEXED_TOOL_RE.findall(sections.get("semantic_tool", ""))
    bundles: list[tuple[str, ...]] = []
    for body in BUNDLE_RE.findall(sections.get("cf_bundle", "")):
        tools = tuple(TOOL_TOKEN_RE.findall(body))
        if tools:
            bundles.append(tools)
    channels["cf_bundle"] = bundles
    return channels


def remove_injected(
    channels: dict[str, list[Any]], injected_llm: bool, gold_llm: str,
    injected_tools: set[str],
) -> dict[str, list[Any]]:
    """Remove forced gold items before reconstructing original ranked lists."""
    clean = {name: list(values) for name, values in channels.items()}
    # build_pairs inserts missing components into the semantic candidate lists.
    # It does not mutate the CF rankings or the contents of retrieved CF bundles.
    if injected_llm:
        clean["semantic_llm"] = [x for x in clean["semantic_llm"] if x != gold_llm]
    if injected_tools:
        clean["semantic_tool"] = [x for x in clean["semantic_tool"]
                                  if x not in injected_tools]
    return clean


def issue(line_no: int, qid: str, code: str, detail: str) -> dict[str, Any]:
    return {"line_no": line_no, "qid": qid, "code": code, "detail": detail}


def parse_record(line_no: int, record: Mapping[str, Any]) -> tuple[Example | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    qid = str(record.get("qid", f"__line_{line_no}"))
    meta = record.get("rag_meta")
    if not isinstance(meta, Mapping):
        return None, [issue(line_no, qid, "missing_rag_meta", "rag_meta is absent or not an object")]

    gold_llm = meta.get("gold_llm")
    gold_tools_raw = meta.get("gold_tools")
    if not isinstance(gold_llm, str) or not gold_llm:
        issues.append(issue(line_no, qid, "invalid_gold_llm", repr(gold_llm)))
    if not isinstance(gold_tools_raw, list) or not all(isinstance(x, str) for x in gold_tools_raw):
        issues.append(issue(line_no, qid, "invalid_gold_tools", repr(gold_tools_raw)))
    if issues:
        return None, issues

    gold_tools = frozenset(gold_tools_raw)
    covered_llm = meta.get("covered_llm")
    covered_tools_raw = meta.get("covered_tools")
    if not isinstance(covered_llm, bool):
        issues.append(issue(line_no, qid, "invalid_covered_llm", repr(covered_llm)))
        return None, issues
    if not isinstance(covered_tools_raw, list) or not all(isinstance(x, str) for x in covered_tools_raw):
        issues.append(issue(line_no, qid, "invalid_covered_tools", repr(covered_tools_raw)))
        return None, issues
    covered_tools = frozenset(covered_tools_raw)

    missing_tools_raw = meta.get("missing_tools", [])
    injected_tools_raw = meta.get("injected_tools", [])
    if not isinstance(missing_tools_raw, list) or not all(isinstance(x, str) for x in missing_tools_raw):
        issues.append(issue(line_no, qid, "invalid_missing_tools", repr(missing_tools_raw)))
        missing_tools_raw = []
    if not isinstance(injected_tools_raw, list) or not all(isinstance(x, str) for x in injected_tools_raw):
        issues.append(issue(line_no, qid, "invalid_injected_tools", repr(injected_tools_raw)))
        injected_tools_raw = []

    if not covered_tools.issubset(gold_tools):
        issues.append(issue(line_no, qid, "covered_tools_not_gold",
                            repr(sorted(covered_tools - gold_tools))))
    expected_missing = gold_tools - covered_tools
    if set(missing_tools_raw) != expected_missing:
        issues.append(issue(line_no, qid, "missing_tools_inconsistent",
                            f"expected={sorted(expected_missing)!r}; logged={sorted(set(missing_tools_raw))!r}"))
    if set(injected_tools_raw) != expected_missing:
        issues.append(issue(line_no, qid, "injected_tools_inconsistent",
                            f"expected={sorted(expected_missing)!r}; logged={sorted(set(injected_tools_raw))!r}"))
    injected_llm = meta.get("injected_llm")
    if not isinstance(injected_llm, bool):
        issues.append(issue(line_no, qid, "invalid_injected_llm", repr(injected_llm)))
        injected_llm = not covered_llm
    elif injected_llm == covered_llm:
        issues.append(issue(line_no, qid, "injected_llm_inconsistent",
                            f"covered_llm={covered_llm}; injected_llm={injected_llm}"))

    context = record.get("context", "")
    if not isinstance(context, str):
        issues.append(issue(line_no, qid, "invalid_context", type(context).__name__))
        context = ""
    parsed = parse_channels(context)
    missing_sections = [name for name, values in parsed.items() if not values and name != "cf_bundle"]
    if missing_sections:
        issues.append(issue(line_no, qid, "empty_candidate_channels", ",".join(missing_sections)))
    clean = remove_injected(parsed, injected_llm, gold_llm, set(injected_tools_raw))

    # Validate parser reconstruction against the authoritative pre-injection log.
    full_llms = set(clean["cf_llm"]) | set(clean["semantic_llm"])
    full_tools = set(clean["semantic_tool"])
    for bundle in clean["cf_bundle"]:
        full_tools.update(bundle)
    parsed_covered_llm = gold_llm in full_llms
    parsed_covered_tools = gold_tools & full_tools
    if parsed_covered_llm != covered_llm:
        issues.append(issue(line_no, qid, "parser_llm_coverage_mismatch",
                            f"logged={covered_llm}; parsed={parsed_covered_llm}"))
    if parsed_covered_tools != covered_tools:
        issues.append(issue(line_no, qid, "parser_tool_coverage_mismatch",
                            f"logged={sorted(covered_tools)!r}; parsed={sorted(parsed_covered_tools)!r}"))

    return Example(
        line_no=line_no,
        qid=qid,
        part=str(record.get("part", "UNKNOWN")),
        gold_llm=gold_llm,
        gold_tools=gold_tools,
        logged_llm_covered=covered_llm,
        logged_tools_covered=covered_tools,
        channels=clean,
    ), issues


def read_examples(
    path: Path, encoding: str
) -> tuple[list[Example], list[dict[str, Any]], str, int, dict[str, Any]]:
    examples: list[Example] = []
    issues: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    records_seen = 0
    with path.open("rb") as raw:
        for line_no, raw_line in enumerate(raw, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            records_seen += 1
            try:
                line = raw_line.decode(encoding)
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                issues.append(issue(line_no, "", "invalid_json", str(exc)))
                continue
            if not isinstance(record, Mapping):
                issues.append(issue(line_no, "", "record_not_object", type(record).__name__))
                continue
            example, record_issues = parse_record(line_no, record)
            issues.extend(record_issues)
            if example is not None:
                examples.append(example)

    qid_groups: defaultdict[str, list[Example]] = defaultdict(list)
    for example in examples:
        qid_groups[example.qid].append(example)
    positive_counts = Counter(len(group) for group in qid_groups.values())
    mixed_part_qids = 0
    for qid, group in qid_groups.items():
        parts = sorted({example.part for example in group})
        if len(parts) > 1:
            mixed_part_qids += 1
            issues.append(issue(
                0, qid, "qid_part_mismatch",
                f"the same qid occurs in multiple parts: {parts!r}",
            ))
    qid_statistics = {
        "num_unique_queries": len(qid_groups),
        "num_configuration_rows": len(examples),
        "num_rows_beyond_first_positive": len(examples) - len(qid_groups),
        "num_multi_positive_queries": sum(1 for group in qid_groups.values() if len(group) > 1),
        "positive_configurations_per_query": {
            str(count): frequency for count, frequency in sorted(positive_counts.items())
        },
        "mean_positive_configurations_per_query": (
            len(examples) / len(qid_groups) if qid_groups else None
        ),
        "num_qids_with_mixed_parts": mixed_part_qids,
    }
    return examples, issues, digest.hexdigest(), records_seen, qid_statistics


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mean_interval(values: Sequence[float], z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return max(0.0, mean - half), min(1.0, mean + half)


def metric_values(
    llm_hit: bool | None,
    gold_tools: frozenset[str],
    candidate_tools: set[str] | None,
) -> dict[str, Any]:
    """Return per-positive metrics, preserving metric applicability by scope."""
    gold_count = len(gold_tools)
    tool_applicable = candidate_tools is not None
    hits = len(gold_tools & candidate_tools) if tool_applicable else None
    tool_recall = (
        hits / gold_count if tool_applicable and gold_count else None
    )
    tool_recall_all = (
        tool_recall if tool_recall is not None else (1.0 if tool_applicable else None)
    )
    all_tools = int(hits == gold_count) if tool_applicable else None
    llm_value = int(llm_hit) if llm_hit is not None else None
    config_applicable = llm_hit is not None and tool_applicable
    return {
        "llm_metric_applicable": llm_hit is not None,
        "tool_metrics_applicable": tool_applicable,
        "config_metric_applicable": config_applicable,
        "llm_hit": llm_value,
        "gold_tool_count": gold_count,
        "tool_hits": hits,
        "tool_recall": tool_recall,
        "tool_recall_all": tool_recall_all,
        "all_tools_hit": all_tools,
        "config_hit": int(bool(llm_hit) and bool(all_tools)) if config_applicable else None,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    llm_rows = [row for row in rows if row.get("llm_metric_applicable", True)]
    applicable_tool_rows = [row for row in rows if row.get("tool_metrics_applicable", True)]
    tool_rows = [row for row in applicable_tool_rows if row["gold_tool_count"] > 0]
    config_rows = [row for row in rows if row.get("config_metric_applicable", True)]
    total_gold_tools = sum(int(row["gold_tool_count"]) for row in tool_rows)
    total_tool_hits = sum(int(row["tool_hits"]) for row in tool_rows)

    def binary(name: str, selected: Sequence[Mapping[str, Any]] = rows) -> dict[str, Any]:
        successes = sum(int(row[name]) for row in selected)
        low, high = wilson_interval(successes, len(selected))
        return {"value": successes / len(selected) if selected else None,
                "numerator": successes, "denominator": len(selected),
                "ci95_low": low, "ci95_high": high, "ci_method": "Wilson"}

    recalls = [float(row["tool_recall"]) for row in tool_rows]
    recall_low, recall_high = mean_interval(recalls)
    all_recalls = [float(row["tool_recall_all"]) for row in applicable_tool_rows]
    all_low, all_high = mean_interval(all_recalls)
    return {
        "num_examples": n,
        "num_llm_metric_examples": len(llm_rows),
        "num_tool_metric_examples": len(applicable_tool_rows),
        "num_configuration_metric_examples": len(config_rows),
        "num_tool_requiring_examples": len(tool_rows),
        "num_zero_tool_examples": len(applicable_tool_rows) - len(tool_rows),
        "llm_coverage": binary("llm_hit", llm_rows),
        "tool_recall_macro": {
            "value": statistics.fmean(recalls) if recalls else None,
            "denominator": len(recalls), "ci95_low": recall_low, "ci95_high": recall_high,
            "ci_method": "normal approximation over per-example recall",
        },
        "tool_recall_micro": {
            "value": total_tool_hits / total_gold_tools if total_gold_tools else None,
            "numerator": total_tool_hits, "denominator": total_gold_tools,
        },
        "all_gold_tools_covered": binary("all_tools_hit", tool_rows),
        "complete_configuration_recoverable": binary("config_hit", config_rows),
        "tool_recall_macro_all": {
            "value": statistics.fmean(all_recalls) if all_recalls else None,
            "denominator": len(all_recalls), "ci95_low": all_low, "ci95_high": all_high,
            "ci_method": "normal approximation; zero-tool recall defined as 1",
        },
        "all_gold_tools_covered_all": binary("all_tools_hit", applicable_tool_rows),
    }


def aggregate_multi_positive_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate positive configurations without constructing a tool union.

    Each metric is an existential/best-positive quantity.  For macro tool
    recall, the best recall among tool-requiring positives is used.  The
    selected positive's hit/count pair is retained so the query-level micro
    recall remains internally consistent.  A query with only zero-tool
    positives receives the standard zero-tool representation (0 hits, 0 gold).
    """
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["qid"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for qid, positives in groups.items():
        llm_values = [int(row["llm_hit"]) for row in positives
                      if row.get("llm_metric_applicable", True)]
        applicable_tools = [row for row in positives
                            if row.get("tool_metrics_applicable", True)]
        tool_positives = [row for row in applicable_tools
                          if int(row["gold_tool_count"]) > 0]
        if tool_positives:
            best_tool = max(
                tool_positives,
                key=lambda row: (
                    float(row["tool_recall"]),
                    int(row["tool_hits"]),
                    -int(row["gold_tool_count"]),
                ),
            )
            gold_tool_count = int(best_tool["gold_tool_count"])
            tool_hits = int(best_tool["tool_hits"])
            tool_recall = float(best_tool["tool_recall"])
        else:
            gold_tool_count = 0
            tool_hits = 0 if applicable_tools else None
            tool_recall = None

        all_tools_values = [int(row["all_tools_hit"]) for row in applicable_tools]
        config_values = [int(row["config_hit"]) for row in positives
                         if row.get("config_metric_applicable", True)]

        parts = {str(row.get("part", "UNKNOWN")) for row in positives}
        row_out: dict[str, Any] = {
            "qid": qid,
            "part": next(iter(parts)) if len(parts) == 1 else "MIXED",
            "num_positive_configurations": len(positives),
            "llm_metric_applicable": bool(llm_values),
            "tool_metrics_applicable": bool(applicable_tools),
            "config_metric_applicable": bool(config_values),
            "llm_hit": max(llm_values) if llm_values else None,
            "gold_tool_count": gold_tool_count,
            "tool_hits": tool_hits,
            "tool_recall": tool_recall,
            "tool_recall_all": (
                max(float(row["tool_recall_all"]) for row in applicable_tools)
                if applicable_tools else None
            ),
            "all_tools_hit": max(all_tools_values) if all_tools_values else None,
            "config_hit": max(config_values) if config_values else None,
        }
        for key in ("retrieval_scope", "cutoff_label", "top_k"):
            if key in positives[0]:
                row_out[key] = positives[0][key]
        aggregated.append(row_out)
    return aggregated


def candidate_sets(
    example: Example, point: EvaluationPoint,
) -> tuple[set[str] | None, set[str] | None, dict[str, int]]:
    """Build candidates for one scope; CF bundle ranks are expanded to tools."""
    selected_channels = set(SCOPE_CHANNELS[point.retrieval_scope])
    cutoff = point.cutoff
    selected: dict[str, list[Any]] = {}
    for channel in CHANNELS:
        values = example.channels[channel]
        selected[channel] = list(values if cutoff is None else values[:cutoff])

    llm_applicable = bool(selected_channels & {"cf_llm", "semantic_llm"})
    tool_applicable = bool(selected_channels & {"cf_bundle", "semantic_tool"})
    llms: set[str] | None = set() if llm_applicable else None
    tools: set[str] | None = set() if tool_applicable else None
    if llms is not None:
        if "cf_llm" in selected_channels:
            llms.update(selected["cf_llm"])
        if "semantic_llm" in selected_channels:
            llms.update(selected["semantic_llm"])
    if tools is not None:
        if "semantic_tool" in selected_channels:
            tools.update(selected["semantic_tool"])
        if "cf_bundle" in selected_channels:
            for bundle in selected["cf_bundle"]:
                tools.update(bundle)

    available: dict[str, int] = {
        "llm_candidates": len(llms) if llms is not None else 0,
        "tool_candidates": len(tools) if tools is not None else 0,
    }
    for channel in CHANNELS:
        is_selected = channel in selected_channels
        available[f"{channel}_observed"] = len(selected[channel]) if is_selected else 0
        available[f"{channel}_available"] = len(example.channels[channel]) if is_selected else 0
        available[f"{channel}_shorter_than_k"] = int(
            is_selected and cutoff is not None and len(example.channels[channel]) < cutoff
        )
    return llms, tools, available


def evaluate_logged(
    examples: Sequence[Example],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for ex in examples:
        candidate_tools = set(ex.logged_tools_covered)
        row = metric_values(ex.logged_llm_covered, ex.gold_tools, candidate_tools)
        row.update({"qid": ex.qid, "part": ex.part})
        rows.append(row)
    query_rows = aggregate_multi_positive_rows(rows)
    return summarize(rows), summarize(query_rows), rows, query_rows


def evaluate_sweep(
    examples: Sequence[Example], points: Sequence[EvaluationPoint]
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    configuration_results: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for point in points:
        rows: list[dict[str, Any]] = []
        costs: defaultdict[str, list[int]] = defaultdict(list)
        costs_by_qid: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for ex in examples:
            llms, tools, available = candidate_sets(ex, point)
            llm_hit = ex.gold_llm in llms if llms is not None else None
            row = metric_values(llm_hit, ex.gold_tools, tools)
            row.update({"qid": ex.qid, "part": ex.part, **point.as_dict()})
            rows.append(row)
            for key, value in available.items():
                costs[key].append(value)
                costs_by_qid[ex.qid][key].append(value)
            per_example.append(row | {
                "candidate_llm_count": len(llms) if llms is not None else None,
                "candidate_tool_count": len(tools) if tools is not None else None,
            })
        summary = summarize(rows)
        summary.update(point.as_dict())
        selected_channels = SCOPE_CHANNELS[point.retrieval_scope]
        summary["candidate_statistics"] = {
            "mean_deduplicated_llms": statistics.fmean(costs["llm_candidates"]),
            "mean_deduplicated_tools": statistics.fmean(costs["tool_candidates"]),
            "median_deduplicated_llms": statistics.median(costs["llm_candidates"]),
            "median_deduplicated_tools": statistics.median(costs["tool_candidates"]),
            **{f"mean_{channel}_observed": statistics.fmean(costs[f"{channel}_observed"])
               for channel in CHANNELS},
            **{f"max_{channel}_observed": max(costs[f"{channel}_observed"])
               for channel in CHANNELS},
            **{f"mean_{channel}_available": statistics.fmean(costs[f"{channel}_available"])
               for channel in CHANNELS},
            **{f"max_{channel}_available": max(costs[f"{channel}_available"])
               for channel in CHANNELS},
            **{f"fraction_{channel}_shorter_than_k":
               statistics.fmean(costs[f"{channel}_shorter_than_k"])
               for channel in CHANNELS},
        }
        summary["candidate_statistics"]["mean_ranked_items_selected"] = sum(
            summary["candidate_statistics"][f"mean_{channel}_observed"]
            for channel in selected_channels
        )
        configuration_results.append(summary)
        query_rows = aggregate_multi_positive_rows(rows)
        for row in query_rows:
            qid_costs = costs_by_qid[row["qid"]]
            row["candidate_llm_count"] = statistics.fmean(qid_costs["llm_candidates"])
            row["candidate_tool_count"] = statistics.fmean(qid_costs["tool_candidates"])
        per_query.extend(query_rows)
        query_summary = summarize(query_rows)
        query_summary.update(point.as_dict())
        query_summary["candidate_statistics"] = {
            "mean_deduplicated_llms": statistics.fmean(
                statistics.fmean(values["llm_candidates"])
                for values in costs_by_qid.values()
            ),
            "mean_deduplicated_tools": statistics.fmean(
                statistics.fmean(values["tool_candidates"])
                for values in costs_by_qid.values()
            ),
            "median_deduplicated_llms": statistics.median(
                statistics.fmean(values["llm_candidates"])
                for values in costs_by_qid.values()
            ),
            "median_deduplicated_tools": statistics.median(
                statistics.fmean(values["tool_candidates"])
                for values in costs_by_qid.values()
            ),
            **{
                f"mean_{channel}_observed": statistics.fmean(
                    statistics.fmean(values[f"{channel}_observed"])
                    for values in costs_by_qid.values()
                )
                for channel in CHANNELS
            },
            **{
                f"max_{channel}_observed": max(
                    max(values[f"{channel}_observed"])
                    for values in costs_by_qid.values()
                )
                for channel in CHANNELS
            },
            **{
                f"mean_{channel}_available": statistics.fmean(
                    statistics.fmean(values[f"{channel}_available"])
                    for values in costs_by_qid.values()
                )
                for channel in CHANNELS
            },
            **{
                f"max_{channel}_available": max(
                    max(values[f"{channel}_available"])
                    for values in costs_by_qid.values()
                )
                for channel in CHANNELS
            },
            **{
                f"fraction_{channel}_shorter_than_k": statistics.fmean(
                    statistics.fmean(values[f"{channel}_shorter_than_k"])
                    for values in costs_by_qid.values()
                )
                for channel in CHANNELS
            },
        }
        query_summary["candidate_statistics"]["mean_ranked_items_selected"] = sum(
            query_summary["candidate_statistics"][f"mean_{channel}_observed"]
            for channel in selected_channels
        )
        query_results.append(query_summary)
    return configuration_results, query_results, per_example, per_query


def flatten_metrics(base: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: value for key, value in base.items()
        if not isinstance(value, Mapping)
    }
    for key in ("llm_coverage", "tool_recall_macro", "tool_recall_micro",
                "all_gold_tools_covered", "complete_configuration_recoverable",
                "tool_recall_macro_all", "all_gold_tools_covered_all"):
        data = base.get(key, {})
        if isinstance(data, Mapping):
            row[key] = data.get("value")
            if key in {"llm_coverage", "tool_recall_macro", "all_gold_tools_covered",
                       "complete_configuration_recoverable"}:
                row[f"{key}_ci95_low"] = data.get("ci95_low")
                row[f"{key}_ci95_high"] = data.get("ci95_high")
    candidate_stats = base.get("candidate_statistics", {})
    if isinstance(candidate_stats, Mapping):
        row.update(candidate_stats)
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat = [flatten_metrics(row) for row in rows]
    if not flat:
        return
    fieldnames = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def summary_markdown(
    input_path: Path, logged_rows: Sequence[Mapping[str, Any]], sweep: Sequence[Mapping[str, Any]],
    issue_counts: Counter[str], valid: int, skipped: int, qid_statistics: Mapping[str, Any],
) -> str:
    overall_rows = [row for row in logged_rows if row["group"] == "ALL"]
    lines = [
        "# Candidate Retrieval Upper Bound",
        "",
        f"- Input: `{input_path}`",
        f"- Valid configuration rows: {valid:,}",
        f"- Unique queries: {int(qid_statistics['num_unique_queries']):,}",
        ("- Mean positive configurations per query: "
         f"{float(qid_statistics['mean_positive_configurations_per_query']):.2f}"),
        f"- Skipped malformed examples: {skipped:,}",
        f"- Genuine validation issues: {sum(issue_counts.values()):,}",
        "",
        "## Authoritative pre-injection coverage at the stored context",
        "",
        "| Evaluation level | Gold LLM coverage | Gold tool recall (macro) | All gold tools covered | Complete configuration recoverable |",
        "|---|---:|---:|---:|---:|",
    ]
    for overall in overall_rows:
        lines.append(
            f"| {overall['evaluation_level']} | {pct(overall['llm_coverage']['value'])} | "
            f"{pct(overall['tool_recall_macro']['value'])} | "
            f"{pct(overall['all_gold_tools_covered']['value'])} | "
            f"{pct(overall['complete_configuration_recoverable']['value'])} |"
        )
    lines.extend([
        "",
        "Tool recall and all-tools coverage above are conditional on at least one gold tool. "
        "Complete-configuration recoverability includes zero-tool examples.",
        "At query level, each metric takes the best/any valid positive configuration; tools "
        "from different positives are never merged into a synthetic gold bundle.",
        "",
        "## Retrieval scope and cutoff comparison (gold injection removed)",
        "",
        "The main table uses query-level multi-positive evaluation. Configuration-level "
        "results are retained in the CSV and JSON outputs.",
        "",
        "| Retrieval scope | Cutoff | Mean ranked units | Mean # LLMs | Mean # tools | LLM coverage | Tool recall | All tools | Full config |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sweep:
        if (row.get("group", "ALL") != "ALL"
                or row.get("evaluation_level") != "query_multi_positive"):
            continue
        stats = row["candidate_statistics"]
        llm_count = (
            f"{stats['mean_deduplicated_llms']:.2f}"
            if row["num_llm_metric_examples"] else "NA"
        )
        tool_count = (
            f"{stats['mean_deduplicated_tools']:.2f}"
            if row["num_tool_metric_examples"] else "NA"
        )
        lines.append(
            f"| {row['retrieval_scope_display']} | {row['cutoff_label']} | "
            f"{stats['mean_ranked_items_selected']:.2f} | "
            f"{llm_count} | {tool_count} | {pct(row['llm_coverage']['value'])} | "
            f"{pct(row['tool_recall_macro']['value'])} | "
            f"{pct(row['all_gold_tools_covered']['value'])} | "
            f"{pct(row['complete_configuration_recoverable']['value'])} |"
        )
    lines.extend([
        "",
        "`all_retrieved` has no common k: each example uses every pre-injection item "
        "actually stored in the selected channel(s). For CF-tool-bundle, a ranked unit is "
        "one bundle; `Mean # tools` is the number of distinct individual tools after bundle expansion.",
        "",
        "## Multi-positive data statistics",
        "",
        ("- Queries with more than one positive: "
         f"{int(qid_statistics['num_multi_positive_queries']):,}"),
        ("- Positive-count distribution (`positives: queries`): `"
         + json.dumps(qid_statistics["positive_configurations_per_query"], sort_keys=True)
         + "`"),
        "- Repeated `qid` values are expected multi-positive observations and are not validation issues.",
    ])
    if issue_counts:
        lines.extend(["", "## Validation diagnostics", ""])
        for code, count in issue_counts.most_common():
            lines.append(f"- `{code}`: {count:,}")
    lines.extend([
        "",
        "## Reporting note",
        "",
        'For a finite k, each selected retrieval is truncated independently at that k. '
        "If a stored list is shorter, all available items are used and the corresponding "
        "`fraction_*_shorter_than_k` column records the shortfall. Such a row is an observed-log "
        'evaluation, not proof that the upstream retrieval actually returned k results. '
        "Use `all_retrieved` when reporting the no-cutoff effect and use the mean deduplicated "
        "candidate counts when discussing search-space cost.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        points = build_evaluation_points(
            args.depths, args.scopes, include_all=not args.no_all_retrieved
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: input does not exist or is not a file: {args.input}", file=sys.stderr)
        return 2

    examples, issues, sha256, records_seen, qid_statistics = read_examples(
        args.input, args.encoding
    )
    if not examples:
        print("error: no valid examples", file=sys.stderr)
        return 2
    if args.strict and issues:
        print(f"error: strict validation failed with {len(issues)} issue(s)", file=sys.stderr)
        for item in issues[:20]:
            print(json.dumps(item, ensure_ascii=False), file=sys.stderr)
        return 1

    logged_conf_all, logged_query_all, logged_example_rows, logged_query_rows = evaluate_logged(
        examples
    )
    logged_rows: list[dict[str, Any]] = [
        {"group": "ALL", "evaluation_level": "configuration", **logged_conf_all},
        {"group": "ALL", "evaluation_level": "query_multi_positive", **logged_query_all},
    ]
    if args.group_by == "part":
        groups: defaultdict[str, list[Example]] = defaultdict(list)
        for ex in examples:
            groups[ex.part].append(ex)
        for name in sorted(groups):
            conf_metrics, query_metrics, _, _ = evaluate_logged(groups[name])
            logged_rows.extend([
                {"group": name, "evaluation_level": "configuration", **conf_metrics},
                {"group": name, "evaluation_level": "query_multi_positive", **query_metrics},
            ])

    sweep_conf_all, sweep_query_all, sweep_per_example, sweep_per_query = evaluate_sweep(
        examples, points
    )
    sweep: list[dict[str, Any]] = [
        *({"group": "ALL", "evaluation_level": "configuration", **row}
          for row in sweep_conf_all),
        *({"group": "ALL", "evaluation_level": "query_multi_positive", **row}
          for row in sweep_query_all),
    ]
    if args.group_by == "part":
        sweep_groups: defaultdict[str, list[Example]] = defaultdict(list)
        for ex in examples:
            sweep_groups[ex.part].append(ex)
        for name in sorted(sweep_groups):
            group_conf, group_query, _, _ = evaluate_sweep(sweep_groups[name], points)
            sweep.extend(
                {"group": name, "evaluation_level": "configuration", **row}
                for row in group_conf
            )
            sweep.extend(
                {"group": name, "evaluation_level": "query_multi_positive", **row}
                for row in group_query
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_path = args.output_dir / "validation_issues.jsonl"
    with issue_path.open("w", encoding="utf-8") as handle:
        for item in issues:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    issue_counts = Counter(item["code"] for item in issues)
    payload = {
        "schema_version": "3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input.resolve()), "sha256": sha256},
        "evaluation_protocol": {
            "gold_injection_excluded": True,
            "logged_full_context_is_authoritative": True,
            "zero_tool_policy": {
                "primary_tool_metrics": "exclude zero-tool examples",
                "all_suffix_metrics": "include zero-tool examples with recall/coverage=1",
                "configuration_recoverability": "include; requires gold LLM only",
            },
            "confidence_intervals": {
                "binary_metrics": "95% Wilson score interval",
                "macro_tool_recall": "95% normal approximation over per-example recall",
            },
            "multi_positive_query_metrics": {
                "llm_coverage": "any positive LLM is covered",
                "tool_recall_macro": "maximum recall over positive configurations",
                "all_gold_tools_covered": "all tools of any one positive are covered",
                "complete_configuration_recoverable": (
                    "LLM and all tools of any one positive are jointly covered"
                ),
                "tool_union_across_positives": False,
            },
            "retrieval_protocol": {
                "channel_order": list(CHANNELS),
                "retrieval_scopes": {
                    scope: list(channels) for scope, channels in SCOPE_CHANNELS.items()
                },
                "finite_top_k_values": parse_positive_ints(args.depths, "--depths"),
                "all_retrieved_enabled": not args.no_all_retrieved,
                "cf_tool_k_unit": "ranked tool bundle",
                "cf_tool_coverage_unit": (
                    "deduplicated individual tool after bundle expansion"
                ),
                "semantic_tool_k_unit": "individual tool",
                "short_list_policy": (
                    "use all stored items and report fraction_*_shorter_than_k"
                ),
            },
        },
        "data_quality": {
            "valid_examples": len(examples),
            "records_seen": records_seen,
            "skipped_examples": records_seen - len(examples),
            "issue_count": len(issues),
            "issue_counts": dict(sorted(issue_counts.items())),
            "multi_positive_statistics": qid_statistics,
        },
        "logged_full_context": logged_rows,
        "retrieval_sweep": sweep,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_csv(args.output_dir / "metrics_by_scope_and_k.csv", sweep)
    write_csv(args.output_dir / "metrics_by_depth.csv", sweep)
    write_csv(args.output_dir / "metrics_logged.csv", logged_rows)

    if args.save_per_example:
        with (args.output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
            for row in logged_example_rows:
                handle.write(json.dumps(
                    {"evaluation_source": "logged_full_context",
                     "evaluation_level": "configuration", **row},
                    ensure_ascii=False,
                ) + "\n")
            for row in sweep_per_example:
                handle.write(json.dumps(
                    {"evaluation_source": "retrieval_sweep",
                     "evaluation_level": "configuration", **row},
                    ensure_ascii=False,
                ) + "\n")
        with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
            for row in logged_query_rows:
                handle.write(json.dumps(
                    {"evaluation_source": "logged_full_context",
                     "evaluation_level": "query_multi_positive", **row},
                    ensure_ascii=False,
                ) + "\n")
            for row in sweep_per_query:
                handle.write(json.dumps(
                    {"evaluation_source": "retrieval_sweep",
                     "evaluation_level": "query_multi_positive", **row},
                    ensure_ascii=False,
                ) + "\n")

    skipped = payload["data_quality"]["skipped_examples"]
    report = summary_markdown(
        args.input, logged_rows, sweep, issue_counts,
        len(examples), skipped, qid_statistics,
    )
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
