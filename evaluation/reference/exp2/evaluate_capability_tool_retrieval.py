#!/usr/bin/env python3
"""Evaluate capability-aware semantic tool retrieval on PartII pairs.

All experiment parameters are fixed in the CONFIG section. Run directly:

    python evaluate_capability_tool_retrieval.py

The script calls the unified 8504 `/retrieve` route with targets=["tool"], so
the service performs capability-aware query rewriting and tool retrieval but
does not run LLM retrieval. Each rewritten tool subquery contributes up to 20
candidates; the service returns a globally score-sorted list of at most 50.

Evaluation uses the same deterministic 1,000-query sample (seed 42) and the
same tool-level metrics as evaluate_cf_tool_bundle_retrieval.py.

Outputs under OUTPUT_DIR:
  summary.md
  aggregate_metrics.csv
  aggregate_metrics.json
  per_example.jsonl
  retrieval_cache.jsonl
  validation_issues.jsonl / retrieval_errors.jsonl (only when non-empty)

Only the Python standard library is required.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# =============================================================================
# CONFIG: fixed experiment parameters; no command-line/shell arguments needed.
# =============================================================================

INPUT_PATH = Path(
    str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/PartII_pairs.jsonl')
)
AGENT_CATALOG_PATH = Path(
    str(AC_ROOT / 'datasets/PartII/agents/merge.json')
)
OUTPUT_DIR = INPUT_PATH.parent / "capability_tool_retrieval_eval_hybrid_50_50_False"

SEMANTIC_RETR_URL = "http://127.0.0.1:8504/retrieve"
EVAL_TOP_KS = (5, 10, 15, 20, 50)
SEMANTIC_TOOL_TOPK = 50          # Per rewritten subquery.
MAX_SEMANTIC_TOOL = 50           # Final globally ranked tool list.
AGGREGATION = "hybrid"
REWRITE_QUERY = False

SAMPLE_UNIQUE_QUERIES = 1000
RANDOM_SEED = 42
WORKERS = 8                      # Set to 1 if the rewrite/model service is serial.
REQUEST_TIMEOUT_SEC = 180
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0
RESUME = True
CASE_SENSITIVE_TOOL_IDS = True
FAIL_ON_MALFORMED_ROW = False


TOOL_TOKEN_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")
CITATION_RE = re.compile(r"\s*\[\d+\]\s*$")
WS_RE = re.compile(r"\s+")
TOOL_VALUE_KEYS = (
    "token", "tool_token", "component", "tool", "tool_name", "name",
    "api_token", "id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_tool(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    match = TOOL_TOKEN_RE.fullmatch(text)
    if match:
        text = match.group(1)
    text = CITATION_RE.sub("", text).strip().strip("'").strip('"').strip()
    text = WS_RE.sub(" ", text)
    return text if CASE_SENSITIVE_TOOL_IDS else text.casefold()


def parse_tool_value(value: Any) -> list[str]:
    """Parse gold tool values from PartII labels/configurations."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "T" in value and isinstance(value["T"], Mapping):
            nested = parse_tool_value(value["T"].get("tools"))
            if nested or "tools" in value["T"]:
                return nested
        for key in ("tools", "tool_list", "tool_names", "tool_bundle", "bundle"):
            if key in value:
                return parse_tool_value(value[key])
        for key in ("agent", "configuration", "config", "target_agent"):
            if key in value:
                parsed = parse_tool_value(value[key])
                if parsed:
                    return parsed
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parsed = parse_tool_value(item)
                if parsed:
                    output.extend(parsed)
                    continue
                for key in TOOL_VALUE_KEYS:
                    if key in item:
                        output.extend(parse_tool_value(item[key]))
                        break
            else:
                output.extend(parse_tool_value(item))
        return stable_unique(output)

    text = str(value).strip()
    if not text:
        return []
    tokens = [normalize_tool(token) for token in TOOL_TOKEN_RE.findall(text)]
    if tokens:
        return stable_unique(tokens)
    if text[:1] in "[{" and text[-1:] in "]}":
        try:
            return parse_tool_value(json.loads(text))
        except json.JSONDecodeError:
            pass
    tool = normalize_tool(text)
    return [tool] if tool else []


def first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def extract_query(record: Mapping[str, Any]) -> str:
    value = first_nonempty(record, ("query", "question", "input", "prompt", "text"))
    if isinstance(value, Mapping):
        value = first_nonempty(value, ("input", "query", "question", "text", "prompt"))
    if isinstance(value, list):
        value = "\n".join(
            str(item["content"])
            for item in value
            if isinstance(item, Mapping) and item.get("content")
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cannot find a non-empty query/question/input field")
    return value.strip()


def extract_qid(record: Mapping[str, Any], line_no: int) -> str:
    value = first_nonempty(
        record, ("qid", "question_id", "query_id", "id", "pair_id", "agent_id")
    )
    return str(value) if value is not None else f"line_{line_no}"


def extract_gold_tools(
    record: Mapping[str, Any], agent_catalog: Mapping[str, tuple[str, ...]],
) -> list[str]:
    # Never inspect query text: only explicit labels/configurations are allowed.
    for key in ("gold_tools", "target_tools", "tools", "tool_bundle"):
        if key in record:
            return parse_tool_value(record[key])
    for key in ("agent", "target_agent", "gold_agent", "configuration", "config"):
        if key in record:
            parsed = parse_tool_value(record[key])
            if parsed or isinstance(record[key], Mapping):
                return parsed
    for key in ("target", "label", "answer", "output", "response"):
        if key in record:
            parsed = parse_tool_value(record[key])
            if parsed:
                return parsed
    agent_id = first_nonempty(record, ("agent_id", "target_agent_id", "gold_agent_id"))
    if agent_id is not None and str(agent_id) in agent_catalog:
        return list(agent_catalog[str(agent_id)])
    raise ValueError(
        "cannot find gold tools; expected gold_tools/tools, agent T.tools, "
        "agent_id resolvable in the catalog, or <<tool>> tokens in the target"
    )


@dataclass(frozen=True)
class Example:
    line_no: int
    qid: str
    query: str
    gold_tools: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedTool:
    tool: str
    raw_score: float | None
    normalized_score: float | None
    matched_subquery: Any
    item_id: str | None


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                if FAIL_ON_MALFORMED_ROW:
                    raise ValueError(f"Malformed JSON at line {line_no}: {exc}") from exc
                print(f"WARNING: skip malformed JSON line {line_no}: {exc}", file=sys.stderr)


def load_agent_catalog(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        print(
            f"WARNING: agent catalog not found: {path}; agent_id-only gold labels cannot be expanded",
            file=sys.stderr,
        )
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Agent catalog must be a JSON object: {path}")
    return {str(agent_id): tuple(parse_tool_value(agent)) for agent_id, agent in raw.items()}


def load_examples(
    path: Path, agent_catalog: Mapping[str, tuple[str, ...]],
) -> tuple[list[Example], list[dict[str, Any]]]:
    examples: list[Example] = []
    issues: list[dict[str, Any]] = []
    for line_no, record in iter_jsonl(path):
        if not isinstance(record, Mapping):
            issues.append({"line_no": line_no, "error": "JSON value is not an object"})
            continue
        try:
            query = extract_query(record)
            gold = extract_gold_tools(record, agent_catalog)
            if not gold:
                raise ValueError("empty gold tool bundle in a PartII example")
            examples.append(
                Example(line_no, extract_qid(record, line_no), query, tuple(gold))
            )
        except ValueError as exc:
            issues.append({"line_no": line_no, "error": str(exc)})
            if FAIL_ON_MALFORMED_ROW:
                raise
    return examples, issues


def query_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def sample_examples_by_unique_query(examples: Sequence[Example]) -> list[Example]:
    """Exactly match the CF script's deterministic unique-query sampling."""
    if SAMPLE_UNIQUE_QUERIES <= 0:
        return list(examples)
    unique_keys = sorted({query_key(example.query) for example in examples})
    if len(unique_keys) <= SAMPLE_UNIQUE_QUERIES:
        return list(examples)
    selected = set(random.Random(RANDOM_SEED).sample(unique_keys, SAMPLE_UNIQUE_QUERIES))
    return [example for example in examples if query_key(example.query) in selected]


def request_payload(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "targets": ["tool"],
        "rewrite": REWRITE_QUERY,
        "agg": AGGREGATION,
        "top_k": {"tool": SEMANTIC_TOOL_TOPK},
        "tool_recall_mode": True,
        "tool_per_subquery_top_k": SEMANTIC_TOOL_TOPK,
        "tool_final_top_k": MAX_SEMANTIC_TOOL,
        "include_raw_tool": False,
        "include_doc_text": False,
    }


def post_json(url: str, payload: Mapping[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def retrieve_one(query: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = post_json(SEMANTIC_RETR_URL, request_payload(query))
            if not isinstance(response, Mapping) or response.get("ok") is False:
                raise ValueError(
                    'retrieval returned an unsuccessful response: '
                    + json.dumps(response, ensure_ascii=False)[:1000]
                )
            # Validate exact Flask response structure before caching it.
            tool_group = response.get("results_by_type", {}).get("tool", {})
            if not isinstance(tool_group, Mapping) or not isinstance(tool_group.get("results"), list):
                raise ValueError(
                    "missing results_by_type.tool.results: "
                    + json.dumps(response, ensure_ascii=False)[:1000]
                )
            return response
        except HTTPError as exc:
            last_error = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}")
            if 400 <= exc.code < 500:
                break
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        if attempt + 1 < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"semantic tool retrieval failed after {MAX_RETRIES} attempts: {last_error}")


def parse_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_retrieved_tool(item: Any) -> RetrievedTool | None:
    if isinstance(item, str):
        tool = normalize_tool(item)
        return RetrievedTool(tool, None, None, None, None) if tool else None
    if not isinstance(item, Mapping):
        return None

    value: Any = None
    for key in TOOL_VALUE_KEYS:
        if key in item and item[key] not in (None, ""):
            value = item[key]
            break
    if isinstance(value, Mapping):
        for key in TOOL_VALUE_KEYS:
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
    tools = parse_tool_value(value)
    if not tools:
        # Backend records sometimes put the canonical token in metadata.
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            for key in TOOL_VALUE_KEYS:
                if key in metadata and metadata[key] not in (None, ""):
                    tools = parse_tool_value(metadata[key])
                    if tools:
                        break
    if not tools:
        return None

    raw_score = parse_float(
        first_nonempty(item, ("raw_score", "raw", "score", "similarity"))
    )
    normalized_score = parse_float(
        first_nonempty(item, ("norm_score", "normalized_score", "score_normalized"))
    )
    matched_subquery = first_nonempty(
        item, ("matched_subquery", "best_subquery", "subquery", "matched_subqueries")
    )
    item_id_value = first_nonempty(item, ("item_id", "tool_id", "record_id", "id"))
    return RetrievedTool(
        tools[0], raw_score, normalized_score, matched_subquery,
        None if item_id_value is None else str(item_id_value),
    )


def parse_ranked_tools(response: Mapping[str, Any]) -> tuple[list[RetrievedTool], int]:
    items = response.get("results_by_type", {}).get("tool", {}).get("results", [])
    parsed = [tool for item in items if (tool := parse_retrieved_tool(item)) is not None]
    # The Flask server already sorts descending by raw_score. Preserve that
    # order exactly and only discard later occurrences of duplicate tool IDs.
    seen: set[str] = set()
    unique: list[RetrievedTool] = []
    for tool in parsed:
        if tool.tool not in seen:
            seen.add(tool.tool)
            unique.append(tool)
    return unique[:MAX_SEMANTIC_TOOL], len(parsed) - len(unique)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not RESUME or not path.exists():
        return cache
    for _, row in iter_jsonl(path):
        if isinstance(row, Mapping) and row.get("query_sha256") and "response" in row:
            cache[str(row["query_sha256"])] = dict(row)
    return cache


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def retrieve_all(
    examples: Sequence[Example], cache_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = load_cache(cache_path)
    unique_queries = {query_key(example.query): example.query for example in examples}
    pending = [(key, query) for key, query in unique_queries.items() if key not in cache]
    errors: list[dict[str, Any]] = []
    print(
        f"Retrieval: unique_queries={len(unique_queries):,}, "
        f"cached={len(unique_queries)-len(pending):,}, pending={len(pending):,}, "
        f"per_subquery_topk={SEMANTIC_TOOL_TOPK}, final_topk={MAX_SEMANTIC_TOOL}, "
        f"workers={WORKERS}",
        flush=True,
    )

    def task(key: str, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = retrieve_one(query)
        ranked, _ = parse_ranked_tools(response)
        if not ranked:
            raise ValueError(
                "response has no parseable tools: "
                + json.dumps(response, ensure_ascii=False)[:1000]
            )
        return {
            "query_sha256": key,
            "query": query,
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "request": request_payload(query),
            "response": response,
        }

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            futures = {pool.submit(task, key, query): (key, query) for key, query in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                key, query = futures[future]
                try:
                    row = future.result()
                    cache[key] = row
                    append_jsonl(cache_path, [row])
                except Exception as exc:
                    errors.append({"query_sha256": key, "query": query, "error": repr(exc)})
                if completed % 50 == 0 or completed == len(pending):
                    print(
                        f"  completed {completed:,}/{len(pending):,}; errors={len(errors):,}",
                        flush=True,
                    )
    return {key: row["response"] for key, row in cache.items()}, errors


def ranked_metrics(gold: set[str], ranked: Sequence[str]) -> dict[str, float]:
    hits = [1 if tool in gold else 0 for tool in ranked]
    hit_count = sum(hits)
    precision = safe_div(hit_count, len(ranked))
    recall = safe_div(hit_count, len(gold))
    first_hit = next((rank for rank, hit in enumerate(hits, 1) if hit), None)

    running_hits = 0
    ap_sum = 0.0
    dcg = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            running_hits += 1
            ap_sum += running_hits / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), len(ranked))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    r_cut = min(len(gold), len(ranked))

    return {
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": safe_div(2 * precision * recall, precision + recall),
        "tool_hit": float(hit_count > 0),
        "tool_all_covered": float(hit_count == len(gold)),
        "tool_mrr": safe_div(1.0, first_hit) if first_hit else 0.0,
        "tool_ap": safe_div(ap_sum, len(gold)),
        "tool_ndcg": safe_div(dcg, idcg),
        "tool_r_precision": safe_div(sum(hits[:r_cut]), len(gold)),
        "retrieved_unique_tools": float(len(ranked)),
        "tool_hits": float(hit_count),
    }


def evaluate_example(
    example: Example, ranked_tools: Sequence[RetrievedTool], duplicate_count: int, k: int,
) -> dict[str, Any]:
    selected = list(ranked_tools[:k])
    metrics = ranked_metrics(set(example.gold_tools), [item.tool for item in selected])
    return {
        "qid": example.qid,
        "line_no": example.line_no,
        "query_sha256": query_key(example.query),
        "k_tools": k,
        "gold_tools": list(example.gold_tools),
        "gold_tool_count": len(set(example.gold_tools)),
        "retrieved_tools": [
            {
                "rank": rank,
                "tool": item.tool,
                "raw_score": item.raw_score,
                "normalized_score": item.normalized_score,
                "matched_subquery": item.matched_subquery,
                "item_id": item.item_id,
            }
            for rank, item in enumerate(selected, 1)
        ],
        "duplicates_removed_from_full_response": duplicate_count,
        "metrics": metrics,
    }


def aggregate(rows: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    metric_names = list(rows[0]["metrics"]) if rows else []
    result: dict[str, Any] = {"k_tools": k, "examples": len(rows)}
    for name in metric_names:
        result[name] = mean([float(row["metrics"][name]) for row in rows])

    gold_total = sum(int(row["gold_tool_count"]) for row in rows)
    hit_total = sum(int(row["metrics"]["tool_hits"]) for row in rows)
    retrieved_total = sum(int(row["metrics"]["retrieved_unique_tools"]) for row in rows)
    micro_precision = safe_div(hit_total, retrieved_total)
    micro_recall = safe_div(hit_total, gold_total)
    result.update({
        "tool_micro_precision": micro_precision,
        "tool_micro_recall": micro_recall,
        "tool_micro_f1": safe_div(
            2 * micro_precision * micro_recall, micro_precision + micro_recall
        ),
        "gold_tools_total": gold_total,
        "tool_hits_total": hit_total,
        "retrieved_unique_tools_total": retrieved_total,
    })
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    return f"{100 * float(value):.2f}%"


def write_summary(
    path: Path, aggregates: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any],
) -> None:
    lines = [
        "# Capability-Aware Tool Retrieval Evaluation",
        "",
        f"- Input: `{metadata['input']}`",
        f"- Evaluated positive rows: {metadata['evaluated_examples']:,}",
        f"- Sampled unique queries: {metadata['unique_queries']:,}",
        f"- Sampling seed: {RANDOM_SEED}",
        f"- Endpoint: `{SEMANTIC_RETR_URL}`",
        f"- Query rewrite: `{REWRITE_QUERY}`; aggregation: `{AGGREGATION}`",
        f"- Per-subquery tool K: {SEMANTIC_TOOL_TOPK}; final tool K: {MAX_SEMANTIC_TOOL}",
        "",
        "## Ranked tool retrieval",
        "",
        "| K tools | Mean #returned | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 | Hit | All covered | MRR | MAP | nDCG | R-Prec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['k_tools']} | {row['retrieved_unique_tools']:.2f} | "
            f"{pct(row['tool_precision'])} | {pct(row['tool_recall'])} | "
            f"{pct(row['tool_f1'])} | {pct(row['tool_micro_precision'])} | "
            f"{pct(row['tool_micro_recall'])} | {pct(row['tool_micro_f1'])} | "
            f"{pct(row['tool_hit'])} | {pct(row['tool_all_covered'])} | "
            f"{row['tool_mrr']:.4f} | {row['tool_ap']:.4f} | "
            f"{row['tool_ndcg']:.4f} | {row['tool_r_precision']:.4f} |"
        )
    lines.extend([
        "",
        "## Evaluation protocol",
        "",
        "- The same SHA-256-based random sample and seed as the CF bundle experiment are used; all positive configurations belonging to a sampled query are retained.",
        "- The endpoint rewrites each query into tool-oriented subqueries. Each subquery retrieves up to 20 tools in recall mode, and the server globally sorts candidates by raw score.",
        "- The evaluator preserves that server order and only removes repeated tool identifiers, retaining their first occurrence.",
        "- AP divides by the total number of gold tools, so unretrieved gold tools are penalized. nDCG uses binary tool relevance. R-Precision evaluates the first |gold| ranked positions available at each cutoff.",
        "- Macro metrics average examples; micro metrics pool gold, retrieved, and hit tool counts.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}\nRun this script on the machine where "
            + str(AC_ROOT / ' is mounted, or edit INPUT_PATH in CONFIG.')
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / "retrieval_cache.jsonl"

    agent_catalog = load_agent_catalog(AGENT_CATALOG_PATH)
    all_examples, validation_issues = load_examples(INPUT_PATH, agent_catalog)
    if not all_examples:
        raise RuntimeError("No valid PartII examples were loaded")
    examples = sample_examples_by_unique_query(all_examples)
    unique_query_count = len({query_key(example.query) for example in examples})
    print(
        f"Loaded {len(all_examples):,} valid rows; randomly sampled "
        f"{unique_query_count:,} unique queries / {len(examples):,} evaluation rows; "
        f"seed={RANDOM_SEED}; validation issues={len(validation_issues):,}"
    )
    if validation_issues:
        append_jsonl(OUTPUT_DIR / "validation_issues.jsonl", validation_issues)

    responses, retrieval_errors = retrieve_all(examples, cache_path)
    if retrieval_errors:
        append_jsonl(OUTPUT_DIR / "retrieval_errors.jsonl", retrieval_errors)

    rows_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    subquery_counts: list[int] = []
    for example in examples:
        response = responses.get(query_key(example.query))
        if response is None:
            continue
        ranked_tools, duplicate_count = parse_ranked_tools(response)
        if not ranked_tools:
            parse_errors.append({
                "qid": example.qid, "line_no": example.line_no,
                "error": "no parseable ranked tools",
            })
            continue
        rewrite = response.get("rewrite", {})
        if isinstance(rewrite, Mapping) and isinstance(rewrite.get("tool_subqueries"), list):
            subquery_counts.append(len(rewrite["tool_subqueries"]))
        for k in EVAL_TOP_KS:
            row = evaluate_example(example, ranked_tools, duplicate_count, k)
            rows_by_k[k].append(row)
            all_rows.append(row)

    if parse_errors:
        append_jsonl(OUTPUT_DIR / "parse_errors.jsonl", parse_errors)
    if not all_rows:
        raise RuntimeError("No examples could be evaluated; inspect retrieval_errors.jsonl")

    with (OUTPUT_DIR / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    aggregates = [aggregate(rows_by_k[k], k) for k in EVAL_TOP_KS]
    metadata = {
        "created_at_utc": utc_now(),
        "input": str(INPUT_PATH),
        "agent_catalog": str(AGENT_CATALOG_PATH),
        "agent_catalog_size": len(agent_catalog),
        "output_dir": str(OUTPUT_DIR),
        "endpoint": SEMANTIC_RETR_URL,
        "request_template": request_payload("<QUERY>"),
        "evaluation_top_ks": list(EVAL_TOP_KS),
        "semantic_tool_topk_per_subquery": SEMANTIC_TOOL_TOPK,
        "max_semantic_tool": MAX_SEMANTIC_TOOL,
        "rewrite": REWRITE_QUERY,
        "aggregation": AGGREGATION,
        "all_valid_examples_before_sampling": len(all_examples),
        "sample_unique_queries_requested": SAMPLE_UNIQUE_QUERIES,
        "random_seed": RANDOM_SEED,
        "unique_queries": unique_query_count,
        "sampled_positive_rows": len(examples),
        "evaluated_examples": len(rows_by_k[EVAL_TOP_KS[0]]),
        "mean_tool_subqueries": mean([float(value) for value in subquery_counts]),
        "validation_issue_count": len(validation_issues),
        "retrieval_error_count": len(retrieval_errors),
        "parse_error_count": len(parse_errors),
        "case_sensitive_tool_ids": CASE_SENSITIVE_TOOL_IDS,
        "workers": WORKERS,
        "resume": RESUME,
    }
    (OUTPUT_DIR / "aggregate_metrics.json").write_text(
        json.dumps({"metadata": metadata, "metrics_by_k": aggregates}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregates)
    write_summary(OUTPUT_DIR / "summary.md", aggregates, metadata)

    print(f"Done. Evaluated examples: {metadata['evaluated_examples']:,}/{len(examples):,}")
    print(f"Mean tool subqueries/query: {metadata['mean_tool_subqueries']:.2f}")
    print(f"Summary: {OUTPUT_DIR / 'summary.md'}")
    print(f"CSV:     {OUTPUT_DIR / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
