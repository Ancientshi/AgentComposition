#!/usr/bin/env python3
'Evaluate the tool-level TF-IDF retrieval served on port 9002.\n\nAll paths and experiment parameters are fixed below. Run directly:\n\n    python evaluate_cf_tool_level_retrieval.py\n\nThe script uses exactly the same deterministic 1,000-unique-query sample\n(SHA-256 query identity, seed 42) and tool-ranking metrics as the semantic tool\nretrieval evaluation. It requests Top-50 once per query and evaluates prefixes\nat K={5,10,15,20,50}.\n\nOutputs under OUTPUT_DIR:\n  summary.md\n  aggregate_metrics.csv\n  aggregate_metrics.json\n  per_example.jsonl\n  retrieval_cache.jsonl\n  validation_issues.jsonl / retrieval_errors.jsonl (only when non-empty)\n\nOnly the Python standard library is required.\n'

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
# CONFIG: fixed experiment parameters; no shell arguments are required.
# =============================================================================

INPUT_PATH = Path(
    str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/PartII_pairs.jsonl')
)
AGENT_CATALOG_PATH = Path(
    str(AC_ROOT / 'datasets/PartII/agents/merge.json')
)
OUTPUT_DIR = INPUT_PATH.parent / "cf_tool_level_retrieval_eval"

RETRIEVAL_URL = "http://127.0.0.1:9002/predict"
EVAL_TOP_KS = (5, 10, 15, 20, 50)
RETRIEVAL_TOPK = 50

SAMPLE_UNIQUE_QUERIES = 1000
RANDOM_SEED = 42
WORKERS = 8
REQUEST_TIMEOUT_SEC = 120
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
RESUME = True
CASE_SENSITIVE_TOOL_IDS = True
FAIL_ON_MALFORMED_ROW = False


TOOL_TOKEN_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")
CITATION_RE = re.compile(r"\s*\[\d+\]\s*$")
WS_RE = re.compile(r"\s+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


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
    """Parse gold tools from lists, agent configurations, or target strings."""
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
                for key in ("tool_id", "tool_name", "name", "token", "id"):
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
    # Gold is read only from labels/configurations, never from query text.
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
        "a catalog-resolvable agent_id, or <<tool>> tokens in target"
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
    score: float | None
    service_rank: int | None
    description: str | None


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
    """Match the prior evaluators' deterministic unique-query sample exactly."""
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
        "topk": RETRIEVAL_TOPK,
        "include_meta": False,
        "include_indexed_text": False,
    }


def post_json(url: str, payload: Mapping[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_ranked_tools(response: Mapping[str, Any]) -> tuple[list[RetrievedTool], int]:
    items = response.get("results", [])
    if not isinstance(items, list):
        return [], 0
    parsed: list[RetrievedTool] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw_tool = first_nonempty(item, ("tool_id", "tool_name", "token", "name", "id"))
        tool = normalize_tool(raw_tool)
        if not tool:
            continue
        description = item.get("description")
        parsed.append(
            RetrievedTool(
                tool=tool,
                score=parse_float(item.get("score")),
                service_rank=parse_int(item.get("rank")),
                description=str(description) if description is not None else None,
            )
        )

    # Preserve the service order and retain only each tool's first occurrence.
    seen: set[str] = set()
    unique: list[RetrievedTool] = []
    for item in parsed:
        if item.tool not in seen:
            seen.add(item.tool)
            unique.append(item)
    return unique[:RETRIEVAL_TOPK], len(parsed) - len(unique)


def retrieve_one(query: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = post_json(RETRIEVAL_URL, request_payload(query))
            if not isinstance(response, Mapping) or response.get("ok") is not True:
                raise ValueError(
                    'retrieval returned an unsuccessful response: '
                    + json.dumps(response, ensure_ascii=False)[:1000]
                )
            if response.get("retrieval_level") != "tool":
                raise ValueError(f"unexpected retrieval_level: {response.get('retrieval_level')!r}")
            ranked, _ = parse_ranked_tools(response)
            if not ranked:
                raise ValueError(
                    "response contains no parseable tools: "
                    + json.dumps(response, ensure_ascii=False)[:1000]
                )
            return response
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:1000]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if 400 <= exc.code < 500:
                break
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        if attempt + 1 < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"tool-level retrieval failed after {MAX_RETRIES} attempts: {last_error}")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not RESUME or not path.exists():
        return cache
    for _, row in iter_jsonl(path):
        if isinstance(row, Mapping) and row.get("query_sha256") and "response" in row:
            # Reject cache entries created with a different requested depth.
            if int(row.get("retrieval_topk", -1)) == RETRIEVAL_TOPK:
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
        f"topk={RETRIEVAL_TOPK}, workers={WORKERS}",
        flush=True,
    )

    def make_row(key: str, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = retrieve_one(query)
        return {
            "query_sha256": key,
            "query": query,
            "retrieval_topk": RETRIEVAL_TOPK,
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "response": response,
        }

    # Synchronous preflight: fail clearly before launching many requests.
    if pending:
        key, query = pending[0]
        try:
            row = make_row(key, query)
        except Exception as exc:
            raise RuntimeError(
                "Preflight retrieval failed; no batch requests were sent. "
                f"query_sha256={key}, error={exc!r}"
            ) from exc
        cache[key] = row
        append_jsonl(cache_path, [row])
        pending = pending[1:]
        print("  preflight succeeded", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            futures = {pool.submit(make_row, key, query): (key, query) for key, query in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                key, query = futures[future]
                try:
                    row = future.result()
                    cache[key] = row
                    append_jsonl(cache_path, [row])
                except Exception as exc:
                    errors.append({"query_sha256": key, "query": query, "error": repr(exc)})
                if completed % 100 == 0 or completed == len(pending):
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
                "score": item.score,
                "service_rank": item.service_rank,
                "description": item.description,
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
        "# Tool-Level TF-IDF Retrieval Evaluation",
        "",
        f"- Input: `{metadata['input']}`",
        f"- Evaluated positive rows: {metadata['evaluated_examples']:,}",
        f"- Sampled unique queries: {metadata['unique_queries']:,}",
        f"- Sampling seed: {RANDOM_SEED}",
        f"- Endpoint: `{RETRIEVAL_URL}`",
        f"- Requested tool depth: {RETRIEVAL_TOPK}",
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
        "- The same SHA-256 query identity, random seed, and 1,000-query sample as the other retrieval experiments are used; all positive configurations for a sampled query are retained.",
        "- The service is queried once at Top-50. Evaluation at smaller K uses ranked prefixes without rescoring.",
        "- Returned tool identifiers are stably deduplicated while preserving the service order.",
        "- AP divides by the total number of gold tools, so missing gold tools are penalized. nDCG uses binary tool relevance. R-Precision evaluates the first |gold| available positions.",
        "- Macro metrics average examples; micro metrics pool gold, retrieved, and hit tool counts.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}\nRun on the machine where "
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
    duplicate_counts: list[int] = []
    for example in examples:
        response = responses.get(query_key(example.query))
        if response is None:
            continue
        ranked_tools, duplicate_count = parse_ranked_tools(response)
        if not ranked_tools:
            parse_errors.append({
                "qid": example.qid,
                "line_no": example.line_no,
                "error": "no parseable ranked tools",
            })
            continue
        duplicate_counts.append(duplicate_count)
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
        "endpoint": RETRIEVAL_URL,
        "request_template": request_payload("<QUERY>"),
        "evaluation_top_ks": list(EVAL_TOP_KS),
        "retrieval_topk": RETRIEVAL_TOPK,
        "all_valid_examples_before_sampling": len(all_examples),
        "sample_unique_queries_requested": SAMPLE_UNIQUE_QUERIES,
        "random_seed": RANDOM_SEED,
        "unique_queries": unique_query_count,
        "sampled_positive_rows": len(examples),
        "evaluated_examples": len(rows_by_k[EVAL_TOP_KS[0]]),
        "mean_duplicates_removed": mean([float(value) for value in duplicate_counts]),
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
    print(f"Summary: {OUTPUT_DIR / 'summary.md'}")
    print(f"CSV:     {OUTPUT_DIR / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
