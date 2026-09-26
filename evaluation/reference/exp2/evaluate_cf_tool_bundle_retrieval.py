#!/usr/bin/env python3
'Evaluate the CF tool-bundle retrieval on PartII_pairs.jsonl.\n\nThis is intentionally a self-contained experiment script: edit the constants in\nthe CONFIG section if paths or the local endpoint change, then run\n\n    python evaluate_cf_tool_bundle_retrieval.py\n\nThe retrieval is called once per unique query at MAX(TOP_KS).  Results are\nevaluated at K={5,10,15,20} in two complementary ways:\n\n1. Bundle level: each retrieved bundle remains one ranked item.  We report\n   exact-gold hits, single-bundle full coverage, partial overlap, best bundle\n   similarity, and whether full coverage is achieved only by merging bundles.\n2. Tool level: tools in the first K bundles are flattened in bundle/tool order\n   and stably deduplicated.  We report standard ranked-retrieval metrics.\n\nOutputs:\n  summary.md             human-readable tables and metric definitions\n  aggregate_metrics.csv  one row per K\n  aggregate_metrics.json full metrics and run metadata\n  per_example.jsonl      query/gold/retrieval details for error analysis\n  retrieval_cache.jsonl  resumable raw API responses\n\nOnly the Python standard library is required.\n'

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
# CONFIG: all experiment parameters are internalized here; no shell is needed.
# =============================================================================

INPUT_PATH = Path(
    str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/PartII_pairs.jsonl')
)
AGENT_CATALOG_PATH = Path(
    str(AC_ROOT / 'datasets/PartII/agents/merge.json')
)
OUTPUT_DIR = INPUT_PATH.parent / "cf_tool_bundle_retrieval_eval"
CF_TOOL_RETR_URL = "http://127.0.0.1:9001/predict"
TOP_KS = (5, 10, 15, 20)

REQUEST_TIMEOUT_SEC = 120
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5
WORKERS = 8                    # Set to 1 if the local service is not concurrent.
RESUME = True
SAMPLE_UNIQUE_QUERIES = 1000   # 0 means evaluate every unique query.
RANDOM_SEED = 42               # Makes query sampling exactly reproducible.
CASE_SENSITIVE_TOOL_IDS = True
FAIL_ON_MALFORMED_ROW = False

# Most versions of the CF service accept {"query": ..., "topk": ...}.
# On HTTP 400/422 the client automatically tries the other common spellings.
REQUEST_PAYLOAD_STYLES = (
    ("query", "topk"),
    ("query", "top_k"),
    ("query", "k"),
    ("question", "topk"),
    ("question", "top_k"),
    ("text", "topk"),
    ("text", "top_k"),
    ("input", "topk"),
    ("input", "top_k"),
)


TOOL_TOKEN_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")
CITATION_RE = re.compile(r"\s*\[\d+\]\s*$")
WS_RE = re.compile(r"\s+")
RESULT_CONTAINER_KEYS = (
    "results", "predictions", "recommendations", "retrieval_results",
    "retrieved_tool_bundles", "tool_bundles", "bundles", "data", "items",
    "top_agents", "agents", "candidates", "ranked_results", "matches",
    "predicted_agents", "recommended_items",
)
TOOL_FIELD_KEYS = (
    "tools", "tool_list", "tool_names", "tool_tokens", "tool_bundle",
    "bundle", "toolkit", "selected_tools",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(n: float | int, d: float | int) -> float:
    return float(n / d) if d else 0.0


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_tool(value: Any) -> str:
    """Normalize wrappers/whitespace while preserving the tool identifier."""
    if value is None:
        return ""
    text = str(value).strip()
    token_match = TOOL_TOKEN_RE.fullmatch(text)
    if token_match:
        text = token_match.group(1)
    text = CITATION_RE.sub("", text).strip().strip("'").strip('"').strip()
    text = WS_RE.sub(" ", text)
    return text if CASE_SENSITIVE_TOOL_IDS else text.casefold()


def parse_tool_value(value: Any) -> list[str]:
    """Parse a list/string/nested agent object into ordered tool identifiers."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # Agent schema used by PartII: {"M": ..., "T": {"tools": [...]}, ...}
        if "T" in value and isinstance(value["T"], Mapping):
            nested = parse_tool_value(value["T"].get("tools"))
            if nested or "tools" in value["T"]:
                return nested
        for key in TOOL_FIELD_KEYS:
            if key in value:
                return parse_tool_value(value[key])
        for key in ("agent", "configuration", "config", "target_agent"):
            if key in value:
                parsed = parse_tool_value(value[key])
                if parsed:
                    return parsed
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parsed = parse_tool_value(item)
                if parsed:
                    out.extend(parsed)
                else:
                    for key in ("name", "tool_name", "token", "id"):
                        if key in item:
                            out.extend(parse_tool_value(item[key]))
                            break
            else:
                out.extend(parse_tool_value(item))
        return stable_unique(out)

    text = str(value).strip()
    if not text:
        return []
    tokens = [normalize_tool(x) for x in TOOL_TOKEN_RE.findall(text)]
    if tokens:
        return stable_unique(tokens)
    if text[:1] in "[{" and text[-1:] in "]}":
        try:
            return parse_tool_value(json.loads(text))
        except json.JSONDecodeError:
            pass
    # A plain string is treated as one tool.  We deliberately do not split on
    # commas because commas may legally occur in a tool identifier.
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
        # Support chat-style messages without adding role markup not seen by the
        # original retriever.
        parts = []
        for item in value:
            if isinstance(item, Mapping) and item.get("content"):
                parts.append(str(item["content"]))
        value = "\n".join(parts)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cannot find a non-empty query/question/input field")
    return value.strip()


def extract_gold_tools(
    record: Mapping[str, Any], agent_catalog: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """Extract gold only from labels/configurations, never from query text."""
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
        "cannot find gold tools; expected gold_tools/tools, an agent T.tools, "
        "or <<tool>> tokens in target/label/output"
    )


def extract_qid(record: Mapping[str, Any], line_no: int) -> str:
    value = first_nonempty(
        record, ("qid", "question_id", "query_id", "id", "pair_id", "agent_id")
    )
    return str(value) if value is not None else f"line_{line_no}"


@dataclass(frozen=True)
class Example:
    line_no: int
    qid: str
    query: str
    gold_tools: tuple[str, ...]


@dataclass(frozen=True)
class Bundle:
    tools: tuple[str, ...]
    score: float | None = None
    item_id: str | None = None


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
            f"WARNING: agent catalog not found: {path}; ID-only API responses cannot be expanded",
            file=sys.stderr,
        )
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Agent catalog must be a JSON object: {path}")
    catalog: dict[str, tuple[str, ...]] = {}
    for agent_id, agent in raw.items():
        tools = parse_tool_value(agent)
        # PartII is expected to be non-empty, but retaining empty entries makes
        # lookup behavior faithful to the catalog and easier to validate.
        catalog[str(agent_id)] = tuple(tools)
    return catalog


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
            examples.append(Example(line_no, extract_qid(record, line_no), query, tuple(gold)))
        except ValueError as exc:
            issues.append({"line_no": line_no, "error": str(exc)})
            if FAIL_ON_MALFORMED_ROW:
                raise
    return examples, issues


def sample_examples_by_unique_query(examples: Sequence[Example]) -> list[Example]:
    """Randomly select query identities and retain all their positive rows."""
    if SAMPLE_UNIQUE_QUERIES <= 0:
        return list(examples)
    unique_keys = sorted({query_key(ex.query) for ex in examples})
    if len(unique_keys) <= SAMPLE_UNIQUE_QUERIES:
        return list(examples)
    rng = random.Random(RANDOM_SEED)
    selected = set(rng.sample(unique_keys, SAMPLE_UNIQUE_QUERIES))
    return [ex for ex in examples if query_key(ex.query) in selected]


def find_result_list(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    if not isinstance(response, Mapping):
        return []
    for key in RESULT_CONTAINER_KEYS:
        value = response.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            nested = find_result_list(value)
            if nested:
                return nested
    # Some APIs return rank-keyed objects: {"1": {...}, "2": {...}}.
    numeric = [(int(k), v) for k, v in response.items() if str(k).isdigit()]
    if numeric:
        return [v for _, v in sorted(numeric)]
    return []


def parse_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_bundle_item(
    item: Any, agent_catalog: Mapping[str, tuple[str, ...]] | None = None,
) -> Bundle | None:
    catalog = agent_catalog or {}
    if isinstance(item, str):
        item_id = item.strip()
        if item_id in catalog:
            return Bundle(catalog[item_id], item_id=item_id) if catalog[item_id] else None
        if re.fullmatch(r"(?:PartII_)?agent[_-]?\d+", item_id, re.IGNORECASE):
            return None
        tools = parse_tool_value(item)
        return Bundle(tuple(tools)) if tools else None
    if isinstance(item, (list, tuple, set)):
        tools = parse_tool_value(item)
        return Bundle(tuple(tools)) if tools else None
    if not isinstance(item, Mapping):
        return None

    tools: list[str] = []
    for key in TOOL_FIELD_KEYS:
        if key in item:
            tools = parse_tool_value(item[key])
            if tools or item[key] == []:
                break
    if not tools:
        for key in ("agent", "matched_agent", "configuration", "config", "candidate"):
            if key in item:
                tools = parse_tool_value(item[key])
                if tools:
                    break
    if not tools:
        tools = parse_tool_value(item)
    score = parse_float(first_nonempty(item, ("score", "similarity", "pred_score", "value")))
    item_id_value = first_nonempty(
        item, ("bundle_id", "agent_id", "item_id", "id", "candidate_id")
    )
    if not tools and item_id_value is not None and str(item_id_value) in catalog:
        tools = list(catalog[str(item_id_value)])
    if not tools:
        return None
    return Bundle(tuple(stable_unique(tools)), score, None if item_id_value is None else str(item_id_value))


def parse_bundles(
    response: Any, agent_catalog: Mapping[str, tuple[str, ...]] | None = None,
) -> list[Bundle]:
    items = find_result_list(response)
    bundles = [
        bundle for item in items
        if (bundle := parse_bundle_item(item, agent_catalog)) is not None
    ]
    return bundles


def post_json(url: str, payload: Mapping[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def retrieve_one(
    query: str, topk: int, agent_catalog: Mapping[str, tuple[str, ...]],
) -> tuple[Any, str]:
    """Return raw response and the request spelling accepted by the service."""
    last_error: Exception | None = None
    for text_key, k_key in REQUEST_PAYLOAD_STYLES:
        payload = {text_key: query, k_key: topk}
        for attempt in range(MAX_RETRIES):
            try:
                response = post_json(CF_TOOL_RETR_URL, payload)
                if parse_bundles(response, agent_catalog):
                    return response, f"{text_key}+{k_key}"
                # A permissive endpoint can return HTTP 200 for an ignored
                # payload.  Try the next spelling instead of caching it.
                last_error = ValueError(
                    "successful response had no parseable non-empty bundles: "
                    + json.dumps(response, ensure_ascii=False)[:500]
                )
                break
            except HTTPError as exc:
                last_error = exc
                # Schema mismatch: immediately try the next payload spelling.
                if exc.code in (400, 404, 405, 415, 422):
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * (2**attempt))
    raise RuntimeError(f"retrieval request failed for all payload styles: {last_error}")


def query_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


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
    agent_catalog: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = load_cache(cache_path)
    unique_queries = {query_key(ex.query): ex.query for ex in examples}
    pending = [(key, query) for key, query in unique_queries.items() if key not in cache]
    errors: list[dict[str, Any]] = []
    max_k = max(TOP_KS)

    print(
        f"Retrieval: unique_queries={len(unique_queries):,}, cached={len(unique_queries)-len(pending):,}, "
        f"pending={len(pending):,}, topk={max_k}, workers={WORKERS}",
        flush=True,
    )

    def task(key: str, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        response, style = retrieve_one(query, max_k, agent_catalog)
        bundles = parse_bundles(response, agent_catalog)
        if not bundles:
            preview = json.dumps(response, ensure_ascii=False)[:1000]
            raise ValueError(f"response contains no parseable non-empty tool bundle: {preview}")
        return {
            "query_sha256": key,
            "query": query,
            "requested_topk": max_k,
            "request_style": style,
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "response": response,
        }

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            future_map = {pool.submit(task, key, query): (key, query) for key, query in pending}
            completed = 0
            for future in as_completed(future_map):
                key, query = future_map[future]
                try:
                    row = future.result()
                    cache[key] = row
                    append_jsonl(cache_path, [row])
                except Exception as exc:  # Preserve all other successful queries.
                    errors.append({"query_sha256": key, "query": query, "error": repr(exc)})
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    print(f"  completed {completed:,}/{len(pending):,}; errors={len(errors):,}", flush=True)
    return {key: row["response"] for key, row in cache.items()}, errors


def flatten_tools(bundles: Sequence[Bundle]) -> list[str]:
    return stable_unique(tool for bundle in bundles for tool in bundle.tools)


def rank_metrics(gold: set[str], ranked: Sequence[str]) -> dict[str, float]:
    hits = [1 if item in gold else 0 for item in ranked]
    hit_count = sum(hits)
    precision = safe_div(hit_count, len(ranked))
    recall = safe_div(hit_count, len(gold))
    first_hit = next((i + 1 for i, hit in enumerate(hits) if hit), None)

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
    r_precision = safe_div(sum(hits[:r_cut]), len(gold))

    return {
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": safe_div(2 * precision * recall, precision + recall),
        "tool_hit": float(hit_count > 0),
        "tool_all_covered": float(hit_count == len(gold)),
        "tool_mrr": safe_div(1.0, first_hit) if first_hit else 0.0,
        "tool_ap": safe_div(ap_sum, len(gold)),
        "tool_ndcg": safe_div(dcg, idcg),
        "tool_r_precision": r_precision,
        "retrieved_unique_tools": float(len(ranked)),
        "tool_hits": float(hit_count),
    }


def bundle_metrics(gold: set[str], bundles: Sequence[Bundle]) -> dict[str, float]:
    exact_ranks: list[int] = []
    cover_ranks: list[int] = []
    overlap_ranks: list[int] = []
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    jaccards: list[float] = []

    for rank, bundle in enumerate(bundles, 1):
        candidate = set(bundle.tools)
        hit_count = len(gold & candidate)
        p = safe_div(hit_count, len(candidate))
        r = safe_div(hit_count, len(gold))
        union_size = len(gold | candidate)
        recalls.append(r)
        precisions.append(p)
        f1s.append(safe_div(2 * p * r, p + r))
        jaccards.append(safe_div(hit_count, union_size))
        if candidate == gold:
            exact_ranks.append(rank)
        if gold.issubset(candidate):
            cover_ranks.append(rank)
        if hit_count:
            overlap_ranks.append(rank)

    merged = set(flatten_tools(bundles))
    merged_hits = len(gold & merged)
    first_exact = exact_ranks[0] if exact_ranks else None
    first_cover = cover_ranks[0] if cover_ranks else None
    first_overlap = overlap_ranks[0] if overlap_ranks else None
    merged_complete = merged_hits == len(gold)
    any_single_complete = bool(cover_ranks)
    return {
        "bundle_exact_hit": float(bool(exact_ranks)),
        "bundle_exact_mrr": safe_div(1.0, first_exact) if first_exact else 0.0,
        # With one gold bundle, AP@K equals reciprocal rank of its first exact hit.
        "bundle_exact_ap": safe_div(1.0, first_exact) if first_exact else 0.0,
        "bundle_complete_hit": float(any_single_complete),
        "bundle_complete_mrr": safe_div(1.0, first_cover) if first_cover else 0.0,
        "bundle_any_overlap_hit": float(bool(overlap_ranks)),
        "bundle_overlap_mrr": safe_div(1.0, first_overlap) if first_overlap else 0.0,
        "bundle_best_tool_recall": max(recalls, default=0.0),
        "bundle_best_tool_precision": max(precisions, default=0.0),
        "bundle_best_tool_f1": max(f1s, default=0.0),
        "bundle_best_jaccard": max(jaccards, default=0.0),
        "merged_bundle_tool_recall": safe_div(merged_hits, len(gold)),
        "merged_bundle_all_covered": float(merged_complete),
        "merge_only_complete": float(merged_complete and not any_single_complete),
        "retrieved_bundles": float(len(bundles)),
    }


def evaluate_example(ex: Example, all_bundles: Sequence[Bundle], k: int) -> dict[str, Any]:
    bundles = list(all_bundles[:k])
    ranked_tools = flatten_tools(bundles)
    gold = set(ex.gold_tools)
    metrics = {**bundle_metrics(gold, bundles), **rank_metrics(gold, ranked_tools)}
    return {
        "qid": ex.qid,
        "line_no": ex.line_no,
        "query_sha256": query_key(ex.query),
        "k_bundles": k,
        "gold_tools": list(ex.gold_tools),
        "gold_tool_count": len(gold),
        "retrieved_bundles": [
            {"rank": rank, "tools": list(bundle.tools), "score": bundle.score, "item_id": bundle.item_id}
            for rank, bundle in enumerate(bundles, 1)
        ],
        "merged_ranked_tools": ranked_tools,
        "metrics": metrics,
    }


def aggregate(rows: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    metric_names = list(rows[0]["metrics"]) if rows else []
    result: dict[str, Any] = {"k_bundles": k, "examples": len(rows)}
    for name in metric_names:
        result[name] = mean([float(row["metrics"][name]) for row in rows])

    gold_total = sum(int(row["gold_tool_count"]) for row in rows)
    hit_total = sum(int(row["metrics"]["tool_hits"]) for row in rows)
    retrieved_total = sum(int(row["metrics"]["retrieved_unique_tools"]) for row in rows)
    result["tool_micro_precision"] = safe_div(hit_total, retrieved_total)
    result["tool_micro_recall"] = safe_div(hit_total, gold_total)
    result["tool_micro_f1"] = safe_div(
        2 * result["tool_micro_precision"] * result["tool_micro_recall"],
        result["tool_micro_precision"] + result["tool_micro_recall"],
    )
    result["gold_tools_total"] = gold_total
    result["tool_hits_total"] = hit_total
    result["retrieved_unique_tools_total"] = retrieved_total
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


def write_summary(path: Path, aggregates: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    lines = [
        "# CF Tool-Bundle Retrieval Evaluation",
        "",
        f"- Input: `{metadata['input']}`",
        f"- Valid examples: {metadata['valid_examples']:,}",
        f"- Unique queries sent/cached: {metadata['unique_queries']:,}",
        f"- Endpoint: `{metadata['endpoint']}`",
        f"- Cutoffs count ranked bundles: {list(TOP_KS)}",
        "",
        "## Bundle-level results",
        "",
        "| K bundles | Exact Hit | Exact MRR | Single-bundle complete | Any overlap | Best recall | Best Jaccard | Merged complete | Merge-only complete |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['k_bundles']} | {pct(row['bundle_exact_hit'])} | "
            f"{row['bundle_exact_mrr']:.4f} | {pct(row['bundle_complete_hit'])} | "
            f"{pct(row['bundle_any_overlap_hit'])} | {pct(row['bundle_best_tool_recall'])} | "
            f"{row['bundle_best_jaccard']:.4f} | {pct(row['merged_bundle_all_covered'])} | "
            f"{pct(row['merge_only_complete'])} |"
        )
    lines.extend([
        "",
        "## Tool-level results after stable merge/deduplication",
        "",
        "| K bundles | Mean #tools | P | R | F1 | Micro P | Micro R | All covered | MRR | MAP | nDCG | R-Prec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in aggregates:
        lines.append(
            f"| {row['k_bundles']} | {row['retrieved_unique_tools']:.2f} | "
            f"{pct(row['tool_precision'])} | {pct(row['tool_recall'])} | {pct(row['tool_f1'])} | "
            f"{pct(row['tool_micro_precision'])} | {pct(row['tool_micro_recall'])} | "
            f"{pct(row['tool_all_covered'])} | {row['tool_mrr']:.4f} | "
            f"{row['tool_ap']:.4f} | {row['tool_ndcg']:.4f} | {row['tool_r_precision']:.4f} |"
        )
    lines.extend([
        "",
        "## Definitions",
        "",
        "- **Exact Hit@K**: at least one of the first K bundles equals the gold tool set.",
        "- **Single-bundle complete@K**: one retrieved bundle contains every gold tool; it may contain extra tools.",
        "- **Best recall/Jaccard@K**: the maximum gold-tool recall/Jaccard achieved by any single retrieved bundle.",
        "- **Merged complete@K**: the stable union of tools from the first K bundles contains every gold tool.",
        "- **Merge-only complete@K**: merged complete is true but no individual bundle contains all gold tools.",
        "- Tool ranking is produced by traversing bundles by rank and tools by their returned within-bundle order, keeping only the first occurrence of each tool.",
        "- Tool AP uses the total number of gold tools as its denominator, so missing gold tools are penalized. nDCG uses binary tool relevance and the ideal placement of all reachable gold tools.",
        "- Macro P/R/F1 average per example; micro P/R/F1 pool tool counts across examples.",
        "",
        "Exact bundle retrieval is the strict primary result. Merged-tool recall is a candidate-pool coverage result and should not be described as exact bundle retrieval.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}\n"
            "This script has the requested server path internalized. Run it on the machine "
            "where /root/yunxshi/NIPS2026 is mounted, or edit INPUT_PATH in CONFIG."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / "retrieval_cache.jsonl"

    agent_catalog = load_agent_catalog(AGENT_CATALOG_PATH)
    all_examples, validation_issues = load_examples(INPUT_PATH, agent_catalog)
    if not all_examples:
        raise RuntimeError("No valid examples were loaded")
    examples = sample_examples_by_unique_query(all_examples)
    print(
        f"Loaded {len(all_examples):,} valid rows; randomly sampled "
        f"{len({query_key(ex.query) for ex in examples}):,} unique queries / "
        f"{len(examples):,} evaluation rows; seed={RANDOM_SEED}; "
        f"validation issues={len(validation_issues):,}"
    )

    responses, retrieval_errors = retrieve_all(examples, cache_path, agent_catalog)
    if retrieval_errors:
        append_jsonl(OUTPUT_DIR / "retrieval_errors.jsonl", retrieval_errors)
    if validation_issues:
        append_jsonl(OUTPUT_DIR / "validation_issues.jsonl", validation_issues)

    rows_by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for ex in examples:
        response = responses.get(query_key(ex.query))
        if response is None:
            continue
        bundles = parse_bundles(response, agent_catalog)
        if not bundles:
            parse_errors.append({"qid": ex.qid, "line_no": ex.line_no, "error": "no parsed bundles"})
            continue
        for k in TOP_KS:
            row = evaluate_example(ex, bundles, k)
            rows_by_k[k].append(row)
            all_rows.append(row)

    if parse_errors:
        append_jsonl(OUTPUT_DIR / "parse_errors.jsonl", parse_errors)
    if not all_rows:
        raise RuntimeError("No examples could be evaluated; inspect retrieval_errors.jsonl")

    with (OUTPUT_DIR / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    aggregates = [aggregate(rows_by_k[k], k) for k in TOP_KS]
    metadata = {
        "created_at_utc": utc_now(),
        "input": str(INPUT_PATH),
        "agent_catalog": str(AGENT_CATALOG_PATH),
        "agent_catalog_size": len(agent_catalog),
        "output_dir": str(OUTPUT_DIR),
        "endpoint": CF_TOOL_RETR_URL,
        "top_ks": list(TOP_KS),
        "all_valid_examples_before_sampling": len(all_examples),
        "sample_unique_queries_requested": SAMPLE_UNIQUE_QUERIES,
        "random_seed": RANDOM_SEED,
        "valid_examples": len(examples),
        "evaluated_examples": len(rows_by_k[TOP_KS[0]]),
        "unique_queries": len({query_key(ex.query) for ex in examples}),
        "validation_issue_count": len(validation_issues),
        "retrieval_error_count": len(retrieval_errors),
        "parse_error_count": len(parse_errors),
        "case_sensitive_tool_ids": CASE_SENSITIVE_TOOL_IDS,
        "workers": WORKERS,
        "resume": RESUME,
    }
    payload = {"metadata": metadata, "metrics_by_k": aggregates}
    (OUTPUT_DIR / "aggregate_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregates)
    write_summary(OUTPUT_DIR / "summary.md", aggregates, metadata)

    print(f"Done. Evaluated examples: {metadata['evaluated_examples']:,}/{len(examples):,}")
    print(f"Summary: {OUTPUT_DIR / 'summary.md'}")
    print(f"CSV:     {OUTPUT_DIR / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()