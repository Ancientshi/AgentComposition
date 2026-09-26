#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Evaluate the Part-I CF LLM retrieval on ranked multi-positive labels.\n\nRun directly (all experiment settings are defined in ``CONFIG`` below):\n\n    python evaluate_cf_llm_retrieval.py\n\nThe evaluator sends one request per unique qid at the maximum cutoff and reuses\nthe returned ranking for every smaller cutoff.  Gold LLMs are identified by the\n``agent_id`` field; their reference order is given by the input ``topk`` field.\nNo tool-related metric is computed.\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CONFIG = {
    "input_path": str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/PartI_pairs.jsonl'),
    "cf_llm_retr_url": "http://127.0.0.1:9000/predict",
    "output_dir": str(AC_ROOT / 'data_preparation/rag_synthesis/scripts/cf_llm_retrieval_eval'),
    "top_ks": (5, 10, 15, 20, 50),
    "request_timeout_seconds": 120,
    "max_retries": 3,
    "retry_backoff_seconds": 2.0,
    "resume": True,
    # Camera-ready default: never silently report metrics on a successful subset.
    "fail_on_retrieval_error": True,
    # 0 means all unique queries. This is useful only for a quick smoke test.
    "max_unique_queries": 0,
}


@dataclass(frozen=True)
class QueryExample:
    qid: str
    query: str
    gold_ids: Tuple[str, ...]  # ordered by the reference topk/rank
    gold_ranks: Tuple[int, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return statistics.fmean(xs) if xs else 0.0


def safe_f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def atomic_json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            yield line_no, obj


def load_examples(path: Path, limit: int = 0) -> Tuple[List[QueryExample], Dict[str, Any]]:
    """Group rows by qid and validate their query/rank annotations."""
    grouped: Dict[str, Dict[str, Any]] = {}
    duplicate_rows = 0
    total_rows = 0

    for line_no, row in read_jsonl(path):
        total_rows += 1
        qid = str(row.get("qid", "")).strip()
        query = str(row.get("query", "")).strip()
        agent_id = str(row.get("agent_id", "")).strip()
        try:
            gold_rank = int(row.get("topk"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid topk at {path}:{line_no}") from exc
        if not qid or not query or not agent_id or gold_rank < 1:
            raise ValueError(
                f"Required qid/query/agent_id/topk is empty or invalid at {path}:{line_no}"
            )

        entry = grouped.setdefault(qid, {"query": query, "rank_to_id": {}, "id_to_rank": {}})
        if entry["query"] != query:
            raise ValueError(f"Conflicting query text for qid={qid!r} at line {line_no}")
        previous_id = entry["rank_to_id"].get(gold_rank)
        previous_rank = entry["id_to_rank"].get(agent_id)
        if previous_id is not None and previous_id != agent_id:
            raise ValueError(f"Two agent_ids use gold rank {gold_rank} for qid={qid!r}")
        if previous_rank is not None and previous_rank != gold_rank:
            raise ValueError(f"agent_id={agent_id!r} has two gold ranks for qid={qid!r}")
        if previous_id == agent_id:
            duplicate_rows += 1
        entry["rank_to_id"][gold_rank] = agent_id
        entry["id_to_rank"][agent_id] = gold_rank

    examples: List[QueryExample] = []
    non_contiguous_rank_queries = 0
    for qid, entry in grouped.items():
        ordered = sorted(entry["rank_to_id"].items())
        ranks = tuple(rank for rank, _ in ordered)
        if ranks != tuple(range(1, len(ranks) + 1)):
            non_contiguous_rank_queries += 1
        examples.append(
            QueryExample(
                qid=qid,
                query=entry["query"],
                gold_ids=tuple(agent_id for _, agent_id in ordered),
                gold_ranks=ranks,
            )
        )

    if limit > 0:
        examples = examples[:limit]
    stats = {
        "input_rows": total_rows,
        "unique_queries_before_limit": len(grouped),
        "evaluated_queries_requested": len(examples),
        "duplicate_rows": duplicate_rows,
        "non_contiguous_gold_rank_queries": non_contiguous_rank_queries,
        "mean_gold_llms_per_query": mean(len(x.gold_ids) for x in examples),
        "min_gold_llms_per_query": min((len(x.gold_ids) for x in examples), default=0),
        "max_gold_llms_per_query": max((len(x.gold_ids) for x in examples), default=0),
    }
    return examples, stats


def extract_ranked_ids(payload: Mapping[str, Any]) -> Tuple[List[str], int]:
    """Parse the strict Flask response and stably remove duplicate item ids."""
    if payload.get("ok") is False:
        raise RuntimeError(f"Retriever returned ok=false: {payload.get('error', payload)}")
    items = payload.get("topk")
    if not isinstance(items, list):
        raise ValueError("Retriever response does not contain a list-valued 'topk' field")

    ids: List[str] = []
    seen = set()
    duplicates = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"topk[{index - 1}] is not an object")
        item_id = str(item.get("llm_item_id", "")).strip()
        if not item_id:
            raise ValueError(f"topk[{index - 1}] has no llm_item_id")
        if item_id in seen:
            duplicates += 1
            continue
        seen.add(item_id)
        ids.append(item_id)
    return ids, duplicates


def request_retrieval(query: str, topk: int) -> Tuple[List[str], int]:
    body = json.dumps({"query": query, "topk": topk}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        str(CONFIG["cf_llm_retr_url"]),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[BaseException] = None
    for attempt in range(1, int(CONFIG["max_retries"]) + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=float(CONFIG["request_timeout_seconds"])
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("Retriever response is not a JSON object")
            return extract_ranked_ids(payload)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < int(CONFIG["max_retries"]):
                time.sleep(float(CONFIG["retry_backoff_seconds"]) * attempt)
    raise RuntimeError(f"Retrieval failed after {CONFIG['max_retries']} attempts: {last_error}")


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists() or not bool(CONFIG["resume"]):
        return cache
    for _, obj in read_jsonl(path):
        qid = str(obj.get("qid", ""))
        if qid and obj.get("status") == "ok":
            cache[qid] = obj
    return cache


def retrieve_all(examples: Sequence[QueryExample], output_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    cache_path = output_dir / "retrieval_cache.jsonl"
    cache = load_cache(cache_path)
    rankings: Dict[str, List[str]] = {}
    cache_hits = 0
    duplicate_returned_ids = 0
    max_k = max(int(k) for k in CONFIG["top_ks"])

    with cache_path.open("a", encoding="utf-8") as cache_file:
        for index, example in enumerate(examples, start=1):
            cached = cache.get(example.qid)
            if (
                cached
                and cached.get("query") == example.query
                and int(cached.get("requested_topk", 0)) >= max_k
                and isinstance(cached.get("retrieved_ids"), list)
            ):
                rankings[example.qid] = [str(x) for x in cached["retrieved_ids"]]
                cache_hits += 1
            else:
                try:
                    ids, duplicate_count = request_retrieval(example.query, max_k)
                    duplicate_returned_ids += duplicate_count
                    record = {
                        "status": "ok",
                        "created_at_utc": utc_now(),
                        "qid": example.qid,
                        "query": example.query,
                        "requested_topk": max_k,
                        "retrieved_ids": ids,
                        "duplicate_ids_removed": duplicate_count,
                    }
                    rankings[example.qid] = ids
                except Exception as exc:
                    record = {
                        "status": "error",
                        "created_at_utc": utc_now(),
                        "qid": example.qid,
                        "query": example.query,
                        "requested_topk": max_k,
                        "error": str(exc),
                    }
                cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache_file.flush()

            if index == 1 or index % 100 == 0 or index == len(examples):
                print(f"[retrieve] {index}/{len(examples)} queries; successful={len(rankings)}")

    return rankings, {
        "successful_queries": len(rankings),
        "failed_queries": len(examples) - len(rankings),
        "cache_hits": cache_hits,
        "duplicate_returned_ids_removed": duplicate_returned_ids,
    }


def dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(rank + 1.0) for rank, rel in enumerate(relevances, start=1))


def per_query_at_k(example: QueryExample, retrieved: Sequence[str], k: int) -> Dict[str, Any]:
    pred = list(retrieved[:k])
    gold_set = set(example.gold_ids)
    gold_rank = dict(zip(example.gold_ids, example.gold_ranks))
    hit_flags = [1 if item_id in gold_set else 0 for item_id in pred]
    hit_count = sum(hit_flags)
    precision = hit_count / len(pred) if pred else 0.0
    recall = hit_count / len(gold_set)
    first_hit_rank = next((rank for rank, rel in enumerate(hit_flags, start=1) if rel), None)

    precision_sum = 0.0
    for rank, rel in enumerate(hit_flags, start=1):
        if rel:
            precision_sum += sum(hit_flags[:rank]) / rank
    # Standard truncated AP@K: denominator is min(number of gold items, K).
    ap = precision_sum / min(len(gold_set), k)

    binary_dcg = dcg(hit_flags)
    ideal_binary = dcg([1.0] * min(len(gold_set), k))
    binary_ndcg = binary_dcg / ideal_binary if ideal_binary else 0.0

    # Reference-rank gain makes retrieval of a higher-ranked gold LLM worth more.
    # Direct gain 1/log2(gold_rank+1) is used in the outer DCG discount.
    graded_rels = [1.0 / math.log2(gold_rank[x] + 1.0) if x in gold_rank else 0.0 for x in pred]
    ideal_graded = sorted(
        (1.0 / math.log2(rank + 1.0) for rank in example.gold_ranks), reverse=True
    )[:k]
    graded_idcg = dcg(ideal_graded)
    graded_ndcg = dcg(graded_rels) / graded_idcg if graded_idcg else 0.0

    return {
        "qid": example.qid,
        "k": k,
        "gold_count": len(gold_set),
        "gold_ids": list(example.gold_ids),
        "gold_ranks": list(example.gold_ranks),
        "returned_count": len(pred),
        "hit_count": hit_count,
        "precision": precision,
        "recall": recall,
        "f1": safe_f1(precision, recall),
        "hit": float(hit_count > 0),
        "all_covered": float(hit_count == len(gold_set)),
        "reciprocal_rank": 0.0 if first_hit_rank is None else 1.0 / first_hit_rank,
        "average_precision": ap,
        "binary_ndcg": binary_ndcg,
        "graded_ndcg": graded_ndcg,
        "first_hit_rank": first_hit_rank,
        "retrieved_ids": pred,
        "hit_ids": [x for x in pred if x in gold_set],
        "missed_gold_ids": [x for x in example.gold_ids if x not in set(pred)],
    }


def aggregate_at_k(rows: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    total_hits = sum(int(x["hit_count"]) for x in rows)
    total_returned = sum(int(x["returned_count"]) for x in rows)
    total_gold = sum(int(x["gold_count"]) for x in rows)
    micro_p = total_hits / total_returned if total_returned else 0.0
    micro_r = total_hits / total_gold if total_gold else 0.0
    return {
        "k": k,
        "queries": len(rows),
        "mean_returned": mean(float(x["returned_count"]) for x in rows),
        "macro_precision": mean(float(x["precision"]) for x in rows),
        "macro_recall": mean(float(x["recall"]) for x in rows),
        "macro_f1": mean(float(x["f1"]) for x in rows),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": safe_f1(micro_p, micro_r),
        "hit_rate": mean(float(x["hit"]) for x in rows),
        "all_covered_rate": mean(float(x["all_covered"]) for x in rows),
        "mrr": mean(float(x["reciprocal_rank"]) for x in rows),
        "map": mean(float(x["average_precision"]) for x in rows),
        "binary_ndcg": mean(float(x["binary_ndcg"]) for x in rows),
        "graded_ndcg": mean(float(x["graded_ndcg"]) for x in rows),
    }


def compute_r_precision(
    examples: Sequence[QueryExample], rankings: Mapping[str, Sequence[str]], max_k: int
) -> Dict[str, Any]:
    values: List[float] = []
    for example in examples:
        if example.qid not in rankings or len(example.gold_ids) > max_k:
            continue
        r = len(example.gold_ids)
        values.append(sum(x in set(example.gold_ids) for x in rankings[example.qid][:r]) / r)
    successful = sum(x.qid in rankings for x in examples)
    return {
        "r_precision": mean(values),
        "evaluable_queries": len(values),
        "evaluable_query_rate": len(values) / successful if successful else 0.0,
        "note": f"Computed only where the full R cutoff is observable within retrieval depth {max_k}.",
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def write_csv(path: Path, aggregates: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)


def write_markdown(
    path: Path,
    metadata: Mapping[str, Any],
    aggregates: Sequence[Mapping[str, Any]],
    r_precision: Mapping[str, Any],
) -> None:
    lines = [
        "# CF LLM Retrieval Evaluation",
        "",
        f"- Input: `{metadata['input_path']}`",
        f"- Retriever: `{metadata['cf_llm_retr_url']}`",
        f"- Successfully evaluated unique queries: {metadata['retrieval']['successful_queries']:,}",
        f"- Mean gold LLMs per query: {metadata['data']['mean_gold_llms_per_query']:.2f}",
        "",
        "## Ranked retrieval",
        "",
        "| K | Mean returned | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 | Hit | All covered | MRR | MAP | nDCG | Graded nDCG |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in aggregates:
        lines.append(
            "| {k} | {mean_returned:.2f} | {mp} | {mr} | {mf} | {mip} | {mir} | {mif} | "
            "{hit} | {allc} | {mrr:.4f} | {map:.4f} | {ndcg:.4f} | {gndcg:.4f} |".format(
                k=x["k"], mean_returned=x["mean_returned"],
                mp=pct(x["macro_precision"]), mr=pct(x["macro_recall"]), mf=pct(x["macro_f1"]),
                mip=pct(x["micro_precision"]), mir=pct(x["micro_recall"]), mif=pct(x["micro_f1"]),
                hit=pct(x["hit_rate"]), allc=pct(x["all_covered_rate"]),
                mrr=x["mrr"], map=x["map"], ndcg=x["binary_ndcg"], gndcg=x["graded_ndcg"],
            )
        )
    lines += [
        "",
        "## R-Precision",
        "",
        f"R-Precision: **{r_precision['r_precision']:.4f}** over "
        f"{r_precision['evaluable_queries']:,} evaluable queries "
        f"({pct(r_precision['evaluable_query_rate'])} of successfully retrieved queries).",
        "",
        "## Metric protocol",
        "",
        "Each qid is one query and all of its distinct `agent_id` values are binary-relevant LLMs. "
        "The input `topk` field supplies their reference ordering. Precision uses the number actually "
        "returned at each cutoff; AP@K is normalized by `min(number of gold LLMs, K)`. Binary nDCG "
        "treats all gold LLMs equally. Graded nDCG additionally assigns reference-rank gain "
        "`1/log2(gold_rank+1)`. All-covered requires every gold LLM for the query to appear by K. "
        "Macro metrics average query-level values; micro metrics pool hits, returned items, and gold items.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = Path(str(CONFIG["input_path"]))
    output_dir = Path(str(CONFIG["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = tuple(sorted(set(int(k) for k in CONFIG["top_ks"])))
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not top_ks or top_ks[0] < 1:
        raise ValueError("CONFIG['top_ks'] must contain positive integers")

    examples, data_stats = load_examples(input_path, int(CONFIG["max_unique_queries"]))
    print(f"[data] rows={data_stats['input_rows']}; unique queries={len(examples)}")
    rankings, retrieval_stats = retrieve_all(examples, output_dir)
    if bool(CONFIG["fail_on_retrieval_error"]) and retrieval_stats["failed_queries"]:
        raise RuntimeError(
            f"{retrieval_stats['failed_queries']} retrieval requests failed. "
            "No aggregate metrics were produced; inspect retrieval_cache.jsonl and rerun."
        )
    successful_examples = [x for x in examples if x.qid in rankings]
    if not successful_examples:
        raise RuntimeError("No query was retrieved successfully; inspect retrieval_cache.jsonl")

    per_query_rows: List[Dict[str, Any]] = []
    aggregates: List[Dict[str, Any]] = []
    for k in top_ks:
        rows = [per_query_at_k(x, rankings[x.qid], k) for x in successful_examples]
        per_query_rows.extend(rows)
        aggregates.append(aggregate_at_k(rows, k))

    max_k = max(top_ks)
    r_precision = compute_r_precision(successful_examples, rankings, max_k)
    metadata = {
        "created_at_utc": utc_now(),
        "input_path": str(input_path),
        "cf_llm_retr_url": str(CONFIG["cf_llm_retr_url"]),
        "output_dir": str(output_dir),
        "top_ks": list(top_ks),
        "config": dict(CONFIG),
        "data": data_stats,
        "retrieval": retrieval_stats,
        "r_precision": r_precision,
    }

    with (output_dir / "per_query_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json_dump(output_dir / "aggregate_metrics.json", aggregates)
    atomic_json_dump(output_dir / "metadata.json", metadata)
    write_csv(output_dir / "aggregate_metrics.csv", aggregates)
    write_markdown(output_dir / "summary.md", metadata, aggregates, r_precision)

    print(f"[done] Results written to: {output_dir}")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
