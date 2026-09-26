#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tqdm import tqdm

from .context_renderer import build_compact_retrieval_context
from .gold_injector import ensure_gold_coverage
from .io_utils import iter_jsonl, load_json_or_empty, write_json, write_jsonl
from .meta_client import MetadataClient
from .retriever_client import RetrievalConfig, call_hybrid_retrievers
from .token_utils import parse_target_tokens

JsonDict = Dict[str, Any]


def resolve_output_path(input_path: str, output_path: str, output_dir: str, default_name: str) -> Path:
    if output_path:
        return Path(output_path)
    if output_dir:
        return Path(output_dir) / default_name
    inp = Path(input_path)
    return inp.with_name(default_name)


def build_one_pair(
    pair: JsonDict,
    *,
    cfg: RetrievalConfig,
    meta_client: MetadataClient,
    max_cf_llm: int,
    max_cf_tool_bundle: int,
    max_semantic_llm: int,
    max_semantic_tool: int,
    max_desc_chars: int,
    include_score: bool,
    include_query: bool,
    fallback_score: float,
    allow_score_fallback: bool,
    keep_debug_fields: bool,
) -> JsonDict:
    query = str(pair.get("query") or "").strip()
    target = str(pair.get("target") or "").strip()
    if not query:
        raise ValueError("pair missing query")
    if not target:
        raise ValueError("pair missing target")

    gold = parse_target_tokens(target)
    retrieval = call_hybrid_retrievers(query, cfg)
    report = ensure_gold_coverage(
        retrieval,
        gold,
        query=query,
        cfg=cfg,
        meta_client=meta_client,
        fallback_score=fallback_score,
        allow_score_fallback=allow_score_fallback,
        semantic_llm_keep=max_semantic_llm,
        semantic_tool_keep=max_semantic_tool,
    )

    context = build_compact_retrieval_context(
        {"query": query, "retrieval": retrieval},
        max_cf_llm=max_cf_llm,
        max_cf_tool_bundle=max_cf_tool_bundle,
        max_semantic_llm=max_semantic_llm,
        max_semantic_tool=max_semantic_tool,
        max_desc_chars=max_desc_chars,
        include_score=include_score,
        include_query=include_query,
    )

    out: JsonDict = {
        "qid": pair.get("qid", ""),
        "agent_id": pair.get("agent_id", ""),
        "part": pair.get("part", ""),
        "topk": pair.get("topk", 0),
        "query": query,
        "target": target,
        "context": context,
    }
    if keep_debug_fields:
        out["rag_meta"] = {
            "gold_llm": report.gold_llm,
            "gold_tools": report.gold_tools,
            "covered_llm": report.covered_llm,
            "covered_tools": report.covered_tools,
            "missing_llm": report.missing_llm,
            "missing_tools": report.missing_tools,
            "injected_llm": report.injected_llm,
            "injected_tools": report.injected_tools,
            "retrieval_source": retrieval.get("source"),
            "retrieval_latency_sec": retrieval.get("latency_sec"),
            "retrieval_errors": retrieval.get("errors", {}),
        }
    return out


def build_file(
    *,
    input_path: str,
    output_path: str,
    cfg: RetrievalConfig,
    meta_client: MetadataClient,
    limit: int,
    resume: bool,
    max_cf_llm: int,
    max_cf_tool_bundle: int,
    max_semantic_llm: int,
    max_semantic_tool: int,
    max_desc_chars: int,
    include_score: bool,
    include_query: bool,
    fallback_score: float,
    allow_score_fallback: bool,
    keep_debug_fields: bool,
    errors_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = Path(errors_path) if errors_path else out_path.with_suffix(out_path.suffix + ".errors.jsonl")

    done_keys = set()
    if resume and out_path.is_file():
        for row in iter_jsonl(out_path):
            key = (str(row.get("qid", "")), str(row.get("agent_id", "")), str(row.get("target", "")))
            done_keys.add(key)

    mode = "a" if resume and out_path.is_file() else "w"
    n_total = 0
    n_written = 0
    n_skipped = 0
    n_errors = 0
    n_injected_llm = 0
    n_injected_tool_rows = 0
    n_missing_tools = 0

    with out_path.open(mode, encoding="utf-8") as fout, err_path.open("a", encoding="utf-8") as ferr:
        for pair in tqdm(iter_jsonl(input_path, limit=limit), desc=f"build {Path(output_path).name}", unit="pair"):
            n_total += 1
            key = (str(pair.get("qid", "")), str(pair.get("agent_id", "")), str(pair.get("target", "")))
            if key in done_keys:
                n_skipped += 1
                continue
            try:
                row = build_one_pair(
                    pair,
                    cfg=cfg,
                    meta_client=meta_client,
                    max_cf_llm=max_cf_llm,
                    max_cf_tool_bundle=max_cf_tool_bundle,
                    max_semantic_llm=max_semantic_llm,
                    max_semantic_tool=max_semantic_tool,
                    max_desc_chars=max_desc_chars,
                    include_score=include_score,
                    include_query=include_query,
                    fallback_score=fallback_score,
                    allow_score_fallback=allow_score_fallback,
                    keep_debug_fields=keep_debug_fields,
                )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
                meta = row.get("rag_meta", {})
                if meta.get("injected_llm"):
                    n_injected_llm += 1
                if meta.get("injected_tools"):
                    n_injected_tool_rows += 1
                    n_missing_tools += len(meta.get("injected_tools") or [])
            except Exception as e:
                n_errors += 1
                ferr.write(json.dumps({
                    "qid": pair.get("qid", ""),
                    "agent_id": pair.get("agent_id", ""),
                    "error": repr(e),
                    "query": pair.get("query", "")[:500],
                    "target": pair.get("target", ""),
                }, ensure_ascii=False) + "\n")
                ferr.flush()
                if not cfg.allow_partial_retrieval:
                    # Continue by default for long dataset construction; errors are logged.
                    pass

    return {
        "input_path": input_path,
        "output_path": str(out_path),
        "errors_path": str(err_path),
        "n_total_seen": n_total,
        "n_written": n_written,
        "n_skipped_resume": n_skipped,
        "n_errors": n_errors,
        "n_injected_llm_rows": n_injected_llm,
        "n_injected_tool_rows": n_injected_tool_rows,
        "n_injected_tool_components": n_missing_tools,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build retrieval-grounded RAG pair_rag.jsonl for AgentRec generative SFT.")

    ap.add_argument("--pairs", type=str, required=True, help="Input clean pairs.jsonl")
    ap.add_argument("--pairs_valid", type=str, default="", help="Optional input pairs_valid.jsonl")
    ap.add_argument("--output_dir", type=str, default="", help="Output dir; default writes next to input")
    ap.add_argument("--output", type=str, default="", help="Output train RAG jsonl; overrides output_dir/pair_rag.jsonl")
    ap.add_argument("--output_valid", type=str, default="", help="Output valid RAG jsonl; overrides output_dir/pair_rag_valid.jsonl")
    ap.add_argument("--summary", type=str, default="", help="Path for summary JSON")

    ap.add_argument("--cf_llm_retr_url", type=str, default=os.getenv("CF_LLM_RETR_URL", "http://127.0.0.1:9000/predict"))
    ap.add_argument("--cf_tool_retr_url", type=str, default=os.getenv("CF_TOOL_RETR_URL", "http://127.0.0.1:9001/predict"))
    ap.add_argument("--semantic_retr_url", type=str, default=os.getenv("SEMANTIC_RETR_URL", "http://127.0.0.1:8504/retrieve"))
    ap.add_argument("--score_url", type=str, default=os.getenv("SCORE_URL", "http://127.0.0.1:8504/compute_score"))
    ap.add_argument("--meta_api_base", type=str, default=os.getenv("META_API_BASE", ""), help="Optional metadata API, e.g. http://127.0.0.1:6000")
    ap.add_argument("--timeout", type=int, default=300)

    ap.add_argument("--cf_llm_topk", type=int, default=8)
    ap.add_argument("--cf_tool_topk", type=int, default=12)
    ap.add_argument("--semantic_llm_topk", type=int, default=8)
    ap.add_argument("--semantic_tool_topk", type=int, default=12)
    ap.add_argument("--rewrite", type=int, default=1)
    ap.add_argument("--cf_rewrite", type=int, default=-1)
    ap.add_argument("--semantic_rewrite", type=int, default=-1)
    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "hybrid"])
    ap.add_argument("--candidate_multiplier", type=int, default=5)
    ap.add_argument("--include_raw", type=int, default=0)
    ap.add_argument("--include_doc_text", type=int, default=0)
    ap.add_argument("--tool_recall_mode", type=int, default=0)
    ap.add_argument("--per_subquery_top_k", type=int, default=0)
    ap.add_argument("--final_top_k", type=int, default=0)
    ap.add_argument("--retrieval_extra_json", type=str, default="")

    ap.add_argument("--max_cf_llm", type=int, default=8)
    ap.add_argument("--max_cf_tool_bundle", type=int, default=12)
    ap.add_argument("--max_semantic_llm", type=int, default=8)
    ap.add_argument("--max_semantic_tool", type=int, default=12)
    ap.add_argument("--max_desc_chars", type=int, default=260)
    ap.add_argument("--include_query", type=int, default=0)
    ap.add_argument("--no_score_in_context", type=int, default=0)

    ap.add_argument("--fallback_score", type=float, default=0.55)
    ap.add_argument("--allow_score_fallback", type=int, default=1)
    ap.add_argument("--allow_partial_retrieval", type=int, default=0)
    ap.add_argument("--dry_run_retrieval", type=int, default=0, help="No server calls; produces contexts with only injected gold components.")
    ap.add_argument("--keep_debug_fields", type=int, default=1)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return args


def main() -> None:
    args = parse_args()
    started = time.time()

    retrieval_extra = load_json_or_empty(args.retrieval_extra_json)
    cf_rewrite = None if args.cf_rewrite < 0 else bool(args.cf_rewrite)
    semantic_rewrite = None if args.semantic_rewrite < 0 else bool(args.semantic_rewrite)
    cfg = RetrievalConfig(
        cf_llm_retr_url=args.cf_llm_retr_url,
        cf_tool_retr_url=args.cf_tool_retr_url,
        semantic_retr_url=args.semantic_retr_url,
        score_url=args.score_url,
        timeout=args.timeout,
        cf_llm_topk=args.cf_llm_topk,
        cf_tool_topk=args.cf_tool_topk,
        semantic_llm_topk=args.semantic_llm_topk,
        semantic_tool_topk=args.semantic_tool_topk,
        rewrite=bool(args.rewrite),
        cf_rewrite=cf_rewrite,
        semantic_rewrite=semantic_rewrite,
        agg=args.agg,
        candidate_multiplier=args.candidate_multiplier,
        include_raw=bool(args.include_raw),
        include_doc_text=bool(args.include_doc_text),
        tool_recall_mode=bool(args.tool_recall_mode),
        per_subquery_top_k=args.per_subquery_top_k,
        final_top_k=args.final_top_k,
        allow_partial_retrieval=bool(args.allow_partial_retrieval),
        retrieval_extra=retrieval_extra,
        dry_run_retrieval=bool(args.dry_run_retrieval),
    )
    meta_client = MetadataClient(args.meta_api_base, timeout=min(args.timeout, 10))

    train_out = resolve_output_path(args.pairs, args.output, args.output_dir, "pair_rag.jsonl")
    results = []
    results.append(build_file(
        input_path=args.pairs,
        output_path=str(train_out),
        cfg=cfg,
        meta_client=meta_client,
        limit=args.limit,
        resume=bool(args.resume),
        max_cf_llm=args.max_cf_llm,
        max_cf_tool_bundle=args.max_cf_tool_bundle,
        max_semantic_llm=args.max_semantic_llm,
        max_semantic_tool=args.max_semantic_tool,
        max_desc_chars=args.max_desc_chars,
        include_score=not bool(args.no_score_in_context),
        include_query=bool(args.include_query),
        fallback_score=args.fallback_score,
        allow_score_fallback=bool(args.allow_score_fallback),
        keep_debug_fields=bool(args.keep_debug_fields),
    ))

    if args.pairs_valid:
        valid_out = resolve_output_path(args.pairs_valid, args.output_valid, args.output_dir, "pair_rag_valid.jsonl")
        results.append(build_file(
            input_path=args.pairs_valid,
            output_path=str(valid_out),
            cfg=cfg,
            meta_client=meta_client,
            limit=args.limit,
            resume=bool(args.resume),
            max_cf_llm=args.max_cf_llm,
            max_cf_tool_bundle=args.max_cf_tool_bundle,
            max_semantic_llm=args.max_semantic_llm,
            max_semantic_tool=args.max_semantic_tool,
            max_desc_chars=args.max_desc_chars,
            include_score=not bool(args.no_score_in_context),
            include_query=bool(args.include_query),
            fallback_score=args.fallback_score,
            allow_score_fallback=bool(args.allow_score_fallback),
            keep_debug_fields=bool(args.keep_debug_fields),
        ))

    summary = {
        "ok": True,
        "elapsed_sec": round(time.time() - started, 4),
        "config": {
            "cf_llm_retr_url": cfg.cf_llm_retr_url,
            "cf_tool_retr_url": cfg.cf_tool_retr_url,
            "semantic_retr_url": cfg.semantic_retr_url,
            "score_url": cfg.score_url,
            "meta_api_base": args.meta_api_base,
            "cf_llm_topk": cfg.cf_llm_topk,
            "cf_tool_topk": cfg.cf_tool_topk,
            "semantic_llm_topk": cfg.semantic_llm_topk,
            "semantic_tool_topk": cfg.semantic_tool_topk,
            "fallback_score": args.fallback_score,
            "allow_score_fallback": bool(args.allow_score_fallback),
            "dry_run_retrieval": bool(args.dry_run_retrieval),
        },
        "files": results,
    }
    summary_path = Path(args.summary) if args.summary else (Path(args.output_dir) / "rag_synth_summary.json" if args.output_dir else train_out.with_name("rag_synth_summary.json"))
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
