from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

JsonDict = Dict[str, Any]


@dataclass
class RetrievalConfig:
    cf_llm_retr_url: str = "http://127.0.0.1:9000/predict"
    cf_tool_retr_url: str = "http://127.0.0.1:9001/predict"
    semantic_retr_url: str = "http://127.0.0.1:8504/retrieve"
    score_url: str = "http://127.0.0.1:8504/compute_score"
    timeout: int = 300
    cf_llm_topk: int = 8
    cf_tool_topk: int = 12
    semantic_llm_topk: int = 8
    semantic_tool_topk: int = 12
    rewrite: bool = True
    cf_rewrite: Optional[bool] = None
    semantic_rewrite: Optional[bool] = None
    agg: str = "max"
    candidate_multiplier: int = 5
    include_raw: bool = False
    include_doc_text: bool = False
    tool_recall_mode: bool = False
    per_subquery_top_k: int = 0
    final_top_k: int = 0
    allow_partial_retrieval: bool = False
    retrieval_extra: Optional[JsonDict] = None
    dry_run_retrieval: bool = False


def post_json(url: str, payload: JsonDict, *, timeout: int = 300) -> JsonDict:
    resp = requests.post(url, json=payload, timeout=timeout)
    text = resp.text
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} failed: status={resp.status_code}, body={text[:2000]}")
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"POST {url} returned non-JSON body: {text[:2000]}") from e
    if isinstance(data, dict) and data.get("ok") is False:
        raise RuntimeError(f"POST {url} returned ok=false: {json.dumps(data, ensure_ascii=False)[:2000]}")
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-object JSON: {type(data)}")
    return data


def build_payloads(query: str, cfg: RetrievalConfig) -> Tuple[JsonDict, JsonDict, JsonDict]:
    cf_rewrite = cfg.rewrite if cfg.cf_rewrite is None else cfg.cf_rewrite
    semantic_rewrite = cfg.rewrite if cfg.semantic_rewrite is None else cfg.semantic_rewrite

    cf_llm_payload: JsonDict = {
        "query": query,
        "top_k": int(cfg.cf_llm_topk),
        "topk": int(cfg.cf_llm_topk),
        "rewrite": bool(cf_rewrite),
        "agg": cfg.agg,
        "include_raw_agent": bool(cfg.include_raw),
    }
    cf_tool_payload: JsonDict = {
        "query": query,
        "top_k": int(cfg.cf_tool_topk),
        "topk": int(cfg.cf_tool_topk),
        "rewrite": bool(cf_rewrite),
        "agg": cfg.agg,
        "candidate_multiplier": int(cfg.candidate_multiplier),
        "include_raw_tool": bool(cfg.include_raw),
        "include_doc_text": bool(cfg.include_doc_text),
    }
    semantic_payload: JsonDict = {
        "query": query,
        "targets": ["llm", "tool"],
        "top_k": {"llm": int(cfg.semantic_llm_topk), "tool": int(cfg.semantic_tool_topk)},
        "rewrite": bool(semantic_rewrite),
        "agg": cfg.agg,
        "candidate_multiplier": int(cfg.candidate_multiplier),
        "include_raw": bool(cfg.include_raw),
        "include_doc_text": bool(cfg.include_doc_text),
        "tool_recall_mode": bool(cfg.tool_recall_mode),
    }
    if cfg.per_subquery_top_k > 0:
        semantic_payload["per_subquery_top_k"] = int(cfg.per_subquery_top_k)
    if cfg.final_top_k > 0:
        semantic_payload["final_top_k"] = int(cfg.final_top_k)
    if cfg.retrieval_extra:
        semantic_payload.update(cfg.retrieval_extra)
    return cf_llm_payload, cf_tool_payload, semantic_payload


def call_hybrid_retrievers(query: str, cfg: RetrievalConfig) -> JsonDict:
    cf_llm_payload, cf_tool_payload, semantic_payload = build_payloads(query, cfg)
    record: JsonDict = {
        "source": "hybrid_cf_plus_semantic",
        "cf_llm_url": cfg.cf_llm_retr_url,
        "cf_tool_url": cfg.cf_tool_retr_url,
        "semantic_url": cfg.semantic_retr_url,
        "cf_llm_request": cf_llm_payload,
        "cf_tool_request": cf_tool_payload,
        "semantic_request": semantic_payload,
        "errors": {},
    }

    if cfg.dry_run_retrieval:
        record.update({
            "cf_llm_response": {"ok": True, "topk": []},
            "cf_tool_response": {"ok": True, "results": []},
            "semantic_response": {"ok": True, "results_by_type": {"llm": {"results": []}, "tool": {"results": []}}},
            "source": "dry_run_empty_retrieval",
        })
        return record

    started = time.time()
    partial: JsonDict = {}
    errors: JsonDict = {}
    try:
        partial["cf_llm_response"] = post_json(cfg.cf_llm_retr_url, cf_llm_payload, timeout=cfg.timeout)
    except Exception as e:
        errors["cf_llm_error"] = {"error": repr(e), "traceback": traceback.format_exc()}
    try:
        partial["cf_tool_response"] = post_json(cfg.cf_tool_retr_url, cf_tool_payload, timeout=cfg.timeout)
    except Exception as e:
        errors["cf_tool_error"] = {"error": repr(e), "traceback": traceback.format_exc()}
    try:
        partial["semantic_response"] = post_json(cfg.semantic_retr_url, semantic_payload, timeout=cfg.timeout)
    except Exception as e:
        errors["semantic_error"] = {"error": repr(e), "traceback": traceback.format_exc()}

    have_cf = "cf_llm_response" in partial and "cf_tool_response" in partial
    have_sem = "semantic_response" in partial
    if not (have_cf and have_sem):
        if not cfg.allow_partial_retrieval:
            raise RuntimeError(f"Hybrid retrieval failed. errors={json.dumps(errors, ensure_ascii=False)[:4000]}")
        record["source"] = "partial_retrieval"
        partial.setdefault("cf_llm_response", {"ok": False, "topk": []})
        partial.setdefault("cf_tool_response", {"ok": False, "results": []})
        partial.setdefault("semantic_response", {"ok": False, "results_by_type": {"llm": {"results": []}, "tool": {"results": []}}})

    record.update(partial)
    record["errors"] = errors
    record["latency_sec"] = round(time.time() - started, 4)
    return record


def call_score(
    *,
    score_url: str,
    query: str,
    component_type: str,
    component: str,
    timeout: int = 60,
    fallback_score: float = 0.55,
    allow_fallback: bool = True,
) -> JsonDict:
    payload = {
        "query": query,
        "component_type": component_type,
        "component": component,
        "token": component,
    }
    try:
        data = post_json(score_url, payload, timeout=timeout)
        raw = data.get("raw_score", data.get("score", data.get("norm_score")))
        score = data.get("score", data.get("norm_score", raw))
        if raw is None:
            raw = fallback_score
        if score is None:
            score = raw
        return {
            "ok": True,
            "raw_score": float(raw),
            "score": float(score),
            "score_source": "compute_score",
            "response": data,
        }
    except Exception as e:
        if not allow_fallback:
            raise
        return {
            "ok": False,
            "raw_score": float(fallback_score),
            "score": float(fallback_score),
            "score_source": "fallback_stub",
            "error": repr(e),
        }
