#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Component-Ret baseline for Agent Generative Recommendation.\n\nDefinition\n----------\nFor each query q:\n    m* = top-1 item from the CF LLM retrieval\n    T*_K = top-K independently retrieved tools from the semantic/capability tool retrieval\n    A_K = (m*, T*_K)\n\nNo generator, beam search, bundle critic, gold tool-count oracle, or target field is\nused to construct the prediction.\n\nThe script evaluates/saves K in {1,2,3,4,5} (configurable) in ONE retrieval pass.\nEach K gets its own experiment directory with the same layout as the v13 batch run:\n    <experiment_root>/k1/\n        sample_manifest.jsonl\n        experiment_metadata.json\n        results.jsonl\n        failures.jsonl\n        run_summary.json\n        per_sample/*.json\n    ...\n\nThis allows the existing evaluator to be run independently on k1..k5 and the best\nGLOBAL K to be selected on validation data without leaking per-query gold bundle size.\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import os
import random
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_JSONL = str(AC_ROOT / 'datasets/generative_v9_sft/sft_valid.jsonl')
DEFAULT_CF_LLM_RETR_URL = "http://127.0.0.1:9000/predict"
DEFAULT_SEMANTIC_RETR_URL = "http://127.0.0.1:8504/retrieve"

TOOL_SEP_TOKEN = "<TOOL_SEP>"
END_TOKEN = "<SPECIAL_END>"
TOOL_EMPTY_TOKEN = "<TOOL_EMPTY>"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return str(obj)


def _write_json(path: str, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def _append_jsonl(path: str, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(obj), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_jsonl_atomic(path: str, rows: Sequence[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def _post_json(url: str, payload: Dict[str, Any], *, timeout: int) -> Dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("The 'requests' package is required for retrieval calls") from exc

    resp = requests.post(url, json=payload, timeout=timeout)
    text = resp.text
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} failed: status={resp.status_code}, body={text[:2000]}")
    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"POST {url} returned non-JSON body: {text[:2000]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-object JSON: {type(data)}")
    if data.get("ok") is False:
        raise RuntimeError(f"POST {url} returned ok=false: {json.dumps(data, ensure_ascii=False)[:2000]}")
    return data


def normalize_raw_name(s: str) -> str:
    s = (s or "").strip()
    return "".join("_" if ch.isspace() else ch for ch in s)


def wrap_llm_name(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("<LLM_") and raw.endswith(">"):
        return raw
    return f"<LLM_{normalize_raw_name(raw)}>"


def wrap_tool_name(raw: str) -> str:
    """Match the current SFT target style for tool phrases."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("<<") and raw.endswith(">>"):
        return raw
    if raw.startswith("<TOOL_") and raw.endswith(">"):
        return raw
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    return f"<<{raw}>>"


def flatten_desc(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(flatten_desc(v) for v in x if v is not None).strip()
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            vv = flatten_desc(v)
            if vv:
                parts.append(f"{k}: {vv}")
        return "; ".join(parts)
    return str(x).strip()


def _pick_response_rows(obj: Dict[str, Any], keys: Sequence[str]) -> List[Any]:
    if not isinstance(obj, dict):
        return []
    for key in keys:
        value = obj.get(key)
        if isinstance(value, list):
            return value
    return []


def _get_semantic_items(retrieval_json: Dict[str, Any], target: str) -> List[Dict[str, Any]]:
    by_type = retrieval_json.get("results_by_type", {}) if isinstance(retrieval_json, dict) else {}
    if isinstance(by_type, dict):
        group = by_type.get(target, {}) or {}
        if isinstance(group, dict) and isinstance(group.get("results"), list):
            return [x for x in group["results"] if isinstance(x, dict)]

    evidence = retrieval_json.get("evidence", []) if isinstance(retrieval_json, dict) else []
    if isinstance(evidence, list):
        return [
            x for x in evidence
            if isinstance(x, dict) and x.get("component_type") == target
        ]
    return []


def _pick_llm_name(item: Dict[str, Any]) -> str:
    llm_obj = item.get("llm")
    if isinstance(llm_obj, dict):
        return str(
            llm_obj.get("name")
            or llm_obj.get("id")
            or llm_obj.get("model")
            or item.get("name")
            or item.get("component")
            or ""
        )
    return str(item.get("llm") or item.get("name") or item.get("component") or "")


def parse_cf_llm_candidates(response: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    rows = _pick_response_rows(response, ["topk", "results", "evidence"])
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(rows[: max(0, int(limit))], 1):
        if not isinstance(item, dict):
            continue
        name = _pick_llm_name(item)
        token = wrap_llm_name(name)
        if not token or token in seen:
            continue
        seen.add(token)
        llm_obj = item.get("llm", {}) if isinstance(item.get("llm"), dict) else {}
        out.append({
            "token": token,
            "name": name,
            "rank": item.get("rank") if item.get("rank") is not None else idx,
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "norm_score": item.get("norm_score") or item.get("normalized_score"),
            "item_id": item.get("item_id") or item.get("aid") or item.get("agent_id"),
            "description": flatten_desc(item.get("description") or item.get("desc") or llm_obj.get("description")),
            "source": "cf_llm_9000_predict",
        })
    return out


def _semantic_tool_key(item: Dict[str, Any]) -> str:
    tool_obj = item.get("tool", {}) if isinstance(item.get("tool"), dict) else {}
    return str(
        item.get("component")
        or tool_obj.get("key")
        or tool_obj.get("item_id")
        or item.get("tid")
        or item.get("tool_id")
        or item.get("key")
        or ""
    )


def parse_semantic_tool_candidates(response: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    rows = _get_semantic_items(response, "tool")
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(rows[: max(0, int(limit))], 1):
        key = _semantic_tool_key(item)
        token = wrap_tool_name(key)
        if not token or token in seen:
            continue
        seen.add(token)
        tool_obj = item.get("tool", {}) if isinstance(item.get("tool"), dict) else {}
        out.append({
            "token": token,
            "component": key,
            "rank": item.get("rank") if item.get("rank") is not None else idx,
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "norm_score": item.get("norm_score") or item.get("normalized_score"),
            "matched_subquery": item.get("matched_subquery"),
            "item_id": item.get("item_id") or tool_obj.get("item_id") or tool_obj.get("key"),
            "description": flatten_desc(tool_obj.get("description") or item.get("description") or item.get("doc_text")),
            "source": "semantic_8504",
        })
    return out


def build_strict_text(llm_token: str, tool_tokens: Sequence[str]) -> str:
    tools = [t for t in tool_tokens if t]
    if not tools:
        tools = [TOOL_EMPTY_TOKEN]
    return " ".join([llm_token or "<LLM_UNK>", TOOL_SEP_TOKEN] + list(tools) + [END_TOKEN])


def _load_validation_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            enriched = dict(row)
            enriched["_source_line"] = line_no
            rows.append(enriched)
    if not rows:
        raise ValueError(f"No valid non-empty-query rows found in {path}")
    return rows


def _load_manifest_file(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"sample_id", "query"}
    for i, row in enumerate(rows, 1):
        missing = [k for k in required if not row.get(k)]
        if missing:
            raise ValueError(f"Manifest row {i} missing required fields: {missing}")
    return rows


def prepare_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.manifest_file:
        return _load_manifest_file(args.manifest_file)

    source_rows = _load_validation_rows(args.input_jsonl)
    requested = int(args.sample_size)
    take = len(source_rows) if requested == 0 else min(requested, len(source_rows))
    rng = random.Random(int(args.sample_seed))
    chosen_indices = sorted(rng.sample(range(len(source_rows)), take))
    manifest: List[Dict[str, Any]] = []
    for sample_number, source_index in enumerate(chosen_indices, 1):
        source = source_rows[source_index]
        manifest.append({
            "sample_id": f"sample_{sample_number:06d}",
            "sample_number": sample_number,
            "source_index_zero_based": source_index,
            "source_line": source.get("_source_line"),
            "qid": source.get("qid"),
            "agent_id": source.get("agent_id"),
            "part": source.get("part"),
            "topk": source.get("topk"),
            "query": source.get("query"),
            "target": source.get("target"),
            "target_context_explanation": source.get("target_context_explanation"),
            "source_example": {k: v for k, v in source.items() if k != "_source_line"},
        })
    return manifest


def _read_successful_sample_ids(path: str) -> set:
    completed = set()
    p = Path(path)
    if not p.exists():
        return completed
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            sample = row.get("dataset_example", {}) if isinstance(row, dict) else {}
            if row.get("ok", True) and sample.get("sample_id"):
                completed.add(str(sample["sample_id"]))
    return completed


def parse_k_values(text: str) -> List[int]:
    vals: List[int] = []
    seen = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError("All component K values must be >= 1")
        if k not in seen:
            vals.append(k)
            seen.add(k)
    if not vals:
        raise ValueError("At least one component K is required")
    return sorted(vals)


def retrieve_components(args: argparse.Namespace, query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cf_llm_payload = {
        "query": query,
        "top_k": int(args.cf_llm_retrieval_depth),
        "topk": int(args.cf_llm_retrieval_depth),
        "rewrite": bool(args.cf_rewrite),
        "agg": args.agg,
        "include_raw_agent": bool(args.include_raw),
    }
    semantic_payload: Dict[str, Any] = {
        "query": query,
        "targets": ["tool"],
        "top_k": {"tool": int(args.semantic_tool_retrieval_depth)},
        "rewrite": bool(args.semantic_rewrite),
        "agg": args.agg,
        "candidate_multiplier": int(args.candidate_multiplier),
        "include_raw": bool(args.include_raw),
        "include_doc_text": bool(args.include_doc_text),
        "tool_recall_mode": bool(args.tool_recall_mode),
        "final_top_k": int(args.semantic_tool_retrieval_depth),
    }
    if int(args.per_subquery_top_k) > 0:
        semantic_payload["per_subquery_top_k"] = int(args.per_subquery_top_k)

    started = time.time()
    responses: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    jobs = {
        "cf_llm_response": (args.cf_llm_retr_url, cf_llm_payload),
        "semantic_tool_response": (args.semantic_retr_url, semantic_payload),
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_key = {
            executor.submit(_post_json, url, payload, timeout=int(args.retriever_timeout)): key
            for key, (url, payload) in jobs.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                responses[key] = future.result()
            except Exception as exc:
                errors[key] = repr(exc)

    if errors:
        raise RuntimeError(f"Component retrieval failed: {errors}")

    llms = parse_cf_llm_candidates(
        responses["cf_llm_response"],
        limit=int(args.cf_llm_retrieval_depth),
    )
    tools = parse_semantic_tool_candidates(
        responses["semantic_tool_response"],
        limit=int(args.semantic_tool_retrieval_depth),
    )
    if not llms:
        raise RuntimeError('CF LLM retrieval returned no usable LLM candidates')
    if not tools:
        raise RuntimeError('Semantic tool retrieval returned no usable tool candidates')

    retrieval_record = {
        "source": "component_retrieve_cf_llm_plus_semantic_tool",
        "started_at": _now_iso(),
        "cf_llm_url": args.cf_llm_retr_url,
        "semantic_tool_url": args.semantic_retr_url,
        "cf_llm_request": cf_llm_payload,
        "semantic_tool_request": semantic_payload,
        "cf_llm_response": responses["cf_llm_response"],
        "semantic_tool_response": responses["semantic_tool_response"],
        "parsed_cf_llm_candidates": llms,
        "parsed_semantic_tool_candidates": tools,
        "latency_sec": round(time.time() - started, 4),
        "finished_at": _now_iso(),
    }
    return llms, tools, retrieval_record


def build_record(
    *,
    args: argparse.Namespace,
    sample: Dict[str, Any],
    k: int,
    llms: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    retrieval_record: Dict[str, Any],
    experiment_dir: Path,
    sample_latency_sec: float,
) -> Dict[str, Any]:
    selected_llm = llms[0]
    selected_tools = tools[: int(k)]
    llm_token = selected_llm["token"]
    tool_tokens = [row["token"] for row in selected_tools]
    strict_text = build_strict_text(llm_token, tool_tokens)

    result = {
        "rank": 1,
        "gen_text": strict_text,
        "llm_token": llm_token,
        "tool_tokens": tool_tokens if tool_tokens else [TOOL_EMPTY_TOKEN],
        "strict_text": strict_text,
        "explanation": "",
        "baseline": {
            "name": "Component-Ret",
            "component_k": int(k),
            "selection_rule": "Top-1 CF-retrieved LLM + Top-K independently semantic-retrieved tools",
            "llm": selected_llm,
            "tools": selected_tools,
            "cross_type_score_aggregation": None,
            "note": 'LLM and tool scores are not added because the two retrievals have different score scales; composition is rank-based.',
        },
    }

    record = {
        "ok": True,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8],
        "created_at": _now_iso(),
        "script": str(Path(__file__).resolve()),
        "query": sample.get("query"),
        "config": {
            "baseline_name": "Component-Ret",
            "baseline_category": "retrieval_based_agent_composition",
            "composition_rule": "top1_cf_llm_plus_topk_independent_semantic_tools",
            "component_k": int(k),
            "k_is_global_hyperparameter": True,
            "gold_tool_count_used": False,
            "cf_llm_retr_url": args.cf_llm_retr_url,
            "semantic_retr_url": args.semantic_retr_url,
            "cf_llm_retrieval_depth": int(args.cf_llm_retrieval_depth),
            "semantic_tool_retrieval_depth": int(args.semantic_tool_retrieval_depth),
            "cf_rewrite": bool(args.cf_rewrite),
            "semantic_rewrite": bool(args.semantic_rewrite),
            "agg": args.agg,
            "candidate_multiplier": int(args.candidate_multiplier),
            "tool_sep_token": TOOL_SEP_TOKEN,
            "end_token": END_TOKEN,
            "tool_empty_token": TOOL_EMPTY_TOKEN,
            "generator_used": False,
            "critic_used": False,
            "beam_search_used": False,
        },
        "dataset_example": sample,
        "retrieval": retrieval_record,
        "context": {
            "source": "retrieval_only_baseline",
            "meta": {
                "num_llm_candidates": len(llms),
                "num_tool_candidates": len(tools),
                "selected_llm": selected_llm,
                "selected_tools": selected_tools,
            },
        },
        "composition": {
            "llm_selection": selected_llm,
            "tool_selection": selected_tools,
            "requested_tool_k": int(k),
            "actual_tool_count": len(selected_tools),
            "strict_text": strict_text,
        },
        "generation": {
            "skipped": True,
            "reason": "Component-Ret is a deterministic retrieval-composition baseline; no generator is used.",
            "results": [result],
        },
        "results": [result],
        "batch": {
            "experiment_dir": str(experiment_dir.resolve()),
            "sample_latency_sec": round(sample_latency_sec, 4),
            "completed_at": _now_iso(),
        },
        "checkpoint": {"stage": "sample_complete", "saved_at": _now_iso()},
        "finished_at": _now_iso(),
    }
    return record


def ensure_experiment_layout(
    args: argparse.Namespace,
    manifest: List[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[int, Dict[str, Path]]:
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)

    root_meta = {
        "created_at": _now_iso(),
        "baseline_name": "Component-Ret",
        "definition": "Top-1 CF LLM + Top-K independently retrieved semantic/capability tools",
        "component_k_values": list(k_values),
        "selection_protocol": "Run all global K values on the same validation manifest; choose one global K by validation metric. Never use per-query gold tool count.",
        "input_jsonl": str(Path(args.input_jsonl).resolve()) if args.input_jsonl else "",
        "reference_manifest": str(Path(args.manifest_file).resolve()) if args.manifest_file else "",
        "sample_size_requested": int(args.sample_size),
        "sample_size_actual": len(manifest),
        "sample_seed": int(args.sample_seed),
    }
    _write_json(str(root / "baseline_metadata.json"), root_meta)
    _write_jsonl_atomic(str(root / "sample_manifest.jsonl"), manifest)

    paths: Dict[int, Dict[str, Path]] = {}
    for k in k_values:
        exp_dir = root / f"k{k}"
        per_sample = exp_dir / "per_sample"
        per_sample.mkdir(parents=True, exist_ok=True)
        _write_jsonl_atomic(str(exp_dir / "sample_manifest.jsonl"), manifest)
        _write_json(str(exp_dir / "experiment_metadata.json"), {
            **root_meta,
            "component_k": int(k),
            "experiment_dir": str(exp_dir.resolve()),
            "required_evaluation_fields": ["query", "target", "target_context_explanation"],
        })
        paths[k] = {
            "exp_dir": exp_dir,
            "per_sample": per_sample,
            "results": exp_dir / "results.jsonl",
            "failures": exp_dir / "failures.jsonl",
            "summary": exp_dir / "run_summary.json",
        }
    return paths


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    k_values = parse_k_values(args.component_ks)
    if max(k_values) > int(args.semantic_tool_retrieval_depth):
        raise ValueError(
            f"max(component_ks)={max(k_values)} exceeds semantic_tool_retrieval_depth={args.semantic_tool_retrieval_depth}"
        )

    manifest = prepare_manifest(args)
    paths = ensure_experiment_layout(args, manifest, k_values)

    completed_by_k = {
        k: _read_successful_sample_ids(str(paths[k]["results"])) if bool(args.resume) else set()
        for k in k_values
    }
    if not bool(args.resume):
        for k in k_values:
            if paths[k]["results"].exists() and paths[k]["results"].stat().st_size:
                raise RuntimeError(
                    f"{paths[k]['results']} already exists. Use --resume 1 or choose a new --experiment_root."
                )

    pending: List[Dict[str, Any]] = []
    for sample in manifest:
        sid = str(sample["sample_id"])
        if not all(sid in completed_by_k[k] for k in k_values):
            pending.append(sample)

    print(
        f"[BATCH] Component-Ret K={k_values} manifest={len(manifest)} "
        f"pending_shared_retrievals={len(pending)}",
        flush=True,
    )

    started_at = _now_iso()
    succeeded_now = {k: 0 for k in k_values}
    failed_now = {k: 0 for k in k_values}

    try:
        from tqdm.auto import tqdm
        iterator: Iterable[Dict[str, Any]] = tqdm(
            pending, total=len(pending), desc="Component-Ret", unit="sample", dynamic_ncols=True
        )
    except Exception:
        iterator = pending

    for sample in iterator:
        sid = str(sample["sample_id"])
        qid = str(sample.get("qid") or "no_qid")
        safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)[:80]
        sample_started = time.time()
        missing_ks = [k for k in k_values if sid not in completed_by_k[k]]
        try:
            llms, tools, retrieval_record = retrieve_components(args, str(sample["query"]))
            retrieval_latency = time.time() - sample_started
            for k in missing_ks:
                record = build_record(
                    args=args,
                    sample=sample,
                    k=k,
                    llms=llms,
                    tools=tools,
                    retrieval_record=retrieval_record,
                    experiment_dir=paths[k]["exp_dir"],
                    sample_latency_sec=retrieval_latency,
                )
                sample_output = paths[k]["per_sample"] / f"{sid}_{safe_qid}.json"
                _write_json(str(sample_output), record)
                _append_jsonl(str(paths[k]["results"]), record)
                completed_by_k[k].add(sid)
                succeeded_now[k] += 1
        except Exception as exc:
            tb = traceback.format_exc()
            for k in missing_ks:
                failure = {
                    "ok": False,
                    "sample_id": sid,
                    "qid": sample.get("qid"),
                    "agent_id": sample.get("agent_id"),
                    "query": sample.get("query"),
                    "target": sample.get("target"),
                    "target_context_explanation": sample.get("target_context_explanation"),
                    "component_k": int(k),
                    "error": repr(exc),
                    "traceback": tb,
                    "latency_sec": round(time.time() - sample_started, 4),
                    "finished_at": _now_iso(),
                }
                _append_jsonl(str(paths[k]["failures"]), failure)
                failed_now[k] += 1
            print(f"[BATCH][ERROR] {sid} qid={qid}: {exc!r}", flush=True)
            if bool(args.fail_fast):
                raise

        for k in k_values:
            _write_json(str(paths[k]["summary"]), {
                "started_at": started_at,
                "updated_at": _now_iso(),
                "baseline_name": "Component-Ret",
                "component_k": int(k),
                "experiment_dir": str(paths[k]["exp_dir"].resolve()),
                "manifest_samples": len(manifest),
                "completed_total": len(completed_by_k[k]),
                "succeeded_this_invocation": succeeded_now[k],
                "failed_this_invocation": failed_now[k],
                "results_jsonl": str(paths[k]["results"].resolve()),
                "failures_jsonl": str(paths[k]["failures"].resolve()),
                "sample_manifest_jsonl": str((paths[k]["exp_dir"] / "sample_manifest.jsonl").resolve()),
            })

    summary = {
        "ok": all(failed_now[k] == 0 for k in k_values),
        "baseline_name": "Component-Ret",
        "started_at": started_at,
        "finished_at": _now_iso(),
        "manifest_samples": len(manifest),
        "component_k_values": k_values,
        "succeeded_this_invocation": succeeded_now,
        "failed_this_invocation": failed_now,
        "experiments": {
            str(k): {
                "experiment_dir": str(paths[k]["exp_dir"].resolve()),
                "results_jsonl": str(paths[k]["results"].resolve()),
                "per_sample_dir": str(paths[k]["per_sample"].resolve()),
            }
            for k in k_values
        },
    }
    _write_json(str(Path(args.experiment_root) / "run_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Component-Ret baseline: Top-1 CF LLM + Top-K independent semantic tools."
    )
    ap.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    ap.add_argument("--manifest_file", type=str, default="", help="Optional existing v13 sample_manifest.jsonl to guarantee exactly the same samples.")
    ap.add_argument("--sample_size", type=int, default=100, help="Used only when --manifest_file is not supplied; 0 means all valid rows.")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--experiment_root", type=str, required=True)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--fail_fast", type=int, default=0)

    ap.add_argument("--component_ks", type=str, default="1,2,3,4,5")

    ap.add_argument("--cf_llm_retr_url", type=str, default=os.getenv("CF_LLM_RETR_URL", DEFAULT_CF_LLM_RETR_URL))
    ap.add_argument("--semantic_retr_url", type=str, default=os.getenv("SEMANTIC_RETR_URL", DEFAULT_SEMANTIC_RETR_URL))
    ap.add_argument('--retrieval_timeout', type=int, default=300)

    # Keep upstream retrieval settings aligned with the supplied v13 shell.
    ap.add_argument("--cf_llm_retrieval_depth", type=int, default=10)
    ap.add_argument("--semantic_tool_retrieval_depth", type=int, default=25)
    ap.add_argument("--cf_rewrite", type=int, default=0)
    ap.add_argument("--semantic_rewrite", type=int, default=0)
    ap.add_argument("--agg", choices=["max", "mean", "hybrid"], default="max")
    ap.add_argument("--candidate_multiplier", type=int, default=5)
    ap.add_argument("--include_raw", type=int, default=0)
    ap.add_argument("--include_doc_text", type=int, default=0)
    ap.add_argument("--tool_recall_mode", type=int, default=0)
    ap.add_argument("--per_subquery_top_k", type=int, default=0)

    args = ap.parse_args()
    if int(args.sample_size) < 0:
        raise SystemExit("--sample_size must be >= 0")
    if int(args.cf_llm_retrieval_depth) < 1:
        raise SystemExit("--cf_llm_retrieval_depth must be >= 1")
    if int(args.semantic_tool_retrieval_depth) < 1:
        raise SystemExit("--semantic_tool_retrieval_depth must be >= 1")
    parse_k_values(args.component_ks)
    return args


def main() -> None:
    args = parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
