#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Bundle-Ret baseline for Agent Generative Recommendation.\n\nFORMAT_V3: preserve complete <<...>> tokens; plain tools use uppercase <TOOL_...>.\n\nDefinition\n----------\nFor each query q:\n    m* = top-1 item from the CF LLM retrieval\n    B* = top-1 historical tool bundle from the CF tool-bundle retrieval\n    A  = (m*, B*)\n\nThe tool bundle is kept intact: tools are NOT independently reranked, truncated,\nmerged with lower-ranked bundles, or selected using the gold tool count.\n\nNo generator, beam search, bundle critic, semantic tool retrieval, gold tool-count\noracle, or target field is used to construct the prediction.\n\nOutput layout (aligned with the v13/baseline-1 per-sample schema):\n    <experiment_root>/\n        sample_manifest.jsonl\n        experiment_metadata.json\n        results.jsonl\n        failures.jsonl\n        run_summary.json\n        per_sample/*.json\n\nThe output is intentionally compatible with the same target-only baseline\nevaluation protocol used for Component-Ret.\n'

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_INPUT_JSONL = str(AC_ROOT / 'datasets/generative_v9_sft/sft_valid.jsonl')
DEFAULT_CF_LLM_RETR_URL = "http://127.0.0.1:9000/predict"
DEFAULT_CF_TOOL_BUNDLE_RETR_URL = "http://127.0.0.1:9001/predict"
DEFAULT_AGENT_CATALOG = str(AC_ROOT / 'datasets/PartII/agents/merge.json')

TOOL_SEP_TOKEN = "<TOOL_SEP>"
END_TOKEN = "<SPECIAL_END>"
TOOL_EMPTY_TOKEN = "<TOOL_EMPTY>"

TOOL_TOKEN_RE = re.compile(r"<<[^<>\n\r]+>>")
SINGLE_TOOL_RE = re.compile(r"<TOOL_[^<>\n\r]+>")
ANY_TOOL_TOKEN_RE = re.compile(r"<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>")
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


def _post_cf_bundle(
    url: str,
    query: str,
    topk: int,
    *,
    timeout: int,
    agent_catalog: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call the 9001 CF bundle service with conservative schema fallback.

    The deployed service has historically accepted {query, topk}; older/newer
    copies sometimes use top_k or k.  We try only these equivalent spellings.
    """
    payloads = [
        {"query": query, "topk": int(topk)},
        {"query": query, "top_k": int(topk)},
        {"query": query, "k": int(topk)},
    ]
    errors: List[str] = []
    for payload in payloads:
        try:
            response = _post_json(url, payload, timeout=timeout)
            # Some permissive server versions can return HTTP 200 even when a
            # request spelling was ignored. Accept a payload only if it yields
            # at least one parseable historical bundle.
            parsed = parse_cf_tool_bundles(
                response,
                agent_catalog=agent_catalog,
                limit=int(topk),
            )
            if parsed:
                return response, payload
            errors.append(
                f"{payload}: HTTP 200 but no parseable non-empty bundle; "
                f"response={json.dumps(response, ensure_ascii=False)[:500]}"
            )
        except Exception as exc:
            errors.append(f"{payload}: {exc!r}")
    raise RuntimeError("CF tool-bundle retrieval failed for all supported payload spellings: " + " | ".join(errors))


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
    'Serialize one tool using the datasets/generator token convention.\n\n    Rules:\n      * ``<<...>>`` is already a complete tool token and is preserved exactly.\n      * ``<TOOL_...>`` is already complete and is preserved exactly.\n      * any other non-empty tool name becomes ``<TOOL_{raw}>``.\n      * ``<TOOL_EMPTY>`` is reserved for an empty bundle.\n    '
    raw = str(raw or "").strip()
    if not raw:
        return ""
    if raw == TOOL_EMPTY_TOKEN:
        return raw
    if raw.startswith("<<"):
        return raw
    if raw.startswith("<TOOL_") and raw.endswith(">"):
        return raw
    return f"<TOOL_{raw}>"


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


def _stable_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _normalize_tool(value: Any) -> str:
    """Normalize spacing/quotes without destroying complete tool tokens."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    # Existing serialized tool tokens must survive unchanged.
    if text.startswith("<<"):
        return text
    if text.startswith("<TOOL_") and text.endswith(">"):
        return text
    if text == TOOL_EMPTY_TOKEN:
        return text

    text = CITATION_RE.sub("", text).strip().strip("'").strip('\"').strip()
    text = WS_RE.sub(" ", text)
    return text


def _parse_tool_value(value: Any) -> List[str]:
    """Parse current PartII/CF-bundle tool representations into ordered IDs."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # PartII catalog schema: {"M": ..., "T": {"tools": [...]}, ...}
        if "T" in value and isinstance(value["T"], Mapping):
            nested = _parse_tool_value(value["T"].get("tools"))
            if nested or "tools" in value["T"]:
                return nested
        for key in TOOL_FIELD_KEYS:
            if key in value:
                return _parse_tool_value(value[key])
        for key in ("agent", "configuration", "config", "target_agent", "matched_agent", "candidate"):
            if key in value:
                parsed = _parse_tool_value(value[key])
                if parsed:
                    return parsed
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parsed = _parse_tool_value(item)
                if parsed:
                    out.extend(parsed)
                else:
                    for key in ("name", "tool_name", "token", "id", "key"):
                        if key in item:
                            out.extend(_parse_tool_value(item[key]))
                            break
            else:
                out.extend(_parse_tool_value(item))
        return _stable_unique(out)

    text = str(value).strip()
    if not text:
        return []

    # Preserve already-serialized tool tokens exactly, in their original order.
    # This is the critical rule: <<...>> must never be converted to <TOOL_...>.
    complete_tokens = [m.group(0).strip() for m in ANY_TOOL_TOKEN_RE.finditer(text)]
    if complete_tokens:
        return _stable_unique(complete_tokens)

    if text[:1] in "[{" and text[-1:] in "]}":
        try:
            return _parse_tool_value(json.loads(text))
        except json.JSONDecodeError:
            pass

    tool = _normalize_tool(text)
    return [tool] if tool else []


def _find_result_list(response: Any) -> List[Any]:
    if isinstance(response, list):
        return response
    if not isinstance(response, Mapping):
        return []
    for key in RESULT_CONTAINER_KEYS:
        value = response.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            nested = _find_result_list(value)
            if nested:
                return nested
    numeric = [(int(k), v) for k, v in response.items() if str(k).isdigit()]
    if numeric:
        return [v for _, v in sorted(numeric)]
    return []


def _first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def load_agent_catalog(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"PartII agent catalog is required for ID-only CF bundle responses but was not found: {p}"
        )
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Agent catalog must be a JSON object: {p}")
    out: Dict[str, List[str]] = {}
    for agent_id, agent in raw.items():
        out[str(agent_id)] = _parse_tool_value(agent)
    return out


def parse_cf_tool_bundles(
    response: Dict[str, Any],
    *,
    agent_catalog: Mapping[str, Sequence[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    rows = _find_result_list(response)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows[: max(0, int(limit))], 1):
        tools: List[str] = []
        item_id: Optional[str] = None
        score: Optional[float] = None

        if isinstance(item, str):
            item_id = item.strip()
            if item_id in agent_catalog:
                tools = list(agent_catalog[item_id])
            else:
                tools = _parse_tool_value(item)
        elif isinstance(item, (list, tuple, set)):
            tools = _parse_tool_value(item)
        elif isinstance(item, Mapping):
            item_id_value = _first_nonempty(
                item, ("bundle_id", "agent_id", "item_id", "id", "candidate_id")
            )
            item_id = None if item_id_value is None else str(item_id_value)
            score = _parse_float(_first_nonempty(item, ("score", "similarity", "pred_score", "value")))

            for key in TOOL_FIELD_KEYS:
                if key in item:
                    tools = _parse_tool_value(item[key])
                    if tools or item[key] == []:
                        break
            if not tools:
                for key in ("agent", "matched_agent", "configuration", "config", "candidate"):
                    if key in item:
                        tools = _parse_tool_value(item[key])
                        if tools:
                            break
            if not tools:
                tools = _parse_tool_value(item)
            if not tools and item_id and item_id in agent_catalog:
                tools = list(agent_catalog[item_id])
        else:
            continue

        tools = _stable_unique([t for t in tools if t])
        if not tools:
            continue
        tool_tokens = [wrap_tool_name(t) for t in tools]
        tool_tokens = [t for t in tool_tokens if t]
        if not tool_tokens:
            continue

        out.append({
            "rank": idx,
            "item_id": item_id,
            "score": score,
            "tools": tools,
            "tool_tokens": tool_tokens,
            "num_tools": len(tool_tokens),
            "source": "cf_tool_bundle_9001_predict",
            "raw_item": item,
        })
    return out


def build_strict_text(llm_token: str, tool_tokens: Sequence[str]) -> str:
    llm_token = (llm_token or "").strip()
    if not (llm_token.startswith("<LLM_") and llm_token.endswith(">")):
        raise ValueError(f"Invalid/missing LLM token: {llm_token!r}; refusing to emit an UNK agent")

    tools = [wrap_tool_name(t) for t in tool_tokens if str(t).strip()]
    tools = [t for t in tools if t]
    if not tools:
        tools = [TOOL_EMPTY_TOKEN]

    text = " ".join([llm_token, TOOL_SEP_TOKEN] + tools + [END_TOKEN])
    if "<<<<" in text or ">>>>" in text:
        raise ValueError(f"Malformed nested tool brackets after canonicalization: {text}")
    if not text.endswith(END_TOKEN):
        raise ValueError(f"Malformed agent output (missing {END_TOKEN}): {text}")
    return text


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


def retrieve_bundle_components(
    args: argparse.Namespace,
    query: str,
    agent_catalog: Mapping[str, Sequence[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cf_llm_payload = {
        "query": query,
        "top_k": int(args.cf_llm_retrieval_depth),
        "topk": int(args.cf_llm_retrieval_depth),
        "rewrite": bool(args.cf_rewrite),
        "agg": args.agg,
        "include_raw_agent": bool(args.include_raw),
    }

    started = time.time()
    responses: Dict[str, Dict[str, Any]] = {}
    requests_used: Dict[str, Dict[str, Any]] = {"cf_llm_request": cf_llm_payload}
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_key = {
            executor.submit(
                _post_json,
                args.cf_llm_retr_url,
                cf_llm_payload,
                timeout=int(args.retriever_timeout),
            ): "cf_llm_response",
            executor.submit(
                _post_cf_bundle,
                args.cf_tool_bundle_retr_url,
                query,
                int(args.cf_tool_bundle_retrieval_depth),
                timeout=int(args.retriever_timeout),
                agent_catalog=agent_catalog,
            ): "cf_tool_bundle_response",
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                value = future.result()
                if key == "cf_tool_bundle_response":
                    response, request_payload = value
                    responses[key] = response
                    requests_used["cf_tool_bundle_request"] = request_payload
                else:
                    responses[key] = value
            except Exception as exc:
                errors[key] = repr(exc)

    if errors:
        raise RuntimeError(f"Bundle-Ret retrieval failed: {errors}")

    llms = parse_cf_llm_candidates(
        responses["cf_llm_response"],
        limit=int(args.cf_llm_retrieval_depth),
    )
    bundles = parse_cf_tool_bundles(
        responses["cf_tool_bundle_response"],
        agent_catalog=agent_catalog,
        limit=int(args.cf_tool_bundle_retrieval_depth),
    )
    if not llms:
        raise RuntimeError('CF LLM retrieval returned no usable LLM candidates')
    if not bundles:
        raise RuntimeError(
            'CF tool-bundle retrieval returned no parseable non-empty bundles. '
            "Check the 9001 response schema and PartII agent catalog."
        )

    retrieval_record = {
        "source": "bundle_retrieve_cf_llm_plus_cf_tool_bundle",
        "started_at": _now_iso(),
        "cf_llm_url": args.cf_llm_retr_url,
        "cf_tool_bundle_url": args.cf_tool_bundle_retr_url,
        **requests_used,
        "cf_llm_response": responses["cf_llm_response"],
        "cf_tool_bundle_response": responses["cf_tool_bundle_response"],
        "parsed_cf_llm_candidates": llms,
        "parsed_cf_tool_bundles": bundles,
        "latency_sec": round(time.time() - started, 4),
        "finished_at": _now_iso(),
    }
    return llms, bundles, retrieval_record


def build_record(
    *,
    args: argparse.Namespace,
    sample: Dict[str, Any],
    llms: List[Dict[str, Any]],
    bundles: List[Dict[str, Any]],
    retrieval_record: Dict[str, Any],
    experiment_dir: Path,
    sample_latency_sec: float,
) -> Dict[str, Any]:
    selected_llm = llms[0]
    selected_bundle = bundles[0]
    llm_token = selected_llm["token"]
    tool_tokens = list(selected_bundle["tool_tokens"])
    strict_text = build_strict_text(llm_token, tool_tokens)

    result = {
        "rank": 1,
        "gen_text": strict_text,
        "llm_token": llm_token,
        "tool_tokens": tool_tokens if tool_tokens else [TOOL_EMPTY_TOKEN],
        "strict_text": strict_text,
        "explanation": "",
        "baseline": {
            "name": "Bundle-Ret",
            "selection_rule": "Top-1 CF-retrieved LLM + intact Top-1 CF-retrieved historical tool bundle",
            "llm": selected_llm,
            "tool_bundle": selected_bundle,
            "bundle_rank_used": 1,
            "lower_ranked_bundles_merged": False,
            "within_bundle_tool_reranking": False,
            "gold_tool_count_used": False,
            "note": "The historical bundle is treated as one coherent retrieved item; all tools in rank-1 bundle are retained in returned order.",
        },
    }

    record = {
        "ok": True,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8],
        "created_at": _now_iso(),
        "script": str(Path(__file__).resolve()),
        "query": sample.get("query"),
        "config": {
            "baseline_name": "Bundle-Ret",
            "baseline_category": "retrieval_based_agent_composition",
            "composition_rule": "top1_cf_llm_plus_top1_cf_historical_tool_bundle",
            "bundle_rank_used": 1,
            "gold_tool_count_used": False,
            "cf_llm_retr_url": args.cf_llm_retr_url,
            "cf_tool_bundle_retr_url": args.cf_tool_bundle_retr_url,
            "cf_llm_retrieval_depth": int(args.cf_llm_retrieval_depth),
            "cf_tool_bundle_retrieval_depth": int(args.cf_tool_bundle_retrieval_depth),
            "agent_catalog": str(Path(args.agent_catalog).resolve()),
            "cf_rewrite": bool(args.cf_rewrite),
            "agg": args.agg,
            "tool_sep_token": TOOL_SEP_TOKEN,
            "end_token": END_TOKEN,
            "tool_empty_token": TOOL_EMPTY_TOKEN,
            'semantic_tool_retrieval_used': False,
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
                "num_tool_bundle_candidates": len(bundles),
                "selected_llm": selected_llm,
                "selected_tool_bundle": selected_bundle,
                "selected_tool_count": len(tool_tokens),
            },
        },
        "composition": {
            "llm_selection": selected_llm,
            "tool_bundle_selection": selected_bundle,
            "bundle_rank_used": 1,
            "actual_tool_count": len(tool_tokens),
            "strict_text": strict_text,
        },
        "generation": {
            "skipped": True,
            "reason": "Bundle-Ret is a deterministic retrieval-composition baseline; no generator is used.",
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
) -> Dict[str, Path]:
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    per_sample = root / "per_sample"
    per_sample.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": _now_iso(),
        "baseline_name": "Bundle-Ret",
        "definition": "Top-1 CF LLM + intact Top-1 CF-retrieved historical tool bundle",
        "selection_protocol": (
            "Retrieve ranked LLMs and historical PartII tool bundles independently; "
            "compose only the top-1 LLM with the top-1 bundle. Never merge lower-ranked "
            "bundles, rerank tools within the bundle, or use the gold tool count."
        ),
        "input_jsonl": str(Path(args.input_jsonl).resolve()) if args.input_jsonl else "",
        "reference_manifest": str(Path(args.manifest_file).resolve()) if args.manifest_file else "",
        "sample_size_requested": int(args.sample_size),
        "sample_size_actual": len(manifest),
        "sample_seed": int(args.sample_seed),
        "cf_llm_retrieval_depth": int(args.cf_llm_retrieval_depth),
        "cf_tool_bundle_retrieval_depth": int(args.cf_tool_bundle_retrieval_depth),
        "bundle_rank_used": 1,
        "agent_catalog": str(Path(args.agent_catalog).resolve()),
        "required_evaluation_fields": ["query", "target", "target_context_explanation"],
        "gold_source_for_baseline_evaluation": "dataset_example.target only",
    }
    _write_json(str(root / "experiment_metadata.json"), meta)
    _write_json(str(root / "baseline_metadata.json"), meta)
    _write_jsonl_atomic(str(root / "sample_manifest.jsonl"), manifest)

    return {
        "exp_dir": root,
        "per_sample": per_sample,
        "results": root / "results.jsonl",
        "failures": root / "failures.jsonl",
        "summary": root / "run_summary.json",
    }


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = prepare_manifest(args)
    paths = ensure_experiment_layout(args, manifest)
    agent_catalog = load_agent_catalog(args.agent_catalog)

    completed = _read_successful_sample_ids(str(paths["results"])) if bool(args.resume) else set()
    if not bool(args.resume) and paths["results"].exists() and paths["results"].stat().st_size:
        raise RuntimeError(
            f"{paths['results']} already exists. Use --resume 1 or choose a new --experiment_root."
        )

    pending = [sample for sample in manifest if str(sample["sample_id"]) not in completed]

    print(
        f"[BATCH] Bundle-Ret manifest={len(manifest)} pending={len(pending)} "
        f"rule=Top1-CF-LLM+Top1-CF-Bundle",
        flush=True,
    )

    started_at = _now_iso()
    succeeded_now = 0
    failed_now = 0

    try:
        from tqdm.auto import tqdm
        iterator: Iterable[Dict[str, Any]] = tqdm(
            pending, total=len(pending), desc="Bundle-Ret", unit="sample", dynamic_ncols=True
        )
    except Exception:
        iterator = pending

    for sample in iterator:
        sid = str(sample["sample_id"])
        qid = str(sample.get("qid") or "no_qid")
        safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)[:80]
        sample_started = time.time()
        try:
            llms, bundles, retrieval_record = retrieve_bundle_components(
                args,
                str(sample["query"]),
                agent_catalog=agent_catalog,
            )
            record = build_record(
                args=args,
                sample=sample,
                llms=llms,
                bundles=bundles,
                retrieval_record=retrieval_record,
                experiment_dir=paths["exp_dir"],
                sample_latency_sec=time.time() - sample_started,
            )
            sample_output = paths["per_sample"] / f"{sid}_{safe_qid}.json"
            _write_json(str(sample_output), record)
            _append_jsonl(str(paths["results"]), record)
            completed.add(sid)
            succeeded_now += 1
        except Exception as exc:
            failure = {
                "ok": False,
                "sample_id": sid,
                "qid": sample.get("qid"),
                "agent_id": sample.get("agent_id"),
                "query": sample.get("query"),
                "target": sample.get("target"),
                "target_context_explanation": sample.get("target_context_explanation"),
                "baseline_name": "Bundle-Ret",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "latency_sec": round(time.time() - sample_started, 4),
                "finished_at": _now_iso(),
            }
            _append_jsonl(str(paths["failures"]), failure)
            failed_now += 1
            print(f"[BATCH][ERROR] {sid} qid={qid}: {exc!r}", flush=True)
            if bool(args.fail_fast):
                raise

        _write_json(str(paths["summary"]), {
            "started_at": started_at,
            "updated_at": _now_iso(),
            "baseline_name": "Bundle-Ret",
            "experiment_dir": str(paths["exp_dir"].resolve()),
            "manifest_samples": len(manifest),
            "completed_total": len(completed),
            "succeeded_this_invocation": succeeded_now,
            "failed_this_invocation": failed_now,
            "results_jsonl": str(paths["results"].resolve()),
            "failures_jsonl": str(paths["failures"].resolve()),
            "sample_manifest_jsonl": str((paths["exp_dir"] / "sample_manifest.jsonl").resolve()),
        })

    summary = {
        "ok": failed_now == 0,
        "baseline_name": "Bundle-Ret",
        "started_at": started_at,
        "finished_at": _now_iso(),
        "manifest_samples": len(manifest),
        "completed_total": len(completed),
        "succeeded_this_invocation": succeeded_now,
        "failed_this_invocation": failed_now,
        "experiment_dir": str(paths["exp_dir"].resolve()),
        "results_jsonl": str(paths["results"].resolve()),
        "per_sample_dir": str(paths["per_sample"].resolve()),
    }
    _write_json(str(Path(args.experiment_root) / "run_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bundle-Ret baseline: Top-1 CF LLM + intact Top-1 CF historical tool bundle."
    )
    ap.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    ap.add_argument(
        "--manifest_file",
        type=str,
        default="",
        help="Optional existing sample_manifest.jsonl to guarantee exactly the same queries as another run.",
    )
    ap.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Used only when --manifest_file is not supplied; 0 means all valid rows.",
    )
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--experiment_root", type=str, required=True)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--fail_fast", type=int, default=0)

    ap.add_argument(
        "--cf_llm_retr_url",
        type=str,
        default=os.getenv("CF_LLM_RETR_URL", DEFAULT_CF_LLM_RETR_URL),
    )
    ap.add_argument(
        "--cf_tool_bundle_retr_url",
        type=str,
        default=os.getenv("CF_TOOL_RETR_URL", DEFAULT_CF_TOOL_BUNDLE_RETR_URL),
    )
    ap.add_argument("--agent_catalog", type=str, default=DEFAULT_AGENT_CATALOG)
    ap.add_argument('--retrieval_timeout', type=int, default=300)

    # Aligned with Baseline 1 / the current v13 retrieval setup.
    ap.add_argument("--cf_llm_retrieval_depth", type=int, default=10)
    # v13 uses 5 CF historical bundles in context; Bundle-Ret still predicts only rank 1.
    ap.add_argument("--cf_tool_bundle_retrieval_depth", type=int, default=5)
    ap.add_argument("--cf_rewrite", type=int, default=0)
    ap.add_argument("--agg", choices=["max", "mean", "hybrid"], default="max")
    ap.add_argument("--include_raw", type=int, default=0)

    args = ap.parse_args()
    if int(args.sample_size) < 0:
        raise SystemExit("--sample_size must be >= 0")
    if int(args.cf_llm_retrieval_depth) < 1:
        raise SystemExit("--cf_llm_retrieval_depth must be >= 1")
    if int(args.cf_tool_bundle_retrieval_depth) < 1:
        raise SystemExit("--cf_tool_bundle_retrieval_depth must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()