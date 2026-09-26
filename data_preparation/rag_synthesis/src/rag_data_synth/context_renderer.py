from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .text_utils import as_dict, as_list, clip, fmt_score, join_nonempty
from .token_utils import canonical_llm_token, canonical_tool_token

JsonDict = Dict[str, Any]


def _safe_rank(item: JsonDict, ev: Optional[JsonDict] = None) -> Any:
    ev = ev or {}
    return item.get("rank", ev.get("rank", "NA"))


def _safe_component(item: JsonDict, ev: Optional[JsonDict] = None) -> Any:
    ev = ev or {}
    return item.get("component", ev.get("component", "NA"))


def _optional_score(label: str, x: Any, include_score: bool) -> str:
    if not include_score:
        return ""
    return f" | {label}={fmt_score(x)}"


def _semantic_results_by_type(semantic_response: JsonDict, target: str) -> List[JsonDict]:
    by_type = as_dict(semantic_response.get("results_by_type"))
    block = as_dict(by_type.get(target))
    results = as_list(block.get("results"))
    if results:
        return [x for x in results if isinstance(x, dict)]
    return [x for x in as_list(semantic_response.get("evidence")) if isinstance(x, dict) and x.get("component_type") == target]


def _tool_desc_from_meta(meta: JsonDict, max_chars: int) -> str:
    doc = as_dict(meta.get("documentation"))
    return clip(doc.get("description") or meta.get("description") or meta.get("desc") or meta, max_chars)


def _tool_desc_from_semantic_tool(tool: JsonDict, max_chars: int) -> str:
    return clip(tool.get("description") or tool.get("documentation") or tool.get("desc") or "", max_chars)


def build_compact_retrieval_context(
    data: JsonDict,
    *,
    max_cf_llm: Optional[int] = None,
    max_cf_tool_bundle: Optional[int] = None,
    max_semantic_llm: Optional[int] = None,
    max_semantic_tool: Optional[int] = None,
    max_desc_chars: int = 260,
    include_score: bool = True,
    include_query: bool = False,
) -> str:
    """Build the five-section compact context used by current v11 inference."""
    retrieval = as_dict(data.get("retrieval"))
    evidence_lines: List[str] = []
    next_eid = 1

    def add_evidence(line: str) -> int:
        nonlocal next_eid
        eid = next_eid
        next_eid += 1
        evidence_lines.append(f"[{eid}] {line}")
        return eid

    # 1. CF LLM
    cf_llm_items: List[str] = []
    cf_llm_resp = as_dict(retrieval.get("cf_llm_response"))
    cf_llm_rows = as_list(cf_llm_resp.get("topk")) or as_list(cf_llm_resp.get("results")) or as_list(cf_llm_resp.get("evidence"))
    for item in cf_llm_rows[:max_cf_llm]:
        if not isinstance(item, dict):
            continue
        ev = as_dict(item.get("evidence"))
        llm = as_dict(ev.get("llm")) if ev else as_dict(item.get("llm"))
        name = llm.get("name") or item.get("llm") or item.get("name") or item.get("canonical_id") or _safe_component(item, ev)
        token = canonical_llm_token(str(name or ""))
        if not token:
            continue
        desc = item.get("description") or llm.get("description") or item.get("desc")
        strengths = llm.get("strengths") or item.get("strengths") or []
        srcs = llm.get("source_evidence") or item.get("source_evidence") or []
        raw_score = ev.get("raw_score", item.get("raw_score", item.get("score")))
        norm_score = ev.get("score", item.get("normalized_score", item.get("score")))
        eid = add_evidence(
            "CF-LLM"
            f" | rank={_safe_rank(item, ev)}"
            f"{_optional_score('raw_score', raw_score, include_score)}"
            f"{_optional_score('norm_score', norm_score, include_score)}"
            f" | token={token}"
            f" | item_id={llm.get('item_id') or item.get('llm_item_id') or _safe_component(item, ev)}"
            f" | strengths={join_nonempty(strengths)}"
            f" | sources={join_nonempty(srcs)}"
            f" | desc={clip(desc, max_desc_chars)}"
        )
        cf_llm_items.append(f"{token} [{eid}]")

    # 2. CF Tool bundle
    cf_bundle_items: List[str] = []
    cf_tool_resp = as_dict(retrieval.get("cf_tool_response"))
    cf_tool_rows = as_list(cf_tool_resp.get("results")) or as_list(cf_tool_resp.get("topk")) or as_list(cf_tool_resp.get("evidence"))
    for item in cf_tool_rows[:max_cf_tool_bundle]:
        if not isinstance(item, dict):
            continue
        tool_ids = [canonical_tool_token(str(t.get("id") if isinstance(t, dict) else t)) for t in as_list(item.get("tool_ids"))]
        tool_ids = [t for t in tool_ids if t]
        if not tool_ids:
            # Older single-tool response fallback.
            direct = item.get("tid") or item.get("tool") or item.get("component") or item.get("key") or item.get("token")
            tok = canonical_tool_token(str(direct or ""))
            if tok:
                tool_ids = [tok]
        if not tool_ids:
            continue
        bundle_text = "{" + ", ".join(tool_ids) + "}"
        meta_by_id: Dict[str, JsonDict] = {}
        for meta in as_list(item.get("tools_meta")):
            if not isinstance(meta, dict):
                continue
            doc = as_dict(meta.get("documentation"))
            tid = canonical_tool_token(str(meta.get("id") or doc.get("name") or meta.get("name") or ""))
            if tid:
                meta_by_id[tid] = meta
        tool_desc_budget = max(80, max_desc_chars // 2)
        tool_descs = []
        for tid in tool_ids:
            desc = _tool_desc_from_meta(meta_by_id.get(tid, {}), tool_desc_budget)
            tool_descs.append(f"{tid}: {desc}" if desc else tid)
        ev = as_dict(item.get("evidence"))
        support_examples = as_list(ev.get("support_examples"))
        support = ""
        if support_examples and isinstance(support_examples[0], dict):
            ex = support_examples[0]
            support = (
                f" | support_qid={ex.get('qid', 'NA')}"
                f" | support_rank={ex.get('rank_in_reference', 'NA')}"
                f" | support_question={clip(ex.get('question'), max_desc_chars)}"
            )
        eid = add_evidence(
            "CF-TOOL-BUNDLE"
            f" | rank={item.get('rank', 'NA')}"
            f"{_optional_score('score', item.get('score', item.get('raw_score')), include_score)}"
            f" | matched_agent={item.get('agent_id', item.get('aid', 'NA'))}"
            f" | bundle_size={len(tool_ids)}"
            f"{support}"
            f" | tools=" + "; ".join(tool_descs)
        )
        cf_bundle_items.append(f"{bundle_text} [{eid}]")

    # 3. Semantic LLM
    semantic_llm_items: List[str] = []
    sem_resp = as_dict(retrieval.get("semantic_response"))
    for item in _semantic_results_by_type(sem_resp, "llm")[:max_semantic_llm]:
        llm = as_dict(item.get("llm"))
        name = llm.get("name") or llm.get("canonical_id") or item.get("component")
        token = canonical_llm_token(str(name or ""))
        if not token:
            continue
        desc = llm.get("description") or item.get("description")
        strengths = llm.get("strengths") or []
        srcs = llm.get("source_evidence") or ([item.get("score_source")] if item.get("score_source") else [])
        eid = add_evidence(
            "SEMANTIC-LLM"
            f" | rank={item.get('rank', 'NA')}"
            f"{_optional_score('raw_score', item.get('raw_score'), include_score)}"
            f"{_optional_score('norm_score', item.get('score'), include_score)}"
            f" | token={token}"
            f" | item_id={llm.get('item_id') or item.get('component', 'NA')}"
            f" | matched_subquery={clip(item.get('matched_subquery'), max_desc_chars)}"
            f" | strengths={join_nonempty(strengths)}"
            f" | sources={join_nonempty(srcs)}"
            f" | desc={clip(desc, max_desc_chars)}"
        )
        semantic_llm_items.append(f"{token} [{eid}]")

    # 4. Semantic single tools
    semantic_tool_items: List[str] = []
    for item in _semantic_results_by_type(sem_resp, "tool")[:max_semantic_tool]:
        tool = as_dict(item.get("tool"))
        token = canonical_tool_token(str(tool.get("key") or tool.get("item_id") or item.get("component") or ""))
        if not token:
            continue
        desc = _tool_desc_from_semantic_tool(tool, max_desc_chars)
        matched_subqueries = as_list(item.get("matched_subqueries"))
        matched_text = join_nonempty(matched_subqueries) if matched_subqueries else clip(item.get("matched_subquery"), max_desc_chars)
        eid = add_evidence(
            "SEMANTIC-TOOL"
            f" | rank={item.get('rank', 'NA')}"
            f"{_optional_score('raw_score', item.get('raw_score'), include_score)}"
            f"{_optional_score('norm_score', item.get('score'), include_score)}"
            f" | token={token}"
            f" | matched_subquery={clip(matched_text, max_desc_chars)}"
            f" | desc={desc}"
        )
        semantic_tool_items.append(f"{token} [{eid}]")

    sections: List[Tuple[str, str]] = []
    if include_query:
        sections.append(("query", clip(data.get("query"), 1200)))
    sections.extend([
        ("cf_retrieved_llm", join_nonempty(cf_llm_items)),
        ("cf_retrieved_tool_bundle", join_nonempty(cf_bundle_items)),
        ("semantic_retrieved_llm", join_nonempty(semantic_llm_items)),
        ("semantic_retrieved_tool", join_nonempty(semantic_tool_items)),
        ("evidence", "\n  ".join(evidence_lines) if evidence_lines else "NONE"),
    ])
    return "\n\n".join(f"{name}:\n  {content}" for name, content in sections)
