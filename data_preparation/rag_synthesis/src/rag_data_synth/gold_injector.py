from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .meta_client import MetadataClient
from .retriever_client import RetrievalConfig, call_score
from .text_utils import as_dict, as_list, textify
from .token_utils import GoldTarget, canonical_llm_token, canonical_tool_token, token_key

JsonDict = Dict[str, Any]


@dataclass
class InjectionReport:
    gold_llm: str
    gold_tools: List[str]
    covered_llm: bool
    covered_tools: List[str]
    missing_llm: str
    missing_tools: List[str]
    injected_llm: bool
    injected_tools: List[str]


def _pick_rows(obj: JsonDict, keys: List[str]) -> List[Any]:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return []


def _llm_token_from_cf_item(item: JsonDict) -> str:
    ev = as_dict(item.get("evidence"))
    llm = as_dict(ev.get("llm"))
    name = llm.get("name") or item.get("llm") or item.get("name") or item.get("canonical_id") or ev.get("component") or item.get("component")
    return canonical_llm_token(str(name or ""))


def _llm_token_from_sem_item(item: JsonDict) -> str:
    llm = as_dict(item.get("llm"))
    name = llm.get("name") or llm.get("canonical_id") or item.get("component") or item.get("llm")
    return canonical_llm_token(str(name or ""))


def _tool_tokens_from_cf_bundle(item: JsonDict) -> List[str]:
    toks: List[str] = []
    for tid in as_list(item.get("tool_ids")):
        if isinstance(tid, dict):
            tid = tid.get("id") or tid.get("name") or tid.get("tool_id") or tid.get("tid")
        tok = canonical_tool_token(str(tid or ""))
        if tok:
            toks.append(tok)
    # gen_text fallback is intentionally omitted here; compact renderer also uses tool_ids.
    return toks


def _tool_token_from_sem_item(item: JsonDict) -> str:
    tool = as_dict(item.get("tool"))
    key = tool.get("key") or tool.get("item_id") or item.get("component") or item.get("tid") or item.get("tool")
    return canonical_tool_token(str(key or ""))


def collect_candidate_keys(retrieval: JsonDict) -> Tuple[set, set]:
    llm_keys = set()
    tool_keys = set()

    cf_llm_resp = as_dict(retrieval.get("cf_llm_response"))
    for item in _pick_rows(cf_llm_resp, ["topk", "results", "evidence"]):
        if isinstance(item, dict):
            tok = _llm_token_from_cf_item(item)
            if tok:
                llm_keys.add(token_key(tok))

    sem_resp = as_dict(retrieval.get("semantic_response"))
    by_type = as_dict(sem_resp.get("results_by_type"))
    sem_llm_rows = as_list(as_dict(by_type.get("llm")).get("results"))
    if not sem_llm_rows:
        sem_llm_rows = [x for x in as_list(sem_resp.get("evidence")) if isinstance(x, dict) and x.get("component_type") == "llm"]
    for item in sem_llm_rows:
        if isinstance(item, dict):
            tok = _llm_token_from_sem_item(item)
            if tok:
                llm_keys.add(token_key(tok))

    cf_tool_resp = as_dict(retrieval.get("cf_tool_response"))
    for item in _pick_rows(cf_tool_resp, ["results", "topk", "evidence"]):
        if isinstance(item, dict):
            for tok in _tool_tokens_from_cf_bundle(item):
                tool_keys.add(token_key(tok))

    sem_tool_rows = as_list(as_dict(by_type.get("tool")).get("results"))
    if not sem_tool_rows:
        sem_tool_rows = [x for x in as_list(sem_resp.get("evidence")) if isinstance(x, dict) and x.get("component_type") == "tool"]
    for item in sem_tool_rows:
        if isinstance(item, dict):
            tok = _tool_token_from_sem_item(item)
            if tok:
                tool_keys.add(token_key(tok))

    return llm_keys, tool_keys


def _ensure_semantic_shape(retrieval: JsonDict) -> JsonDict:
    sem = retrieval.setdefault("semantic_response", {})
    if not isinstance(sem, dict):
        sem = {}
        retrieval["semantic_response"] = sem
    by_type = sem.setdefault("results_by_type", {})
    if not isinstance(by_type, dict):
        by_type = {}
        sem["results_by_type"] = by_type
    for t in ["llm", "tool"]:
        group = by_type.setdefault(t, {})
        if not isinstance(group, dict):
            group = {}
            by_type[t] = group
        group.setdefault("results", [])
        if not isinstance(group["results"], list):
            group["results"] = []
    return sem


def _rerank_and_clip_semantic(results: List[JsonDict], *, max_keep: int, gold_keys: set, component_type: str) -> None:
    def get_tok(item: JsonDict) -> str:
        return _llm_token_from_sem_item(item) if component_type == "llm" else _tool_token_from_sem_item(item)

    # Dedup by loose key, keep highest score.
    best: Dict[str, JsonDict] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        key = token_key(get_tok(item))
        if not key:
            continue
        cur = best.get(key)
        if cur is None or float(item.get("score", item.get("raw_score", -1e30)) or -1e30) > float(cur.get("score", cur.get("raw_score", -1e30)) or -1e30):
            best[key] = item
    ordered = list(best.values())
    ordered.sort(key=lambda x: (float(x.get("score", x.get("raw_score", -1e30)) or -1e30), -int(x.get("rank", 10**9) or 10**9)), reverse=True)

    if max_keep and len(ordered) > max_keep:
        kept = ordered[:max_keep]
        kept_keys = {token_key(get_tok(x)) for x in kept}
        forced = [x for x in ordered[max_keep:] if token_key(get_tok(x)) in gold_keys and token_key(get_tok(x)) not in kept_keys]
        for f in forced:
            if kept:
                # Replace lowest non-gold if possible; otherwise append and clip.
                repl = None
                for i in range(len(kept) - 1, -1, -1):
                    if token_key(get_tok(kept[i])) not in gold_keys:
                        repl = i
                        break
                if repl is not None:
                    kept[repl] = f
                else:
                    kept.append(f)
            else:
                kept.append(f)
        ordered = kept[:max_keep]

    # Reassign display ranks after injection/rerank.
    for i, item in enumerate(ordered, start=1):
        item["rank"] = i
    results[:] = ordered


def ensure_gold_coverage(
    retrieval: JsonDict,
    gold: GoldTarget,
    *,
    query: str,
    cfg: RetrievalConfig,
    meta_client: MetadataClient,
    fallback_score: float = 0.55,
    allow_score_fallback: bool = True,
    semantic_llm_keep: int = 8,
    semantic_tool_keep: int = 12,
) -> InjectionReport:
    llm_keys, tool_keys = collect_candidate_keys(retrieval)
    gold_llm = canonical_llm_token(gold.llm_token)
    gold_tools = [canonical_tool_token(t) for t in gold.tool_tokens if canonical_tool_token(t)]

    covered_llm = bool(gold_llm and token_key(gold_llm) in llm_keys)
    covered_tools = [t for t in gold_tools if token_key(t) in tool_keys]
    missing_llm = "" if covered_llm else gold_llm
    missing_tools = [t for t in gold_tools if token_key(t) not in tool_keys]

    sem = _ensure_semantic_shape(retrieval)
    by_type = sem["results_by_type"]
    injected_tools: List[str] = []
    injected_llm = False

    if missing_llm:
        sc = call_score(
            score_url=cfg.score_url,
            query=query,
            component_type="llm",
            component=missing_llm,
            timeout=min(cfg.timeout, 60),
            fallback_score=fallback_score,
            allow_fallback=allow_score_fallback,
        )
        desc = meta_client.get_desc(missing_llm, "llm") or "Gold LLM injected because retrieval did not cover the supervised target."
        by_type["llm"]["results"].append({
            "rank": 10**6,
            "component_type": "llm",
            "component": missing_llm,
            "raw_score": sc["raw_score"],
            "score": sc["score"],
            "matched_subquery": "gold component forced by supervision",
            "llm": {
                "name": missing_llm[len("<LLM_"):-1] if missing_llm.startswith("<LLM_") else missing_llm,
                "item_id": "GOLD_INJECTED",
                "description": desc,
                "strengths": ["gold supervision target"],
                "source_evidence": [sc.get("score_source", "compute_score")],
            },
            "gold_injected": True,
            "score_source": sc.get("score_source"),
            "score_error": sc.get("error", ""),
        })
        injected_llm = True

    for tool_tok in missing_tools:
        sc = call_score(
            score_url=cfg.score_url,
            query=query,
            component_type="tool",
            component=tool_tok,
            timeout=min(cfg.timeout, 60),
            fallback_score=fallback_score,
            allow_fallback=allow_score_fallback,
        )
        desc = meta_client.get_desc(tool_tok, "tool") or "Gold tool injected because retrieval did not cover the supervised target."
        by_type["tool"]["results"].append({
            "rank": 10**6,
            "component_type": "tool",
            "component": tool_tok,
            "raw_score": sc["raw_score"],
            "score": sc["score"],
            "matched_subquery": "gold component forced by supervision",
            "matched_subqueries": ["gold component forced by supervision"],
            "tool": {
                "key": tool_tok,
                "item_id": tool_tok,
                "description": desc,
            },
            "gold_injected": True,
            "score_source": sc.get("score_source"),
            "score_error": sc.get("error", ""),
        })
        injected_tools.append(tool_tok)

    _rerank_and_clip_semantic(
        by_type["llm"]["results"],
        max_keep=max(0, int(semantic_llm_keep)),
        gold_keys={token_key(gold_llm)} if gold_llm else set(),
        component_type="llm",
    )
    _rerank_and_clip_semantic(
        by_type["tool"]["results"],
        max_keep=max(0, int(semantic_tool_keep)),
        gold_keys={token_key(t) for t in gold_tools},
        component_type="tool",
    )

    return InjectionReport(
        gold_llm=gold_llm,
        gold_tools=gold_tools,
        covered_llm=covered_llm,
        covered_tools=covered_tools,
        missing_llm=missing_llm,
        missing_tools=missing_tools,
        injected_llm=injected_llm,
        injected_tools=injected_tools,
    )
