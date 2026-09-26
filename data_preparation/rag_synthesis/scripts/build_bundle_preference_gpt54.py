#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else []


DEFAULT_INPUT = "pair_rag.context_target_rationales.v2.jsonl"
DEFAULT_OUTPUT = "bundle_preference_data.gpt54.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

LLM_RE = re.compile(r"<LLM_[^>\s]+>")
TOOL_RE = re.compile(r"<<[^<>\n]+?&&[^<>\n]+?>>")
BUNDLE_RE = re.compile(r"\{([^{}]+)\}\s*\[(\d+)\]")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build GPT-5.4 teacher-ranked bundle preference data from "
            "pair_rag.context_target_rationales.v2.jsonl."
        )
    )
    ap.add_argument("--input", default=os.getenv("INPUT_JSONL", DEFAULT_INPUT))
    ap.add_argument("--output", default=os.getenv("OUTPUT_JSONL", DEFAULT_OUTPUT))
    ap.add_argument("--error_output", default=os.getenv("ERROR_JSONL", ""))
    ap.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL") or os.getenv("QUERY_REWRITE_MODEL") or DEFAULT_MODEL,
    )
    ap.add_argument("--max_qids", type=int, default=int(os.getenv("MAX_QIDS", "0")))
    ap.add_argument("--max_context_chars", type=int, default=int(os.getenv("MAX_CONTEXT_CHARS", "12000")))
    ap.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "8")))
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    ap.add_argument("--sleep", type=float, default=float(os.getenv("SLEEP_SEC", "0.0")))
    ap.add_argument("--dry_run", action="store_true", help="Parse/filter only; do not call GPT.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming.")
    return ap.parse_args()


def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            yield 1, data
            return
        if isinstance(data, list):
            for idx, obj in enumerate(data, 1):
                if isinstance(obj, dict):
                    yield idx, obj
            return

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[bad-json] {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield line_no, obj


def append_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_records(path: str) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return 1
        if isinstance(data, list):
            return sum(1 for x in data if isinstance(x, dict))
        return None
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_resume_state(path: str) -> Tuple[set[str], Dict[str, set[int]], Dict[str, int]]:
    counts: Dict[str, int] = {}
    variants: Dict[str, set[int]] = {}
    if not path or not os.path.exists(path):
        return set(), variants, counts
    for _, obj in read_jsonl(path):
        qid = str(obj.get("qid", "") or "")
        if qid:
            counts[qid] = counts.get(qid, 0) + 1
            try:
                variants.setdefault(qid, set()).add(int(obj.get("variant_id")))
            except Exception:
                pass
    completed = {qid for qid, seen in variants.items() if {0, 1, 2}.issubset(seen)}
    completed.update(qid for qid, n in counts.items() if n >= 3 and qid not in variants)
    return completed, variants, counts


def write_error(
    path: str,
    *,
    obj: Dict[str, Any],
    line_no: int,
    stage: str,
    exc: BaseException,
    model: str,
    base_url: str,
) -> None:
    row = {
        "qid": str(obj.get("qid", "") or ""),
        "agent_id": str(obj.get("agent_id", "") or ""),
        "part": str(obj.get("part", "") or ""),
        "source_line_no": line_no,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc(limit=6),
        "query": str(obj.get("query", "") or ""),
        "model": model,
        "base_url": base_url,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_jsonl(path, [row])


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.68)
    tail = max_chars - head - 80
    return text[:head] + "\n\n...[TRUNCATED]...\n\n" + text[-max(0, tail):]


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = str(item or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_target_tools(target: str) -> List[str]:
    if "<TOOL_EMPTY>" in (target or ""):
        return []
    return unique_keep_order(TOOL_RE.findall(target or ""))


def parse_context(context: str, gold_llm: str, gold_tools: Sequence[str]) -> Dict[str, Any]:
    llms = unique_keep_order(LLM_RE.findall(context or ""))
    tools = unique_keep_order(TOOL_RE.findall(context or ""))
    bundles: List[Dict[str, Any]] = []
    for match in BUNDLE_RE.finditer(context or ""):
        bundle_tools = unique_keep_order(TOOL_RE.findall(match.group(1)))
        if bundle_tools:
            bundles.append({"rank": int(match.group(2)), "tools": bundle_tools})

    if gold_llm and gold_llm not in llms:
        llms.insert(0, gold_llm)
    for tool in gold_tools:
        if tool not in tools:
            tools.insert(0, tool)

    llm_ranks = {llm: i + 1 for i, llm in enumerate(llms)}
    return {
        "retrieved_llms": llms,
        "retrieved_tools": tools,
        "retrieved_tool_bundles": bundles[:30],
        "llm_ranks": llm_ranks,
    }


def is_eligible(obj: Dict[str, Any]) -> Tuple[bool, str, str, List[str]]:
    meta = obj.get("rag_meta") or {}
    gold_llm = str(meta.get("gold_llm", "") or "").strip()
    gold_tools = meta.get("gold_tools")
    if not isinstance(gold_tools, list):
        gold_tools = parse_target_tools(str(obj.get("target", "") or ""))
    gold_tools = unique_keep_order(str(x) for x in gold_tools)
    if not gold_llm:
        return False, "missing_gold_llm", gold_llm, gold_tools
    if len(gold_tools) < 3:
        return False, "gold_tools_lt_3", gold_llm, gold_tools
    return True, "", gold_llm, gold_tools


def strip_json_markdown(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GPTClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def json_chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        retries: int = 3,
    ) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                }
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return json.loads(strip_json_markdown(content))
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(2 * attempt, 8))
        raise RuntimeError(f"GPT JSON call failed after {retries} retries: {last_error}")


TEACHER_SYSTEM = """You are a careful data-augmentation teacher for AgentSelect.
You build preference data for a discriminative bundle critic.
Use only the provided retrieved LLM/tool candidates and evidence. Do not invent model names or tools.
Prefer fine-grained, plausible negatives over random negatives.
Return strict JSON only."""


def build_variant_prompt(
    *,
    query: str,
    gold_llm: str,
    gold_tools: Sequence[str],
    parsed: Dict[str, Any],
    context: str,
) -> str:
    return f"""Task: generate exactly two query variants for preference-data augmentation.

The variants should be close to the original query but change the task intent enough that the best bundle may slightly change. For each variant, select a gold bundle from the retrieved candidates. The selected LLM and tools must come from the allowed lists below.

Rules:
- Do not invent tools or LLMs.
- Prefer at least 3 tools when the original task is tool-heavy.
- The selected gold can keep the original LLM/tools if the retrieved context does not support a better change.
- Make the variants realistic user queries, not labels or explanations.

Original query:
{query}

Original gold:
LLM: {gold_llm}
Tools: {json.dumps(list(gold_tools), ensure_ascii=False)}

Allowed LLMs with retrieved rank:
{json.dumps(parsed["llm_ranks"], ensure_ascii=False)}

Allowed tools:
{json.dumps(parsed["retrieved_tools"][:80], ensure_ascii=False)}

Retrieved tool bundles:
{json.dumps(parsed["retrieved_tool_bundles"][:20], ensure_ascii=False)}

Evidence context excerpt:
{context}

Return JSON:
{{
  "variants": [
    {{
      "query": "...",
      "gold_llm": "<LLM_...>",
      "gold_tools": ["<<...&&...>>"],
      "change_type": "more_specific|broader|extra_constraint|multi_step|tool_shift",
      "reason": "short grounded reason"
    }}
  ]
}}"""


def build_candidate_prompt(
    *,
    query: str,
    gold_llm: str,
    gold_tools: Sequence[str],
    parsed: Dict[str, Any],
    context: str,
) -> str:
    gold_rank = parsed["llm_ranks"].get(gold_llm, 999)
    weaker = [
        {"llm": llm, "rank": rank}
        for llm, rank in parsed["llm_ranks"].items()
        if rank >= gold_rank + 5
    ]
    if not weaker:
        weaker = [
            {"llm": llm, "rank": rank}
            for llm, rank in parsed["llm_ranks"].items()
            if llm != gold_llm
        ][-5:]

    return f"""Task: construct a candidate bundle pool for one query.

Create 6 to 8 candidates. A1 must be the exact gold bundle. The remaining candidates must be fine-grained negatives using these perturbation types:
- remove_critical_tool
- add_irrelevant_or_redundant_tool
- swap_llm_weaker_rank_gap_ge_5
- correct_llm_wrong_tools
- swap_correct_tool_with_wrong_tool
- redundant_or_overcomplete_bundle

Rules:
- Use only allowed LLMs and tools.
- For the weaker LLM candidate, prefer an LLM at least 5 retrieved ranks worse than the gold LLM when available.
- Keep candidates plausible enough that ranking requires evidence-grounded judgment.
- Avoid random nonsense bundles.
- Candidate ids must be A1, A2, ...

Query:
{query}

Gold bundle:
LLM: {gold_llm}
Tools: {json.dumps(list(gold_tools), ensure_ascii=False)}
Gold LLM rank: {gold_rank}

Allowed LLM ranks:
{json.dumps(parsed["llm_ranks"], ensure_ascii=False)}

Preferred weaker LLM options:
{json.dumps(weaker, ensure_ascii=False)}

Allowed tools:
{json.dumps(parsed["retrieved_tools"][:100], ensure_ascii=False)}

Retrieved tool bundles:
{json.dumps(parsed["retrieved_tool_bundles"][:25], ensure_ascii=False)}

Evidence context excerpt:
{context}

Return JSON:
{{
  "candidates": [
    {{
      "id": "A1",
      "llm": "{gold_llm}",
      "tools": {json.dumps(list(gold_tools), ensure_ascii=False)},
      "perturbation": "gold",
      "reason": "short reason"
    }}
  ],
  "pool_reason": "one short sentence"
}}"""


def build_ranking_prompt(*, query: str, candidates: Sequence[Dict[str, Any]], context: str) -> str:
    return f"""Task: rank candidate agent bundles from best to worst for the query.

Judge by query relevance, capability coverage, component complementarity, evidence support, and lack of redundancy. The gold candidate is usually best, but still rank by the evidence and the actual query intent.

Query:
{query}

Candidates:
{json.dumps(list(candidates), ensure_ascii=False)}

Evidence context excerpt:
{context}

Return JSON:
{{
  "ranking": ["A1", "A2"],
  "reason": "1-3 concise sentences explaining why the top candidate wins and why the main negatives are worse.",
  "scores": {{"A1": 1.0, "A2": 0.5}}
}}"""


def sanitize_gold(
    item: Dict[str, Any],
    *,
    fallback_llm: str,
    fallback_tools: Sequence[str],
    parsed: Dict[str, Any],
) -> Tuple[str, List[str]]:
    llm = str(item.get("gold_llm", "") or "").strip()
    if llm not in parsed["retrieved_llms"]:
        llm = fallback_llm
    allowed_tools = set(parsed["retrieved_tools"])
    tools = unique_keep_order(str(x) for x in item.get("gold_tools", []) if str(x) in allowed_tools)
    if len(tools) < 3:
        tools = list(fallback_tools)
    return llm, tools


def make_fallback_variants(query: str, gold_llm: str, gold_tools: Sequence[str]) -> List[Dict[str, Any]]:
    return [
        {
            "query": f"{query} Please solve it with any necessary supporting tool calls.",
            "gold_llm": gold_llm,
            "gold_tools": list(gold_tools),
            "change_type": "extra_constraint",
            "reason": "Fallback variant preserving the original grounded gold bundle.",
        },
        {
            "query": f"{query} Also explain which external information or computation each selected tool supports.",
            "gold_llm": gold_llm,
            "gold_tools": list(gold_tools),
            "change_type": "multi_step",
            "reason": "Fallback variant preserving the original grounded gold bundle.",
        },
    ]


def choose_wrong_tools(gold_tools: Sequence[str], parsed: Dict[str, Any], n: int) -> List[str]:
    gold_set = set(gold_tools)
    for bundle in parsed.get("retrieved_tool_bundles", []):
        tools = [t for t in bundle.get("tools", []) if t not in gold_set]
        if len(tools) >= n:
            return tools[:n]
    return [t for t in parsed["retrieved_tools"] if t not in gold_set][:n]


def choose_weaker_llm(gold_llm: str, parsed: Dict[str, Any]) -> str:
    ranks = parsed["llm_ranks"]
    gold_rank = ranks.get(gold_llm, 999)
    weaker = [llm for llm, rank in ranks.items() if rank >= gold_rank + 5]
    if weaker:
        return weaker[0]
    for llm in reversed(parsed["retrieved_llms"]):
        if llm != gold_llm:
            return llm
    return gold_llm


def fallback_candidates(gold_llm: str, gold_tools: Sequence[str], parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = list(gold_tools)
    wrong = choose_wrong_tools(tools, parsed, max(3, len(tools)))
    extra = wrong[0] if wrong else (parsed["retrieved_tools"][0] if parsed["retrieved_tools"] else "")
    weaker_llm = choose_weaker_llm(gold_llm, parsed)
    rows = [
        {"id": "A1", "llm": gold_llm, "tools": tools, "perturbation": "gold", "reason": "Gold bundle."},
        {
            "id": "A2",
            "llm": gold_llm,
            "tools": tools[1:],
            "perturbation": "remove_critical_tool",
            "reason": "Removes one gold tool.",
        },
        {
            "id": "A3",
            "llm": gold_llm,
            "tools": unique_keep_order(tools + ([extra] if extra else [])),
            "perturbation": "add_irrelevant_or_redundant_tool",
            "reason": "Adds an extra retrieved tool.",
        },
        {
            "id": "A4",
            "llm": weaker_llm,
            "tools": tools,
            "perturbation": "swap_llm_weaker_rank_gap_ge_5",
            "reason": "Keeps tools but swaps the LLM.",
        },
        {
            "id": "A5",
            "llm": gold_llm,
            "tools": wrong[: len(tools)] if wrong else tools[:1],
            "perturbation": "correct_llm_wrong_tools",
            "reason": "Keeps the LLM but uses wrong tools.",
        },
        {
            "id": "A6",
            "llm": gold_llm,
            "tools": unique_keep_order((tools[:-1] or tools) + wrong[:1]),
            "perturbation": "swap_correct_tool_with_wrong_tool",
            "reason": "Swaps one gold tool with a wrong retrieved tool.",
        },
    ]
    return [row for row in rows if row["llm"] and row["tools"]]


def sanitize_candidates(
    raw: Dict[str, Any],
    *,
    gold_llm: str,
    gold_tools: Sequence[str],
    parsed: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    allowed_llms = set(parsed["retrieved_llms"])
    allowed_tools = set(parsed["retrieved_tools"])
    rows: List[Dict[str, Any]] = []
    seen = set()

    raw_candidates = raw.get("candidates", []) if isinstance(raw, dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    gold = {
        "id": "A1",
        "llm": gold_llm,
        "tools": list(gold_tools),
        "perturbation": "gold",
        "reason": "Exact grounded gold bundle.",
    }

    def add_candidate(candidate: Dict[str, Any]) -> None:
        llm = str(candidate.get("llm", "") or "").strip()
        if llm not in allowed_llms:
            return
        tools = unique_keep_order(str(x) for x in candidate.get("tools", []) if str(x) in allowed_tools)
        key = (llm, tuple(tools))
        if not tools or key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "id": f"A{len(rows) + 1}",
                "llm": llm,
                "tools": tools,
                "perturbation": str(candidate.get("perturbation", "") or "unknown"),
                "reason": str(candidate.get("reason", "") or "").strip(),
            }
        )

    add_candidate(gold)
    for candidate in raw_candidates:
        if isinstance(candidate, dict):
            add_candidate(candidate)

    if len(rows) < 6:
        for candidate in fallback_candidates(gold_llm, gold_tools, parsed):
            add_candidate(candidate)

    for i, row in enumerate(rows, 1):
        row["id"] = f"A{i}"
    return rows[:8], str(raw.get("pool_reason", "") if isinstance(raw, dict) else "").strip()


def sanitize_ranking(raw: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Tuple[List[str], str, Dict[str, float]]:
    ids = [c["id"] for c in candidates]
    id_set = set(ids)
    ranking = raw.get("ranking", []) if isinstance(raw, dict) else []
    if not isinstance(ranking, list):
        ranking = []
    clean = [str(x) for x in ranking if str(x) in id_set]
    clean = unique_keep_order(clean)
    for cid in ids:
        if cid not in clean:
            clean.append(cid)
    reason = str(raw.get("reason", "") if isinstance(raw, dict) else "").strip()
    scores_raw = raw.get("scores", {}) if isinstance(raw, dict) else {}
    scores: Dict[str, float] = {}
    if isinstance(scores_raw, dict):
        for cid, val in scores_raw.items():
            if str(cid) in id_set:
                try:
                    scores[str(cid)] = float(val)
                except Exception:
                    pass
    return clean, reason, scores


@dataclass
class QueryCase:
    variant_id: int
    query: str
    gold_llm: str
    gold_tools: List[str]
    variant_type: str
    variant_reason: str


def build_query_cases(
    *,
    client: Optional[GPTClient],
    dry_run: bool,
    original_query: str,
    gold_llm: str,
    gold_tools: Sequence[str],
    parsed: Dict[str, Any],
    context: str,
) -> Tuple[List[QueryCase], Dict[str, Any]]:
    cases = [
        QueryCase(0, original_query, gold_llm, list(gold_tools), "original", "Original query and gold bundle.")
    ]
    meta: Dict[str, Any] = {"variant_source": "gpt"}
    if dry_run or client is None:
        variants = make_fallback_variants(original_query, gold_llm, gold_tools)
        meta["variant_source"] = "fallback_dry_run"
    else:
        raw = client.json_chat(
            system=TEACHER_SYSTEM,
            user=build_variant_prompt(
                query=original_query,
                gold_llm=gold_llm,
                gold_tools=gold_tools,
                parsed=parsed,
                context=context,
            ),
            temperature=0.35,
        )
        variants = raw.get("variants", [])
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError("variant response missing two variants")
        meta["variant_raw"] = raw

    for idx, item in enumerate(variants[:2], 1):
        if not isinstance(item, dict):
            continue
        vq = str(item.get("query", "") or "").strip()
        if not vq:
            continue
        v_llm, v_tools = sanitize_gold(
            item,
            fallback_llm=gold_llm,
            fallback_tools=gold_tools,
            parsed=parsed,
        )
        cases.append(
            QueryCase(
                idx,
                vq,
                v_llm,
                v_tools,
                str(item.get("change_type", "") or "variant"),
                str(item.get("reason", "") or "").strip(),
            )
        )

    while len(cases) < 3:
        if not dry_run:
            raise ValueError("could not build three query cases from teacher variants")
        fb = make_fallback_variants(original_query, gold_llm, gold_tools)[len(cases) - 1]
        cases.append(
            QueryCase(
                len(cases),
                fb["query"],
                fb["gold_llm"],
                list(fb["gold_tools"]),
                fb["change_type"],
                fb["reason"],
            )
        )
    return cases[:3], meta


def build_preference_row(
    *,
    client: Optional[GPTClient],
    dry_run: bool,
    case: QueryCase,
    obj: Dict[str, Any],
    source_line_no: int,
    parsed: Dict[str, Any],
    context: str,
    model: str,
    base_url: str,
) -> Dict[str, Any]:
    candidate_source = "gpt"
    ranking_source = "gpt"
    candidate_error = ""
    ranking_error = ""

    if dry_run or client is None:
        candidates, pool_reason = sanitize_candidates(
            {"candidates": fallback_candidates(case.gold_llm, case.gold_tools, parsed), "pool_reason": "dry-run fallback"},
            gold_llm=case.gold_llm,
            gold_tools=case.gold_tools,
            parsed=parsed,
        )
        candidate_source = "fallback_dry_run"
        ranking, rank_reason, scores = sanitize_ranking(
            {"ranking": [c["id"] for c in candidates], "reason": "dry-run fallback ranking", "scores": {}},
            candidates,
        )
        ranking_source = "fallback_dry_run"
    else:
        raw_candidates = client.json_chat(
            system=TEACHER_SYSTEM,
            user=build_candidate_prompt(
                query=case.query,
                gold_llm=case.gold_llm,
                gold_tools=case.gold_tools,
                parsed=parsed,
                context=context,
            ),
            temperature=0.25,
        )
        candidates, pool_reason = sanitize_candidates(
            raw_candidates,
            gold_llm=case.gold_llm,
            gold_tools=case.gold_tools,
            parsed=parsed,
        )
        if len(candidates) < 3:
            raise ValueError(f"candidate pool too small after sanitization: {len(candidates)}")

        raw_ranking = client.json_chat(
            system=TEACHER_SYSTEM,
            user=build_ranking_prompt(query=case.query, candidates=candidates, context=context),
            temperature=0,
        )
        ranking, rank_reason, scores = sanitize_ranking(raw_ranking, candidates)
        if not rank_reason:
            raise ValueError("ranking response missing reason")

    return {
        "qid": str(obj.get("qid", "") or ""),
        "agent_id": str(obj.get("agent_id", "") or ""),
        "part": str(obj.get("part", "") or ""),
        "source_line_no": source_line_no,
        "variant_id": case.variant_id,
        "variant_type": case.variant_type,
        "original_query": str(obj.get("query", "") or ""),
        "query": case.query,
        "evidence_context": context,
        "gold": {"llm": case.gold_llm, "tools": case.gold_tools},
        "candidates": candidates,
        "ranking": ranking,
        "reason": rank_reason,
        "scores": scores,
        "candidate_pool_reason": pool_reason,
        "variant_reason": case.variant_reason,
        "teacher_meta": {
            "model": model,
            "base_url": base_url,
            "candidate_source": candidate_source,
            "ranking_source": ranking_source,
            "candidate_error": candidate_error,
            "ranking_error": ranking_error,
        },
    }


def process_qid(
    *,
    client: Optional[GPTClient],
    dry_run: bool,
    obj: Dict[str, Any],
    line_no: int,
    gold_llm: str,
    gold_tools: Sequence[str],
    max_context_chars: int,
    model: str,
    base_url: str,
) -> List[Dict[str, Any]]:
    original_query = str(obj.get("query", "") or "").strip()
    raw_context = str(obj.get("context", "") or "")
    context = truncate_text(raw_context, max_context_chars)
    parsed = parse_context(raw_context, gold_llm, gold_tools)

    cases, variant_meta = build_query_cases(
        client=client,
        dry_run=dry_run,
        original_query=original_query,
        gold_llm=gold_llm,
        gold_tools=gold_tools,
        parsed=parsed,
        context=context,
    )

    out_rows = [
        build_preference_row(
            client=client,
            dry_run=dry_run,
            case=case,
            obj=obj,
            source_line_no=line_no,
            parsed=parsed,
            context=context,
            model=model,
            base_url=base_url,
        )
        for case in cases
    ]
    for row in out_rows:
        row["teacher_meta"]["variant_source"] = variant_meta.get("variant_source", "gpt")
    return out_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if not args.error_output:
        args.error_output = args.output + ".errors.jsonl"

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
    if args.overwrite and os.path.exists(args.error_output):
        os.remove(args.error_output)

    completed_qids, existing_variants, existing_counts = load_resume_state(args.output)
    if existing_counts:
        partial = {
            qid: sorted(existing_variants.get(qid, set()))
            for qid in existing_counts
            if qid not in completed_qids
        }
        print(
            f"[resume] existing rows={sum(existing_counts.values())}, "
            f"completed_qids={len(completed_qids)}, partial_qids={len(partial)}",
            file=sys.stderr,
        )
    if os.path.exists(args.error_output):
        error_qids = set()
        for _, err in read_jsonl(args.error_output):
            qid = str(err.get("qid", "") or "")
            if qid:
                error_qids.add(qid)
        print(
            f"[resume] previous_error_qids={len(error_qids)}; "
            "they will be retried unless already completed in output",
            file=sys.stderr,
        )

    client = None if args.dry_run else GPTClient(args.base_url, args.model)

    stats = {
        "read": 0,
        "eligible": 0,
        "submitted_qids": 0,
        "skipped_duplicate_qid": 0,
        "skipped_completed_qid": 0,
        "skipped_ineligible": 0,
        "failed_qids": 0,
        "written_rows": 0,
        "processed_qids": 0,
    }
    seen_this_run: set[str] = set()
    total = count_records(args.input)

    def update_progress(pbar: Any) -> None:
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                ok=stats["processed_qids"],
                fail=stats["failed_qids"],
                submitted=stats["submitted_qids"],
                skip=stats["skipped_completed_qid"],
                rows=stats["written_rows"],
            )

    def handle_done(fut: Future, fut_meta: Dict[Future, Dict[str, Any]], pbar: Any) -> None:
        meta = fut_meta.pop(fut)
        qid = meta["qid"]
        obj = meta["obj"]
        line_no = meta["line_no"]
        try:
            out_rows = fut.result()
            old_variants = existing_variants.get(qid, set())
            rows_to_write = [
                row for row in out_rows
                if int(row.get("variant_id", -1)) not in old_variants
            ]
            if rows_to_write:
                append_jsonl(args.output, rows_to_write)
                existing_variants.setdefault(qid, set()).update(
                    int(row["variant_id"]) for row in rows_to_write
                )
                if {0, 1, 2}.issubset(existing_variants[qid]):
                    completed_qids.add(qid)
            stats["written_rows"] += len(rows_to_write)
            stats["processed_qids"] += 1
        except Exception as exc:
            stats["failed_qids"] += 1
            write_error(
                args.error_output,
                obj=obj,
                line_no=line_no,
                stage="qid_generation",
                exc=exc,
                model=args.model,
                base_url=args.base_url,
            )
            msg = f"[error] qid={qid} line={line_no}: {type(exc).__name__}: {exc}"
            if hasattr(tqdm, "write"):
                tqdm.write(msg)
            else:
                print(msg, file=sys.stderr)
        update_progress(pbar)

    workers = max(1, int(args.workers or 1))
    max_pending = max(workers * 2, workers)
    print(f"[config] workers={workers}", file=sys.stderr)

    pending: set[Future] = set()
    fut_meta: Dict[Future, Dict[str, Any]] = {}
    stop_submitting = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pbar = tqdm(read_jsonl(args.input), total=total, desc="scan/submit", unit="row")
        for line_no, obj in pbar:
            stats["read"] += 1
            qid = str(obj.get("qid", "") or "")
            if not qid:
                continue
            if qid in completed_qids:
                stats["skipped_completed_qid"] += 1
                update_progress(pbar)
                continue
            if qid in seen_this_run:
                stats["skipped_duplicate_qid"] += 1
                update_progress(pbar)
                continue

            ok, reason, gold_llm, gold_tools = is_eligible(obj)
            if not ok:
                stats["skipped_ineligible"] += 1
                update_progress(pbar)
                continue

            stats["eligible"] += 1
            seen_this_run.add(qid)

            if not stop_submitting:
                fut = executor.submit(
                    process_qid,
                    client=client,
                    dry_run=args.dry_run,
                    obj=obj,
                    line_no=line_no,
                    gold_llm=gold_llm,
                    gold_tools=gold_tools,
                    max_context_chars=args.max_context_chars,
                    model=args.model,
                    base_url=args.base_url,
                )
                pending.add(fut)
                fut_meta[fut] = {"qid": qid, "obj": obj, "line_no": line_no}
                stats["submitted_qids"] += 1
                if args.max_qids and stats["submitted_qids"] >= args.max_qids:
                    stop_submitting = True

            if pending:
                done_now, pending = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
                for fut in done_now:
                    handle_done(fut, fut_meta, pbar)

            while len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle_done(fut, fut_meta, pbar)

            update_progress(pbar)

            if stop_submitting:
                break

        if pending:
            done_pbar = tqdm(total=len(pending), desc="finish", unit="qid")
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle_done(fut, fut_meta, pbar)
                    if hasattr(done_pbar, "update"):
                        done_pbar.update(1)
                    if hasattr(done_pbar, "set_postfix"):
                        done_pbar.set_postfix(
                            ok=stats["processed_qids"],
                            fail=stats["failed_qids"],
                            rows=stats["written_rows"],
                        )
            if hasattr(done_pbar, "close"):
                done_pbar.close()

    if args.sleep > 0:
        print(
            "[note] --sleep is ignored in threaded mode; reduce WORKERS if the API is rate-limited.",
            file=sys.stderr,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"[done] output={args.output}", file=sys.stderr)
    print(f"[done] errors={args.error_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
