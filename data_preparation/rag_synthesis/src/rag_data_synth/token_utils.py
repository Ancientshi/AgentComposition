from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .text_utils import normalize_raw_name, extract_double_wrapped_tools, extract_wrapped_tokens


@dataclass
class GoldTarget:
    llm_token: str
    tool_tokens: List[str]


def is_llm_token(s: str) -> bool:
    return bool(s) and s.startswith("<LLM_") and s.endswith(">")


def is_legacy_tool_token(s: str) -> bool:
    return bool(s) and s.startswith("<TOOL_") and s.endswith(">")


def is_double_tool_token(s: str) -> bool:
    return bool(s) and s.startswith("<<") and s.endswith(">>")


def canonical_llm_token(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if is_llm_token(raw):
        inner = raw[len("<LLM_"):-1]
        return f"<LLM_{inner.replace('/', '__')}>"
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    raw = normalize_raw_name(raw).replace("/", "__")
    if raw.startswith("LLM_"):
        raw = raw[len("LLM_"):]
    return f"<LLM_{raw}>"


def canonical_tool_token(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw or raw == "<TOOL_EMPTY>":
        return ""
    if is_double_tool_token(raw) or is_legacy_tool_token(raw):
        return raw
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    return f"<<{raw}>>"


def token_key(tok: str) -> str:
    """Loose equality key for coverage checking."""
    tok = (tok or "").strip()
    if not tok:
        return ""
    if tok.startswith("<LLM_") and tok.endswith(">"):
        inner = tok[len("<LLM_"):-1]
        return "llm:" + inner.replace("/", "__").lower()
    if tok.startswith("<TOOL_") and tok.endswith(">"):
        inner = tok[len("<TOOL_"):-1]
        return "tool:" + inner.strip().lower()
    if tok.startswith("<<") and tok.endswith(">>"):
        inner = tok[2:-2]
        return "tool:" + inner.strip().lower()
    if tok.startswith("<") and tok.endswith(">"):
        inner = tok[1:-1]
        return "tool:" + inner.strip().lower()
    return "tool:" + tok.lower()


def parse_target_tokens(
    target: str,
    *,
    tool_sep_token: str = "<TOOL_SEP>",
    end_token: str = "<SPECIAL_END>",
    tool_empty_token: str = "<TOOL_EMPTY>",
    max_tools: int = 64,
) -> GoldTarget:
    raw = (target or "").strip()
    if end_token in raw:
        raw = raw.split(end_token, 1)[0].strip()

    llm_token = ""
    tool_tokens: List[str] = []

    if tool_sep_token in raw:
        left, right = raw.split(tool_sep_token, 1)
        left_wrapped = extract_wrapped_tokens(left)
        if left_wrapped:
            llm_token = canonical_llm_token(left_wrapped[0])
        else:
            left_toks = left.strip().split()
            if left_toks:
                llm_token = canonical_llm_token(left_toks[0])

        # Extract double-wrapped first, then legacy single-wrapped tool tokens.
        # Also preserve raw one-token tools such as `Calculator`; this matters because
        # some old targets mixed `<<API&&Endpoint>>` and plain tool names.
        candidates = extract_double_wrapped_tools(right)
        candidates += [t for t in extract_wrapped_tokens(right) if t.startswith("<TOOL_") and t.endswith(">")]

        residue = right
        for t in candidates + [tool_empty_token, tool_sep_token, end_token]:
            residue = residue.replace(t, " ")
        raw_candidates = [t for t in residue.split() if t not in {tool_empty_token, tool_sep_token, end_token}]
        candidates += raw_candidates

        for t in candidates:
            ct = canonical_tool_token(t)
            if ct and ct not in tool_tokens:
                tool_tokens.append(ct)
    else:
        wrapped = extract_wrapped_tokens(raw)
        if wrapped:
            llm_token = canonical_llm_token(wrapped[0])
            for t in wrapped[1:]:
                if t in {tool_empty_token, tool_sep_token, end_token}:
                    continue
                ct = canonical_tool_token(t)
                if ct and ct not in tool_tokens:
                    tool_tokens.append(ct)

    return GoldTarget(llm_token=llm_token, tool_tokens=tool_tokens[:max_tools])
