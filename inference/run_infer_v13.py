#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for the v13 delayed-critic tree/beam structured recommender.

Aligned with the current SFT builder:
prompt =
    ### Task:
    Select the best target agent from the provided context, then explain the choice
    using the retrieved evidence. Output the target agent first.

    ### Context:
    {retrieval evidence context}

    ### User Query:
    {query}

    ### Answer:

target =
    <LLM_xxx> <TOOL_SEP> <TOOL_...> <SPECIAL_END>
or:
    <LLM_xxx> <TOOL_SEP> <TOOL_...> <SPECIAL_END>

    Explanation: ...

IMPORTANT:
- No candidate-space mapping
- No stats_json dependency
- No constrained decoding based on single-token atomic symbols
- Works for full model or PEFT adapter dir

UPGRADE (Scheme B, strict version):
- Context-adaptive controllable generation for LLM/TOOL names with TOKEN-LEVEL prefix constraints
- Supports BOTH tool formats in context:
    <TOOL_xxx>   and   <<ToolName&&Endpoint>>
- LLM selection: token-level constrained phrase selection (same spirit as tool selection)
- Tool selection: token-level constrained decoding over a trie of candidate phrases
  (Once a tool prefix is generated, next tokens are constrained to valid continuations
   until the tool phrase is completed.)
- Enforce "at least min_tools tool phrases before END"
- Penalize TOOL_EMPTY to avoid dominating
- Reserve TOOL_EMPTY for the zero-tool configuration; once emitted at the root, append SPECIAL_END and terminate immediately
- Delay Bundle Critic guidance: no critic at LLM roots or the first-tool layer;
  critic scoring starts once a candidate contains at least two real tools
- Parser recognizes <<...>> in outputs

v13 builds on the v12 retrieval/context format:
- Retrieval results are compacted into the same evidence-style context used by
  build_sft_from_pair_rag.py:
    cf_retrieved_llm / cf_retrieved_tool_bundle /
    semantic_retrieved_llm / semantic_retrieved_tool / evidence
- Controlled decoding still constrains the target agent string to candidates
  appearing in the context.
- When --answer_mode=target_with_explanation, the target string is decoded under
  constraints first, then the free-form Explanation is generated after it.

Notes:
- The critic-tree path performs exact multi-branch traversal over a token-level candidate trie.
  Invalid token transitions are never expanded, while complete phrases are ranked by
  average token log-probability under the original unmasked model distribution.
- The greedy fallback keeps a single trie-constrained path.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import math
import hashlib
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Sequence, Mapping

try:
    import torch
except Exception:  # keep --help / --dry_run usable in lightweight environments
    class _NoGradStub:
        def __call__(self, fn=None):
            if fn is None:
                return self
            return fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _TorchMissing:
        def no_grad(self):
            return _NoGradStub()

        def __getattr__(self, name):
            raise RuntimeError("torch is required for model inference")

    torch = _TorchMissing()  # type: ignore
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # keep --dry_run usable in environments without transformers
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore


# -----------------------------
# Small helpers
# -----------------------------
def longest_common_prefix_len(list_of_ids: List[List[int]]) -> int:
    if not list_of_ids:
        return 0
    m = min(len(x) for x in list_of_ids)
    L = 0
    for i in range(m):
        v = list_of_ids[0][i]
        if all(x[i] == v for x in list_of_ids):
            L += 1
        else:
            break
    return L


# -----------------------------
# Data structure
# -----------------------------
@dataclass
class ParsedOutput:
    raw: str
    llm_token: str
    tool_tokens: List[str]
    strict_text: str


# -----------------------------
# IO utils
# -----------------------------
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Token string utils
# -----------------------------
def normalize_raw_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "".join("_" if ch.isspace() else ch for ch in s)


def is_wrapped_token(s: str) -> bool:
    s = (s or "").strip()
    return len(s) >= 2 and s[0] == "<" and s[-1] == ">"


def wrap_atomic(raw: str, prefix: str) -> str:
    raw = normalize_raw_name(raw)
    if not raw:
        return ""
    if is_wrapped_token(raw):
        return raw
    if not prefix:
        return raw
    return f"{prefix}{raw}>"


# -----------------------------
# Parsing
# -----------------------------
def _spaceify_struct_tokens(text: str, struct_tokens: List[str]) -> str:
    out = text
    for t in struct_tokens:
        if t:
            out = out.replace(t, f" {t} ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_wrapped_tokens(text: str) -> List[str]:
    # match <...> but not across > chars
    return re.findall(r"<[^<>\n\r]+>", text or "")


def _extract_double_wrapped_tools(text: str) -> List[str]:
    # match <<...>>
    return re.findall(r"<<[^<>\n\r]+>>", text or "")


def parse_structured_tokens(
    text: str,
    *,
    tool_sep_token: str = "<TOOL_SEP>",
    end_token: str = "<SPECIAL_END>",
    tool_empty_token: str = "<TOOL_EMPTY>",
    max_tools: int = 16,
) -> Tuple[str, List[str]]:
    """
    Best-effort parse on decoded text (not token ids).
    Supports tools as <TOOL_xxx> or <<...>>.
    Expected shape:
      <LLM_xxx> <TOOL_SEP> <TOOL_a/<<...>>> ... <SPECIAL_END>
    """
    raw = (text or "").strip()
    raw = re.sub(r">\s*(?=<)", "> ", raw)
    raw = _spaceify_struct_tokens(raw, [tool_sep_token, end_token])

    if not raw:
        return "", []

    if end_token in raw:
        raw = raw.split(end_token, 1)[0].strip()

    def _all_wrapped(s: str) -> List[str]:
        return _extract_wrapped_tokens(s) + _extract_double_wrapped_tools(s)

    llm_token = ""
    tool_tokens: List[str] = []

    if tool_sep_token in raw:
        left, right = raw.split(tool_sep_token, 1)

        left_wrapped = _extract_wrapped_tokens(left)
        if left_wrapped:
            llm_token = left_wrapped[0]
        else:
            left_toks = left.strip().split()
            llm_token = left_toks[0] if left_toks else ""

        right_wrapped = _all_wrapped(right)
        for t in right_wrapped:
            if t in (tool_sep_token, end_token, tool_empty_token):
                continue
            # accept only real tool formats
            if t.startswith("<<") and t.endswith(">>"):
                tool_tokens.append(t)
            elif t.startswith("<TOOL_") and t.endswith(">"):
                tool_tokens.append(t)
            else:
                continue
    else:
        wrapped = _all_wrapped(raw)
        if wrapped:
            llm_token = wrapped[0]
            for t in wrapped[1:]:
                if t in (tool_sep_token, end_token, tool_empty_token):
                    continue
                if t.startswith("<<") and t.endswith(">>"):
                    tool_tokens.append(t)
                elif t.startswith("<TOOL_") and t.endswith(">"):
                    tool_tokens.append(t)

    # dedup preserve order
    dedup: List[str] = []
    seen = set()
    for t in tool_tokens:
        if t not in seen:
            dedup.append(t)
            seen.add(t)

    if max_tools > 0:
        dedup = dedup[:max_tools]

    return llm_token, dedup


def build_strict_text(
    *,
    llm_token: str,
    tool_tokens: List[str],
    tool_sep_token: str,
    end_token: str,
    tool_empty_token: str,
) -> str:
    llm = (llm_token or "").strip()
    tools = [(t or "").strip() for t in (tool_tokens or []) if (t or "").strip()]

    if not llm:
        llm = "<LLM_UNK>"

    if not tools:
        tools = [tool_empty_token]

    return " ".join([llm, tool_sep_token] + tools + [end_token])


def build_prompt_v12(*, context: str, query: str, want_explanation: bool, task_text: str = "") -> str:
    if task_text.strip():
        task = task_text.strip()
    elif want_explanation:
        task = (
            "Select the best target agent from the provided context, then explain the choice "
            "using the retrieved evidence. Output the target agent first."
        )
    else:
        task = "Select the best target agent from the provided context. Output only the target agent."
    return (
        f"### Task:\n{task}\n\n"
        f"### Context:\n{context.strip()}\n\n"
        f"### User Query:\n{query.strip()}\n\n"
        "### Answer:\n"
    )


def extract_explanation(text: str, *, end_token: str = "<SPECIAL_END>") -> str:
    raw = text or ""
    if end_token in raw:
        raw = raw.split(end_token, 1)[1]
    m = re.search(r"Explanation\s*:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1)
    raw = re.split(r"\n\s*###\s+", raw, maxsplit=1)[0]
    raw = raw.replace("<|endoftext|>", " ").replace("</s>", " ").replace("<s>", " ")
    return re.sub(r"\s+", " ", raw).strip()


# -----------------------------
# PEFT-aware loading
# -----------------------------
def _load_model_peft_aware(
    model_dir: str,
    *,
    base_model_name: str = "",
    torch_dtype: str = "auto",
    device_map: str = "auto",
    token: bool = True,
):
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required for model loading")

    adapter_cfg = os.path.join(model_dir, "adapter_config.json")

    dtype_obj = "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)

    if os.path.isfile(adapter_cfg):
        from peft import PeftModel  # type: ignore

        cfg = _read_json(adapter_cfg)
        base = base_model_name.strip() or str(cfg.get("base_model_name_or_path", "")).strip()
        if not base:
            raise SystemExit(
                "[infer] adapter_config.json found, but base_model_name_or_path missing. "
                "Please pass --base_model_name explicitly."
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype_obj,
            device_map=device_map,
            token=token,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype_obj,
        device_map=device_map,
        token=token,
    )
    model.eval()
    return model


# -----------------------------
# Candidate extraction
# -----------------------------
def extract_candidates_from_context(context: str) -> Tuple[List[str], List[str]]:
    """
    Extract dynamic candidates from context.

    LLM candidates:
      <LLM_xxx>

    Tool candidates:
      <TOOL_xxx>   (legacy)
      <<Name&&Endpoint>>  (your current format)
    """
    ctx = context or ""

    toks = _extract_wrapped_tokens(ctx)
    llms = [t for t in toks if t.startswith("<LLM_") and t.endswith(">")]

    tools = [
        t
        for t in toks
        if t.startswith("<TOOL_") and t.endswith(">")
        and t not in ("<TOOL_SEP>", "<TOOL_EMPTY>", "<SPECIAL_END>")
    ]

    tools2 = _extract_double_wrapped_tools(ctx)
    tools.extend(tools2)

    # dedup keep order
    def _dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return _dedup(llms), _dedup(tools)


def encode_phrase(tokenizer, phrase: str, leading_space: bool) -> List[int]:
    s = (" " + phrase) if leading_space else phrase
    return tokenizer.encode(s, add_special_tokens=False)


# -----------------------------
# Cached scoring utilities (LLM selection + optional tool ranking)
# -----------------------------
@torch.no_grad()
def _prefix_next_logits(
    model,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
):
    out = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]
    return past, next_logits


@torch.no_grad()
def phrase_logprob_cached(
    model,
    *,
    prefix_past,
    prefix_next_logits: torch.Tensor,
    prefix_attn: torch.Tensor,
    phrase_ids: List[int],
) -> float:
    """
    Correct autoregressive scoring with cached prefix:
      logP(t0|prefix) + logP(t1|prefix,t0) + ...
    """
    if not phrase_ids:
        return -1e30

    device = prefix_next_logits.device
    total = 0.0
    past = prefix_past
    attn = prefix_attn
    next_logits = prefix_next_logits

    for tid in phrase_ids:
        lp = torch.log_softmax(next_logits, dim=-1)
        total += float(lp[0, tid].detach().cpu())

        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)],
            dim=1,
        )
        tok = torch.tensor([[tid]], device=device, dtype=torch.long)
        out = model(input_ids=tok, attention_mask=attn, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    return total





# -----------------------------
# TOKEN-LEVEL constrained decoding over a trie (for tools)
# -----------------------------
class TrieNode:
    __slots__ = ("children", "is_end", "phrase")
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False
        self.phrase: Optional[str] = None  # store the phrase at end


def build_phrase_trie(phrases_with_ids: List[Tuple[str, List[int]]]) -> TrieNode:
    root = TrieNode()
    for phrase, ids in phrases_with_ids:
        node = root
        for tid in ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True
        node.phrase = phrase
    return root


@torch.no_grad()
def step_with_cache(
    model,
    token_id: int,
    past_key_values,
    attention_mask: torch.Tensor,
):
    """
    Advance model by one token using cached kv.
    Returns: new_past, next_logits, new_attention_mask
    """
    device = attention_mask.device
    tok = torch.tensor([[token_id]], device=device, dtype=torch.long)
    new_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
        dim=1,
    )
    out = model(
        input_ids=tok,
        attention_mask=new_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return out.past_key_values, out.logits[:, -1, :], new_mask


@torch.no_grad()
def decode_one_phrase_constrained_greedy(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    candidate_phrases: List[str],
    leading_space: bool,
    max_steps: int = 256,
) -> Tuple[str, List[int], float, float]:
    """Token-level constrained greedy decoding over the candidate trie.

    Returns ``(phrase, token_ids, sum_logprob, avg_logprob)``.  The token
    transition at every step is restricted to valid trie children, while the
    recorded likelihood is taken from the original unmasked model distribution.
    """
    if not candidate_phrases:
        return "", [], -1e30, -1e30

    phrases_with_ids: List[Tuple[str, List[int]]] = []
    for phrase in candidate_phrases:
        ids = encode_phrase(tokenizer, phrase, leading_space=leading_space)
        if ids:
            phrases_with_ids.append((phrase, ids))
    if not phrases_with_ids:
        return "", [], -1e30, -1e30

    trie_root = build_phrase_trie(phrases_with_ids)
    past, next_logits = _prefix_next_logits(model, prefix_ids, prefix_mask)
    attn = prefix_mask
    node = trie_root
    generated: List[int] = []
    total = 0.0

    for _ in range(max_steps):
        allowed = list(node.children.keys())
        if not allowed:
            break
        log_probs = torch.log_softmax(next_logits, dim=-1)[0]
        best_tid = max(allowed, key=lambda tid: float(log_probs[tid].detach().cpu()))
        best_logprob = float(log_probs[best_tid].detach().cpu())
        generated.append(best_tid)
        total += best_logprob
        node = node.children[best_tid]
        past, next_logits, attn = step_with_cache(
            model,
            best_tid,
            past_key_values=past,
            attention_mask=attn,
        )
        if node.is_end and node.phrase:
            return node.phrase, generated, total, total / max(1, len(generated))

    return "", generated, total, total / max(1, len(generated))


@torch.no_grad()
def rank_phrases_trie_constrained(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    candidate_phrases: List[str],
    leading_space: bool,
    top_k: int = 1,
    max_steps: int = 256,
) -> List[Tuple[str, List[int], float, float]]:
    """Exactly rank complete candidate phrases through trie traversal.

    Only valid trie transitions are expanded.  Each completed phrase is scored
    with the original unmasked autoregressive distribution and ranked by its
    average token log-probability.  The returned tuple contains both the raw sum
    and the average so path likelihood accounting remains exact.
    """
    if not candidate_phrases:
        return []

    deduped = list(dict.fromkeys(candidate_phrases))
    phrases_with_ids: List[Tuple[str, List[int]]] = []
    for phrase in deduped:
        ids = encode_phrase(tokenizer, phrase, leading_space=leading_space)
        if ids and len(ids) <= max_steps:
            phrases_with_ids.append((phrase, ids))
    if not phrases_with_ids:
        return []

    trie_root = build_phrase_trie(phrases_with_ids)
    prefix_past, prefix_next_logits = _prefix_next_logits(model, prefix_ids, prefix_mask)
    scored: List[Tuple[str, List[int], float, float]] = []

    def visit(
        node: TrieNode,
        past_key_values,
        next_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_ids: List[int],
        sum_logprob: float,
    ) -> None:
        if node.is_end and node.phrase:
            scored.append((
                node.phrase,
                list(generated_ids),
                sum_logprob,
                sum_logprob / max(1, len(generated_ids)),
            ))
        if len(generated_ids) >= max_steps or not node.children:
            return

        log_probs = torch.log_softmax(next_logits, dim=-1)[0]
        for token_id, child in node.children.items():
            token_logprob = float(log_probs[token_id].detach().cpu())
            new_past, new_logits, new_mask = step_with_cache(
                model,
                token_id,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            visit(
                child,
                new_past,
                new_logits,
                new_mask,
                generated_ids + [token_id],
                sum_logprob + token_logprob,
            )

    visit(
        trie_root,
        prefix_past,
        prefix_next_logits,
        prefix_mask,
        [],
        0.0,
    )
    scored.sort(key=lambda row: (row[3], row[2]), reverse=True)
    return scored[: max(1, int(top_k))]


def rank_phrases_full_score(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    candidate_phrases: List[str],
    leading_space: bool,
    top_k: int = 1,
    max_steps: int = 256,
) -> List[Tuple[str, List[int], float, float]]:
    """Backward-compatible alias for exact trie-constrained phrase ranking."""
    return rank_phrases_trie_constrained(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        prefix_mask=prefix_mask,
        candidate_phrases=candidate_phrases,
        leading_space=leading_space,
        top_k=top_k,
        max_steps=max_steps,
    )

@torch.no_grad()
def decode_llm_candidates_constrained(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    llm_cands: List[str],
    top_k: int = 1,
    phrase_max_steps: int = 256,
) -> List[Tuple[str, List[int], float, float]]:
    """
    LLM selection without two-stage preselection.
    Strategy:
      - if top_k == 1: use trie-based constrained greedy decoding
      - if top_k > 1: fully score all candidates and return top-k
    """
    if not llm_cands:
        return []

    if int(top_k) == 1:
        s, ids, sum_sc, avg_sc = decode_one_phrase_constrained_greedy(
            model,
            tokenizer,
            prefix_ids=prefix_ids,
            prefix_mask=prefix_mask,
            candidate_phrases=llm_cands,
            leading_space=False,
            max_steps=phrase_max_steps,
        )
        if not s:
            return []
        return [(s, ids, sum_sc, avg_sc)]

    return rank_phrases_full_score(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        prefix_mask=prefix_mask,
        candidate_phrases=llm_cands,
        leading_space=False,
        top_k=top_k,
        max_steps=phrase_max_steps,
    )
    
    
# -----------------------------
# Controlled generation (LLM: phrase scoring; Tools: constrained trie decoding)
# -----------------------------
@torch.no_grad()
def append_tools_tokenwise(
    model,
    tokenizer,
    *,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    tool_cands: List[str],
    tool_empty_token: str,
    end_token: str,
    max_tools: int,
    empty_penalty: float = 10.0,
    min_tools: int = 2,
    phrase_max_steps: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Append tool phrases with TOKEN-LEVEL constraints.

    Enforce:
      - At least `min_tools` phrases before allowing END.
      - TOOL_EMPTY is available only before the first real tool and receives a proposal penalty.
    """
    gen_ids = prefix_ids
    gen_mask = prefix_mask

    if not tool_cands:
        tool_cands = [tool_empty_token]

    picked = 0
    for _step in range(max_tools):
        # TOOL_EMPTY is reserved for the zero-tool configuration and is only
        # legal before any real tool has been selected. SPECIAL_END becomes
        # available after the ordinary minimum-tool requirement is satisfied.
        step_cands = list(tool_cands)
        if picked == 0:
            step_cands.append(tool_empty_token)
        if picked >= min_tools:
            step_cands.append(end_token)

        # Decode one full phrase under trie constraints (greedy)
        best_s, best_ids, best_sum_score, best_avg_score = decode_one_phrase_constrained_greedy(
            model,
            tokenizer,
            prefix_ids=gen_ids,
            prefix_mask=gen_mask,
            candidate_phrases=step_cands,
            leading_space=True,
            max_steps=phrase_max_steps,
        )

        if not best_s:
            # Could not complete any candidate phrase; force END to keep output valid
            end_ids = encode_phrase(tokenizer, end_token, leading_space=True)
            gen_ids = torch.cat([gen_ids, torch.tensor(end_ids, device=gen_ids.device).unsqueeze(0)], dim=1)
            gen_mask = torch.cat(
                [gen_mask, torch.ones((gen_mask.shape[0], len(end_ids)), device=gen_mask.device, dtype=gen_mask.dtype)],
                dim=1,
            )
            return gen_ids, gen_mask

        # Penalize TOOL_EMPTY and possibly re-pick
        if best_s == tool_empty_token:
            empty_proposal_score = best_avg_score - float(empty_penalty)
            alts = [c for c in step_cands if c != tool_empty_token]
            if alts:
                alt_s, alt_ids, alt_sum_score, alt_avg_score = decode_one_phrase_constrained_greedy(
                    model,
                    tokenizer,
                    prefix_ids=gen_ids,
                    prefix_mask=gen_mask,
                    candidate_phrases=alts,
                    leading_space=True,
                    max_steps=phrase_max_steps,
                )
                if alt_s and alt_avg_score > empty_proposal_score:
                    best_s, best_ids = alt_s, alt_ids
                    best_sum_score, best_avg_score = alt_sum_score, alt_avg_score

        # Append the phrase token ids
        append_tensor = torch.tensor(best_ids, device=gen_ids.device, dtype=torch.long).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, append_tensor], dim=1)
        gen_mask = torch.cat(
            [gen_mask, torch.ones((gen_mask.shape[0], len(best_ids)), device=gen_mask.device, dtype=gen_mask.dtype)],
            dim=1,
        )

        if best_s == end_token:
            return gen_ids, gen_mask

        # TOOL_EMPTY is the root-only zero-tool control action, not a normal tool component.
        # Once emitted, close this path immediately so no later tool layer is entered.
        if best_s == tool_empty_token:
            end_ids = encode_phrase(tokenizer, end_token, leading_space=True)
            gen_ids = torch.cat(
                [gen_ids, torch.tensor(end_ids, device=gen_ids.device, dtype=torch.long).unsqueeze(0)],
                dim=1,
            )
            gen_mask = torch.cat(
                [
                    gen_mask,
                    torch.ones(
                        (gen_mask.shape[0], len(end_ids)),
                        device=gen_mask.device,
                        dtype=gen_mask.dtype,
                    ),
                ],
                dim=1,
            )
            return gen_ids, gen_mask

        picked += 1

    # If max_tools reached, force END
    end_ids = encode_phrase(tokenizer, end_token, leading_space=True)
    gen_ids = torch.cat([gen_ids, torch.tensor(end_ids, device=gen_ids.device).unsqueeze(0)], dim=1)
    gen_mask = torch.cat(
        [gen_mask, torch.ones((gen_mask.shape[0], len(end_ids)), device=gen_mask.device, dtype=gen_mask.dtype)],
        dim=1,
    )
    return gen_ids, gen_mask


@torch.no_grad()
def controlled_generate_structured(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    *,
    context: str,
    tool_sep_token: str = "<TOOL_SEP>",
    end_token: str = "<SPECIAL_END>",
    tool_empty_token: str = "<TOOL_EMPTY>",
    max_tools: int = 8,
    top_k: int = 1,
    empty_penalty: float = 10.0,
    min_tools: int = 1,
    phrase_max_steps: int = 256,
) -> List[str]:
    """
    Return top-k sequences by:
      1) choosing top-k LLM candidates via constrained phrase selection / full phrase scoring
      2) appending TOOL_SEP
      3) selecting tools with token-level prefix constraints (trie), then END
    """
    llm_cands, tool_cands = extract_candidates_from_context(context)

    if not llm_cands:
        return []

    prefix_ids = enc["input_ids"]
    prefix_mask = enc.get("attention_mask", torch.ones_like(prefix_ids))

    # 1) LLM selection by strict constrained decoding / full scoring (no two-stage preselect)
    llm_top = decode_llm_candidates_constrained(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        prefix_mask=prefix_mask,
        llm_cands=llm_cands,
        top_k=max(1, int(top_k)),
        phrase_max_steps=phrase_max_steps,
    )
    if not llm_top:
        return []

    outs: List[str] = []
    for llm_s, llm_ids, _sum_score, _avg_score in llm_top:
        # append LLM
        gen_ids = torch.cat(
            [prefix_ids, torch.tensor(llm_ids, device=prefix_ids.device).unsqueeze(0)],
            dim=1,
        )
        gen_mask = torch.cat(
            [prefix_mask, torch.ones((prefix_mask.shape[0], len(llm_ids)), device=prefix_mask.device, dtype=prefix_mask.dtype)],
            dim=1,
        )

        # append TOOL_SEP
        sep_ids = encode_phrase(tokenizer, tool_sep_token, leading_space=True)
        gen_ids = torch.cat([gen_ids, torch.tensor(sep_ids, device=gen_ids.device).unsqueeze(0)], dim=1)
        gen_mask = torch.cat(
            [gen_mask, torch.ones((gen_mask.shape[0], len(sep_ids)), device=gen_mask.device, dtype=gen_mask.dtype)],
            dim=1,
        )

        # append tools with strict token-level constraints
        gen_ids, gen_mask = append_tools_tokenwise(
            model,
            tokenizer,
            prefix_ids=gen_ids,
            prefix_mask=gen_mask,
            tool_cands=tool_cands,
            tool_empty_token=tool_empty_token,
            end_token=end_token,
            max_tools=max_tools,
            empty_penalty=empty_penalty,
            min_tools=min_tools,
            phrase_max_steps=phrase_max_steps,
        )

        suffix = tokenizer.decode(gen_ids[0, prefix_ids.shape[1]:], skip_special_tokens=False)
        suffix = re.sub(r"\s+", " ", suffix).strip()
        outs.append(suffix)

    # Dedup keep order
    dedup_outs: List[str] = []
    seen = set()
    for t in outs:
        if t not in seen:
            dedup_outs.append(t)
            seen.add(t)
    return dedup_outs


@torch.no_grad()
def generate_explanation_for_target(
    model,
    tokenizer,
    *,
    prompt: str,
    target_text: str,
    bundle_effect_context: str = "",
    max_new_tokens: int = 160,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Free-generate a rationale grounded by target and optional critic statistics."""
    prefix = f"{prompt}{target_text.strip()}"
    if bundle_effect_context.strip():
        prefix += f"\n\n### Bundle-Effect Statistics:\n{bundle_effect_context.strip()}"
    prefix += "\n\nExplanation:"
    enc = tokenizer(prefix, return_tensors="pt", truncation=True)
    if hasattr(model, "device") and str(model.device) != "cpu":
        enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs: Dict[str, Any] = dict(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        pad_token_id=tokenizer.pad_token_id,
    )
    if bool(do_sample):
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    out_ids = model.generate(**gen_kwargs)
    suffix_ids = out_ids[:, enc["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(suffix_ids, skip_special_tokens=False)[0]
    text = text.replace(tokenizer.eos_token or "", " ")
    text = text.replace(tokenizer.pad_token or "", " ")
    text = re.split(r"\n\s*###\s+", text, maxsplit=1)[0]
    text = re.split(r"\s*###\s+", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------
# v13: delayed-critic tree + beam search
# -----------------------------
# The Bundle Critic is intentionally withheld from LLM-root and one-tool
# decisions. It becomes eligible only after a candidate contains this many
# real tools; all completed paths are still critic-scored during final reranking.
DELAYED_CRITIC_MIN_REAL_TOOLS = 2

@dataclass
class SearchNode:
    node_id: str
    parent_id: str
    depth: int
    llm: str
    tools: Tuple[str, ...]
    suffix_ids: List[int]
    generator_logprob: float
    generator_token_count: int
    last_action: str
    is_complete: bool = False
    critic_raw: Optional[float] = None
    critic_sigmoid: Optional[float] = None
    search_score: Optional[float] = None
    stage_rank: Optional[int] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def generator_avg_logprob(self) -> float:
        return self.generator_logprob / max(1, self.generator_token_count)

    def canonical_bundle_key(self) -> Tuple[str, Tuple[str, ...]]:
        return self.llm, tuple(sorted(set(self.tools)))

    def target_text(
        self,
        *,
        tool_sep_token: str,
        end_token: str,
        tool_empty_token: str,
    ) -> str:
        tools = list(self.tools) if self.tools else [tool_empty_token]
        return " ".join([self.llm, tool_sep_token] + tools + [end_token])

    def to_json(self, *, include_suffix_ids: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "llm": self.llm,
            "tools": list(self.tools),
            "last_action": self.last_action,
            "is_complete": self.is_complete,
            "generator_logprob": self.generator_logprob,
            "generator_token_count": self.generator_token_count,
            "generator_avg_logprob": self.generator_avg_logprob,
            "critic_raw": self.critic_raw,
            "critic_sigmoid": self.critic_sigmoid,
            "search_score": self.search_score,
            "stage_rank": self.stage_rank,
            "history": self.history,
        }
        if include_suffix_ids:
            out["suffix_ids"] = list(self.suffix_ids)
        return out


def _progress_log(enabled: bool, message: str, *, stream=None) -> None:
    """Timestamped, immediately flushed progress output for long searches."""
    if not enabled:
        return
    if stream is None:
        stream = sys.stdout
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=stream, flush=True)


def _node_brief(node: "SearchNode") -> str:
    tools = ", ".join(node.tools) if node.tools else "<no-tools>"
    critic = "n/a" if node.critic_raw is None else f"{node.critic_raw:.4f}"
    search = "n/a" if node.search_score is None else f"{node.search_score:.4f}"
    return (
        f"{node.node_id} llm={node.llm} tools=[{tools}] "
        f"critic={critic} search={search} gen_avg={node.generator_avg_logprob:.4f}"
    )


def _log_top_nodes(enabled: bool, label: str, nodes: Sequence["SearchNode"], top_n: int) -> None:
    if not enabled or not nodes:
        return
    ranked = sorted(
        nodes,
        key=lambda n: (
            float(n.search_score if n.search_score is not None else -1e30),
            float(n.critic_raw if n.critic_raw is not None else -1e30),
            n.generator_avg_logprob,
        ),
        reverse=True,
    )
    _progress_log(True, f"[SEARCH] {label}: showing top {min(len(ranked), max(1, top_n))}/{len(ranked)}")
    for idx, node in enumerate(ranked[: max(1, int(top_n))], 1):
        _progress_log(True, f"[SEARCH]   #{idx} {_node_brief(node)}")


class BundleCriticClient:
    """Small synchronous client with per-run score caching.

    Search-depth candidates are sent together through /v1/rerank, which keeps the
    number of HTTP requests proportional to tree depth rather than node count.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int = 300,
        required: bool = True,
        verbose: bool = False,
        progress_topn: int = 5,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = int(timeout)
        self.required = bool(required)
        self.verbose = bool(verbose)
        self.progress_topn = max(1, int(progress_topn))
        self.score_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}
        self.events: List[Dict[str, Any]] = []

    def _url(self, path: str) -> str:
        if self.base_url.endswith(path):
            return self.base_url
        if "/v1/" in self.base_url:
            return self.base_url.rsplit("/v1/", 1)[0] + path
        return self.base_url + path

    def score_nodes(
        self,
        *,
        query: str,
        nodes: Sequence[SearchNode],
        evidence_context: str,
        stage: str,
    ) -> Dict[str, Any]:
        missing: List[SearchNode] = []
        for node in nodes:
            key = node.canonical_bundle_key()
            cached = self.score_cache.get(key)
            if cached is None:
                missing.append(node)
            else:
                node.critic_raw = float(cached["raw_score"])
                node.critic_sigmoid = float(cached["sigmoid_score"])

        event: Dict[str, Any] = {
            "stage": stage,
            "candidate_count": len(nodes),
            "cache_hits": len(nodes) - len(missing),
            "requested_count": len(missing),
            "started_at": _now_iso(),
        }
        _progress_log(
            self.verbose,
            f"[CRITIC] stage={stage} candidates={len(nodes)} "
            f"cache_hits={len(nodes) - len(missing)} request={len(missing)}",
        )
        if missing:
            payload = {
                "query": query,
                "candidates": [
                    {"id": n.node_id, "llm": n.llm, "tools": list(n.tools)}
                    for n in missing
                ],
                "evidence_context": evidence_context,
                "top_k": len(missing),
            }
            started = time.time()
            try:
                response = _post_json(self._url("/v1/rerank"), payload, timeout=self.timeout)
                by_id = {
                    str(row.get("id")): row
                    for row in response.get("ranking", [])
                    if isinstance(row, dict)
                }
                for node in missing:
                    row = by_id.get(node.node_id)
                    if row is None:
                        raise RuntimeError(f"Critic rerank response missing node id={node.node_id}")
                    node.critic_raw = float(row["raw_score"])
                    node.critic_sigmoid = float(row["sigmoid_score"])
                    self.score_cache[node.canonical_bundle_key()] = {
                        "raw_score": node.critic_raw,
                        "sigmoid_score": node.critic_sigmoid,
                    }
                event["response_count"] = len(by_id)
                event["latency_sec"] = round(time.time() - started, 4)
                raw_values = [n.critic_raw for n in nodes if n.critic_raw is not None]
                score_range = (
                    f" range=[{min(raw_values):.4f}, {max(raw_values):.4f}]"
                    if raw_values else ""
                )
                _progress_log(
                    self.verbose,
                    f"[CRITIC] stage={stage} completed in {event['latency_sec']:.2f}s" + score_range,
                )
            except Exception as exc:
                event["error"] = repr(exc)
                event["traceback"] = traceback.format_exc()
                if self.required:
                    self.events.append(event)
                    raise
                for node in missing:
                    node.critic_raw = 0.0
                    node.critic_sigmoid = 0.5
        if not missing:
            event["latency_sec"] = 0.0
            _progress_log(self.verbose, f"[CRITIC] stage={stage} served fully from cache")
        event["finished_at"] = _now_iso()
        self.events.append(event)
        _log_top_nodes(self.verbose, f"critic ranking after {stage}", nodes, self.progress_topn)
        return event

    def analyze_node(
        self,
        *,
        query: str,
        node: SearchNode,
        evidence_context: str,
        include_pairwise: bool,
        max_pairs: int,
    ) -> Dict[str, Any]:
        payload = {
            "id": node.node_id,
            "query": query,
            "llm": node.llm,
            "tools": list(node.tools),
            "evidence_context": evidence_context,
            "include_pairwise": bool(include_pairwise),
            "max_pairs": int(max_pairs),
        }
        _progress_log(
            self.verbose,
            f"[ANALYZE] node={node.node_id} tools={len(node.tools)} pairwise={bool(include_pairwise)}",
        )
        started = time.time()
        try:
            response = _post_json(self._url("/v1/analyze"), payload, timeout=self.timeout)
            analyze_latency = round(time.time() - started, 4)
            self.events.append({
                "stage": "final_analyze",
                "node_id": node.node_id,
                "latency_sec": analyze_latency,
                "finished_at": _now_iso(),
            })
            _progress_log(self.verbose, f"[ANALYZE] node={node.node_id} completed in {analyze_latency:.2f}s")
            return response
        except Exception as exc:
            event = {
                "stage": "final_analyze",
                "node_id": node.node_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "finished_at": _now_iso(),
            }
            self.events.append(event)
            if self.required:
                raise
            return {"error": repr(exc)}


def _prefix_with_suffix(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    suffix_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not suffix_ids:
        return prompt_ids, prompt_mask
    suffix = torch.tensor(list(suffix_ids), device=prompt_ids.device, dtype=torch.long).unsqueeze(0)
    ids = torch.cat([prompt_ids, suffix], dim=1)
    mask = torch.cat(
        [
            prompt_mask,
            torch.ones(
                (prompt_mask.shape[0], len(suffix_ids)),
                device=prompt_mask.device,
                dtype=prompt_mask.dtype,
            ),
        ],
        dim=1,
    )
    return ids, mask



@torch.no_grad()
def _score_suffix_extension(
    model,
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    suffix_ids: Sequence[int],
    extension_ids: Sequence[int],
) -> float:
    """
    Compute the exact autoregressive log-probability of `extension_ids`
    conditioned on prompt + current suffix.

    This is used when a structural phrase such as SPECIAL_END is appended
    deterministically rather than selected as an ordinary search action.
    """
    extension_ids = list(extension_ids)
    if not extension_ids:
        return 0.0

    prefix_ids, prefix_mask = _prefix_with_suffix(
        prompt_ids,
        prompt_mask,
        suffix_ids,
    )
    prefix_past, prefix_next_logits = _prefix_next_logits(
        model,
        prefix_ids,
        prefix_mask,
    )

    return float(
        phrase_logprob_cached(
            model,
            prefix_past=prefix_past,
            prefix_next_logits=prefix_next_logits,
            prefix_attn=prefix_mask,
            phrase_ids=extension_ids,
        )
    )

def _zscore_map(nodes: Sequence[SearchNode], attr: str) -> Dict[str, float]:
    vals = [float(getattr(n, attr) or 0.0) for n in nodes]
    if not vals:
        return {}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    if std < 1e-8:
        return {n.node_id: 0.0 for n in nodes}
    return {n.node_id: (float(getattr(n, attr) or 0.0) - mean) / std for n in nodes}


def assign_search_scores(
    nodes: Sequence[SearchNode],
    *,
    mode: str,
    critic_weight: float,
    generator_weight: float,
    length_penalty: float,
) -> None:
    if not nodes:
        return
    if mode == "critic":
        critic_z = _zscore_map(nodes, "critic_raw")
        for node in nodes:
            node.search_score = (
                float(critic_weight) * critic_z.get(node.node_id, 0.0)
                - float(length_penalty) * len(node.tools)
            )
        return
    critic_z = _zscore_map(nodes, "critic_raw")
    gen_vals = {n.node_id: n.generator_avg_logprob for n in nodes}
    mean = sum(gen_vals.values()) / len(gen_vals)
    var = sum((v - mean) ** 2 for v in gen_vals.values()) / len(gen_vals)
    std = math.sqrt(var)
    for node in nodes:
        gen_z = 0.0 if std < 1e-8 else (gen_vals[node.node_id] - mean) / std
        node.search_score = (
            float(critic_weight) * critic_z.get(node.node_id, 0.0)
            + float(generator_weight) * gen_z
            - float(length_penalty) * len(node.tools)
        )


def assign_generator_only_scores(
    nodes: Sequence[SearchNode],
    *,
    length_penalty: float = 0.0,
) -> None:
    """Score an early search stage without calling the Bundle Critic.

    LLM roots and one-tool candidates are ranked only by the generator's
    average token log-probability. The optional length term is retained for
    consistency, although all candidates at these stages normally have the
    same number of real tools.
    """
    for node in nodes:
        node.search_score = (
            node.generator_avg_logprob
            - float(length_penalty) * len(node.tools)
        )


def prune_nodes(
    nodes: Sequence[SearchNode],
    *,
    stage: str,
    retain_ratio: float,
    min_beam_size: int,
    max_beam_size: int,
) -> Tuple[List[SearchNode], List[SearchNode], Dict[str, Any]]:
    ranked = sorted(
        nodes,
        key=lambda n: (
            float(n.search_score if n.search_score is not None else -1e30),
            float(n.critic_raw if n.critic_raw is not None else -1e30),
            n.generator_avg_logprob,
        ),
        reverse=True,
    )
    if not ranked:
        return [], [], {"stage": stage, "input_count": 0, "retained_count": 0}
    ratio_keep = max(1, int(math.ceil(len(ranked) * max(0.0, min(1.0, retain_ratio)))))
    keep_n = max(int(min_beam_size), ratio_keep)
    keep_n = min(len(ranked), int(max_beam_size), keep_n)
    kept = ranked[:keep_n]
    pruned = ranked[keep_n:]
    kept_ids = {n.node_id for n in kept}
    for rank, node in enumerate(ranked, 1):
        node.stage_rank = rank
        node.history.append({
            "stage": stage,
            "rank": rank,
            "candidate_count": len(ranked),
            "retained": node.node_id in kept_ids,
            "critic_raw": node.critic_raw,
            "critic_sigmoid": node.critic_sigmoid,
            "search_score": node.search_score,
            "generator_avg_logprob": node.generator_avg_logprob,
        })
    summary = {
        "stage": stage,
        "input_count": len(ranked),
        "retain_ratio": retain_ratio,
        "ratio_keep": ratio_keep,
        "min_beam_size": min_beam_size,
        "max_beam_size": max_beam_size,
        "retained_count": len(kept),
        "pruned_count": len(pruned),
        "retained_node_ids": [n.node_id for n in kept],
        "pruned_node_ids": [n.node_id for n in pruned],
    }
    return kept, pruned, summary


def deduplicate_nodes(nodes: Sequence[SearchNode], *, by_tool_set: bool) -> List[SearchNode]:
    if not by_tool_set:
        return list(nodes)
    best: Dict[Tuple[str, Tuple[str, ...], bool], SearchNode] = {}
    for node in nodes:
        key = (node.llm, tuple(sorted(set(node.tools))), bool(node.is_complete))
        cur = best.get(key)
        if cur is None or node.generator_avg_logprob > cur.generator_avg_logprob:
            best[key] = node
    return list(best.values())


def _cap_complete_pool(nodes: Sequence[SearchNode], cap: int) -> List[SearchNode]:
    deduped = deduplicate_nodes(nodes, by_tool_set=True)
    deduped.sort(
        key=lambda n: (
            float(n.critic_raw if n.critic_raw is not None else -1e30),
            n.generator_avg_logprob,
        ),
        reverse=True,
    )
    return deduped[: max(1, int(cap))]


@torch.no_grad()
def critic_guided_tree_search(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    *,
    query: str,
    context: str,
    critic: BundleCriticClient,
    tool_sep_token: str,
    end_token: str,
    tool_empty_token: str,
    min_tools: int,
    max_tools: int,
    llm_branch_factor: int,
    tool_branch_factor: int,
    retain_ratio: float,
    min_beam_size: int,
    max_beam_size: int,
    num_results: int,
    complete_pool_size: int,
    search_score_mode: str,
    critic_weight: float,
    generator_weight: float,
    length_penalty: float,
    empty_penalty: float,
    dedup_tool_sets: bool,
    phrase_max_steps: int = 256,
    progress: bool = True,
    progress_topn: int = 5,
    progress_parent_interval: int = 1,
) -> Tuple[List[SearchNode], Dict[str, Any]]:
    """Expand one complete component at a time under delayed critic guidance.

    Tree levels:
      level 0: complete LLM roots, generator-only pruning
      level 1: first real tool or the zero-tool TOOL_EMPTY path, generator-only pruning
      later levels: critic guidance is applied only to candidates containing at
                    least ``DELAYED_CRITIC_MIN_REAL_TOOLS`` real tools
      final stage: every completed path, including zero/one-tool paths, is
                   critic-scored and globally reranked
    """
    llm_cands, tool_cands = extract_candidates_from_context(context)
    if not llm_cands:
        raise RuntimeError("No LLM candidates found in retrieval context")

    progress_parent_interval = max(1, int(progress_parent_interval))
    progress_topn = max(1, int(progress_topn))
    _progress_log(
        progress,
        f"[SEARCH] start delayed-critic tree search: llms={len(llm_cands)} "
        f"tools={len(tool_cands)} max_tools={max_tools} retain_ratio={retain_ratio:.2f} "
        f"beam=[{min_beam_size},{max_beam_size}] tool_branch={tool_branch_factor}",
    )

    prompt_ids = enc["input_ids"]
    prompt_mask = enc.get("attention_mask", torch.ones_like(prompt_ids))
    sep_ids = encode_phrase(tokenizer, tool_sep_token, leading_space=True)
    end_ids = encode_phrase(tokenizer, end_token, leading_space=True)
    node_counter = 0

    def next_id() -> str:
        nonlocal node_counter
        node_counter += 1
        return f"n{node_counter:06d}"

    trace: Dict[str, Any] = {
        "algorithm": "component_level_delayed_critic_tree_beam_search",
        "critic_start_real_tools": DELAYED_CRITIC_MIN_REAL_TOOLS,
        "started_at": _now_iso(),
        "llm_candidate_count": len(llm_cands),
        "tool_candidate_count": len(tool_cands),
        "levels": [],
    }

    llm_limit = len(llm_cands) if int(llm_branch_factor) <= 0 else min(len(llm_cands), int(llm_branch_factor))
    _progress_log(progress, f"[SEARCH] scoring {len(llm_cands)} complete LLM phrases; keeping generator top {llm_limit}")
    llm_score_started = time.time()
    llm_ranked = rank_phrases_full_score(
        model,
        tokenizer,
        prefix_ids=prompt_ids,
        prefix_mask=prompt_mask,
        candidate_phrases=llm_cands,
        leading_space=False,
        top_k=llm_limit,
        max_steps=phrase_max_steps,
    )
    _progress_log(progress, f"[SEARCH] LLM phrase scoring completed in {time.time() - llm_score_started:.2f}s")
    roots: List[SearchNode] = []
    for llm, ids, sum_score, avg_score in llm_ranked:
        roots.append(SearchNode(
            node_id=next_id(),
            parent_id="",
            depth=0,
            llm=llm,
            tools=tuple(),
            suffix_ids=list(ids) + list(sep_ids),
            generator_logprob=float(sum_score),
            generator_token_count=len(ids),
            last_action=llm,
        ))
    # Stage 0 deliberately does not call the Bundle Critic.
    # Root pruning is based only on the generator's complete-phrase likelihood.
    critic_event = {
        "stage": "llm_roots",
        "skipped": True,
        "reason": f"Bundle Critic starts only when a candidate contains at least {DELAYED_CRITIC_MIN_REAL_TOOLS} real tools",
    }
    assign_generator_only_scores(roots, length_penalty=length_penalty)
    active, root_pruned, prune_summary = prune_nodes(
        roots,
        stage="llm_roots",
        retain_ratio=retain_ratio,
        min_beam_size=min_beam_size,
        max_beam_size=max_beam_size,
    )
    _progress_log(
        progress,
        f"[PRUNE] llm_roots input={len(roots)} retained={len(active)} pruned={len(root_pruned)}",
    )
    _log_top_nodes(progress, "retained LLM roots", active, progress_topn)
    trace["levels"].append({
        "stage": "llm_roots",
        "critic_event": critic_event,
        "prune": prune_summary,
        "nodes": [n.to_json() for n in roots],
    })

    completed: List[SearchNode] = []
    for depth_index in range(max(1, int(max_tools))):
        if not active:
            _progress_log(progress, f"[SEARCH] stop before depth {depth_index + 1}: no active branches")
            break
        stage = f"tool_depth_{depth_index + 1}"
        depth_started = time.time()
        _progress_log(
            progress,
            f"[SEARCH] {stage} begin: active_parents={len(active)} complete_pool={len(completed)}",
        )
        expansions: List[SearchNode] = []
        action_trace: List[Dict[str, Any]] = []
        for parent_idx, parent in enumerate(active, 1):
            remaining = [t for t in tool_cands if t not in set(parent.tools)]
            actions = list(remaining)

            # TOOL_EMPTY is reserved for the zero-tool configuration. It is
            # therefore legal only before any real tool has been selected and,
            # once chosen, is immediately followed by SPECIAL_END.
            if not parent.tools:
                actions.append(tool_empty_token)

            # SPECIAL_END remains available once the ordinary min-tools
            # constraint has been satisfied.
            if len(parent.tools) >= int(min_tools):
                actions.append(end_token)

            if parent_idx == 1 or parent_idx % progress_parent_interval == 0 or parent_idx == len(active):
                _progress_log(
                    progress,
                    f"[EXPAND] {stage} parent {parent_idx}/{len(active)} {parent.node_id}: "
                    f"tools={len(parent.tools)} candidate_actions={len(actions)}; scoring...",
                )
            parent_score_started = time.time()
            prefix_ids, prefix_mask = _prefix_with_suffix(prompt_ids, prompt_mask, parent.suffix_ids)
            ranked_actions = rank_phrases_full_score(
                model,
                tokenizer,
                prefix_ids=prefix_ids,
                prefix_mask=prefix_mask,
                candidate_phrases=actions,
                leading_space=True,
                top_k=len(actions),
                max_steps=phrase_max_steps,
            )
            adjusted: List[Tuple[str, List[int], float, float, float]] = []
            for action, ids, sum_score, avg_score in ranked_actions:
                proposal_score = float(avg_score) - (
                    float(empty_penalty) if action == tool_empty_token else 0.0
                )
                adjusted.append((action, ids, float(sum_score), float(avg_score), proposal_score))
            adjusted.sort(key=lambda row: (row[4], row[3]), reverse=True)
            chosen = adjusted[: max(1, int(tool_branch_factor))]
            if end_token in actions and not any(row[0] == end_token for row in chosen):
                end_row = next((row for row in adjusted if row[0] == end_token), None)
                if end_row is not None:
                    chosen.append(end_row)

            if parent_idx == 1 or parent_idx % progress_parent_interval == 0 or parent_idx == len(active):
                chosen_text = "; ".join(
                    f"{a}(avg={avg:.3f}, proposal={proposal:.3f})"
                    for a, _ids, _sum, avg, proposal in chosen
                )
                _progress_log(
                    progress,
                    f"[EXPAND] {stage} parent {parent_idx}/{len(active)} completed in "
                    f"{time.time() - parent_score_started:.2f}s; chosen={chosen_text}",
                )

            action_trace.append({
                "parent_id": parent.node_id,
                "depth": parent.depth,
                "top_actions": [
                    {
                        "action": action,
                        "phrase_sum_logprob": sum_score,
                        "phrase_avg_logprob": avg_score,
                        "proposal_score": proposal_score,
                        "token_count": len(ids),
                    }
                    for action, ids, sum_score, avg_score, proposal_score in chosen
                ],
            })
            for action, ids, action_sum_score, action_avg_score, proposal_score in chosen:
                child_tools = parent.tools
                is_empty = action == tool_empty_token
                is_complete = action in (end_token, tool_empty_token)
                suffix = list(parent.suffix_ids) + list(ids)
                child_depth = parent.depth

                # The selected action itself has already been scored.
                child_generator_logprob = (
                    parent.generator_logprob + float(action_sum_score)
                )
                child_generator_token_count = (
                    parent.generator_token_count + len(ids)
                )

                if action not in (end_token, tool_empty_token):
                    child_tools = parent.tools + (action,)
                    child_depth = parent.depth + 1

                if is_empty:
                    # TOOL_EMPTY is selected as an action, but SPECIAL_END is appended
                    # deterministically. Score the appended END under the current prefix.
                    forced_end_logprob = _score_suffix_extension(
                        model,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        suffix_ids=suffix,
                        extension_ids=end_ids,
                    )

                    suffix = suffix + list(end_ids)
                    child_generator_logprob += forced_end_logprob
                    child_generator_token_count += len(end_ids)

                elif not is_complete and child_depth >= int(max_tools):
                    # The final real tool has already been scored. Score SPECIAL_END
                    # conditioned on the path including that final tool.
                    forced_end_logprob = _score_suffix_extension(
                        model,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        suffix_ids=suffix,
                        extension_ids=end_ids,
                    )

                    suffix = suffix + list(end_ids)
                    child_generator_logprob += forced_end_logprob
                    child_generator_token_count += len(end_ids)
                    is_complete = True

                expansions.append(SearchNode(
                    node_id=next_id(),
                    parent_id=parent.node_id,
                    depth=child_depth,
                    llm=parent.llm,
                    tools=tuple(child_tools),
                    suffix_ids=suffix,
                    generator_logprob=child_generator_logprob,
                    generator_token_count=child_generator_token_count,
                    last_action=action,
                    is_complete=is_complete,
                    history=list(parent.history),
                ))

        before_dedup = len(expansions)
        expansions = deduplicate_nodes(expansions, by_tool_set=bool(dedup_tool_sets))
        if not expansions:
            _progress_log(progress, f"[SEARCH] {stage} stopped: no expansions")
            break

        # Delayed critic gating is based on bundle content, not the loop index.
        # Therefore the first-tool layer naturally remains generator-only, while
        # critic guidance starts exactly when a candidate reaches the configured
        # number of real tools. Earlier completed paths (including TOOL_EMPTY)
        # remain in the completed pool and are scored during final reranking.
        critic_eligible = [
            n for n in expansions
            if (not n.is_complete)
            and len(n.tools) >= DELAYED_CRITIC_MIN_REAL_TOOLS
        ]
        critic_ineligible = [n for n in expansions if n not in critic_eligible]

        _progress_log(
            progress,
            f"[SEARCH] {stage} proposals={before_dedup} after_dedup={len(expansions)}; "
            f"critic_eligible={len(critic_eligible)} "
            f"(search-critic skipped={len(critic_ineligible)})",
        )

        if critic_eligible:
            critic_event = critic.score_nodes(
                query=query,
                nodes=critic_eligible,
                evidence_context=context,
                stage=stage,
            )
            assign_search_scores(
                critic_eligible,
                mode=search_score_mode,
                critic_weight=critic_weight,
                generator_weight=generator_weight,
                length_penalty=length_penalty,
            )
        else:
            critic_event = {
                "stage": stage,
                "skipped": True,
                "reason": (
                    "No candidate at this stage contains at least "
                    f"{DELAYED_CRITIC_MIN_REAL_TOOLS} real tools"
                ),
                "candidate_count": len(expansions),
                "critic_start_real_tools": DELAYED_CRITIC_MIN_REAL_TOOLS,
            }

        # Completed paths and zero/one-tool candidates are never passed to the
        # critic during search. They retain generator-only scores until the final
        # global reranking.
        assign_generator_only_scores(critic_ineligible, length_penalty=length_penalty)

        newly_completed = [n for n in expansions if n.is_complete]
        expandable = [n for n in expansions if not n.is_complete]
        completed.extend(newly_completed)
        # Keep early-terminated, not-yet-critic-scored paths until final reranking.
        # Deduplication controls growth without biasing against zero/one-tool paths.
        completed = deduplicate_nodes(completed, by_tool_set=True)

        active, pruned, prune_summary = prune_nodes(
            expandable,
            stage=stage,
            retain_ratio=retain_ratio,
            min_beam_size=min_beam_size,
            max_beam_size=max_beam_size,
        )
        _progress_log(
            progress,
            f"[PRUNE] {stage} expandable={len(expandable)} retained={len(active)} "
            f"pruned={len(pruned)} new_complete={len(newly_completed)} "
            f"complete_pool={len(completed)} elapsed={time.time() - depth_started:.2f}s",
        )
        _log_top_nodes(progress, f"retained branches after {stage}", active, progress_topn)
        if newly_completed:
            _log_top_nodes(progress, f"new complete paths at {stage}", newly_completed, progress_topn)
        trace["levels"].append({
            "stage": stage,
            "critic_event": critic_event,
            "action_proposals": action_trace,
            "new_complete_count": len(newly_completed),
            "complete_pool_count": len(completed),
            "prune": prune_summary,
            "nodes": [n.to_json() for n in expansions],
        })

    if not completed and active:
        forced: List[SearchNode] = []

        for parent in active:
            forced_end_logprob = _score_suffix_extension(
                model,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                suffix_ids=parent.suffix_ids,
                extension_ids=end_ids,
            )

            forced.append(SearchNode(
                node_id=next_id(),
                parent_id=parent.node_id,
                depth=parent.depth,
                llm=parent.llm,
                tools=parent.tools,
                suffix_ids=list(parent.suffix_ids) + list(end_ids),
                generator_logprob=(
                    parent.generator_logprob + forced_end_logprob
                ),
                generator_token_count=(
                    parent.generator_token_count + len(end_ids)
                ),
                last_action=end_token,
                is_complete=True,
                history=list(parent.history),
            ))

        assign_generator_only_scores(
            forced,
            length_penalty=length_penalty,
        )
        completed.extend(forced)

    completed = deduplicate_nodes(completed, by_tool_set=True)
    if not completed:
        raise RuntimeError("Tree search produced no completed candidate paths")

    final_rerank_candidate_count = len(completed)
    _progress_log(
        progress,
        f"[SEARCH] final rerank: scoring all {final_rerank_candidate_count} completed paths; "
        f"complete_pool_cap={complete_pool_size} return_top={num_results}",
    )
    critic.score_nodes(query=query, nodes=completed, evidence_context=context, stage="final_rerank")
    assign_search_scores(
        completed,
        mode=search_score_mode,
        critic_weight=critic_weight,
        generator_weight=generator_weight,
        length_penalty=length_penalty,
    )
    completed.sort(
        key=lambda n: (
            float(n.search_score if n.search_score is not None else -1e30),
            float(n.critic_raw if n.critic_raw is not None else -1e30),
            n.generator_avg_logprob,
        ),
        reverse=True,
    )
    completed = completed[: max(1, int(complete_pool_size))]
    final_nodes = completed[: max(1, int(num_results))]
    for rank, node in enumerate(final_nodes, 1):
        node.stage_rank = rank
    _log_top_nodes(progress, "final candidates", final_nodes, progress_topn)
    _progress_log(progress, f"[SEARCH] finished: generated_nodes={node_counter} completed_pool={len(completed)}")
    trace["final_candidates"] = [n.to_json() for n in final_nodes]
    trace["final_rerank_candidate_count"] = final_rerank_candidate_count
    trace["completed_pool_count"] = len(completed)
    trace["generated_node_count"] = node_counter
    trace["finished_at"] = _now_iso()
    return final_nodes, trace


def _fmt_signed(x: Any, digits: int = 4) -> str:
    try:
        value = float(x)
    except Exception:
        return "n/a"
    return f"{value:+.{digits}f}"


def format_path_bundle_effect_text(node: SearchNode, analysis: Mapping[str, Any]) -> str:
    summary = analysis.get("summary", {}) if isinstance(analysis, Mapping) else {}
    marginals = analysis.get("component_marginals", []) if isinstance(analysis, Mapping) else []
    interactions = analysis.get("pairwise_interactions", []) if isinstance(analysis, Mapping) else []
    lines = [
        f"The bundle critic assigns raw score {float(node.critic_raw or 0.0):.4f} "
        f"(sigmoid {float(node.critic_sigmoid or 0.0):.4f}) to this completed path.",
        f"The path contains {len(node.tools)} tool(s), has cumulative generator log-probability "
        f"{node.generator_logprob:.4f}, and average generated-token log-probability "
        f"{node.generator_avg_logprob:.4f}.",
    ]
    if node.history:
        retained = [h for h in node.history if h.get("retained")]
        if retained:
            stages = ", ".join(
                f"{h.get('stage')} rank {h.get('rank')}/{h.get('candidate_count')}"
                for h in retained
            )
            lines.append("The branch survived pruning at: " + stages + ".")

    if marginals:
        positive = [m for m in marginals if float(m.get("marginal_raw", 0.0)) > 0]
        negative = [m for m in marginals if float(m.get("marginal_raw", 0.0)) < 0]
        if positive:
            lines.append(
                "Positive component contributions: "
                + "; ".join(
                    f"{m.get('component')} {_fmt_signed(m.get('marginal_raw'))}"
                    for m in positive[:4]
                )
                + "."
            )
        if negative:
            lines.append(
                "Potentially redundant or harmful components: "
                + "; ".join(
                    f"{m.get('component')} {_fmt_signed(m.get('marginal_raw'))}"
                    for m in negative[:4]
                )
                + "."
            )
    if interactions:
        synergies = [x for x in interactions if float(x.get("interaction_raw", 0.0)) > 0]
        conflicts = [x for x in interactions if float(x.get("interaction_raw", 0.0)) < 0]
        if synergies:
            lines.append(
                "Strongest estimated synergies: "
                + "; ".join(
                    f"{x.get('component_i')} + {x.get('component_j')} "
                    f"{_fmt_signed(x.get('interaction_raw'))}"
                    for x in synergies[:3]
                )
                + "."
            )
        if conflicts:
            lines.append(
                "Strongest estimated redundancy/conflict signals: "
                + "; ".join(
                    f"{x.get('component_i')} + {x.get('component_j')} "
                    f"{_fmt_signed(x.get('interaction_raw'))}"
                    for x in conflicts[-3:]
                )
                + "."
            )
    if summary:
        lines.append(
            "Aggregate critic diagnostics: positive component rate="
            f"{summary.get('positive_component_rate')}, positive synergy rate="
            f"{summary.get('positive_synergy_rate')}, mean marginal="
            f"{summary.get('mean_marginal_raw')}, mean interaction="
            f"{summary.get('mean_interaction_raw')}."
        )
    lines.append("These are critic-estimated associations under counterfactual removal, not causal effects.")
    return " ".join(lines)

def build_global_bundle_effect_text(
    final_nodes: Sequence[SearchNode],
    analyses: Mapping[str, Mapping[str, Any]],
    *,
    search_score_mode: str = "hybrid",
    critic_weight: float = 1.0,
    generator_weight: float = 0.15,
    length_penalty: float = 0.10,
) -> str:
    if not final_nodes:
        return "No completed candidate bundles were available for global comparison."

    # `final_nodes` already follows the final search ranking produced by
    # critic_guided_tree_search. Preserve that ordering rather than re-sorting
    # by raw critic score, since hybrid mode also uses generator likelihood and
    # the explicit tool-count penalty.
    ranked = list(final_nodes)
    llm_counts: Dict[str, int] = {}
    tool_counts: Dict[str, int] = {}
    for node in ranked:
        llm_counts[node.llm] = llm_counts.get(node.llm, 0) + 1
        for tool in set(node.tools):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    lines = [
        f"The global comparison covers {len(ranked)} completed candidates.",
        "Final search ranking: "
        + "; ".join(
            f"#{idx} {node.llm} with {len(node.tools)} tool(s), "
            f"search={float(node.search_score or 0.0):.4f}, "
            f"critic_raw={float(node.critic_raw or 0.0):.4f}, "
            f"generator_avg={node.generator_avg_logprob:.4f}"
            for idx, node in enumerate(ranked, 1)
        )
        + ".",
    ]

    if search_score_mode == "hybrid":
        lines.append(
            "The final ranking uses a composite score that combines "
            f"stage-normalized critic utility (weight={float(critic_weight):.3f}), "
            f"stage-normalized generator likelihood (weight={float(generator_weight):.3f}), "
            f"and a tool-count penalty of {float(length_penalty):.3f} per selected tool."
        )
    else:
        lines.append(
            "The final ranking uses stage-normalized critic utility "
            f"(weight={float(critic_weight):.3f}) with a tool-count penalty of "
            f"{float(length_penalty):.3f} per selected tool."
        )

    if len(ranked) > 1:
        search_margin = float(ranked[0].search_score or 0.0) - float(ranked[1].search_score or 0.0)
        critic_margin = float(ranked[0].critic_raw or 0.0) - float(ranked[1].critic_raw or 0.0)
        lines.append(
            f"The top candidate leads the runner-up by {_fmt_signed(search_margin)} composite "
            f"search-score points; their raw critic-score difference is {_fmt_signed(critic_margin)}."
        )

    stable_llms = sorted(llm_counts.items(), key=lambda x: (-x[1], x[0]))
    stable_tools = sorted(tool_counts.items(), key=lambda x: (-x[1], x[0]))
    if stable_llms:
        lines.append(
            "LLM consensus: "
            + "; ".join(f"{name} appears in {count}/{len(ranked)}" for name, count in stable_llms[:4])
            + "."
        )
    if stable_tools:
        lines.append(
            "Tool stability across candidates: "
            + "; ".join(f"{name} appears in {count}/{len(ranked)}" for name, count in stable_tools[:8])
            + "."
        )

    all_synergies: List[Tuple[float, str, str, str]] = []
    all_negative_marginals: List[Tuple[float, str, str]] = []
    for node in ranked:
        ana = analyses.get(node.node_id, {})
        for item in ana.get("pairwise_interactions", []) if isinstance(ana, Mapping) else []:
            val = float(item.get("interaction_raw", 0.0))
            if val > 0:
                all_synergies.append((val, node.node_id, str(item.get("component_i")), str(item.get("component_j"))))
        for item in ana.get("component_marginals", []) if isinstance(ana, Mapping) else []:
            val = float(item.get("marginal_raw", 0.0))
            if val < 0:
                all_negative_marginals.append((val, node.node_id, str(item.get("component"))))
    all_synergies.sort(reverse=True)
    all_negative_marginals.sort()
    if all_synergies:
        lines.append(
            "Globally strongest positive interactions: "
            + "; ".join(
                f"{a} + {b} {_fmt_signed(v)} in {nid}"
                for v, nid, a, b in all_synergies[:5]
            )
            + "."
        )
    if all_negative_marginals:
        lines.append(
            "Components with the strongest negative leave-one-out signals: "
            + "; ".join(
                f"{tool} {_fmt_signed(v)} in {nid}"
                for v, nid, tool in all_negative_marginals[:5]
            )
            + "."
        )
    lines.append(
        "The top bundle is selected by the configured search objective"
        "score alone; critic marginals and pairwise interactions are used to explain the component-level "
        "signals that differentiate the candidates."
    )
    lines.append("All bundle-effect quantities are model-estimated associations rather than causal effects.")
    return " ".join(lines)


@torch.no_grad()
def generate_global_explanation(
    model,
    tokenizer,
    *,
    query: str,
    global_statistics_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prefix = (
        "### Task:\nExplain the global bundle-effect comparison across all generated agent candidates. "
        "Use only the supplied statistics, identify why the top candidate is preferred, describe consensus "
        "and disagreements, and mention uncertainty.\n\n"
        f"### User Query:\n{query.strip()}\n\n"
        f"### Global Bundle-Effect Statistics:\n{global_statistics_text.strip()}\n\n"
        "### Global Explanation:\n"
    )
    enc = tokenizer(prefix, return_tensors="pt", truncation=True)
    if hasattr(model, "device") and str(model.device) != "cpu":
        enc = {k: v.to(model.device) for k, v in enc.items()}
    kwargs: Dict[str, Any] = {
        **enc,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    out = model.generate(**kwargs)
    suffix = out[:, enc["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(suffix, skip_special_tokens=False)[0]
    text = text.replace(tokenizer.eos_token or "", " ").replace(tokenizer.pad_token or "", " ")
    text = re.split(r"\n\s*###\s+", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip()



# -----------------------------
# Retrieval + context building
# -----------------------------
DEFAULT_BASE_MODEL = str(AC_BASE_MODEL)
DEFAULT_SEMANTIC_RETRIEVER_URL = "http://127.0.0.1:8504/retrieve"
# CF-signal retrievers. These are the standalone LLM / Tool retrievers.
DEFAULT_CF_LLM_RETR_URL = "http://127.0.0.1:9000/predict"
DEFAULT_CF_TOOL_RETR_URL = "http://127.0.0.1:9001/predict"
# Backward-compatible aliases used by the first v11 draft.
DEFAULT_UNIFIED_RETRIEVER_URL = DEFAULT_SEMANTIC_RETRIEVER_URL
DEFAULT_LEGACY_LLM_RETR_URL = DEFAULT_CF_LLM_RETR_URL
DEFAULT_LEGACY_TOOL_RETR_URL = DEFAULT_CF_TOOL_RETR_URL


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _default_model_dir() -> str:
    # run_infer_v11.py is expected to live at repo root.
    # If you keep the old layout where scripts/*.sh calls ../run_infer_v11.py,
    # this resolves to <repo_root>/model_save/generative_v9_sft/checkpoint-2600.
    return str(Path(__file__).resolve().parent / "model_save" / "generative_v9_sft" / "checkpoint-2600")


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parent / "infer_logs_v13")


def _json_safe(obj: Any) -> Any:
    """Make an object JSON-serializable without losing useful debug information."""
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)


def _load_json_or_empty(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _post_json(url: str, payload: Dict[str, Any], *, timeout: int = 300) -> Dict[str, Any]:
    try:
        import requests  # imported lazily so --context_file / --dry_run can still work without it
    except Exception as e:
        raise RuntimeError("The 'requests' package is required for retrieval calls") from e

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


def flatten_desc(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(flatten_desc(v) for v in x if v is not None).strip()
    if isinstance(x, dict):
        # Keep this compact; context should be textual, not raw JSON.
        parts = []
        for k, v in x.items():
            vv = flatten_desc(v)
            if vv:
                parts.append(f"{k}: {vv}")
        return "; ".join(parts)
    s = str(x).strip()
    # Some tool descriptions are JSON-encoded strings like '["..."]'.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return flatten_desc(json.loads(s)) or s
        except Exception:
            return s
    return s


def yaml_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ")
    return s


def compact_text(x: Any, max_chars: int = 260) -> str:
    s = flatten_desc(x)
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars > 0 and len(s) > max_chars:
        return s[: max(0, max_chars - 1)].rstrip() + "…"
    return s


def fmt_score(x: Any) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def first_present(item: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        v = item.get(k)
        if v not in (None, ""):
            return v
    return ""


def evidence_line(label: str, item: Dict[str, Any], *, token: str = "", tokens: Optional[List[str]] = None) -> str:
    fields = [label]
    rank = item.get("rank")
    if rank not in (None, ""):
        fields.append(f"rank={rank}")
    raw_score = first_present(item, ["raw_score", "score"])
    if raw_score not in (None, ""):
        score_name = "raw_score" if item.get("raw_score") not in (None, "") else "score"
        fields.append(f"{score_name}={fmt_score(raw_score)}")
    norm_score = first_present(item, ["norm_score", "normalized_score"])
    if norm_score not in (None, ""):
        fields.append(f"norm_score={fmt_score(norm_score)}")
    if token:
        fields.append(f"token={token}")
    if tokens:
        fields.append("tools=" + ", ".join(tokens))
        fields.append(f"bundle_size={len(tokens)}")
    item_id = first_present(item, ["item_id", "component", "agent_id", "aid", "matched_agent"])
    if item_id:
        fields.append(f"item_id={compact_text(item_id, 120)}")
    matched_agent = item.get("matched_agent")
    if matched_agent and matched_agent != item_id:
        fields.append(f"matched_agent={compact_text(matched_agent, 120)}")
    matched_subquery = item.get("matched_subquery")
    if matched_subquery:
        fields.append(f"matched_subquery={compact_text(matched_subquery, 180)}")
    support_qid = item.get("support_qid")
    if support_qid:
        fields.append(f"support_qid={compact_text(support_qid, 120)}")
    support_rank = item.get("support_rank")
    if support_rank not in (None, ""):
        fields.append(f"support_rank={support_rank}")
    support_question = item.get("support_question")
    if support_question:
        fields.append(f"support_question={compact_text(support_question, 220)}")
    strengths = item.get("strengths")
    if strengths:
        fields.append(f"strengths={compact_text(strengths, 180)}")
    sources = item.get("sources")
    if sources:
        fields.append(f"sources={compact_text(sources, 180)}")
    desc = item.get("description") or item.get("desc") or item.get("doc_text")
    if desc:
        fields.append(f"desc={compact_text(desc, 280)}")
    return " | ".join(str(x) for x in fields if str(x).strip())


def _empty_section_line() -> str:
    return "  []"


def wrap_llm_name(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("<LLM_") and raw.endswith(">"):
        return raw
    # Prefer training-style model names: spaces -> underscores, but keep / only if it already exists.
    return f"<LLM_{normalize_raw_name(raw)}>"


def wrap_tool_name(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("<<") and raw.endswith(">>"):
        return raw
    if raw.startswith("<TOOL_") and raw.endswith(">"):
        return raw
    return f"<TOOL_{normalize_raw_name(raw)}>"


def wrap_cf_tool_name(raw: str) -> str:
    """Wrap tool ids returned by the PartII two-tower CF service.

    The current Tool twotower service emits bundle tools as plain tool ids
    in `tool_ids` / `gen_text`, while the generative SFT target uses
    double-wrapped tool phrases such as `<<ToolName&&Endpoint>>`.
    Keep already wrapped tokens unchanged; otherwise prefer `<<...>>` for
    CF tool candidates.
    """
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


def _dedup_candidates(items: List[Dict[str, Any]], token_key: str = "token") -> List[Dict[str, Any]]:
    """Deduplicate by generation token while preserving multi-source evidence.

    CF and semantic retrieval can return the same LLM/tool.  For generation we only
    need one token in CONTEXT, but for debugging/evaluation we keep all source tags
    and a compact evidence trail in the JSON meta.
    """
    out: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}

    def _as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]


    for item in items:
        tok = (item.get(token_key) or "").strip()
        if not tok:
            continue
        source = item.get("source") or item.get("retrieval_source")
        evidence = {
            "source": source,
            "rank": item.get("rank"),
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "component": item.get("component"),
            "matched_subquery": item.get("matched_subquery"),
        }
        if tok in index:
            cur = index[tok]
            for src in _as_list(source):
                if src and src not in cur.setdefault("sources", []):
                    cur["sources"].append(src)
            cur.setdefault("evidence", []).append(evidence)
            # Keep the first non-empty description, but fill it if the first source lacked one.
            if not cur.get("description") and item.get("description"):
                cur["description"] = item.get("description")
            continue
        item = dict(item)
        item["sources"] = [x for x in _as_list(source) if x]
        item["evidence"] = [evidence]
        index[tok] = item
        out.append(item)
    return out


def _get_unified_items(retrieval_json: Dict[str, Any], target: str) -> List[Dict[str, Any]]:
    by_type = retrieval_json.get("results_by_type", {}) if isinstance(retrieval_json, dict) else {}
    if isinstance(by_type, dict):
        group = by_type.get(target, {}) or {}
        if isinstance(group, dict) and isinstance(group.get("results"), list):
            return group.get("results", [])

    # Fallback for saved compact examples: use flat evidence and filter component_type.
    evidence = retrieval_json.get("evidence", []) if isinstance(retrieval_json, dict) else []
    if isinstance(evidence, list):
        return [x for x in evidence if isinstance(x, dict) and x.get("component_type") == target]
    return []


def _context_from_unified_response(
    retrieval_json: Dict[str, Any],
    *,
    llm_topk: int,
    tool_topk: int,
) -> Tuple[str, Dict[str, Any]]:
    llm_candidates: List[Dict[str, Any]] = []
    for item in _get_unified_items(retrieval_json, "llm")[: max(0, int(llm_topk))]:
        llm_obj = item.get("llm", {}) if isinstance(item.get("llm"), dict) else {}
        # Use llm.name first because the generation target in your v9/v10 style is <LLM_{name}>.
        name = (
            llm_obj.get("name")
            or item.get("llm")
            or llm_obj.get("canonical_id")
            or item.get("component")
            or ""
        )
        tok = wrap_llm_name(str(name))
        desc = flatten_desc(llm_obj.get("description") or item.get("description"))
        if not desc:
            desc = flatten_desc(llm_obj.get("strengths") or llm_obj.get("source_evidence"))
        if not desc:
            desc = "Retrieved LLM candidate."
        llm_candidates.append({
            "token": tok,
            "description": desc,
            "rank": item.get("rank"),
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "norm_score": item.get("norm_score") or item.get("normalized_score"),
            "component": item.get("component"),
            "item_id": item.get("item_id") or llm_obj.get("item_id") or llm_obj.get("canonical_id"),
            "strengths": llm_obj.get("strengths") or item.get("strengths"),
            "sources": llm_obj.get("sources") or item.get("sources"),
            "matched_subquery": item.get("matched_subquery"),
            "source": "semantic_8504",
        })

    tool_candidates: List[Dict[str, Any]] = []
    for item in _get_unified_items(retrieval_json, "tool")[: max(0, int(tool_topk))]:
        tool_obj = item.get("tool", {}) if isinstance(item.get("tool"), dict) else {}
        key = (
            item.get("component")
            or tool_obj.get("key")
            or tool_obj.get("item_id")
            or item.get("tid")
            or ""
        )
        tok = wrap_cf_tool_name(str(key))
        desc = flatten_desc(tool_obj.get("description") or item.get("description"))
        if not desc:
            desc = flatten_desc(item.get("doc_text") or tool_obj.get("top_level_fields"))
        if not desc:
            desc = "Retrieved tool candidate."
        tool_candidates.append({
            "token": tok,
            "description": desc,
            "rank": item.get("rank"),
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "norm_score": item.get("norm_score") or item.get("normalized_score"),
            "component": item.get("component"),
            "item_id": item.get("item_id") or tool_obj.get("item_id") or tool_obj.get("key"),
            "matched_subquery": item.get("matched_subquery"),
            "source": "semantic_8504",
        })

    return _context_from_v12_evidence_groups(
        semantic_llm_candidates=llm_candidates,
        semantic_tool_candidates=tool_candidates,
    )



def _tag_candidates(candidates: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for item in candidates:
        x = dict(item)
        x["source"] = source
        tagged.append(x)
    return tagged


def _pick_response_rows(obj: Dict[str, Any], keys: List[str]) -> List[Any]:
    """Return the first list-valued response field.

    Important: the Tool PartII /predict service returns both
      {"topk": <int>, "results": [...]}
    so `obj.get("topk") or obj.get("results")` is wrong because
    the integer topk wins and later `for item in tool_rows` crashes.
    """
    if not isinstance(obj, dict):
        return []
    for k in keys:
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return []


def _context_from_hybrid_responses(
    cf_llm_json: Dict[str, Any],
    cf_tool_json: Dict[str, Any],
    semantic_json: Dict[str, Any],
    *,
    llm_topk: int,
    cf_tool_bundle_topk: int,
    semantic_llm_topk: Optional[int] = None,
    semantic_tool_topk: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    'Build one CONTEXT from CF-signal retrievals plus semantic retrieval.\n\n    - 9000/9001 /predict are treated as CF-signal twotower candidate sources.\n    - 8504 is treated as semantic/capability candidate source.\n    - Candidates are concatenated then de-duplicated by the generation token.\n      CF candidates come first by default so the original CF signal remains visible;\n      semantic candidates expand recall when CF misses implicit requirements.\n    '
    _, cf_meta = _context_from_legacy_responses(
        cf_llm_json,
        cf_tool_json,
        llm_topk=llm_topk,
        tool_bundle_topk=cf_tool_bundle_topk,
    )
    _, sem_meta = _context_from_unified_response(
        semantic_json,
        llm_topk=semantic_llm_topk if semantic_llm_topk is not None else llm_topk,
        tool_topk=semantic_tool_topk if semantic_tool_topk is not None else 0,
    )

    context, meta = _context_from_v12_evidence_groups(
        cf_llm_candidates=_tag_candidates(cf_meta.get("cf_llm_candidates", cf_meta.get("llm_candidates", [])), "cf_llm_9000_predict"),
        cf_tool_bundles=cf_meta.get("cf_tool_bundles", []),
        cf_tool_candidates=_tag_candidates(cf_meta.get("cf_tool_candidates", cf_meta.get("tool_candidates", [])), "cf_tool_9001_predict"),
        semantic_llm_candidates=_tag_candidates(
            sem_meta.get("semantic_llm_candidates", sem_meta.get("llm_candidates", [])),
            "semantic_8504",
        ),
        semantic_tool_candidates=_tag_candidates(
            sem_meta.get("semantic_tool_candidates", sem_meta.get("tool_candidates", [])),
            "semantic_8504",
        ),
    )
    meta["hybrid_source_order"] = ["cf_llm_9000_predict", "cf_tool_9001_predict", "semantic_8504"]
    meta["cf_num_llm_candidates_before_dedup"] = len(cf_meta.get("cf_llm_candidates", cf_meta.get("llm_candidates", [])))
    meta["cf_num_tool_candidates_before_dedup"] = len(cf_meta.get("cf_tool_candidates", cf_meta.get("tool_candidates", [])))
    meta["semantic_num_llm_candidates_before_dedup"] = len(sem_meta.get("semantic_llm_candidates", sem_meta.get("llm_candidates", [])))
    meta["semantic_num_tool_candidates_before_dedup"] = len(sem_meta.get("semantic_tool_candidates", sem_meta.get("tool_candidates", [])))
    meta["max_possible_num_llm_candidates_before_dedup"] = int(llm_topk) + int(semantic_llm_topk if semantic_llm_topk is not None else llm_topk)
    # A CF retrieval depth counts bundles, not flattened tools.  The number of
    # tool candidates contributed by those bundles is therefore data-dependent.
    meta["cf_tool_bundle_topk"] = int(cf_tool_bundle_topk)
    meta["semantic_tool_topk"] = int(semantic_tool_topk if semantic_tool_topk is not None else 0)
    meta["max_possible_num_tool_candidates_before_dedup"] = None
    return context, meta


def _context_from_legacy_responses(
    llm_json: Dict[str, Any],
    tool_json: Dict[str, Any],
    *,
    llm_topk: int,
    tool_bundle_topk: int,
) -> Tuple[str, Dict[str, Any]]:
    """Build CONTEXT from the two standalone CF twotower services.

    Expected current services:
      - LLM CF service:  http://127.0.0.1:9000/predict
        Response shape: {"ok": true, "topk": [{"aid", "llm", "name", "description", ...}]}

      - Tool CF service: http://127.0.0.1:9001/predict
        Response shape: {"ok": true, "results": [{"agent_id", "tool_ids", "tools_meta", "gen_text", ...}]}

    Older single-tool responses with `topk=[{"tid": ...}]` are still supported.
    """

    def _as_list(v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

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

    def _tool_desc_from_meta(tid: str, tools_meta: Any) -> str:
        tid_norm = (tid or "").strip().strip("<>")
        for meta in _as_list(tools_meta):
            if not isinstance(meta, dict):
                continue
            mids = [
                meta.get("id"),
                meta.get("name"),
                meta.get("tool_name"),
                meta.get("tid"),
                meta.get("api_name"),
            ]
            mids_norm = [(str(x).strip().strip("<>") if x is not None else "") for x in mids]
            if tid_norm and tid_norm in mids_norm:
                desc = flatten_desc(
                    meta.get("description")
                    or meta.get("desc")
                    or meta.get("summary")
                    or meta.get("api_description")
                    or meta.get("expressions")
                    or meta
                )
                if desc:
                    return desc
        return ""

    def _extract_tool_tokens_from_gen_text(gen_text: str) -> List[str]:
        out: List[str] = []
        for tok in _extract_double_wrapped_tools(gen_text or ""):
            if tok not in out:
                out.append(tok)
        for tok in _extract_wrapped_tokens(gen_text or ""):
            if tok.startswith("<TOOL_") and tok.endswith(">") and tok not in out:
                out.append(tok)
        return out

    llm_candidates: List[Dict[str, Any]] = []
    # LLM /predict uses topk=[...]; allow results=[...] for compatibility.
    llm_rows = _pick_response_rows(llm_json, ["topk", "results", "evidence"])
    for item in llm_rows[: max(0, int(llm_topk))]:
        if not isinstance(item, dict):
            continue
        name = _pick_llm_name(item)
        tok = wrap_llm_name(name)
        if not tok:
            continue
        desc = flatten_desc(item.get("description") or item.get("desc"))
        if not desc and isinstance(item.get("llm"), dict):
            desc = flatten_desc(item["llm"].get("description") or item["llm"].get("meta"))
        if not desc:
            desc = "Retrieved LLM candidate from CF twotower."
        llm_candidates.append({
            "token": tok,
            "description": desc,
            "rank": item.get("rank"),
            "score": item.get("score"),
            "raw_score": item.get("raw_score"),
            "norm_score": item.get("norm_score") or item.get("normalized_score"),
            "component": item.get("component") or item.get("aid") or item.get("agent_id") or name,
            "item_id": item.get("item_id") or item.get("aid") or item.get("agent_id"),
            "strengths": item.get("strengths"),
            "sources": item.get("sources"),
            "source": "cf_llm_9000_predict",
        })

    tool_candidates: List[Dict[str, Any]] = []
    cf_tool_bundles: List[Dict[str, Any]] = []
    seen_tool_tokens = set()
    # Tool /predict uses topk=<requested integer> and results=[...].
    # Do NOT use `tool_json.get("topk") or ...` here.
    tool_rows = _pick_response_rows(tool_json, ["results", "topk", "evidence"])

    # `tool_bundle_topk` limits retrieved bundle rows.  Do not stop after the
    # same number of flattened tools: every tool in each retained bundle must
    # remain available to constrained generation.
    for item in tool_rows[: max(0, int(tool_bundle_topk))]:
        if not isinstance(item, dict):
            continue

        # Case A: older single-tool response, e.g. {"tid": "<<...>>", "description": ...}
        direct_key = item.get("tid") or item.get("tool") or item.get("component") or item.get("key") or item.get("token")
        if direct_key:
            tok = wrap_cf_tool_name(str(direct_key))
            if tok and tok not in seen_tool_tokens:
                desc = flatten_desc(item.get("description") or item.get("desc")) or "Retrieved tool candidate from CF twotower."
                tool_candidates.append({
                    "token": tok,
                    "description": desc,
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "raw_score": item.get("raw_score"),
                    "component": direct_key,
                    "matched_agent": item.get("agent_id") or item.get("aid"),
                    "source": "cf_tool_9001_predict",
                })
                cf_tool_bundles.append({
                    "tokens": [tok],
                    "tool_descriptions": {tok: desc},
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "raw_score": item.get("raw_score"),
                    "matched_agent": item.get("agent_id") or item.get("aid"),
                    "support_qid": item.get("support_qid") or item.get("qid"),
                    "description": desc,
                    "source": "cf_tool_9001_predict",
                })
                seen_tool_tokens.add(tok)

        # Case B: current PartII twotower bundle response, e.g. {"tool_ids": [...], "tools_meta": [...], "gen_text": ...}
        bundled_tool_ids: List[str] = []
        for tid in _as_list(item.get("tool_ids")):
            if isinstance(tid, dict):
                tid = tid.get("id") or tid.get("name") or tid.get("tool_id") or tid.get("tid")
            if tid:
                bundled_tool_ids.append(str(tid))

        # If tool_ids are absent, recover tool tokens from gen_text.
        if not bundled_tool_ids:
            bundled_tool_ids.extend(_extract_tool_tokens_from_gen_text(str(item.get("gen_text") or "")))

        bundle_tokens: List[str] = []
        bundle_desc: Dict[str, str] = {}
        for tid in bundled_tool_ids:
            tok = wrap_cf_tool_name(tid)
            if not tok or tok in seen_tool_tokens:
                continue
            desc = (
                _tool_desc_from_meta(tid, item.get("tools_meta"))
                or flatten_desc(item.get("description") or item.get("desc"))
                or "Retrieved tool candidate from CF twotower bundle."
            )
            tool_candidates.append({
                "token": tok,
                "description": desc,
                "rank": item.get("rank"),
                "score": item.get("score"),
                "raw_score": item.get("raw_score"),
                "component": tid,
                "matched_agent": item.get("agent_id") or item.get("aid"),
                "source": "cf_tool_9001_predict",
            })
            bundle_tokens.append(tok)
            bundle_desc[tok] = desc
            seen_tool_tokens.add(tok)

        if bundle_tokens:
            bundle_desc_text = "; ".join(
                f"{tok}: {desc}" for tok, desc in bundle_desc.items() if desc
            )
            cf_tool_bundles.append({
                "tokens": bundle_tokens,
                "tool_descriptions": bundle_desc,
                "rank": item.get("rank"),
                "score": item.get("score"),
                "raw_score": item.get("raw_score"),
                "matched_agent": item.get("agent_id") or item.get("aid"),
                "support_qid": item.get("support_qid") or item.get("qid"),
                "support_rank": item.get("support_rank"),
                "support_question": item.get("support_question") or item.get("query") or item.get("question"),
                "description": flatten_desc(item.get("description") or item.get("desc")) or bundle_desc_text,
                "source": "cf_tool_9001_predict",
            })

    return _context_from_v12_evidence_groups(
        cf_llm_candidates=llm_candidates,
        cf_tool_bundles=cf_tool_bundles,
        cf_tool_candidates=tool_candidates,
    )


def _context_from_v12_evidence_groups(
    *,
    cf_llm_candidates: Optional[List[Dict[str, Any]]] = None,
    cf_tool_bundles: Optional[List[Dict[str, Any]]] = None,
    cf_tool_candidates: Optional[List[Dict[str, Any]]] = None,
    semantic_llm_candidates: Optional[List[Dict[str, Any]]] = None,
    semantic_tool_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    cf_llm_candidates = _dedup_candidates(cf_llm_candidates or [])
    cf_tool_bundles = cf_tool_bundles or []
    cf_tool_candidates = _dedup_candidates(cf_tool_candidates or [])
    semantic_llm_candidates = _dedup_candidates(semantic_llm_candidates or [])
    semantic_tool_candidates = _dedup_candidates(semantic_tool_candidates or [])

    lines: List[str] = []
    evidence_rows: List[Dict[str, Any]] = []
    ref = 1

    def add_ref(label: str, item: Dict[str, Any], *, token: str = "", tokens: Optional[List[str]] = None) -> int:
        nonlocal ref
        row = {
            "ref_id": ref,
            "label": label,
            "token": token,
            "tokens": tokens or [],
            "item": item,
            "line": evidence_line(label, item, token=token, tokens=tokens),
        }
        evidence_rows.append(row)
        ref += 1
        return row["ref_id"]

    lines.append("cf_retrieved_llm:")
    if cf_llm_candidates:
        parts = []
        for item in cf_llm_candidates:
            tok = item.get("token", "")
            parts.append(f"{tok} [{add_ref('CF-LLM', item, token=tok)}]")
        lines.append("  " + ", ".join(parts))
    else:
        lines.append(_empty_section_line())
    lines.append("")

    lines.append("cf_retrieved_tool_bundle:")
    flattened_cf_tools: List[Dict[str, Any]] = []
    if cf_tool_bundles:
        parts = []
        for bundle in cf_tool_bundles:
            toks = [t for t in bundle.get("tokens", []) if t]
            if not toks:
                continue
            parts.append("{" + ", ".join(toks) + f"}} [{add_ref('CF-TOOL-BUNDLE', bundle, tokens=toks)}]")
            for tok in toks:
                flattened_cf_tools.append({
                    "token": tok,
                    "description": (bundle.get("tool_descriptions") or {}).get(tok, ""),
                    "rank": bundle.get("rank"),
                    "score": bundle.get("score"),
                    "raw_score": bundle.get("raw_score"),
                    "component": tok,
                    "matched_agent": bundle.get("matched_agent"),
                    "source": "cf_tool_bundle",
                })
        lines.append("  " + ", ".join(parts) if parts else _empty_section_line())
    elif cf_tool_candidates:
        parts = []
        for item in cf_tool_candidates:
            tok = item.get("token", "")
            parts.append("{" + tok + f"}} [{add_ref('CF-TOOL', item, token=tok)}]")
            flattened_cf_tools.append(item)
        lines.append("  " + ", ".join(parts) if parts else _empty_section_line())
    else:
        lines.append(_empty_section_line())
    lines.append("")

    lines.append("semantic_retrieved_llm:")
    if semantic_llm_candidates:
        parts = []
        for item in semantic_llm_candidates:
            tok = item.get("token", "")
            parts.append(f"{tok} [{add_ref('SEMANTIC-LLM', item, token=tok)}]")
        lines.append("  " + ", ".join(parts))
    else:
        lines.append(_empty_section_line())
    lines.append("")

    lines.append("semantic_retrieved_tool:")
    if semantic_tool_candidates:
        parts = []
        for item in semantic_tool_candidates:
            tok = item.get("token", "")
            parts.append(f"{tok} [{add_ref('SEMANTIC-TOOL', item, token=tok)}]")
        lines.append("  " + ", ".join(parts))
    else:
        lines.append(_empty_section_line())
    lines.append("")

    lines.append("evidence:")
    if evidence_rows:
        for row in evidence_rows:
            lines.append(f"  [{row['ref_id']}] {row['line']}")
    else:
        lines.append(_empty_section_line())

    all_llms = _dedup_candidates(cf_llm_candidates + semantic_llm_candidates)
    all_tools = _dedup_candidates(flattened_cf_tools + cf_tool_candidates + semantic_tool_candidates)
    meta = {
        "format": "v12_evidence_context",
        "llm_candidates": all_llms,
        "tool_candidates": all_tools,
        "cf_llm_candidates": cf_llm_candidates,
        "cf_tool_bundles": cf_tool_bundles,
        "cf_tool_candidates": cf_tool_candidates,
        "semantic_llm_candidates": semantic_llm_candidates,
        "semantic_tool_candidates": semantic_tool_candidates,
        "evidence": evidence_rows,
        "num_llm_candidates": len(all_llms),
        "num_tool_candidates": len(all_tools),
        "num_cf_llm_candidates": len(cf_llm_candidates),
        "num_cf_tool_bundles": len(cf_tool_bundles),
        "num_semantic_llm_candidates": len(semantic_llm_candidates),
        "num_semantic_tool_candidates": len(semantic_tool_candidates),
    }
    return "\n".join(lines), meta


def _context_from_candidates(
    llm_candidates: List[Dict[str, Any]],
    tool_candidates: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    llm_candidates = _dedup_candidates(llm_candidates)
    tool_candidates = _dedup_candidates(tool_candidates)

    lines = ["llms:"]
    for item in llm_candidates:
        lines.append(f'  "{yaml_escape(item["token"])}": "{yaml_escape(item.get("description", ""))}"')

    lines.append("tools:")
    for item in tool_candidates:
        lines.append(f'  "{yaml_escape(item["token"])}": "{yaml_escape(item.get("description", ""))}"')

    context = "\n".join(lines)
    meta = {
        "llm_candidates": llm_candidates,
        "tool_candidates": tool_candidates,
        "num_llm_candidates": len(llm_candidates),
        "num_tool_candidates": len(tool_candidates),
    }
    return context, meta


def retrieve_and_build_context(args: argparse.Namespace, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    'Return (context, context_meta, retrieval_record).\n\n    v11 default is hybrid retrieval:\n      1) 9000/predict: LLM CF-signal twotower retrieval\n      2) 9001/predict: CF tool-bundle retrieval\n      3) 8504/retrieve: selectable semantic/capability targets\n      4) Python-side CONTEXT construction + JSON logging\n    '
    retrieval_record: Dict[str, Any] = {
        "mode": args.retriever_mode,
        "started_at": _now_iso(),
    }

    # Useful for replay/debugging a previous retrieval without re-calling servers.
    if args.retrieval_json_file:
        retrieval_json = _read_json(args.retrieval_json_file)
        context, meta = _context_from_unified_response(
            retrieval_json,
            llm_topk=args.semantic_llm_topk,
            tool_topk=args.semantic_tool_topk,
        )
        retrieval_record.update({
            "source": "retrieval_json_file",
            "retrieval_json_file": args.retrieval_json_file,
            "response": retrieval_json,
            "finished_at": _now_iso(),
        })
        return context, meta, retrieval_record

    mode = args.retriever_mode
    if mode == "unified":
        mode = "semantic"
    if mode == "legacy":
        mode = "cf"

    semantic_targets = [
        target.strip().lower()
        for target in str(args.semantic_targets).split(",")
        if target.strip()
    ]
    invalid_targets = [target for target in semantic_targets if target not in {"llm", "tool"}]
    if invalid_targets or not semantic_targets:
        raise ValueError(
            "--semantic_targets must be a comma-separated subset of {llm,tool}; "
            f"got {args.semantic_targets!r}"
        )

    semantic_top_k: Dict[str, int] = {}
    if "llm" in semantic_targets:
        semantic_top_k["llm"] = int(args.semantic_llm_topk)
    if "tool" in semantic_targets:
        semantic_top_k["tool"] = int(args.semantic_tool_topk)

    semantic_payload: Dict[str, Any] = {
        "query": query,
        "targets": semantic_targets,
        "top_k": semantic_top_k,
        "rewrite": bool(args.semantic_rewrite),
        "agg": args.agg,
        "candidate_multiplier": int(args.candidate_multiplier),
        "include_raw": bool(args.include_raw),
        "include_doc_text": bool(args.include_doc_text),
        "tool_recall_mode": bool(args.tool_recall_mode),
    }
    if args.per_subquery_top_k > 0:
        semantic_payload["per_subquery_top_k"] = int(args.per_subquery_top_k)
    if args.final_top_k > 0:
        semantic_payload["final_top_k"] = int(args.final_top_k)
    semantic_payload.update(_load_json_or_empty(args.retrieval_extra_json))

    cf_llm_payload: Dict[str, Any] = {
        "query": query,
        "top_k": int(args.llm_topk),
        # Keep topk too for compatibility with older /predict-style retrievers.
        "topk": int(args.llm_topk),
        "rewrite": bool(args.cf_rewrite),
        "agg": args.agg,
        "include_raw_agent": bool(args.include_raw),
    }
    cf_tool_payload: Dict[str, Any] = {
        "query": query,
        "top_k": int(args.cf_tool_bundle_topk),
        "topk": int(args.cf_tool_bundle_topk),
        "rewrite": bool(args.cf_rewrite),
        "agg": args.agg,
        "candidate_multiplier": int(args.candidate_multiplier),
        "include_raw_tool": bool(args.include_raw),
        "include_doc_text": bool(args.include_doc_text),
    }

    if mode in {"hybrid", "auto"}:
        started = time.time()
        partial: Dict[str, Any] = {}
        errors: Dict[str, Any] = {}
        try:
            partial["cf_llm_response"] = _post_json(args.cf_llm_retr_url, cf_llm_payload, timeout=int(args.retriever_timeout))
        except Exception as e:
            errors["cf_llm_error"] = {"error": repr(e), "traceback": traceback.format_exc()}
        try:
            partial["cf_tool_response"] = _post_json(args.cf_tool_retr_url, cf_tool_payload, timeout=int(args.retriever_timeout))
        except Exception as e:
            errors["cf_tool_error"] = {"error": repr(e), "traceback": traceback.format_exc()}
        try:
            partial["semantic_response"] = _post_json(args.semantic_retr_url, semantic_payload, timeout=int(args.retriever_timeout))
        except Exception as e:
            errors["semantic_error"] = {"error": repr(e), "traceback": traceback.format_exc()}

        have_cf = "cf_llm_response" in partial and "cf_tool_response" in partial
        have_sem = "semantic_response" in partial
        if have_cf and have_sem:
            context, meta = _context_from_hybrid_responses(
                partial["cf_llm_response"],
                partial["cf_tool_response"],
                partial["semantic_response"],
                llm_topk=args.llm_topk,
                cf_tool_bundle_topk=args.cf_tool_bundle_topk,
                semantic_llm_topk=args.semantic_llm_topk,
                semantic_tool_topk=args.semantic_tool_topk,
            )
            source = "hybrid_cf_plus_semantic"
        elif have_sem and bool(args.allow_partial_retrieval):
            context, meta = _context_from_unified_response(
                partial["semantic_response"],
                llm_topk=args.semantic_llm_topk,
                tool_topk=args.semantic_tool_topk,
            )
            source = "partial_semantic_only"
        elif have_cf and bool(args.allow_partial_retrieval):
            context, meta = _context_from_legacy_responses(
                partial["cf_llm_response"],
                partial["cf_tool_response"],
                llm_topk=args.llm_topk,
                tool_bundle_topk=args.cf_tool_bundle_topk,
            )
            source = "partial_cf_only"
        else:
            raise RuntimeError(f"Hybrid retrieval failed. errors={json.dumps(errors, ensure_ascii=False)[:4000]}")

        retrieval_record.update({
            "source": source,
            "cf_llm_url": args.cf_llm_retr_url,
            "cf_tool_url": args.cf_tool_retr_url,
            "semantic_url": args.semantic_retr_url,
            "cf_llm_request": cf_llm_payload,
            "cf_tool_request": cf_tool_payload,
            "semantic_request": semantic_payload,
            **partial,
            "errors": errors,
            "latency_sec": round(time.time() - started, 4),
            "finished_at": _now_iso(),
        })
        return context, meta, retrieval_record

    if mode == "semantic":
        started = time.time()
        response = _post_json(args.semantic_retr_url, semantic_payload, timeout=int(args.retriever_timeout))
        context, meta = _context_from_unified_response(
            response,
            llm_topk=args.semantic_llm_topk,
            tool_topk=args.semantic_tool_topk,
        )
        retrieval_record.update({
            "source": 'semantic_unified_retrieval',
            "url": args.semantic_retr_url,
            "request": semantic_payload,
            "response": response,
            "latency_sec": round(time.time() - started, 4),
            "finished_at": _now_iso(),
        })
        return context, meta, retrieval_record

    if mode == "cf":
        started = time.time()
        cf_llm_json = _post_json(args.cf_llm_retr_url, cf_llm_payload, timeout=int(args.retriever_timeout))
        cf_tool_json = _post_json(args.cf_tool_retr_url, cf_tool_payload, timeout=int(args.retriever_timeout))
        context, meta = _context_from_legacy_responses(
            cf_llm_json,
            cf_tool_json,
            llm_topk=args.llm_topk,
            tool_bundle_topk=args.cf_tool_bundle_topk,
        )
        retrieval_record.update({
            "source": 'cf_two_endpoint_retrievals',
            "cf_llm_url": args.cf_llm_retr_url,
            "cf_tool_url": args.cf_tool_retr_url,
            "cf_llm_request": cf_llm_payload,
            "cf_tool_request": cf_tool_payload,
            "cf_llm_response": cf_llm_json,
            "cf_tool_response": cf_tool_json,
            "latency_sec": round(time.time() - started, 4),
            "finished_at": _now_iso(),
        })
        return context, meta, retrieval_record

    raise ValueError(f"Unsupported retriever_mode={args.retriever_mode!r}")


def load_tokenizer_peft_aware(model_dir: str, base_model_name: str = ""):
    """Prefer adapter tokenizer, fallback to base model tokenizer for PEFT adapter dirs."""
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for tokenizer loading")
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception as first_err:
        adapter_cfg = os.path.join(model_dir, "adapter_config.json")
        base = base_model_name.strip()
        if not base and os.path.isfile(adapter_cfg):
            try:
                cfg = _read_json(adapter_cfg)
                base = str(cfg.get("base_model_name_or_path", "")).strip()
            except Exception:
                base = ""
        if base:
            print(f"[WARN] Failed to load tokenizer from model_dir; fallback to base model tokenizer: {base}", file=sys.stderr)
            return AutoTokenizer.from_pretrained(base, use_fast=True)
        raise first_err


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end AgentRec generation: retrieve -> build CONTEXT -> generate -> JSON log.")

    # Query input. In v11, shell only needs to pass --query.
    ap.add_argument("--query", type=str, default="", help="User query. Required unless --query_file is provided.")
    ap.add_argument("--query_file", type=str, default="", help="Read query from a text file.")

    # Optional manual context override. If provided, retrieval is skipped.
    ap.add_argument("--context", type=str, default="")
    ap.add_argument("--context_file", type=str, default="")
    ap.add_argument("--retrieval_json_file", type=str, default="", help="Replay/build context from a saved unified retrieval JSON.")

    # Model paths. Defaults match your v10 layout.
    ap.add_argument("--model_dir", type=str, default=_default_model_dir())
    ap.add_argument("--base_model_name", type=str, default=DEFAULT_BASE_MODEL)

    # Retrieval endpoints.
    # Default v11 mode is hybrid: CF signal from 9000/9001 /predict + semantic signal from 8504 /retrieve.
    ap.add_argument('--retrieval_mode', choices=["hybrid", "cf", "semantic", "unified", "legacy", "auto"], default="hybrid")
    ap.add_argument("--semantic_retr_url", type=str, default=os.getenv("SEMANTIC_RETR_URL", os.getenv("UNIFIED_RETR_URL", DEFAULT_SEMANTIC_RETRIEVER_URL)))
    ap.add_argument("--cf_llm_retr_url", type=str, default=os.getenv("CF_LLM_RETR_URL", os.getenv("LLM_RETR_URL", DEFAULT_CF_LLM_RETR_URL)))
    ap.add_argument("--cf_tool_retr_url", type=str, default=os.getenv("CF_TOOL_RETR_URL", os.getenv("TOOL_RETR_URL", DEFAULT_CF_TOOL_RETR_URL)))
    # Backward-compatible aliases. They are copied onto the new names below if explicitly supplied.
    ap.add_argument('--retrieval_url', type=str, default="")
    ap.add_argument("--legacy_llm_retr_url", type=str, default="")
    ap.add_argument("--legacy_tool_retr_url", type=str, default="")
    ap.add_argument("--allow_partial_retrieval", type=int, default=0, help="In hybrid/auto mode, allow falling back to only CF or only semantic if one side fails.")
    ap.add_argument('--retrieval_timeout', type=int, default=300)
    ap.add_argument("--retrieval_extra_json", type=str, default="", help='Optional JSON object merged into semantic/unified retrieval payload.')

    # Retrieval/context config.
    ap.add_argument("--llm_topk", type=int, default=10, help="CF LLM top-k from 9000/predict.")
    ap.add_argument(
        "--cf_tool_bundle_topk",
        type=int,
        default=5,
        help="Number of CF tool bundles requested from 9001/predict. All tools in each retained bundle are merged.",
    )
    ap.add_argument(
        "--tool_topk",
        type=int,
        default=-1,
        help="Deprecated alias for --cf_tool_bundle_topk.",
    )
    ap.add_argument(
        "--semantic_targets",
        type=str,
        default="tool",
        help="Comma-separated capability targets: tool, llm, or llm,tool.",
    )
    ap.add_argument("--semantic_llm_topk", type=int, default=0, help="Capability LLM top-k from 8504; ignored unless semantic_targets includes llm.")
    ap.add_argument("--semantic_tool_topk", type=int, default=25, help="Capability tool top-k from 8504; ignored unless semantic_targets includes tool.")
    ap.add_argument("--rewrite", type=int, default=0, help="Backward-compatible switch. Used as default for both cf_rewrite and semantic_rewrite if those are not set.")
    ap.add_argument("--cf_rewrite", type=int, default=-1)
    ap.add_argument("--semantic_rewrite", type=int, default=-1)
    ap.add_argument("--agg", choices=["max", "mean", "hybrid"], default="max")
    ap.add_argument("--candidate_multiplier", type=int, default=5)
    ap.add_argument("--include_raw", type=int, default=0)
    ap.add_argument("--include_doc_text", type=int, default=0)
    ap.add_argument("--tool_recall_mode", type=int, default=0)
    ap.add_argument("--per_subquery_top_k", type=int, default=0)
    ap.add_argument("--final_top_k", type=int, default=0)

    # Generation config from v9.
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--max_source_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument(
        "--answer_mode",
        choices=["target_only", "target_with_explanation", "auto"],
        default="target_with_explanation",
        help="v12 prompt/completion style. auto currently behaves like target_with_explanation for inference.",
    )
    ap.add_argument("--task_text", type=str, default="", help="Optional override for the ### Task text.")
    ap.add_argument("--max_explanation_tokens", type=int, default=160)

    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--tool_sep_token", type=str, default="<TOOL_SEP>")
    ap.add_argument("--end_token", type=str, default="<SPECIAL_END>")
    ap.add_argument("--tool_empty_token", type=str, default="<TOOL_EMPTY>")
    ap.add_argument("--max_tools", type=int, default=8)

    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default="auto")
    ap.add_argument("--token", type=int, default=1)

    ap.add_argument("--with_parsed", type=int, default=1)
    ap.add_argument("--print_raw", type=int, default=1)

    ap.add_argument("--controlled", type=int, default=1, help="1: strict controllable; 0: free generate() fallback")
    ap.add_argument("--empty_penalty", type=float, default=10.0, help="Proposal penalty for the root-only zero-tool action TOOL_EMPTY")
    ap.add_argument("--min_tools", type=int, default=1, help="Minimum tool phrases before allowing END")
    ap.add_argument("--phrase_max_steps", type=int, default=256, help="Max tokens to decode for one LLM/tool phrase")

    # v13 delayed-critic tree/beam search.
    ap.add_argument("--search_mode", choices=["critic_tree", "greedy"], default="critic_tree", help="critic_tree uses delayed critic guidance: no critic at LLM roots or one-tool candidates")
    ap.add_argument("--critic_url", type=str, default=os.getenv("BUNDLE_CRITIC_URL", AC_ENV('CRITIC_URL','http://127.0.0.1:8015')))
    ap.add_argument("--critic_timeout", type=int, default=300)
    ap.add_argument("--critic_required", type=int, default=1)
    ap.add_argument("--beam_retain_ratio", type=float, default=0.30)
    ap.add_argument("--beam_min_size", type=int, default=3)
    ap.add_argument("--beam_max_size", type=int, default=24)
    ap.add_argument("--llm_branch_factor", type=int, default=0, help="0 means score all retrieved LLMs before pruning")
    ap.add_argument("--tool_branch_factor", type=int, default=5)
    ap.add_argument("--num_results", type=int, default=5)
    ap.add_argument("--complete_pool_size", type=int, default=50)
    ap.add_argument("--search_score_mode", choices=["critic", "hybrid"], default="hybrid")
    ap.add_argument("--critic_score_weight", type=float, default=1.0)
    ap.add_argument("--generator_score_weight", type=float, default=0.15)
    ap.add_argument("--search_length_penalty", type=float, default=0.1)
    ap.add_argument("--dedup_tool_sets", type=int, default=1)
    ap.add_argument("--include_pairwise_analysis", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--generate_global_explanation", type=int, default=1)
    ap.add_argument("--max_global_explanation_tokens", type=int, default=256)
    ap.add_argument("--allow_free_fallback", type=int, default=0)
    ap.add_argument("--progress", type=int, default=1, help="Print timestamped tree-search progress logs")
    ap.add_argument("--progress_topn", type=int, default=5, help="Number of best paths shown after each prune")
    ap.add_argument(
        "--progress_parent_interval",
        type=int,
        default=1,
        help="Print one parent-expansion update every N active parents",
    )

    # JSON logging.
    ap.add_argument("--output_dir", type=str, default=_default_output_dir())
    ap.add_argument("--output_json", type=str, default="", help="Path to write the full process JSON. Default: output_dir/run_<timestamp>.json")
    ap.add_argument("--print_full_json", type=int, default=0, help="Print full process JSON instead of only generation results.")
    ap.add_argument("--dry_run", type=int, default=0, help="Only retrieve/build context and write JSON; do not load/generate with model.")

    args = ap.parse_args()

    # Backward compatibility with first v11 draft / old shell envs.
    if args.retriever_url:
        args.semantic_retr_url = args.retriever_url
    if args.legacy_llm_retr_url:
        args.cf_llm_retr_url = args.legacy_llm_retr_url
    if args.legacy_tool_retr_url:
        args.cf_tool_retr_url = args.legacy_tool_retr_url
    if args.cf_rewrite < 0:
        args.cf_rewrite = int(args.rewrite)
    if args.semantic_rewrite < 0:
        args.semantic_rewrite = int(args.rewrite)
    if args.tool_topk >= 0:
        args.cf_tool_bundle_topk = int(args.tool_topk)

    semantic_targets = {
        target.strip().lower()
        for target in str(args.semantic_targets).split(",")
        if target.strip()
    }
    if not semantic_targets or not semantic_targets.issubset({"llm", "tool"}):
        raise SystemExit("--semantic_targets must be tool, llm, or llm,tool")
    if "llm" not in semantic_targets:
        args.semantic_llm_topk = 0
    if "tool" not in semantic_targets:
        args.semantic_tool_topk = 0
    if int(args.llm_topk) < 0 or int(args.cf_tool_bundle_topk) < 0:
        raise SystemExit("--llm_topk and --cf_tool_bundle_topk must be >= 0")
    if int(args.semantic_llm_topk) < 0 or int(args.semantic_tool_topk) < 0:
        raise SystemExit("Semantic top-k values must be >= 0")

    if not (0.0 < float(args.beam_retain_ratio) <= 1.0):
        raise SystemExit("--beam_retain_ratio must be in (0, 1]")
    if int(args.beam_min_size) < 1 or int(args.beam_max_size) < int(args.beam_min_size):
        raise SystemExit("Require 1 <= --beam_min_size <= --beam_max_size")
    if int(args.tool_branch_factor) < 1:
        raise SystemExit("--tool_branch_factor must be >= 1")
    if int(args.num_results) < 1 or int(args.complete_pool_size) < int(args.num_results):
        raise SystemExit("Require 1 <= --num_results <= --complete_pool_size")
    if int(args.min_tools) < 0 or int(args.max_tools) < max(1, int(args.min_tools)):
        raise SystemExit("Require 0 <= --min_tools <= --max_tools and --max_tools >= 1")
    if args.search_mode == "critic_tree" and not str(args.critic_url).strip():
        raise SystemExit("--critic_url is required for critic_tree search")
    if int(args.progress_topn) < 1 or int(args.progress_parent_interval) < 1:
        raise SystemExit("Require --progress_topn >= 1 and --progress_parent_interval >= 1")

    return args


# -----------------------------
# Main
# -----------------------------
def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_json = args.output_json.strip()
    if not output_json:
        output_json = str(Path(args.output_dir) / f"run_infer_v13_delayed_critic_{run_id}.json")

    query = (args.query or "").strip()
    if args.query_file:
        query = _read_text(args.query_file).strip()
    if not query:
        raise SystemExit("[infer_v13] Missing query. Pass --query or --query_file.")

    want_explanation = args.answer_mode in {"target_with_explanation", "auto"}

    record: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "script": str(Path(__file__).resolve()),
        "query": query,
        "config": {
            "model_dir": args.model_dir,
            "base_model_name": args.base_model_name,
            'retrieval_mode': args.retriever_mode,
            "semantic_retr_url": args.semantic_retr_url,
            "cf_llm_retr_url": args.cf_llm_retr_url,
            "cf_tool_retr_url": args.cf_tool_retr_url,
            "cf_llm_topk": args.llm_topk,
            "cf_tool_bundle_topk": args.cf_tool_bundle_topk,
            "semantic_targets": sorted(
                target.strip().lower()
                for target in str(args.semantic_targets).split(",")
                if target.strip()
            ),
            "semantic_llm_topk": args.semantic_llm_topk,
            "semantic_tool_topk": args.semantic_tool_topk,
            "max_context_llm_candidates_before_dedup": args.llm_topk + args.semantic_llm_topk if args.retriever_mode in {"hybrid", "auto"} else args.llm_topk,
            # CF depth counts bundles, whose flattened tool count varies.
            "max_context_tool_candidates_before_dedup": None,
            "rewrite": bool(args.rewrite),
            "agg": args.agg,
            "candidate_multiplier": args.candidate_multiplier,
            "tool_recall_mode": bool(args.tool_recall_mode),
            "top_k": args.top_k,
            "num_beams": args.num_beams,
            "max_source_length": args.max_source_length,
            "max_tools": args.max_tools,
            "controlled": bool(args.controlled),
            "answer_mode": args.answer_mode,
            "max_explanation_tokens": args.max_explanation_tokens,
            "min_tools": args.min_tools,
            "empty_penalty": args.empty_penalty,
            "phrase_max_steps": args.phrase_max_steps,
            "search_mode": args.search_mode,
            "critic_start_real_tools": DELAYED_CRITIC_MIN_REAL_TOOLS,
            "critic_url": args.critic_url,
            "critic_required": bool(args.critic_required),
            "beam_retain_ratio": args.beam_retain_ratio,
            "beam_min_size": args.beam_min_size,
            "beam_max_size": args.beam_max_size,
            "llm_branch_factor": args.llm_branch_factor,
            "tool_branch_factor": args.tool_branch_factor,
            "num_results": args.num_results,
            "complete_pool_size": args.complete_pool_size,
            "search_score_mode": args.search_score_mode,
            "critic_score_weight": args.critic_score_weight,
            "generator_score_weight": args.generator_score_weight,
            "search_length_penalty": args.search_length_penalty,
            "dedup_tool_sets": bool(args.dedup_tool_sets),
            "include_pairwise_analysis": bool(args.include_pairwise_analysis),
            "max_pairs": args.max_pairs,
            "generate_global_explanation": bool(args.generate_global_explanation),
            "progress": bool(args.progress),
            "progress_topn": args.progress_topn,
            "progress_parent_interval": args.progress_parent_interval,
            "dry_run": bool(args.dry_run),
            "output_json": output_json,
        },
    }

    # 1) Build context: manual override OR Python-side retrieval + context construction.
    context = (args.context or "").strip()
    if args.context_file:
        context = _read_text(args.context_file).strip()

    if context:
        context_meta = {
            "source": "manual_context",
            "num_llm_candidates": len(extract_candidates_from_context(context)[0]),
            "num_tool_candidates": len(extract_candidates_from_context(context)[1]),
        }
        retrieval_record = {"source": "skipped_manual_context", "finished_at": _now_iso()}
    else:
        context, context_meta, retrieval_record = retrieve_and_build_context(args, query)

    record["retrieval"] = retrieval_record
    record["context"] = {
        "text": context,
        "meta": context_meta,
    }
    prompt = build_prompt_v12(
        context=context,
        query=query,
        want_explanation=want_explanation,
        task_text=args.task_text,
    )
    record["prompt"] = {
        "text": prompt,
        "want_explanation": want_explanation,
        "answer_mode": args.answer_mode,
    }

    print("[INFO] Constructed CONTEXT:")
    print("----------------------------------------")
    print(context)
    print("----------------------------------------")

    if bool(args.dry_run):
        record["generation"] = {"skipped": True, "reason": "dry_run"}
        record["finished_at"] = _now_iso()
        _write_json(output_json, record)
        print(f"[INFO] Full process JSON saved to: {output_json}")
        return record

    # 2) Load model/tokenizer.
    record["model_loading"] = {"started_at": _now_iso()}
    model_load_started = time.time()
    _progress_log(bool(args.progress), f"[MODEL] loading tokenizer/model from {args.model_dir}")
    tokenizer = load_tokenizer_peft_aware(args.model_dir, args.base_model_name)

    model = _load_model_peft_aware(
        args.model_dir,
        base_model_name=args.base_model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        token=bool(args.token),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    record["model_loading"].update({
        "finished_at": _now_iso(),
        "latency_sec": round(time.time() - model_load_started, 4),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
    })
    _progress_log(bool(args.progress), f"[MODEL] loaded in {time.time() - model_load_started:.2f}s")

    # 3) Prompt tokenization.
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_source_length),
    )

    if hasattr(model, "device") and str(model.device) != "cpu":
        enc = {k: v.to(model.device) for k, v in enc.items()}

    answer_start_idx = int(enc["input_ids"].shape[1])
    record["prompt"].update({
        "input_token_count": answer_start_idx,
        "max_source_length": int(args.max_source_length),
        "truncated": bool(answer_start_idx >= int(args.max_source_length)),
    })

    # 4) Generate.
    generation_record: Dict[str, Any] = {
        "started_at": _now_iso(),
        "controlled": bool(args.controlled),
        "want_explanation": want_explanation,
        "used_fallback_free_generate": False,
    }
    cleaned: List[str] = []

    if bool(args.controlled) and args.search_mode == "critic_tree":
        critic = BundleCriticClient(
            args.critic_url,
            timeout=int(args.critic_timeout),
            required=bool(args.critic_required),
            verbose=bool(args.progress),
            progress_topn=int(args.progress_topn),
        )
        final_nodes, search_trace = critic_guided_tree_search(
            model,
            tokenizer,
            enc,
            query=query,
            context=context,
            critic=critic,
            tool_sep_token=args.tool_sep_token,
            end_token=args.end_token,
            tool_empty_token=args.tool_empty_token,
            min_tools=int(args.min_tools),
            max_tools=int(args.max_tools),
            llm_branch_factor=int(args.llm_branch_factor),
            tool_branch_factor=int(args.tool_branch_factor),
            retain_ratio=float(args.beam_retain_ratio),
            min_beam_size=int(args.beam_min_size),
            max_beam_size=int(args.beam_max_size),
            num_results=int(args.num_results),
            complete_pool_size=int(args.complete_pool_size),
            search_score_mode=args.search_score_mode,
            critic_weight=float(args.critic_score_weight),
            generator_weight=float(args.generator_score_weight),
            length_penalty=float(args.search_length_penalty),
            empty_penalty=float(args.empty_penalty),
            dedup_tool_sets=bool(args.dedup_tool_sets),
            phrase_max_steps=int(args.phrase_max_steps),
            progress=bool(args.progress),
            progress_topn=int(args.progress_topn),
            progress_parent_interval=int(args.progress_parent_interval),
        )
        analyses: Dict[str, Dict[str, Any]] = {}
        results: List[Dict[str, Any]] = []
        for final_rank, node in enumerate(final_nodes, 1):
            _progress_log(
                bool(args.progress),
                f"[RESULT] processing candidate {final_rank}/{len(final_nodes)}: {_node_brief(node)}",
            )
            analysis = critic.analyze_node(
                query=query,
                node=node,
                evidence_context=context,
                include_pairwise=bool(args.include_pairwise_analysis),
                max_pairs=int(args.max_pairs),
            )
            analyses[node.node_id] = analysis
            bundle_effect_text = format_path_bundle_effect_text(node, analysis)
            target_text = node.target_text(
                tool_sep_token=args.tool_sep_token,
                end_token=args.end_token,
                tool_empty_token=args.tool_empty_token,
            )
            explanation = ""
            if want_explanation:
                _progress_log(bool(args.progress), f"[EXPLAIN] generating explanation for rank {final_rank}")
                explanation_started = time.time()
                explanation = generate_explanation_for_target(
                    model,
                    tokenizer,
                    prompt=prompt,
                    target_text=target_text,
                    bundle_effect_context=bundle_effect_text,
                    max_new_tokens=int(args.max_explanation_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
                _progress_log(
                    bool(args.progress),
                    f"[EXPLAIN] rank {final_rank} completed in {time.time() - explanation_started:.2f}s",
                )
            gen_text = target_text
            if want_explanation:
                gen_text += f"\n\nExplanation: {explanation}" if explanation else "\n\nExplanation:"
            results.append({
                "rank": final_rank,
                "node_id": node.node_id,
                "gen_text": gen_text,
                "llm_token": node.llm,
                "tool_tokens": list(node.tools) if node.tools else [args.tool_empty_token],
                "strict_text": target_text,
                "explanation": explanation,
                "bundle_effect_explanation": bundle_effect_text,
                "critic": {
                    "raw_score": node.critic_raw,
                    "sigmoid_score": node.critic_sigmoid,
                    "analysis": analysis,
                },
                "search_path": node.to_json(),
            })

        global_stats_text = build_global_bundle_effect_text(final_nodes, analyses)
        global_explanation = ""
        if bool(args.generate_global_explanation):
            _progress_log(bool(args.progress), "[GLOBAL] generating cross-candidate bundle-effect explanation")
            global_explain_started = time.time()
            global_explanation = generate_global_explanation(
                model,
                tokenizer,
                query=query,
                global_statistics_text=global_stats_text,
                max_new_tokens=int(args.max_global_explanation_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            _progress_log(
                bool(args.progress),
                f"[GLOBAL] explanation completed in {time.time() - global_explain_started:.2f}s",
            )
        generation_record.update({
            "finished_at": _now_iso(),
            "search_mode": "critic_tree",
            "used_fallback_free_generate": False,
            "search_trace": search_trace,
            "critic_api_events": critic.events,
            "global_bundle_effect_statistics": global_stats_text,
            "global_explanation": global_explanation,
            "results": results,
        })
        record["generation"] = generation_record
        record["results"] = results
        record["global_explanation"] = {
            "bundle_effect_statistics": global_stats_text,
            "text": global_explanation,
        }
        record["finished_at"] = _now_iso()
        _write_json(output_json, record)
        print(f"[INFO] Full process JSON saved to: {output_json}")
        return record

    if bool(args.controlled):
        outs = controlled_generate_structured(
            model,
            tokenizer,
            enc,
            context=context,
            tool_sep_token=args.tool_sep_token,
            end_token=args.end_token,
            tool_empty_token=args.tool_empty_token,
            max_tools=int(args.max_tools),
            top_k=max(1, int(args.top_k)),
            empty_penalty=float(args.empty_penalty),
            min_tools=int(args.min_tools),
            phrase_max_steps=int(args.phrase_max_steps),
        )
        cleaned = [re.sub(r"\s+", " ", t).strip() for t in outs]
        generation_record["controlled_targets"] = list(cleaned)

        if cleaned and want_explanation:
            with_explanations: List[str] = []
            for target_text in cleaned:
                explanation = generate_explanation_for_target(
                    model,
                    tokenizer,
                    prompt=prompt,
                    target_text=target_text,
                    max_new_tokens=int(args.max_explanation_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
                if explanation:
                    with_explanations.append(f"{target_text}\n\nExplanation: {explanation}")
                else:
                    with_explanations.append(f"{target_text}\n\nExplanation:")
            cleaned = with_explanations

    if not cleaned and not bool(args.allow_free_fallback):
        raise RuntimeError("Controlled generation produced no valid candidates and free fallback is disabled")

    if not cleaned:
        generation_record["used_fallback_free_generate"] = True
        num_return_sequences = max(1, int(args.top_k))
        num_beams = max(int(args.num_beams), num_return_sequences)

        gen_kwargs: Dict[str, Any] = dict(
            **enc,
            max_new_tokens=int(args.max_new_tokens),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=bool(args.do_sample),
            pad_token_id=tokenizer.pad_token_id,
        )

        if bool(args.do_sample):
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)
        else:
            gen_kwargs["early_stopping"] = True

        with torch.no_grad():
            out_ids = model.generate(**gen_kwargs)

        suffix_ids = out_ids[:, answer_start_idx:]
        texts = tokenizer.batch_decode(suffix_ids, skip_special_tokens=False)

        strip_list = [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]
        strip_list = [x for x in strip_list if x]

        for t in texts:
            for s in strip_list:
                t = t.replace(s, " ")
            t = re.sub(r">\s*(?=<)", "> ", t)
            t = re.sub(r"\s+", " ", t).strip()
            if args.end_token in t and not want_explanation:
                t = t.split(args.end_token, 1)[0].strip() + f" {args.end_token}"
            cleaned.append(t)

    # 5) Parse / format.
    results: List[Dict[str, Any]] = []
    for t in cleaned:
        row: Dict[str, Any] = {}

        if args.print_raw:
            row["gen_text"] = t

        if bool(args.with_parsed):
            llm_tok, tool_toks = parse_structured_tokens(
                t,
                tool_sep_token=args.tool_sep_token,
                end_token=args.end_token,
                tool_empty_token=args.tool_empty_token,
                max_tools=int(args.max_tools),
            )

            strict_text = build_strict_text(
                llm_token=llm_tok,
                tool_tokens=tool_toks,
                tool_sep_token=args.tool_sep_token,
                end_token=args.end_token,
                tool_empty_token=args.tool_empty_token,
            )

            row["llm_token"] = llm_tok
            row["tool_tokens"] = tool_toks if tool_toks else [args.tool_empty_token]
            row["strict_text"] = strict_text
            row["explanation"] = extract_explanation(
                t,
                end_token=args.end_token,
            ) if want_explanation else ""

        results.append(row if row else {"gen_text": t})

    generation_record.update({
        "finished_at": _now_iso(),
        "raw_outputs": cleaned,
        "results": results,
    })
    record["generation"] = generation_record
    record["results"] = results
    record["finished_at"] = _now_iso()

    _write_json(output_json, record)
    print(f"[INFO] Full process JSON saved to: {output_json}")
    return record


def main() -> None:
    try:
        args = parse_args()
        record = run_pipeline(args)
        if bool(args.print_full_json):
            print(json.dumps(_json_safe(record), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(record.get("results", []), ensure_ascii=False, indent=2))
    except Exception as e:
        # Best-effort failure JSON if args are parseable and output path can be resolved.
        err = {
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "finished_at": _now_iso(),
        }
        try:
            args = locals().get("args") or parse_args()
            output_json = (getattr(args, "output_json", "") or "").strip()
            if not output_json:
                rid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_error_" + uuid.uuid4().hex[:8]
                output_json = str(Path(getattr(args, "output_dir", _default_output_dir())) / f"run_infer_v13_delayed_critic_{rid}.json")
            err["query"] = getattr(args, "query", "")
            err["output_json"] = output_json
            _write_json(output_json, err)
            print(f"[ERROR] Full error JSON saved to: {output_json}", file=sys.stderr)
        except Exception:
            pass
        print(json.dumps(err, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
