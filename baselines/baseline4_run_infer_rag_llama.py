#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'RAG-Llama baseline for Agent Generative Recommendation.\n\nPurpose\n-------\nThis baseline isolates *free Llama generation* from the proposed controllable\nagent generator while keeping the retrieval/context and evaluation protocol\naligned with ``inference/run_infer_v13_batch_eval_sota.py``.\n\nAligned with SOTA\n-----------------\n- same hybrid retrieval implementation imported from the SOTA script;\n- CF LLM retrieval: top-10 by default;\n- CF tool-bundle retrieval: top-5 bundles by default;\n- semantic/capability tool retrieval: top-25 by default, rewrite disabled;\n- same context builder/evidence formatting as SOTA;\n- same deterministic random-sample manifest logic as SOTA;\n- target-only gold remains ``dataset_example.target`` in saved records;\n- per-sample JSON keeps ``results`` / ``generation.results`` fields compatible\n  with the existing target-only baseline evaluators.\n\nWhat is intentionally REMOVED\n-----------------------------\n- no SFT/PEFT task adapter (default model is the original Llama-3-8B backbone);\n- no constrained decoding;\n- no candidate trie / phrase restriction;\n- no beam search (one greedy autoregressive completion);\n- no Bundle Critic;\n- no fuzzy/semantic post-hoc mapping of hallucinated names back into the retrieved space;\n- evaluation-side protocol normalization may restore an EXACT retrieved identifier when\n  the generated payload differs only by wrappers such as <...>, <<...>>, or TOOL_.\n\nThe prompt tells Llama the required serialization format and explicitly asks it\nnot to invent components. This is *instruction-only grounding*: the decoder is\nstill completely free to violate the format or hallucinate, and such failures\nare preserved and logged.\n'

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_BASE_MODEL = "/root/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B"
DEFAULT_INPUT_JSONL = str(AC_ROOT / 'datasets/generative_v9_sft/sft_valid_PartII.jsonl')
DEFAULT_EXP_DIR = str(AC_ROOT / 'outputs/baseline4_rag_llama_instruct_seed42_n100_PartII')

_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


def _resolve_sota_script(explicit: str = "") -> Path:
    """Locate the SOTA script whose retrieval/context code is reused exactly."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SOTA script not found: {p}")
        return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / 'inference/run_infer_v13_batch_eval_sota.py',
        here.parent / 'inference/run_infer_v13_batch_eval_sota.py',
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        'Could not find inference/run_infer_v13_batch_eval_sota.py. '
        'Pass --sota_script /path/to/inference/run_infer_v13_batch_eval_sota.py'
    )


def _load_sota_module(path: Path):
    spec = importlib.util.spec_from_file_location("agentrec_sota_shared", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SOTA module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Some decorators/types expect the module to be registered while executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(obj), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rag_llama_messages(
    *,
    context: str,
    conversation_history: str,
    max_tools: int,
    context_llms: Sequence[str],
    context_tools: Sequence[str],
) -> List[Dict[str, str]]:
    """Build system/user messages for the prompt-only RAG-Llama baseline.

    Important design choice:
      - no fictitious few-shot identifiers;
      - no placeholder string such as ``<LLM_...> TOOL_1 TOOL_2``;
      - only REAL retrieved identifiers are shown.
    """
    llm_inventory = "\n".join(f"- {x}" for x in context_llms) if context_llms else "- (none)"
    tool_inventory = "\n".join(f"- {x}" for x in context_tools) if context_tools else "- (none)"

    system = """You are an AGENT CONFIGURATION RECOMMENDER.

You do not solve the user's task and you do not continue the user-assistant conversation.
The supplied conversation is historical task context that you must ANALYZE.

Your only job is to select resources for a downstream agent:
- exactly one LLM from the retrieved LLM inventory;
- only the necessary tools from the retrieved tool inventory.

Never browse, calculate, fetch information, summarize the requested content, or answer the user.
Never repeat a previous assistant answer from the conversation.
Never output explanations, JSON, markdown, URLs, headings, or commentary.

Grounding is strict at the IDENTIFIER level:
- copy every selected identifier verbatim from the corresponding retrieved inventory;
- never invent, rename, normalize, split, merge, or reconstruct an identifier;
- a string written as <<...>> is one indivisible tool;
- a string written as <TOOL_...> is also one indivisible tool."""

    user = f"""RETRIEVED LLM INVENTORY
{llm_inventory}

RETRIEVED TOOL INVENTORY
{tool_inventory}

RETRIEVED CANDIDATE EVIDENCE
Use this only for deciding which resources are suitable. Do not continue any conversation or answer embedded in this evidence.
{context.strip()}

FULL CONVERSATION HISTORY
<BEGIN_HISTORY>
{conversation_history.strip()}
<END_HISTORY>

SELECTION REQUIREMENT
Select one retrieved LLM and at most {int(max_tools)} necessary retrieved tools for handling the needs expressed in the full conversation.

OUTPUT CONTRACT
Return exactly one line.
1. Start by copying one exact identifier from RETRIEVED LLM INVENTORY.
2. Then output the literal token <TOOL_SEP>.
3. Then copy the selected complete tool identifier(s) from RETRIEVED TOOL INVENTORY, separated by spaces.
4. If no tool is needed, output the literal token <TOOL_EMPTY> instead.
5. Finish with the literal token <SPECIAL_END>.
6. Stop immediately after <SPECIAL_END>.

Do not write the next assistant response to the conversation.
Do not solve the conversation's task.
Output the resource-selection line only."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rag_llama_plain_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Fallback serialization for tokenizers without a chat template."""
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role") or "").upper()
        content = str(msg.get("content") or "")
        parts.append(f"### {role}\n{content}")
    parts.append("### ASSISTANT\n")
    return "\n\n".join(parts)

def _strip_model_special_tokens(text: str, tokenizer) -> str:
    out = text or ""
    for token in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token):
        if token:
            out = out.replace(token, " ")
    return re.sub(r"\s+", " ", out).strip()


def _trim_after_first_end(text: str, end_token: str) -> str:
    """Evaluation view only; raw completion is separately preserved."""
    if end_token in text:
        return text.split(end_token, 1)[0].strip() + f" {end_token}"
    return text.strip()


def _canonical_component_payload(text: Any, *, strip_tool_prefix: bool = True) -> str:
    """Normalize serialization wrappers only; never fuzzy-match component names.

    Examples that become the same payload::

        <<House Plants&&Get By Origin>>
        <House Plants&&Get By Origin>
        <TOOL_House Plants&&Get By Origin>
        <<TOOL_House Plants&&Get By Origin>>

    all normalize to ``House Plants&&Get By Origin``.  Internal spelling, spacing,
    punctuation, and endpoint names are otherwise preserved, so a generated
    ``Get By Climate`` does NOT become ``Get By Origin``.
    """
    s = str(text or "").strip()
    if not s:
        return ""

    # Tolerate malformed repeated angle wrappers, e.g. <<<<tool>>>>.
    s = re.sub(r"^\s*<+", "", s)
    s = re.sub(r">+\s*$", "", s)
    s = s.strip()

    if strip_tool_prefix and s.startswith("TOOL_"):
        s = s[len("TOOL_"):].strip()

    return s


def _inventory_payload_lookup(tokens: Sequence[str], *, strip_tool_prefix: bool) -> Dict[str, List[str]]:
    """Map wrapper-insensitive payloads to exact retrieved identifiers."""
    lookup: Dict[str, List[str]] = {}
    for tok in tokens:
        exact = str(tok or "").strip()
        key = _canonical_component_payload(exact, strip_tool_prefix=strip_tool_prefix)
        if not exact or not key:
            continue
        lookup.setdefault(key, [])
        if exact not in lookup[key]:
            lookup[key].append(exact)
    return lookup


def _serialize_unknown_tool_payload(payload: str) -> str:
    """Keep an unmatched generated tool explicit so hallucinations remain measurable."""
    payload = str(payload or "").strip()
    return f"<<{payload}>>" if payload else ""


def parse_structured_tokens_relaxed_protocol(
    text: Any,
    *,
    context_llms: Sequence[str],
    context_tools: Sequence[str],
    tool_sep_token: str,
    end_token: str,
    tool_empty_token: str,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Parse free generation while ignoring only serialization-shell mistakes.

    Matching policy:
      1. Strip only outer ``<``/``>`` wrappers and an optional ``TOOL_`` prefix.
      2. If the resulting payload exactly equals a retrieved inventory payload,
         restore the inventory's exact identifier.
      3. Otherwise keep the generated payload as an explicit unknown tool.

    There is deliberately NO fuzzy, edit-distance, embedding, alias, or semantic
    repair.  Therefore hallucinated names remain hallucinations.
    """
    raw = str(text or "").strip()
    before_end = raw.split(end_token, 1)[0] if end_token in raw else raw

    llm_lookup = _inventory_payload_lookup(context_llms, strip_tool_prefix=False)
    tool_lookup = _inventory_payload_lookup(context_tools, strip_tool_prefix=True)

    llm_token = ""
    llm_generated = ""
    llm_match = re.search(r"<+\s*(LLM_[^<>\n\r]+?)\s*>+", before_end)
    if llm_match:
        llm_generated = llm_match.group(1).strip()
        llm_key = _canonical_component_payload(llm_generated, strip_tool_prefix=False)
        exact_llms = llm_lookup.get(llm_key, [])
        # Exact wrapper-insensitive restoration only when unambiguous.
        llm_token = exact_llms[0] if len(exact_llms) == 1 else f"<{llm_key}>"

    if tool_sep_token in before_end:
        tool_body = before_end.split(tool_sep_token, 1)[1]
    else:
        # Diagnostic fallback only: still try to recover bracketed tool payloads.
        tool_body = before_end
        if llm_match:
            tool_body = tool_body[llm_match.end():]

    explicit_empty = tool_empty_token in tool_body
    generated_chunks = re.findall(r"<+\s*([^<>\n\r]+?)\s*>+", tool_body)

    restored_tools: List[str] = []
    seen = set()
    trace: List[Dict[str, Any]] = []

    for chunk in generated_chunks:
        chunk = str(chunk or "").strip()
        if not chunk:
            continue
        if chunk in {"TOOL_SEP", "TOOL_EMPTY", "SPECIAL_END"}:
            continue
        if chunk.startswith("LLM_"):
            continue

        payload = _canonical_component_payload(chunk, strip_tool_prefix=True)
        if not payload:
            continue

        exact_matches = tool_lookup.get(payload, [])
        if len(exact_matches) == 1:
            restored = exact_matches[0]
            status = "exact_payload_inventory_match"
        elif len(exact_matches) > 1:
            # Extremely unlikely duplicate canonical payload.  Do not guess which
            # inventory token was intended; retain it as unmatched/ambiguous.
            restored = _serialize_unknown_tool_payload(payload)
            status = "ambiguous_inventory_payload"
        else:
            restored = _serialize_unknown_tool_payload(payload)
            status = "unmatched_generated_payload"

        trace.append({
            "generated_chunk": chunk,
            "canonical_payload": payload,
            "restored_identifier": restored,
            "status": status,
            "inventory_matches": exact_matches,
        })

        if restored and restored not in seen:
            seen.add(restored)
            restored_tools.append(restored)

    # Respect an explicit TOOL_EMPTY only when no tool payload was actually emitted.
    if explicit_empty and not restored_tools:
        restored_tools = []

    meta = {
        "parser": "wrapper_insensitive_exact_payload_v1",
        "policy": "strip_outer_angles_and_optional_TOOL_prefix_only; exact_inventory_payload_match; no_fuzzy_repair",
        "generated_llm_payload": llm_generated,
        "restored_llm_token": llm_token,
        "explicit_tool_empty": explicit_empty,
        "generated_tool_chunk_count": len(generated_chunks),
        "restored_tool_count": len(restored_tools),
        "tool_trace": trace,
    }
    return llm_token, restored_tools, meta


def _format_diagnostics(
    *,
    text: str,
    llm_token: str,
    tool_tokens: Sequence[str],
    context_llms: Sequence[str],
    context_tools: Sequence[str],
    tool_sep_token: str,
    end_token: str,
    tool_empty_token: str,
    max_tools: int,
) -> Dict[str, Any]:
    """Measure formatting/grounding without repairing the model output."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    llm_set = set(context_llms)
    tool_set = set(context_tools)
    parsed_tools = [t for t in tool_tokens if t and t != tool_empty_token]

    has_sep = tool_sep_token in normalized
    has_end = end_token in normalized
    has_empty = tool_empty_token in normalized

    hallucinated_llm = bool(llm_token) and llm_token not in llm_set
    hallucinated_tools = [t for t in parsed_tools if t not in tool_set]
    llm_grounded = bool(llm_token) and not hallucinated_llm
    tools_grounded = len(hallucinated_tools) == 0

    # Reconstruct the exact accepted serialization and compare after whitespace
    # normalization. This checks format only; grounding is reported separately.
    if llm_token:
        body_tools = parsed_tools if parsed_tools else ([tool_empty_token] if has_empty else [])
        reconstructed = " ".join(
            [llm_token, tool_sep_token] + body_tools + [end_token]
        ) if body_tools else ""
    else:
        reconstructed = ""
    exact_format_valid = bool(reconstructed) and normalized == reconstructed

    return {
        "exact_format_valid": exact_format_valid,
        "has_tool_sep": has_sep,
        "has_special_end": has_end,
        "explicit_tool_empty": has_empty,
        "parsed_llm": llm_token,
        "parsed_tool_count": len(parsed_tools),
        "within_requested_max_tools": len(parsed_tools) <= int(max_tools),
        "llm_grounded": llm_grounded,
        "tools_grounded": tools_grounded,
        "fully_grounded": llm_grounded and tools_grounded,
        "hallucinated_llm": llm_token if hallucinated_llm else "",
        "hallucinated_tools": hallucinated_tools,
        "hallucinated_components": ([llm_token] if hallucinated_llm else []) + hallucinated_tools,
        "context_llm_count": len(context_llms),
        "context_tool_count": len(context_tools),
    }


def _load_base_model(model_dir: str, *, torch_dtype: str, device_map: str):
    key = (str(Path(model_dir).resolve()), torch_dtype, device_map)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key][0], _MODEL_CACHE[key][1], True

    dtype_obj: Any = "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype_obj,
        device_map=device_map,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model, False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RAG-Llama baseline: SOTA-aligned retrieval + unconstrained one-shot Llama generation"
    )

    # Shared SOTA implementation.
    ap.add_argument("--sota_script", type=str, default="")

    # Batch protocol.
    ap.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    ap.add_argument("--sample_size", type=int, default=100)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--generation_seed", type=int, default=42)
    ap.add_argument("--experiment_dir", type=str, default=DEFAULT_EXP_DIR)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--fail_fast", type=int, default=0)

    # Model: intentionally the unadapted backbone, not generative_v9 SFT.
    ap.add_argument("--model_dir", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--torch_dtype", type=str, default="auto")
    ap.add_argument("--device_map", type=str, default="auto")

    # Retrieval: defaults mirror the supplied v13 SOTA launcher.
    ap.add_argument('--retrieval_mode', choices=["hybrid", "cf", "semantic", "auto"], default="hybrid")
    ap.add_argument("--cf_llm_retr_url", type=str, default="http://127.0.0.1:9000/predict")
    ap.add_argument("--cf_tool_retr_url", type=str, default="http://127.0.0.1:9001/predict")
    ap.add_argument("--semantic_retr_url", type=str, default="http://127.0.0.1:8504/retrieve")
    ap.add_argument('--retrieval_timeout', type=int, default=300)
    ap.add_argument("--allow_partial_retrieval", type=int, default=0)
    ap.add_argument("--retrieval_json_file", type=str, default="")
    ap.add_argument("--retrieval_extra_json", type=str, default="")

    ap.add_argument("--llm_topk", type=int, default=10)
    ap.add_argument("--cf_tool_bundle_topk", type=int, default=5)
    ap.add_argument("--semantic_targets", type=str, default="tool")
    ap.add_argument("--semantic_llm_topk", type=int, default=0)
    ap.add_argument("--semantic_tool_topk", type=int, default=25)
    ap.add_argument("--cf_rewrite", type=int, default=0)
    ap.add_argument("--semantic_rewrite", type=int, default=0)
    ap.add_argument("--agg", choices=["max", "mean", "hybrid"], default="max")
    ap.add_argument("--candidate_multiplier", type=int, default=5)
    ap.add_argument("--include_raw", type=int, default=0)
    ap.add_argument("--include_doc_text", type=int, default=0)
    ap.add_argument("--tool_recall_mode", type=int, default=0)
    ap.add_argument("--per_subquery_top_k", type=int, default=0)
    ap.add_argument("--final_top_k", type=int, default=25)

    # Free generation. Keep one deterministic recommendation per query.
    ap.add_argument("--max_source_length", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_tools", type=int, default=6)
    ap.add_argument("--tool_sep_token", type=str, default="<TOOL_SEP>")
    ap.add_argument("--tool_empty_token", type=str, default="<TOOL_EMPTY>")
    ap.add_argument("--end_token", type=str, default="<SPECIAL_END>")
    ap.add_argument("--progress", type=int, default=1)
    ap.add_argument("--use_chat_template", type=int, default=1, help="Use tokenizer chat template when available (recommended for Instruct models).")
    ap.add_argument("--require_chat_template", type=int, default=1, help="Fail closed on a base LM; recommended for the main RAG-Llama baseline.")
    ap.add_argument("--dry_run", type=int, default=0)
    return ap.parse_args()


def _ensure_sota_retrieval_namespace(args: argparse.Namespace) -> None:
    """Add legacy/shared attributes read by SOTA retrieval helper code."""
    # retrieve_and_build_context reads these fields from its Namespace.
    defaults = {
        "rewrite": 0,
        "tool_topk": -1,
        'retrieval_url': "",
        "legacy_llm_retr_url": "",
        "legacy_tool_retr_url": "",
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def run_one(args: argparse.Namespace, sota, sample: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    query = str(sample.get("query") or "").strip()
    if not query:
        raise ValueError("Sample has an empty conversation history")

    _set_all_seeds(int(args.generation_seed))
    started = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    # Exact same retrieval/context function as the SOTA implementation.
    context, context_meta, retrieval_record = sota.retrieve_and_build_context(args, query)
    context_llms, context_tools = sota.extract_candidates_from_context(context)

    messages = build_rag_llama_messages(
        context=context,
        conversation_history=query,
        max_tools=int(args.max_tools),
        context_llms=context_llms,
        context_tools=context_tools,
    )
    prompt = build_rag_llama_plain_prompt(messages)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "baseline": "RAG-Llama-Instruct",
        "policy": "sota_aligned_retrieval_plus_instruct_prompt_only_greedy_generation",
        "query": query,
        "conversation_history": query,
        "config": {
            "model_dir": args.model_dir,
            "task_adapter": None,
            "sample_seed": int(args.sample_seed),
            "generation_seed": int(args.generation_seed),
            'retrieval_mode': args.retriever_mode,
            "cf_llm_topk": int(args.llm_topk),
            "cf_tool_bundle_topk": int(args.cf_tool_bundle_topk),
            "semantic_targets": args.semantic_targets,
            "semantic_llm_topk": int(args.semantic_llm_topk),
            "semantic_tool_topk": int(args.semantic_tool_topk),
            "cf_rewrite": bool(args.cf_rewrite),
            "semantic_rewrite": bool(args.semantic_rewrite),
            "final_top_k": int(args.final_top_k),
            "max_source_length": int(args.max_source_length),
            "max_new_tokens": int(args.max_new_tokens),
            "requested_max_tools": int(args.max_tools),
            "controlled": False,
            "num_beams": 1,
            "do_sample": False,
            "num_return_sequences": 1,
            "critic": False,
            "use_chat_template": bool(args.use_chat_template),
        },
        "retrieval": retrieval_record,
        "context": {"text": context, "meta": context_meta},
        "prompt": {"text": prompt, "answer_mode": "target_only"},
        "dataset_example": sample,
    }
    _write_json(output_path, record)

    if bool(args.dry_run):
        record["generation"] = {"skipped": True, "reason": "dry_run"}
        record["finished_at"] = _now_iso()
        _write_json(output_path, record)
        return record

    model_started = time.time()
    tokenizer, model, reused = _load_base_model(
        args.model_dir,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    record["model_loading"] = {
        "reused_from_batch_cache": reused,
        "latency_sec": round(time.time() - model_started, 4),
        "finished_at": _now_iso(),
    }

    use_chat_template = bool(args.use_chat_template) and bool(getattr(tokenizer, "chat_template", None))
    if bool(args.require_chat_template) and not use_chat_template:
        raise RuntimeError(
            "RAG-Llama v4 requires an instruction/chat checkpoint with a tokenizer chat template. "
            "The current model appears to be a base LM. Use Meta-Llama-3-8B-Instruct (recommended), "
            "or explicitly pass --require_chat_template 0 for the weaker base-LM diagnostic."
        )
    if use_chat_template:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=int(args.max_source_length),
        )
    else:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_source_length),
        )
    record["prompt"]["chat_template_used"] = use_chat_template
    record["prompt"]["message_roles"] = [m["role"] for m in messages]
    if hasattr(model, "device") and str(model.device) != "cpu":
        enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    record["prompt"]["input_token_count"] = prompt_len
    record["prompt"]["truncated"] = bool(prompt_len >= int(args.max_source_length))

    gen_started = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    suffix = out_ids[:, prompt_len:]
    raw_decoded = tokenizer.batch_decode(suffix, skip_special_tokens=False)[0]
    cleaned_raw = _strip_model_special_tokens(raw_decoded, tokenizer)
    eval_text = _trim_after_first_end(cleaned_raw, args.end_token)

    # Evaluation-side protocol normalization only.  We ignore wrapper mistakes
    # (<tool>, <<tool>>, <TOOL_tool>) but require an EXACT canonical payload
    # match to restore a retrieved identifier.  Hallucinated names are retained.
    llm_tok, tool_toks, relaxed_parse = parse_structured_tokens_relaxed_protocol(
        eval_text,
        context_llms=context_llms,
        context_tools=context_tools,
        tool_sep_token=args.tool_sep_token,
        end_token=args.end_token,
        tool_empty_token=args.tool_empty_token,
    )
    diagnostics = _format_diagnostics(
        text=eval_text,
        llm_token=llm_tok,
        tool_tokens=tool_toks,
        context_llms=context_llms,
        context_tools=context_tools,
        tool_sep_token=args.tool_sep_token,
        end_token=args.end_token,
        tool_empty_token=args.tool_empty_token,
        max_tools=int(args.max_tools),
    )

    # Compatibility field only.  Exact wrapper-insensitive payload matches may
    # restore the retrieved identifier; unknown/hallucinated names are never
    # fuzzily mapped to another candidate. Missing parses remain EMPTY markers.
    strict_text = sota.build_strict_text(
        llm_token=llm_tok,
        tool_tokens=tool_toks,
        tool_sep_token=args.tool_sep_token,
        end_token=args.end_token,
        tool_empty_token=args.tool_empty_token,
    )
    result = {
        "rank": 1,
        "gen_text": eval_text,
        "raw_decoded": raw_decoded,
        "raw_cleaned": cleaned_raw,
        "llm_token": llm_tok,
        "tool_tokens": list(tool_toks) if tool_toks else [args.tool_empty_token],
        "strict_text": strict_text,
        "protocol_normalization": relaxed_parse,
        "validation": diagnostics,
    }

    record["generation"] = {
        "started_at": _now_iso(),
        "mode": "free_autoregressive_greedy",
        "controlled": False,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "latency_sec": round(time.time() - gen_started, 4),
        "raw_output": raw_decoded,
        "results": [result],
    }
    record["results"] = [result]
    record["finished_at"] = _now_iso()
    record["latency_sec"] = round(time.time() - started, 4)
    _write_json(output_path, record)
    return record


def _read_completed_sample_ids(results_path: Path) -> set:
    done = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("ok", True):
                continue
            sample = row.get("dataset_example", {})
            sid = sample.get("sample_id") if isinstance(sample, dict) else None
            if sid:
                done.add(str(sid))
    return done


def run_batch(args: argparse.Namespace, sota) -> Dict[str, Any]:
    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    per_sample = exp_dir / "per_sample"
    per_sample.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.jsonl"
    failures_path = exp_dir / "failures.jsonl"
    summary_path = exp_dir / "run_summary.json"

    # Reuse the SOTA's exact manifest/sampling implementation.
    manifest = sota._prepare_sample_manifest(args, exp_dir)
    completed = _read_completed_sample_ids(results_path) if bool(args.resume) else set()
    pending = [x for x in manifest if str(x["sample_id"]) not in completed]

    metadata_path = exp_dir / "baseline_config.json"
    _write_json(metadata_path, {
        "baseline": "RAG-Llama",
        "created_at": _now_iso(),
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "sample_size": int(args.sample_size),
        "sample_seed": int(args.sample_seed),
        "generation_seed": int(args.generation_seed),
        "model_dir": args.model_dir,
        "query_view": "full_conversation_history",
        "retrieval_alignment": {
            'retrieval_mode': args.retriever_mode,
            "cf_llm_topk": int(args.llm_topk),
            "cf_tool_bundle_topk": int(args.cf_tool_bundle_topk),
            "semantic_targets": args.semantic_targets,
            "semantic_tool_topk": int(args.semantic_tool_topk),
            "semantic_rewrite": bool(args.semantic_rewrite),
            "final_top_k": int(args.final_top_k),
        },
        "generation": {
            "free_generation": True,
            "controlled": False,
            "num_beams": 1,
            "do_sample": False,
            "num_return_sequences": 1,
            "critic": False,
            "task_adapter": None,
            "use_chat_template_when_available": bool(args.use_chat_template),
            "require_chat_template": bool(args.require_chat_template),
        },
    })

    print(
        f"[RAG-Llama] samples={len(manifest)} completed={len(completed)} "
        f"pending={len(pending)} seed={args.sample_seed}"
    )
    print(
        "[RAG-Llama] retrieval: "
        f"CF-LLM={args.llm_topk}, CF-bundles={args.cf_tool_bundle_topk}, "
        f"semantic-tools={args.semantic_tool_topk}, rewrite={args.semantic_rewrite}"
    )
    print("[RAG-Llama] generation: FREE greedy; constrained=OFF beam=OFF critic=OFF")

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(pending, total=len(pending), desc="RAG-Llama", unit="sample", dynamic_ncols=True)
    except Exception:
        iterator = pending

    ok_now = 0
    failed_now = 0
    started_at = _now_iso()
    for sample in iterator:
        sid = str(sample["sample_id"])
        qid = str(sample.get("qid") or "no_qid")
        safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)[:80]
        out_path = per_sample / f"{sid}_{safe_qid}.json"
        t0 = time.time()
        try:
            record = run_one(args, sota, sample, out_path)
            record["ok"] = True
            record["batch"] = {
                "experiment_dir": str(exp_dir),
                "sample_latency_sec": round(time.time() - t0, 4),
                "completed_at": _now_iso(),
            }
            _write_json(out_path, record)
            _append_jsonl(results_path, record)
            ok_now += 1
        except Exception as exc:
            failed_now += 1
            failure = {
                "ok": False,
                "sample_id": sid,
                "qid": sample.get("qid"),
                "query": sample.get("query"),
                "target": sample.get("target"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "latency_sec": round(time.time() - t0, 4),
                "finished_at": _now_iso(),
            }
            _append_jsonl(failures_path, failure)
            print(f"[RAG-Llama][ERROR] {sid} qid={qid}: {exc!r}", file=sys.stderr, flush=True)
            if bool(args.fail_fast):
                raise
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(ok=ok_now, failed=failed_now, refresh=True)

        _write_json(summary_path, {
            "started_at": started_at,
            "updated_at": _now_iso(),
            "manifest_samples": len(manifest),
            "already_completed_at_start": len(completed),
            "pending_at_start": len(pending),
            "succeeded_this_invocation": ok_now,
            "failed_this_invocation": failed_now,
            "remaining_estimate": len(pending) - ok_now - failed_now,
            "results_jsonl": str(results_path),
            "failures_jsonl": str(failures_path),
        })

    summary = {
        "ok": failed_now == 0,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "manifest_samples": len(manifest),
        "already_completed_at_start": len(completed),
        "succeeded_this_invocation": ok_now,
        "failed_this_invocation": failed_now,
        "results_jsonl": str(results_path),
        "failures_jsonl": str(failures_path),
        "per_sample_dir": str(per_sample),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    _ensure_sota_retrieval_namespace(args)
    sota_path = _resolve_sota_script(args.sota_script)
    sota = _load_sota_module(sota_path)
    print(f"[RAG-Llama] shared SOTA retrieval module: {sota_path}")
    print(f"[RAG-Llama] input: {args.input_jsonl}")
    print(f"[RAG-Llama] model: {args.model_dir}")
    summary = run_batch(args, sota)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()