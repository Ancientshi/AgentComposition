#!/usr/bin/env python3
"""Baseline5: replay baseline4 inputs, replacing only generation with a GPT API.

No retrieval calls, Llama weights, gold labels or baseline4 outputs go to the API.
The Llama tokenizer is used only to reproduce baseline4's input truncation.
Use --dry_run to audit every input without any network request.
"""
from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = AC_ROOT
DEFAULT_SOURCE = ROOT / 'outputs/baseline4_rag_llama_instruct_seed42_n100'
ALLOWED_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.4-2026-03-05", "gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17")


def baseline_name(model):
    if model.startswith("gpt-5.6-"):
        return "RAG-GPT-5.6-" + model.removeprefix("gpt-5.6-").capitalize()
    return "RAG-" + model.replace("gpt-", "GPT-", 1)


def default_output(model):
    return ROOT / f"EXP/baseline5_rag_{model.replace('-', '_').replace('.', '_')}_seed42_n100"


def ensure_env_cuda_library():
    """Re-exec once so the dynamic loader prefers this environment's nvJitLink."""
    nvjitlink = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/nvjitlink/lib"
    if not (nvjitlink / "libnvJitLink.so.12").is_file():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current.split(":")[0] == str(nvjitlink):
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(nvjitlink) + (":" + current if current else "")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def reference_inputs(source):
    manifest = read_jsonl(source / "sample_manifest.jsonl")
    records = read_jsonl(source / "results.jsonl")
    by_id = {}
    for record in records:
        sid = record["dataset_example"]["sample_id"]
        if sid in by_id:
            raise ValueError(f"Duplicate baseline4 result: {sid}")
        if not record.get("ok", True) or not record.get("results"):
            raise ValueError(f"Incomplete baseline4 result: {sid}")
        by_id[sid] = record
    if len({s["sample_id"] for s in manifest}) != len(manifest):
        raise ValueError("Duplicate manifest IDs")
    if set(by_id) != {s["sample_id"] for s in manifest}:
        raise ValueError("Baseline4 results and manifest IDs differ")
    ordered = []
    for sample in manifest:
        row = by_id[sample["sample_id"]]
        if row["dataset_example"] != sample:
            raise ValueError("Baseline4 result/manifest sample mismatch")
        ordered.append(row)
    return manifest, ordered


def replay_prompt(row, b4, sota, tokenizers):
    context = row["context"]["text"]
    llms, tools = sota.extract_candidates_from_context(context)
    messages = b4.build_rag_llama_messages(
        context=context, conversation_history=row["query"],
        max_tools=row["config"]["requested_max_tools"],
        context_llms=llms, context_tools=tools,
    )
    if b4.build_rag_llama_plain_prompt(messages) != row["prompt"]["text"]:
        raise ValueError("Current baseline4 prompt builder differs from saved experiment")
    cfg = row["config"]
    model_dir = cfg["model_dir"]
    if model_dir not in tokenizers:
        from transformers import AutoTokenizer
        tokenizers[model_dir] = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok = tokenizers[model_dir]
    if not row["prompt"]["chat_template_used"]:
        raise ValueError("Expected baseline4 Instruct chat template")
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                  truncation=True, max_length=cfg["max_source_length"])
    if len(ids) != row["prompt"]["input_token_count"]:
        raise ValueError("Reference tokenizer no longer reproduces baseline4 input length")
    visible = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not rendered.startswith(visible):
        raise ValueError("Cannot reproduce baseline4 prefix truncation exactly")
    sent, offset = [], 0
    for message in messages:
        start = rendered.index(message["content"], offset)
        end = start + len(message["content"])
        content = visible[start:min(end, len(visible))] if start < len(visible) else ""
        if content:
            sent.append({"role": message["role"], "content": content})
        offset = end
    audit = {
        "reference_input_token_count": len(ids),
        "reference_max_source_length": cfg["max_source_length"],
        "reference_truncated": row["prompt"]["truncated"],
        "content_truncated": sent != messages,
        "input_truncation": "same visible message content as baseline4 Llama tokenizer",
        "api_messages_sha256": digest(sent),
        "source_prompt_sha256": digest(row["prompt"]["text"]),
    }
    return sent, llms, tools, audit


def build_payload(args, messages):
    common = {"model": args.model, "temperature": 0, "stream": False, "store": False}
    if args.api_mode == "responses":
        return {**common, "input": messages, "max_output_tokens": args.max_new_tokens,
                "reasoning": {"effort": "none"}}
    return {**common, "messages": messages, "max_completion_tokens": args.max_new_tokens,
            "reasoning_effort": "none", "n": 1}


def parse_response(body, api_mode, model):
    actual_model = body.get("model")
    if actual_model and actual_model != model and not actual_model.startswith(model + "-"):
        raise ValueError(f"Proxy returned a different model: {actual_model!r}")
    if body.get("error"):
        raise ValueError("API returned an error object")
    if api_mode == "responses":
        if body.get("status") not in ("completed", "incomplete"):
            raise ValueError(f"Unexpected response status: {body.get('status')!r}")
        if not isinstance(body.get("output"), list):
            raise ValueError("Missing API output array")
        parts, refusals = [], []
        for item in body["output"]:
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal", ""))
        text = "".join(parts)
        finish = body.get("incomplete_details") or body.get("status")
    else:
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected exactly one chat completion")
        message = choices[0]["message"]
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise ValueError("Expected text chat content")
        refusals = [message["refusal"]] if message.get("refusal") else []
        finish = choices[0].get("finish_reason")
    return text, {"response_id": body.get("id"), "requested_model": model, "returned_model": actual_model,
                  "usage": body.get("usage"), "finish_reason": finish, "refusals": refusals}


def call_api(args, messages):
    endpoint = args.api_base_url.rstrip("/") + ("/responses" if args.api_mode == "responses" else "/chat/completions")
    payload = build_payload(args, messages)
    key = os.environ.get("OPENAI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    # Never inherit global HTTP(S)_PROXY: only this API call uses the explicit tunnel.
    proxies = {"http": args.http_proxy, "https": args.http_proxy} if args.http_proxy else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    for attempt in range(args.api_retries + 1):
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with opener.open(request, timeout=args.api_timeout) as response:
                body = json.load(response)
            return parse_response(body, args.api_mode, args.model)
        except urllib.error.HTTPError as exc:
            # Do not persist error bodies/headers; a proxy may echo credentials.
            retryable = exc.code in (408, 429, 500, 502, 503, 504)
            if not retryable or attempt == args.api_retries:
                raise RuntimeError(f"API HTTP {exc.code}; check endpoint, model access and authentication") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == args.api_retries:
                raise RuntimeError("API connection failed; check local proxy and SSH reverse tunnel") from None
        time.sleep(min(2 ** attempt, 10))


def make_record(row, messages, llms, tools, audit, text, api, b4, sota, config_hash):
    cleaned = re.sub(r"\s+", " ", text).strip()
    evaluated = b4._trim_after_first_end(cleaned, "<SPECIAL_END>")
    llm, selected, normalization = b4.parse_structured_tokens_relaxed_protocol(
        evaluated, context_llms=llms, context_tools=tools,
        tool_sep_token="<TOOL_SEP>", end_token="<SPECIAL_END>", tool_empty_token="<TOOL_EMPTY>")
    strict = sota.build_strict_text(llm_token=llm, tool_tokens=selected,
                                   tool_sep_token="<TOOL_SEP>", end_token="<SPECIAL_END>", tool_empty_token="<TOOL_EMPTY>")
    result = {"rank": 1, "gen_text": evaluated, "raw_decoded": text, "raw_cleaned": cleaned,
              "llm_token": llm, "tool_tokens": selected or ["<TOOL_EMPTY>"], "strict_text": strict,
              "protocol_normalization": normalization,
              "validation": b4._format_diagnostics(text=evaluated, llm_token=llm, tool_tokens=selected,
                  context_llms=llms, context_tools=tools, tool_sep_token="<TOOL_SEP>",
                  end_token="<SPECIAL_END>", tool_empty_token="<TOOL_EMPTY>",
                  max_tools=row["config"]["requested_max_tools"])}
    return {"ok": True, "baseline": baseline_name(api["requested_model"]), "config_sha256": config_hash,
            "dataset_example": row["dataset_example"], "query": row["query"],
            "context": row["context"], "retrieval": row["retrieval"],
            "prompt": {"messages": messages, **audit}, "api": api,
            "generation": {"mode": "api_temperature_zero_reasoning_none", "num_return_sequences": 1,
                           "raw_output": text, "results": [result]}, "results": [result]}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_experiment", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--experiment_dir", type=Path, default=None)
    ap.add_argument("--api_base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18080/v1"))
    ap.add_argument("--http_proxy", default=os.environ.get("BASELINE5_HTTP_PROXY", ""))
    ap.add_argument("--api_mode", choices=["responses", "chat_completions"], default="responses")
    ap.add_argument("--model", choices=ALLOWED_MODELS, default="gpt-5.6-luna")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--api_timeout", type=float, default=180)
    ap.add_argument("--api_retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 replays all source samples")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.experiment_dir is None:
        args.experiment_dir = default_output(args.model)
    if args.limit < 0 or args.api_retries < 0 or args.api_timeout <= 0:
        ap.error("Invalid limit/retry/timeout")
    for url in [args.api_base_url] + ([args.http_proxy] if args.http_proxy else []):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query:
            ap.error("Use HTTP(S) URLs without embedded credentials or query parameters")
    if args.http_proxy and not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        ap.error("HTTP forward-proxy mode requires OPENAI_API_KEY in the environment")
    return args


def main():
    args = parse_args()
    source, output = args.source_experiment.resolve(), args.experiment_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate")
    b4 = load_module(ROOT / 'baselines/baseline4_run_infer_rag_llama.py', "baseline4_shared")
    sota_path = b4._resolve_sota_script()
    sota = b4._load_sota_module(sota_path)
    manifest, rows = reference_inputs(source)
    if args.limit:
        rows, manifest = rows[:args.limit], manifest[:args.limit]
    for row in rows:
        if row["config"]["max_new_tokens"] != args.max_new_tokens:
            raise ValueError("Output token budget must match baseline4")
    config = {"baseline": baseline_name(args.model), "source_experiment": str(source),
              "source_results_sha256": file_digest(source / "results.jsonl"),
              "source_manifest_sha256": file_digest(source / "sample_manifest.jsonl"),
              "baseline4_code_sha256": file_digest(ROOT / 'baselines/baseline4_run_infer_rag_llama.py'),
              "sota_code_sha256": file_digest(sota_path),
              "runner_sha256": file_digest(__file__),
              "model": args.model, "api_base_url": args.api_base_url, "http_proxy": args.http_proxy,
              "api_mode": args.api_mode, "temperature": 0, "reasoning_effort": "none",
              "max_output_tokens": args.max_new_tokens, "sample_count": len(rows),
              "limit": args.limit, "dry_run": args.dry_run,
              "api_seed": None, "api_determinism_guaranteed": False,
              "protocol": "baseline4 exact manifest/context/prompt content/truncation/parser/evaluator"}
    config_hash = digest(config)
    config_file = output / "baseline_config.json"
    if config_file.exists():
        stored_config = json.loads(config_file.read_text())
        if stored_config != config:
            stable = lambda c: {k: v for k, v in c.items() if k != "runner_sha256"}
            summary_file = output / "run_summary.json"
            summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            complete = (summary.get("ok") and summary.get("expected_samples") == len(rows)
                        and summary.get("completed_samples") == len(rows)
                        and len(list((output / "per_sample").glob("*.json"))) == len(rows))
            if args.model == "gpt-5.6-luna" and stable(stored_config) == stable(config) and complete:
                print(f"[resume] Existing Luna experiment is complete and preserved: {output}")
                return 0
            raise ValueError("Output configuration changed; use a new experiment_dir to avoid mixed results")
    if output.exists() and not config_file.exists() and any(output.glob("per_sample/*.json")):
        raise ValueError("Existing predictions have no configuration; use a new output directory")
    write_json(config_file, config)
    (output / "sample_manifest.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in manifest))
    tokenizers, prepared, audits = {}, [], []
    # Validate ALL prompts before spending API tokens.
    for row in rows:
        messages, llms, tools, audit = replay_prompt(row, b4, sota, tokenizers)
        prepared.append((row, messages, llms, tools, audit))
        audits.append({"sample_id": row["dataset_example"]["sample_id"], **audit})
    write_json(output / "alignment_audit.json", {"samples": len(rows), "source": str(source),
               "truncated_samples": sum(x["content_truncated"] for x in audits), "per_sample": audits})
    print(f"[alignment] {len(rows)} exact source samples; {sum(x['content_truncated'] for x in audits)} truncated prompts", flush=True)
    if args.dry_run:
        print("[done] dry_run: no API requests, no predictions or scores")
        return 0
    results, failures = [], []
    for row, messages, llms, tools, audit in prepared:
        sid = row["dataset_example"]["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sid):
            raise ValueError("Unsafe sample ID")
        path = output / "per_sample" / (sid + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("config_sha256") != config_hash or not cached.get("ok") or cached.get("dataset_example") != row["dataset_example"]:
                raise ValueError(f"Invalid resume record: {sid}")
            results.append(cached)
            continue
        started = time.time()
        try:
            text, api = call_api(args, messages)
            record = make_record(row, messages, llms, tools, audit, text, api, b4, sota, config_hash)
            record["latency_sec"] = round(time.time() - started, 4)
            write_json(path, record)
            results.append(record)
            print(f"[ok] {sid} ({len(results)}/{len(rows)})", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": str(exc)})
            print(f"[ERROR] {sid}: {exc}", file=sys.stderr, flush=True)
            break  # Fail early on tunnel/auth errors; resume retries this sample.
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    summary = {"ok": len(results) == len(rows), "expected_samples": len(rows),
               "completed_samples": len(results), "failures": failures, "output": str(output)}
    write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    ensure_env_cuda_library()
    raise SystemExit(main())
