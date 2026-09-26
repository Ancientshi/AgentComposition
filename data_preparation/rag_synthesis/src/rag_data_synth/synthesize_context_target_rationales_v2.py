#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from tqdm import tqdm

try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires requests: pip install requests") from e

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ExplainConfig:
    base_url: str
    model: str
    api_key: str
    timeout: int
    max_retries: int
    retry_sleep: float
    temperature: float
    max_tokens: int
    token_limit_param: str
    max_context_chars: int
    max_query_chars: int
    max_target_chars: int
    max_sentences: int
    field_name: str
    error_field_name: str
    meta_field_name: str
    require_context: bool
    require_target: bool


def clip_text(x: Any, n: int) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    s = "\n".join(line.rstrip() for line in x.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    s = s.strip()
    if n and n > 0 and len(s) > n:
        return s[: max(0, n - 20)].rstrip() + "\n...[truncated]"
    return s


def iter_jsonl(path: Path, limit: int = 0) -> Iterator[Tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit and line_no > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL row is not an object")
                yield line_no, obj
            except Exception as e:
                yield line_no, {"__parse_error__": repr(e), "__raw_line__": line[:1000]}


def row_key(row: JsonDict, line_no: Optional[int] = None) -> Tuple[str, str, str, str]:
    """Stable enough key for resume. qid/agent_id/target/query usually identify a row."""
    qid = str(row.get("qid", ""))
    aid = str(row.get("agent_id", ""))
    target = str(row.get("target", ""))
    query = str(row.get("query", ""))[:300]
    if not any([qid, aid, target, query]) and line_no is not None:
        return ("__line__", str(line_no), "", "")
    return (qid, aid, target, query)


def load_done_keys(output_path: Path, field_name: str, error_field_name: str, meta_field_name: str) -> set:
    done = set()
    if not output_path.is_file():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            # Rows with error are intentionally NOT done, so rerun can retry them.
            if row.get(error_field_name):
                continue
            if not str(row.get(field_name, "")).strip():
                continue
            meta = row.get(meta_field_name) if isinstance(row.get(meta_field_name), dict) else {}
            line_no = meta.get("source_line_no")
            done.add(row_key(row, line_no=line_no))
    return done


def clean_explanation(text: str, max_sentences: int) -> str:
    text = (text or "").strip()
    # Remove common wrappers if the model returns JSON-ish or markdown labels.
    text = re.sub(r"^```(?:json|text|markdown)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            for k in ["explanation", "rationale", "selection_explanation", "target_context_explanation"]:
                if isinstance(obj, dict) and isinstance(obj.get(k), str):
                    text = obj[k].strip()
                    break
        except Exception:
            pass
    text = re.sub(r"\s+", " ", text).strip()
    # Keep it compact. This is a light sentence limiter, not a parser.
    if max_sentences and max_sentences > 0:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) > max_sentences:
            text = " ".join(parts[:max_sentences]).strip()
    return text


def build_messages(query: str, context: str, target: str, max_sentences: int) -> List[JsonDict]:
    system = (
        "You write concise, context-grounded rationales for an agent recommendation dataset. "
        "Use the retrieved context as evidence, and explain why the final target is selected for the query. "
        "You may mention one or two backup alternatives from the retrieved context if they are also plausible, "
        "but make clear why the target remains the best final choice. "
        "Do not invent components that are not in the target or retrieved context. "
        "Do not mention gold labels, injection, training data, retrieval failure, or hidden metadata. "
        "Return plain English only, not JSON, not markdown bullets."
    )
    user = f"""Query:
{query}

Retrieved context:
{context}

Final target:
{target}

Task:
Write one concise paragraph within {max_sentences} sentences.
Required structure:
1. State the core capability required by the query.
2. Explain why the final target is selected, using evidence from the retrieved context.
3. Briefly explain why one or two backup choices from the context could also be reasonable, but why the final target is preferable overall.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _make_payload(messages: List[JsonDict], cfg: ExplainConfig, token_param: str, include_temperature: bool) -> JsonDict:
    payload: JsonDict = {
        "model": cfg.model,
        "messages": messages,
    }
    if include_temperature:
        payload["temperature"] = cfg.temperature
    if token_param == "max_completion_tokens":
        payload["max_completion_tokens"] = cfg.max_tokens
    elif token_param == "max_tokens":
        payload["max_tokens"] = cfg.max_tokens
    else:
        raise ValueError(f"Unknown token limit parameter: {token_param}")
    return payload


def _token_param_order(cfg: ExplainConfig) -> List[str]:
    mode = (cfg.token_limit_param or "auto").strip().lower()
    if mode == "max_completion_tokens":
        return ["max_completion_tokens"]
    if mode == "max_tokens":
        return ["max_tokens"]
    # For GPT-5/o-series style endpoints, max_completion_tokens is required.
    # Keep max_tokens as fallback for older OpenAI-compatible servers.
    return ["max_completion_tokens", "max_tokens"]


def _extract_content(data: JsonDict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {json.dumps(data, ensure_ascii=False)[:1000]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        # Some local servers use text instead of message.content.
        content = choices[0].get("text", "")
    if isinstance(content, list):
        # Some newer APIs return typed content blocks.
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        content = "\n".join(x for x in parts if x)
    content = str(content or "").strip()
    if not content:
        raise RuntimeError(f"Empty model response: {json.dumps(data, ensure_ascii=False)[:1000]}")
    return content


def call_chat_completion(messages: List[JsonDict], cfg: ExplainConfig) -> str:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    token_params = _token_param_order(cfg)
    include_temperature_options = [True, False] if cfg.temperature is not None else [False]

    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries + 1):
        for token_param in token_params:
            for include_temperature in include_temperature_options:
                payload = _make_payload(messages, cfg, token_param=token_param, include_temperature=include_temperature)
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
                    if resp.status_code >= 400:
                        body = resp.text[:2000]
                        lower = body.lower()
                        # Try next payload variant when the server rejects a compatibility parameter.
                        if (
                            "unsupported parameter" in lower
                            or "unsupported value" in lower
                            or "max_tokens" in lower
                            or "max_completion_tokens" in lower
                            or "temperature" in lower
                        ):
                            last_err = RuntimeError(f"HTTP {resp.status_code}: {body}")
                            continue
                        raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                    data = resp.json()
                    return _extract_content(data)
                except Exception as e:
                    last_err = e
                    # Network / JSON errors should not burn through all payload variants instantly.
                    break
        if attempt < cfg.max_retries:
            time.sleep(cfg.retry_sleep * (attempt + 1))
    raise RuntimeError(repr(last_err))

def explain_one(task: Tuple[int, JsonDict, Dict[str, Any]]) -> Dict[str, Any]:
    line_no, row, cfg_dict = task
    cfg = ExplainConfig(**cfg_dict)
    started = time.time()

    if "__parse_error__" in row:
        return {"ok": False, "line_no": line_no, "row": row, "error": row.get("__parse_error__", "parse error")}

    query = clip_text(row.get("query", ""), cfg.max_query_chars)
    context = clip_text(row.get("context", ""), cfg.max_context_chars)
    target = clip_text(row.get("target", ""), cfg.max_target_chars)

    if cfg.require_target and not target:
        return {"ok": False, "line_no": line_no, "row": row, "error": "missing target"}
    if cfg.require_context and not context:
        return {"ok": False, "line_no": line_no, "row": row, "error": "missing context"}
    if not query:
        return {"ok": False, "line_no": line_no, "row": row, "error": "missing query"}

    try:
        messages = build_messages(query=query, context=context, target=target, max_sentences=cfg.max_sentences)
        raw = call_chat_completion(messages, cfg)
        explanation = clean_explanation(raw, cfg.max_sentences)
        if not explanation:
            raise RuntimeError("empty cleaned explanation")
        out = dict(row)
        out[cfg.field_name] = explanation
        out.pop(cfg.error_field_name, None)
        out[cfg.meta_field_name] = {
            "source_line_no": line_no,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "used_fields": ["query", "context", "target"],
            "max_context_chars": cfg.max_context_chars,
            "max_sentences": cfg.max_sentences,
            "elapsed_sec": round(time.time() - started, 4),
        }
        return {"ok": True, "line_no": line_no, "row": out, "error": ""}
    except Exception as e:
        return {"ok": False, "line_no": line_no, "row": row, "error": repr(e)}


def write_jsonl_line(f, obj: JsonDict) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    cfg: ExplainConfig,
    workers: int,
    resume: bool,
    limit: int,
    write_failed_rows: bool,
    max_pending: int,
) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(output_path, cfg.field_name, cfg.error_field_name, cfg.meta_field_name) if resume else set()
    mode = "a" if resume and output_path.is_file() else "w"

    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "errors_path": str(errors_path),
        "workers": workers,
        "resume": resume,
        "n_seen": 0,
        "n_submitted": 0,
        "n_written": 0,
        "n_skipped_resume": 0,
        "n_errors": 0,
        "n_failed_rows_written": 0,
    }

    cfg_dict = asdict(cfg)
    pending: Dict[cf.Future, int] = {}

    def drain_done(executor_done: Iterable[cf.Future], fout, ferr) -> None:
        for fut in executor_done:
            pending.pop(fut, None)
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "line_no": -1, "row": {}, "error": repr(e)}
            if res.get("ok"):
                write_jsonl_line(fout, res["row"])
                stats["n_written"] += 1
            else:
                stats["n_errors"] += 1
                row = res.get("row") if isinstance(res.get("row"), dict) else {}
                err_obj = {
                    "line_no": res.get("line_no"),
                    "error": res.get("error", "unknown error"),
                    "qid": row.get("qid", ""),
                    "agent_id": row.get("agent_id", ""),
                    "query": str(row.get("query", ""))[:500],
                    "target": str(row.get("target", ""))[:500],
                }
                write_jsonl_line(ferr, err_obj)
                if write_failed_rows:
                    failed = dict(row)
                    failed[cfg.field_name] = ""
                    failed[cfg.error_field_name] = res.get("error", "unknown error")
                    failed[cfg.meta_field_name] = {
                        "source_line_no": res.get("line_no"),
                        "model": cfg.model,
                        "base_url": cfg.base_url,
                        "used_fields": ["query", "context", "target"],
                        "failed": True,
                    }
                    write_jsonl_line(fout, failed)
                    stats["n_failed_rows_written"] += 1

    with output_path.open(mode, encoding="utf-8") as fout, errors_path.open("a", encoding="utf-8") as ferr:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for line_no, row in tqdm(iter_jsonl(input_path, limit=limit), desc="Submitting rows"):
                stats["n_seen"] += 1
                if resume and row_key(row, line_no=line_no) in done:
                    stats["n_skipped_resume"] += 1
                    continue
                fut = ex.submit(explain_one, (line_no, row, cfg_dict))
                pending[fut] = line_no
                stats["n_submitted"] += 1

                if len(pending) >= max_pending:
                    done_futs, _ = cf.wait(pending.keys(), return_when=cf.FIRST_COMPLETED)
                    drain_done(done_futs, fout, ferr)

            while pending:
                done_futs, _ = cf.wait(pending.keys(), return_when=cf.FIRST_COMPLETED)
                drain_done(done_futs, fout, ferr)

    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate context-aware explanations for pair_rag JSONL rows using query, context, and target."
    )
    ap.add_argument("--input", required=True, help="Input pair_rag.jsonl")
    ap.add_argument("--output", required=True, help="Output JSONL with generated explanation field")
    ap.add_argument("--errors", default="", help="Error JSONL path; default output.errors.jsonl")
    ap.add_argument("--summary", default="", help="Summary JSON path")

    ap.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:18080/v1"))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL") or os.getenv("QUERY_REWRITE_MODEL", "gpt-5.4"))
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT", "120")))
    ap.add_argument("--max_retries", type=int, default=int(os.getenv("MAX_RETRIES", "2")))
    ap.add_argument("--retry_sleep", type=float, default=float(os.getenv("RETRY_SLEEP", "2.0")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.2")))
    ap.add_argument("--max_tokens", type=int, default=int(os.getenv("MAX_TOKENS", "220")))
    ap.add_argument("--token_limit_param", default=os.getenv("TOKEN_LIMIT_PARAM", "auto"), choices=["auto", "max_completion_tokens", "max_tokens"])

    ap.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "8")))
    ap.add_argument("--max_pending", type=int, default=int(os.getenv("MAX_PENDING", "64")))
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")))
    ap.add_argument("--resume", type=int, default=int(os.getenv("RESUME", "1")))
    ap.add_argument("--write_failed_rows", type=int, default=int(os.getenv("WRITE_FAILED_ROWS", "0")))

    ap.add_argument("--max_context_chars", type=int, default=int(os.getenv("MAX_CONTEXT_CHARS", "12000")))
    ap.add_argument("--max_query_chars", type=int, default=int(os.getenv("MAX_QUERY_CHARS", "2000")))
    ap.add_argument("--max_target_chars", type=int, default=int(os.getenv("MAX_TARGET_CHARS", "2000")))
    ap.add_argument("--max_sentences", type=int, default=int(os.getenv("MAX_SENTENCES", "5")))
    ap.add_argument("--field_name", default=os.getenv("FIELD_NAME", "target_context_explanation"))
    ap.add_argument("--error_field_name", default=os.getenv("ERROR_FIELD_NAME", "target_context_explanation_error"))
    ap.add_argument("--meta_field_name", default=os.getenv("META_FIELD_NAME", "target_context_explanation_meta"))
    ap.add_argument("--require_context", type=int, default=int(os.getenv("REQUIRE_CONTEXT", "1")))
    ap.add_argument("--require_target", type=int, default=int(os.getenv("REQUIRE_TARGET", "1")))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    input_path = Path(args.input)
    output_path = Path(args.output)
    errors_path = Path(args.errors) if args.errors else output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")

    cfg = ExplainConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        token_limit_param=args.token_limit_param,
        max_context_chars=args.max_context_chars,
        max_query_chars=args.max_query_chars,
        max_target_chars=args.max_target_chars,
        max_sentences=args.max_sentences,
        field_name=args.field_name,
        error_field_name=args.error_field_name,
        meta_field_name=args.meta_field_name,
        require_context=bool(args.require_context),
        require_target=bool(args.require_target),
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stats = process_file(
        input_path=input_path,
        output_path=output_path,
        errors_path=errors_path,
        cfg=cfg,
        workers=max(1, args.workers),
        resume=bool(args.resume),
        limit=max(0, args.limit),
        write_failed_rows=bool(args.write_failed_rows),
        max_pending=max(1, args.max_pending),
    )
    summary = {
        "ok": True,
        "elapsed_sec": round(time.time() - started, 4),
        "config": {k: v for k, v in asdict(cfg).items() if k != "api_key"},
        "stats": stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
