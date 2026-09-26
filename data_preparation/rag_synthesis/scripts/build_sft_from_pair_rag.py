#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else []


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert pair_rag JSONL plus rationale JSONL into SFT JSONL."
    )
    ap.add_argument("--input", type=str, nargs="+", default=[], help="Single-file/fallback input JSONL file(s).")
    ap.add_argument("--pair_rag_jsonl", type=str, default="pair_rag.jsonl", help="Main pair_rag.jsonl file with query/context/target.")
    ap.add_argument(
        "--rationale_jsonl",
        type=str,
        default="pair_rag.context_target_rationales.v2.jsonl",
        help="pair_rag.context_target_rationales.v2.jsonl file with target_context_explanation.",
    )
    ap.add_argument("--output", type=str, default="sft_train.jsonl", help="Output SFT JSONL file.")
    ap.add_argument("--valid_output", type=str, default="sft_valid.jsonl", help="Optional validation SFT JSONL file.")
    ap.add_argument("--valid_ratio", type=float, default=0.1, help="If >0 and valid_output is set, split train/valid.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all.")
    ap.add_argument(
        "--mode",
        type=str,
        default="target_with_explanation",
        choices=["auto", "target_only", "target_with_explanation"],
        help=(
            "auto: use target+explanation when target_context_explanation exists, otherwise target only; "
            "target_only: ignore explanation; "
            "target_with_explanation: keep only rows with explanation."
        ),
    )
    ap.add_argument("--dedup", type=int, default=1)
    ap.add_argument("--debug_samples", type=int, default=3)
    return ap.parse_args()


def read_jsonl(paths: Sequence[str]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"[bad-json] {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    continue
                yield path, line_no, obj


def row_keys(obj: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    qid = str(obj.get("qid", "") or "")
    agent_id = str(obj.get("agent_id", "") or "")
    target = str(obj.get("target", "") or "")
    query = str(obj.get("query", "") or "")
    keys = [
        ("qid_agent_target", f"{qid}\t{agent_id}\t{target}"),
        ("qid_target", f"{qid}\t{target}"),
        ("agent_target", f"{agent_id}\t{target}"),
        ("query_target", f"{query}\t{target}"),
    ]
    return [(name, key) for name, key in keys if key.strip()]


def load_rationale_maps(path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    maps: Dict[str, Dict[str, str]] = {
        "qid_agent_target": {},
        "qid_target": {},
        "agent_target": {},
        "query_target": {},
    }
    stats = {"rows": 0, "with_explanation": 0, "collisions": 0}
    if not path:
        return maps, stats

    for _, _, obj in read_jsonl([path]):
        stats["rows"] += 1
        explanation = str(obj.get("target_context_explanation", "") or "").strip()
        if not explanation:
            continue
        stats["with_explanation"] += 1
        for name, key in row_keys(obj):
            if key in maps[name] and maps[name][key] != explanation:
                stats["collisions"] += 1
                continue
            maps[name][key] = explanation
    return maps, stats


def find_rationale(obj: Dict[str, Any], rationale_maps: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    for name, key in row_keys(obj):
        explanation = rationale_maps.get(name, {}).get(key, "")
        if explanation:
            return explanation, name
    return "", ""


def build_prompt(*, context: str, query: str, want_explanation: bool) -> str:
    if want_explanation:
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


def build_completion(*, target: str, explanation: str, want_explanation: bool) -> str:
    target = target.strip()
    explanation = explanation.strip()
    if want_explanation:
        return f"{target}\n\nExplanation: {explanation}"
    return target


def convert_row(
    obj: Dict[str, Any],
    *,
    source_path: str,
    line_no: int,
    mode: str,
    explanation_override: str = "",
    explanation_match_key: str = "",
) -> Dict[str, Any] | None:
    query = str(obj.get("query", "") or "").strip()
    context = str(obj.get("context", "") or "").strip()
    target = str(obj.get("target", "") or "").strip()
    explanation = explanation_override.strip() or str(obj.get("target_context_explanation", "") or "").strip()

    if not query or not context or not target:
        return None

    if mode == "target_only":
        want_explanation = False
    elif mode == "target_with_explanation":
        if not explanation:
            return None
        want_explanation = True
    else:
        want_explanation = bool(explanation)

    out = {
        "qid": str(obj.get("qid", "") or ""),
        "agent_id": str(obj.get("agent_id", "") or ""),
        "part": str(obj.get("part", "") or ""),
        "topk": int(obj.get("topk", 0) or 0),
        "query": query,
        "context": context,
        "target": target,
        "target_context_explanation": explanation,
        "sft_mode": "target_with_explanation" if want_explanation else "target_only",
        "source_file": os.path.basename(source_path),
        "source_line_no": line_no,
        "explanation_match_key": explanation_match_key,
    }
    out["prompt"] = build_prompt(context=context, query=query, want_explanation=want_explanation)
    out["completion"] = build_completion(
        target=target,
        explanation=explanation,
        want_explanation=want_explanation,
    )
    return out


def dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("qid", ""),
            row.get("agent_id", ""),
            row.get("target", ""),
            row.get("sft_mode", ""),
            row.get("completion", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def split_train_valid(rows: List[Dict[str, Any]], valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    if valid_ratio <= 0 or len(rows) <= 1:
        return rows, []
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_ratio)) if rows else 0
    return rows[n_valid:], rows[:n_valid]


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    skipped = 0
    matched_explanations = 0
    rationale_stats = {"rows": 0, "with_explanation": 0, "collisions": 0}

    if args.pair_rag_jsonl:
        rationale_maps, rationale_stats = load_rationale_maps(args.rationale_jsonl)
        source_iter = read_jsonl([args.pair_rag_jsonl])
        for source_path, line_no, obj in tqdm(source_iter, desc="convert", unit="row"):
            explanation, match_key = find_rationale(obj, rationale_maps)
            if explanation:
                matched_explanations += 1
            row = convert_row(
                obj,
                source_path=source_path,
                line_no=line_no,
                mode=args.mode,
                explanation_override=explanation,
                explanation_match_key=match_key,
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)
    else:
        if not args.input:
            raise SystemExit("Provide --pair_rag_jsonl plus --rationale_jsonl, or provide --input.")
        for source_path, line_no, obj in tqdm(read_jsonl(args.input), desc="convert", unit="row"):
            row = convert_row(obj, source_path=source_path, line_no=line_no, mode=args.mode)
            if row is None:
                skipped += 1
                continue
            if row.get("target_context_explanation"):
                matched_explanations += 1
            rows.append(row)

    if args.dedup:
        before = len(rows)
        rows = dedup_rows(rows)
        deduped = before - len(rows)
    else:
        deduped = 0

    if args.max_examples and args.max_examples > 0:
        rows = rows[: int(args.max_examples)]

    train_rows, valid_rows = split_train_valid(rows, float(args.valid_ratio), int(args.seed))
    write_jsonl(args.output, train_rows)
    if args.valid_output and valid_rows:
        write_jsonl(args.valid_output, valid_rows)

    print(
        json.dumps(
            {
                "input_files": args.input,
                "pair_rag_jsonl": args.pair_rag_jsonl,
                "rationale_jsonl": args.rationale_jsonl,
                "output": args.output,
                "valid_output": args.valid_output if valid_rows else "",
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "skipped_rows": skipped,
                "deduped_rows": deduped,
                "matched_explanations": matched_explanations,
                "rationale_stats": rationale_stats,
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for i, row in enumerate(train_rows[: max(0, int(args.debug_samples))]):
        print(f"[sample {i}] mode={row['sft_mode']} qid={row.get('qid')}")
        print(row["prompt"][:500].replace("\n", "\\n"))
        print(row["completion"][:500].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
