#!/usr/bin/env python3
"""
Read generated bundle critic API test data, call the Flask API, and save all
responses into a JSON file.

Default input:
  ./bundle_critic_api_test_data.jsonl

Default output:
  ./bundle_critic_api_results.json
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_BASE_URL = AC_ENV('CRITIC_URL','http://127.0.0.1:8015')
DEFAULT_DATA = Path(__file__).resolve().parent / "bundle_critic_api_test_data.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bundle_critic_api_results.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc

    if not records:
        raise RuntimeError(f"No records found in {path}")

    return records


def parse_response(
    response: requests.Response,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status_code": response.status_code,
        "ok": response.ok,
    }

    try:
        result["response"] = response.json()
    except ValueError:
        result["response_text"] = response.text

    return result


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        response = session.request(
            method=method,
            url=url,
            json=payload,
            timeout=timeout,
        )

        result = parse_response(response)
        print(
            f"{method} {url} "
            f"status={result['status_code']} ok={result['ok']}"
        )
        return result

    except requests.RequestException as exc:
        print(f"{method} {url} request_failed={exc}")
        return {
            "status_code": None,
            "ok": False,
            "request_error": str(exc),
        }


def call_health(
    session: requests.Session,
    base_url: str,
) -> Dict[str, Any]:
    return request_json(
        session=session,
        method="GET",
        url=base_url + "/health",
        timeout=30,
    )


def call_score(
    session: requests.Session,
    base_url: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    gold = record["gold"]

    payload = {
        "query": record["query"],
        "llm": gold["llm"],
        "tools": gold["tools"],
        "evidence_context": gold["evidence_context"],
        "include_pairwise": True,
    }

    result = request_json(
        session=session,
        method="POST",
        url=base_url + "/v1/score",
        payload=payload,
        timeout=120,
    )
    result["request_payload"] = payload
    return result


def call_analyze(
    session: requests.Session,
    base_url: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    gold = record["gold"]

    payload = {
        "query": record["query"],
        "llm": gold["llm"],
        "tools": gold["tools"],
        "evidence_context": gold["evidence_context"],
        "include_pairwise": True,
    }

    result = request_json(
        session=session,
        method="POST",
        url=base_url + "/v1/analyze",
        payload=payload,
        timeout=180,
    )
    result["request_payload"] = payload
    return result


def call_rerank(
    session: requests.Session,
    base_url: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "query": record["query"],
        "candidates": [
            {
                "id": candidate["id"],
                "llm": candidate["llm"],
                "tools": candidate["tools"],
                "evidence_context": candidate["evidence_context"],
            }
            for candidate in record["candidates"]
        ],
    }

    result = request_json(
        session=session,
        method="POST",
        url=base_url + "/v1/rerank",
        payload=payload,
        timeout=300,
    )
    result["request_payload"] = payload
    return result


def save_results(
    output_path: Path,
    data: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[done] results saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_url",
        default=DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--record_index",
        type=int,
        default=None,
        help=(
            "Run only one zero-based record index. "
            "By default, all records are tested."
        ),
    )
    parser.add_argument(
        "--skip_analyze",
        action="store_true",
    )
    parser.add_argument(
        "--skip_rerank",
        action="store_true",
    )
    parser.add_argument(
        "--omit_request_payloads",
        action="store_true",
        help=(
            "Do not store request payloads in the output JSON. "
            "Useful when evidence_context makes the file too large."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_url = args.base_url.rstrip("/")
    data_path = Path(args.data)
    output_path = Path(args.output)

    records = load_jsonl(data_path)

    if args.record_index is not None:
        if args.record_index < 0 or args.record_index >= len(records):
            raise IndexError(
                f"record_index={args.record_index}, "
                f"but data contains {len(records)} records"
            )
        selected = [(args.record_index, records[args.record_index])]
    else:
        selected = list(enumerate(records))

    output: Dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "data_path": str(data_path),
            "output_path": str(output_path),
            "record_count": len(selected),
            "skip_analyze": args.skip_analyze,
            "skip_rerank": args.skip_rerank,
        },
        "health": None,
        "records": [],
    }

    with requests.Session() as session:
        output["health"] = call_health(session, base_url)

        for index, record in selected:
            print("\n" + "=" * 80)
            print(
                f"record_index={index} "
                f"qid={record.get('qid')}"
            )

            item: Dict[str, Any] = {
                "record_index": index,
                "qid": record.get("qid"),
                "query": record.get("query"),
                "expected_ranking": record.get("expected_ranking"),
                "teacher_scores": record.get("teacher_scores"),
                "score": call_score(session, base_url, record),
                "analyze": None,
                "rerank": None,
            }

            if not args.skip_analyze:
                item["analyze"] = call_analyze(
                    session,
                    base_url,
                    record,
                )

            if not args.skip_rerank:
                item["rerank"] = call_rerank(
                    session,
                    base_url,
                    record,
                )

            if args.omit_request_payloads:
                for endpoint in ("score", "analyze", "rerank"):
                    result = item.get(endpoint)
                    if isinstance(result, dict):
                        result.pop("request_payload", None)

            output["records"].append(item)

            # Save incrementally so completed results are retained if a later
            # request fails or the process is interrupted.
            save_results(output_path, output)

    save_results(output_path, output)


if __name__ == "__main__":
    main()
