#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
from typing import Any, Dict


def recommend(
    query: str,
    topk: int = 10,
    rewrite_tool_query: bool = True,
    base_url: str = "http://127.0.0.1:6000",
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    POST /recommend

    input:
      {
        "query": str,
        "topk": int = 10,
        "rewrite_tool_query": bool = true
      }

    output:
      {
        "query": str,
        "tool_query": str,
        "llm_recs":  [{"name": str, "score": float, "metadata": object}, ...],
        "tool_recs": [{"name": str, "score": float, "metadata": object}, ...]
      }
    """
    url = f"{base_url.rstrip('/')}/recommend"
    payload = {
        "query": query,
        "topk": topk,
        "rewrite_tool_query": rewrite_tool_query,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Response is not valid JSON: {e}") from e


if __name__ == "__main__":
    query = "I want an agent that can summarize arXiv papers and search recent AI news."

    result = recommend(
        query=query,
        topk=10,
        rewrite_tool_query=True,
        base_url="http://127.0.0.1:6000",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))