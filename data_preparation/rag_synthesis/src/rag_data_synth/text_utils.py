from __future__ import annotations

import json
import re
from typing import Any, Iterable, List


def as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def as_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def textify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return textify(json.loads(s)) or s
            except Exception:
                return s
        return s
    if isinstance(x, list):
        return " ".join(textify(v) for v in x if v is not None).strip()
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            sv = textify(v)
            if sv:
                parts.append(f"{k}={sv}")
        return " ".join(parts).strip()
    return str(x).strip()


def clip(x: Any, n: int = 260) -> str:
    s = " ".join(textify(x).replace("\n", " ").replace("\r", " ").split())
    if n is None or n <= 0:
        return s
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def fmt_score(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{float(x):.4f}"
    return "NA"


def join_nonempty(items: Iterable[Any], sep: str = ", ") -> str:
    vals = [str(x) for x in items if x is not None and str(x) != ""]
    return sep.join(vals) if vals else "NA"


def normalize_raw_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "".join("_" if ch.isspace() else ch for ch in s)


def extract_wrapped_tokens(text: str) -> List[str]:
    return re.findall(r"<[^<>\n\r]+>", text or "")


def extract_double_wrapped_tools(text: str) -> List[str]:
    return re.findall(r"<<[^<>\n\r]+>>", text or "")
