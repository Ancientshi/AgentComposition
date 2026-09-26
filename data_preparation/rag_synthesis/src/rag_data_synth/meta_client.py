from __future__ import annotations

from typing import Any, Dict
import requests

from .text_utils import textify


def _extract_description(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    # Broad fallback order across your LLM/tool metadata shapes.
    doc = meta.get("documentation") if isinstance(meta.get("documentation"), dict) else {}
    for key in ["description", "desc", "summary", "api_description", "expressions"]:
        val = doc.get(key) if isinstance(doc, dict) and key in doc else meta.get(key)
        s = textify(val)
        if s:
            return s
    s = textify(meta)
    return s


class MetadataClient:
    def __init__(self, api_base: str = "", timeout: float = 5.0):
        self.api_base = (api_base or "").rstrip("/")
        self.timeout = timeout
        self.llm_cache: Dict[str, str] = {}
        self.tool_cache: Dict[str, str] = {}

    @staticmethod
    def normalize_lookup_name(token: str, component_type: str) -> str:
        t = (token or "").strip()
        if component_type == "llm":
            if t.startswith("<LLM_") and t.endswith(">"):
                return t[len("<LLM_"):-1].replace("__", "/")
            return t
        if t.startswith("<<") and t.endswith(">>"):
            return t[2:-2]
        if t.startswith("<TOOL_") and t.endswith(">"):
            return t[len("<TOOL_"):-1]
        if t.startswith("<") and t.endswith(">"):
            return t[1:-1]
        return t

    def get_desc(self, token: str, component_type: str) -> str:
        if not self.api_base:
            return ""
        cache = self.llm_cache if component_type == "llm" else self.tool_cache
        if token in cache:
            return cache[token]
        endpoint = "/llm_meta" if component_type == "llm" else "/tool_meta"
        name = self.normalize_lookup_name(token, component_type)
        try:
            resp = requests.get(self.api_base + endpoint, params={"name": name}, timeout=self.timeout)
            if resp.status_code != 200:
                cache[token] = ""
                return ""
            obj = resp.json()
            meta = obj.get("metadata") if isinstance(obj, dict) else None
            desc = _extract_description(meta)
        except Exception:
            desc = ""
        cache[token] = desc
        return desc
