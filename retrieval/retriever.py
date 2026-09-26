#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Tower (TF-IDF) inference -> Flask API

POST /recommend
  input:  { "query": str, "topk": int=10, "rewrite_tool_query": bool=true }
  output: {
    "query": str,
    "tool_query": str,
    "llm_recs":  [{"name": str, "score": float, "metadata": object}, ...],
    "tool_recs": [{"name": str, "score": float, "metadata": object}, ...]
  }
"""

from __future__ import annotations
import time
import random
import re
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request

from agent_rec.config import TFIDF_MAX_FEATURES
from agent_rec.features import (
    UNK_LLM_TOKEN,
    UNK_TOOL_TOKEN,
    build_agent_content_view,
    feature_cache_exists,
    load_feature_cache,
    load_vectorizers,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import bootstrap_run, shared_cache_dir
from agent_rec.data import load_tools as load_tools_json, load_LLMs as load_llms_json



# ----------------------------
# Optional: GPT Tool-Query Rewriting (Stage2)
# ----------------------------
TOOL_QUERY_PROMPT = """You are a search query rewriter for TOOL retrieval in an agent recommender.

Given a user's natural-language request, rewrite it into a compact "tool search query" that helps match tools.
Rules:
- Output ONLY JSON (no markdown fences).
- Keys: tool_query, rationale
- tool_query should be <= 32 tokens, English preferred, include concrete actions/APIs.
- Do NOT include model names.
- If multiple intents, keep top 2-3 tool intents, separated by "; ".
- Keep important constraints (location, format, source) if present.

User query:
{{query}}
"""

def normalize_key(s: str) -> str:
    """
    Normalize lookup key:
      - lowercase
      - remove spaces and all special symbols
      - keep only alphanumeric (unicode-aware via str.isalnum)
    """
    s = (s or "").strip().lower()
    if not s:
        return ""
    return "".join(ch for ch in s if ch.isalnum())

def _time_seed_mod_1000() -> int:
    # timestamp in ms, then mod 1000 -> 0..999
    return int(time.time() * 1000) % 1000

def _random_pick_names(cands: List[str]) -> Tuple[int, List[str]]:
    """
    Randomly pick k names from candidates.
    k is in [1, 10], capped by len(cands).
    Seed is timestamp(ms) % 1000.
    Returns (seed, picked_names).
    """
    seed = _time_seed_mod_1000()
    rng = random.Random(seed)
    if not cands:
        return seed, []
    k = rng.randint(1, 10)
    k = min(k, len(cands))
    picked = rng.sample(cands, k=k)
    return seed, picked



def _try_import_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None


def gpt_qa_not_stream(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. tool_query rewriting disabled.")
    openai = _try_import_openai()
    if openai is None:
        raise RuntimeError("openai python package not found. Please `pip install openai`.")

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        stream=False,
        service_tier="default",
    )
    return resp.choices[0].message.content


def rewrite_tool_query(query: str, model_name: str = "gpt-5-nano") -> Tuple[str, str, str]:
    prompt = TOOL_QUERY_PROMPT.replace("{{query}}", query)
    raw = gpt_qa_not_stream(prompt, model_name=model_name, temperature=0.0)

    try:
        obj = json.loads(raw)
        tq = str(obj.get("tool_query", "")).strip()
        if not tq:
            tq = query.strip()
        return tq, prompt, raw
    except Exception:
        tq = raw.strip()
        if not tq or len(tq) > 300:
            tq = query.strip()
        return tq, prompt, raw


# ----------------------------
# core inference
# ----------------------------
def _device_from_arg(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA is not available; falling back to CPU (requested: {device_str}).")
        return torch.device("cpu")
    return device


def _load_checkpoint(model_path: str, device: torch.device) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint format: {model_path}")
    return ckpt


def _resolve_feature_cache_dir(data_root: str, max_features: int, data_sig: str) -> str:
    return shared_cache_dir(data_root, "features", f"twotower_tfidf_{max_features}_{data_sig}")


def _build_encoder(*, ckpt: dict, feature_cache, device: torch.device) -> TwoTowerTFIDF:
    flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
    dims = ckpt.get("dims", {}) if isinstance(ckpt, dict) else {}

    encoder = TwoTowerTFIDF(
        d_q=int(dims.get("d_q", feature_cache.Q.shape[1])),
        d_a=int(
            dims.get(
                "d_a",
                feature_cache.A_text_full.shape[1]
                if hasattr(feature_cache, "A_text_full")
                else feature_cache.A_model_content.shape[1],
            )
        ),
        hid=int(dims.get("hid", 256)),
        num_tools=int(dims.get("num_tools", len(feature_cache.tool_id_vocab))),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_idx_padded=torch.tensor(feature_cache.agent_tool_idx_padded, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(feature_cache.agent_tool_mask, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        use_tool_id_emb=bool(flags.get("use_tool_id_emb", True)),
        use_llm_id_emb=bool(flags.get("use_llm_id_emb", False)),
        num_agents=len(feature_cache.a_ids),
        num_queries=len(feature_cache.q_ids),
        use_query_id_emb=bool(flags.get("use_query_id_emb", False)),
    ).to(device)
    encoder.load_state_dict(ckpt["state_dict"], strict=False)
    encoder.eval()
    return encoder


class TwoTowerInference:
    def __init__(
        self,
        *,
        data_root: str,
        model_path: str,
        device: torch.device,
        max_features: int,
        topk: int,
    ) -> None:
        self.data_root = data_root
        self.model_path = model_path
        self.device = device
        self.max_features = max_features
        self.topk = topk

        boot = bootstrap_run(
            data_root=data_root,
            exp_name="infer_twotower_tfidf",
            topk=topk,
            seed=1234,
            with_tools=True,
        )
        self.bundle = boot.bundle

        ckpt = _load_checkpoint(model_path, device)
        self.ckpt = ckpt
        self.ckpt_data_sig = ckpt.get("data_sig", boot.data_sig)

        cache_dir = _resolve_feature_cache_dir(data_root, max_features, self.ckpt_data_sig)
        if not feature_cache_exists(cache_dir):
            raise RuntimeError(
                f"Feature cache not found: {cache_dir}\n"
                "Make sure you trained the TwoTower TF-IDF model with the same data_root and max_features."
            )
        self.feature_cache = load_feature_cache(cache_dir)

        vecs = load_vectorizers(cache_dir)
        if vecs is None or not hasattr(vecs, "q_vec"):
            raise RuntimeError(
                f"TF-IDF q_vectorizer not found in: {cache_dir}\n"
                "Make sure training saved q_vectorizer.pkl."
            )
        self.q_vectorizer = vecs.q_vec

        flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
        use_model_content_vector = bool(flags.get("use_model_content_vector", True))
        use_tool_content_vector = bool(flags.get("use_tool_content_vector", True))

        self.agent_content = build_agent_content_view(
            cache=self.feature_cache,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        )

        self.encoder = _build_encoder(ckpt=ckpt, feature_cache=self.feature_cache, device=self.device)
        self.encoder.set_agent_features(self.agent_content)
        self.agent_embeddings = self.encoder.export_agent_embeddings()

        self.agent_ids = list(self.feature_cache.a_ids)

        # For reverse indices
        self.agent_llm_ids = list(getattr(self.feature_cache, "llm_ids", []))
        self.agent_tools: List[List[str]] = []
        self.agent_llm_names: List[str] = []
        for aid in self.agent_ids:
            agent = self.bundle.all_agents.get(aid, {}) or {}
            m = (agent.get("M") or {}) if isinstance(agent, dict) else {}
            t = (agent.get("T") or {}) if isinstance(agent, dict) else {}
            self.agent_llm_names.append((m.get("name") or m.get("id") or "").strip())
            self.agent_tools.append(list((t.get("tools") or [])))

        # GLOBAL candidate sets from merge.json (key=name)
        llm_json = load_llms_json(data_root) or {}
        tool_json = load_tools_json(data_root) or {}

        # Keep full metadata (not only description)
        self.llm_meta_map: Dict[str, object] = {k: (v or {}) for k, v in llm_json.items() if k}
        self.tool_meta_map: Dict[str, object] = {k: (v or {}) for k, v in tool_json.items() if k}

        # Normalized key -> original key (for fast meta lookup)
        self._llm_norm_to_keys: Dict[str, List[str]] = {}
        for k in self.llm_meta_map.keys():
            nk = normalize_key(k)
            if nk:
                self._llm_norm_to_keys.setdefault(nk, []).append(k)

        self._tool_norm_to_keys: Dict[str, List[str]] = {}
        for k in self.tool_meta_map.keys():
            nk = normalize_key(k)
            if nk:
                self._tool_norm_to_keys.setdefault(nk, []).append(k)
                
        self.llm_candidates: List[str] = [k for k in llm_json.keys() if k and k != UNK_LLM_TOKEN]
        self.tool_candidates: List[str] = [k for k in tool_json.keys() if k and k != UNK_TOOL_TOKEN]

        # Reverse indices (normalized)
        self._llm_to_agent_indices: Dict[str, List[int]] = {}
        self._tool_to_agent_indices: Dict[str, List[int]] = {}

        for i in range(len(self.agent_ids)):
            llm_id = (self.agent_llm_ids[i] if i < len(self.agent_llm_ids) else "") or ""
            llm_name = (self.agent_llm_names[i] if i < len(self.agent_llm_names) else "") or ""

            for key in {llm_id.strip(), llm_name.strip()}:
                nk = normalize_key(key)
                if nk:
                    self._llm_to_agent_indices.setdefault(nk, []).append(i)

            for t in (self.agent_tools[i] if i < len(self.agent_tools) else []):
                nk = normalize_key(t)
                if nk:
                    self._tool_to_agent_indices.setdefault(nk, []).append(i)


    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.q_vectorizer.transform([query]).toarray().astype(np.float32)
        q = torch.from_numpy(vec).to(self.device)
        q_idx = None
        if getattr(self.encoder, "use_query_id_emb", False):
            q_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            qe = self.encoder.encode_q(q, q_idx=q_idx).cpu().numpy()
        return qe

    def score_all_agents(self, query: str) -> np.ndarray:
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty.")
        qe = self._encode_query(query)
        scores = np.dot(qe, self.agent_embeddings.T).reshape(-1)
        return scores

    def recommend_llms_from_candidates(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for llm in self.llm_candidates:
            key = normalize_key(llm)
            idxs = self._llm_to_agent_indices.get(key, [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(scores[i]))
            out.append(
                {
                    "name": llm,
                    "score": float(scores[best_i]),
                    "metadata": self.llm_meta_map.get(llm, {}) or {},
                }
            )
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

    def recommend_tools_from_candidates(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for tool in self.tool_candidates:
            key = normalize_key(tool)
            idxs = self._tool_to_agent_indices.get(key, [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(scores[i]))
            out.append(
                {
                    "name": tool,
                    "score": float(scores[best_i]),
                    "metadata": self.tool_meta_map.get(tool, {}) or {},
                }
            )
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

def build_app(infer: TwoTowerInference) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64KB

    def _get_name_from_req() -> str:
        # GET: /xx?name=...
        name = (request.args.get("name") or "").strip()
        if name:
            return name
        # POST: {"name": "..."}
        data = request.get_json(force=True, silent=True) or {}
        return (data.get("name") or "").strip()

    def _lookup_meta(meta_map: Dict[str, object], name: str, norm_index: Dict[str, List[str]]) -> Tuple[bool, str, object]:
        """
        Returns: (found, resolved_key, metadata)
        Matching priority:
          1) exact key
          2) case-insensitive key
          3) normalized key (remove special symbols/spaces + lowercase)
        """
        if not name:
            return False, "", {}

        # 1) exact
        if name in meta_map:
            return True, name, meta_map.get(name, {}) or {}

        # 2) case-insensitive
        low = name.lower()
        for k in meta_map.keys():
            if (k or "").lower() == low:
                return True, k, meta_map.get(k, {}) or {}

        # 3) normalized
        nk = normalize_key(name)
        if nk:
            keys = norm_index.get(nk, [])
            if keys:
                # pick the first deterministically (could add collision info if you want)
                k0 = keys[0]
                return True, k0, meta_map.get(k0, {}) or {}

        return False, "", {}


    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    # ----------------------------
    # NEW: LLM meta lookup
    # ----------------------------
    @app.route("/llm_meta", methods=["GET", "POST"])
    def llm_meta():
        name = _get_name_from_req()
        if not name:
            return jsonify({"error": "missing_name"}), 400

        found, key, meta = _lookup_meta(infer.llm_meta_map, name, infer._llm_norm_to_keys)
        if not found:
            return (
                jsonify(
                    {
                        "error": "not_found",
                        "name": name,
                    }
                ),
                404,
            )

        return jsonify(
            {
                "name": name,
                "resolved_key": key,
                "metadata": meta,
            }
        )

    # ----------------------------
    # NEW: Tool meta lookup
    # ----------------------------
    @app.route("/tool_meta", methods=["GET", "POST"])
    def tool_meta():
        name = _get_name_from_req()
        if not name:
            return jsonify({"error": "missing_name"}), 400
        
        found, key, meta = _lookup_meta(infer.tool_meta_map, name, infer._tool_norm_to_keys)
        if not found:
            return (
                jsonify(
                    {
                        "error": "not_found",
                        "name": name,
                    }
                ),
                404,
            )

        return jsonify(
            {
                "name": name,
                "resolved_key": key,
                "metadata": meta,
                "source": os.path.join(infer.data_root, "Tools", "merge.json"),
            }
        )

    @app.route("/recommend", methods=["POST"])
    def recommend():
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        topk = int(data.get("topk", infer.topk))
        rewrite_flag = bool(data.get("rewrite_tool_query", True))

        if not query:
            return jsonify({"error": "missing_query"}), 400

        # Stage 1: LLM retrieval with original query
        scores = infer.score_all_agents(query)
        llm_recs = infer.recommend_llms_from_candidates(scores, topk=topk)

        # Stage 2: Tool retrieval with tool_query (optional rewrite)
        tool_query = query
        tool_query_error = None
        if rewrite_flag:
            try:
                tool_query, _p, _raw = rewrite_tool_query(query, model_name="gpt-5-nano")
            except Exception as e:
                tool_query_error = str(e)
                tool_query = query

        tool_scores = infer.score_all_agents(tool_query)
        tool_recs = infer.recommend_tools_from_candidates(tool_scores, topk=topk)

        return jsonify(
            {
                "query": query,
                "tool_query": tool_query,
                "tool_query_error": tool_query_error,
                "topk": topk,
                "llm_recs": llm_recs,
                "tool_recs": tool_recs,
                "meta_sources": {
                    "llm_merge_json": os.path.join(infer.data_root, "LLMs", "merge.json"),
                    "tool_merge_json": os.path.join(infer.data_root, "Tools", "merge.json"),
                },
            }
        )
        

    @app.route("/random_llms", methods=["GET", "POST"])
    def random_llms():
        seed, picked = _random_pick_names(infer.llm_candidates)
        out = [
            {
                "name": name,
                "metadata": infer.llm_meta_map.get(name, {}) or {},
            }
            for name in picked
        ]
        return jsonify(
            {
                "llms": out,
            }
        )

    @app.route("/random_tools", methods=["GET", "POST"])
    def random_tools():
        seed, picked = _random_pick_names(infer.tool_candidates)
        out = [
            {
                "name": name,
                "metadata": infer.tool_meta_map.get(name, {}) or {},
            }
            for name in picked
        ]
        return jsonify(
            {
                "tools": out,
            }
        )

    return app



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TwoTower TF-IDF inference -> Flask API")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root dir (contains Part I/II/III).")
    ap.add_argument("--model_path", type=str, required=True, help="Trained TwoTowerTFIDF checkpoint (.pt).")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES, help="TF-IDF max features.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Device: cpu / cuda:0 / etc.")
    ap.add_argument("--topk", type=int, default=10, help="Default TopK.")
    ap.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    ap.add_argument("--port", type=int, default=6000, help="Bind port.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    infer = TwoTowerInference(
        data_root=args.data_root,
        model_path=args.model_path,
        device=device,
        max_features=args.max_features,
        topk=args.topk,
    )

    app = build_app(infer)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
