#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable evaluation for agent generation quality vs GT agents.

What's new in this version (NO CLI args changed):
1) More detailed printing:
   - For each @K, prints averaged:
       * Soft Precision/Recall/F1/MRR/Config-nDCG
       * Avg LLM-match score (model part)
       * Avg Tool-recall score (tool part)
   - Additionally prints a HARD exact-match evaluation table (@K) in parallel.

2) Per-sample detail (if you pass --save_per_sample):
   - Saves per-rank breakdown (for top-K ranks):
       * best matched GT index
       * LLM part score (mm: 0/1)
       * Tool part score (tr: recall in [0,1])
       * alpha used (may adapt based on GT)
       * soft similarity
       * whether it is counted as a hit under theta
   - Also includes hard-exact match info per rank

3) Metric explanations printed at the end.

Important:
- "Soft" metrics use your similarity:
    sim = alpha * model_match + (1-alpha) * tool_recall
  then threshold by theta for TP coverage.
- "Hard" metrics require exact match of BOTH:
    M_set == M_gt AND T_set == T_gt
  (no theta; hit iff exact)

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------
# 0) Soft evaluation code
# -----------------------------
from math import log2

def _alpha_for_gt(gt: Tuple[Set[str], Set[str]], alpha: float) -> Optional[float]:
    """
    New rule:
      - M_gt empty, T_gt non-empty -> return 0.0 (tool only)
      - T_gt empty, M_gt non-empty -> return 1.0 (model only)
      - both empty -> None (treat as sim=1)
      - both non-empty -> alpha
    """
    M_gt, T_gt = gt
    if len(M_gt) == 0 and len(T_gt) == 0:
        return None
    if len(M_gt) == 0 and len(T_gt) > 0:
        return 0.0
    if len(T_gt) == 0 and len(M_gt) > 0:
        return 1.0
    return alpha


def tool_recall(T_rec: Set[str], T_gt: Set[str]) -> float:
    if len(T_gt) == 0:
        return 1.0
    return len(T_rec & T_gt) / len(T_gt)


def model_match(M_rec: Set[str], M_gt: Set[str]) -> float:
    if len(M_gt) == 0:
        return 1.0
    return 1.0 if (M_rec & M_gt) else 0.0


def agent_similarity_components(
    A: Tuple[Set[str], Set[str]],
    B: Tuple[Set[str], Set[str]],
    alpha: float = 0.4,
) -> Tuple[float, float, float]:
    """
    Return (mm, tr, sim).
      mm: model_match in {0,1}
      tr: tool_recall in [0,1]
      sim: alpha*mm + (1-alpha)*tr
    """
    M_rec, T_rec = A
    M_gt,  T_gt  = B
    mm = model_match(M_rec, M_gt)
    tr = tool_recall(T_rec, T_gt)
    sim = alpha * mm + (1 - alpha) * tr
    return mm, tr, sim


def agent_similarity(
    A: Tuple[Set[str], Set[str]],
    B: Tuple[Set[str], Set[str]],
    alpha: float = 0.4
) -> float:
    return agent_similarity_components(A, B, alpha=alpha)[2]


def soft_rel(
    rec_agent: Tuple[Set[str], Set[str]],
    gt_agents: List[Tuple[Set[str], Set[str]]],
    beta: List[float],
    alpha: float
) -> float:
    best = 0.0
    for i, g in enumerate(gt_agents):
        alpha_i = _alpha_for_gt(g, alpha)
        if alpha_i is None:
            s = 1.0
        else:
            s = agent_similarity(rec_agent, g, alpha=alpha_i)
        val = beta[i] * s
        if val > best:
            best = val
    return best


def config_ndcg_at_k(R, G, K, alpha, beta) -> float:
    if K <= 0 or not R or not G:
        return 0.0
    rels_all = [soft_rel(r, G, beta, alpha) for r in R]
    k = min(K, len(rels_all))
    top_rels = rels_all[:k]
    dcg = sum(top_rels[j] / log2(j + 2) for j in range(k))
    ideal_rels = sorted(rels_all, reverse=True)[:k]
    idcg = sum(ideal_rels[j] / log2(j + 2) for j in range(k))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_agents_soft_detailed(
    gt_agents: List[Tuple[Set[str], Set[str]]],
    rec_agents: List[Tuple[Set[str], Set[str]]],
    ks: List[int] = [5, 10, 50],
    alpha: float = 0.4,
    theta: float = 0.67,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, List[Dict[str, Any]]]]:
    """
    Returns:
      - results_by_k: metrics dict per K (soft)
      - breakdown_by_k: list of per-rank breakdown rows (for each K separately)
    """
    G = [a for a in gt_agents]
    R = [a for a in rec_agents]
    L = len(G)
    beta = [1.0] * L

    results: Dict[int, Dict[str, float]] = {}
    breakdown_by_k: Dict[int, List[Dict[str, Any]]] = {}

    for K in ks:
        k = min(K, len(R))
        if k == 0 or L == 0:
            results[K] = {
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Config-nDCG": 0.0,
                "MRR": 0.0,
                "Avg-LLM": 0.0,
                "Avg-ToolRecall": 0.0,
            }
            breakdown_by_k[K] = []
            continue

        per_rank = []
        best_scores: List[float] = []
        best_idx: List[Optional[int]] = []
        mm_list: List[float] = []
        tr_list: List[float] = []

        for j in range(k):
            r = R[j]
            s_best, i_best = 0.0, None
            best_mm, best_tr, best_alpha_i = 0.0, 0.0, None

            for i, g in enumerate(G):
                alpha_i = _alpha_for_gt(g, alpha)
                if alpha_i is None:
                    mm, tr, s = 1.0, 1.0, 1.0
                else:
                    mm, tr, s = agent_similarity_components(r, g, alpha=alpha_i)
                s *= beta[i]

                if s > s_best:
                    s_best, i_best = s, i
                    best_mm, best_tr, best_alpha_i = mm, tr, alpha_i

            hit = (i_best is not None) and (s_best >= theta)

            best_scores.append(s_best)
            best_idx.append(i_best)
            mm_list.append(best_mm)
            tr_list.append(best_tr)

            per_rank.append({
                "rank": j + 1,
                "best_gt_index": i_best,
                "alpha_used": best_alpha_i,
                "llm_score_mm": best_mm,
                "tool_score_tr": best_tr,
                "soft_similarity": s_best,
                "hit_theta": hit,
            })

        # coverage-based TP (unique GT covered)
        covered = set()
        for row in per_rank:
            if row["best_gt_index"] is not None and row["hit_theta"]:
                covered.add(row["best_gt_index"])

        tp = len(covered)
        precision = tp / max(1, k)
        recall = tp / max(1, L)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # MRR: first rank that hits a new GT with s>=theta
        mrr = 0.0
        seen = set()
        for row in per_rank:
            gi = row["best_gt_index"]
            if gi is not None and row["hit_theta"] and gi not in seen:
                mrr = 1.0 / row["rank"]
                break

        cndcg = config_ndcg_at_k(R, G, K, alpha, beta)

        avg_mm = sum(mm_list) / len(mm_list) if mm_list else 0.0
        avg_tr = sum(tr_list) / len(tr_list) if tr_list else 0.0

        results[K] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Config-nDCG": cndcg,
            "MRR": mrr,
            "Avg-LLM": avg_mm,
            "Avg-ToolRecall": avg_tr,
        }
        breakdown_by_k[K] = per_rank

    return results, breakdown_by_k


# -----------------------------
# 0b) Hard exact-match evaluation
# -----------------------------
def hard_exact_match(A: Tuple[Set[str], Set[str]], B: Tuple[Set[str], Set[str]]) -> bool:
    """
    Hard exact match: both model-set and tool-set must match exactly.
    """
    return (A[0] == B[0]) and (A[1] == B[1])


def config_ndcg_at_k_hard(R, G, K) -> float:
    """
    Hard nDCG: rel(r) = 1 if any GT exactly matches, else 0.
    """
    if K <= 0 or not R or not G:
        return 0.0
    rels_all = []
    for r in R:
        rel = 1.0 if any(hard_exact_match(r, g) for g in G) else 0.0
        rels_all.append(rel)

    k = min(K, len(rels_all))
    top_rels = rels_all[:k]
    dcg = sum(top_rels[j] / log2(j + 2) for j in range(k))
    ideal_rels = sorted(rels_all, reverse=True)[:k]
    idcg = sum(ideal_rels[j] / log2(j + 2) for j in range(k))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate_agents_hard_detailed(
    gt_agents: List[Tuple[Set[str], Set[str]]],
    rec_agents: List[Tuple[Set[str], Set[str]]],
    ks: List[int] = [5, 10, 50],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, List[Dict[str, Any]]]]:
    """
    Hard evaluation: exact match only.

    We still follow the same "coverage" style:
      - At rank j, if rec agent exactly matches ANY GT, it "covers" that GT (first matched GT index).
      - TP counts unique GTs covered in top-K.

    Returns:
      - metrics per K
      - per-rank breakdown per K
    """
    G = [a for a in gt_agents]
    R = [a for a in rec_agents]
    L = len(G)

    results: Dict[int, Dict[str, float]] = {}
    breakdown_by_k: Dict[int, List[Dict[str, Any]]] = {}

    for K in ks:
        k = min(K, len(R))
        if k == 0 or L == 0:
            results[K] = {
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Config-nDCG": 0.0,
                "MRR": 0.0,
            }
            breakdown_by_k[K] = []
            continue

        per_rank = []
        covered = set()
        mrr = 0.0
        for j in range(k):
            r = R[j]
            hit_i = None
            for i, g in enumerate(G):
                if hard_exact_match(r, g):
                    hit_i = i
                    break
            hit = hit_i is not None
            if hit:
                covered.add(hit_i)
                if mrr == 0.0:
                    mrr = 1.0 / (j + 1)
            per_rank.append({
                "rank": j + 1,
                "hit_exact": hit,
                "matched_gt_index": hit_i,
            })

        tp = len(covered)
        precision = tp / max(1, k)
        recall = tp / max(1, L)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        cndcg = config_ndcg_at_k_hard(R, G, K)

        results[K] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Config-nDCG": cndcg,
            "MRR": mrr,
        }
        breakdown_by_k[K] = per_rank

    return results, breakdown_by_k


# -----------------------------
# 1) Parsing utilities (robust)
# -----------------------------
WRAPPED_LLM_RE = re.compile(r"<LLM_[^<>\n\r]+>")
WRAPPED_TOOL1_RE = re.compile(r"<TOOL_[^<>\n\r]+>")
WRAPPED_TOOL2_RE = re.compile(r"<<[^<>\n\r]+>>")

TOOL_SEP = "<TOOL_SEP>"
END_TOK = "<SPECIAL_END>"

def extract_wrapped_llms(text: str) -> List[str]:
    return WRAPPED_LLM_RE.findall(text or "")

def extract_wrapped_tools(text: str) -> List[str]:
    # support both <TOOL_...> and <<Tool&&Endpoint>>
    return WRAPPED_TOOL1_RE.findall(text or "") + WRAPPED_TOOL2_RE.findall(text or "")

def parse_agent_from_text(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Convert a single generated string into (M_set, T_set).
    - M_set: all <LLM_...> tokens found
    - T_set: all tool tokens found (<TOOL_...> or <<...>>)
    """
    M = set(extract_wrapped_llms(text))
    T = set(extract_wrapped_tools(text))
    return M, T

def parse_ranked_agents_from_text(text: str, max_tools: int = 8) -> List[Tuple[Set[str], Set[str]]]:
    """
    Convert output into a ranked list of agents.
    Your model outputs ONE agent config normally, but sometimes may contain multiple tools.
    Here we interpret it as a single agent with potentially many tools.
    If you later extend to multi-agent outputs, you can change this function only.
    """
    M, T = parse_agent_from_text(text)
    if max_tools > 0 and len(T) > max_tools:
        # keep deterministic order by sorting
        T = set(sorted(list(T))[:max_tools])
    return [(M, T)]

def parse_gt_agents_from_target(target: str) -> List[Tuple[Set[str], Set[str]]]:
    """
    GT target is usually a single agent string.
    If your GT sometimes encodes multiple agents, modify here.
    """
    return parse_ranked_agents_from_text(target, max_tools=999999)


# -----------------------------
# 2) Dataset
# -----------------------------
@dataclass
class Sample:
    idx: int
    qid: Optional[str]
    query: str
    context: str
    target: str
    raw: Dict[str, Any]

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_dataset(data_path: str) -> List[Sample]:
    rows = read_jsonl(data_path)
    out: List[Sample] = []
    for i, r in enumerate(rows):
        query = r.get("query") or r.get("question") or ""
        context = r.get("context") or ""
        target = r.get("target") or ""
        qid = r.get("qid") or r.get("id") or r.get("question_id")
        out.append(Sample(idx=i, qid=qid, query=query, context=context, target=target, raw=r))
    return out


# -----------------------------
# 3) Generator interface + implementations
# -----------------------------
class Generator:
    def generate_one(self, query: str, context: str, meta: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

class FilePredGenerator(Generator):
    """
    Reads predictions from a jsonl file.
    Each line should contain:
      - "pred": string
    optionally:
      - "qid": to match by id
    If no qid, align by line index.
    """
    def __init__(self, pred_jsonl: str):
        self.pred_jsonl = pred_jsonl
        rows = read_jsonl(pred_jsonl)

        self.by_qid: Dict[str, str] = {}
        self.by_idx: List[str] = []
        for r in rows:
            pred = r.get("pred") or r.get("gen_text") or r.get("prediction") or ""
            qid = r.get("qid")
            if isinstance(qid, str) and qid:
                self.by_qid[qid] = pred
            self.by_idx.append(pred)

    def generate_one(self, query: str, context: str, meta: Optional[Dict[str, Any]] = None) -> str:
        if meta and isinstance(meta.get("qid"), str) and meta["qid"] in self.by_qid:
            return self.by_qid[meta["qid"]]
        idx = int(meta.get("idx")) if meta and meta.get("idx") is not None else None
        if idx is not None and 0 <= idx < len(self.by_idx):
            return self.by_idx[idx]
        return ""

class SubprocessGenerator(Generator):
    """
    Calls your existing inference script per sample:
      python infer_py --base_model_name ... --model_dir ... --query ... --context ... --controlled ... etc

    Assumptions:
    - The script prints something containing the generated structured text.
    - We'll try to extract the first occurrence of "<LLM_...> ... <SPECIAL_END>" from stdout.
      If your infer script prints JSON, adjust _extract_pred() below.
    """
    def __init__(
        self,
        infer_py: str,
        model_dir: str,
        base_model: Optional[str],
        top_k: int = 3,
        num_beams: int = 5,
        max_tools: int = 8,
        controlled: int = 0,
        preselect_k: int = 50,
        python_bin: str = "python",
        extra_args: Optional[List[str]] = None,
    ):
        self.infer_py = infer_py
        self.model_dir = model_dir
        self.base_model = base_model
        self.top_k = top_k
        self.num_beams = num_beams
        self.max_tools = max_tools
        self.controlled = controlled
        self.preselect_k = preselect_k
        self.python_bin = python_bin
        self.extra_args = extra_args or []

    def _extract_pred(self, stdout: str) -> str:
        # Try to find a substring that looks like your strict output.
        m = re.search(r"(<LLM_[^<>\n\r]+>.*?<SPECIAL_END>)", stdout, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # fallback: last non-empty line
        lines = [x.strip() for x in stdout.splitlines() if x.strip()]
        return lines[-1] if lines else ""

    def generate_one(self, query: str, context: str, meta: Optional[Dict[str, Any]] = None) -> str:
        cmd = [
            self.python_bin, self.infer_py,
            "--model_dir", self.model_dir,
            "--query", query,
            "--context", context,
            "--top_k", str(self.top_k),
            "--num_beams", str(self.num_beams),
            "--max_tools", str(self.max_tools),
            "--controlled", str(self.controlled),
            "--preselect_k", str(self.preselect_k),
        ]
        if self.base_model:
            cmd += ["--base_model_name", self.base_model]
        cmd += self.extra_args

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return self._extract_pred(p.stdout)


# -----------------------------
# 4) Evaluation runner
# -----------------------------
@dataclass
class EvalConfig:
    ks: List[int] = dataclasses.field(default_factory=lambda: [5, 10, 50])
    alpha: float = 0.4
    theta: float = 0.67
    max_tools: int = 8

def avg_metrics(all_results: List[Dict[int, Dict[str, float]]], ks: List[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {k: {} for k in ks}
    if not all_results:
        for k in ks:
            out[k] = {}
        return out

    keys = list(next(iter(all_results))[ks[0]].keys()) if all_results else []
    for k in ks:
        for kk in keys:
            out[k][kk] = sum(r[k][kk] for r in all_results) / len(all_results)
    return out

def format_table_soft(avg: Dict[int, Dict[str, float]], ks: List[int]) -> str:
    headers = ["@K", "Prec", "Rec", "F1", "nDCG", "MRR", "Avg-LLM", "Avg-ToolRec"]
    lines = []
    lines.append("".join([f"{h:>12s}" for h in headers]))
    lines.append("-" * (12 * len(headers)))
    for k in ks:
        m = avg[k]
        lines.append(
            f"{('@'+str(k)):>12s}"
            f"{m.get('Precision',0.0):12.4f}"
            f"{m.get('Recall',0.0):12.4f}"
            f"{m.get('F1',0.0):12.4f}"
            f"{m.get('Config-nDCG',0.0):12.4f}"
            f"{m.get('MRR',0.0):12.4f}"
            f"{m.get('Avg-LLM',0.0):12.4f}"
            f"{m.get('Avg-ToolRecall',0.0):12.4f}"
        )
    return "\n".join(lines)

def format_table_hard(avg: Dict[int, Dict[str, float]], ks: List[int]) -> str:
    headers = ["@K", "Prec", "Rec", "F1", "nDCG", "MRR"]
    lines = []
    lines.append("".join([f"{h:>12s}" for h in headers]))
    lines.append("-" * (12 * len(headers)))
    for k in ks:
        m = avg[k]
        lines.append(
            f"{('@'+str(k)):>12s}"
            f"{m.get('Precision',0.0):12.4f}"
            f"{m.get('Recall',0.0):12.4f}"
            f"{m.get('F1',0.0):12.4f}"
            f"{m.get('Config-nDCG',0.0):12.4f}"
            f"{m.get('MRR',0.0):12.4f}"
        )
    return "\n".join(lines)

def metric_explanations(alpha: float, theta: float) -> str:
    return (
        "\nMetric explanations:\n"
        f"- Soft similarity per (rec, gt): sim = alpha*LLM_match + (1-alpha)*Tool_recall\n"
        f"  where alpha may be overridden by GT type:\n"
        f"    * GT tool-only => alpha=0 (only Tool_recall matters)\n"
        f"    * GT model-only => alpha=1 (only LLM_match matters)\n"
        f"    * GT empty => sim=1 by definition\n"
        f"- LLM_match (mm): 1 if predicted <LLM_...> overlaps GT <LLM_...>, else 0. (If GT has no LLM, mm=1.)\n"
        f"- Tool_recall (tr): |T_pred ∩ T_gt| / |T_gt| (If GT has no tools, tr=1.)\n"
        f"- Hit under theta: for each rank, we match the rec agent to its best GT; hit iff soft_similarity >= theta ({theta}).\n"
        f"- Precision@K (soft): among top-K ranks, how many UNIQUE GT agents are covered (hit) / K.\n"
        f"- Recall@K (soft): among all GT agents, how many UNIQUE GT agents are covered (hit) / |GT|.\n"
        f"- F1@K (soft): harmonic mean of Precision@K and Recall@K.\n"
        f"- MRR@K (soft): 1/rank of the first hit in top-K (rank starts at 1).\n"
        f"- Config-nDCG@K (soft): ranking quality using soft relevance (max similarity to any GT) with logarithmic discount.\n"
        f"- Avg-LLM@K / Avg-ToolRec@K: average of mm and tr for each rank's best GT match within top-K.\n"
        f"\nHard evaluation:\n"
        f"- Exact match: hit iff (M_pred == M_gt) AND (T_pred == T_gt).\n"
        f"- Precision/Recall/F1/MRR/nDCG are computed the same style as above but using exact-match hits.\n"
    )

def evaluate_dataset(
    samples: List[Sample],
    generator: Generator,
    cfg: EvalConfig,
    save_per_sample_jsonl: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    per_sample_rows = []
    all_soft_metrics = []
    all_hard_metrics = []

    n = len(samples) if max_samples is None else min(len(samples), max_samples)

    for s in samples[:n]:
        pred_text = generator.generate_one(s.query, s.context, meta={"qid": s.qid, "idx": s.idx})
        gt_agents = parse_gt_agents_from_target(s.target)
        rec_agents = parse_ranked_agents_from_text(pred_text, max_tools=cfg.max_tools)

        soft_metrics, soft_breakdown = evaluate_agents_soft_detailed(
            gt_agents=gt_agents,
            rec_agents=rec_agents,
            ks=cfg.ks,
            alpha=cfg.alpha,
            theta=cfg.theta,
        )
        hard_metrics, hard_breakdown = evaluate_agents_hard_detailed(
            gt_agents=gt_agents,
            rec_agents=rec_agents,
            ks=cfg.ks,
        )

        all_soft_metrics.append(soft_metrics)
        all_hard_metrics.append(hard_metrics)

        row = {
            "idx": s.idx,
            "qid": s.qid,
            "query": s.query,
            "pred": pred_text,
            "target": s.target,
            "soft_metrics": soft_metrics,
            "hard_metrics": hard_metrics,
            "parsed": {
                "gt": [{"M": sorted(list(m)), "T": sorted(list(t))} for (m, t) in gt_agents],
                "rec": [{"M": sorted(list(m)), "T": sorted(list(t))} for (m, t) in rec_agents],
            },
            "breakdown": {
                "soft_by_k": soft_breakdown,   # per K: list of per-rank rows
                "hard_by_k": hard_breakdown,
            },
        }
        per_sample_rows.append(row)

    soft_avg = avg_metrics(all_soft_metrics, cfg.ks)
    hard_avg = avg_metrics(all_hard_metrics, cfg.ks)

    if save_per_sample_jsonl:
        os.makedirs(os.path.dirname(save_per_sample_jsonl) or ".", exist_ok=True)
        with open(save_per_sample_jsonl, "w", encoding="utf-8") as f:
            for r in per_sample_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "num_samples": n,
        "soft_avg": soft_avg,
        "hard_avg": hard_avg,
        "per_sample_path": save_per_sample_jsonl,
    }


# -----------------------------
# 5) CLI (ARGS UNCHANGED)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="pair_rag_valid.jsonl path")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")
    ap.add_argument("--save_per_sample", type=str, default="", help="optional jsonl to save per-sample metrics")

    # metrics knobs
    ap.add_argument("--ks", type=str, default="5,10,50")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--theta", type=float, default=0.67)
    ap.add_argument("--max_tools", type=int, default=8)

    # generator mode A: pred file
    ap.add_argument("--pred_jsonl", type=str, default="", help="predictions jsonl. If set, use FilePredGenerator.")

    # generator mode B: subprocess infer
    ap.add_argument("--infer_py", type=str, default="", help="your inference script path")
    ap.add_argument("--model_dir", type=str, default="", help="checkpoint/adaptor dir")
    ap.add_argument("--base_model", type=str, default="", help="base model (for PEFT)")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--controlled", type=int, default=0)
    ap.add_argument("--preselect_k", type=int, default=50)

    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    cfg = EvalConfig(ks=ks, alpha=args.alpha, theta=args.theta, max_tools=args.max_tools)

    samples = load_dataset(args.data)
    max_samples = None if args.max_samples == 0 else args.max_samples

    # choose generator
    if args.pred_jsonl:
        gen: Generator = FilePredGenerator(args.pred_jsonl)
    else:
        if not (args.infer_py and args.model_dir):
            raise SystemExit("Need either --pred_jsonl OR (--infer_py and --model_dir).")
        base_model = args.base_model.strip() or None
        gen = SubprocessGenerator(
            infer_py=args.infer_py,
            model_dir=args.model_dir,
            base_model=base_model,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_tools=args.max_tools,
            controlled=args.controlled,
            preselect_k=args.preselect_k,
        )

    save_path = args.save_per_sample.strip() or None
    report = evaluate_dataset(
        samples=samples,
        generator=gen,
        cfg=cfg,
        save_per_sample_jsonl=save_path,
        max_samples=max_samples,
    )

    print(f"\nEvaluated samples: {report['num_samples']}")

    print("\nAveraged metrics (SOFT evaluation):")
    print(format_table_soft(report["soft_avg"], cfg.ks))

    print("\nAveraged metrics (HARD exact-match evaluation):")
    print(format_table_hard(report["hard_avg"], cfg.ks))

    print(metric_explanations(alpha=cfg.alpha, theta=cfg.theta))

    if report.get("per_sample_path"):
        print(f"\nPer-sample (with per-rank breakdown) saved to: {report['per_sample_path']}")

if __name__ == "__main__":
    main()