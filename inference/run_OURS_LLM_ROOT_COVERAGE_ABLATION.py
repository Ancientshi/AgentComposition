#!/usr/bin/env python3
"LLM ROOT COVERAGE ABLATION: separate LLM roots from the tool beam.\n\nImports the existing final-only SOTA implementation. No gold is passed into\ngeneration. No retrieval, rewriting or external GPT call is made. Explanations\nare skipped after selection while the ablation's generation prompt is preserved.\n"
from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

ROOT = AC_ROOT
SETTINGS = {
    "controlled": 1, "search_mode": "critic_tree", "allow_free_fallback": 0,
    "beam_retain_ratio": 0.4, "beam_min_size": 1, "beam_max_size": 2,
    "llm_branch_factor": 4, "tool_branch_factor": 5, "num_results": 10,
    "complete_pool_size": 50, "search_score_mode": "hybrid",
    "critic_required": 1, "critic_score_weight": 0.5,
    "generator_score_weight": 0.15, "search_length_penalty": 0.1,
    "min_tools": 1, "max_tools": 6, "dedup_tool_sets": 1,
    'retrieval_mode': "hybrid", "llm_topk": 10, "cf_tool_bundle_topk": 5,
    "semantic_targets": "tool", "semantic_llm_topk": 0, "semantic_tool_topk": 25,
    "final_top_k": 25, "rewrite": 0, "cf_rewrite": 0, "semantic_rewrite": 0,
    "answer_mode": "target_only", "include_pairwise_analysis": 0,
    "generate_global_explanation": 0, "max_source_length": 8192,
    "phrase_score_batch_size": 8, "search_checkpoint_interval": 1,
    "progress": 1, "progress_topn": 3, "progress_parent_interval": 5,
    "device_map": "auto", "torch_dtype": "auto", "do_sample": 0, "token": 0,
    # Same prompt text as the original ablations; only post-ranking narrative
    # computation is omitted, so switching answer_mode does not alter search.
    "task_text": "Select the best target agent from the provided context, then explain the choice "
                 "using the retrieved evidence. Output the target agent first.",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def source_records(source):
    manifest = read_lines(source / "sample_manifest.jsonl")
    records = read_lines(source / "results.jsonl")
    ids = [s["sample_id"] for s in manifest]
    by_id = {r["dataset_example"]["sample_id"]: r for r in records}
    if len(set(ids)) != len(ids) or len(by_id) != len(records) or set(ids) != set(by_id):
        raise ValueError("Reference manifest/results must contain the same unique sample IDs")
    ordered = []
    for sample in manifest:
        row = by_id[sample["sample_id"]]
        if sample != row["dataset_example"] or not row.get("ok", True):
            raise ValueError("Reference result does not match manifest")
        cfg = row["config"]
        if cfg["cf_rewrite"] or cfg["semantic_rewrite"]:
            raise ValueError("Reference retrieval used rewriting")
        if (cfg["cf_llm_topk"], cfg["cf_tool_bundle_topk"], cfg["semantic_tool_topk"]) != (10, 5, 25):
            raise ValueError("Reference retrieval budgets differ from Table 1")
        if row["query"] != sample["query"].strip() or not row["context"]["text"].strip():
            raise ValueError("Reference query/context missing or mismatched")
        ordered.append(row)
    return manifest, ordered


def critic_health(url):
    # Local service must not inherit an external HTTP(S) proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + "/health", timeout=10) as response:
            health = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Critic unavailable at {url}; start the Bundle Critic service first") from exc
    if health.get("status") != "ok":
        raise RuntimeError("Critic health is not OK")
    return health


def configure(sota, opts):
    values = {**SETTINGS, "model_dir": str(opts.model_dir), "base_model_name": str(opts.base_model),
              "critic_url": opts.critic_url}
    tokens = [word for key, value in values.items() for word in ("--" + key, str(value))]
    previous = sys.argv
    try:
        sys.argv = [str(sota.__file__), *tokens]
        args = sota.parse_args()
    finally:
        sys.argv = previous
    original = sota.BundleCriticClient

    class FinalOnlyClient(original):
        def score_nodes(self, *, stage, nodes, **kwargs):
            if stage != "final_rerank" or not all(n.is_complete for n in nodes):
                raise RuntimeError("Only completed bundles may be passed to the final critic")
            return super().score_nodes(stage=stage, nodes=nodes, **kwargs)

        def analyze_node(self, **kwargs):
            # Post-selection explanatory diagnostics cannot affect ranking.
            return {"skipped": True, "reason": "Table 1 evaluates ranking only"}

    sota.BundleCriticClient = FinalOnlyClient
    return args


def validate_prediction(record):
    generation = record["generation"]
    trace = generation["search_trace"]
    if trace.get("search_critic_enabled") is not False or trace.get("final_critic_reranking") is not True:
        raise ValueError("Unexpected critic placement in search trace")
    root_keep = trace["llm_root_keep"]
    scope = trace["tool_beam_scope"]
    if root_keep > 0:
        expected = min(root_keep, trace["llm_candidate_count"])
        if trace["levels"][0]["prune"]["retained_count"] != expected:
            raise ValueError("Independent LLM root count was not preserved")
    for level in trace["levels"][1:]:
        pruning = level["prune"]
        if scope == "per_llm":
            if pruning.get("scope") != "per_llm":
                raise ValueError("Tool beam unexpectedly reverted to global pruning")
            if any(g["retained_count"] > 2 for g in pruning["groups"]):
                raise ValueError("Per-LLM tool beam exceeded 2")
        elif pruning["retained_count"] > 2:
            raise ValueError("Global tool beam exceeded 2")
    events = generation["critic_api_events"]
    if len(events) != 1 or events[0].get("stage") != "final_rerank" or events[0].get("error"):
        raise ValueError("Expected exactly one successful final critic event")
    results = record["results"]
    if not 1 <= len(results) <= 10:
        raise ValueError("Expected 1..10 ranked configurations")
    keys = [(r["llm_token"], tuple(sorted(r["tool_tokens"]))) for r in results]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate ranked configurations")
    if [r["rank"] for r in results] != list(range(1, len(results) + 1)):
        raise ValueError("Unexpected ranks")


def evaluate(output):
    evaluator = load_module(ROOT / 'evaluation/reference/exp3/evaluate_ranked_recall_baseline4.py', "ours_table1_evaluator")
    manifest = read_lines(output / "sample_manifest.jsonl")
    config = json.loads((output / "ours_config.json").read_text())
    rows, counts = [], []
    for sample in manifest:
        record = json.loads((output / "per_sample" / (sample["sample_id"] + ".json")).read_text())
        if not record.get("ok") or record["dataset_example"] != sample or record.get("config_sha256") != sha(config):
            raise ValueError("Missing, failed or mismatched prediction; refusing partial Table 1 scores")
        validate_prediction(record)
        ev = evaluator.evaluate_one(record)
        ev["sample_id"] = sample["sample_id"]
        rows.append(ev)
        counts.append(len(record["results"]))
    if not rows:
        raise ValueError("No samples to evaluate")
    fields = ["top1_tool_recall", "tool_hit@10", "top1_component_recall", "top1_complete_recall",
              "cr_hit@10", "cr_mrr@10", "rdcr@10"]
    labels = ["ToolR@1", "Tool-Hit@10", "CompR@1", "CR-Hit@1", "CR-Hit@10", "CR-MRR@10", "RDCR@10"]
    raw = {label: statistics.fmean(r[field] for r in rows) for label, field in zip(labels, fields)}
    table = {key: value if key == "CR-MRR@10" else value * 100 for key, value in raw.items()}
    latex = r"\textbf{Ours} & " + " & ".join(f"{v:.4f}" if k == "CR-MRR@10" else f"{v:.2f}" for k,v in table.items()) + r" \\"
    out = output / "evaluation"
    write_json(out / "ours_ranked_recall_metrics.json", {"n": len(rows), "gold_source": "target_only",
        "missing_ranks": "zero", "raw": raw, "table1": table,
        "result_count_min": min(counts), "result_count_max": max(counts), "result_count_mean": statistics.fmean(counts)})
    (out / "table1_row.tex").write_text(latex + "\n")
    (out / "per_sample_evaluation.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"[EVAL] n={len(rows)}; {latex}", flush=True)
    from llm_root_coverage_diagnostics import audit_experiment
    audit_experiment(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_experiment", type=Path, default=ROOT / 'outputs/baseline4_rag_llama_instruct_seed42_n100')
    ap.add_argument("--experiment_dir", type=Path, default=None)
    ap.add_argument("--llm_root_keep", type=int, choices=[0, 2, 4, 10], default=10)
    ap.add_argument("--tool_beam_scope", choices=["global", "per_llm"], default="per_llm")
    ap.add_argument("--model_dir", type=Path, default=ROOT / 'checkpoints/generative_v9')
    ap.add_argument("--base_model", type=Path, default=Path(str(AC_BASE_MODEL)))
    ap.add_argument("--critic_url", default=AC_ENV('CRITIC_URL','http://127.0.0.1:8015'))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--evaluate_only", action="store_true")
    opts = ap.parse_args()
    if opts.limit < 0:
        ap.error("limit must be nonnegative")
    if opts.llm_root_keep == 0 and opts.tool_beam_scope != "global":
        ap.error("Legacy root pruning is a global-beam control only")
    SETTINGS.update(llm_root_keep=opts.llm_root_keep, tool_beam_scope=opts.tool_beam_scope,
                    llm_branch_factor=4 if opts.llm_root_keep == 0 else 0)
    variant = (f"ROOTS_{opts.llm_root_keep:02d}__TOOL_BEAM_{opts.tool_beam_scope.upper()}_1-2"
               "__CRITIC_FINAL_0p5__NO_REWRITE__TOP10__seed42_n100")
    if opts.limit:
        variant += f"__SMOKE_{opts.limit}"
    if opts.experiment_dir is None:
        opts.experiment_dir = ROOT / 'outputs/ABLATION_LLM_ROOT_COVERAGE' / variant
    source, output = opts.source_experiment.resolve(), opts.experiment_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Reference and output directories must be separate")
    if opts.evaluate_only:
        evaluate(output)
        return 0
    manifest, rows = source_records(source)
    if opts.limit:
        manifest, rows = manifest[:opts.limit], rows[:opts.limit]
    if not rows:
        raise ValueError("No reference samples")
    for path in (opts.model_dir / "adapter_model.safetensors", opts.base_model / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    health = critic_health(opts.critic_url)
    sota_path = ROOT / "run_infer_v13_sota_LLM_ROOT_COVERAGE_ABLATION.py"
    sota = load_module(sota_path, "ours_sota")
    args = configure(sota, opts)
    # Verify preserved pre-generation prompt and critic rejection before loading weights.
    example = rows[0]
    assert sota.build_prompt_v12(context=example["context"]["text"], query=example["query"], want_explanation=True) == sota.build_prompt_v12(
        context=example["context"]["text"], query=example["query"], want_explanation=False, task_text=args.task_text)
    try:
        sota.BundleCriticClient(opts.critic_url).score_nodes(query="test", nodes=[], evidence_context="", stage="tool_depth_2")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Search-time critic was not blocked")
    print(f"[LLM ROOT COVERAGE ABLATION] {variant}; {len(rows)} samples; output={output}", flush=True)
    if opts.check_only:
        print(f"[CHECK] Passed; Python={sys.executable}; critic={health['status']}; no model loaded")
        return 0
    config = {"method": "Ours LLM root coverage ablation", "variant": variant,
              "evaluation_role": "paired diagnostic on existing Table-1 samples, not held-out hyperparameter selection",
              "seed": 42, "sample_count": len(rows), "limit": opts.limit,
              "settings": SETTINGS, "source_experiment": str(source), "source_results_sha256": file_sha(source / "results.jsonl"),
              "sota_sha256": file_sha(sota_path), "runner_sha256": file_sha(__file__),
              "model_dir": str(opts.model_dir.resolve()), "base_model": str(opts.base_model.resolve()),
              "adapter_config_sha256": file_sha(opts.model_dir / "adapter_config.json"),
              "critic_url": opts.critic_url, "critic_model": health.get("model"), "python": sys.executable}
    config_path = output / "ours_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing configuration differs; choose a new experiment_dir")
    write_json(config_path, config)
    fingerprint = sha(config)
    (output / "sample_manifest.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in manifest))
    import torch
    import random
    import numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; check agentrec environment and CUDA_VISIBLE_DEVICES")
    torch.cuda.manual_seed_all(42)
    completed, failures = [], []
    for index, (sample, source_row) in enumerate(zip(manifest, rows), 1):
        sid = sample["sample_id"]
        if not re.fullmatch(r"[\w.-]+", sid):
            raise ValueError("Invalid sample ID")
        path = output / "per_sample" / (sid + ".json")
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("ok"):
                if prior.get("config_sha256") != fingerprint or prior.get("dataset_example") != sample:
                    raise ValueError("Resume record mismatches config/sample")
                validate_prediction(prior)
                completed.append(prior)
                print(f"[RESUME] {index}/{len(rows)} {sid}", flush=True)
                continue
        sample_args = copy.copy(args)
        sample_args.query = source_row["query"]
        sample_args.context = source_row["context"]["text"]
        sample_args.output_json = str(path)
        log_path = output / "logs" / (sid + ".log")
        log_path.parent.mkdir(exist_ok=True)
        started = time.time()
        try:
            with log_path.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                record = sota.run_pipeline(sample_args)
            validate_prediction(record)
            record.update(ok=True, method="Ours", config_sha256=fingerprint, dataset_example=sample,
                          context=source_row["context"], retrieval=source_row["retrieval"],
                          latency_sec=round(time.time()-started, 3),
                          input_provenance={"source_experiment": str(source), "context_sha256": sha(source_row["context"]["text"])})
            write_json(path, record)
            completed.append(record)
            print(f"[OK] {index}/{len(rows)} {sid}; candidates={len(record['results'])}; {record['latency_sec']}s", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sid, "error": repr(exc), "log": str(log_path)})
            print(f"[ERROR] {sid}: {exc}; see {log_path}", file=sys.stderr, flush=True)
            break
        finally:
            write_json(output / "run_summary.json", {"ok": len(completed) == len(rows), "expected_samples": len(rows),
                       "completed_samples": len(completed), "failures": failures})
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in completed))
    write_json(output / "run_summary.json", {"ok": len(completed) == len(rows), "expected_samples": len(rows),
               "completed_samples": len(completed), "failures": failures})
    if len(completed) != len(rows):
        return 1
    evaluate(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
