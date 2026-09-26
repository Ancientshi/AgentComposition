#!/usr/bin/env python3
"""Run the five-setting v13 controllable-generation structural ablation.

This driver imports the existing run_infer_v13_batch_eval.py and changes only
the component gates needed by the ablation. It does not duplicate or replace
the retrieval, context construction, trie expansion, scoring, explanation, or
batch-record implementation from the baseline program.

The five settings are:
  1. direct deterministic free generation;
  2. trie-constrained greedy generation;
  3. trie-constrained generator-guided bundle beam search;
  4. the same beam search with the critic used only for final reranking; and
  5. the full delayed-critic search plus final critic reranking.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlparse


CURRENT_VARIANT: Dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_inference_script(config: Mapping[str, Any]) -> Path:
    candidates = config["paths"].get("inference_script_candidates", [])
    for raw_path in candidates:
        path = Path(str(raw_path)).expanduser().resolve()
        if path.is_file():
            return path
    rendered = "\n  - ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "Could not find run_infer_v13_batch_eval.py. Checked:\n  - " + rendered
    )


def variants_by_id(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    variants: Dict[str, Dict[str, Any]] = {}
    for raw_variant in config["variants"]:
        variant = dict(raw_variant)
        variant_id = str(variant["id"])
        if variant_id in variants:
            raise ValueError(f"Duplicate structural variant id: {variant_id}")
        variants[variant_id] = variant
    if len(variants) != 5:
        raise ValueError(f"Expected exactly five structural variants, got {len(variants)}")
    return variants


def validate_component_matrix(config: Mapping[str, Any]) -> None:
    variants = variants_by_id(config)
    expected = {
        "01_direct_free_generation": (False, False, False, False),
        "02_constrained_greedy": (True, False, False, False),
        "03_generator_guided_beam": (True, True, False, False),
        "04_critic_final_reranking_only": (True, True, False, True),
        "05_full_model": (True, True, True, True),
    }
    for variant_id, target in expected.items():
        row = variants.get(variant_id)
        if row is None:
            raise ValueError(f"Missing required structural variant: {variant_id}")
        actual = (
            bool(row["constrained_decoding"]),
            bool(row["bundle_beam_search"]),
            bool(row["search_critic"]),
            bool(row["final_critic_reranking"]),
        )
        if actual != target:
            raise ValueError(
                f"Invalid component matrix for {variant_id}: expected {target}, got {actual}"
            )


def select_baseline_manifest(config: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], str]:
    manifest_path = Path(config["paths"]["baseline_manifest"]).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Baseline manifest not found: {manifest_path}")

    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object at {manifest_path}:{line_number}")
            rows.append(row)

    expected = int(config["sample"].get("baseline_manifest_expected_size", 0))
    if expected and len(rows) != expected:
        raise ValueError(
            f"Expected {expected} baseline rows, found {len(rows)}. "
            "Refusing to silently change the controlled sample."
        )
    count = int(config["sample"]["count"])
    if count < 1 or len(rows) < count:
        raise ValueError(f"Cannot take first {count} rows from a {len(rows)}-row manifest")

    selected = rows[:count]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    return selected, text


def prepare_experiment(config_path: Path, config: Dict[str, Any]) -> Path:
    root = Path(config["paths"]["experiment_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    variants = variants_by_id(config)
    selected, manifest_text = select_baseline_manifest(config)
    manifest_hash = sha256_text(manifest_text)
    input_jsonl = str(Path(config["paths"]["input_jsonl"]).expanduser().resolve())

    for variant_id, variant in variants.items():
        run_dir = root / "runs" / variant_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "sample_manifest.jsonl"
        if manifest_path.exists():
            existing = manifest_path.read_text(encoding="utf-8")
            if sha256_text(existing) != manifest_hash:
                raise RuntimeError(
                    f"Existing manifest differs from the controlled first-100 prefix: {manifest_path}"
                )
        else:
            write_text_atomic(manifest_path, manifest_text)

        metadata = {
            "created_at": now_iso(),
            "input_jsonl": input_jsonl,
            "sample_size_requested": len(selected),
            "sample_size_actual": len(selected),
            "sample_seed": int(config["sample"]["seed"]),
            "sampling": "first N rows copied verbatim from the baseline 1000-sample manifest",
            "baseline_manifest": str(Path(config["paths"]["baseline_manifest"]).resolve()),
            "baseline_manifest_prefix_sha256": manifest_hash,
            "required_evaluation_fields": [
                "query",
                "target",
                "target_context_explanation",
            ],
        }
        write_json_atomic(run_dir / "experiment_metadata.json", metadata)
        write_json_atomic(
            run_dir / "variant_settings.json",
            {
                "experiment_name": config["experiment_name"],
                "variant": variant,
                "sample": config["sample"],
                "fixed_parameters": config["fixed_parameters"],
                "services": config["services"],
            },
        )

    resolved = copy.deepcopy(config)
    resolved["resolved_at"] = now_iso()
    resolved["source_config"] = str(config_path.resolve())
    resolved["resolved_inference_script"] = str(resolve_inference_script(config))
    resolved["selected_manifest_prefix_sha256"] = manifest_hash
    resolved["variant_count"] = len(variants)
    resolved_path = root / "structural_ablation_settings.json"
    write_json_atomic(resolved_path, resolved)
    return resolved_path


def tcp_preflight(url: str, timeout: float = 2.0) -> None:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid service URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with socket.create_connection((parsed.hostname, port), timeout=timeout):
        pass


def validate_environment(config: Mapping[str, Any]) -> None:
    validate_component_matrix(config)
    required_files = [
        Path(config["paths"]["input_jsonl"]),
        Path(config["paths"]["baseline_manifest"]),
    ]
    required_dirs = [Path(config["paths"]["model_dir"])]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise FileNotFoundError("Missing required paths:\n  - " + "\n  - ".join(missing))
    resolve_inference_script(config)

    if bool(config["services"].get("preflight_tcp_check", True)):
        failures = []
        for key in (
            "cf_llm_retr_url",
            "cf_tool_retr_url",
            "semantic_retr_url",
            "bundle_critic_url",
        ):
            try:
                tcp_preflight(str(config["services"][key]))
            except Exception as exc:
                failures.append(f"{key}={config['services'][key]} ({exc})")
        if failures:
            raise ConnectionError(
                "Required retrieval/critic services are not reachable:\n  - "
                + "\n  - ".join(failures)
            )

    requested = {int(value) for value in config["parallelism"]["gpu_ids"]}
    assignments = config["parallelism"]["assignments"]
    assigned = {int(value) for value in assignments}
    if requested != assigned:
        raise ValueError(
            f"GPU ids and assignment keys differ: gpu_ids={sorted(requested)}, "
            f"assignments={sorted(assigned)}"
        )

    assigned_variants = [item for values in assignments.values() for item in values]
    expected_variants = list(variants_by_id(config))
    if sorted(assigned_variants) != sorted(expected_variants):
        raise ValueError("Every structural variant must be assigned exactly once")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        available = {
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        }
        unavailable = sorted(requested - available)
        if unavailable:
            raise RuntimeError(f"Requested GPU IDs are unavailable: {unavailable}")
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found; skipping GPU index validation.", flush=True)


def import_inference_module(path: Path):
    module_name = "run_infer_v13_batch_eval_structural_import"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def structural_mode() -> str:
    return str(CURRENT_VARIANT.get("id", ""))


def use_search_critic() -> bool:
    return bool(CURRENT_VARIANT.get("search_critic", False))


def use_final_critic() -> bool:
    return bool(CURRENT_VARIANT.get("final_critic_reranking", False))


def install_structural_hooks(module: Any) -> None:
    """Install explicit critic-stage gates into the imported baseline module.

    The source transformation is deliberately fail-closed. If the baseline
    function changes and either expected snippet is no longer present, the
    experiment stops before loading the model instead of silently running a
    different ablation.
    """

    module._structural_mode = structural_mode
    module._structural_use_search_critic = use_search_critic
    module._structural_use_final_critic = use_final_critic

    source = inspect.getsource(module.critic_guided_tree_search)
    old_ineligible = (
        "        critic_ineligible = [n for n in expansions if n not in critic_eligible]\n"
    )
    new_ineligible = (
        "        if _structural_use_search_critic():\n"
        "            critic_ineligible = [n for n in expansions if n not in critic_eligible]\n"
        "        else:\n"
        "            critic_ineligible = list(expansions)\n"
        "            critic_eligible = []\n"
    )
    if source.count(old_ineligible) != 1:
        raise RuntimeError(
            "Baseline tree-search source changed: search-critic gate anchor not found exactly once"
        )
    source = source.replace(old_ineligible, new_ineligible, 1)

    old_no_eligible_reason = '''                "reason": (
                    "No candidate at this stage contains at least "
                    f"{DELAYED_CRITIC_MIN_REAL_TOOLS} real tools"
                ),
'''
    new_no_eligible_reason = '''                "reason": (
                    "Search-time Bundle Critic disabled by structural ablation"
                    if not _structural_use_search_critic()
                    else (
                        "No candidate at this stage contains at least "
                        f"{DELAYED_CRITIC_MIN_REAL_TOOLS} real tools"
                    )
                ),
'''
    if source.count(old_no_eligible_reason) != 1:
        raise RuntimeError(
            "Baseline tree-search source changed: critic-skip reason anchor not found exactly once"
        )
    source = source.replace(old_no_eligible_reason, new_no_eligible_reason, 1)

    old_final = '''    critic.score_nodes(query=query, nodes=completed, evidence_context=context, stage="final_rerank")
    assign_search_scores(
        completed,
        mode=search_score_mode,
        critic_weight=critic_weight,
        generator_weight=generator_weight,
        length_penalty=length_penalty,
    )
'''
    new_final = '''    if _structural_use_final_critic():
        critic.score_nodes(query=query, nodes=completed, evidence_context=context, stage="final_rerank")
        assign_search_scores(
            completed,
            mode=search_score_mode,
            critic_weight=critic_weight,
            generator_weight=generator_weight,
            length_penalty=length_penalty,
        )
    else:
        assign_generator_only_scores(completed, length_penalty=length_penalty)
'''
    if source.count(old_final) != 1:
        raise RuntimeError(
            "Baseline tree-search source changed: final-critic gate anchor not found exactly once"
        )
    source = source.replace(old_final, new_final, 1)

    old_algorithm = (
        '        "algorithm": "component_level_delayed_critic_tree_beam_search",\n'
    )
    new_algorithm = (
        '        "algorithm": f"component_level_tree_beam_search::{_structural_mode()}",\n'
    )
    if source.count(old_algorithm) != 1:
        raise RuntimeError(
            "Baseline tree-search source changed: algorithm trace anchor not found exactly once"
        )
    source = source.replace(old_algorithm, new_algorithm, 1)

    old_final_log = '''        f"[SEARCH] final rerank: scoring all {final_rerank_candidate_count} completed paths; "
        f"complete_pool_cap={complete_pool_size} return_top={num_results}",
'''
    new_final_log = '''        f"[SEARCH] final selection: candidates={final_rerank_candidate_count}; "
        f"critic_rerank={_structural_use_final_critic()} "
        f"complete_pool_cap={complete_pool_size} return_top={num_results}",
'''
    if source.count(old_final_log) != 1:
        raise RuntimeError(
            "Baseline tree-search source changed: final-selection log anchor not found exactly once"
        )
    source = source.replace(old_final_log, new_final_log, 1)
    exec(compile(source, str(Path(module.__file__).resolve()), "exec"), module.__dict__)

    original_client = module.BundleCriticClient

    class StructuralBundleCriticClient(original_client):
        def analyze_node(self, **kwargs):
            if _variant_final_critic_enabled():
                return super().analyze_node(**kwargs)
            node = kwargs.get("node")
            return {
                "skipped": True,
                "reason": "critic disabled by structural ablation",
                "variant": structural_mode(),
                "node_id": getattr(node, "node_id", None),
            }

    def _variant_final_critic_enabled() -> bool:
        return use_final_critic()

    module.BundleCriticClient = StructuralBundleCriticClient

    original_path_text = module.format_path_bundle_effect_text
    original_global_text = module.build_global_bundle_effect_text

    def structural_path_text(node: Any, analysis: Mapping[str, Any]) -> str:
        if use_final_critic():
            return original_path_text(node, analysis)
        return (
            "This candidate is ranked without Bundle Critic input. "
            f"It contains {len(node.tools)} selected tool token(s), has cumulative "
            f"generator log-probability {node.generator_logprob:.4f}, average "
            f"generated-token log-probability {node.generator_avg_logprob:.4f}, "
            f"and generator-only search score {float(node.search_score or 0.0):.4f}."
        )

    def structural_global_text(
        final_nodes: Sequence[Any], analyses: Mapping[str, Mapping[str, Any]], **kwargs: Any
    ) -> str:
        if use_final_critic():
            return original_global_text(final_nodes, analyses, **kwargs)
        if not final_nodes:
            return "No completed candidate bundles were available for comparison."
        ranking = "; ".join(
            f"#{index} {node.llm} with {len(node.tools)} tool(s), "
            f"generator-only score={float(node.search_score or 0.0):.4f}, "
            f"generator_avg={node.generator_avg_logprob:.4f}"
            for index, node in enumerate(final_nodes, 1)
        )
        return (
            f"The comparison covers {len(final_nodes)} completed candidates. "
            f"Generator-only ranking: {ranking}. No Bundle Critic score, marginal, "
            "or interaction signal is used in this structural variant."
        )

    module.format_path_bundle_effect_text = structural_path_text
    module.build_global_bundle_effect_text = structural_global_text

    original_run_pipeline = module.run_pipeline

    def tagged_run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
        record = original_run_pipeline(args)
        tag = {
            "experiment_name": "controllable_generation_structural_ablation",
            "variant": copy.deepcopy(CURRENT_VARIANT),
            "critic_policy": {
                "search": use_search_critic(),
                "final_reranking": use_final_critic(),
            },
        }
        record["structural_ablation"] = tag
        if isinstance(record.get("config"), dict):
            record["config"]["structural_ablation_mode"] = structural_mode()
        if isinstance(record.get("generation"), dict):
            record["generation"]["structural_ablation"] = tag
        output_json = str(getattr(args, "output_json", "") or "").strip()
        if output_json:
            module._write_json(output_json, record)
        return record

    module.run_pipeline = tagged_run_pipeline


def cli_tokens_for_variant(
    config: Mapping[str, Any], variant: Mapping[str, Any], run_dir: Path
) -> List[str]:
    fixed = config["fixed_parameters"]
    services = config["services"]
    sample = config["sample"]
    pairs: List[tuple[str, Any]] = [
        ("input_jsonl", config["paths"]["input_jsonl"]),
        ("sample_size", sample["count"]),
        ("sample_seed", sample["seed"]),
        ("experiment_dir", str(run_dir)),
        ("resume", fixed["resume"]),
        ("fail_fast", fixed["fail_fast"]),
        ("model_dir", config["paths"]["model_dir"]),
        ('retrieval_mode', fixed['retrieval_mode']),
        ("cf_llm_retr_url", services["cf_llm_retr_url"]),
        ("cf_tool_retr_url", services["cf_tool_retr_url"]),
        ("semantic_retr_url", services["semantic_retr_url"]),
        ("llm_topk", fixed["cf_llm_topk"]),
        ("cf_tool_bundle_topk", fixed["cf_tool_bundle_topk"]),
        ("semantic_targets", fixed["semantic_targets"]),
        ("semantic_tool_topk", fixed["semantic_tool_topk"]),
        ("semantic_rewrite", fixed["semantic_rewrite"]),
        ("final_top_k", fixed["final_top_k"]),
        ("controlled", variant["controlled"]),
        ("search_mode", variant["search_mode"]),
        ("top_k", variant["top_k"]),
        ("num_beams", variant["num_beams"]),
        ("allow_free_fallback", variant["allow_free_fallback"]),
        ("critic_url", services["bundle_critic_url"]),
        ("critic_required", variant["critic_required"]),
        ("beam_retain_ratio", fixed["beam_retain_ratio"]),
        ("beam_min_size", fixed["beam_min_size"]),
        ("beam_max_size", fixed["beam_max_size"]),
        ("llm_branch_factor", fixed["llm_branch_factor"]),
        ("tool_branch_factor", fixed["tool_branch_factor"]),
        ("num_results", fixed["num_results"]),
        ("complete_pool_size", fixed["complete_pool_size"]),
        ("search_score_mode", fixed["search_score_mode"]),
        ("critic_score_weight", fixed["critic_score_weight"]),
        ("generator_score_weight", fixed["generator_score_weight"]),
        ("search_length_penalty", fixed["search_length_penalty"]),
        ("max_tools", fixed["max_tools"]),
        ("min_tools", fixed["min_tools"]),
        ("dedup_tool_sets", fixed["dedup_tool_sets"]),
        ("include_pairwise_analysis", fixed["include_pairwise_analysis"]),
        ("generate_global_explanation", fixed["generate_global_explanation"]),
        ("answer_mode", fixed["answer_mode"]),
        ("max_source_length", fixed["max_source_length"]),
        ("max_new_tokens", fixed["max_new_tokens"]),
        ("max_explanation_tokens", fixed["max_explanation_tokens"]),
        ("max_global_explanation_tokens", fixed["max_global_explanation_tokens"]),
        ("phrase_score_batch_size", fixed["phrase_score_batch_size"]),
        ("explanation_batch_size", fixed["explanation_batch_size"]),
        ("search_checkpoint_interval", fixed["search_checkpoint_interval"]),
        ("progress", fixed["progress"]),
        ("progress_topn", fixed["progress_topn"]),
        ("progress_parent_interval", fixed["progress_parent_interval"]),
        ("device_map", fixed["device_map"]),
        ("torch_dtype", fixed["torch_dtype"]),
        ("do_sample", fixed["do_sample"]),
    ]
    tokens: List[str] = []
    for key, value in pairs:
        tokens.extend([f"--{key}", str(value)])
    return tokens


def parse_inference_args(module: Any, infer_script: Path, tokens: Sequence[str]):
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(infer_script), *tokens]
        return module.parse_args()
    finally:
        sys.argv = old_argv


def run_worker(config: Dict[str, Any], gpu: int, variant_ids: Sequence[str]) -> int:
    global CURRENT_VARIANT
    root = Path(config["paths"]["experiment_root"]).expanduser().resolve()
    variants = variants_by_id(config)
    status_path = root / f"worker_gpu{gpu}_status.json"
    worker_status: Dict[str, Any] = {
        "gpu_id": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": now_iso(),
        "variants": [],
    }
    write_json_atomic(status_path, worker_status)

    infer_script = resolve_inference_script(config)
    module = import_inference_module(infer_script)
    install_structural_hooks(module)
    failures = 0

    for position, variant_id in enumerate(variant_ids, 1):
        variant = variants[variant_id]
        CURRENT_VARIANT = copy.deepcopy(variant)
        run_dir = root / "runs" / variant_id
        console_path = run_dir / "console.log"
        print(
            f"[GPU {gpu}] ({position}/{len(variant_ids)}) starting "
            f"{variant_id}: {variant['label']}",
            flush=True,
        )
        entry: Dict[str, Any] = {
            "variant": variant,
            "started_at": now_iso(),
        }
        try:
            tokens = cli_tokens_for_variant(config, variant, run_dir)
            args = parse_inference_args(module, infer_script, tokens)
            with console_path.open("a", encoding="utf-8", buffering=1) as console:
                console.write(
                    f"\n===== STRUCTURAL ABLATION {variant_id} | GPU {gpu} | {now_iso()} =====\n"
                )
                old_stdout, old_stderr = sys.stdout, sys.stderr
                try:
                    sys.stdout = console
                    sys.stderr = console
                    summary = module.run_batch(args)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            entry["summary"] = summary
            entry["status"] = "succeeded" if bool(summary.get("ok")) else "failed"
            if entry["status"] != "succeeded":
                failures += 1
        except Exception as exc:
            failures += 1
            entry.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            with console_path.open("a", encoding="utf-8") as console:
                console.write(entry["traceback"] + "\n")
        entry["finished_at"] = now_iso()
        worker_status["variants"].append(entry)
        worker_status["updated_at"] = now_iso()
        write_json_atomic(status_path, worker_status)
        print(f"[GPU {gpu}] {variant_id}: {entry['status']}", flush=True)

    worker_status["finished_at"] = now_iso()
    worker_status["ok"] = failures == 0
    write_json_atomic(status_path, worker_status)
    return 0 if failures == 0 else 1


def run_master(config_path: Path, config: Dict[str, Any]) -> int:
    validate_environment(config)
    resolved_config = prepare_experiment(config_path, config)
    root = Path(config["paths"]["experiment_root"]).expanduser().resolve()
    assignments = {
        int(gpu): [str(item) for item in variants]
        for gpu, variants in config["parallelism"]["assignments"].items()
    }
    launch_record: Dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "started_at": now_iso(),
        "resolved_config": str(resolved_config),
        "variant_count": len(config["variants"]),
        "assignments": {str(gpu): values for gpu, values in assignments.items()},
        "workers": [],
    }
    status_path = root / "structural_ablation_status.json"
    write_json_atomic(status_path, launch_record)

    script_path = Path(__file__).resolve()
    children: List[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    print(f"[ABLATION] Prepared five runs under {root}", flush=True)
    print("[ABLATION] Launching persistent workers on GPUs 0,1,2,3", flush=True)
    try:
        for gpu in sorted(assignments):
            variant_ids = assignments[gpu]
            env = os.environ.copy()
            env.update({str(key): str(value) for key, value in config["environment"].items()})
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["PYTHONUNBUFFERED"] = "1"
            log_path = root / f"worker_gpu{gpu}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                "-u",
                str(script_path),
                "--config",
                str(config_path.resolve()),
                "--worker-gpu",
                str(gpu),
                "--variant-ids",
                ",".join(variant_ids),
            ]
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            children.append((gpu, process, log_handle, log_path))
            launch_record["workers"].append(
                {
                    "gpu_id": gpu,
                    "pid": process.pid,
                    "variant_ids": variant_ids,
                    "log": str(log_path),
                    "status": "running",
                }
            )
            print(f"[ABLATION] GPU {gpu}: {variant_ids}; log={log_path}", flush=True)
        write_json_atomic(status_path, launch_record)

        pending = {process.pid for _, process, _, _ in children}
        while pending:
            for gpu, process, _, _ in children:
                if process.pid in pending and process.poll() is not None:
                    pending.remove(process.pid)
                    print(
                        f"[ABLATION] GPU {gpu} worker exited with code {process.returncode}",
                        flush=True,
                    )
            if pending:
                time.sleep(5)
    except KeyboardInterrupt:
        print("[ABLATION] Interrupted; terminating GPU workers...", file=sys.stderr, flush=True)
        for _, process, _, _ in children:
            if process.poll() is None:
                process.terminate()
        for _, process, _, _ in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        launch_record["status"] = "interrupted"
        launch_record["finished_at"] = now_iso()
        write_json_atomic(status_path, launch_record)
        return 130
    finally:
        for _, _, handle, _ in children:
            handle.close()

    failed_workers = []
    for worker in launch_record["workers"]:
        process = next(item[1] for item in children if item[0] == worker["gpu_id"])
        worker["return_code"] = process.returncode
        worker["status"] = "succeeded" if process.returncode == 0 else "failed"
        if process.returncode != 0:
            failed_workers.append(worker["gpu_id"])
    launch_record["finished_at"] = now_iso()
    launch_record["status"] = "succeeded" if not failed_workers else "failed"
    launch_record["failed_gpu_workers"] = failed_workers
    write_json_atomic(status_path, launch_record)
    if failed_workers:
        print(
            f"[ABLATION][ERROR] Workers failed on GPUs {failed_workers}. "
            "Fix the reported issue and rerun the same shell; completed samples will resume.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"[ABLATION] All five runs finished. Status: {status_path}", flush=True)
    return 0


def parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--variant-ids", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_main_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if args.worker_gpu is not None:
        variant_ids = [item for item in args.variant_ids.split(",") if item]
        return run_worker(config, int(args.worker_gpu), variant_ids)
    return run_master(config_path, config)


if __name__ == "__main__":
    raise SystemExit(main())
