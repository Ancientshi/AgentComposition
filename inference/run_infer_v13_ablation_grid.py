#!/usr/bin/env python3
"""Run the 5 x 5 v13 beam-size/critic-weight ablation on eight GPUs.

The master process prepares one identical 100-sample manifest for every grid
point, balances the 25 jobs across the requested GPUs, and launches one worker
per GPU.  Each worker imports the existing v13 inference program once and runs
its assigned grid points sequentially, allowing the inference module's model
cache to reuse one loaded model on that GPU.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import copy
import hashlib
import importlib.util
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


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug_number(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def resolve_inference_script(config: Mapping[str, Any]) -> Path:
    candidates = config["paths"].get("inference_script_candidates", [])
    for raw in candidates:
        path = Path(str(raw)).expanduser().resolve()
        if path.is_file():
            return path
    rendered = "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not find run_infer_v13_batch_eval.py. Checked:\n  - " + rendered
    )


def build_jobs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    root = Path(config["paths"]["experiment_root"]).expanduser().resolve()
    for beam_index, beam in enumerate(config["grid"]["beam_settings"]):
        beam_min = int(beam["beam_min_size"])
        beam_max = int(beam["beam_max_size"])
        if not (1 <= beam_min <= beam_max):
            raise ValueError(f"Invalid beam range: {beam}")
        for critic_index, critic in enumerate(config["grid"]["critic_score_weights"]):
            weight = float(critic["value"])
            run_name = (
                f"beam_{beam_index + 1:02d}_{beam['label']}_{beam_min}-{beam_max}"
                f"__critic_{critic_index + 1:02d}_{critic['label']}_{slug_number(weight)}"
            )
            jobs.append(
                {
                    "job_index": len(jobs),
                    "run_name": run_name,
                    "experiment_dir": str(root / "runs" / run_name),
                    "beam_label": str(beam["label"]),
                    "beam_min_size": beam_min,
                    "beam_max_size": beam_max,
                    "critic_label": str(critic["label"]),
                    "critic_score_weight": weight,
                }
            )
    if len(jobs) != 25:
        raise ValueError(f"Expected a 5 x 5 grid (25 jobs), got {len(jobs)}")
    return jobs


def select_baseline_manifest(config: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], str]:
    sample_cfg = config["sample"]
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

    expected = int(sample_cfg.get("baseline_manifest_expected_size", 0))
    if expected and len(rows) != expected:
        raise ValueError(
            f"Expected {expected} rows in baseline manifest, found {len(rows)}. "
            "Refusing to silently change the controlled sample."
        )
    count = int(sample_cfg["count"])
    if count < 1 or len(rows) < count:
        raise ValueError(f"Cannot take first {count} rows from a {len(rows)}-row manifest")

    selected = rows[:count]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    return selected, text


def prepare_grid(config_path: Path, config: Dict[str, Any], jobs: List[Dict[str, Any]]) -> Path:
    root = Path(config["paths"]["experiment_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected, manifest_text = select_baseline_manifest(config)
    manifest_hash = sha256_text(manifest_text)
    input_jsonl = str(Path(config["paths"]["input_jsonl"]).expanduser().resolve())

    for job in jobs:
        run_dir = Path(job["experiment_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "sample_manifest.jsonl"
        if manifest_path.exists():
            existing = manifest_path.read_text(encoding="utf-8")
            if sha256_text(existing) != manifest_hash:
                raise RuntimeError(
                    f"Existing manifest differs from the controlled first-100 sample: {manifest_path}"
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
            "required_evaluation_fields": ["query", "target", "target_context_explanation"],
        }
        write_json_atomic(run_dir / "experiment_metadata.json", metadata)
        write_json_atomic(
            run_dir / "grid_point.json",
            {
                "experiment_name": config["experiment_name"],
                "job": job,
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
    resolved["grid_point_count"] = len(jobs)
    resolved["jobs"] = jobs
    resolved_path = root / "ablation_settings.json"
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

    requested = {int(x) for x in config["parallelism"]["gpu_ids"]}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        available = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        unavailable = sorted(requested - available)
        if unavailable:
            raise RuntimeError(f"Requested GPU IDs are unavailable: {unavailable}")
    except FileNotFoundError:
        print("[WARN] nvidia-smi was not found; skipping GPU index validation.", flush=True)


def balanced_assignments(
    jobs: Sequence[Dict[str, Any]], gpu_ids: Sequence[int], max_workers: int
) -> Dict[int, List[int]]:
    workers = [int(x) for x in gpu_ids[: max(1, int(max_workers))]]
    if not workers:
        raise ValueError("At least one GPU ID is required")
    assignment: Dict[int, List[int]] = {gpu: [] for gpu in workers}
    load: Dict[int, int] = {gpu: 0 for gpu in workers}
    # Beam maximum is a simple, deterministic proxy for expected runtime.
    ordered = sorted(jobs, key=lambda row: (-int(row["beam_max_size"]), row["job_index"]))
    for job in ordered:
        gpu = min(workers, key=lambda item: (load[item], item))
        assignment[gpu].append(int(job["job_index"]))
        load[gpu] += int(job["beam_max_size"])
    return assignment


def import_inference_module(path: Path):
    module_name = "run_infer_v13_batch_eval_ablation_import"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cli_tokens_for_job(config: Mapping[str, Any], job: Mapping[str, Any]) -> List[str]:
    fixed = config["fixed_parameters"]
    services = config["services"]
    sample = config["sample"]

    pairs: List[tuple[str, Any]] = [
        ("input_jsonl", config["paths"]["input_jsonl"]),
        ("sample_size", sample["count"]),
        ("sample_seed", sample["seed"]),
        ("experiment_dir", job["experiment_dir"]),
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
        ("search_mode", fixed["search_mode"]),
        ("critic_url", services["bundle_critic_url"]),
        ("critic_required", fixed["critic_required"]),
        ("beam_retain_ratio", fixed["beam_retain_ratio"]),
        ("beam_min_size", job["beam_min_size"]),
        ("beam_max_size", job["beam_max_size"]),
        ("llm_branch_factor", fixed["llm_branch_factor"]),
        ("tool_branch_factor", fixed["tool_branch_factor"]),
        ("num_results", fixed["num_results"]),
        ("complete_pool_size", fixed["complete_pool_size"]),
        ("search_score_mode", fixed["search_score_mode"]),
        ("critic_score_weight", job["critic_score_weight"]),
        ("generator_score_weight", fixed["generator_score_weight"]),
        ("search_length_penalty", fixed["search_length_penalty"]),
        ("max_tools", fixed["max_tools"]),
        ("min_tools", fixed["min_tools"]),
        ("dedup_tool_sets", fixed["dedup_tool_sets"]),
        ("include_pairwise_analysis", fixed["include_pairwise_analysis"]),
        ("generate_global_explanation", fixed["generate_global_explanation"]),
        ("answer_mode", fixed["answer_mode"]),
        ("max_source_length", fixed["max_source_length"]),
        ("max_explanation_tokens", fixed["max_explanation_tokens"]),
        ("max_global_explanation_tokens", fixed["max_global_explanation_tokens"]),
        ("allow_free_fallback", fixed["allow_free_fallback"]),
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


def run_worker(config: Dict[str, Any], gpu: int, indices: Sequence[int]) -> int:
    jobs = build_jobs(config)
    selected = [jobs[index] for index in indices]
    root = Path(config["paths"]["experiment_root"]).resolve()
    status_path = root / f"worker_gpu{gpu}_status.json"
    worker_status: Dict[str, Any] = {
        "gpu_id": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": now_iso(),
        "jobs": [],
    }
    write_json_atomic(status_path, worker_status)

    infer_script = resolve_inference_script(config)
    module = import_inference_module(infer_script)
    failures = 0
    for position, job in enumerate(selected, 1):
        run_dir = Path(job["experiment_dir"])
        print(
            f"[GPU {gpu}] ({position}/{len(selected)}) starting {job['run_name']}",
            flush=True,
        )
        entry: Dict[str, Any] = {"job": job, "started_at": now_iso()}
        console_path = run_dir / "console.log"
        try:
            args = parse_inference_args(module, infer_script, cli_tokens_for_job(config, job))
            with console_path.open("a", encoding="utf-8", buffering=1) as console:
                console.write(
                    f"\n===== ABLATION RUN {job['run_name']} | GPU {gpu} | {now_iso()} =====\n"
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
        worker_status["jobs"].append(entry)
        worker_status["updated_at"] = now_iso()
        write_json_atomic(status_path, worker_status)
        print(f"[GPU {gpu}] {job['run_name']}: {entry['status']}", flush=True)

    worker_status["finished_at"] = now_iso()
    worker_status["ok"] = failures == 0
    write_json_atomic(status_path, worker_status)
    return 0 if failures == 0 else 1


def run_master(config_path: Path, config: Dict[str, Any]) -> int:
    validate_environment(config)
    jobs = build_jobs(config)
    gpu_ids = [int(x) for x in config["parallelism"]["gpu_ids"]]
    assignments = balanced_assignments(
        jobs, gpu_ids, int(config["parallelism"].get("max_workers", len(gpu_ids)))
    )
    config_for_record = copy.deepcopy(config)
    config_for_record["resolved_gpu_assignments"] = {
        str(gpu): indices for gpu, indices in assignments.items()
    }
    resolved_config = prepare_grid(config_path, config_for_record, jobs)
    root = Path(config["paths"]["experiment_root"]).resolve()

    launch_record = {
        "experiment_name": config["experiment_name"],
        "started_at": now_iso(),
        "resolved_config": str(resolved_config),
        "job_count": len(jobs),
        "assignments": {str(gpu): values for gpu, values in assignments.items()},
        "workers": [],
    }
    status_path = root / "grid_status.json"
    write_json_atomic(status_path, launch_record)

    script_path = Path(__file__).resolve()
    children: List[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    print(f"[GRID] Prepared {len(jobs)} runs under {root}", flush=True)
    print(f"[GRID] Launching {len(assignments)} persistent GPU workers", flush=True)
    try:
        for gpu, indices in assignments.items():
            if not indices:
                continue
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in config.get("environment", {}).items()})
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
                "--job-indices",
                ",".join(str(index) for index in indices),
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
                    "job_indices": indices,
                    "log": str(log_path),
                    "status": "running",
                }
            )
            print(f"[GRID] GPU {gpu}: jobs={indices}, log={log_path}", flush=True)
        write_json_atomic(status_path, launch_record)

        pending = {process.pid for _, process, _, _ in children}
        while pending:
            for gpu, process, _, _ in children:
                if process.pid in pending and process.poll() is not None:
                    pending.remove(process.pid)
                    print(f"[GRID] GPU {gpu} worker exited with code {process.returncode}", flush=True)
            if pending:
                time.sleep(5)
    except KeyboardInterrupt:
        print("[GRID] Interrupted; terminating GPU workers...", file=sys.stderr, flush=True)
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
            f"[GRID][ERROR] Workers failed on GPUs {failed_workers}. "
            "Fix the reported issue and rerun the same shell; completed samples will resume.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"[GRID] All 25 runs finished successfully. Status: {status_path}", flush=True)
    return 0


def parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--job-indices", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_main_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if args.worker_gpu is not None:
        indices = [int(x) for x in args.job_indices.split(",") if x.strip()]
        return run_worker(config, int(args.worker_gpu), indices)
    return run_master(config_path, config)


if __name__ == "__main__":
    raise SystemExit(main())
