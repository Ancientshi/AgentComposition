#!/usr/bin/env python3
"""Baseline5 evaluation: use baseline4's exact scoring functions and Table 1 units."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.experiment_root.resolve()
    path = Path(__file__).with_name("evaluate_ranked_recall_baseline4.py")
    spec = importlib.util.spec_from_file_location("baseline4_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    config = json.loads((root / "baseline_config.json").read_text())
    model = config.get("model")
    if model not in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        raise SystemExit("Unknown GPT model in experiment config")
    label = "RAG-GPT-5.6-" + model.removeprefix("gpt-5.6-").capitalize()
    manifest = [json.loads(x) for x in (root / "sample_manifest.jsonl").read_text().splitlines() if x.strip()]
    if config.get("dry_run"):
        raise SystemExit("Cannot evaluate a dry run")
    if not manifest or len({x["sample_id"] for x in manifest}) != len(manifest):
        raise SystemExit("Empty or duplicate manifest")
    # Fail on missing API responses rather than silently changing the denominator.
    rows = []
    for sample in manifest:
        record = json.loads((root / "per_sample" / (sample["sample_id"] + ".json")).read_text())
        if not record.get("ok") or record["dataset_example"] != sample or record.get("baseline") != label:
            raise SystemExit("Incomplete/mismatched prediction")
        ev = evaluator.evaluate_one(record)
        ev["sample_id"] = sample["sample_id"]
        rows.append(ev)
    mapping = {"tool_recall": "top1_tool_recall", "top1_component_recall": "top1_component_recall",
               "complete_recall": "top1_complete_recall", "llm_accuracy": "top1_llm_accuracy",
               "tool_precision": "top1_tool_precision", "tool_f1": "top1_tool_f1"}
    mapping.update({f"{name}@{k}": f"{name}@{k}" for k in evaluator.CUTOFFS
                    for name in ("cr_hit", "cr_mrr", "tool_hit", "tool_mrr", "rdcr", "rdtr")})
    means = {key: statistics.fmean(r[field] for r in rows) for key, field in mapping.items()}
    means["cr_hit@1"] = means["complete_recall"]
    metrics = {"evaluation_name": "Baseline5 " + label, "model": model, "n": len(rows),
               "gold_source": "dataset_example.target only", "missing_ranks": "zero", "mean": means}
    table = {"ToolR@1": means["tool_recall"] * 100, "Tool-Hit@10": means["tool_hit@10"] * 100,
             "CompR@1": means["top1_component_recall"] * 100,
             "CR-Hit@1": means["cr_hit@1"] * 100, "CR-Hit@10": means["cr_hit@10"] * 100,
             "CR-MRR@10": means["cr_mrr@10"], "RDCR@10": means["rdcr@10"] * 100}
    latex = label + " & " + " & ".join(
        f"{value:.4f}" if key == "CR-MRR@10" else f"{value:.2f}" for key, value in table.items()) + r" \\"
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    (out / "rag_gpt_ranked_recall_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "table1_metrics.json").write_text(json.dumps({"units": "MRR raw; other columns x100", **table}, indent=2))
    (out / "table1_row.tex").write_text(latex + "\n")
    (out / "per_sample_evaluation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[evaluation] n={len(rows)}; {out}")
    print(latex)


if __name__ == "__main__":
    main()
