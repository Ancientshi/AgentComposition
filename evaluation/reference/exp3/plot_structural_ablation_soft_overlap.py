#!/usr/bin/env python3
"""Plot TPAMI-style retrieval-grounded soft-overlap results for the structural ablation.

The input must be a summary produced by evaluate_beam_critic_experiments.py
v1.2 or later.  For each query, that evaluator averages component overlap
over up to the first five available candidates and then macro-averages across
queries.  Paper-facing retrieval-grounded Recall is the coverage-oriented primary metric; F1 is the
redundancy-sensitive companion metric. Any candidate containing an LLM or tool outside the
per-sample retrieval universe contributes zero soft-overlap credit.

Runs with no parsed candidates are shown as N/A, never as zero.  Runs with
fewer than five candidates remain plottable, but their actual average number
of evaluated candidates is exported to CSV and reported on stdout.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_SUMMARY = Path(
    str(AC_ROOT / 'outputs/') +
    "exp2_v13_structural_ablation_seed42_first100/"
    "evaluation/structural_ablation_evaluation_summary.json"
)

SETTINGS = (
    ("01_direct_free_generation", "Free"),
    ("02_constrained_greedy", "Greedy"),
    ("03_generator_guided_beam", "Gen. beam"),
    ("04_critic_final_reranking_only", "Final rerank"),
    ("05_full_model", "Full"),
)

METRICS = (
    (
        "component_recall",
        "Retrieval-Grounded Component Recall@$\\leq$5",
        ("metrics", "ranked", "top_5", "retrieval_grounded_candidate_mean_overlap", "component_recall"),
        "#315b7d",
    ),
    (
        "component_f1",
        "Retrieval-Grounded Component F1@$\\leq$5",
        ("metrics", "ranked", "top_5", "retrieval_grounded_candidate_mean_overlap", "component_f1"),
        "#a64b2a",
    ),
)

AVAILABILITY_PATH = (
    "metrics",
    "ranked",
    "top_5",
    "candidate_availability",
)


def nested_get(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def distribution_stats(value: Any) -> tuple[Optional[float], Optional[float], int]:
    if not isinstance(value, Mapping):
        return None, None, 0
    mean = as_float(value.get("mean"))
    std = as_float(value.get("std"))
    count_value = as_float(value.get("count"))
    count = int(count_value) if count_value is not None and count_value > 0 else 0
    if mean is None:
        return None, None, count
    ci95 = None
    if std is not None and count > 1:
        ci95 = 1.96 * std / math.sqrt(count)
    return 100.0 * mean, 100.0 * ci95 if ci95 is not None else None, count


def collect_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = summary.get("runs") or {}
    if not isinstance(runs, Mapping):
        raise ValueError("The summary has no 'runs' mapping.")

    rows: list[dict[str, Any]] = []
    missing_metric_fields: list[str] = []
    for index, (run_name, label) in enumerate(SETTINGS, start=1):
        run = runs.get(run_name)
        if not isinstance(run, Mapping):
            raise ValueError(f"Required structural-ablation run is missing: {run_name}")

        availability = nested_get(run, AVAILABILITY_PATH)
        available_mean = as_float(
            nested_get(availability or {}, ("available_candidate_count", "mean"))
        )
        evaluated_mean = as_float(
            nested_get(availability or {}, ("evaluated_candidate_count", "mean"))
        )
        full_top5_rate = as_float(
            nested_get(availability or {}, ("samples_with_at_least_cutoff", "rate"))
        )
        evaluable = available_mean is not None and available_mean > 0

        row: dict[str, Any] = {
            "setting_index": index,
            "run_name": run_name,
            "label": label,
            "available_candidate_count_mean": available_mean,
            "evaluated_candidate_count_mean": evaluated_mean,
            "samples_with_at_least_5_rate": full_top5_rate,
            "evaluable": evaluable,
        }
        for metric_key, _, path, _ in METRICS:
            distribution_value = nested_get(run, path)
            if distribution_value is None:
                missing_metric_fields.append(f"{run_name}: {'.'.join(path)}")
                row[f"{metric_key}_mean_percent"] = None
                row[f"{metric_key}_ci95_percent"] = None
                row[f"{metric_key}_sample_count"] = 0
                continue
            mean, ci95, count = distribution_stats(distribution_value)
            if not evaluable:
                mean, ci95 = None, None
            row[f"{metric_key}_mean_percent"] = mean
            row[f"{metric_key}_ci95_percent"] = ci95
            row[f"{metric_key}_sample_count"] = count
        rows.append(row)

    if missing_metric_fields:
        preview = "\n".join(f"  - {item}" for item in missing_metric_fields[:5])
        raise ValueError(
            "The summary does not contain the retrieval-grounded mean-candidate-overlap fields. "
            "Rerun evaluate_beam_critic_experiments.py before plotting. "
            "Missing examples:\n"
            f"{preview}"
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "setting_index",
        "run_name",
        "setting_label",
        "evaluable",
        "mean_component_recall_at_leq5_percent",
        "component_recall_ci95_percent",
        "mean_component_f1_at_leq5_percent",
        "component_f1_ci95_percent",
        "available_candidate_count_mean",
        "evaluated_candidate_count_mean",
        "samples_with_at_least_5_rate",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting_index": row["setting_index"],
                    "run_name": row["run_name"],
                    "setting_label": str(row["label"]).replace("\n", " "),
                    "evaluable": row["evaluable"],
                    "mean_component_recall_at_leq5_percent": row[
                        "component_recall_mean_percent"
                    ],
                    "component_recall_ci95_percent": row[
                        "component_recall_ci95_percent"
                    ],
                    "mean_component_f1_at_leq5_percent": row[
                        "component_f1_mean_percent"
                    ],
                    "component_f1_ci95_percent": row["component_f1_ci95_percent"],
                    "available_candidate_count_mean": row[
                        "available_candidate_count_mean"
                    ],
                    "evaluated_candidate_count_mean": row[
                        "evaluated_candidate_count_mean"
                    ],
                    "samples_with_at_least_5_rate": row[
                        "samples_with_at_least_5_rate"
                    ],
                }
            )


def plot(rows: Sequence[Mapping[str, Any]], output_prefix: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(rows), dtype=float)
    labels = [str(row["label"]) for row in rows]
    all_values = [
        float(row[f"{key}_mean_percent"])
        for key, _, _, _ in METRICS
        for row in rows
        if row[f"{key}_mean_percent"] is not None
    ]
    if not all_values:
        raise ValueError("No structural-ablation setting has evaluable soft-overlap scores.")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.82), sharex=True)
    for panel_index, (axis, (metric_key, title, _, color)) in enumerate(
        zip(axes, METRICS)
    ):
        means = np.array(
            [
                np.nan
                if row[f"{metric_key}_mean_percent"] is None
                else float(row[f"{metric_key}_mean_percent"])
                for row in rows
            ],
            dtype=float,
        )
        ci95 = np.array(
            [
                0.0
                if row[f"{metric_key}_ci95_percent"] is None
                else float(row[f"{metric_key}_ci95_percent"])
                for row in rows
            ],
            dtype=float,
        )
        valid = np.isfinite(means)
        upper_values = means[valid] + ci95[valid]
        panel_max = float(np.nanmax(upper_values)) if len(upper_values) else 1.0
        y_max = max(5.0, math.ceil((panel_max * 1.20) / 2.0) * 2.0)

        axis.plot(
            x,
            means,
            color=color,
            linewidth=1.25,
            alpha=0.72,
            zorder=2,
        )
        for point_index, row in enumerate(rows):
            mean = means[point_index]
            if not math.isfinite(float(mean)):
                axis.scatter(
                    [x[point_index]],
                    [0.0],
                    marker="x",
                    s=34,
                    color="#7f7f7f",
                    linewidths=1.1,
                    zorder=4,
                    clip_on=False,
                )
                axis.annotate(
                    "N/A",
                    (x[point_index], 0.0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="#666666",
                )
                continue

            is_full = row["run_name"] == "05_full_model"
            axis.errorbar(
                [x[point_index]],
                [mean],
                yerr=[[ci95[point_index]], [ci95[point_index]]],
                fmt="o",
                markersize=6.2 if is_full else 5.4,
                markerfacecolor=color,
                markeredgecolor="black" if is_full else "white",
                markeredgewidth=1.25 if is_full else 0.65,
                ecolor=color,
                elinewidth=0.85,
                capsize=2.3,
                capthick=0.8,
                zorder=4,
            )
            axis.annotate(
                f"{mean:.1f}",
                (x[point_index], mean + ci95[point_index]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold" if is_full else "normal",
                color="#222222",
            )

        axis.set_title(f"({chr(97 + panel_index)}) {title}", pad=6)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_xlim(-0.45, len(rows) - 0.55)
        axis.set_ylim(0.0, y_max)
        axis.set_ylabel("Score (%)")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.55, linestyle="--", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.015,
        "Error bars show 95% confidence intervals over queries; black outline denotes the full model.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.88, wspace=0.22)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Default: <summary-dir>/structural_ablation_grounded_soft_overlap",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    if not summary_path.exists():
        print(f"ERROR: summary does not exist: {summary_path}", file=sys.stderr)
        return 2
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    try:
        rows = collect_rows(summary)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix is not None
        else summary_path.parent / "structural_ablation_grounded_soft_overlap"
    )
    try:
        plot(rows, output_prefix)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    csv_path = output_prefix.with_suffix(".csv")
    write_csv(rows, csv_path)
    for row in rows:
        label = str(row["label"]).replace("\n", " ")
        evaluated = row["evaluated_candidate_count_mean"]
        if not row["evaluable"]:
            print(f"WARNING: {label}: no parsed candidates; plotted as N/A.")
        elif evaluated is not None and evaluated < 5.0:
            print(
                f"NOTE: {label}: averages over {evaluated:.2f} available candidates "
                "per query, not five."
            )
    print(f"Saved: {output_prefix.with_suffix('.pdf')}")
    print(f"Saved: {output_prefix.with_suffix('.png')}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())