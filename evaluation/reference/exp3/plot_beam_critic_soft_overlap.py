#!/usr/bin/env python3
"""Create a TPAMI-style beam/critic scatter matrix from the evaluation JSON.

The evaluator must be v1.2 or later so that Top-5 mean candidate overlap is
available.  Each plotted value first averages over the five candidates for a
query and then macro-averages over queries.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_SUMMARY = Path(
    str(AC_ROOT / 'outputs/') +
    "exp2_v13_ablation_beam_critic_seed42_first100/"
    "evaluation/beam_critic_evaluation_summary.json"
)

METRICS = (
    (
        "component_recall",
        "Mean Component Recall@5",
        ("metrics", "ranked", "top_5", "candidate_mean_overlap", "component_recall", "mean"),
    ),
    (
        "component_f1",
        "Mean Component F1@5",
        ("metrics", "ranked", "top_5", "candidate_mean_overlap", "component_f1", "mean"),
    ),
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


def beam_from_run(run_name: str, run: Mapping[str, Any]) -> tuple[int, int]:
    hyperparameters = run.get("hyperparameters") or {}
    beam_min = as_float(hyperparameters.get("beam_min_size"))
    beam_max = as_float(hyperparameters.get("beam_max_size"))
    if beam_min is not None and beam_max is not None:
        return int(beam_min), int(beam_max)
    match = re.search(r"beam_\d+_[^_]+_(\d+)-(\d+)", run_name)
    if not match:
        raise ValueError(f"Cannot determine beam range for {run_name!r}")
    return int(match.group(1)), int(match.group(2))


def critic_from_run(run_name: str, run: Mapping[str, Any]) -> float:
    hyperparameters = run.get("hyperparameters") or {}
    weight = as_float(hyperparameters.get("critic_score_weight"))
    if weight is not None:
        return weight
    match = re.search(r"critic_\d+_[^_]+_([0-9]+(?:p[0-9]+)?)$", run_name)
    if not match:
        raise ValueError(f"Cannot determine critic weight for {run_name!r}")
    return float(match.group(1).replace("p", "."))


def collect_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_name, run in (summary.get("runs") or {}).items():
        row: dict[str, Any] = {
            "run_name": run_name,
            "beam": beam_from_run(run_name, run),
            "critic_weight": critic_from_run(run_name, run),
        }
        for key, _, path in METRICS:
            value = as_float(nested_get(run, path))
            if value is None:
                missing.append(f"{run_name}: {'.'.join(path)}")
            else:
                row[key] = 100.0 * value
        rows.append(row)
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:5])
        raise ValueError(
            "The summary does not contain the v1.2 Top-5 mean-overlap fields. "
            "Rerun evaluate_beam_critic_experiments.py first. Missing examples:\n"
            f"{preview}"
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_name",
                "beam_min",
                "beam_max",
                "critic_weight",
                "mean_component_recall_at_5_percent",
                "mean_component_f1_at_5_percent",
            ),
        )
        writer.writeheader()
        for row in rows:
            beam_min, beam_max = row["beam"]
            writer.writerow(
                {
                    "run_name": row["run_name"],
                    "beam_min": beam_min,
                    "beam_max": beam_max,
                    "critic_weight": row["critic_weight"],
                    "mean_component_recall_at_5_percent": row["component_recall"],
                    "mean_component_f1_at_5_percent": row["component_f1"],
                }
            )


def plot(
    rows: Sequence[Mapping[str, Any]],
    output_prefix: Path,
    current_beam: tuple[int, int],
    current_critic: float,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    beams = sorted({tuple(row["beam"]) for row in rows})
    critic_weights = sorted({float(row["critic_weight"]) for row in rows})
    beam_position = {beam: index for index, beam in enumerate(beams)}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), sharey=True)
    for panel_index, (axis, (metric_key, title, _)) in enumerate(zip(axes, METRICS)):
        values = [float(row[metric_key]) for row in rows]
        value_min, value_max = min(values), max(values)
        if math.isclose(value_min, value_max):
            value_min -= 0.5
            value_max += 0.5
        normalizer = mpl.colors.Normalize(vmin=value_min, vmax=value_max)
        cmap = mpl.colormaps["cividis"]

        for row in rows:
            x = beam_position[tuple(row["beam"])]
            y = float(row["critic_weight"])
            value = float(row[metric_key])
            is_current = tuple(row["beam"]) == current_beam and math.isclose(
                y, current_critic
            )
            axis.scatter(
                [x],
                [y],
                c=[value],
                cmap=cmap,
                norm=normalizer,
                marker="s",
                s=520,
                edgecolors="black" if is_current else "white",
                linewidths=1.8 if is_current else 0.6,
                zorder=3,
            )
            relative = normalizer(value)
            axis.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if relative < 0.56 else "black",
                fontsize=7.2,
                fontweight="bold" if is_current else "normal",
                zorder=4,
            )

        axis.set_title(f"({chr(97 + panel_index)}) {title}", pad=6)
        axis.set_xlabel("Beam range (min–max)")
        axis.set_xticks(range(len(beams)))
        axis.set_xticklabels([f"{low}–{high}" for low, high in beams])
        axis.set_yticks(critic_weights)
        axis.set_ylim(min(critic_weights) - 0.25, max(critic_weights) + 0.25)
        axis.grid(color="#d9d9d9", linewidth=0.55, linestyle="--", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap),
            ax=axis,
            fraction=0.052,
            pad=0.025,
        )
        colorbar.set_label("Score (%)", labelpad=3)
        colorbar.ax.tick_params(labelsize=7)

    axes[0].set_ylabel(r"Critic weight $\lambda_c$")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.88, wspace=0.23)
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
        help="Default: <summary-dir>/beam_critic_soft_overlap_scatter",
    )
    parser.add_argument("--current-beam-min", type=int, default=3)
    parser.add_argument("--current-beam-max", type=int, default=5)
    parser.add_argument("--current-critic", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
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
        else summary_path.parent / "beam_critic_soft_overlap_scatter"
    )
    plot(
        rows,
        output_prefix,
        (args.current_beam_min, args.current_beam_max),
        args.current_critic,
    )
    csv_path = output_prefix.with_suffix(".csv")
    write_csv(rows, csv_path)
    print(f"Saved: {output_prefix.with_suffix('.pdf')}")
    print(f"Saved: {output_prefix.with_suffix('.png')}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
