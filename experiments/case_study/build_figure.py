"""Render the square, vector-first DNS bundle case-study figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


from agentcomposition.paths import ROOT
HERE = ROOT / 'outputs/case_study'
case = json.loads((HERE / "case_data.json").read_text())
cells = case["cells"]
critic = {key: value["critic_raw"] for key, value in cells.items()}
e2e = {key: value["e2e_total"] for key, value in cells.items()}
interaction = case["interaction_raw"]
assert abs(interaction - (critic["SDN"] - critic["SD"] - critic["SN"] + critic["S"])) < 1e-9
assert abs(case["additive_prediction_raw"] - (critic["SD"] + critic["SN"] - critic["S"])) < 1e-9

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=480)
fig.patch.set_facecolor("white")
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

navy = "#193449"
muted = "#4B6070"
light_rule = "#C8D8DE"
rose = "#FFF1ED"
sky = "#E7F4F9"
teal = "#CBE7E9"
selected_edge = "#157A8C"

ax.text(0.5, 0.956, "DNS bundle effect", ha="center", va="center",
        fontsize=13.0, weight="bold", color=navy)
ax.text(0.5, 0.900, "S similarity  |  D DNS  |  N NS", ha="center", va="center",
        fontsize=8.8, color=muted)
ax.text(0.5, 0.858, "Raw critic logit (large)  |  J: Terra judge /10",
        ha="center", va="center", fontsize=7.0, color=muted)
ax.text(0.43, 0.816, "N absent", ha="center", va="center", fontsize=9.2, color=navy)
ax.text(0.75, 0.816, "N present", ha="center", va="center", fontsize=9.2, color=navy)
ax.text(0.135, 0.660, "D off", ha="center", va="center", fontsize=9.0, color=navy)
ax.text(0.135, 0.400, "D on", ha="center", va="center", fontsize=9.0, color=navy)

cell_spec = [
    ("S", 0.278, 0.545, rose),
    ("SN", 0.606, 0.545, sky),
    ("SD", 0.278, 0.285, sky),
    ("SDN", 0.606, 0.285, teal),
]
for key, x, y, face in cell_spec:
    chosen = key == "SN"
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.292, 0.230,
        boxstyle="round,pad=0.008,rounding_size=0.020",
        linewidth=2.0 if chosen else 0.85,
        edgecolor=selected_edge if chosen else light_rule,
        facecolor=face,
    ))
    label = "+".join(key)
    ax.text(x + 0.146, y + 0.193, label, ha="center", va="center",
            fontsize=9.8 if len(label) <= 3 else 9.0, color=navy, weight="bold")
    ax.text(x + 0.146, y + 0.132, f"{critic[key]:+.2f}", ha="center", va="center",
            fontsize=13.6, color=navy, weight="bold")
    ax.text(x + 0.146, y + 0.071,
            f"J {e2e[key]:g}/10", ha="center", va="center",
            fontsize=8.7, color=muted)
    if key == "SDN":
        ax.text(x + 0.146, y + 0.022, "N not called", ha="center", va="center",
                fontsize=7.1, color=muted)
    if chosen:
        ax.text(x + 0.146, y + 0.022, "selected", ha="center", va="center",
                fontsize=7.1, weight="bold", color=selected_edge)

ax.plot([0.278, 0.906], [0.258, 0.258], color=light_rule, lw=1.0)
ax.text(0.425, 0.220, "Additive", ha="center", va="center", fontsize=8.5, color=muted)
ax.text(0.752, 0.220, "Observed", ha="center", va="center", fontsize=8.5, color=muted)
ax.text(0.425, 0.160, f"{case['additive_prediction_raw']:+.2f}",
        ha="center", va="center", fontsize=13.0, color=muted, weight="bold")
ax.text(0.752, 0.160, f"{critic['SDN']:+.2f}",
        ha="center", va="center", fontsize=13.0, color=navy, weight="bold")
ax.add_patch(FancyBboxPatch(
    (0.278, 0.030), 0.628, 0.086,
    boxstyle="round,pad=0.006,rounding_size=0.018",
    linewidth=0, facecolor=navy,
))
ax.text(0.592, 0.072, f"Raw critic Δ = {interaction:+.2f}", ha="center", va="center",
        fontsize=8.6, color="white", weight="bold")

for suffix in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"bundle_effect_case.{suffix}", dpi=480, facecolor="white")
plt.close(fig)
