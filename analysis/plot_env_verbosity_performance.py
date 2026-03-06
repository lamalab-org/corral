"""
Two-panel figure:
  Left  - horizontal bar plot of Average Score per environment x verbosity
  Right - Δ dot plot: change relative to "brief" baseline

Aggregation:
- Average over: models, agent types, categories (task/subtask), levels
- Group by: environment x Tool Verbosity

Output: analysis/results/figures/app_fig5_ver1.pdf
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "app_fig5_ver1.pdf"

records = []
with DATA_PATH.open() as fh:
    for raw_line in fh:
        stripped_line = raw_line.strip()
        if stripped_line:
            records.append(json.loads(stripped_line))

performance_df = pd.DataFrame(records)
performance_df = performance_df[
    [
        "model",
        "agent_type",
        "environment",
        "level",
        "category",
        "Tool Verbosity",
        "Average Score",
    ]
].copy()
performance_df["Average Score"] = pd.to_numeric(
    performance_df["Average Score"], errors="coerce"
)

agg = performance_df.groupby(["environment", "Tool Verbosity"], as_index=False)[
    "Average Score"
].mean()

VERBOSITIES = ["brief", "workflow", "comprehensive"]
VERBOSITY_LABELS = {
    "brief": "Brief",
    "workflow": "Workflow",
    "comprehensive": "Comprehensive",
}
COLORS = {"brief": "#4C72B0", "workflow": "#DD8452", "comprehensive": "#55A868"}
MARKERS = {"workflow": "D", "comprehensive": "o"}

envs = sorted(agg["environment"].unique())

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
}
yticklabels = [ENV_LABELS.get(e, e.capitalize()) for e in envs]

pivot = agg.pivot_table(
    index="environment", columns="Tool Verbosity", values="Average Score"
)
for v in VERBOSITIES:
    if v not in pivot.columns:
        pivot[v] = np.nan
pivot = pivot[VERBOSITIES]

BASELINE = "brief"
NON_BASELINE = ["workflow", "comprehensive"]
for v in NON_BASELINE:
    pivot[f"delta_{v}"] = pivot[v] - pivot[BASELINE]

n_envs = len(envs)
n_verb = len(VERBOSITIES)
bar_w = 0.22
gap = 0.1
x_centers = np.arange(n_envs) * (n_verb * bar_w + gap + bar_w * 0.3)

JITTER = {"workflow": -0.08, "comprehensive": 0.08}

fig, (ax_bar, ax_delta) = plt.subplots(
    1,
    2,
    figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

all_scores = []
for vi, verb in enumerate(VERBOSITIES):
    offsets = x_centers + (vi - (n_verb - 1) / 2) * bar_w
    scores = [pivot.loc[e, verb] if e in pivot.index else np.nan for e in envs]
    all_scores.extend([s for s in scores if not np.isnan(s)])
    valid_offsets = [o for o, s in zip(offsets, scores, strict=True) if not np.isnan(s)]
    valid_scores = [s for s in scores if not np.isnan(s)]
    ax_bar.hlines(
        valid_offsets,
        0,
        valid_scores,
        label=VERBOSITY_LABELS[verb],
        color=COLORS[verb],
        alpha=0.5,
        linewidth=5,
    )
    ax_bar.plot(
        valid_scores,
        valid_offsets,
        "o",
        markersize=5,
        color=COLORS[verb],
        alpha=0.6,
    )

ax_bar.set_yticks(x_centers)
ax_bar.set_yticklabels(yticklabels)
ax_bar.set_xlabel("Average Score")
ax_bar.set_ylabel("Environment")
range_frame(ax_bar, np.array([0, 1]), x_centers)

ax_delta.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

all_deltas = []
for v in NON_BASELINE:
    deltas = [pivot.loc[e, f"delta_{v}"] if e in pivot.index else np.nan for e in envs]
    all_deltas.extend([d for d in deltas if not np.isnan(d)])
    y_pos = x_centers + JITTER[v]
    ax_delta.scatter(
        deltas,
        y_pos,
        color=COLORS[v],
        marker=MARKERS[v],
        s=90,
        alpha=0.75,
        zorder=3,
        label=f"{VERBOSITY_LABELS[v]} - Brief",
        edgecolors="white",
        linewidths=0.6,
    )
    for yi, di in zip(y_pos, deltas, strict=True):
        if not np.isnan(di):
            ax_delta.plot(
                [0, di], [yi, yi], color=COLORS[v], linewidth=0.8, alpha=0.5, zorder=2
            )

ax_delta.set_yticks(x_centers)
ax_delta.set_yticklabels([])
ax_delta.set_xlabel("Δ Average Score  (vs. Brief)")
range_frame(ax_delta, np.array([-0.05, 0.05]), x_centers)

bar_handles, bar_labels = ax_bar.get_legend_handles_labels()
ax_bar.legend(bar_handles, bar_labels, title="Tool Verbosity", loc="upper right")

plt.tight_layout()
fig.savefig(OUT_FILE, bbox_inches="tight")
logger.info(f"Saved figure to {OUT_FILE}")
