"""
Figures summarizing Average Score as a function of tool verbosity.

Outputs:
- Two-panel environment summary:
    - Left  - horizontal bar plot of Average Score per environment x verbosity
    - Right - Δ dot plot: change relative to the "brief" baseline
- One-column model summary: average over environments for each model x verbosity
- One-column agent summary: average over environments for each agent type x verbosity
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "app_fig5_ver1.pdf"
OUT_FILE_MODEL = OUT_DIR / "app_fig5_ver1_model_avg.pdf"
OUT_FILE_AGENT = OUT_DIR / "app_fig5_ver1_agent_type_avg.pdf"

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

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
}
MODEL_LABELS = {
    "claude-4.5": "Claude 4.5",
    "gpt-4o": "GPT-4o",
    "gpt-oss-120b": "GPT-OSS-120B",
}
AGENT_TYPE_LABELS = {
    "react": "ReAct",
    "tool_calling": "Tool calling",
}


def build_pivot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg_df = df.groupby([group_col, "Tool Verbosity"], as_index=False)[
        "Average Score"
    ].mean()
    pivot_df = agg_df.pivot_table(
        index=group_col, columns="Tool Verbosity", values="Average Score"
    )
    for verbosity in VERBOSITIES:
        if verbosity not in pivot_df.columns:
            pivot_df[verbosity] = np.nan
    return pivot_df[VERBOSITIES]


def get_group_centers(
    n_groups: int, bar_width: float = 0.22, gap_width: float = 0.1
) -> np.ndarray:
    return np.arange(n_groups) * (
        len(VERBOSITIES) * bar_width + gap_width + bar_width * 0.3
    )


def get_group_frame(_bar_width: float, y_centers: np.ndarray) -> np.ndarray:
    return np.array([y_centers.min(), y_centers.max()])


def plot_horizontal_verbosity_bars(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    groups: list[str],
    group_labels: list[str],
    y_label: str,
    *,
    bar_width: float = 0.22,
    gap_width: float = 0.1,
    show_legend: bool = True,
) -> np.ndarray:
    y_centers = get_group_centers(len(groups), bar_width=bar_width, gap_width=gap_width)

    for vi, verbosity in enumerate(VERBOSITIES):
        offsets = y_centers + (vi - (len(VERBOSITIES) - 1) / 2) * bar_width
        scores = [
            pivot_df.loc[group, verbosity] if group in pivot_df.index else np.nan
            for group in groups
        ]
        valid_offsets = [
            offset
            for offset, score in zip(offsets, scores, strict=True)
            if not np.isnan(score)
        ]
        valid_scores = [score for score in scores if not np.isnan(score)]
        ax.hlines(
            valid_offsets,
            0,
            valid_scores,
            label=VERBOSITY_LABELS[verbosity],
            color=COLORS[verbosity],
            alpha=0.5,
            linewidth=5,
        )
        ax.plot(
            valid_scores,
            valid_offsets,
            "o",
            markersize=5,
            color=COLORS[verbosity],
            alpha=0.6,
        )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel("Average Score")
    ax.set_ylabel(y_label)
    range_frame(ax, np.array([0, 1]), get_group_frame(bar_width, y_centers))

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Tool Verbosity", loc="upper right")

    return y_centers


envs = sorted(agg["environment"].unique())
yticklabels = [ENV_LABELS.get(e, e.capitalize()) for e in envs]

env_pivot = build_pivot(performance_df, "environment")

BASELINE = "brief"
NON_BASELINE = ["workflow", "comprehensive"]
for v in NON_BASELINE:
    env_pivot[f"delta_{v}"] = env_pivot[v] - env_pivot[BASELINE]

x_centers = get_group_centers(len(envs))

JITTER = {"workflow": -0.08, "comprehensive": 0.08}

fig, (ax_bar, ax_delta) = plt.subplots(
    1,
    2,
    figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
    gridspec_kw={"width_ratios": [1.4, 1]},
)

all_scores = []
plot_horizontal_verbosity_bars(
    ax_bar,
    env_pivot,
    envs,
    yticklabels,
    "Environment",
    show_legend=False,
)

ax_delta.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

all_deltas = []
for v in NON_BASELINE:
    deltas = [
        env_pivot.loc[e, f"delta_{v}"] if e in env_pivot.index else np.nan for e in envs
    ]
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

models = sorted(performance_df["model"].dropna().unique())
model_labels = [MODEL_LABELS.get(model, model) for model in models]
model_pivot = build_pivot(performance_df, "model")

fig_model, ax_model = plt.subplots(
    1,
    1,
    figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT),
)
model_centers = plot_horizontal_verbosity_bars(
    ax_model,
    model_pivot,
    models,
    model_labels,
    "Model",
    bar_width=0.16,
    show_legend=False,
)
range_frame(ax_model, np.array([0, 1]), get_group_frame(0.16, model_centers), pad=0.2)
plt.tight_layout()
fig_model.savefig(OUT_FILE_MODEL, bbox_inches="tight")
logger.info(f"Saved figure to {OUT_FILE_MODEL}")

agent_types = sorted(performance_df["agent_type"].dropna().unique())
agent_type_labels = [
    AGENT_TYPE_LABELS.get(agent_type, agent_type) for agent_type in agent_types
]
agent_type_pivot = build_pivot(performance_df, "agent_type")

fig_agent, ax_agent = plt.subplots(
    1,
    1,
    figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT),
)
agent_centers = plot_horizontal_verbosity_bars(
    ax_agent,
    agent_type_pivot,
    agent_types,
    agent_type_labels,
    "Scaffold",
    bar_width=0.16,
    show_legend=False,
)
range_frame(ax_agent, np.array([0, 1]), np.array([0, 1]), pad=0.2)
plt.tight_layout()
fig_agent.savefig(OUT_FILE_AGENT, bbox_inches="tight")
logger.info(f"Saved figure to {OUT_FILE_AGENT}")
