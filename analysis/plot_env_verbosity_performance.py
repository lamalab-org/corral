"""Summarize how benchmark scores shift as tool verbosity increases.

The figure combines absolute score comparisons with a delta view relative to the
brief setting so small but consistent gains or regressions remain visible even
when the raw scores cluster tightly near one another.
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.transforms import blended_transform_factory

lama_aesthetics.get_style("main")


def to_title_case_label(text: str) -> str:
    """Capitalize the first alphabetic character in the full label.

    The requested label style uses an initial capital letter followed by
    lowercase text rather than conventional title case.
    """

    lowered = text.lower()
    for idx, char in enumerate(lowered):
        if char.isalpha():
            return f"{lowered[:idx]}{char.upper()}{lowered[idx + 1:]}"
    return lowered


DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "fig_5_app"
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
    "brief": to_title_case_label("brief"),
    "workflow": to_title_case_label("workflow"),
    "comprehensive": to_title_case_label("comprehensive"),
}
COLORS = {"brief": "#4C72B0", "workflow": "#DD8452", "comprehensive": "#55A868"}
MARKERS = {"workflow": "D", "comprehensive": "o"}

ENV_LABELS = {
    "afm": to_title_case_label("afm experimental\nexecution"),
    "catalyst": to_title_case_label("adsorption surface\nconstruction"),
    "md": to_title_case_label("molecular\nsimulation"),
    "ml": to_title_case_label("ml-based\nproperty"),
    "resistor": to_title_case_label("circuit inference"),
    "retro": to_title_case_label("retrosynthetic\nplanning"),
    "spectra": to_title_case_label("spectroscopic\nstructure elucidation"),
    "wetlab": to_title_case_label("inorganic\nqualitative analysis"),
}
MODEL_LABELS = {
    "claude-4.5": "Claude-4.5-Sonnet",
    "gpt-4o": "GPT-4o",
    "gpt-oss-120b": "gpt-oss-120b",
}
AGENT_TYPE_LABELS = {
    "react": "ReAct",
    "tool_calling": to_title_case_label("tool calling"),
}


def build_pivot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Average scores by verbosity for one comparison axis.

    Missing verbosity levels are inserted explicitly so every panel shares the
    same ordering even when a slice has incomplete data.

    Args:
        df: Benchmark score table.
        group_col: Column that defines the rows of the pivot table.

    Returns:
        pd.DataFrame: Pivot table with verbosity columns in canonical order.
    """
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
    n_groups: int, bar_width: float = 0.22, gap_width: float = 0.04
) -> np.ndarray:
    """Compute categorical anchors for grouped horizontal plots.

    Args:
        n_groups: Number of groups to place on the axis.
        bar_width: Vertical spacing allocated to one verbosity level.
        gap_width: Additional spacing between neighboring groups.

    Returns:
        np.ndarray: Center coordinate for each group.
    """
    return np.arange(n_groups) * (len(VERBOSITIES) * bar_width + gap_width)


def get_group_frame(_bar_width: float, y_centers: np.ndarray) -> np.ndarray:
    """Return the y-range that encloses the grouped positions.

    Args:
        _bar_width: Unused legacy parameter kept for call-site symmetry.
        y_centers: Group center coordinates on the y-axis.

    Returns:
        np.ndarray: Two-value range passed to `range_frame()`.
    """
    return np.array([y_centers.min(), y_centers.max()])


def plot_horizontal_verbosity_bars(
    ax: plt.Axes,
    pivot_df: pd.DataFrame,
    groups: list[str],
    group_labels: list[str],
    y_label: str | None = None,
    *,
    bar_width: float = 0.22,
    gap_width: float = 0.1,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw grouped horizontal lollipops for verbosity-specific mean scores.

    Args:
        ax: Axis that receives the plot.
        pivot_df: Pivot table indexed by group with verbosity columns.
        groups: Ordered raw group identifiers to plot.
        group_labels: Display labels corresponding to `groups`.
        y_label: Label shown on the categorical axis.
        bar_width: Vertical spacing allocated to one verbosity level.
        gap_width: Additional spacing between neighboring groups.
        show_legend: Whether to draw the verbosity legend on this axis.

    Returns:
        np.ndarray: Group-center coordinates used for further annotations.
    """
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

    ax.set_xlabel(to_title_case_label("average score"))
    if y_label is not None:
        ax.set_ylabel(y_label)
    range_frame(ax, np.array([0, 1]), y_centers)
    # Override y-axis: spine bounds must match categorical tick positions
    ax.spines["left"].set_bounds(y_centers[0], y_centers[-1])
    y_pad = bar_width * 1.8
    ax.set_ylim(y_centers[0] - y_pad, y_centers[-1] + y_pad)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(group_labels)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title=to_title_case_label("tool verbosity"),
            loc="upper right",
        )

    return y_centers


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.18) -> None:
    """Place a bold panel tag outside the axis bounds.

    Args:
        ax: Axis that receives the panel tag.
        label: Single-letter panel identifier.
        x: Horizontal offset in axes coordinates.

    Returns:
        None: The function mutates the provided axis.
    """

    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=16,
        color="black",
        ha="center",
        va="center",
        clip_on=False,
    )


envs = sorted(agg["environment"].unique())
yticklabels = [
    ENV_LABELS.get(e, to_title_case_label(e.replace("_", " "))) for e in envs
]

env_pivot = build_pivot(performance_df, "environment")

BASELINE = "brief"
NON_BASELINE = ["workflow", "comprehensive"]
for v in NON_BASELINE:
    env_pivot[f"delta_{v}"] = env_pivot[v] - env_pivot[BASELINE]

JITTER = {"workflow": -0.08, "comprehensive": 0.08}


def plot_environment_summary(ax_bar: plt.Axes, ax_delta: plt.Axes) -> None:
    """Draw the environment summary and the delta-from-brief companion view.

    The delta panel exists to surface consistent but small verbosity effects
    that can disappear in the absolute-score panel.

    Args:
        ax_bar: Axis for the absolute environment means.
        ax_delta: Axis for the environment deltas relative to `brief`.

    Returns:
        None: The function mutates both provided axes.
    """
    x_centers = get_group_centers(len(envs))

    plot_horizontal_verbosity_bars(
        ax_bar,
        env_pivot,
        envs,
        yticklabels,
        show_legend=False,
    )

    ax_delta.axvline(
        0, color="black", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1
    )

    all_deltas = [0]
    for v in NON_BASELINE:
        deltas = [
            env_pivot.loc[e, f"delta_{v}"] if e in env_pivot.index else np.nan
            for e in envs
        ]
        all_deltas.extend(d for d in deltas if not np.isnan(d))
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
                    [0, di],
                    [yi, yi],
                    color=COLORS[v],
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=2,
                )

    ax_delta.set_xlabel(to_title_case_label("Δ average score (vs. brief)"))
    range_frame(ax_delta, np.array([-0.05, 0.05]), x_centers, nice=False)
    ax_delta.set_xticks([-0.05, 0, 0.05])
    # Override y-axis to match environment bar plot
    ax_delta.spines["left"].set_bounds(x_centers[0], x_centers[-1])
    y_pad = 0.22 * 1.8
    ax_delta.set_ylim(x_centers[0] - y_pad, x_centers[-1] + y_pad)
    ax_delta.set_yticks(x_centers)
    ax_delta.set_yticklabels([])

    bar_handles, bar_labels = ax_bar.get_legend_handles_labels()
    ax_bar.legend(
        bar_handles,
        bar_labels,
        title=to_title_case_label("tool verbosity"),
        loc="upper right",
    )


models = sorted(performance_df["model"].dropna().unique())
model_labels = [MODEL_LABELS.get(model, model) for model in models]
model_pivot = build_pivot(performance_df, "model")


def plot_model_summary(ax: plt.Axes) -> None:
    """Draw the model-level verbosity summary averaged over environments.

    Args:
        ax: Axis that receives the plot.

    Returns:
        None: The function mutates the provided axis.
    """
    plot_horizontal_verbosity_bars(
        ax,
        model_pivot,
        models,
        model_labels,
        to_title_case_label("model"),
        show_legend=False,
    )


fig = plt.figure(figsize=(TWO_COL_WIDTH, 3 * ONE_COL_HEIGHT))
outer_grid = fig.add_gridspec(
    2,
    1,
    height_ratios=[TWO_COL_HEIGHT, ONE_COL_HEIGHT],
    hspace=0.38,
)
top_grid = outer_grid[0].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.22)
bottom_grid = outer_grid[1].subgridspec(1, 2, wspace=0.32)

ax_bar = fig.add_subplot(top_grid[0, 0])
ax_delta = fig.add_subplot(top_grid[0, 1])
ax_model_grid = fig.add_subplot(bottom_grid[0, 0])
ax_agent_grid = fig.add_subplot(bottom_grid[0, 1])

plot_environment_summary(ax_bar, ax_delta)
plot_model_summary(ax_model_grid)


agent_types = sorted(performance_df["agent_type"].dropna().unique())
agent_type_labels = [
    AGENT_TYPE_LABELS.get(agent_type, agent_type) for agent_type in agent_types
]
agent_type_pivot = build_pivot(performance_df, "agent_type")


def plot_agent_summary(ax: plt.Axes) -> None:
    """Draw the scaffold-level verbosity summary averaged over environments.

    Args:
        ax: Axis that receives the plot.

    Returns:
        None: The function mutates the provided axis.
    """
    plot_horizontal_verbosity_bars(
        ax,
        agent_type_pivot,
        agent_types,
        agent_type_labels,
        to_title_case_label("agent type"),
        show_legend=False,
    )


plot_agent_summary(ax_agent_grid)

plt.tight_layout()

# Place aligned panel labels using figure coordinates
pos_bar = ax_bar.get_position()
pos_delta = ax_delta.get_position()
pos_model = ax_model_grid.get_position()
pos_agent = ax_agent_grid.get_position()

left_x = min(pos_bar.x0, pos_model.x0)
right_x = min(pos_delta.x0, pos_agent.x0)
LABEL_X_OFFSET = 0.135
RIGHT_LABEL_SHIFT = 0.1

for ax_curr, label, col_x in [
    (ax_bar, "A", left_x),
    (ax_delta, "B", right_x + RIGHT_LABEL_SHIFT),
    (ax_model_grid, "C", left_x),
    (ax_agent_grid, "D", right_x + RIGHT_LABEL_SHIFT),
]:
    trans = blended_transform_factory(fig.transFigure, ax_curr.transAxes)
    ax_curr.text(
        col_x - LABEL_X_OFFSET,
        1.08,
        label,
        transform=trans,
        fontweight="bold",
        fontsize=16,
        color="black",
        ha="center",
        va="center",
        clip_on=False,
    )

fig.savefig(OUT_FILE, bbox_inches="tight")
logger.info(f"Saved figure to {OUT_FILE}")
