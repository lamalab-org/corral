"""Compare how agent scaffolds respond to the benchmark verbosity settings.

The figures reuse the visual language of the environment-verbosity analysis but
shift the analytical question: instead of asking whether a verbosity level is
globally better, they highlight whether ReAct and tool-calling scaffolds react
differently to the same verbosity regime overall and within each environment.
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
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_VERBOSITY_GRID = OUT_DIR / "avg_score_by_agent_type_and_verbosity.pdf"
OUT_FILE_ENV_AGENT = (
    OUT_DIR / "avg_score_by_environment_and_agent_type_avg_over_verbosity.pdf"
)

VERBOSITIES = ["brief", "workflow", "comprehensive"]
VERBOSITY_LABELS = {
    "brief": "Brief",
    "workflow": "Workflow",
    "comprehensive": "Comprehensive",
}
AGENT_TYPES = ["react", "tool_calling"]
AGENT_TYPE_LABELS = {
    "react": "ReAct",
    "tool_calling": "Tool Calling",
}
AGENT_TYPE_COLORS = {
    "react": "#4C72B0",
    "tool_calling": "#DD8452",
}
AGENT_TYPE_MARKERS = {"react": "o", "tool_calling": "D"}

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
}


def safe_float(value: object) -> float:
    """Convert a scalar-like value to float without collapsing missingness.

    Args:
        value: Value pulled from a pivot table or aggregation result.

    Returns:
        float: Parsed numeric value, or `np.nan` when the input is missing or
        non-numeric.
    """
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return np.nan
    return float(numeric_value)


def load_performance_df() -> pd.DataFrame:
    """Load the score columns needed for scaffold-versus-verbosity comparisons.

    Returns:
        pd.DataFrame: Filtered benchmark table containing only the grouping keys
        and the numeric score used in the plots.
    """
    records = []
    with DATA_PATH.open() as fh:
        for raw_line in fh:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            records.append(json.loads(stripped_line))

    performance_df = pd.DataFrame(records)
    performance_df = performance_df[
        ["environment", "agent_type", "Tool Verbosity", "Average Score"]
    ].copy()
    performance_df["Average Score"] = pd.to_numeric(
        performance_df["Average Score"], errors="coerce"
    )
    return performance_df.dropna(
        subset=["environment", "agent_type", "Tool Verbosity", "Average Score"]
    )


def build_pivot(
    df: pd.DataFrame,
    *,
    index_col: str,
    column_col: str,
    value_col: str = "Average Score",
    index_order: list[str] | None = None,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate mean scores into a pivot table with stable plotting order.

    Args:
        df: Source table containing the grouping columns and score column.
        index_col: Column to place on the pivot index.
        column_col: Column to place on the pivot columns.
        value_col: Numeric column to aggregate before pivoting.
        index_order: Optional preferred order for index labels.
        column_order: Optional preferred order for column labels.

    Returns:
        pd.DataFrame: Mean-value pivot table ready for direct plotting.
    """
    agg_df = df.groupby([index_col, column_col], as_index=False)[value_col].mean()
    pivot_df = agg_df.pivot_table(index=index_col, columns=column_col, values=value_col)

    if index_order is not None:
        pivot_df = pivot_df.reindex(index_order)
    if column_order is not None:
        for col in column_order:
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan
        pivot_df = pivot_df[column_order]

    return pivot_df


def get_group_centers(
    n_groups: int, *, bar_width: float = 0.22, gap_width: float = 0.1
) -> np.ndarray:
    """Compute group anchors with enough spacing for paired scaffold markers.

    Args:
        n_groups: Number of categorical groups to place on the axis.
        bar_width: Horizontal footprint allocated to one scaffold within a group.
        gap_width: Extra space inserted between neighboring groups.

    Returns:
        np.ndarray: Center coordinate for each categorical group.
    """
    return np.arange(n_groups) * (
        len(AGENT_TYPES) * bar_width + gap_width + bar_width * 0.3
    )


def plot_agent_vs_verbosity_panel(
    ax: Axes,
    pivot_df: pd.DataFrame,
    title: str,
    *,
    show_ylabel: bool = False,
) -> None:
    """Draw one panel of the scaffold-versus-verbosity small-multiples grid.

    Args:
        ax: Axis that receives the panel.
        pivot_df: Pivot table indexed by verbosity with one column per scaffold.
        title: Panel title, typically an environment name.
        show_ylabel: Whether to keep the shared y-axis label on this panel.

    Returns:
        None: The function mutates the provided axis.
    """
    x_centers = np.arange(len(VERBOSITIES), dtype=float)
    offset_width = 0.16

    for idx, agent_type in enumerate(AGENT_TYPES):
        offsets = x_centers + (idx - (len(AGENT_TYPES) - 1) / 2) * offset_width
        scores = [
            safe_float(pivot_df.loc[verbosity, agent_type])
            if verbosity in pivot_df.index and agent_type in pivot_df.columns
            else np.nan
            for verbosity in VERBOSITIES
        ]
        valid_points = [
            (offset, score)
            for offset, score in zip(offsets, scores, strict=True)
            if not pd.isna(score)
        ]
        if not valid_points:
            continue

        valid_offsets = np.array([offset for offset, _ in valid_points], dtype=float)
        valid_scores = np.array([score for _, score in valid_points], dtype=float)
        color = AGENT_TYPE_COLORS[agent_type]

        ax.vlines(
            valid_offsets,
            0,
            valid_scores,
            color=color,
            alpha=0.45,
            linewidth=4,
            zorder=1,
        )
        ax.plot(
            valid_offsets,
            valid_scores,
            color=color,
            linewidth=1.2,
            alpha=0.7,
            zorder=2,
        )
        ax.scatter(
            valid_offsets,
            valid_scores,
            s=42,
            color=color,
            marker=AGENT_TYPE_MARKERS[agent_type],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x_centers)
    ax.set_ylim(0, 1.0)
    if show_ylabel:
        ax.set_ylabel("Average Score")

    range_frame(
        ax,
        np.array([-0.5, len(VERBOSITIES) - 0.5]),
        np.array([0, 1.0]),
        pad=0.15,
    )
    ax.set_xticklabels([VERBOSITY_LABELS[v] for v in VERBOSITIES], rotation=45)


def plot_agent_vs_verbosity_grid(performance_df: pd.DataFrame) -> None:
    """Write the full grid of scaffold-versus-verbosity comparisons.

    The first panel aggregates all environments so local deviations can be read
    against the benchmark-wide trend.

    Args:
        performance_df: Benchmark score table.

    Returns:
        None: The function saves the grid figure to disk.
    """
    environments = sorted(performance_df["environment"].dropna().unique())
    panels = [(None, "All Environments")] + [
        (environment, ENV_LABELS.get(environment, str(environment).capitalize()))
        for environment in environments
    ]

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(TWO_COL_WIDTH, max(TWO_COL_HEIGHT * 1.2, nrows * 2.25)),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (environment, title) in zip(axes, panels, strict=True):
        subset_df = (
            performance_df
            if environment is None
            else performance_df.loc[performance_df["environment"].eq(environment)]
        )
        pivot_df = build_pivot(
            subset_df,
            index_col="Tool Verbosity",
            column_col="agent_type",
            index_order=VERBOSITIES,
            column_order=AGENT_TYPES,
        )
        plot_agent_vs_verbosity_panel(
            ax,
            pivot_df,
            title,
            show_ylabel=ax in axes[::ncols],
        )

    for ax in axes[len(panels) :]:
        ax.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=AGENT_TYPE_COLORS[agent_type],
            marker=AGENT_TYPE_MARKERS[agent_type],
            linewidth=1.4,
            markersize=6,
            label=AGENT_TYPE_LABELS[agent_type],
        )
        for agent_type in AGENT_TYPES
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(AGENT_TYPES),
        title="Scaffold",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_FILE_VERBOSITY_GRID, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {OUT_FILE_VERBOSITY_GRID}")


def plot_horizontal_agent_comparison(
    ax: Axes,
    pivot_df: pd.DataFrame,
    groups: list[str],
    group_labels: list[str],
    *,
    bar_width: float = 0.18,
    gap_width: float = 0.14,
) -> None:
    """Draw environment-level scaffold averages as horizontal lollipop groups.

    Args:
        ax: Axis that receives the plot.
        pivot_df: Pivot table indexed by group with scaffold columns.
        groups: Ordered raw group identifiers to plot.
        group_labels: Display labels corresponding to `groups`.
        bar_width: Vertical separation between scaffold markers in one group.
        gap_width: Extra spacing between neighboring groups.

    Returns:
        None: The function mutates the provided axis.
    """
    y_centers = get_group_centers(len(groups), bar_width=bar_width, gap_width=gap_width)

    for idx, agent_type in enumerate(AGENT_TYPES):
        offsets = y_centers + (idx - (len(AGENT_TYPES) - 1) / 2) * bar_width
        scores = [
            safe_float(pivot_df.loc[group, agent_type])
            if group in pivot_df.index and agent_type in pivot_df.columns
            else np.nan
            for group in groups
        ]
        valid_points = [
            (offset, score)
            for offset, score in zip(offsets, scores, strict=True)
            if not pd.isna(score)
        ]
        if not valid_points:
            continue

        valid_offsets = np.array([offset for offset, _ in valid_points], dtype=float)
        valid_scores = np.array([score for _, score in valid_points], dtype=float)
        color = AGENT_TYPE_COLORS[agent_type]

        ax.hlines(
            valid_offsets,
            0,
            valid_scores,
            color=color,
            alpha=0.5,
            linewidth=5,
            label=AGENT_TYPE_LABELS[agent_type],
        )
        ax.scatter(
            valid_scores,
            valid_offsets,
            s=42,
            color=color,
            marker=AGENT_TYPE_MARKERS[agent_type],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel("Average Score (Mean Over Verbosities)")
    ax.set_ylabel("Environment")
    range_frame(ax, np.array([0, 1]), np.array([y_centers.min(), y_centers.max()]))
    ax.legend(title="Scaffold", loc="upper right", frameon=False)


def plot_environment_agent_summary(performance_df: pd.DataFrame) -> None:
    """Write the environment summary after collapsing over verbosity levels.

    This complementary view isolates scaffold differences that remain once the
    verbosity choice has been averaged out.

    Args:
        performance_df: Benchmark score table.

    Returns:
        None: The function saves the summary figure to disk.
    """
    environments = sorted(performance_df["environment"].dropna().unique())
    env_labels = [
        ENV_LABELS.get(environment, str(environment)) for environment in environments
    ]
    pivot_df = build_pivot(
        performance_df,
        index_col="environment",
        column_col="agent_type",
        index_order=environments,
        column_order=AGENT_TYPES,
    )

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 0.75))
    plot_horizontal_agent_comparison(ax, pivot_df, environments, env_labels)

    plt.tight_layout()
    fig.savefig(OUT_FILE_ENV_AGENT, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {OUT_FILE_ENV_AGENT}")


def main() -> None:
    """Generate both scaffold comparison figures from the combined reports.

    Returns:
        None: The function saves the requested PDFs.
    """
    performance_df = load_performance_df()
    if performance_df.empty:
        raise ValueError(f"No performance rows found in {DATA_PATH}")

    plot_agent_vs_verbosity_grid(performance_df)
    plot_environment_agent_summary(performance_df)


if __name__ == "__main__":
    main()
