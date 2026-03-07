"""Performance matrix heatmaps showing model x environment performance by agent type.

Creates two side-by-side heatmaps (ReAct and Tool-Calling), each showing a 3x7 matrix
where rows are models and columns are environments. The similarity between the two
heatmaps shows minimal agent effect, while the strong row-wise color variation within
each heatmap shows large model effect.

Usage:
    python 8_performance_heatmap.py --verbosity_strategy=average --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger

lama_aesthetics.get_style("main")

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import (  # noqa: E402
    ENVIRONMENT_NAMES,
    FONT_SIZES,
    MODEL_NAMES,
)
from plot_utils import (  # noqa: E402
    filter_by_level,
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

# ==================== CONFIGURATION ====================

AGENT_NAMES = {
    "react": "ReAct",
    "tool_calling": "Tool-Calling",
}


# ==================== DATA COLLECTION ====================


def collect_heatmap_data(
    df: pd.DataFrame,
    metric_column: str,
) -> dict:
    """Collect data for two heatmaps (one per agent type).

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        Dict with {agent_type: DataFrame with models as rows, environments as columns}
    """
    heatmap_data = {}
    agent_types = sorted(df["agent_type"].unique())

    for agent_type in agent_types:
        agent_df = df[df["agent_type"] == agent_type]

        # Pivot to create model x environment matrix
        pivot_df = agent_df.pivot_table(
            values=metric_column,
            index="model",
            columns="environment",
            aggfunc="mean",
        )

        if not pivot_df.empty:
            heatmap_data[agent_type] = pivot_df

    return heatmap_data


# ==================== PLOTTING ====================


def plot_dual_heatmaps(
    heatmap_data: dict,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create two side-by-side heatmaps for ReAct and Tool-Calling.

    Args:
        heatmap_data: Dict with {agent_type: DataFrame (models x environments)}
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
    """
    if not heatmap_data or len(heatmap_data) < 2:
        logger.warning("Need data for both agent types!")
        return

    # Get data for both agent types
    agent_types = sorted(heatmap_data.keys())
    if "react" not in agent_types or "tool_calling" not in agent_types:
        logger.warning("Missing react or tool_calling data!")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.2),
        sharey=True,
    )

    # Get global min/max for consistent color scale
    all_values = []
    for df in heatmap_data.values():
        all_values.extend(df.to_numpy().flatten())
    all_values = [v for v in all_values if not np.isnan(v)]

    if all_values:
        vmin = min(all_values)
        vmax = max(all_values)
    else:
        vmin, vmax = 0, 1

    # Plot ReAct heatmap (left)
    react_df = heatmap_data["react"].copy()
    react_df.index = [MODEL_NAMES.get(m, m) for m in react_df.index]
    react_df.columns = [ENVIRONMENT_NAMES.get(e, e.upper()) for e in react_df.columns]

    sns.heatmap(
        react_df,
        annot=True,
        fmt=".2f",
        cmap="Purples",  # "RdYlGn",
        vmin=vmin,
        vmax=vmax,
        cbar=False,  # No colorbar on left
        ax=ax1,
        linewidths=1,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"] - 2},
    )

    # Plot Tool-Calling heatmap (right)
    tool_df = heatmap_data["tool_calling"].copy()
    tool_df.index = [MODEL_NAMES.get(m, m) for m in tool_df.index]
    tool_df.columns = [ENVIRONMENT_NAMES.get(e, e.upper()) for e in tool_df.columns]

    sns.heatmap(
        tool_df,
        annot=True,
        fmt=".2f",
        cmap="Purples",  # RdYlGn",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": metric_display_name, "shrink": 0.8},
        ax=ax2,
        linewidths=1,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"] - 2},
    )

    # Styling for left heatmap (ReAct)
    ax1.set_title(
        AGENT_NAMES["react"],
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("Model", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
    ax1.set_xlabel("")
    ax1.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"] - 1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Styling for right heatmap (Tool-Calling)
    ax2.set_title(
        AGENT_NAMES["tool_calling"],
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("")  # Shared y-axis, no label needed
    ax2.set_xlabel("")
    ax2.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"] - 1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(
        output_path.with_suffix(".png"),
        bbox_inches="tight",
        dpi=300,
    )
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


# ==================== MAIN ====================


def main(
    verbosity_strategy: str = "average",
    task_type_strategy: str = "both",
    level_strategy: str = "default_map",
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str = "performance_heatmap.pdf",
) -> None:
    """Generate dual heatmaps showing model x environment matrices per agent type.

    Args:
        verbosity_strategy: Which tool verbosity to use.
            Options: "average", "brief", "workflow", "comprehensive"
        task_type_strategy: Which task type to use.
            Options: "tasks" (category=task), "subtasks" (category=subtask), "both"
        level_strategy: Which level to use for environments.
            Options: "all" (average across all levels),
                     "default_map" (use per-environment mapping),
                     or specific level like "1", "2", "3", "4"
        metric: Metric to plot.
            Options: "average_score", "pass_at_k", "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Dual Performance Heatmap Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Level strategy: {level_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info("")

    # Validate inputs
    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        msg = f"Invalid metric: {metric}. Must be 'average_score', 'pass_at_k', or 'pass_hat_k'"
        raise ValueError(msg)

    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        msg = f"Invalid k_value: {k_value}. Must be between 1 and 5"
        raise ValueError(msg)

    # Get metric column name and display name
    metric_column = get_metric_column_name(metric, k_value)
    if metric == "average_score":
        metric_display_name = "Average Score"
    elif metric == "pass_at_k":
        metric_display_name = f"Pass@{k_value}"
    elif metric == "pass_hat_k":
        metric_display_name = f"Pass^{k_value}"

    # Load data
    logger.info("Loading datasets...")
    reports_df = load_reports_data()

    # Filter reports data
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(filtered_df, verbosity_strategy)
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)
    filtered_df = filter_by_level(filtered_df, level_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect heatmap data
    logger.info("Preparing heatmap data...")
    heatmap_data = collect_heatmap_data(filtered_df, metric_column)
    logger.info(f"Prepared data for {len(heatmap_data)} agent types")
    for agent_type, df in heatmap_data.items():
        logger.info(
            f"  {agent_type}: {df.shape[0]} models x {df.shape[1]} environments"
        )

    # Generate plot
    logger.info("Generating dual performance heatmaps...")
    output_path = Path(output_filename)
    plot_dual_heatmaps(heatmap_data, output_path, metric_display_name)

    # Print summary statistics
    logger.info("")
    logger.info("Heatmap Comparison:")
    logger.info("-" * 40)

    if "react" in heatmap_data and "tool_calling" in heatmap_data:
        react_df = heatmap_data["react"]
        tool_df = heatmap_data["tool_calling"]

        # Calculate difference between agent types (environment-wise)
        diff_df = (react_df - tool_df).abs()
        mean_diff = diff_df.mean().mean()

        logger.info(
            f"Mean absolute difference between ReAct and Tool-Calling: {mean_diff:.4f}"
        )
        logger.info(f"Max difference: {diff_df.max().max():.4f}")
        logger.info("")

        # Calculate row variance (model effect) within each agent type
        react_row_var = react_df.var(axis=1).mean()
        tool_row_var = tool_df.var(axis=1).mean()
        avg_row_var = (react_row_var + tool_row_var) / 2

        logger.info("Within-agent variation across environments (per model):")
        logger.info(f"  ReAct: {react_row_var:.6f}")
        logger.info(f"  Tool-Calling: {tool_row_var:.6f}")
        logger.info(f"  Average: {avg_row_var:.6f}")
        logger.info("")

        # Calculate column variance (environment effect) within each agent type
        react_col_var = react_df.var(axis=0).mean()
        tool_col_var = tool_df.var(axis=0).mean()
        avg_col_var = (react_col_var + tool_col_var) / 2

        logger.info("Within-agent variation across models (per environment):")
        logger.info(f"  ReAct: {react_col_var:.6f}")
        logger.info(f"  Tool-Calling: {tool_col_var:.6f}")
        logger.info(f"  Average: {avg_col_var:.6f}")
        logger.info("")

        logger.info(
            f"Model variance / Environment variance ratio: {avg_col_var / avg_row_var:.2f}x"
        )
        logger.info(
            f"Agent difference / Model variance ratio: {mean_diff / avg_col_var:.2f}x"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dual performance heatmaps generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
