"""Clustered scatter plot showing all model x agent combinations across environments.

Each point represents a (model, agent, environment) combination. Points are colored
by model and shaped by agent type. This visualization shows that model clusters are
more separated than agent variations within each model, demonstrating that model
choice contributes more to performance variance than agent scaffold.

Usage:
    python 7_clustered_scatter.py --verbosity_strategy=average --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import (  # noqa: E402
    ENVIRONMENT_NAMES,
    FONT_SIZES,
    MODEL_COLOUR_MAP,
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

# Agent type markers
AGENT_MARKERS = {
    "react": "o",  # Circle
    "tool_calling": "s",  # Square
}

AGENT_NAMES = {
    "react": "ReAct",
    "tool_calling": "Tool-Calling",
}


# ==================== DATA COLLECTION ====================


def collect_scatter_data(
    df: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Collect data for clustered scatter plot.

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        DataFrame with columns: environment, model, agent_type, score
    """
    # Select relevant columns and compute mean across trials
    return (
        df.groupby(["environment", "model", "agent_type"])[metric_column]
        .mean()
        .reset_index()
        .rename(columns={metric_column: "score"})
    )


# ==================== PLOTTING ====================


def plot_clustered_scatter(
    scatter_df: pd.DataFrame,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create clustered scatter plot.

    Args:
        scatter_df: DataFrame with environment, model, agent_type, score
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
    """
    if scatter_df.empty:
        logger.warning("No data available for scatter plot!")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    # Get unique values
    environments = sorted(scatter_df["environment"].unique())
    models = sorted(scatter_df["model"].unique())
    agent_types = sorted(scatter_df["agent_type"].unique())

    # Create environment positions with some spacing for jitter
    env_positions = {env: i for i, env in enumerate(environments)}

    # Plot points for each model x agent combination
    rng = np.random.default_rng()
    for model in models:
        for agent_type in agent_types:
            # Filter data for this combination
            subset = scatter_df[
                (scatter_df["model"] == model)
                & (scatter_df["agent_type"] == agent_type)
            ]

            if subset.empty:
                continue

            # Extract x (environment) and y (score)
            x_positions = [
                env_positions[env] + rng.uniform(-0.15, 0.15)
                for env in subset["environment"]
            ]
            y_scores = subset["score"].to_numpy()

            # Plot
            ax.scatter(
                x_positions,
                y_scores,
                marker=AGENT_MARKERS.get(agent_type, "o"),
                color=MODEL_COLOUR_MAP.get(model, "gray"),
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=1,
                label=f"{MODEL_NAMES.get(model, model)} - {AGENT_NAMES.get(agent_type, agent_type)}",
                zorder=3,
            )

    # Styling
    ax.set_xticks(list(range(len(environments))))
    ax.set_xticklabels(
        [ENVIRONMENT_NAMES.get(env, env.upper()) for env in environments],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel(
        metric_display_name,
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Legend - organize by model, then agent
    handles, labels = ax.get_legend_handles_labels()
    # Sort to group by model
    ax.legend(
        handles,
        labels,
        loc="best",
        fontsize=FONT_SIZES["legend"] - 1,
        framealpha=0.9,
        ncol=2,
    )

    # Grid for readability
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)

    # Set axis limits with range_frame
    if environments:
        x_range = np.array([-0.5, len(environments) - 0.5])
        y_min = scatter_df["score"].min() * 0.95
        y_max = scatter_df["score"].max() * 1.05
        y_range = np.array([max(0, y_min), y_max])
        range_frame(ax, x_range, y_range, pad=0.05)

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
    output_filename: str = "clustered_scatter.pdf",
) -> None:
    """Generate clustered scatter plot showing model x agent combinations.

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
    logger.info("Clustered Scatter Plot Generation")
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

    # Collect scatter data
    logger.info("Collecting scatter data...")
    scatter_df = collect_scatter_data(filtered_df, metric_column)
    logger.info(f"Prepared {len(scatter_df)} data points")

    # Generate plot
    logger.info("Generating clustered scatter plot...")
    output_path = Path(output_filename)
    plot_clustered_scatter(scatter_df, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Clustered scatter plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
