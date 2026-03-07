"""Bubble chart showing model x agent performance across environment-level combinations.

Creates a bubble chart where:
- X-axis: Environment-level combinations
- Y-axis: Performance score
- Color: Model
- Shape: Agent type
- Size: Number of trials (or uniform if single trial)

This visualization shows clustering by model (color) rather than agent (shape),
demonstrating that model choice matters more than agent scaffold.

Usage:
    python 10_bubble_chart.py --verbosity_strategy=average --task_type_strategy=both
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


def collect_bubble_data(
    data: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Collect data for bubble chart.

    Args:
        data: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        DataFrame with columns: environment, level, env_level, model, agent_type, score, n_runs
    """
    # Create environment-level column
    data = data.copy()
    data["env_level"] = data["environment"] + "-" + data["level"].astype(str)

    # Group by relevant columns and aggregate
    bubble_df = (
        data.groupby(["environment", "level", "env_level", "model", "agent_type"])
        .agg({metric_column: "mean"})
        .reset_index()
    )
    bubble_df = bubble_df.rename(columns={metric_column: "score"})

    # Count number of runs (rows) per configuration
    bubble_df["n_runs"] = (
        data.groupby(["environment", "level", "env_level", "model", "agent_type"])
        .size()
        .to_numpy()
    )

    return bubble_df


# ==================== PLOTTING ====================


def plot_bubble_chart(
    bubble_df: pd.DataFrame,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create bubble chart.

    Args:
        bubble_df: DataFrame with bubble data
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
    """
    if bubble_df.empty:
        logger.warning("No data available for bubble chart!")
        return

    # Sort environment-level combinations for consistent ordering
    env_levels = sorted(bubble_df["env_level"].unique())
    env_level_positions = {el: i for i, el in enumerate(env_levels)}

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.2))

    # Get unique models and agents
    models = sorted(bubble_df["model"].unique())
    agent_types = sorted(bubble_df["agent_type"].unique())

    # Plot bubbles for each model x agent combination
    rng = np.random.default_rng()
    for model in models:
        for agent_type in agent_types:
            # Filter data for this combination
            subset = bubble_df[
                (bubble_df["model"] == model) & (bubble_df["agent_type"] == agent_type)
            ]

            if subset.empty:
                continue

            # Extract positions and scores
            x_positions = [
                env_level_positions[el] + rng.uniform(-0.15, 0.15)
                for el in subset["env_level"]
            ]
            y_scores = subset["score"].to_numpy()

            # Bubble size based on number of runs (or uniform if all same)
            sizes = subset["n_runs"].to_numpy()
            if len(set(sizes)) == 1:
                # All same, use uniform size
                bubble_sizes = [100] * len(sizes)
            else:
                # Scale sizes proportionally
                bubble_sizes = (sizes / sizes.max()) * 200 + 50

            # Plot
            ax.scatter(
                x_positions,
                y_scores,
                marker=AGENT_MARKERS.get(agent_type, "o"),
                color=MODEL_COLOUR_MAP.get(model, "gray"),
                s=bubble_sizes,
                alpha=0.7,
                edgecolors="white",
                linewidths=1.5,
                label=f"{MODEL_NAMES.get(model, model)} - {AGENT_NAMES.get(agent_type, agent_type)}",
                zorder=3,
            )

    # Format x-axis labels
    x_labels = []
    for el in env_levels:
        parts = el.split("-")
        if len(parts) >= 2:
            env = parts[0]
            level = parts[1]
            env_name = ENVIRONMENT_NAMES.get(env, env.upper())
            x_labels.append(f"{env_name}\nL{level}")
        else:
            x_labels.append(el)

    # Styling
    ax.set_xticks(list(range(len(env_levels))))
    ax.set_xticklabels(x_labels, rotation=0, ha="center")
    ax.set_ylabel(
        metric_display_name,
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_xlabel(
        "Environment - Level",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"] - 1)

    # Legend - organize by model, then agent
    handles, labels = ax.get_legend_handles_labels()
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
    if env_levels:
        x_range = np.array([-0.5, len(env_levels) - 0.5])
        y_min = bubble_df["score"].min() * 0.95
        y_max = bubble_df["score"].max() * 1.05
        y_range = np.array([max(0, y_min), y_max])
        range_frame(ax, x_range, y_range, pad=0.05)

    # Add title
    ax.set_title(
        f"Performance Across {len(env_levels)} Environment-Level Combinations",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
        pad=15,
    )

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
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str = "bubble_chart.pdf",
) -> None:
    """Generate bubble chart showing model x agent performance across environments.

    Args:
        verbosity_strategy: Which tool verbosity to use.
            Options: "average", "brief", "workflow", "comprehensive"
        task_type_strategy: Which task type to use.
            Options: "tasks" (category=task), "subtasks" (category=subtask), "both"
        metric: Metric to plot.
            Options: "average_score", "pass_at_k", "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Bubble Chart Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
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

    # Filter reports data (no level filtering - we want all levels)
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(filtered_df, verbosity_strategy)
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect bubble data
    logger.info("Collecting bubble chart data...")
    bubble_df = collect_bubble_data(filtered_df, metric_column)
    logger.info(f"Prepared {len(bubble_df)} data points")
    logger.info(f"Environment-level combinations: {bubble_df['env_level'].nunique()}")

    # Generate plot
    logger.info("Generating bubble chart...")
    output_path = Path(output_filename)
    plot_bubble_chart(bubble_df, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bubble chart generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
