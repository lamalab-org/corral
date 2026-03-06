"""Plot performance gaps across environments.

Shows two lines:
1. Model gap: (best model - worst model) averaged across agents
2. Agent gap: (best agent - worst agent) averaged across models

Usage:
    python 3_gap_plot.py --ordering_strategy=average_qa --verbosity_strategy=average \\
                         --task_type_strategy=both --order_direction=descending
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
    GAP_COLORS,
)
from plot_utils import (  # noqa: E402
    DEFAULT_ENV_LEVEL_MAP,
    filter_by_level,
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_qa_data,
    load_reports_data,
    order_environments,
)

# ==================== CONFIGURATION ====================

# (Data paths, filtering utilities, and ordering functions now imported from plot_utils)

# Gap-specific metric display name handling (customized for gap plots)


def get_gap_metric_display_name(metric: str, k_value: int) -> str:
    """Get the display name for the specified metric (gap plot specific).

    Args:
        metric: Metric type - "average_score", "pass_at_k", or "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics

    Returns:
        Display name for the metric with "Gap" suffix
    """
    if metric == "average_score":
        return "Performance Gap"
    elif metric == "pass_at_k":
        return f"Pass@{k_value} Gap"
    elif metric == "pass_hat_k":
        return f"Pass^{k_value} Gap"
    else:
        msg = f"Invalid metric: {metric}"
        raise ValueError(msg)


# ==================== DATA COLLECTION ====================


def collect_gap_data(
    df: pd.DataFrame,
    environments: list[str],
    metric_column: str,
) -> dict:
    """Collect gap data for plotting.

    Args:
        df: Filtered benchmark reports dataframe
        environments: List of environments to plot
        metric_column: Column name of the metric to plot

    Returns:
        Dict with {env: {"model_gap": float, "agent_gap": float}}
    """
    gap_data = {}

    for env in environments:
        env_df = df[df["environment"] == env]
        if env_df.empty:
            continue

        # Compute model gap: (best model - worst model) averaged across agents
        # Group by model, average across agents
        model_scores = env_df.groupby("model")[metric_column].mean()
        if len(model_scores) > 0:
            model_gap = model_scores.max() - model_scores.min()
        else:
            model_gap = np.nan

        # Compute agent gap: (best agent - worst agent) averaged across models
        # Group by agent_type, average across models
        agent_scores = env_df.groupby("agent_type")[metric_column].mean()
        if len(agent_scores) > 0:
            agent_gap = agent_scores.max() - agent_scores.min()
        else:
            agent_gap = np.nan

        if not np.isnan(model_gap) or not np.isnan(agent_gap):
            gap_data[env] = {
                "model_gap": model_gap,
                "agent_gap": agent_gap,
            }

    return gap_data


# ==================== PLOTTING ====================


def plot_gaps(
    gap_data: dict,
    environments: list[str],
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create gap plot.

    Args:
        gap_data: Dict with gap scores {env: {"model_gap": float, "agent_gap": float}}
        environments: Ordered list of environments
        output_path: Path to save the figure
        metric_display_name: Display name for the metric (y-axis label)
    """
    # Filter environments with data
    available_envs = [env for env in environments if env in gap_data]

    if not available_envs:
        logger.warning("No data available for any environment!")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    x_positions = list(range(len(available_envs)))
    x_labels = [ENVIRONMENT_NAMES.get(env, env.upper()) for env in available_envs]

    # Collect data for both gap types
    model_gaps = []
    agent_gaps = []

    for env in available_envs:
        model_gaps.append(gap_data[env]["model_gap"])
        agent_gaps.append(gap_data[env]["agent_gap"])

    # Plot model gap line
    ax.plot(
        x_positions,
        model_gaps,
        color=GAP_COLORS["model_gap"],
        linewidth=2,
        marker="o",
        markersize=6,
        label="Model Gap",
        zorder=3,
    )

    # Plot agent gap line
    ax.plot(
        x_positions,
        agent_gaps,
        color=GAP_COLORS["agent_gap"],
        linewidth=2,
        marker="o",
        markersize=6,
        label="Agent Gap",
        zorder=3,
    )

    # Add vertical dashed lines at each environment
    for x_pos in x_positions:
        ax.axvline(
            x_pos,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
            zorder=1,
        )

    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel(
        metric_display_name, fontsize=FONT_SIZES["axis_label"], fontweight="bold"
    )
    # Set fontsize for tick labels (override lama_aesthetics defaults)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])

    # Legend
    ax.legend(
        loc="best",
        fontsize=FONT_SIZES["legend"],
        framealpha=0.9,
    )

    # Set axis limits with range_frame
    all_gaps = model_gaps + agent_gaps
    all_gaps = [g for g in all_gaps if not np.isnan(g)]

    if x_positions and all_gaps:
        x_range = np.array([min(x_positions) - 0.5, max(x_positions) + 0.5])
        min_gap = min(all_gaps)
        max_gap = max(all_gaps)
        y_range = np.array([max(0, min_gap - 0.05), max_gap + 0.05])

        # Apply range_frame for clean axis appearance
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
    ordering_strategy: str = "average_qa",
    order_direction: str = "descending",
    model_for_ordering: str | None = None,
    qa_type_for_ordering: str = "qa",
    verbosity_strategy: str = "average",
    task_type_strategy: str = "both",
    level_strategy: str = "default_map",
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str = "gap_plot.pdf",
) -> None:
    """Generate gap plot showing model and agent performance gaps across environments.

    Args:
        ordering_strategy: How to order environments by QA score.
            Options: "average_qa" (average across models),
                     "model_specific_qa" (use specific model's QA score)
        order_direction: "ascending" or "descending"
        model_for_ordering: Model name for model_specific_qa
            (e.g., "claude", "gpt", "gpt_oss")
        qa_type_for_ordering: Type of QA to use for ordering.
            Options: "qa", "reasoning_qa"
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
    logger.info("Performance Gap Plot Generation")
    logger.info("=" * 60)
    logger.info(f"Ordering strategy: {ordering_strategy}")
    logger.info(f"Order direction: {order_direction}")
    if model_for_ordering:
        logger.info(f"Model for ordering: {model_for_ordering}")
    logger.info(f"QA type for ordering: {qa_type_for_ordering}")
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
    metric_display_name = get_gap_metric_display_name(metric, k_value)

    # Load data
    logger.info("Loading datasets...")
    reports_df = load_reports_data()
    qa_df = load_qa_data()

    # Order environments by QA score
    logger.info("Ordering environments by QA score...")
    ordered_envs = order_environments(
        qa_df,
        ordering_strategy,
        order_direction,
        model_for_ordering,
        qa_type_for_ordering,
    )
    logger.info("")

    # Filter reports data
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(filtered_df, verbosity_strategy)
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)
    filtered_df = filter_by_level(filtered_df, level_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    if level_strategy == "default_map":
        logger.info(f"Using default level map: {DEFAULT_ENV_LEVEL_MAP}")
    logger.info("")

    # Collect gap data
    logger.info("Computing performance gaps...")
    gap_data = collect_gap_data(
        filtered_df,
        ordered_envs,
        metric_column,
    )

    # Generate plot
    logger.info("Generating plot...")
    output_path = Path(output_filename)
    plot_gaps(gap_data, ordered_envs, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Gap plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
