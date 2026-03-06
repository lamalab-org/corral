"""Plot slope chart showing model performance across environments.

Each model is represented by a colored line connecting scores across environments.
Environments are ordered by QA scores on the x-axis.

Usage:
    python 2_slope.py --ordering_strategy=average_qa --verbosity_strategy=average \\
                      --task_type_strategy=both --agent_type_strategy=average \\
                      --order_direction=descending
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
    DEFAULT_ENV_LEVEL_MAP,
    filter_by_agent_type,
    filter_by_level,
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    get_metric_display_name,
    load_qa_data,
    load_reports_data,
    order_environments,
)

# ==================== CONFIGURATION ====================

# (Data paths, filtering utilities, and ordering functions now imported from plot_utils)


# ==================== DATA COLLECTION ====================


def collect_plot_data(
    df: pd.DataFrame,
    environments: list[str],
    models: list[str],
    metric_column: str,
) -> dict:
    """Collect aggregated data for plotting.

    Args:
        df: Filtered benchmark reports dataframe
        environments: List of environments to plot
        models: List of models to plot
        metric_column: Column name of the metric to plot

    Returns:
        Nested dict: {model: {env: score}}
    """
    plot_data = {}

    for model in models:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        plot_data[model] = {}

        for env in environments:
            env_df = model_df[model_df["environment"] == env]
            if env_df.empty:
                continue

            # Compute mean score across all filtered rows (agent types, verbosities, etc.)
            score = env_df[metric_column].mean()
            if not np.isnan(score):
                plot_data[model][env] = score

    return plot_data


# ==================== PLOTTING ====================


def plot_slope(
    plot_data: dict,
    environments: list[str],
    models: list[str],
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create slope plot.

    Args:
        plot_data: Nested dict with scores {model: {env: score}}
        environments: Ordered list of environments
        models: List of models to plot
        output_path: Path to save the figure
        metric_display_name: Display name for the metric (y-axis label)
    """
    # Filter environments with data
    available_envs = [
        env for env in environments if any(env in plot_data.get(m, {}) for m in models)
    ]

    if not available_envs:
        logger.warning("No data available for any environment!")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    x_positions = list(range(len(available_envs)))
    x_labels = [ENVIRONMENT_NAMES.get(env, env.upper()) for env in available_envs]

    # Plot lines for each model
    for model_name in models:
        if model_name not in plot_data:
            continue

        model_scores = plot_data[model_name]

        # Collect x and y for this model
        x_vals = []
        y_vals = []
        for x_idx, env in enumerate(available_envs):
            if env in model_scores:
                x_vals.append(x_idx)
                y_vals.append(model_scores[env])

        if not x_vals:
            continue

        # Plot line
        color = MODEL_COLOUR_MAP.get(model_name, "#333333")
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=MODEL_NAMES.get(model_name, model_name),
            zorder=3,
        )

    # Add vertical dashed lines at each environment
    for x_pos in x_positions:
        ax.axvline(
            x_pos,
            color="gray",
            linestyle="--",
            linewidth=1,
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
    all_scores = []
    for model_data in plot_data.values():
        all_scores.extend(model_data.values())

    if x_positions and all_scores:
        x_range = np.array([min(x_positions) - 0.5, max(x_positions) + 0.5])
        min_score = min(all_scores)
        max_score = max(all_scores)
        y_range = np.array([max(0, min_score - 0.05), min(1, max_score + 0.05)])

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
    agent_type_strategy: str = "average",
    metric: str = "average_score",
    k_value: int = 5,
    models: str = "claude-4.5,gpt-4o",
    output_filename: str = "slope_plot.pdf",
) -> None:
    """Generate slope plot showing model performance across environments.

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
        agent_type_strategy: Which agent types to include.
            Options: "average" (average both), "react", "tool_calling"
        metric: Metric to plot.
            Options: "average_score", "pass_at_k", "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        models: Comma-separated list of models to plot
            (e.g., "claude-4.5,gpt-4o,gpt-oss-120b")
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Slope Plot Generation")
    logger.info("=" * 60)
    logger.info(f"Ordering strategy: {ordering_strategy}")
    logger.info(f"Order direction: {order_direction}")
    if model_for_ordering:
        logger.info(f"Model for ordering: {model_for_ordering}")
    logger.info(f"QA type for ordering: {qa_type_for_ordering}")
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Level strategy: {level_strategy}")
    logger.info(f"Agent type strategy: {agent_type_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info(f"Models to plot: {models}")
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
    metric_display_name = get_metric_display_name(metric, k_value)

    # Parse models list
    model_list = [m.strip() for m in models.split(",")]

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
    filtered_df = filter_by_agent_type(filtered_df, agent_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    if level_strategy == "default_map":
        logger.info(f"Using default level map: {DEFAULT_ENV_LEVEL_MAP}")
    logger.info("")

    # Collect plot data
    logger.info("Collecting plot data...")
    plot_data = collect_plot_data(
        filtered_df,
        ordered_envs,
        model_list,
        metric_column,
    )

    # Generate plot
    logger.info("Generating plot...")
    output_path = Path(output_filename)
    plot_slope(plot_data, ordered_envs, model_list, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Slope plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
