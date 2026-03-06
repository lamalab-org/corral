"""Plot dumbbell chart comparing React vs Tool-Calling agents across environments.

Each environment shows two dumbbells (one per model) with React on one end
and Tool-Calling on the other. Environments are ordered by QA scores.

Usage:
    python 1_slope.py --ordering_strategy=average_qa --verbosity_strategy=average \\
                      --task_type_strategy=both --agent_type_strategy=average \\
                      --order_direction=descending
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.lines as mlines
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

# Offsets for plotting
MODEL_OFFSET = 0.1  # Horizontal separation between models
AGENT_OFFSET = 0.2  # Horizontal separation between react/tool_calling


# ==================== DATA COLLECTION ====================


def collect_plot_data(
    df: pd.DataFrame,
    environments: list[str],
    models: list[str],
    agent_types: list[str],
    metric_column: str,
) -> dict:
    """Collect aggregated data for plotting.

    Args:
        df: Filtered benchmark reports dataframe
        environments: List of environments to plot
        models: List of models to plot
        agent_types: List of agent types to plot
        metric_column: Column name of the metric to plot

    Returns:
        Nested dict: {env: {model: {agent: score}}}
    """
    plot_data = {}

    for env in environments:
        env_df = df[df["environment"] == env]
        if env_df.empty:
            continue

        plot_data[env] = {}

        for model in models:
            model_df = env_df[env_df["model"] == model]
            if model_df.empty:
                continue

            plot_data[env][model] = {}

            for agent in agent_types:
                agent_df = model_df[model_df["agent_type"] == agent]
                if agent_df.empty:
                    continue

                # Compute mean score across all filtered rows
                score = agent_df[metric_column].mean()
                if not np.isnan(score):
                    plot_data[env][model][agent] = score

    return plot_data


# ==================== PLOTTING ====================


def plot_dumbbell(
    plot_data: dict,
    environments: list[str],
    models: list[str],
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create dumbbell plot.

    Args:
        plot_data: Nested dict with scores
        environments: Ordered list of environments
        models: List of models to plot
        output_path: Path to save the figure
        metric_display_name: Display name for the metric (y-axis label)
    """
    # Filter environments with data
    available_envs = [env for env in environments if plot_data.get(env)]

    if not available_envs:
        logger.warning("No data available for any environment!")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    x_positions = []
    x_labels = []

    for x_idx, env in enumerate(available_envs):
        env_data = plot_data[env]

        x_positions.append(x_idx)
        x_labels.append(ENVIRONMENT_NAMES.get(env, env.upper()))

        # Plot for each model
        for model_idx, model_name in enumerate(models):
            if model_name not in env_data:
                continue

            agent_scores = env_data[model_name]

            # Need both react and tool_calling
            if "react" not in agent_scores or "tool_calling" not in agent_scores:
                continue

            react_score = agent_scores["react"]
            tool_calling_score = agent_scores["tool_calling"]

            # Determine positions
            model_off = MODEL_OFFSET if model_idx == 1 else -MODEL_OFFSET
            if len(models) == 3:
                model_off = MODEL_OFFSET * (model_idx - 1)

            react_x = x_idx + model_off - AGENT_OFFSET
            tool_x = x_idx + model_off + AGENT_OFFSET

            # Color
            color = MODEL_COLOUR_MAP.get(model_name, "#333333")

            # Plot sloped dumbbell
            ax.plot(
                [react_x, tool_x],
                [react_score, tool_calling_score],
                linestyle="-",
                color=color,
                linewidth=2.5,
                alpha=0.7,
                zorder=2,
            )

            # Plot endpoints: filled for react, unfilled for tool_calling
            ax.scatter(
                react_x,
                react_score,
                s=80,
                color=color,
                marker="o",
                edgecolors=color,
                linewidths=0.5,
                zorder=3,
            )
            ax.scatter(
                tool_x,
                tool_calling_score,
                s=80,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                zorder=3,
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

    # Add vertical separators
    for i in range(len(x_positions) - 1):
        ax.axvline(
            x_positions[i] + 0.5,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.3,
        )

    # Legend
    handles = []
    for model_id, model_display in MODEL_NAMES.items():
        if model_id in models:
            handles.append(
                mlines.Line2D(
                    [0],
                    [0],
                    color=MODEL_COLOUR_MAP[model_id],
                    lw=2.5,
                    linestyle="-",
                    label=model_display,
                )
            )

    handles.extend(
        [
            mlines.Line2D(
                [0],
                [0],
                color="none",
                marker="o",
                markersize=8,
                markerfacecolor="gray",
                markeredgecolor="gray",
                linestyle="None",
                label="React Agent",
            ),
            mlines.Line2D(
                [0],
                [0],
                color="none",
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="gray",
                linestyle="None",
                label="Tool-Calling Agent",
            ),
        ]
    )

    ax.legend(
        handles=handles,
        bbox_to_anchor=(0.95, 0.98),
        fontsize=FONT_SIZES["legend"],
        framealpha=0.9,
    )

    # Set axis limits with range_frame
    all_scores = []
    for env_data in plot_data.values():
        for model_data in env_data.values():
            all_scores.extend(model_data.values())

    if x_positions and all_scores:
        x_range = np.array([min(x_positions) - 0.5, max(x_positions) + 0.5])
        min_score = min(all_scores)
        max_score = max(all_scores)
        y_range = np.array([max(0, min_score), min(1, max_score)])

        # Apply range_frame for clean axis appearance
        range_frame(ax, x_range, y_range, pad=0.05)

    # ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

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
    output_filename: str = "dumbbell_plot.pdf",
) -> None:
    """Generate dumbbell plot comparing React vs Tool-Calling agents.

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
        agent_type_strategy: Which agent types to compare.
            Options: "average" (shows both), "react", "tool_calling"
        metric: Metric to plot.
            Options: "average_score", "pass_at_k", "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        models: Comma-separated list of models to plot
            (e.g., "claude-4.5,gpt-4o,gpt-oss-80b")
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Dumbbell Plot Generation")
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

    # For dumbbell plot, we always show both agent types
    agent_types_to_plot = ["react", "tool_calling"]

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
        agent_types_to_plot,
        metric_column,
    )

    # Generate plot
    logger.info("Generating plot...")
    output_path = Path(output_filename)
    plot_dumbbell(plot_data, ordered_envs, model_list, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dumbbell plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
