"""Grouped bar chart showing model gap vs agent gap across environments.

Side-by-side bars for each environment showing model gap and agent gap.
Environments are sorted by model gap to emphasize that model choice consistently
contributes more variance than agent scaffold choice.

Usage:
    python 6_gap_bars.py --verbosity_strategy=average --task_type_strategy=both \\
                         --sort_by=model_gap
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
    filter_by_level,
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

# ==================== CONFIGURATION ====================

# Bar width for grouped bars
BAR_WIDTH = 0.35


# ==================== DATA COLLECTION ====================


def collect_gap_data(
    df: pd.DataFrame,
    metric_column: str,
) -> dict:
    """Collect gap data for all environments.

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        Dict with {env: {"model_gap": float, "agent_gap": float}}
    """
    gap_data = {}
    environments = df["environment"].unique()

    for env in environments:
        env_df = df[df["environment"] == env]
        if env_df.empty:
            continue

        # Compute model gap: (best model - worst model) averaged across agents
        model_scores = env_df.groupby("model")[metric_column].mean()
        if len(model_scores) > 1:
            model_gap = model_scores.max() - model_scores.min()
        else:
            model_gap = np.nan

        # Compute agent gap: (best agent - worst agent) averaged across models
        agent_scores = env_df.groupby("agent_type")[metric_column].mean()
        if len(agent_scores) > 1:
            agent_gap = agent_scores.max() - agent_scores.min()
        else:
            agent_gap = np.nan

        if not np.isnan(model_gap) and not np.isnan(agent_gap):
            gap_data[env] = {
                "model_gap": model_gap,
                "agent_gap": agent_gap,
            }

    return gap_data


# ==================== PLOTTING ====================


def plot_gap_bars(
    gap_data: dict,
    output_path: Path,
    metric_display_name: str,
    sort_by: str = "model_gap",
) -> None:
    """Create grouped bar chart of model gap vs agent gap.

    Args:
        gap_data: Dict with gap scores {env: {"model_gap": float, "agent_gap": float}}
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
        sort_by: How to sort environments ("model_gap", "agent_gap", or "none")
    """
    if not gap_data:
        logger.warning("No data available for bar chart!")
        return

    # Sort environments
    environments = list(gap_data.keys())
    if sort_by == "model_gap":
        environments = sorted(
            environments, key=lambda e: gap_data[e]["model_gap"], reverse=True
        )
    elif sort_by == "agent_gap":
        environments = sorted(
            environments, key=lambda e: gap_data[e]["agent_gap"], reverse=True
        )
    # else: keep original order

    # Extract data
    model_gaps = [gap_data[env]["model_gap"] for env in environments]
    agent_gaps = [gap_data[env]["agent_gap"] for env in environments]
    env_labels = [ENVIRONMENT_NAMES.get(env, env.upper()) for env in environments]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    # Set up bar positions
    x_positions = np.arange(len(environments))

    # Plot bars
    ax.bar(
        x_positions - BAR_WIDTH / 2,
        model_gaps,
        BAR_WIDTH,
        label="Model Gap",
        color=GAP_COLORS["model_gap"],
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )

    ax.bar(
        x_positions + BAR_WIDTH / 2,
        agent_gaps,
        BAR_WIDTH,
        label="Agent Gap",
        color=GAP_COLORS["agent_gap"],
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )

    # Add value labels on bars (optional, can be disabled if cluttered)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label non-negligible bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["tick_label"] - 2,
                    alpha=0.7,
                )

    # Uncomment to add value labels:
    # add_value_labels(bars1)
    # add_value_labels(bars2)

    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(env_labels, rotation=45, ha="right")
    ax.set_ylabel(
        f"{metric_display_name} Gap",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=FONT_SIZES["legend"],
        framealpha=0.9,
    )

    # Grid for readability
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)

    # Set axis limits with range_frame
    all_gaps = model_gaps + agent_gaps
    if x_positions.size > 0 and all_gaps:
        x_range = np.array([x_positions[0] - 0.6, x_positions[-1] + 0.6])
        y_max = max(all_gaps) * 1.1
        y_range = np.array([0, y_max])
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
    sort_by: str = "model_gap",
    output_filename: str = "gap_bars.pdf",
) -> None:
    """Generate grouped bar chart showing model gap vs agent gap.

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
        sort_by: How to sort environments.
            Options: "model_gap" (sort by model gap, descending),
                     "agent_gap" (sort by agent gap, descending),
                     "none" (keep original order)
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Gap Bar Chart Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Level strategy: {level_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info(f"Sort by: {sort_by}")
    logger.info("")

    # Validate inputs
    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        msg = f"Invalid metric: {metric}. Must be 'average_score', 'pass_at_k', or 'pass_hat_k'"
        raise ValueError(msg)

    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        msg = f"Invalid k_value: {k_value}. Must be between 1 and 5"
        raise ValueError(msg)

    if sort_by not in ["model_gap", "agent_gap", "none"]:
        msg = f"Invalid sort_by: {sort_by}. Must be 'model_gap', 'agent_gap', or 'none'"
        raise ValueError(msg)

    # Get metric column name and display name
    metric_column = get_metric_column_name(metric, k_value)
    if metric == "average_score":
        metric_display_name = "Score"
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

    # Collect gap data
    logger.info("Computing performance gaps...")
    gap_data = collect_gap_data(filtered_df, metric_column)
    logger.info(f"Found data for {len(gap_data)} environments")

    # Generate plot
    logger.info("Generating bar chart...")
    output_path = Path(output_filename)
    plot_gap_bars(gap_data, output_path, metric_display_name, sort_by)

    # Print summary statistics
    logger.info("")
    logger.info("Summary Statistics:")
    logger.info("-" * 40)
    agent_gaps = [gap_data[env]["agent_gap"] for env in gap_data]
    model_gaps = [gap_data[env]["model_gap"] for env in gap_data]

    logger.info(
        f"Agent gap - Mean: {np.mean(agent_gaps):.3f}, Std: {np.std(agent_gaps):.3f}"
    )
    logger.info(
        f"Model gap - Mean: {np.mean(model_gaps):.3f}, Std: {np.std(model_gaps):.3f}"
    )
    logger.info(
        f"Model/Agent ratio - Mean: {np.mean(np.array(model_gaps) / np.array(agent_gaps)):.2f}x"
    )
    larger_model = sum(
        1 for mg, ag in zip(model_gaps, agent_gaps, strict=False) if mg > ag
    )
    logger.info(
        f"Environments where model gap > agent gap: {larger_model}/{len(gap_data)}"
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bar chart generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
