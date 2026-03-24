"""Scatter plot showing model gap vs agent gap across environments.

Each point represents an environment. Points above the diagonal (y=x) indicate
that model choice contributes more performance variance than agent scaffold choice.

Usage:
    python 5_gap_scatter.py --verbosity_strategy=average --task_type_strategy=both \\
                            --order_direction=descending
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
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

# Color for scatter points
SCATTER_COLOR = GAP_COLORS["model_gap"]


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


def plot_gap_scatter(
    gap_data: dict,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create scatter plot of model gap vs agent gap.

    Args:
        gap_data: Dict with gap scores {env: {"model_gap": float, "agent_gap": float}}
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
    """
    if not gap_data:
        logger.warning("No data available for scatter plot!")
        return

    # Extract data
    environments = list(gap_data.keys())
    agent_gaps = [gap_data[env]["agent_gap"] for env in environments]
    model_gaps = [gap_data[env]["model_gap"] for env in environments]
    env_labels = [ENVIRONMENT_NAMES.get(env, env.upper()) for env in environments]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH / 3, ONE_COL_HEIGHT))

    # Plot diagonal line (y=x) first
    max_gap = max(*agent_gaps, *model_gaps)
    min_gap = min(*agent_gaps, *model_gaps)
    diagonal_range = [min_gap * 0.95, max_gap * 1.05]
    ax.plot(
        diagonal_range,
        diagonal_range,
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Equal Gap",
        zorder=1,
    )

    # Plot scatter points
    ax.scatter(
        agent_gaps,
        model_gaps,
        s=100,
        color=SCATTER_COLOR,
        alpha=0.7,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    # Add environment labels to points with adjustText to avoid overlap
    texts = []
    for _i, (x, y, label) in enumerate(
        zip(agent_gaps, model_gaps, env_labels, strict=False)
    ):
        texts.append(
            ax.text(
                x,
                y,
                label,
                fontsize=5,
                color="black",
                alpha=1,
            )
        )

    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "alpha": 1},
        expand_points=(1.5, 1.5),
        force_text=(0.5, 0.5),
    )

    # Styling
    ax.set_xlabel(
        f"Agent Gap ({metric_display_name})",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        f"Model Gap ({metric_display_name})",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Legend
    # ax.legend(
    #     loc="upper left",
    #     fontsize=FONT_SIZES["legend"],
    #     framealpha=0.9,
    # )

    # Add shaded region above diagonal to highlight model dominance
    ax.fill_between(
        diagonal_range,
        diagonal_range,
        [max_gap * 1.1] * 2,
        alpha=0.1,
        color=SCATTER_COLOR,
        zorder=0,
        label="_nolegend_",
    )

    # Set equal aspect ratio for fair comparison
    ax.set_aspect("equal", adjustable="box")

    # Set axis limits with range_frame
    x_range = np.array([min_gap * 0.95, max_gap * 1.05])
    y_range = np.array([min_gap * 0.95, max_gap * 1.05])
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
    output_filename: str | None = None,
) -> None:
    """Generate scatter plot showing model gap vs agent gap.

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
    logger.info("Gap Scatter Plot Generation")
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
    filtered_df = filter_by_verbosity(
        filtered_df, None if verbosity_strategy == "average" else verbosity_strategy
    )
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)
    filtered_df = filter_by_level(filtered_df, level_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect gap data
    logger.info("Computing performance gaps...")
    gap_data = collect_gap_data(filtered_df, metric_column)
    logger.info(f"Found data for {len(gap_data)} environments")

    # Generate plot
    logger.info("Generating scatter plot...")
    if output_filename is None:
        output_path = Path(__file__).parent / "2b_gap_scatter.pdf"
    else:
        output_path = Path(output_filename)
    plot_gap_scatter(gap_data, output_path, metric_display_name)

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
    above_diagonal = sum(
        1 for mg, ag in zip(model_gaps, agent_gaps, strict=False) if mg > ag
    )
    logger.info(
        f"Environments where model gap > agent gap: {above_diagonal}/{len(gap_data)}"
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Scatter plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
