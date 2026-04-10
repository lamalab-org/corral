"""Scatter plot showing model gap vs agent gap across environments.

Each point represents an environment. Points above the diagonal (y=x) indicate
that model choice contributes more performance variance than agent scaffold choice.

Usage:
    python plot_panel2_gap_scatter.py --verbosity_strategy=average --task_type_strategy=both
"""

from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import ENVIRONMENT_GROUPS, FONT_SIZES, GAP_COLORS, GROUP_COLOURS
from plot_utils import (
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color for scatter points
SCATTER_COLOR = GAP_COLORS["model_gap"]


# ==================== DATA COLLECTION ====================


def collect_gap_data(
    df: pd.DataFrame,
    metric_column: str,
) -> dict:
    """Collect gap data for all environments."""
    gap_data = {}
    environments = df["environment"].unique()

    for env in environments:
        env_df = df[df["environment"] == env]
        if env_df.empty:
            continue

        model_scores = env_df.groupby("model")[metric_column].mean()
        if len(model_scores) > 1:
            model_gap = model_scores.max() - model_scores.min()
        else:
            model_gap = np.nan

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
) -> None:
    """Create scatter plot of model gap vs agent gap."""
    if not gap_data:
        logger.warning("No data available for scatter plot!")
        return

    env_to_group = {}
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        for env in group_info["environments"]:
            env_to_group[env] = group_name

    environments = list(gap_data.keys())
    agent_gaps = [gap_data[env]["agent_gap"] for env in environments]
    model_gaps = [gap_data[env]["model_gap"] for env in environments]

    fig, ax = plt.subplots(1, 1, figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    max_gap = max(*agent_gaps, *model_gaps)
    min_gap = min(*agent_gaps, *model_gaps)
    diagonal_range = [min_gap * 0.95, max_gap * 1.05]
    ax.plot(
        diagonal_range,
        diagonal_range,
        "k--",
        linewidth=1.5,
        alpha=0.5,
        zorder=1,
    )

    for group_name in ENVIRONMENT_GROUPS:
        group_envs = [e for e in environments if env_to_group.get(e) == group_name]
        if not group_envs:
            continue
        gx = [gap_data[e]["agent_gap"] for e in group_envs]
        gy = [gap_data[e]["model_gap"] for e in group_envs]
        ax.scatter(
            gx,
            gy,
            s=100,
            color=GROUP_COLOURS[group_name],
            alpha=0.7,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
            label=group_name,
        )

    ax.set_xlabel(
        "Scaffold Spread",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        "Model Spread",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(0.45, 0.2),
        fontsize=FONT_SIZES["legend"],
        framealpha=0.0,
        ncol=1,
    )
    for text, handle in zip(leg.get_texts(), leg.legend_handles, strict=False):
        color = handle.get_facecolor()[0][:3]
        text.set_color(color)

    ax.fill_between(
        diagonal_range,
        diagonal_range,
        [max_gap * 1.1] * 2,
        alpha=0.1,
        color=SCATTER_COLOR,
        zorder=0,
        label="_nolegend_",
    )

    ax.set_aspect("equal", adjustable="box")

    x_range = np.array([min_gap * 0.95, max_gap * 1.05])
    y_range = np.array([min_gap * 0.95, max_gap * 1.05])
    range_frame(ax, x_range, y_range, pad=0.05)

    fig.tight_layout()

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
    output_filename: str | None = None,
) -> None:
    """Generate scatter plot showing model gap vs agent gap."""
    logger.info("=" * 60)
    logger.info("Gap Scatter Plot Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info("")

    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        msg = f"Invalid metric: {metric}. Must be 'average_score', 'pass_at_k', or 'pass_hat_k'"
        raise ValueError(msg)

    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        msg = f"Invalid k_value: {k_value}. Must be between 1 and 5"
        raise ValueError(msg)

    metric_column = get_metric_column_name(metric, k_value)

    logger.info("Loading datasets...")
    reports_df = load_reports_data()

    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(
        filtered_df, None if verbosity_strategy == "average" else verbosity_strategy
    )
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")

    logger.info("Computing performance gaps...")
    gap_data = collect_gap_data(filtered_df, metric_column)
    logger.info(f"Found data for {len(gap_data)} environments")

    logger.info("Generating scatter plot...")
    if output_filename is None:
        output_path = OUT_DIR / "panel2_gap_scatter.pdf"
    else:
        output_path = Path(output_filename)
    plot_gap_scatter(gap_data, output_path)

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


if __name__ == "__main__":
    fire.Fire(main)
