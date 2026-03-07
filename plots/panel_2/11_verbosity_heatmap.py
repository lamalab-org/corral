"""Enhanced coverage heatmap with verbosity bars within each cell.

Like the full coverage heatmap, but each cell contains 3 small bars showing performance
at each verbosity level (brief, workflow, comprehensive). The background shows the
average score, and the bars are semi-transparent overlays.

Usage:
    python 11_verbosity_heatmap.py --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.patches as mpatches
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
    get_metric_column_name,
    load_reports_data,
)

# ==================== CONFIGURATION ====================

AGENT_NAMES = {
    "react": "ReAct",
    "tool_calling": "Tool-Call",
}

VERBOSITY_ORDER = ["brief", "workflow", "comprehensive"]
VERBOSITY_COLORS = {
    "brief": "#D1C4E9",  # Light purple
    "workflow": "#9575CD",  # Medium purple
    "comprehensive": "#5E35B1",  # Dark purple
}


# ==================== DATA COLLECTION ====================


def collect_verbosity_data(
    df: pd.DataFrame,
    metric_column: str,
) -> tuple[pd.DataFrame, dict]:
    """Collect data for verbosity heatmap.

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        Tuple of (average_matrix, verbosity_details)
        - average_matrix: DataFrame with configs x env-levels (averaged across verbosity)
        - verbosity_details: Dict with detailed scores per verbosity level
    """
    # Create configuration column (use environment directly, not env_level)
    data = df.copy()
    data["config"] = data["model"] + "_" + data["agent_type"]

    # Create average matrix (for background colors)
    avg_matrix = data.pivot_table(
        values=metric_column,
        index="config",
        columns="environment",
        aggfunc="mean",
    )

    # Collect verbosity-specific data
    verbosity_details = {}
    for config in data["config"].unique():
        verbosity_details[config] = {}
        for env in data["environment"].unique():
            subset = data[(data["config"] == config) & (data["environment"] == env)]

            if subset.empty:
                continue

            verbosity_scores = {}
            for verb in VERBOSITY_ORDER:
                verb_subset = subset[subset["Tool Verbosity"] == verb]
                if not verb_subset.empty:
                    verbosity_scores[verb] = verb_subset[metric_column].mean()

            if verbosity_scores:
                verbosity_details[config][env] = verbosity_scores

    return avg_matrix, verbosity_details


def format_labels(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Format row and column labels."""
    display_df = pivot_df.copy()

    # Format row labels
    row_labels = []
    for config in display_df.index:
        parts = config.split("_")
        if len(parts) >= 2:
            model = parts[0]
            agent = "_".join(parts[1:])
            model_name = MODEL_NAMES.get(model, model)
            agent_name = AGENT_NAMES.get(agent, agent)
            row_labels.append(f"{model_name}\n{agent_name}")
        else:
            row_labels.append(config)
    display_df.index = row_labels

    # Format column labels (just environment names)
    col_labels = [ENVIRONMENT_NAMES.get(env, env.upper()) for env in display_df.columns]
    display_df.columns = col_labels

    return display_df


# ==================== PLOTTING ====================


def plot_verbosity_heatmap(
    avg_matrix: pd.DataFrame,
    verbosity_details: dict,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create enhanced heatmap with verbosity bars in cells."""
    if avg_matrix.empty:
        logger.warning("No data available for heatmap!")
        return

    # Format labels
    display_df = format_labels(avg_matrix)

    # Calculate figure size
    n_cols = len(display_df.columns)
    n_rows = len(display_df.index)
    fig_width = min(TWO_COL_WIDTH * 1.5, max(TWO_COL_WIDTH, n_cols * 0.5))
    fig_height = max(ONE_COL_HEIGHT * 0.8, n_rows * 0.6)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Create background heatmap showing average scores
    sns.heatmap(
        display_df,
        annot=False,
        cmap="Purples",  # background_cmap,  # White to #efbbff gradient based on average score
        cbar=True,
        cbar_kws={"label": metric_display_name, "shrink": 0.8},
        ax=ax,
        linewidths=1.5,
        linecolor="white",
        vmin=0,
        vmax=1,
    )

    # Add verbosity bars within cells and average score annotations
    for i, (_row_label, config) in enumerate(
        zip(display_df.index, avg_matrix.index, strict=False)
    ):
        for j, (_col_label, env) in enumerate(
            zip(display_df.columns, avg_matrix.columns, strict=False)
        ):
            # Add average score text in center of cell
            avg_score = avg_matrix.loc[config, env]
            if not np.isnan(avg_score):
                # Choose text color based on background brightness
                # Light background (low score) = black text, dark background (high score) = white text
                text_color = "black" if avg_score < 0.6 else "white"
                # text_color = "#9f4e00"
                ax.text(
                    j + 0.5,  # Center of cell horizontally
                    i + 0.5,  # Center of cell vertically
                    f"{avg_score:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZES["tick_label"] - 1,
                    fontweight="bold",
                    color=text_color,
                    zorder=20,  # Above bars
                )

            if config not in verbosity_details:
                continue
            if env not in verbosity_details[config]:
                continue

            verb_scores = verbosity_details[config][env]

            # Calculate bar positions - bars fill full width of cell
            cell_width = 1.0
            cell_height = 0.85  # Leave small margin at top
            bar_width = cell_width / len(VERBOSITY_ORDER)

            for k, verb in enumerate(VERBOSITY_ORDER):
                if verb not in verb_scores:
                    continue

                score = verb_scores[verb]
                bar_height = score * cell_height

                # Position bars within cell (no padding, full width)
                x_pos = j + k * bar_width
                y_pos = i + 1 - bar_height

                # Draw mini bar
                rect = mpatches.Rectangle(
                    (x_pos, y_pos),
                    bar_width,
                    bar_height,
                    facecolor=VERBOSITY_COLORS[verb],
                    alpha=0.5,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=10,
                )
                ax.add_patch(rect)

    # Styling
    ax.set_xlabel(
        "Environment",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        "Model x Agent Configuration",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"] - 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add title
    ax.set_title(
        f"Performance with Verbosity Breakdown: {n_rows} Configs x {n_cols} Environments",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
        pad=15,
    )

    # Add legend for verbosity bars
    legend_elements = [
        mpatches.Patch(facecolor=VERBOSITY_COLORS[v], alpha=0.85, label=v.title())
        for v in VERBOSITY_ORDER
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        fontsize=FONT_SIZES["legend"],
        title="Tool Verbosity",
        framealpha=0.9,
    )

    fig.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


# ==================== MAIN ====================


def main(
    task_type_strategy: str = "both",
    level_strategy: str = "default_map",
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str = "verbosity_heatmap.pdf",
) -> None:
    """Generate enhanced heatmap with verbosity bars in cells.

    Args:
        task_type_strategy: Which task type to use.
        level_strategy: Which level to use (default: "default_map")
        metric: Metric to plot.
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Verbosity Heatmap Generation")
    logger.info("=" * 60)
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Level strategy: {level_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info("")

    # Validate inputs
    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        raise ValueError(f"Invalid metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        raise ValueError(f"Invalid k_value: {k_value}")

    # Get metric column
    metric_column = get_metric_column_name(metric, k_value)
    if metric == "average_score":
        metric_display_name = "Average Score"
    elif metric == "pass_at_k":
        metric_display_name = f"Pass@{k_value}"
    else:
        metric_display_name = f"Pass^{k_value}"

    # Load data
    logger.info("Loading datasets...")
    reports_df = load_reports_data()

    # Filter (keep all verbosity levels, but filter by level using default_map)
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)
    filtered_df = filter_by_level(filtered_df, level_strategy)
    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect data
    logger.info("Preparing verbosity heatmap data...")
    avg_matrix, verbosity_details = collect_verbosity_data(filtered_df, metric_column)
    logger.info(
        f"Matrix: {avg_matrix.shape[0]} configs x {avg_matrix.shape[1]} environments"
    )

    # Generate plot
    logger.info("Generating verbosity heatmap...")
    output_path = Path(output_filename)
    plot_verbosity_heatmap(
        avg_matrix, verbosity_details, output_path, metric_display_name
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Verbosity heatmap generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
