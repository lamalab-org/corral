"""Full coverage heatmap showing all model x agent configurations across all environment-level combinations.

Creates a heatmap where rows are model x agent configurations (6 total) and columns are
all environment-level combinations tested (e.g., AFM-1, AFM-2, Catalyst-1, MD-1, MD-2, etc.).
This visualization shows the complete experimental scope and performance across all conditions.

Usage:
    python 9_full_coverage_heatmap.py --verbosity_strategy=average --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
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
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

# ==================== CONFIGURATION ====================

AGENT_NAMES = {
    "react": "ReAct",
    "tool_calling": "Tool-Call",
}


# ==================== DATA COLLECTION ====================


def collect_full_coverage_data(
    df: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Collect data for full coverage heatmap.

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        DataFrame with model x agent configs as rows, environment-level combos as columns
    """
    # Create model x agent configuration column
    data = df.copy()
    data["config"] = data["model"] + "_" + data["agent_type"]

    # Create environment-level column
    data["env_level"] = data["environment"] + "-" + data["level"].astype(str)

    # Pivot to create config x env_level matrix
    return data.pivot_table(
        values=metric_column,
        index="config",
        columns="env_level",
        aggfunc="mean",
    )


def format_heatmap_labels(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Format row and column labels for display.

    Args:
        pivot_df: Pivot dataframe with raw labels

    Returns:
        DataFrame with formatted labels
    """
    display_df = pivot_df.copy()

    # Format row labels (model_agent -> Model - Agent)
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

    # Format column labels (environment-level -> ENV-L#)
    col_labels = []
    for env_level in display_df.columns:
        parts = env_level.split("-")
        if len(parts) >= 2:
            env = parts[0]
            level = parts[1]
            env_name = ENVIRONMENT_NAMES.get(env, env.upper())
            col_labels.append(f"{env_name}-L{level}")
        else:
            col_labels.append(env_level)
    display_df.columns = col_labels

    return display_df


# ==================== PLOTTING ====================


def plot_full_coverage_heatmap(
    heatmap_data: pd.DataFrame,
    output_path: Path,
    metric_display_name: str,
) -> None:
    """Create full coverage heatmap.

    Args:
        heatmap_data: DataFrame with configs as rows, env-levels as columns
        output_path: Path to save the figure
        metric_display_name: Display name for the metric
    """
    if heatmap_data.empty:
        logger.warning("No data available for heatmap!")
        return

    # Format labels
    display_df = format_heatmap_labels(heatmap_data)

    # Calculate figure width based on number of columns
    n_cols = len(display_df.columns)
    n_rows = len(display_df.index)

    # Adaptive sizing: wider if many columns
    fig_width = min(TWO_COL_WIDTH * 1.5, max(TWO_COL_WIDTH, n_cols * 0.4))
    fig_height = max(ONE_COL_HEIGHT * 0.8, n_rows * 0.5)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Create heatmap
    sns.heatmap(
        display_df,
        annot=True,
        fmt=".2f",
        cmap="Purples",  # "RdYlGn",
        cbar_kws={"label": metric_display_name, "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"] - 3},
    )

    # Styling
    ax.set_xlabel(
        "Environment - Level",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        "Model x Agent Configuration",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"] - 1)

    # Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add title with experimental scope
    n_configs = len(display_df.index)
    n_env_levels = len(display_df.columns)
    ax.set_title(
        f"Full Experimental Coverage: {n_configs} Configurations x {n_env_levels} Environment-Level Combinations",
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
    output_filename: str = "full_coverage_heatmap.pdf",
) -> None:
    """Generate full coverage heatmap showing all tested configurations.

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
    logger.info("Full Coverage Heatmap Generation")
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

    # Collect heatmap data
    logger.info("Preparing full coverage data...")
    heatmap_data = collect_full_coverage_data(filtered_df, metric_column)
    logger.info(
        f"Matrix shape: {heatmap_data.shape[0]} configs x {heatmap_data.shape[1]} env-levels"
    )

    # Print coverage summary
    logger.info("")
    logger.info("Experimental Coverage:")
    logger.info("-" * 40)
    logger.info(f"Configurations tested: {heatmap_data.shape[0]}")
    logger.info(f"Environment-level combinations: {heatmap_data.shape[1]}")
    logger.info(
        f"Total experimental conditions: {heatmap_data.shape[0] * heatmap_data.shape[1]}"
    )

    # Check for missing data
    n_missing = heatmap_data.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Missing data points: {n_missing}")

    # Generate plot
    logger.info("")
    logger.info("Generating full coverage heatmap...")
    output_path = Path(output_filename)
    plot_full_coverage_heatmap(heatmap_data, output_path, metric_display_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Full coverage heatmap generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
