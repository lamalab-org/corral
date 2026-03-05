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

# ==================== CONFIGURATION ====================

# Data paths (relative to analysis directory)
REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "reports.jsonl"
QA_REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "qa_topic_reports.jsonl"

# Default per-environment level selection (used when level_strategy="default_map")
DEFAULT_ENV_LEVEL_MAP = {
    "afm": 1,
    "catalyst": 1,
    "md": 2,
    "ml": 1,
    "resistor": 1,
    "retro": 2,
    "spectra": 1,
    "wetlab": 2,
}


# ==================== HELPER FUNCTIONS ====================


def get_metric_column_name(metric: str, k_value: int) -> str:
    """Get the column name for the specified metric.

    Args:
        metric: Metric type - "average_score", "pass_at_k", or "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics

    Returns:
        Column name in the dataframe
    """
    if metric == "average_score":
        return "Average Score"
    elif metric == "pass_at_k":
        return f"Pass@{k_value}"
    elif metric == "pass_hat_k":
        return f"Pass^{k_value}"
    else:
        msg = f"Invalid metric: {metric}"
        raise ValueError(msg)


def get_metric_display_name(metric: str, k_value: int) -> str:
    """Get the display name for the specified metric.

    Args:
        metric: Metric type - "average_score", "pass_at_k", or "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics

    Returns:
        Display name for the metric
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


# ==================== DATA LOADING ====================


def load_reports_data() -> pd.DataFrame:
    """Load main benchmark reports dataset."""
    if not REPORTS_PATH.exists():
        msg = f"Reports file not found: {REPORTS_PATH}"
        raise FileNotFoundError(msg)

    df = pd.read_json(REPORTS_PATH, lines=True)  # noqa: PD901
    logger.info(f"Loaded {len(df)} rows from reports.jsonl")
    return df


def load_qa_data() -> pd.DataFrame:
    """Load QA topic reports dataset."""
    if not QA_REPORTS_PATH.exists():
        msg = f"QA reports file not found: {QA_REPORTS_PATH}"
        raise FileNotFoundError(msg)

    df = pd.read_json(QA_REPORTS_PATH, lines=True)  # noqa: PD901
    logger.info(f"Loaded {len(df)} rows from qa_topic_reports.jsonl")
    return df


# ==================== FILTERING FUNCTIONS ====================


def filter_by_verbosity(
    df: pd.DataFrame, verbosity_strategy: str, verbosity_value: str | None = None
) -> pd.DataFrame:
    """Filter dataframe by verbosity strategy.

    Args:
        df: Input dataframe with "Tool Verbosity" column
        verbosity_strategy: "average", "brief", "workflow", or "comprehensive"
        verbosity_value: Specific verbosity value (overrides strategy)

    Returns:
        Filtered dataframe
    """
    if verbosity_value:
        verbosity_strategy = verbosity_value

    if verbosity_strategy == "average":
        # Keep all verbosities (will be averaged in aggregation)
        return df
    if verbosity_strategy in ["brief", "workflow", "comprehensive"]:
        return df[df["Tool Verbosity"] == verbosity_strategy]
    msg = f"Invalid verbosity_strategy: {verbosity_strategy}"
    raise ValueError(msg)


def filter_by_task_type(df: pd.DataFrame, task_type_strategy: str) -> pd.DataFrame:
    """Filter dataframe by task type strategy.

    Args:
        df: Input dataframe with "category" column
        task_type_strategy: "tasks", "subtasks", or "both"

    Returns:
        Filtered dataframe
    """
    if task_type_strategy == "both":
        return df
    if task_type_strategy == "tasks":
        return df[df["category"] == "task"]
    if task_type_strategy == "subtasks":
        return df[df["category"] == "subtask"]
    msg = f"Invalid task_type_strategy: {task_type_strategy}"
    raise ValueError(msg)


def filter_by_level(
    df: pd.DataFrame,
    level_strategy: str,
) -> pd.DataFrame:
    """Filter dataframe by level strategy.

    Args:
        df: Input dataframe with "environment" and "level" columns
        level_strategy: "all", "default_map", or specific level (e.g., "1", "2")

    Returns:
        Filtered dataframe
    """
    if level_strategy == "all":
        # Keep all levels
        return df

    if level_strategy == "default_map":
        # Use per-environment level mapping
        mask = pd.Series(False, index=df.index)
        for env, level in DEFAULT_ENV_LEVEL_MAP.items():
            mask |= (df["environment"] == env) & (df["level"] == level)
        filtered_df = df[mask]
        logger.debug(f"Applied default_map: {DEFAULT_ENV_LEVEL_MAP}")
        return filtered_df

    # Try to parse as integer level (apply to all environments)
    try:
        level_int = int(level_strategy)
        if level_int < 1:
            msg = f"Level must be >= 1, got: {level_int}"
            raise ValueError(msg)
        filtered_df = df[df["level"] == level_int]
        logger.debug(f"Filtered to level {level_int} for all environments")
        return filtered_df
    except ValueError as e:
        if "invalid literal" in str(e):
            msg = f"Invalid level_strategy: {level_strategy}. Must be 'all', 'default_map', or an integer"
            raise ValueError(msg) from e
        raise


# ==================== ORDERING FUNCTIONS ====================


def compute_qa_scores_by_env(
    qa_df: pd.DataFrame,
    ordering_strategy: str,
    model_for_ordering: str | None = None,
    qa_type: str = "qa",
) -> dict[str, float]:
    """Compute QA scores for each environment.

    Args:
        qa_df: QA topic reports dataframe
        ordering_strategy: "average_qa" or "model_specific_qa"
        model_for_ordering: Model name for model_specific_qa (e.g., "claude")
        qa_type: Type of QA to use - "qa" or "reasoning_qa"

    Returns:
        Dictionary mapping environment to QA score
    """
    # Filter to specified QA type
    qa_df = qa_df[qa_df["qa_type"] == qa_type].copy()

    env_scores = {}

    if ordering_strategy == "average_qa":
        # Average across all models for each environment
        for env in qa_df["env"].unique():
            env_data = qa_df[qa_df["env"] == env]
            avg_score = env_data["overall_score"].mean()
            env_scores[env] = avg_score

    elif ordering_strategy == "model_specific_qa":
        if not model_for_ordering:
            msg = "model_for_ordering required for model_specific_qa strategy"
            raise ValueError(msg)

        # Use QA score for specific model
        for env in qa_df["env"].unique():
            env_data = qa_df[
                (qa_df["env"] == env) & (qa_df["model"] == model_for_ordering)
            ]
            if not env_data.empty:
                env_scores[env] = env_data["overall_score"].iloc[0]
            else:
                logger.warning(f"No QA data for {env} with model {model_for_ordering}")
                env_scores[env] = 0.0

    else:
        msg = f"Invalid ordering_strategy: {ordering_strategy}"
        raise ValueError(msg)

    return env_scores


def order_environments(
    qa_df: pd.DataFrame,
    ordering_strategy: str,
    order_direction: str,
    model_for_ordering: str | None = None,
    qa_type: str = "qa",
) -> list[str]:
    """Order environments by QA score.

    Args:
        qa_df: QA topic reports dataframe
        ordering_strategy: "average_qa" or "model_specific_qa"
        order_direction: "ascending" or "descending"
        model_for_ordering: Model name for model_specific_qa
        qa_type: Type of QA to use - "qa" or "reasoning_qa"

    Returns:
        Ordered list of environment names
    """
    env_scores = compute_qa_scores_by_env(
        qa_df, ordering_strategy, model_for_ordering, qa_type
    )

    # Sort by score
    sorted_envs = sorted(
        env_scores.items(),
        key=lambda x: x[1],
        reverse=(order_direction == "descending"),
    )

    ordered_env_list = [env for env, _ in sorted_envs]

    logger.info(f"Environment ordering ({order_direction}):")
    for env, score in sorted_envs:
        logger.info(f"  {env}: {score:.3f}")

    return ordered_env_list


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
    metric_display_name = get_metric_display_name(metric, k_value)

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
