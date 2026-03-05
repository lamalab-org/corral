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
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.constants import golden

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import (  # noqa: E402
    ENVIRONMENT_NAMES,
    MODEL_COLOUR_MAP,
    MODEL_NAMES,
)

# ==================== CONFIGURATION ====================

# Data paths (relative to analysis directory)
REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "reports.jsonl"
QA_REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "qa_topic_reports.jsonl"

# Figure dimensions
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Offsets for plotting
MODEL_OFFSET = 0.1  # Horizontal separation between models
AGENT_OFFSET = 0.2  # Horizontal separation between react/tool_calling

# Metric to plot
METRIC = "Average Score"

# Map internal names to dataset column names
MODEL_NAME_MAP = {
    "claude-4.5": "claude-4.5",
    "gpt-4o": "gpt-4o",
    "gpt-oss-120b": "gpt-oss-120b",
}

AGENT_TYPE_MAP = {
    "react": "react",
    "tool_calling": "tool_calling",
}

VERBOSITY_MAP = {
    "brief": "brief",
    "workflow": "workflow",
    "comprehensive": "comprehensive",
}


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
        df: Input dataframe with "level" column
        task_type_strategy: "tasks" (level 1), "subtasks" (level > 1), or "both"

    Returns:
        Filtered dataframe
    """
    if task_type_strategy == "both":
        return df
    if task_type_strategy == "tasks":
        return df[df["level"] == 1]
    if task_type_strategy == "subtasks":
        return df[df["level"] > 1]
    msg = f"Invalid task_type_strategy: {task_type_strategy}"
    raise ValueError(msg)


def filter_by_agent_type(
    df: pd.DataFrame, agent_type_strategy: str, agent_value: str | None = None
) -> pd.DataFrame:
    """Filter dataframe by agent type strategy.

    Args:
        df: Input dataframe with "agent_type" column
        agent_type_strategy: "average", "react", or "tool_calling"
        agent_value: Specific agent value (overrides strategy)

    Returns:
        Filtered dataframe
    """
    if agent_value:
        agent_type_strategy = agent_value

    if agent_type_strategy == "average":
        return df
    if agent_type_strategy in ["react", "tool_calling"]:
        return df[df["agent_type"] == agent_type_strategy]
    msg = f"Invalid agent_type_strategy: {agent_type_strategy}"
    raise ValueError(msg)


# ==================== ORDERING FUNCTIONS ====================


def compute_qa_scores_by_env(
    qa_df: pd.DataFrame, ordering_strategy: str, model_for_ordering: str | None = None
) -> dict[str, float]:
    """Compute QA scores for each environment.

    Args:
        qa_df: QA topic reports dataframe
        ordering_strategy: "average_qa" or "model_specific_qa"
        model_for_ordering: Model name for model_specific_qa (e.g., "claude")

    Returns:
        Dictionary mapping environment to QA score
    """
    # Filter to only "qa" type (not reasoning_qa)
    qa_df = qa_df[qa_df["qa_type"] == "qa"].copy()

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
) -> list[str]:
    """Order environments by QA score.

    Args:
        qa_df: QA topic reports dataframe
        ordering_strategy: "average_qa" or "model_specific_qa"
        order_direction: "ascending" or "descending"
        model_for_ordering: Model name for model_specific_qa

    Returns:
        Ordered list of environment names
    """
    env_scores = compute_qa_scores_by_env(qa_df, ordering_strategy, model_for_ordering)

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


def collect_plot_data(
    df: pd.DataFrame,
    environments: list[str],
    models: list[str],
    agent_types: list[str],
) -> dict:
    """Collect aggregated data for plotting.

    Args:
        df: Filtered benchmark reports dataframe
        environments: List of environments to plot
        models: List of models to plot
        agent_types: List of agent types to plot

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
                score = agent_df[METRIC].mean()
                if not np.isnan(score):
                    plot_data[env][model][agent] = score

    return plot_data


# ==================== PLOTTING ====================


def plot_dumbbell(
    plot_data: dict,
    environments: list[str],
    models: list[str],
    output_path: Path,
) -> None:
    """Create dumbbell plot.

    Args:
        plot_data: Nested dict with scores
        environments: Ordered list of environments
        models: List of models to plot
        output_path: Path to save the figure
    """
    # Filter environments with data
    available_envs = [env for env in environments if plot_data.get(env)]

    if not available_envs:
        logger.warning("No data available for any environment!")
        return

    # Create figure
    fig, ax = plt.subplots(
        1, 1, figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.2)
    )

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
    ax.set_ylabel(METRIC, fontsize=12, fontweight="bold")

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
        fontsize=7,
        framealpha=0.9,
    )

    # Set y-axis limits with padding
    all_scores = []
    for env_data in plot_data.values():
        for model_data in env_data.values():
            all_scores.extend(model_data.values())

    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        padding = (max_score - min_score) * 0.1 if max_score > min_score else 0.1
        ax.set_ylim(max(0, min_score - padding), min(1, max_score + padding))

    # Set x-axis limits
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

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
    verbosity_strategy: str = "average",
    task_type_strategy: str = "both",
    agent_type_strategy: str = "average",
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
        verbosity_strategy: Which tool verbosity to use.
            Options: "average", "brief", "workflow", "comprehensive"
        task_type_strategy: Which task type to use.
            Options: "tasks" (level 1), "subtasks" (level > 1), "both"
        agent_type_strategy: Which agent types to compare.
            Options: "average" (shows both), "react", "tool_calling"
        models: Comma-separated list of models to plot
            (e.g., "claude-4.5,gpt-4o,gpt-oss-120b")
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Dumbbell Plot Generation")
    logger.info("=" * 60)
    logger.info(f"Ordering strategy: {ordering_strategy}")
    logger.info(f"Order direction: {order_direction}")
    if model_for_ordering:
        logger.info(f"Model for ordering: {model_for_ordering}")
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Agent type strategy: {agent_type_strategy}")
    logger.info(f"Models to plot: {models}")
    logger.info("")

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
    )
    logger.info("")

    # Filter reports data
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(filtered_df, verbosity_strategy)
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)

    # For dumbbell plot, we always show both agent types
    agent_types_to_plot = ["react", "tool_calling"]

    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect plot data
    logger.info("Collecting plot data...")
    plot_data = collect_plot_data(
        filtered_df,
        ordered_envs,
        model_list,
        agent_types_to_plot,
    )

    # Generate plot
    logger.info("Generating plot...")
    output_path = Path(output_filename)
    plot_dumbbell(plot_data, ordered_envs, model_list, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dumbbell plot generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
