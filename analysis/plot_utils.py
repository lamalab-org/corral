"""Shared plotting utilities for Corral benchmark analysis.

This module provides common functions for:
- Loading benchmark and QA data
- Filtering by verbosity, task type, level, and agent type
- Ordering environments by QA scores
- Metric column name handling
"""

from pathlib import Path

import pandas as pd
from loguru import logger

# ==================== CONFIGURATION ====================

# Get repo root - this file is in analysis/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data paths
REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "reports.jsonl"
QA_REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "qa_topic_reports.jsonl"
LOGPROBS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "logprobs.jsonl"
REASONING_PATH = REPO_ROOT / "analysis" / "reasoning.json"

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


# ==================== METRIC HELPERS ====================


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
        return "Average Score"
    elif metric == "pass_at_k":
        return f"Pass@{k_value}"
    elif metric == "pass_hat_k":
        return f"Pass^{k_value}"
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


def load_logprobs_data() -> pd.DataFrame:
    """Load per-message log-probability trace dataset."""
    if not LOGPROBS_PATH.exists():
        msg = f"Logprobs file not found: {LOGPROBS_PATH}"
        raise FileNotFoundError(msg)

    df = pd.read_json(LOGPROBS_PATH, lines=True)  # noqa: PD901
    logger.info(f"Loaded {len(df)} rows from logprobs.jsonl")
    return df


def load_reasoning_data() -> dict:
    """Load reasoning heaviness mapping."""
    import json

    if not REASONING_PATH.exists():
        msg = f"Reasoning file not found: {REASONING_PATH}"
        raise FileNotFoundError(msg)

    with REASONING_PATH.open() as f:
        reasoning_data = json.load(f)

    logger.info(f"Loaded reasoning data for {len(reasoning_data)} environments")
    return reasoning_data


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


def filter_by_model(df: pd.DataFrame, model_strategy: str) -> pd.DataFrame:
    """Filter dataframe by model strategy.

    Args:
        df: Input dataframe with "model" column
        model_strategy: "average" or specific model name

    Returns:
        Filtered dataframe
    """
    if model_strategy == "average":
        return df
    # Specific model
    return df[df["model"] == model_strategy]


def filter_by_agent(df: pd.DataFrame, agent_strategy: str) -> pd.DataFrame:
    """Filter dataframe by agent strategy.

    Args:
        df: Input dataframe with "agent_type" column
        agent_strategy: "average", "react", or "tool_calling"

    Returns:
        Filtered dataframe
    """
    if agent_strategy == "average":
        return df
    if agent_strategy in ["react", "tool_calling"]:
        return df[df["agent_type"] == agent_strategy]
    msg = f"Invalid agent_strategy: {agent_strategy}"
    raise ValueError(msg)


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
