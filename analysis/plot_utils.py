"""Shared plotting utilities for Corral benchmark analysis.

This module provides common functions for:
- Loading benchmark and QA data
- Filtering by verbosity, task type, level, and agent type
- Ordering environments by QA scores
- Metric column name handling
"""

import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from plot_config import DEFAULT_ENV_LEVEL_MAP

# ==================== CONFIGURATION ====================

# Get repo root - this file is in analysis/
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data paths
REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "reports.jsonl"
QA_REPORTS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "qa_topic_reports.jsonl"
LOGPROBS_PATH = REPO_ROOT / "analysis" / "results" / "data" / "logprobs.jsonl"
REASONING_PATH = REPO_ROOT / "analysis" / "reasoning.json"


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


def filter_by_verbosity(df: pd.DataFrame, verbosity: str | None = None) -> pd.DataFrame:
    """Filter dataframe by verbosity.

    Args:
        df: Input dataframe with "Tool Verbosity" column
        verbosity: "brief", "workflow", or "comprehensive". None keeps all rows.

    Returns:
        Filtered dataframe
    """
    if verbosity is None:
        return df
    if verbosity in ["brief", "workflow", "comprehensive"]:
        return df[df["Tool Verbosity"] == verbosity]
    msg = f"Invalid verbosity: {verbosity}"
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
    df: pd.DataFrame, agent_type: str | None = None
) -> pd.DataFrame:
    """Filter dataframe by agent type.

    Args:
        df: Input dataframe with "agent_type" column
        agent_type: "react" or "tool_calling". None keeps all rows.

    Returns:
        Filtered dataframe
    """
    if agent_type is None:
        return df
    if agent_type in ["react", "tool_calling"]:
        return df[df["agent_type"] == agent_type]
    msg = f"Invalid agent_type: {agent_type}"
    raise ValueError(msg)


def filter_by_model(df: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Filter dataframe by model.

    Args:
        df: Input dataframe with "model" column
        model: Specific model name. None keeps all rows.

    Returns:
        Filtered dataframe
    """
    if model is None:
        return df
    return df[df["model"] == model]


def filter_by_agent(df: pd.DataFrame, agent: str | None = None) -> pd.DataFrame:
    """Filter dataframe by agent.

    Args:
        df: Input dataframe with "agent_type" column
        agent: "react" or "tool_calling". None keeps all rows.

    Returns:
        Filtered dataframe
    """
    if agent is None:
        return df
    if agent in ["react", "tool_calling"]:
        return df[df["agent_type"] == agent]
    msg = f"Invalid agent: {agent}"
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


# ==================== CATEGORY CLASSIFICATION ====================

CATEGORY_TAGS_PATH = Path(__file__).parent / "subtask_category_tags.json"


def load_category_tags() -> dict:
    """Load subtask category tags from JSON."""
    if not CATEGORY_TAGS_PATH.exists():
        msg = f"Category tags file not found: {CATEGORY_TAGS_PATH}"
        raise FileNotFoundError(msg)
    with CATEGORY_TAGS_PATH.open() as f:
        return json.load(f)


def get_env_key_mapping() -> dict[str, str]:
    """Map environment short names to keys used in subtask_category_tags.json."""
    return {
        "spectra": "sptectra",
        "retro": "retrosynthesis",
        "afm": "afm",
        "catalyst": "catalyst",
        "md": "md",
        "ml": "ml",
        "resistor": "resistor",
        "wetlab": "wetlab",
    }


def classify_subtask(subtask: str, environment: str, category_tags: dict) -> str | None:
    """Classify a subtask into a category using the category tags mapping."""
    env_mapping = get_env_key_mapping()
    tag_env_key = env_mapping.get(environment, environment)
    env_tags = category_tags.get(tag_env_key, {})
    if not env_tags:
        return None

    if environment == "afm":
        if "subtask_level_" in subtask:
            base_name = subtask.split("_level_")[0]
            if base_name + "_level_1" in env_tags:
                return env_tags[base_name + "_level_1"]
        return env_tags.get(subtask)

    if environment == "catalyst":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[1] in env_tags:
            return env_tags[parts[1]]
        return None

    if environment == "md":
        for task_type in ["melting", "quenching", "surface_energy", "surface"]:
            if task_type in subtask:
                task_tags = env_tags.get(task_type, {})
                if "subtask_" in subtask:
                    subtask_name = subtask.split("subtask_")[-1]
                    if subtask_name == "diffusion_coefficient":
                        subtask_name = "diffusivity"
                    elif subtask_name == "tg_calculation":
                        subtask_name = "tg_detection"
                    elif subtask_name == "equilibration":
                        subtask_name = "structure_retrieval"
                    if subtask_name in task_tags:
                        return task_tags[subtask_name]
        return None

    if environment == "ml":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            if (
                task_name.startswith("batch_retrieve_")
                and "batch_retrieve_*" in env_tags
            ):
                return env_tags["batch_retrieve_*"]
            if task_name in env_tags:
                return env_tags[task_name]
        return None

    if environment == "resistor":
        parts = subtask.split("_", 2)
        if len(parts) >= 3 and parts[0] == "task" and parts[1].isdigit():
            pattern = "_".join(parts[2:])
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "retro":
        if "-" in subtask:
            pattern = subtask.split("-", 1)[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "spectra":
        if "_subtask_" in subtask:
            subtask_num = "subtask_" + subtask.split("_subtask_")[-1]
            if subtask_num in env_tags:
                return env_tags[subtask_num]
        return None

    if environment == "wetlab":
        m = re.match(r"qualysis_lvl(\d+)_\d+_(sub\d+)", subtask)
        if m:
            level_key = f"level_{m.group(1)}"
            sub_key = f"qualysis_lvl{m.group(1)}_*_{m.group(2)}"
            level_tags = env_tags.get(level_key, {})
            if sub_key in level_tags:
                return level_tags[sub_key]
        return None

    return None
