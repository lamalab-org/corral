"""
Filtering utilities for selecting subsets of data
"""

import pandas as pd
from loguru import logger

from corral_trace_analyzer.config import (
    SELECTED_AGENT_TYPES,
    SELECTED_ENVIRONMENTS,
    SELECTED_MODELS,
)


def filter_by_config(
    df: pd.DataFrame,
    environments: list[str] | None = None,
    models: list[str] | None = None,
    agent_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter dataframe based on config or provided filters

    Args:
        df: DataFrame to filter
        environments: list of environments to include (overrides config)
        models: list of models to include (overrides config)
        agent_types: list of agent types to include (overrides config)

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Use provided filters or fall back to config
    envs = environments if environments is not None else SELECTED_ENVIRONMENTS
    mods = models if models is not None else SELECTED_MODELS
    agents = agent_types if agent_types is not None else SELECTED_AGENT_TYPES

    # Apply filters
    if envs is not None and "environment" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["environment"].isin(envs)]
        logger.info(f"Filtered to environments: {envs} ({len(filtered_df)} rows)")

    if mods is not None and "model" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["model"].isin(mods)]
        logger.info(f"Filtered to models: {mods} ({len(filtered_df)} rows)")

    if agents is not None and "agent_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["agent_type"].isin(agents)]
        logger.info(f"Filtered to agent types: {agents} ({len(filtered_df)} rows)")

    return filtered_df


def get_environment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics per environment

    Args:
        df: DataFrame with environment column

    Returns:
        DataFrame with stats per environment
    """
    if "environment" not in df.columns:
        return pd.DataFrame()

    stats = []

    for env in df["environment"].unique():
        env_df = df[df["environment"] == env]

        env_stats = {
            "environment": env,
            "n_traces": len(env_df),
            "n_models": env_df["model"].nunique() if "model" in env_df.columns else 0,
            "n_agent_types": (
                env_df["agent_type"].nunique() if "agent_type" in env_df.columns else 0
            ),
        }

        if "score" in env_df.columns:
            env_stats["mean_score"] = env_df["score"].mean()
            env_stats["success_rate"] = env_df["score"].mean()  # If binary
            env_stats["std_score"] = env_df["score"].std()

        if "total_calls" in env_df.columns:
            env_stats["mean_calls"] = env_df["total_calls"].mean()

        if "total_tokens" in env_df.columns:
            env_stats["mean_tokens"] = env_df["total_tokens"].mean()

        stats.append(env_stats)

    return pd.DataFrame(stats).sort_values("n_traces", ascending=False)


def filter_by_success(
    df: pd.DataFrame, success_only: bool = False, failure_only: bool = False
) -> pd.DataFrame:
    """
    Filter to only successful or failed traces

    Args:
        df: DataFrame
        success_only: If True, keep only successful traces
        failure_only: If True, keep only failed traces

    Returns:
        Filtered DataFrame
    """
    if success_only and "score" in df.columns:
        return df[df["score"] == 1]
    elif failure_only and "score" in df.columns:
        return df[df["score"] == 0]
    else:
        return df


def filter_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.DataFrame:
    """
    Filter outliers based on standard deviations

    Args:
        df: DataFrame
        column: Column to check for outliers
        n_std: Number of standard deviations for outlier threshold

    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        return df

    mean = df[column].mean()
    std = df[column].std()

    lower = mean - n_std * std
    upper = mean + n_std * std

    return df[(df[column] >= lower) & (df[column] <= upper)]


def split_by_environment(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split dataframe by environment

    Args:
        df: DataFrame with environment column

    Returns:
        dictionary mapping environment name to DataFrame
    """
    if "environment" not in df.columns:
        return {"all": df}

    return {env: df[df["environment"] == env] for env in df["environment"].unique()}


def get_balanced_sample(
    df: pd.DataFrame,
    target_col: str = "score",
    n_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Get a balanced sample with equal numbers of each class

    Useful for binary classification when classes are imbalanced

    Args:
        df: DataFrame
        target_col: Target column
        n_per_class: Number of samples per class (if None, use min class size)
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    if target_col not in df.columns:
        return df

    class_counts = df[target_col].value_counts()

    if n_per_class is None:
        n_per_class = class_counts.min()

    balanced_dfs = []

    for class_val in class_counts.index:
        class_df = df[df[target_col] == class_val]
        sampled = class_df.sample(
            n=min(n_per_class, len(class_df)), random_state=random_state
        )
        balanced_dfs.append(sampled)

    return pd.concat(balanced_dfs, ignore_index=True)
