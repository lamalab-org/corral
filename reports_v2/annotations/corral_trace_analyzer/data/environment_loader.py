"""
Environment-level data loader for aggregated results
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from corral_trace_analyzer.config import (
    SELECTED_AGENT_TYPES,
    SELECTED_ENVIRONMENTS,
    SELECTED_MODELS,
    SELECTED_VERBOSITY,
)


class EnvironmentDataLoader:
    """Load and process environment-level aggregated results"""

    def __init__(self, results_path: str):
        """
        Initialize environment data loader

        Args:
            results_path: Path to main_results.json file
        """
        self.results_path = Path(results_path)
        self.raw_data = None
        self.df = None

    def load(self) -> pd.DataFrame:
        """
        Load the main results JSON file

        Returns:
            DataFrame with environment-level results
        """
        logger.info(f"Loading environment results from {self.results_path}...")

        with self.results_path.open() as f:
            self.raw_data = json.load(f)

        self.df = pd.DataFrame(self.raw_data)
        logger.info(f"Loaded {len(self.df)} environment configurations")

        return self.df

    def filter_by_config(
        self,
        environments: list[str] | None = None,
        models: list[str] | None = None,
        agent_types: list[str] | None = None,
        verbosity: list[str] | None = None,
        levels: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Filter data based on configuration

        Args:
            environments: list of environments to include
            models: list of models to include
            agent_types: list of agent types to include
            verbosity: list of verbosity levels to include
            levels: list of difficulty levels to include

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            self.load()

        filtered_df = self.df.copy()

        # Use provided filters or fall back to config
        envs = environments if environments is not None else SELECTED_ENVIRONMENTS
        mods = models if models is not None else SELECTED_MODELS
        agents = agent_types if agent_types is not None else SELECTED_AGENT_TYPES
        verb = verbosity if verbosity is not None else SELECTED_VERBOSITY

        # Apply filters
        if envs is not None:
            filtered_df = filtered_df[filtered_df["environment"].isin(envs)]
            logger.info(f"Filtered to environments: {envs} ({len(filtered_df)} rows)")

        if mods is not None:
            filtered_df = filtered_df[filtered_df["model"].isin(mods)]
            logger.info(f"Filtered to models: {mods} ({len(filtered_df)} rows)")

        if agents is not None:
            filtered_df = filtered_df[filtered_df["agent_type"].isin(agents)]
            logger.info(f"Filtered to agent types: {agents} ({len(filtered_df)} rows)")

        if verb is not None:
            filtered_df = filtered_df[filtered_df["tool_verbosity"].isin(verb)]
            logger.info(f"Filtered to verbosity: {verb} ({len(filtered_df)} rows)")

        if levels is not None:
            filtered_df = filtered_df[filtered_df["level"].isin(levels)]
            logger.info(f"Filtered to levels: {levels} ({len(filtered_df)} rows)")

        return filtered_df

    def add_derived_features(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Add derived features to the environment data

        Args:
            df: DataFrame to add features to (if None, uses self.df)

        Returns:
            DataFrame with additional features
        """
        # Ensure there is a DataFrame to work with; load from file if necessary
        if df is None:
            if self.df is None:
                self.load()
            df_ = self.df.copy()
        else:
            df_ = df.copy()

        # Error rates
        df_["failed_call_rate"] = df_["failed_tool_calls"] / df_["total_tool_calls"]
        df_["success_call_rate"] = (
            df_["successful_tool_calls"] / df_["total_tool_calls"]
        )

        # Token efficiency
        df_["tokens_per_call"] = df_["total_overall_tokens"] / df_["total_tool_calls"]
        df_["prompt_to_completion_ratio"] = df_["total_prompt_tokens"] / (
            df_["total_completion_tokens"] + 1
        )  # +1 to avoid division by zero

        # Time efficiency
        df_["time_per_call"] = (
            df_["total_tool_execution_time"] / df_["total_tool_calls"]
        )
        df_["benchmark_time_per_call"] = (
            df_["total_benchmark_time"] / df_["total_tool_calls"]
        )
        df_["execution_overhead"] = (
            df_["total_benchmark_time"] - df_["total_tool_execution_time"]
        ) / df_["total_benchmark_time"]

        # Success-normalized metrics
        df_["calls_per_success"] = df_["total_tool_calls"] / (
            df_["average_score"] + 0.01
        )
        df_["tokens_per_success"] = df_["total_overall_tokens"] / (
            df_["average_score"] + 0.01
        )
        df_["time_per_success"] = df_["total_benchmark_time"] / (
            df_["average_score"] + 0.01
        )

        # Pass rate improvements
        df_["pass@1_to_pass@5_improvement"] = df_["pass@5"] - df_["pass@1"]
        df_["pass^1_to_pass^5_degradation"] = df_["pass^1"] - df_["pass^5"]

        # Binary indicators
        df_["has_failures"] = (df_["failed_tool_calls"] > 0).astype(int)
        df_["perfect_success"] = (df_["average_score"] == 1.0).astype(int)
        df_["zero_success"] = (df_["average_score"] == 0.0).astype(int)

        return df_

    def get_summary_stats(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Get summary statistics for environment data

        Args:
            df: DataFrame to summarize (if None, uses self.df)

        Returns:
            dictionary with summary statistics
        """
        if df is None:
            df = self.df

        return {
            "total_configs": len(df),
            "unique_environments": df["environment"].nunique(),
            "unique_models": df["model"].nunique(),
            "unique_agent_types": df["agent_type"].nunique(),
            "unique_verbosity": df["tool_verbosity"].nunique(),
            "unique_levels": df["level"].nunique(),
            "environments": df["environment"].unique().tolist(),
            "models": df["model"].unique().tolist(),
            "agent_types": df["agent_type"].unique().tolist(),
            "verbosity_levels": df["tool_verbosity"].unique().tolist(),
            "difficulty_levels": sorted(df["level"].unique().tolist()),
            "avg_success_rate": df["average_score"].mean(),
            "avg_total_calls": df["total_tool_calls"].mean(),
            "avg_failed_calls": df["failed_tool_calls"].mean(),
            "avg_tokens": df["total_overall_tokens"].mean(),
        }

    def get_environment_difficulty(
        self, df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Calculate difficulty metrics for each environment

        Args:
            df: DataFrame to analyze (if None, uses self.df)

        Returns:
            DataFrame with difficulty metrics per environment
        """
        if df is None:
            df = self.df

        difficulty = (
            df.groupby("environment")
            .agg(
                {
                    "average_score": ["mean", "std", "min", "max"],
                    "total_tool_calls": "mean",
                    "failed_tool_calls": "mean",
                    "total_benchmark_time": "mean",
                }
            )
            .reset_index()
        )

        difficulty.columns = [
            "environment",
            "avg_success_rate",
            "success_rate_std",
            "min_success_rate",
            "max_success_rate",
            "avg_tool_calls",
            "avg_failed_calls",
            "avg_benchmark_time",
        ]

        # Add difficulty rank (lower success = harder)
        difficulty["difficulty_rank"] = difficulty["avg_success_rate"].rank()
        difficulty = difficulty.sort_values("avg_success_rate")

        return difficulty

    def pivot_by_model_agent(
        self, metric: str = "average_score", df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Create pivot table showing metric by model and agent type

        Args:
            metric: Metric to pivot
            df: DataFrame to pivot (if None, uses self.df)

        Returns:
            Pivot table
        """
        if df is None:
            df = self.df

        return df.pivot_table(
            values=metric,
            index="environment",
            columns=["model", "agent_type"],
            aggfunc="mean",
        )

    def compare_configurations(
        self,
        group_by: str = "agent_type",
        metric: str = "average_score",
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compare different configurations

        Args:
            group_by: Column to group by
            metric: Metric to compare
            df: DataFrame to analyze

        Returns:
            Comparison DataFrame
        """
        if df is None:
            df = self.df

        return (
            df.groupby(group_by)
            .agg(
                {
                    metric: ["mean", "std", "min", "max", "count"],
                    "total_tool_calls": "mean",
                    "failed_tool_calls": "mean",
                    "total_overall_tokens": "mean",
                }
            )
            .reset_index()
        )
