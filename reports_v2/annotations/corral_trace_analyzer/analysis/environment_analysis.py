"""
Analysis for environment-level aggregated data
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import ALPHA, ENVIRONMENT_TARGET_METRIC


class EnvironmentAnalyzer:
    """Analyze environment-level aggregated results"""

    def __init__(self, env_df: pd.DataFrame, target_metric: str = None):
        """
        Initialize environment analyzer

        Args:
            env_df: DataFrame with environment-level data
            target_metric: Target metric to analyze (e.g., "average_score", "pass@1")
        """
        self.env_df = env_df
        self.target_metric = target_metric or ENVIRONMENT_TARGET_METRIC

        if self.target_metric not in env_df.columns:
            raise ValueError(f"Target metric '{self.target_metric}' not found in data")

    def compute_correlations(
        self, feature_cols: list[str] | None = None, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute correlations with target metric

        Args:
            feature_cols: list of feature columns to correlate
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            DataFrame with correlation results
        """
        if feature_cols is None:
            # Auto-select numeric columns
            feature_cols = self.env_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Remove target and identifier columns
            exclude_cols = [
                self.target_metric,
                "level",
            ]
            feature_cols = [c for c in feature_cols if c not in exclude_cols]

        results = []
        target_values = self.env_df[self.target_metric].values

        for col in feature_cols:
            if col not in self.env_df.columns:
                continue

            feature_values = self.env_df[col].values

            # Remove NaN
            mask = ~(np.isnan(target_values) | np.isnan(feature_values))
            if mask.sum() < 3:
                continue

            clean_target = target_values[mask]
            clean_feature = feature_values[mask]

            # Compute correlation
            if method == "pearson":
                corr, pval = stats.pearsonr(clean_target, clean_feature)
            else:
                corr, pval = stats.spearmanr(clean_target, clean_feature)

            results.append(
                {
                    "feature": col,
                    "correlation": corr,
                    "p_value": pval,
                    "significant": pval < ALPHA,
                    "abs_correlation": abs(corr),
                    "n_samples": mask.sum(),
                }
            )

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values("abs_correlation", ascending=False)

        return results_df

    def analyze_tool_calls_impact(self) -> pd.DataFrame:
        """
        Analyze impact of tool call metrics on success

        Returns:
            Correlation results for tool call features
        """
        tool_features = [
            "total_tool_calls",
            "successful_tool_calls",
            "failed_tool_calls",
            "failed_call_rate",
            "success_call_rate",
        ]

        tool_features = [f for f in tool_features if f in self.env_df.columns]

        return self.compute_correlations(tool_features)

    def analyze_token_efficiency(self) -> pd.DataFrame:
        """
        Analyze relationship between token usage and success

        Returns:
            Correlation results for token features
        """
        token_features = [
            "total_overall_tokens",
            "total_prompt_tokens",
            "total_completion_tokens",
            "tokens_per_call",
            "prompt_to_completion_ratio",
            "tokens_per_success",
        ]

        token_features = [f for f in token_features if f in self.env_df.columns]

        return self.compute_correlations(token_features)

    def analyze_time_efficiency(self) -> pd.DataFrame:
        """
        Analyze relationship between execution time and success

        Returns:
            Correlation results for time features
        """
        time_features = [
            "total_tool_execution_time",
            "total_benchmark_time",
            "time_per_call",
            "benchmark_time_per_call",
            "execution_overhead",
            "time_per_success",
        ]

        time_features = [f for f in time_features if f in self.env_df.columns]

        return self.compute_correlations(time_features)

    def analyze_pass_rate_patterns(self) -> pd.DataFrame:
        """
        Analyze pass@k and pass^k patterns

        Returns:
            Correlation results for pass metrics
        """
        pass_features = [
            "pass@1",
            "pass@2",
            "pass@3",
            "pass@4",
            "pass@5",
            "pass^1",
            "pass^2",
            "pass^3",
            "pass^4",
            "pass^5",
            "pass@1_to_pass@5_improvement",
            "pass^1_to_pass^5_degradation",
        ]

        pass_features = [f for f in pass_features if f in self.env_df.columns]

        return self.compute_correlations(pass_features)

    def compare_agent_types(self) -> dict[str, Any]:
        """
        Compare performance across agent types

        Returns:
            dictionary with comparison results
        """
        if "agent_type" not in self.env_df.columns:
            return {}

        comparison = {}

        for agent_type in self.env_df["agent_type"].unique():
            agent_data = self.env_df[self.env_df["agent_type"] == agent_type]

            comparison[agent_type] = {
                "mean_score": agent_data[self.target_metric].mean(),
                "std_score": agent_data[self.target_metric].std(),
                "median_score": agent_data[self.target_metric].median(),
                "n_configs": len(agent_data),
                "mean_tool_calls": agent_data["total_tool_calls"].mean(),
                "mean_failed_calls": agent_data["failed_tool_calls"].mean(),
                "mean_tokens": agent_data["total_overall_tokens"].mean(),
            }

        return comparison

    def compare_models(self) -> dict[str, Any]:
        """
        Compare performance across models

        Returns:
            dictionary with comparison results
        """
        if "model" not in self.env_df.columns:
            return {}

        comparison = {}

        for model in self.env_df["model"].unique():
            model_data = self.env_df[self.env_df["model"] == model]

            comparison[model] = {
                "mean_score": model_data[self.target_metric].mean(),
                "std_score": model_data[self.target_metric].std(),
                "median_score": model_data[self.target_metric].median(),
                "n_configs": len(model_data),
                "mean_tool_calls": model_data["total_tool_calls"].mean(),
                "mean_failed_calls": model_data["failed_tool_calls"].mean(),
                "mean_tokens": model_data["total_overall_tokens"].mean(),
            }

        return comparison

    def compare_verbosity_levels(self) -> dict[str, Any]:
        """
        Compare performance across verbosity levels

        Returns:
            dictionary with comparison results
        """
        if "tool_verbosity" not in self.env_df.columns:
            return {}

        comparison = {}

        for verbosity in self.env_df["tool_verbosity"].unique():
            verb_data = self.env_df[self.env_df["tool_verbosity"] == verbosity]

            comparison[verbosity] = {
                "mean_score": verb_data[self.target_metric].mean(),
                "std_score": verb_data[self.target_metric].std(),
                "median_score": verb_data[self.target_metric].median(),
                "n_configs": len(verb_data),
                "mean_tool_calls": verb_data["total_tool_calls"].mean(),
                "mean_tokens": verb_data["total_overall_tokens"].mean(),
                "mean_prompt_tokens": verb_data["total_prompt_tokens"].mean(),
                "mean_completion_tokens": verb_data["total_completion_tokens"].mean(),
            }

        return comparison

    def anova_by_factor(self, factor: str) -> dict[str, Any]:
        """
        Perform ANOVA to test if factor significantly affects target metric

        Args:
            factor: Column name to test (e.g., 'agent_type', 'model', 'environment')

        Returns:
            dictionary with ANOVA results
        """
        if factor not in self.env_df.columns:
            return {"error": f"Factor '{factor}' not found in data"}

        groups = self.env_df.groupby(factor)[self.target_metric].apply(list).to_dict()

        # Remove groups with too few samples
        groups = {k: v for k, v in groups.items() if len(v) >= 2}

        if len(groups) < 2:
            return {"error": "Not enough groups for ANOVA"}

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups.values())

        # Calculate effect size (eta squared)
        grand_mean = self.env_df[self.target_metric].mean()
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values()
        )
        ss_total = sum((x - grand_mean) ** 2 for g in groups.values() for x in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Group statistics
        group_stats = {}
        for group_name, group_values in groups.items():
            group_stats[group_name] = {
                "mean": np.mean(group_values),
                "std": np.std(group_values),
                "n": len(group_values),
            }

        return {
            "factor": factor,
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < ALPHA,
            "eta_squared": eta_squared,
            "n_groups": len(groups),
            "group_stats": group_stats,
        }

    def run_all_analyses(self) -> dict[str, Any]:
        """
        Run all environment-level analyses

        Returns:
            dictionary with all analysis results
        """
        logger.info(
            f"Running environment-level analyses (target: {self.target_metric})..."
        )

        results = {}

        logger.info("  Computing correlations...")
        results["all_correlations"] = self.compute_correlations()

        logger.info("  Analyzing tool call impact...")
        results["tool_calls"] = self.analyze_tool_calls_impact()

        logger.info("  Analyzing token efficiency...")
        results["tokens"] = self.analyze_token_efficiency()

        logger.info("  Analyzing time efficiency...")
        results["time"] = self.analyze_time_efficiency()

        logger.info("  Analyzing pass rate patterns...")
        results["pass_rates"] = self.analyze_pass_rate_patterns()

        logger.info("  Comparing agent types...")
        results["agent_comparison"] = self.compare_agent_types()

        logger.info("  Comparing models...")
        results["model_comparison"] = self.compare_models()

        logger.info("  Comparing verbosity levels...")
        results["verbosity_comparison"] = self.compare_verbosity_levels()

        logger.info("  Running ANOVA tests...")
        results["anova_agent"] = self.anova_by_factor("agent_type")
        results["anova_model"] = self.anova_by_factor("model")
        results["anova_environment"] = self.anova_by_factor("environment")
        results["anova_verbosity"] = self.anova_by_factor("tool_verbosity")

        return results

    def analyze_multiple_targets(
        self, target_metrics: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze correlations for multiple target metrics

        Args:
            target_metrics: list of target metrics to analyze

        Returns:
            dictionary mapping target metric to correlation results
        """
        results = {}

        for metric in target_metrics:
            if metric not in self.env_df.columns:
                logger.info(f"Warning: Metric '{metric}' not found, skipping...")
                continue

            logger.info(f"\nAnalyzing target metric: {metric}")
            original_target = self.target_metric
            self.target_metric = metric

            results[metric] = self.compute_correlations()

            self.target_metric = original_target

        return results
