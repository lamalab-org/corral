"""
Correlation analysis for trace features
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from corral_trace_analyzer.config import ALPHA


class CorrelationAnalyzer:
    """Analyze correlations between features and outcomes"""

    def __init__(self, features_df: pd.DataFrame, target_col: str = "score"):
        """
        Initialize correlation analyzer

        Args:
            features_df: DataFrame with features
            target_col: Target column for correlation analysis
        """
        self.features_df = features_df
        self.target_col = target_col
        self.correlation_results = {}

    def compute_pearson_correlations(
        self, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compute Pearson correlations with the target variable

        Args:
            feature_cols: list of feature columns to correlate. If None, uses all numeric columns.

        Returns:
            DataFrame with correlations, p-values, and significance
        """
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if self.target_col in feature_cols:
                feature_cols.remove(self.target_col)

        results = []

        target_values = self.features_df[self.target_col].values

        for col in feature_cols:
            if col not in self.features_df.columns:
                continue

            feature_values = self.features_df[col].values

            # Remove NaN values
            mask = ~(np.isnan(target_values) | np.isnan(feature_values))
            if mask.sum() < 3:  # Need at least 3 points
                continue

            clean_target = target_values[mask]
            clean_feature = feature_values[mask]

            # Compute Pearson correlation
            corr, pval = stats.pearsonr(clean_target, clean_feature)

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

        self.correlation_results["pearson"] = results_df
        return results_df

    def compute_spearman_correlations(
        self, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compute Spearman correlations with the target variable (for non-linear relationships)

        Args:
            feature_cols: list of feature columns to correlate. If None, uses all numeric columns.

        Returns:
            DataFrame with correlations, p-values, and significance
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if self.target_col in feature_cols:
                feature_cols.remove(self.target_col)

        results = []

        target_values = self.features_df[self.target_col].values

        for col in feature_cols:
            if col not in self.features_df.columns:
                continue

            feature_values = self.features_df[col].values

            # Remove NaN values
            mask = ~(np.isnan(target_values) | np.isnan(feature_values))
            if mask.sum() < 3:
                continue

            clean_target = target_values[mask]
            clean_feature = feature_values[mask]

            # Compute Spearman correlation
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

        self.correlation_results["spearman"] = results_df
        return results_df

    def analyze_marker_score_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between marker counts and score (Tier 1 analysis)

        Returns:
            DataFrame with marker correlation results
        """
        marker_features = [
            "positive_marker_count",
            "negative_marker_count",
            "marker_balance",
            "marker_balance_ratio",
            "positive_marker_ratio",
            "negative_marker_ratio",
            "early_negative_rate",
        ]

        # Filter to features that exist
        marker_features = [f for f in marker_features if f in self.features_df.columns]

        return self.compute_pearson_correlations(marker_features)

    def analyze_error_score_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between error metrics and score (Tier 1 analysis)

        Returns:
            DataFrame with error correlation results
        """
        error_features = [
            "failed_calls",
            "error_rate",
            "total_errors",
            "execution_errors",
            "error_burstiness",
            "max_consecutive_failures",
        ]

        error_features = [f for f in error_features if f in self.features_df.columns]

        return self.compute_pearson_correlations(error_features)

    def analyze_recovery_score_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between recovery behavior and score (Tier 2 analysis)

        Returns:
            DataFrame with recovery correlation results
        """
        recovery_features = [
            "backtrack_trigger_count",
            "validation_attempt_count",
            "backtrack_to_total_ratio",
            "validation_to_total_ratio",
            "validation_after_error_rate",
            "backtrack_after_error_rate",
            "recovery_marker_count",
            "recovery_marker_ratio",
        ]

        recovery_features = [
            f for f in recovery_features if f in self.features_df.columns
        ]

        return self.compute_pearson_correlations(recovery_features)

    def analyze_efficiency_score_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between efficiency metrics and score (Tier 1 analysis)

        Returns:
            DataFrame with efficiency correlation results
        """
        efficiency_features = [
            "total_calls",
            "total_tokens",
            "tool_execution_duration",
            "step_count_total",
        ]

        efficiency_features = [
            f for f in efficiency_features if f in self.features_df.columns
        ]

        return self.compute_pearson_correlations(efficiency_features)

    def analyze_qa_correlations(self) -> dict[str, pd.DataFrame]:
        """
        Analyze correlations between QA score and various metrics (Tier 1 analysis)

        Returns:
            dictionary with different correlation analyses
        """
        if "qa_score" not in self.features_df.columns:
            return {}

        # Temporarily change target to qa_score
        original_target = self.target_col
        self.target_col = "qa_score"

        results = {
            "qa_vs_score": self._analyze_two_variables("qa_score", original_target),
            "qa_vs_negative_markers": self.compute_pearson_correlations(
                ["negative_marker_count"]
            ),
            "qa_vs_error_rate": self.compute_pearson_correlations(["error_rate"]),
        }

        # Restore original target
        self.target_col = original_target

        return results

    def analyze_planning_score_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between planning/reasoning and score (Tier 2 analysis)

        Returns:
            DataFrame with planning correlation results
        """
        planning_features = [
            "planning_statement_count",
            "reasoning_statement_count",
            "planning_to_total_ratio",
            "reasoning_to_total_ratio",
            "steps_to_first_planning",
            "steps_to_first_reasoning",
        ]

        planning_features = [
            f for f in planning_features if f in self.features_df.columns
        ]

        return self.compute_pearson_correlations(planning_features)

    def analyze_tool_usage_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between tool usage patterns and score (Tier 2 analysis)

        Returns:
            DataFrame with tool usage correlation results
        """
        tool_features = [
            "tools_used_count",
            "tool_diversity_entropy",
            "unnecessary_tool_use_count",
            "retry_rate",
            "unique_tool_calls_ratio",
            "tool_repetition_rate",
        ]

        tool_features = [f for f in tool_features if f in self.features_df.columns]

        return self.compute_pearson_correlations(tool_features)

    def run_all_tier1_analyses(self) -> dict[str, pd.DataFrame]:
        """
        Run all Tier 1 (high priority) correlation analyses

        Returns:
            dictionary with all Tier 1 analysis results
        """
        logger.info("Running Tier 1 correlation analyses...")

        results = {
            "markers_vs_score": self.analyze_marker_score_correlations(),
            "errors_vs_score": self.analyze_error_score_correlations(),
            "efficiency_vs_score": self.analyze_efficiency_score_correlations(),
            "qa_analyses": self.analyze_qa_correlations(),
        }

        return results

    def run_all_tier2_analyses(self) -> dict[str, pd.DataFrame]:
        """
        Run all Tier 2 correlation analyses

        Returns:
            dictionary with all Tier 2 analysis results
        """
        logger.info("Running Tier 2 correlation analyses...")

        results = {
            "recovery_vs_score": self.analyze_recovery_score_correlations(),
            "planning_vs_score": self.analyze_planning_score_correlations(),
            "tool_usage_vs_score": self.analyze_tool_usage_correlations(),
        }

        return results

    def run_all_analyses(self) -> dict[str, Any]:
        """
        Run all correlation analyses

        Returns:
            dictionary with all analysis results
        """
        tier1 = self.run_all_tier1_analyses()
        tier2 = self.run_all_tier2_analyses()

        return {"tier1": tier1, "tier2": tier2}

    def get_top_correlations(
        self, n: int = 10, min_abs_corr: float = 0.0
    ) -> pd.DataFrame:
        """
        Get top N correlations across all analyses

        Args:
            n: Number of top correlations to return
            min_abs_corr: Minimum absolute correlation threshold

        Returns:
            DataFrame with top correlations
        """
        # Compute all correlations if not done yet
        if "pearson" not in self.correlation_results:
            self.compute_pearson_correlations()

        df = self.correlation_results["pearson"]
        df = df[df["abs_correlation"] >= min_abs_corr]
        return df.head(n)

    def _analyze_two_variables(self, var1: str, var2: str) -> dict[str, float]:
        """Helper to analyze correlation between two variables"""
        if var1 not in self.features_df.columns or var2 not in self.features_df.columns:
            return {}

        v1 = self.features_df[var1].values
        v2 = self.features_df[var2].values

        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() < 3:
            return {}

        corr, pval = stats.pearsonr(v1[mask], v2[mask])

        return {
            "correlation": corr,
            "p_value": pval,
            "significant": pval < ALPHA,
            "n_samples": mask.sum(),
        }
