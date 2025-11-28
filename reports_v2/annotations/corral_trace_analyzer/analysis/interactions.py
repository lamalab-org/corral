"""
Interaction effects analysis (ANOVA, multi-way effects)
"""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from corral_trace_analyzer.config import ALPHA


class InteractionAnalyzer:
    """Analyze interaction effects between categorical variables"""

    def __init__(self, features_df: pd.DataFrame, target_col: str = "score"):
        """
        Initialize interaction analyzer

        Args:
            features_df: DataFrame with features
            target_col: Target column for analysis
        """
        self.features_df = features_df
        self.target_col = target_col

    def one_way_anova(self, group_col: str) -> dict[str, Any]:
        """
        Perform one-way ANOVA

        Args:
            group_col: Column to group by

        Returns:
            dictionary with ANOVA results
        """
        if group_col not in self.features_df.columns:
            return {}

        groups = (
            self.features_df.groupby(group_col)[self.target_col].apply(list).to_dict()
        )

        # Remove groups with too few samples
        groups = {k: v for k, v in groups.items() if len(v) >= 2}

        if len(groups) < 2:
            return {"error": "Not enough groups for ANOVA"}

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups.values())

        # Calculate effect size (eta squared)
        grand_mean = self.features_df[self.target_col].mean()
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
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < ALPHA,
            "eta_squared": eta_squared,
            "n_groups": len(groups),
            "group_stats": group_stats,
        }

    def two_way_anova(self, factor1: str, factor2: str) -> dict[str, Any]:
        """
        Perform two-way ANOVA to test for interaction effects

        Args:
            factor1: First factor column
            factor2: Second factor column

        Returns:
            dictionary with ANOVA results including interaction effect
        """
        if (
            factor1 not in self.features_df.columns
            or factor2 not in self.features_df.columns
        ):
            return {}

        # Create interaction term
        df_ = self.features_df[[factor1, factor2, self.target_col]].copy()
        df_ = df_.dropna()

        if len(df_) < 10:  # Need reasonable sample size
            return {"error": "Insufficient data for two-way ANOVA"}

        # Get unique levels
        levels1 = df_[factor1].unique()
        levels2 = df_[factor2].unique()

        # Main effect of factor1
        groups1 = [
            df_[df_[factor1] == level][self.target_col].to_numpy() for level in levels1
        ]
        groups1 = [g for g in groups1 if len(g) >= 2]

        if len(groups1) >= 2:
            f1_stat, f1_pval = stats.f_oneway(*groups1)
        else:
            f1_stat, f1_pval = np.nan, np.nan

        # Main effect of factor2
        groups2 = [
            df_[df_[factor2] == level][self.target_col].to_numpy() for level in levels2
        ]
        groups2 = [g for g in groups2 if len(g) >= 2]

        if len(groups2) >= 2:
            f2_stat, f2_pval = stats.f_oneway(*groups2)
        else:
            f2_stat, f2_pval = np.nan, np.nan

        # Interaction effect (simplified approach)
        # For each combination, get mean score
        interaction_means = {}
        for l1 in levels1:
            for l2 in levels2:
                combo_data = df_[(df_[factor1] == l1) & (df_[factor2] == l2)][
                    self.target_col
                ]
                if len(combo_data) > 0:
                    interaction_means[(l1, l2)] = combo_data.mean()

        # Calculate interaction effect strength
        # Check if the effect of factor1 depends on level of factor2
        interaction_detected = False
        if len(levels1) >= 2 and len(levels2) >= 2:
            # Simple test: compare score differences
            diffs_by_f2 = []
            for l2 in levels2:
                means_at_l2 = [
                    interaction_means.get((l1, l2), np.nan) for l1 in levels1
                ]
                if not any(np.isnan(means_at_l2)):
                    diffs_by_f2.append(max(means_at_l2) - min(means_at_l2))

            if len(diffs_by_f2) >= 2:
                # If differences vary substantially, there's interaction
                interaction_variance = np.var(diffs_by_f2)
                interaction_detected = interaction_variance > 0.01

        return {
            "factor1": factor1,
            "factor2": factor2,
            "factor1_f_statistic": f1_stat,
            "factor1_p_value": f1_pval,
            "factor1_significant": f1_pval < ALPHA if not np.isnan(f1_pval) else False,
            "factor2_f_statistic": f2_stat,
            "factor2_p_value": f2_pval,
            "factor2_significant": f2_pval < ALPHA if not np.isnan(f2_pval) else False,
            "interaction_detected": interaction_detected,
            "interaction_means": interaction_means,
            "n_samples": len(df_),
        }

    def analyze_model_agent_interaction(self) -> dict[str, Any]:
        """
        Analyze interaction between model and agent_type

        Returns:
            dictionary with interaction analysis results
        """
        return self.two_way_anova("model", "agent_type")

    def analyze_model_environment_interaction(self) -> dict[str, Any]:
        """
        Analyze interaction between model and environment

        Returns:
            dictionary with interaction analysis results
        """
        return self.two_way_anova("model", "environment")

    def analyze_agent_environment_interaction(self) -> dict[str, Any]:
        """
        Analyze interaction between agent_type and environment

        Returns:
            dictionary with interaction analysis results
        """
        return self.two_way_anova("agent_type", "environment")

    def three_way_interaction(
        self, factor1: str, factor2: str, factor3: str
    ) -> dict[str, Any]:
        """
        Analyze three-way interaction effects

        Args:
            factor1: First factor
            factor2: Second factor
            factor3: Third factor

        Returns:
            dictionary with three-way interaction results
        """
        df = self.features_df[[factor1, factor2, factor3, self.target_col]].copy()
        df = df.dropna()

        if len(df) < 20:
            return {"error": "Insufficient data for three-way analysis"}

        # Get means for each combination
        combination_means = {}

        for l1 in df[factor1].unique():
            for l2 in df[factor2].unique():
                for l3 in df[factor3].unique():
                    combo_data = df[
                        (df[factor1] == l1) & (df[factor2] == l2) & (df[factor3] == l3)
                    ][self.target_col]

                    if len(combo_data) > 0:
                        combination_means[(l1, l2, l3)] = {
                            "mean": combo_data.mean(),
                            "std": combo_data.std(),
                            "n": len(combo_data),
                        }

        # Find best and worst combinations
        if combination_means:
            best_combo = max(combination_means.items(), key=lambda x: x[1]["mean"])
            worst_combo = min(combination_means.items(), key=lambda x: x[1]["mean"])
        else:
            best_combo = worst_combo = None

        return {
            "factor1": factor1,
            "factor2": factor2,
            "factor3": factor3,
            "n_combinations": len(combination_means),
            "combination_means": combination_means,
            "best_combination": {
                "combination": best_combo[0] if best_combo else None,
                "stats": best_combo[1] if best_combo else None,
            },
            "worst_combination": {
                "combination": worst_combo[0] if worst_combo else None,
                "stats": worst_combo[1] if worst_combo else None,
            },
            "n_samples": len(df),
        }

    def analyze_all_main_effects(self) -> dict[str, dict[str, Any]]:
        """
        Analyze main effects for all categorical variables

        Returns:
            dictionary with ANOVA results for each categorical variable
        """
        logger.info("Analyzing main effects...")

        categorical_cols = ["model", "environment", "agent_type", "level"]
        categorical_cols = [
            c for c in categorical_cols if c in self.features_df.columns
        ]

        results = {}

        for col in categorical_cols:
            logger.info(f"  Analyzing {col}...")
            results[col] = self.one_way_anova(col)

        return results

    def analyze_all_two_way_interactions(self) -> dict[str, dict[str, Any]]:
        """
        Analyze all two-way interaction effects

        Returns:
            dictionary with two-way ANOVA results
        """
        logger.info("Analyzing two-way interactions...")

        categorical_cols = ["model", "environment", "agent_type"]
        categorical_cols = [
            c for c in categorical_cols if c in self.features_df.columns
        ]

        results = {}

        for factor1, factor2 in combinations(categorical_cols, 2):
            key = f"{factor1}_x_{factor2}"
            logger.info(f"  Analyzing {key}...")
            results[key] = self.two_way_anova(factor1, factor2)

        return results

    def analyze_model_agent_environment(self) -> dict[str, Any]:
        """
        Analyze three-way interaction: model , agent_type , environment

        Returns:
            dictionary with three-way interaction results
        """
        return self.three_way_interaction("model", "agent_type", "environment")

    def run_all_interaction_analyses(self) -> dict[str, Any]:
        """
        Run all interaction analyses

        Returns:
            dictionary with all interaction analysis results
        """
        return {
            "main_effects": self.analyze_all_main_effects(),
            "two_way_interactions": self.analyze_all_two_way_interactions(),
            "three_way_interaction": self.analyze_model_agent_environment(),
        }

    def compare_groups(
        self, group_col: str, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compare groups across multiple features

        Args:
            group_col: Column to group by
            feature_cols: Features to compare across groups

        Returns:
            DataFrame with group comparisons
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        results = []

        for feature in feature_cols:
            if feature not in self.features_df.columns:
                continue

            groups = self.features_df.groupby(group_col)[feature]

            for group_name, group_data in groups:
                results.append(
                    {
                        "group": group_name,
                        "feature": feature,
                        "mean": group_data.mean(),
                        "std": group_data.std(),
                        "median": group_data.median(),
                        "min": group_data.min(),
                        "max": group_data.max(),
                        "n": len(group_data),
                    }
                )

        return pd.DataFrame(results)
