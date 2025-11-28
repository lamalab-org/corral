"""
Statistical tests and utilities
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..config import ALPHA


class StatisticalTests:
    """Various statistical tests for trace analysis"""

    @staticmethod
    def t_test_groups(
        df: pd.DataFrame, group_col: str, value_col: str, group1: Any, group2: Any
    ) -> dict[str, Any]:
        """
        Perform t-test between two groups

        Args:
            df: DataFrame
            group_col: Column containing group labels
            value_col: Column containing values to compare
            group1: First group label
            group2: Second group label

        Returns:
            dictionary with t-test results
        """
        data1 = df[df[group_col] == group1][value_col].dropna()
        data2 = df[df[group_col] == group2][value_col].dropna()

        if len(data1) < 2 or len(data2) < 2:
            return {"error": "Insufficient data for t-test"}

        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(data1) - 1) * data1.std() ** 2 + (len(data2) - 1) * data2.std() ** 2)
            / (len(data1) + len(data2) - 2)
        )
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

        return {
            "group1": group1,
            "group2": group2,
            "group1_mean": data1.mean(),
            "group2_mean": data2.mean(),
            "group1_std": data1.std(),
            "group2_std": data2.std(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < ALPHA,
            "cohens_d": cohens_d,
            "n1": len(data1),
            "n2": len(data2),
        }

    @staticmethod
    def mann_whitney_test(
        df: pd.DataFrame, group_col: str, value_col: str, group1: Any, group2: Any
    ) -> dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test)

        Args:
            df: DataFrame
            group_col: Column containing group labels
            value_col: Column containing values to compare
            group1: First group label
            group2: Second group label

        Returns:
            dictionary with test results
        """
        data1 = df[df[group_col] == group1][value_col].dropna()
        data2 = df[df[group_col] == group2][value_col].dropna()

        if len(data1) < 2 or len(data2) < 2:
            return {"error": "Insufficient data for Mann-Whitney test"}

        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")

        return {
            "group1": group1,
            "group2": group2,
            "group1_median": data1.median(),
            "group2_median": data2.median(),
            "u_statistic": u_stat,
            "p_value": p_value,
            "significant": p_value < ALPHA,
            "n1": len(data1),
            "n2": len(data2),
        }

    @staticmethod
    def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict[str, Any]:
        """
        Perform chi-square test of independence

        Args:
            df: DataFrame
            col1: First categorical column
            col2: Second categorical column

        Returns:
            dictionary with chi-square test results
        """
        contingency_table = pd.crosstab(df[col1], df[col2])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        return {
            "col1": col1,
            "col2": col2,
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < ALPHA,
            "cramers_v": cramers_v,
            "contingency_table": contingency_table.to_dict(),
        }

    @staticmethod
    def normality_test(data: np.ndarray) -> dict[str, Any]:
        """
        Test if data follows a normal distribution (Shapiro-Wilk test)

        Args:
            data: Array of values

        Returns:
            dictionary with normality test results
        """
        data = data[~np.isnan(data)]

        if len(data) < 3:
            return {"error": "Insufficient data for normality test"}

        stat, p_value = stats.shapiro(data)

        return {
            "statistic": stat,
            "p_value": p_value,
            "is_normal": p_value >= ALPHA,
            "n": len(data),
        }

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic_func=np.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """
        Calculate bootstrap confidence interval for a statistic

        Args:
            data: Array of values
            statistic_func: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            dictionary with confidence interval
        """
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return {"error": "Insufficient data for bootstrap"}

        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return {
            "statistic": statistic_func(data),
            "lower_ci": lower,
            "upper_ci": upper,
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
        }

    @staticmethod
    def partial_correlation(
        df: pd.DataFrame, x_col: str, y_col: str, control_cols: list[str]
    ) -> dict[str, Any]:
        """
        Compute partial correlation between x and y, controlling for other variables

        Args:
            df: DataFrame
            x_col: First variable
            y_col: Second variable
            control_cols: Variables to control for

        Returns:
            dictionary with partial correlation results
        """
        # Select relevant columns and drop NaNs
        cols = [x_col, y_col] + control_cols
        data = df[cols].dropna()

        if len(data) < len(cols) + 2:
            return {"error": "Insufficient data for partial correlation"}

        # Compute residuals after regressing x and y on control variables
        from sklearn.linear_model import LinearRegression

        X_control = data[control_cols].values
        x_values = data[x_col].values
        y_values = data[y_col].values

        # Regress x on controls
        model_x = LinearRegression()
        model_x.fit(X_control, x_values)
        x_resid = x_values - model_x.predict(X_control)

        # Regress y on controls
        model_y = LinearRegression()
        model_y.fit(X_control, y_values)
        y_resid = y_values - model_y.predict(X_control)

        # Compute correlation of residuals
        partial_corr, p_value = stats.pearsonr(x_resid, y_resid)

        return {
            "x": x_col,
            "y": y_col,
            "controls": control_cols,
            "partial_correlation": partial_corr,
            "p_value": p_value,
            "significant": p_value < ALPHA,
            "n": len(data),
        }
