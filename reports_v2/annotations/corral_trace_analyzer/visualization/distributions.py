"""
Visualization for distributions and group comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import VIZ_CONFIG


class DistributionVisualizer:
    """Visualize distributions and group comparisons"""

    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize visualizer

        Args:
            features_df: DataFrame with features
        """
        self.features_df = features_df
        sns.set_style(VIZ_CONFIG["style"])
        sns.set_context(VIZ_CONFIG["context"])

    def plot_score_distribution(
        self,
        score_col: str = "score",
        group_col: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot score distribution

        Args:
            score_col: Score column name
            group_col: Optional column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if group_col:
            for label in self.features_df[group_col].unique():
                subset = self.features_df[self.features_df[group_col] == label]
                ax.hist(
                    subset[score_col].dropna(),
                    bins=20,
                    alpha=0.5,
                    label=label,
                    edgecolor="black",
                )
        else:
            ax.hist(
                self.features_df[score_col].dropna(),
                bins=20,
                alpha=0.7,
                edgecolor="black",
            )

        ax.set_xlabel(score_col.capitalize(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Distribution of {score_col.capitalize()}", fontsize=14, fontweight="bold"
        )

        if group_col:
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_violin_comparison(
        self,
        value_col: str,
        group_col: str,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot violin plot for group comparison

        Args:
            value_col: Value column to compare
            group_col: Column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.violinplot(
            data=self.features_df,
            x=group_col,
            y=value_col,
            ax=ax,
            palette=VIZ_CONFIG["palette"],
        )

        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(value_col.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"Distribution of {value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_box_comparison(
        self,
        value_col: str,
        group_col: str,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot box plot for group comparison

        Args:
            value_col: Value column to compare
            group_col: Column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=self.features_df,
            x=group_col,
            y=value_col,
            ax=ax,
            palette=VIZ_CONFIG["palette"],
        )

        # Add mean markers
        means = self.features_df.groupby(group_col)[value_col].mean()
        positions = range(len(means))
        ax.scatter(
            positions, means, color="red", s=100, zorder=3, marker="D", label="Mean"
        )

        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(value_col.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"{value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_success_rate_by_group(
        self,
        group_col: str,
        success_col: str = "success",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot success rate by group

        Args:
            group_col: Column to group by
            success_col: Success indicator column
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        success_rates = self.features_df.groupby(group_col)[success_col].agg(
            ["mean", "count"]
        )
        success_rates["success_rate"] = success_rates["mean"] * 100

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(
            range(len(success_rates)),
            success_rates["success_rate"],
            alpha=0.7,
            edgecolor="black",
        )

        # Color bars by success rate
        for i, bar in enumerate(bars):
            if success_rates.iloc[i]["success_rate"] >= 80:
                bar.set_color("green")
            elif success_rates.iloc[i]["success_rate"] >= 50:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.set_xticks(range(len(success_rates)))
        ax.set_xticklabels(success_rates.index, rotation=45, ha="right")
        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            f"Success Rate by {group_col.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, 100)

        # Add count labels
        for i, (idx, row) in enumerate(success_rates.iterrows()):
            ax.text(
                i,
                row["success_rate"] + 2,
                f"n={int(row['count'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_feature_distribution_grid(
        self,
        feature_cols: list[str],
        n_cols: int = 3,
        figsize: tuple[int, int] = (15, 10),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot grid of feature distributions

        Args:
            feature_cols: list of features to plot
            n_cols: Number of columns in grid
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_features = len(feature_cols)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(feature_cols):
            if feature not in self.features_df.columns:
                continue

            data = self.features_df[feature].dropna()

            if len(data) > 0:
                axes[i].hist(data, bins=20, alpha=0.7, edgecolor="black")
                axes[i].set_title(feature.replace("_", " ").title(), fontsize=10)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("Frequency", fontsize=9)

                # Add mean and median lines
                mean_val = data.mean()
                median_val = data.median()
                axes[i].axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                )
                axes[i].axvline(
                    median_val,
                    color="blue",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.2f}",
                )
                axes[i].legend(fontsize=8)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis("off")

        fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_two_way_interaction(
        self,
        value_col: str,
        factor1: str,
        factor2: str,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot two-way interaction effect

        Args:
            value_col: Value column
            factor1: First factor
            factor2: Second factor
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate means for each combination
        grouped = (
            self.features_df.groupby([factor1, factor2])[value_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Plot lines for each level of factor2
        for f2_level in grouped[factor2].unique():
            subset = grouped[grouped[factor2] == f2_level]

            ax.plot(
                subset[factor1],
                subset["mean"],
                marker="o",
                markersize=8,
                linewidth=2,
                label=f"{factor2}={f2_level}",
            )

            # Add error bars
            ax.errorbar(
                subset[factor1],
                subset["mean"],
                yerr=subset["std"],
                fmt="none",
                alpha=0.3,
            )

        ax.set_xlabel(factor1.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(f"Mean {value_col.replace('_', ' ').title()}", fontsize=12)
        ax.set_title(
            f"Interaction: {factor1} × {factor2}", fontsize=14, fontweight="bold"
        )
        ax.legend(title=factor2.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_marker_distribution(
        self,
        group_col: str | None = None,
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot distribution of marker types

        Args:
            group_col: Optional column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        marker_cols = [
            "positive_marker_count",
            "negative_marker_count",
            "neutral_marker_count",
        ]

        marker_cols = [c for c in marker_cols if c in self.features_df.columns]

        if group_col:
            fig, ax = plt.subplots(figsize=figsize)

            x = np.arange(len(self.features_df[group_col].unique()))
            width = 0.25

            for i, marker_col in enumerate(marker_cols):
                means = self.features_df.groupby(group_col)[marker_col].mean()
                ax.bar(
                    x + i * width,
                    means,
                    width,
                    label=marker_col.replace("_", " ").title(),
                    alpha=0.7,
                )

            ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
            ax.set_ylabel("Average Count", fontsize=12)
            ax.set_title("Marker Distribution by Group", fontsize=14, fontweight="bold")
            ax.set_xticks(x + width)
            ax.set_xticklabels(
                self.features_df[group_col].unique(), rotation=45, ha="right"
            )
            ax.legend()

            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=figsize)

            marker_totals = [self.features_df[col].sum() for col in marker_cols]
            marker_labels = [col.replace("_", " ").title() for col in marker_cols]

            ax.bar(marker_labels, marker_totals, alpha=0.7, edgecolor="black")
            ax.set_ylabel("Total Count", fontsize=12)
            ax.set_title("Overall Marker Distribution", fontsize=14, fontweight="bold")

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig
