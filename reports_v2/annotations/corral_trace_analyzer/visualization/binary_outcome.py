"""
Visualizations specifically for binary outcomes
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from corral_trace_analyzer.config import VIZ_CONFIG


class BinaryOutcomeVisualizer:
    """Visualize binary outcome analyses"""

    def __init__(self, features_df: pd.DataFrame, target_col: str = "score"):
        """
        Initialize visualizer

        Args:
            features_df: DataFrame with features
            target_col: Binary target column
        """
        self.features_df = features_df
        self.target_col = target_col
        sns.set_style(VIZ_CONFIG["style"])
        sns.set_context(VIZ_CONFIG["context"])

    def plot_feature_by_outcome(
        self,
        feature_col: str,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot feature distribution split by outcome

        Args:
            feature_col: Feature to plot
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Violin plot
        success_data = self.features_df[self.features_df[self.target_col] == 1][
            feature_col
        ].dropna()
        failure_data = self.features_df[self.features_df[self.target_col] == 0][
            feature_col
        ].dropna()

        _parts = axes[0].violinplot(
            [failure_data, success_data],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )

        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Failure", "Success"])
        axes[0].set_ylabel(feature_col.replace("_", " ").title())
        axes[0].set_title("Distribution by Outcome")
        axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(
            failure_data, bins=20, alpha=0.5, label="Failure", color="red", density=True
        )
        axes[1].hist(
            success_data,
            bins=20,
            alpha=0.5,
            label="Success",
            color="green",
            density=True,
        )
        axes[1].set_xlabel(feature_col.replace("_", " ").title())
        axes[1].set_ylabel("Density")
        axes[1].set_title("Overlapping Distributions")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"{feature_col.replace('_', ' ').title()} by Outcome",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_success_rate_by_quantile(
        self,
        feature_col: str,
        n_bins: int = 5,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot success rate by feature quantiles

        Args:
            feature_col: Feature to analyze
            n_bins: Number of bins
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        df_ = self.features_df[[feature_col, self.target_col]].dropna()

        if len(df_) < n_bins:
            raise ValueError(f"Not enough data for {n_bins} bins")

        # Create quantile bins
        df_["quantile"] = pd.qcut(
            df_[feature_col], q=n_bins, labels=False, duplicates="drop"
        )

        # Calculate success rate per quantile
        quantile_stats = (
            df_.groupby("quantile")
            .agg({feature_col: "mean", self.target_col: ["mean", "count"]})
            .reset_index()
        )

        quantile_stats.columns = ["quantile", "feature_mean", "success_rate", "count"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Success rate plot
        axes[0].bar(
            quantile_stats["quantile"],
            quantile_stats["success_rate"] * 100,
            alpha=0.7,
            edgecolor="black",
        )

        axes[0].set_xlabel(f"{feature_col.replace('_', ' ').title()} Quantile")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Success Rate by Feature Quantile")
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3)

        # Add count labels
        for _i, row in quantile_stats.iterrows():
            axes[0].text(
                row["quantile"],
                row["success_rate"] * 100 + 2,
                f"n={int(row['count'])}",
                ha="center",
                fontsize=9,
            )

        # Scatter plot showing trend
        axes[1].scatter(
            df_[feature_col], df_[self.target_col], alpha=0.3, s=20, label="Data points"
        )

        # Add quantile means
        axes[1].scatter(
            quantile_stats["feature_mean"],
            quantile_stats["success_rate"],
            color="red",
            s=100,
            zorder=5,
            marker="D",
            label="Quantile success rate",
        )

        axes[1].plot(
            quantile_stats["feature_mean"],
            quantile_stats["success_rate"],
            color="red",
            linestyle="--",
            alpha=0.5,
        )

        axes[1].set_xlabel(feature_col.replace("_", " ").title())
        axes[1].set_ylabel("Success (0/1)")
        axes[1].set_title("Success vs Feature Value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Success Rate Analysis: {feature_col.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_roc_curve_approximation(
        self,
        feature_col: str,
        figsize: tuple[int, int] = (8, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot ROC-like curve for a single feature

        Args:
            feature_col: Feature to plot
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import auc, roc_curve

        df_ = self.features_df[[feature_col, self.target_col]].dropna()

        y_true = df_[self.target_col].to_numpy()
        y_score = df_[feature_col].to_numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {feature_col.replace('_', ' ').title()}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = 15,
        figsize: tuple[int, int] = (10, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot feature importance from logistic regression

        Args:
            feature_importance_df: DataFrame with features and coefficients
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        top_features = feature_importance_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=figsize)

        colors = ["green" if c > 0 else "red" for c in top_features["coefficient"]]

        ax.barh(
            range(len(top_features)),
            top_features["coefficient"],
            color=colors,
            alpha=0.7,
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
        ax.set_title(
            f"Top {top_n} Predictive Features (Logistic Regression)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Add text annotation
        ax.text(
            0.98,
            0.02,
            "Green = increases success probability\nRed = decreases success probability",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            fontsize=9,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_confusion_heatmap(
        self,
        feature_col: str,
        n_bins: int = 3,
        figsize: tuple[int, int] = (8, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot confusion-like heatmap showing success rates

        Args:
            feature_col: Feature to bin
            n_bins: Number of bins
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        df_ = self.features_df[[feature_col, self.target_col]].dropna()

        # Create bins
        df_["bin"] = pd.qcut(
            df_[feature_col],
            q=n_bins,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

        # Create contingency table
        contingency = (
            pd.crosstab(df_["bin"], df_[self.target_col], normalize="index") * 100
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            contingency,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Percentage"},
            vmin=0,
            vmax=100,
        )

        ax.set_xlabel("Outcome")
        ax.set_ylabel(f"{feature_col.replace('_', ' ').title()} Level")
        ax.set_title(f"Success Rate by {feature_col.replace('_', ' ').title()} Level")
        ax.set_xticklabels(["Failure", "Success"])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig
