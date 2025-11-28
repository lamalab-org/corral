"""
Visualization for correlation analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from corral_trace_analyzer.config import VIZ_CONFIG


class CorrelationVisualizer:
    """Visualize correlation analysis results"""

    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize visualizer

        Args:
            features_df: DataFrame with features
        """
        self.features_df = features_df
        sns.set_style(VIZ_CONFIG["style"])
        sns.set_context(VIZ_CONFIG["context"])

    def plot_correlation_heatmap(
        self,
        feature_cols: list[str] | None = None,
        figsize: tuple[int, int] = (12, 10),
        method: str = "pearson",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot correlation heatmap

        Args:
            feature_cols: list of features to include
            figsize: Figure size
            method: Correlation method ('pearson' or 'spearman')
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        corr_matrix = self.features_df[feature_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=figsize)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
        )

        ax.set_title(
            f"{method.capitalize()} Correlation Heatmap", fontsize=16, fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_top_correlations(
        self,
        correlation_results: pd.DataFrame,
        n: int = 15,
        figsize: tuple[int, int] = (10, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot top N correlations as a bar chart

        Args:
            correlation_results: DataFrame with correlation results
            n: Number of top correlations to show
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        top_corrs = correlation_results.head(n).copy()

        fig, ax = plt.subplots(figsize=figsize)

        colors = ["green" if c > 0 else "red" for c in top_corrs["correlation"]]

        ax.barh(
            range(len(top_corrs)), top_corrs["correlation"], color=colors, alpha=0.7
        )

        ax.set_yticks(range(len(top_corrs)))
        ax.set_yticklabels(top_corrs["feature"])
        ax.set_xlabel("Correlation with Score", fontsize=12)
        ax.set_title(f"Top {n} Correlations with Score", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        # Add significance markers
        for i, (_, row) in enumerate(top_corrs.iterrows()):
            if row["significant"]:
                ax.text(
                    row["correlation"],
                    i,
                    " *",
                    va="center",
                    ha="left" if row["correlation"] > 0 else "right",
                    fontsize=12,
                    fontweight="bold",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_scatter_with_correlation(
        self,
        x_col: str,
        y_col: str,
        hue_col: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot scatter plot with correlation line

        Args:
            x_col: X-axis column
            y_col: Y-axis column
            hue_col: Optional column for coloring points
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if hue_col:
            for label in self.features_df[hue_col].unique():
                subset = self.features_df[self.features_df[hue_col] == label]
                ax.scatter(subset[x_col], subset[y_col], label=label, alpha=0.6, s=50)
        else:
            ax.scatter(
                self.features_df[x_col], self.features_df[y_col], alpha=0.6, s=50
            )

        # Add regression line
        mask = ~(self.features_df[x_col].isna() | self.features_df[y_col].isna())
        x = self.features_df[mask][x_col].values
        y = self.features_df[mask][y_col].values

        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_line,
                p(x_line),
                "r--",
                alpha=0.8,
                linewidth=2,
                label="Regression line",
            )

            # Calculate correlation
            from scipy import stats

            corr, pval = stats.pearsonr(x, y)
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}\np = {pval:.3e}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f"{y_col} vs {x_col}", fontsize=14, fontweight="bold")

        if hue_col or len(x) > 1:
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig

    def plot_correlation_matrix_interactive(
        self, feature_cols: list[str] | None = None, method: str = "pearson"
    ) -> go.Figure:
        """
        Create interactive correlation heatmap using Plotly

        Args:
            feature_cols: list of features to include
            method: Correlation method

        Returns:
            Plotly figure
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        corr_matrix = self.features_df[feature_cols].corr(method=method)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(title=f"{method.capitalize()} Correlation"),
            )
        )

        fig.update_layout(
            title=f"{method.capitalize()} Correlation Matrix",
            xaxis_title="",
            yaxis_title="",
            height=800,
            width=900,
        )

        return fig

    def plot_pairwise_scatter_matrix(
        self,
        feature_cols: list[str],
        hue_col: str | None = None,
        figsize: tuple[int, int] = (12, 12),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot pairwise scatter matrix

        Args:
            feature_cols: list of features to include
            hue_col: Optional column for coloring
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if hue_col:
            g = sns.pairplot(
                self.features_df[feature_cols + [hue_col]], hue=hue_col, diag_kind="kde"
            )
        else:
            g = sns.pairplot(self.features_df[feature_cols], diag_kind="kde")

        g.fig.suptitle(
            "Pairwise Feature Relationships", y=1.01, fontsize=16, fontweight="bold"
        )

        if save_path:
            g.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return g.fig

    def plot_correlation_comparison(
        self,
        feature_col: str,
        target_col: str,
        group_col: str,
        figsize: tuple[int, int] = (12, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Compare correlations across groups

        Args:
            feature_col: Feature to correlate
            target_col: Target variable
            group_col: Column to group by
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        groups = self.features_df[group_col].unique()

        fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)

        if len(groups) == 1:
            axes = [axes]

        from scipy import stats

        for ax, group in zip(axes, groups, strict=False):
            subset = self.features_df[self.features_df[group_col] == group]
            mask = ~(subset[feature_col].isna() | subset[target_col].isna())
            x = subset[mask][feature_col].values
            y = subset[mask][target_col].values

            if len(x) > 1:
                ax.scatter(x, y, alpha=0.6, s=50)

                # Regression line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

                # Correlation
                corr, pval = stats.pearsonr(x, y)
                ax.set_title(f"{group}\nr = {corr:.3f}, p = {pval:.3e}", fontsize=10)

            ax.set_xlabel(feature_col, fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel(target_col, fontsize=10)

        fig.suptitle(
            f"{target_col} vs {feature_col} by {group_col}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")

        return fig
