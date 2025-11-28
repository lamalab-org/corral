"""
Comparative visualization functions for multi-environment analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Set style
sns.set_style("whitegrid")
sns.set_context("notebook")


class ComparativePlotter:
    """Generate comparative plots across multiple environments"""

    def __init__(
        self, env_features: dict[str, pd.DataFrame], env_results: dict[str, dict]
    ):
        """
        Initialize comparative plotter

        Args:
            env_features: dict mapping environment name to features DataFrame
            env_results: dict mapping environment name to analysis results dict
                        (should contain 'point_biserial', 'mann_whitney', 'logistic_regression')
        """
        self.env_features = env_features
        self.env_results = env_results
        self.environments = list(env_features.keys())

    def plot_success_rate_by_quantile_grid(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] | None = None,
    ):
        """
        Plot 3: Success rate by feature quantile for top feature in each environment
        N subplots (dynamic grid), one per environment
        """
        # Calculate grid dimensions dynamically
        n_envs = len(self.environments)
        ncols = 3  # Fixed number of columns
        nrows = (n_envs + ncols - 1) // ncols  # Ceiling division

        # Auto-size figure if not specified
        if figsize is None:
            figsize = (18, 4 * nrows)  # 4 inches per row

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_envs > 1 else [axes]

        for idx, env_name in enumerate(self.environments):
            ax = axes[idx]
            features_df = self.env_features[env_name]
            results = self.env_results[env_name]

            # Get top feature from point-biserial correlation
            point_biserial = results.get("point_biserial")
            if point_biserial is None or len(point_biserial) == 0:
                ax.text(0.5, 0.5, f"No data for {env_name}", ha="center", va="center")
                ax.set_title(env_name.upper())
                continue

            top_feature = point_biserial.iloc[0]["feature"]
            correlation = point_biserial.iloc[0]["correlation"]

            # Create quantiles (or binary if feature is binary)
            if features_df[top_feature].nunique() == 2:
                # Binary feature
                grouped = features_df.groupby(top_feature)["score"].agg(
                    ["mean", "count"]
                )
                x_labels = ["0", "1"]
                success_rates = grouped["mean"].to_numpy()
                counts = grouped["count"].to_numpy()
            else:
                # Continuous feature - use 5 quantiles
                features_df_copy = features_df.copy()
                features_df_copy["quantile"] = pd.qcut(
                    features_df_copy[top_feature], q=5, labels=False, duplicates="drop"
                )
                grouped = features_df_copy.groupby("quantile")["score"].agg(
                    ["mean", "count"]
                )
                x_labels = [f"Q{i+1}" for i in range(len(grouped))]
                success_rates = grouped["mean"].to_numpy()
                counts = grouped["count"].to_numpy()

            # Plot bars
            bars = ax.bar(x_labels, success_rates, alpha=0.7, edgecolor="black")

            # Color bars by success rate
            for _i, (bar, rate) in enumerate(zip(bars, success_rates, strict=False)):
                bar.set_color(plt.cm.RdYlGn(rate))

            # Add count labels on bars
            for i, (_x, y, count) in enumerate(
                zip(x_labels, success_rates, counts, strict=False)
            ):
                ax.text(i, y + 0.02, f"n={count}", ha="center", fontsize=9)

            # Formatting
            ax.set_title(
                f"{env_name.upper()}\n{top_feature}\n(r={correlation:.3f})", fontsize=11
            )
            ax.set_ylabel("Success Rate", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(n_envs, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        return fig

    def plot_feature_distributions_by_outcome(
        self,
        top_n: int = 3,
        save_path: str | None = None,
        figsize: tuple[int, int] | None = None,
    ):
        """
        Plot 5: Feature distribution by outcome (multi-panel)
        Grid: N rows (envs) x top_n columns (top features per env)
        """
        n_envs = len(self.environments)

        # Auto-size figure if not specified
        if figsize is None:
            figsize = (6 * top_n, 2 * n_envs)  # 6 inches per column, 2 inches per row

        fig, axes = plt.subplots(n_envs, top_n, figsize=figsize)

        # Handle different array shapes
        if n_envs == 1 and top_n == 1:
            axes = np.array([[axes]])
        elif n_envs == 1:
            axes = axes.reshape(1, -1)
        elif top_n == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, env_name in enumerate(self.environments):
            features_df = self.env_features[env_name]
            results = self.env_results[env_name]

            # Get top features
            point_biserial = results.get("point_biserial")
            if point_biserial is None or len(point_biserial) == 0:
                for col_idx in range(top_n):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            top_features = point_biserial.head(top_n)

            for col_idx, (_, feature_row) in enumerate(top_features.iterrows()):
                ax = axes[row_idx, col_idx]
                feature_name = feature_row["feature"]

                if feature_name not in features_df.columns:
                    ax.text(
                        0.5, 0.5, f"{feature_name}\nnot found", ha="center", va="center"
                    )
                    continue

                # Create violin plot
                data_success = features_df[features_df["score"] == 1][
                    feature_name
                ].dropna()
                data_failure = features_df[features_df["score"] == 0][
                    feature_name
                ].dropna()

                positions = [0, 1]
                parts = ax.violinplot(
                    [data_failure, data_success],
                    positions=positions,
                    showmeans=True,
                    showextrema=True,
                )

                # Color violins
                for pc, color in zip(
                    parts["bodies"], ["#ff7f7f", "#7fbf7f"], strict=False
                ):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

                # Formatting
                ax.set_xticks(positions)
                ax.set_xticklabels(["Failure", "Success"])
                ax.set_ylabel(feature_name, fontsize=9)
                ax.grid(axis="y", alpha=0.3)

                # Add title with correlation
                corr = feature_row["correlation"]
                ax.set_title(f"r={corr:.3f}", fontsize=9)

                # Add row label on leftmost column
                if col_idx == 0:
                    ax.text(
                        -0.3,
                        0.5,
                        env_name.upper(),
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                    )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        return fig

    def plot_first_step_features_impact(
        self, save_path: str | None = None, figsize: tuple[int, int] = (16, 5)
    ):
        """
        Plot 10: First-step features impact
        3 panels (3 first-step features)
        For each: success rate when feature=0 vs feature=1
        Separate bars per environment
        """
        first_step_features = [
            "has_planning_in_first_step",
            "has_reasoning_in_first_step",
            "has_tool_error_in_first_step",
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for idx, feature in enumerate(first_step_features):
            ax = axes[idx]

            env_names = []
            success_rate_0 = []
            success_rate_1 = []
            counts_0 = []
            counts_1 = []

            for env_name in self.environments:
                features_df = self.env_features[env_name]

                if feature not in features_df.columns:
                    continue

                # Calculate success rates for feature=0 and feature=1
                grouped = features_df.groupby(feature)["score"].agg(["mean", "count"])

                if 0 in grouped.index:
                    success_rate_0.append(grouped.loc[0, "mean"])
                    counts_0.append(grouped.loc[0, "count"])
                else:
                    success_rate_0.append(0)
                    counts_0.append(0)

                if 1 in grouped.index:
                    success_rate_1.append(grouped.loc[1, "mean"])
                    counts_1.append(grouped.loc[1, "count"])
                else:
                    success_rate_1.append(0)
                    counts_1.append(0)

                env_names.append(env_name.upper())

            # Create grouped bar chart
            x = np.arange(len(env_names))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2, success_rate_0, width, label=f"{feature}=0", alpha=0.8
            )
            bars2 = ax.bar(
                x + width / 2, success_rate_1, width, label=f"{feature}=1", alpha=0.8
            )

            # Add count labels
            for _i, (bar, count) in enumerate(zip(bars1, counts_0, strict=False)):
                if count > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.02,
                        f"n={count}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            for i, (bar, count) in enumerate(zip(bars2, counts_1, strict=False)):
                if count > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.02,
                        f"n={count}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            # Formatting
            ax.set_ylabel("Success Rate", fontsize=10)
            ax.set_title(feature.replace("_", " ").title(), fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(env_names, rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        return fig

    def plot_individual_vs_all_comparison(
        self,
        all_env_name: str = "all",
        save_path: str | None = None,
        figsize: tuple[int, int] = (16, 10),
    ):
        """
        Plot 7: Individual vs ALL - Feature importance comparison
        Scatter plot: x=correlation in individual env, y=correlation in ALL
        Separate panel per environment
        """
        if all_env_name not in self.env_results:
            logger.info(f"Warning: '{all_env_name}' not found in results")
            return None

        all_results = self.env_results[all_env_name]["point_biserial"]
        individual_envs = [env for env in self.environments if env != all_env_name]

        n_envs = len(individual_envs)
        ncols = 3
        nrows = (n_envs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_envs > 1 else [axes]

        for idx, env_name in enumerate(individual_envs):
            ax = axes[idx]
            env_results = self.env_results[env_name]["point_biserial"]

            # Merge on feature name
            merged = env_results[["feature", "correlation"]].merge(
                all_results[["feature", "correlation"]],
                on="feature",
                suffixes=("_env", "_all"),
            )

            # Scatter plot
            ax.scatter(
                merged["correlation_env"], merged["correlation_all"], alpha=0.6, s=50
            )

            # Add diagonal line (x=y)
            lim = max(
                abs(merged["correlation_env"].min()),
                abs(merged["correlation_env"].max()),
                abs(merged["correlation_all"].min()),
                abs(merged["correlation_all"].max()),
            )
            ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5, label="x=y")

            # Add quadrant lines
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

            # Annotate outliers (features with big difference)
            merged["diff"] = abs(merged["correlation_env"] - merged["correlation_all"])
            top_outliers = merged.nlargest(3, "diff")

            for _, row in top_outliers.iterrows():
                ax.annotate(
                    row["feature"][:20],  # Truncate long names
                    xy=(row["correlation_env"], row["correlation_all"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

            # Formatting
            ax.set_xlabel(f"{env_name.upper()} correlation", fontsize=10)
            ax.set_ylabel("ALL environments correlation", fontsize=10)
            ax.set_title(env_name.upper(), fontsize=11)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_envs, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        return fig

    def plot_longest_looping_sequence_analysis(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (16, 10),
    ):
        """
        Plot 11: Longest looping sequence analysis
        Violin plot: longest_looping_sequence by outcome
        Faceted by environment
        """
        feature = "longest_looping_sequence"

        ncols = 3
        nrows = (len(self.environments) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if len(self.environments) > 1 else [axes]

        for idx, env_name in enumerate(self.environments):
            ax = axes[idx]
            features_df = self.env_features[env_name]

            if feature not in features_df.columns:
                ax.text(0.5, 0.5, f"{feature}\nnot available", ha="center", va="center")
                ax.set_title(env_name.upper())
                continue

            # Get data
            data_success = features_df[features_df["score"] == 1][feature].dropna()
            data_failure = features_df[features_df["score"] == 0][feature].dropna()

            # Calculate statistics
            mean_success = data_success.mean()
            mean_failure = data_failure.mean()
            median_success = data_success.median()
            median_failure = data_failure.median()

            # Create violin plot
            positions = [0, 1]
            parts = ax.violinplot(
                [data_failure, data_success],
                positions=positions,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )

            # Color violins
            for pc, color in zip(parts["bodies"], ["#ff7f7f", "#7fbf7f"], strict=False):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            # Add statistics text
            stats_text = (
                f"Failure: μ={mean_failure:.2f}, med={median_failure:.1f}\n"
                f"Success: μ={mean_success:.2f}, med={median_success:.1f}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )

            # Formatting
            ax.set_xticks(positions)
            ax.set_xticklabels(["Failure", "Success"])
            ax.set_ylabel("Longest Looping Sequence (steps)", fontsize=10)
            ax.set_title(env_name.upper(), fontsize=11)
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(self.environments), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        return fig

    def generate_all_priority_plots(self, output_dir: str = "comparative_plots"):
        """
        Generate all 5 priority plots and save to output directory

        Returns:
            dict: Paths to saved plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        logger.info("\n" + "=" * 80)
        logger.info("GENERATING TOP 5 PRIORITY PLOTS")
        logger.info("=" * 80 + "\n")

        # Plot 3: Success rate by quantile
        logger.info("Plot 1/5: Success Rate by Feature Quantile Grid...")
        path = str(output_path / "plot3_success_rate_by_quantile_grid.png")
        self.plot_success_rate_by_quantile_grid(save_path=path)
        saved_plots["plot3"] = path

        # Plot 5: Feature distributions
        logger.info("\nPlot 2/5: Feature Distributions by Outcome...")
        path = str(output_path / "plot5_feature_distributions_by_outcome.png")
        self.plot_feature_distributions_by_outcome(save_path=path)
        saved_plots["plot5"] = path

        # Plot 10: First-step features
        logger.info("\nPlot 3/5: First-Step Features Impact...")
        path = str(output_path / "plot10_first_step_features_impact.png")
        self.plot_first_step_features_impact(save_path=path)
        saved_plots["plot10"] = path

        # Plot 7: Individual vs ALL
        logger.info("\nPlot 4/5: Individual vs ALL Comparison...")
        path = str(output_path / "plot7_individual_vs_all_comparison.png")
        self.plot_individual_vs_all_comparison(save_path=path)
        saved_plots["plot7"] = path

        # Plot 11: Looping analysis
        logger.info("\nPlot 5/5: Longest Looping Sequence Analysis...")
        path = str(output_path / "plot11_longest_looping_sequence_analysis.png")
        self.plot_longest_looping_sequence_analysis(save_path=path)
        saved_plots["plot11"] = path

        logger.info("\n" + "=" * 80)
        logger.info("ALL PRIORITY PLOTS GENERATED!")
        logger.info("=" * 80)
        logger.info(f"\nPlots saved to: {output_path}/")
        for plot_name, plot_path in saved_plots.items():
            logger.info(f"  - {plot_name}: {Path(plot_path).name}")

        return saved_plots
