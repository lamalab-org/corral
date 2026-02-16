import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lama_aesthetics import (
    ONE_COL_HEIGHT,
    ONE_COL_WIDTH,
    TWO_COL_HEIGHT,
    TWO_COL_WIDTH,
)
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

MODEL_COLORS = {"Claude 4.5": "#8900ff", "GPT-4o": "#ff0677", "OSS": "#0051ff"}


def plot_capability_heatmaps(knowledge_df, reasoning_df, output_path):
    """
    Side-by-side heatmaps for theta_K and theta_R.
    """
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    for ax, df, label in zip(
        axes,
        [knowledge_df, reasoning_df],
        [r"$\theta_K$", r"$\theta_R$"],
        strict=False,
    ):
        pivot = df.pivot_table(
            index="model", columns="environment", values="theta_mean"
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="Purples",
            center=0,
            ax=ax,
            cbar_kws={"label": label},
        )
        ax.set_title(f"Capability {label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Environment")
        ax.set_ylabel("")

    axes[0].set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


def plot_capability_profiles(knowledge_df, reasoning_df, output_path):
    """
    Side-by-side profile lines for theta_K and theta_R.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
        sharey=True,
    )

    for ax, df, label in zip(
        axes,
        [knowledge_df, reasoning_df],
        [r"$\theta_K$", r"$\theta_R$"],
        strict=False,
    ):
        models = df["model"].unique()
        for model in models:
            color = MODEL_COLORS.get(model, "gray")
            model_data = df[df["model"] == model].sort_values("environment")
            ax.plot(
                model_data["environment"],
                model_data["theta_mean"],
                marker="o",
                label=model,
                color=color,
                linewidth=2,
                markersize=8,
            )
            if "theta_sd" in model_data.columns:
                ax.fill_between(
                    model_data["environment"],
                    model_data["theta_mean"] - model_data["theta_sd"],
                    model_data["theta_mean"] + model_data["theta_sd"],
                    alpha=0.2,
                    color=color,
                )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Environment", fontsize=9)
        ax.set_title(
            f"Capability {label} Across Domains", fontsize=9, fontweight="bold"
        )
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

        all_y = df["theta_mean"].to_numpy()
        envs = df["environment"].unique()
        range_frame(ax, np.arange(len(envs)), all_y, pad=0.05)

    axes[0].set_ylabel(r"Capability ($\theta$)", fontsize=9)
    axes[0].legend(loc="best", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


def plot_knowledge_vs_reasoning(knowledge_df, reasoning_df, output_path):
    """
    Side-by-side grouped bar chart comparing knowledge and reasoning
    capability across domains for each model.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
        sharey=True,
    )

    for ax, df, title in zip(
        axes,
        [knowledge_df, reasoning_df],
        [r"Knowledge Capability ($\theta_K$)", r"Reasoning Capability ($\theta_R$)"],
        strict=False,
    ):
        envs = sorted(df["environment"].unique())
        models = sorted(df["model"].unique())
        x = np.arange(len(envs))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_data = df[df["model"] == model].set_index("environment").reindex(envs)
            color = MODEL_COLORS.get(model, "gray")
            offset = x + i * width - 0.4 + width / 2

            ax.bar(
                offset,
                model_data["theta_mean"],
                width,
                label=model,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            if "theta_sd" in model_data.columns:
                ax.errorbar(
                    offset,
                    model_data["theta_mean"],
                    yerr=model_data["theta_sd"],
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        all_y = df["theta_mean"].to_numpy()
        range_frame(ax, x, all_y, pad=0.1)

    axes[0].set_ylabel(r"Capability ($\theta$)", fontsize=9)
    axes[0].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


def plot_scatter_knowledge_reasoning(knowledge_df, reasoning_df, output_path):
    """
    Scatter plot of knowledge theta vs reasoning theta for each (model, env) pair.
    """
    merged = knowledge_df.merge(
        reasoning_df,
        on=["model", "environment"],
        suffixes=("_knowledge", "_reasoning"),
    )

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH * 1.5, ONE_COL_HEIGHT * 1.5))

    for model in merged["model"].unique():
        model_data = merged[merged["model"] == model]
        color = MODEL_COLORS.get(model, "gray")
        ax.scatter(
            model_data["theta_mean_knowledge"],
            model_data["theta_mean_reasoning"],
            color=color,
            label=model,
            s=80,
            # edgecolors="black",
            # linewidth=0.5,
            zorder=3,
        )
        # Label each point with environment name
        for _, row in model_data.iterrows():
            ax.annotate(
                row["environment"],
                (row["theta_mean_knowledge"], row["theta_mean_reasoning"]),
                fontsize=5,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

    # Add diagonal reference line
    all_x = merged["theta_mean_knowledge"].to_numpy()
    all_y = merged["theta_mean_reasoning"].to_numpy()
    range_frame(ax, all_x, all_y, pad=0.1)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, zorder=1)

    ax.set_xlabel(r"$\theta_K$ (Knowledge)", fontsize=9)
    ax.set_ylabel(r"$\theta_R$ (Reasoning)", fontsize=9)
    ax.set_title(r"$\theta_K$ vs $\theta_R$", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Plot capability heatmaps and profiles"
    )
    parser.add_argument(
        "--output-dir", default="./capability_plots", help="Output directory for plots"
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    knowledge_theta = pd.read_csv("knowledge_theta.csv")
    reasoning_theta = pd.read_csv("reasoning_theta.csv")

    # Heatmaps (theta_K and theta_R side by side)
    plot_capability_heatmaps(
        knowledge_theta,
        reasoning_theta,
        f"{args.output_dir}/capability_heatmaps.png",
    )

    # Profile lines (theta_K and theta_R side by side)
    plot_capability_profiles(
        knowledge_theta,
        reasoning_theta,
        f"{args.output_dir}/capability_profiles.png",
    )

    # Comparison plots
    plot_knowledge_vs_reasoning(
        knowledge_theta,
        reasoning_theta,
        f"{args.output_dir}/capability_comparison.png",
    )

    plot_scatter_knowledge_reasoning(
        knowledge_theta,
        reasoning_theta,
        f"{args.output_dir}/knowledge_vs_reasoning_scatter.png",
    )
