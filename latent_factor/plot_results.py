"""
Plot all results from latent factor modelling.

Reads saved outputs:
  - knowledge_theta.csv, reasoning_theta.csv  (IRT theta estimates)
  - results/agent_trace.nc                    (MCMC posterior samples)
  - results/agent_df.csv                      (processed agent data)
  - results/results.json                      (variance decomposition)

Usage:
  cd latent_factor && python plot_results.py
  cd latent_factor && python plot_results.py --results-dir ./results --output-dir ./plots
"""

import argparse
import json

import arviz as az
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


# =============================================================================
# Capability plots (from IRT theta estimates)
# =============================================================================


def plot_capability_heatmaps(knowledge_df, reasoning_df, output_path):
    """Side-by-side heatmaps for theta_K and theta_R."""
    from pathlib import Path

    output_path = Path(output_path)
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
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


def plot_capability_profiles(knowledge_df, reasoning_df, output_path):
    """Side-by-side profile lines for theta_K and theta_R."""
    from pathlib import Path

    output_path = Path(output_path)
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
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


# =============================================================================
# Agent model plots (from MCMC trace)
# =============================================================================


def plot_lambda_forest(trace, env_names, output_path):
    """Knowledge loading forest plot."""
    lam_summary = az.summary(trace, var_names=["lambda"], hdi_prob=0.9)
    env_list = list(env_names) if isinstance(env_names, dict) else env_names

    from pathlib import Path

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    order = lam_summary["mean"].argsort()
    y_pos = np.arange(len(lam_summary))

    means = lam_summary["mean"].iloc[order].to_numpy()
    lows = lam_summary["hdi_5%"].iloc[order].to_numpy()
    highs = lam_summary["hdi_95%"].iloc[order].to_numpy()

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lows, highs - means],
        fmt="o",
        capsize=4,
        color="steelblue",
        markersize=8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([env_list[i] for i in order])
    range_frame(ax, np.concatenate([lows, highs]), y_pos, pad=0.05)

    ax.axvline(1, color="gray", linestyle="--", alpha=0.7, label=r"$\lambda$=1")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label=r"$\lambda$=0")

    xlim = ax.get_xlim()
    ax.axvspan(1.2, xlim[1], alpha=0.1, color="#0051ff", label="Knowledge-limited")
    ax.axvspan(xlim[0], 0.5, alpha=0.1, color="#ff0677", label="Execution-limited")

    ax.set_xlabel(r"Knowledge Loading ($\lambda$)", fontsize=8)
    ax.set_title(
        "How Much Does Domain Knowledge Matter?", fontsize=8, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


def plot_psi_forest(trace, env_names, output_path):
    """Reasoning loading forest plot."""
    psi_summary = az.summary(trace, var_names=["psi"], hdi_prob=0.9)
    env_list = list(env_names) if isinstance(env_names, dict) else env_names

    from pathlib import Path

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    order = psi_summary["mean"].argsort()
    y_pos = np.arange(len(psi_summary))

    means = psi_summary["mean"].iloc[order].to_numpy()
    lows = psi_summary["hdi_5%"].iloc[order].to_numpy()
    highs = psi_summary["hdi_95%"].iloc[order].to_numpy()

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lows, highs - means],
        fmt="o",
        capsize=4,
        color="#e84ab5",
        markersize=8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([env_list[i] for i in order])
    range_frame(ax, np.concatenate([lows, highs]), y_pos, pad=0.05)

    ax.axvline(1, color="gray", linestyle="--", alpha=0.7, label=r"$\psi$=1")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label=r"$\psi$=0")

    xlim = ax.get_xlim()
    ax.axvspan(1.2, xlim[1], alpha=0.1, color="#0051ff", label="Reasoning-limited")
    ax.axvspan(xlim[0], 0.5, alpha=0.1, color="#ff0677", label="Execution-limited")

    ax.set_xlabel(r"Reasoning Loading ($\psi$)", fontsize=8)
    ax.set_title(
        "How Much Does Reasoning Capability Matter?", fontsize=8, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


def plot_variance_decomposition(var_dict, output_path):
    from pathlib import Path

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    sorted_items = sorted(var_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0].capitalize() for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ["#5f59d0", "#e84ab5", "#a052c3", "#668fb6", "#6bb5a4", "#2a9d8f"]

    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
        )

    y_pos = np.arange(len(values))
    range_frame(ax, np.array([0, max(values)]), y_pos, pad=0.15)

    ax.set_xlabel("Variance Explained (%)", fontsize=6)
    ax.set_title("Variance Decomposition", fontsize=6, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


def plot_scaffold_effects(trace, scaffold_names, output_path):
    from pathlib import Path

    output_path = Path(output_path)
    gamma_summary = az.summary(trace, var_names=["gamma"], hdi_prob=0.9)
    scaffold_list = (
        list(scaffold_names) if isinstance(scaffold_names, dict) else scaffold_names
    )

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    y_pos = np.arange(len(gamma_summary))
    means = gamma_summary["mean"].to_numpy()
    lows = gamma_summary["hdi_5%"].to_numpy()
    highs = gamma_summary["hdi_95%"].to_numpy()

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lows, highs - means],
        fmt="o",
        capsize=4,
        color="#711c91",
        markersize=8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scaffold_list)
    range_frame(ax, np.concatenate([lows, highs]), y_pos, pad=0.05)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Effect on Log-Odds\n of Success", fontsize=6)
    ax.set_title(r"Scaffold Effects ($\gamma$)", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


def plot_level_effects(trace, output_path):
    from pathlib import Path

    output_path = Path(output_path)
    delta_summary = az.summary(trace, var_names=["delta"], hdi_prob=0.9)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    levels = [f"Level {i + 1}" for i in range(len(delta_summary))]
    y_pos = np.arange(len(delta_summary))
    means = delta_summary["mean"].to_numpy()
    lows = delta_summary["hdi_5%"].to_numpy()
    highs = delta_summary["hdi_95%"].to_numpy()

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lows, highs - means],
        fmt="o",
        capsize=4,
        color="#6300ff",
        markersize=8,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(levels)
    range_frame(ax, np.concatenate([lows, highs]), y_pos, pad=0.05)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Effect on Log-Odds\n of Success", fontsize=6)
    ax.set_title(r"Level Difficulty ($\delta$)", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    # ...existing code...
    plt.close()
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all latent factor model results")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Directory with saved model results (agent_trace.nc, etc.)",
    )
    parser.add_argument(
        "--output-dir", default="./plots", help="Output directory for plots"
    )
    args = parser.parse_args()

    from pathlib import Path

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load saved results
    # ...existing code...
    knowledge_theta_df = pd.read_csv("knowledge_theta.csv")
    reasoning_theta_df = pd.read_csv("reasoning_theta.csv")
    agent_trace = az.from_netcdf(f"{args.results_dir}/agent_trace.nc")
    agent_df = pd.read_csv(f"{args.results_dir}/agent_df.csv")

    from pathlib import Path

    with Path(f"{args.results_dir}/results.json").open() as f:
        saved = json.load(f)
    var_result = saved["variance_decomposition"]

    env_names = sorted(agent_df["environment"].unique())
    scaffold_names = sorted(agent_df["scaffold"].unique())

    # ...existing code...

    # Capability heatmaps (theta_K and theta_R side by side)
    plot_capability_heatmaps(
        knowledge_theta_df,
        reasoning_theta_df,
        f"{args.output_dir}/fig1_capability_heatmaps.png",
    )

    # Loading forest plots
    plot_lambda_forest(
        agent_trace, env_names, f"{args.output_dir}/fig2a_lambda_forest.png"
    )
    plot_psi_forest(agent_trace, env_names, f"{args.output_dir}/fig2b_psi_forest.png")

    # Variance decomposition
    plot_variance_decomposition(
        var_result, f"{args.output_dir}/fig3_variance_decomposition.png"
    )

    # Capability profiles (theta_K and theta_R side by side)
    plot_capability_profiles(
        knowledge_theta_df,
        reasoning_theta_df,
        f"{args.output_dir}/fig4_capability_profiles.png",
    )

    # Scaffold and level effects
    plot_scaffold_effects(
        agent_trace, scaffold_names, f"{args.output_dir}/fig5_scaffold_effects.png"
    )
    plot_level_effects(agent_trace, f"{args.output_dir}/fig6_level_effects.png")

    # ...existing code...
