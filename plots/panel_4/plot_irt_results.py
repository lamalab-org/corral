"""
Unified IRT Results Plotting Script

Combines all IRT model visualizations with user-selectable plots.
Uses data from analysis/results/irt_* and shared plotting utilities.

Available plots:
  1. capability_heatmaps       - Heatmaps of theta_K and theta_R
  2. capability_profiles       - Line plots of theta_K and theta_R across environments
  3. capability_comparison     - Grouped bar charts comparing knowledge vs reasoning
  4. knowledge_vs_reasoning    - Scatter plot of theta_K vs theta_R
  5. lambda_forest            - Knowledge loading forest plot
  6. psi_forest               - Reasoning loading forest plot
  7. variance_decomposition   - Variance explained by each component
  8. scaffold_effects         - Agent scaffold effects
  9. level_effects            - Difficulty level effects

Model types (4 specifications):
  - baseline: Knowledge, reasoning, scaffold, level, verbosity only
  - category: Adds category effect (task vs subtask)
  - task: Adds task-specific random effects
  - category_task: Includes both category and task effects

Usage:
  # Generate all plots for baseline model
  python plot_irt_results.py --model_type=baseline

  # Generate specific plots for category+task model
  python plot_irt_results.py --model_type=category_task --plots=capability_heatmaps,lambda_forest

  # Custom paths
  python plot_irt_results.py --results_dir=../analysis/results/irt_baseline --output_dir=./output

  # List available plots
  python plot_irt_results.py list_plots
"""

import importlib.util
import json
import sys
from pathlib import Path

import arviz as az
import fire
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
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_CONFIG_PATH = REPO_ROOT / "analysis" / "plot_config.py"
plot_config_spec = importlib.util.spec_from_file_location(
    "plot_config", PLOT_CONFIG_PATH
)
if plot_config_spec is None or plot_config_spec.loader is None:
    raise ImportError(f"Could not load plot config from {PLOT_CONFIG_PATH}")
plot_config = importlib.util.module_from_spec(plot_config_spec)
plot_config_spec.loader.exec_module(plot_config)

AGENT_NAMES = plot_config.AGENT_NAMES
ENVIRONMENT_NAMES = plot_config.ENVIRONMENT_NAMES
MODEL_COLOURS = plot_config.MODEL_COLOURS
MODEL_NAMES = plot_config.MODEL_NAMES

# Apply style
lama_aesthetics.get_style("main")

# Create color maps: raw ID -> color and display name -> color
MODEL_COLORS_BY_ID = dict(zip(MODEL_NAMES.keys(), MODEL_COLOURS, strict=False))
MODEL_COLORS_BY_NAME = {
    MODEL_NAMES[model_id]: color for model_id, color in MODEL_COLORS_BY_ID.items()
}


# =============================================================================
# Helper functions
# =============================================================================


def map_display_names(dataframe):
    """Map raw IDs to display names for plotting.

    Args:
        dataframe: DataFrame with raw ID columns (model, environment, scaffold)

    Returns:
        DataFrame with display names
    """
    mapped_df = dataframe.copy()

    if "model" in mapped_df.columns:
        mapped_df["model"] = mapped_df["model"].map(MODEL_NAMES)
    if "environment" in mapped_df.columns:
        mapped_df["environment"] = mapped_df["environment"].map(ENVIRONMENT_NAMES)
    if "scaffold" in mapped_df.columns:
        mapped_df["scaffold"] = mapped_df["scaffold"].map(AGENT_NAMES)

    return mapped_df


def get_model_color(display_name):
    """Get color for a model display name.

    Args:
        display_name: Display name (e.g., "Claude-4.5-Sonnet")

    Returns:
        Color hex string
    """
    return MODEL_COLORS_BY_NAME.get(display_name, "gray")


# =============================================================================
# Capability plots (from IRT theta estimates)
# =============================================================================


def plot_capability_heatmaps(knowledge_df, reasoning_df, output_path):
    """Side-by-side heatmaps for theta_K and theta_R."""
    # Map raw IDs to display names
    knowledge_df = map_display_names(knowledge_df)
    reasoning_df = map_display_names(reasoning_df)

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
    plt.close()
    logger.success(f"Saved capability heatmaps to {output_path}")
    return fig


def plot_capability_profiles(knowledge_df, reasoning_df, output_path):
    """Side-by-side profile lines for theta_K and theta_R."""
    # Map raw IDs to display names
    knowledge_df = map_display_names(knowledge_df)
    reasoning_df = map_display_names(reasoning_df)

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
            # Get color based on display name
            color = get_model_color(model)
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
    plt.close()
    logger.success(f"Saved capability profiles to {output_path}")
    return fig


def plot_knowledge_vs_reasoning(knowledge_df, reasoning_df, output_path):
    """Side-by-side grouped bar chart comparing knowledge and reasoning."""
    # Map raw IDs to display names
    knowledge_df = map_display_names(knowledge_df)
    reasoning_df = map_display_names(reasoning_df)

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
            color = get_model_color(model)
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
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved capability comparison to {output_path}")
    return fig


def plot_scatter_knowledge_reasoning(knowledge_df, reasoning_df, output_path):
    """Scatter plot of knowledge theta vs reasoning theta."""
    # Map raw IDs to display names
    knowledge_df = map_display_names(knowledge_df)
    reasoning_df = map_display_names(reasoning_df)

    merged = knowledge_df.merge(
        reasoning_df,
        on=["model", "environment"],
        suffixes=("_knowledge", "_reasoning"),
    )

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH * 1.5, ONE_COL_HEIGHT * 1.5))

    for model in merged["model"].unique():
        model_data = merged[merged["model"] == model]
        color = get_model_color(model)
        ax.scatter(
            model_data["theta_mean_knowledge"],
            model_data["theta_mean_reasoning"],
            color=color,
            label=model,
            s=80,
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
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved knowledge vs reasoning scatter to {output_path}")
    return fig


# =============================================================================
# Agent model plots (from MCMC trace)
# =============================================================================


def plot_lambda_forest(trace, env_names, output_path):
    """Knowledge loading forest plot."""
    lam_summary = az.summary(trace, var_names=["lambda"], hdi_prob=0.9)
    # Map raw env IDs to display names
    env_list = [ENVIRONMENT_NAMES.get(e, e) for e in env_names]

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
    plt.close()
    logger.success(f"Saved lambda forest plot to {output_path}")
    return fig


def plot_psi_forest(trace, env_names, output_path):
    """Reasoning loading forest plot."""
    psi_summary = az.summary(trace, var_names=["psi"], hdi_prob=0.9)
    # Map raw env IDs to display names
    env_list = [ENVIRONMENT_NAMES.get(e, e) for e in env_names]

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
    plt.close()
    logger.success(f"Saved psi forest plot to {output_path}")
    return fig


def plot_variance_decomposition(var_dict, output_path):
    """Variance decomposition bar chart."""
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    sorted_items = sorted(var_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0].capitalize() for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ["#5f59d0", "#e84ab5", "#a052c3", "#668fb6", "#6bb5a4", "#2a9d8f"]

    bars = ax.barh(labels, values, color=colors)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )

    y_pos = np.arange(len(values))
    range_frame(ax, np.array([0, max(values)]), y_pos, pad=0.15)

    ax.set_xlabel("Variance Explained (%)", fontsize=8)
    ax.set_title("Variance Decomposition", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved variance decomposition to {output_path}")
    return fig


def plot_scaffold_effects(trace, scaffold_names, output_path):
    """Agent scaffold effects forest plot."""
    gamma_summary = az.summary(trace, var_names=["gamma"], hdi_prob=0.9)
    # Map raw scaffold IDs to display names
    scaffold_list = [AGENT_NAMES.get(s, s) for s in scaffold_names]

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
    plt.close()
    logger.success(f"Saved scaffold effects to {output_path}")
    return fig


def plot_level_effects(trace, output_path):
    """Difficulty level effects forest plot."""
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
    plt.close()
    logger.success(f"Saved level effects to {output_path}")
    return fig


# =============================================================================
# Plot registry and dispatcher
# =============================================================================

PLOT_FUNCTIONS = {
    "capability_heatmaps": {
        "func": plot_capability_heatmaps,
        "requires_trace": False,
        "description": "Heatmaps of theta_K and theta_R",
    },
    "capability_profiles": {
        "func": plot_capability_profiles,
        "requires_trace": False,
        "description": "Line plots of theta_K and theta_R",
    },
    "capability_comparison": {
        "func": plot_knowledge_vs_reasoning,
        "requires_trace": False,
        "description": "Grouped bar charts comparing knowledge vs reasoning",
    },
    "knowledge_vs_reasoning": {
        "func": plot_scatter_knowledge_reasoning,
        "requires_trace": False,
        "description": "Scatter plot of theta_K vs theta_R",
    },
    "lambda_forest": {
        "func": plot_lambda_forest,
        "requires_trace": True,
        "description": "Knowledge loading forest plot",
    },
    "psi_forest": {
        "func": plot_psi_forest,
        "requires_trace": True,
        "description": "Reasoning loading forest plot",
    },
    "variance_decomposition": {
        "func": plot_variance_decomposition,
        "requires_trace": True,
        "description": "Variance explained by each component",
    },
    "scaffold_effects": {
        "func": plot_scaffold_effects,
        "requires_trace": True,
        "description": "Agent scaffold effects",
    },
    "level_effects": {
        "func": plot_level_effects,
        "requires_trace": True,
        "description": "Difficulty level effects",
    },
}


# =============================================================================
# Main
# =============================================================================


def list_plots():
    """List all available plots and their descriptions."""
    logger.info("Available plots:")
    for name, info in PLOT_FUNCTIONS.items():
        logger.info(f"  {name:25s} - {info['description']}")


def generate(
    model_type: str = "baseline",
    results_dir: str | None = None,
    output_dir: str = "./output",
    plots: str | list[str] = "all",
):
    """
    Generate IRT model result plots.

    Args:
        model_type: Which IRT model results to plot (baseline, category, task, category_task)
        results_dir: Directory with IRT results (auto-determined from model_type if not provided)
        output_dir: Output directory for plots
        plots: Comma-separated plot names or "all" (default: all)
               Options: capability_heatmaps, capability_profiles, capability_comparison,
                       knowledge_vs_reasoning, lambda_forest, psi_forest,
                       variance_decomposition, scaffold_effects, level_effects

    Examples:
        python plot_irt_results.py generate --model_type=baseline
        python plot_irt_results.py generate --model_type=category_task --plots=lambda_forest,psi_forest
        python plot_irt_results.py generate --results_dir=../analysis/results/irt_baseline
    """
    # Validate model_type
    valid_model_types = ["baseline", "category", "task", "category_task"]
    if model_type not in valid_model_types:
        logger.error(f"Invalid model_type: {model_type}")
        logger.error(f"Valid options: {', '.join(valid_model_types)}")
        sys.exit(1)

    # Set results_dir based on model_type if not provided
    default_dirs = {
        "baseline": "../../analysis/results/irt_baseline",
        "category": "../../analysis/results/irt_category",
        "task": "../../analysis/results/irt_task",
        "category_task": "../../analysis/results/irt_category_task",
    }

    if results_dir is None:
        results_dir = default_dirs[model_type]
        logger.info(f"Using model_type={model_type}, results_dir={results_dir}")

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse plots argument
    if isinstance(plots, str):
        if plots == "all":
            plots_to_generate = list(PLOT_FUNCTIONS.keys())
        else:
            plots_to_generate = [p.strip() for p in plots.split(",")]
    else:
        plots_to_generate = plots

    # Validate plot names
    invalid_plots = [p for p in plots_to_generate if p not in PLOT_FUNCTIONS]
    if invalid_plots:
        logger.error(f"Invalid plot names: {', '.join(invalid_plots)}")
        logger.error(f"Valid options: {', '.join(PLOT_FUNCTIONS.keys())}")
        sys.exit(1)

    logger.info(f"Generating {len(plots_to_generate)} plots...")

    # Load theta estimates (required for all plots)
    logger.info("Loading theta estimates...")
    knowledge_theta_path = results_dir / "knowledge_theta.csv"
    reasoning_theta_path = results_dir / "reasoning_theta.csv"

    if not knowledge_theta_path.exists() or not reasoning_theta_path.exists():
        logger.error(f"Theta CSV files not found in {results_dir}")
        logger.error("Expected files: knowledge_theta.csv, reasoning_theta.csv")
        sys.exit(1)

    knowledge_theta_df = pd.read_csv(knowledge_theta_path)
    reasoning_theta_df = pd.read_csv(reasoning_theta_path)
    logger.success("Loaded theta estimates")

    # Load trace data if needed
    requires_trace = any(
        PLOT_FUNCTIONS[plot]["requires_trace"] for plot in plots_to_generate
    )

    if requires_trace:
        logger.info("Loading MCMC trace and agent data...")
        trace_path = results_dir / "agent_trace.nc"
        agent_df_path = results_dir / "agent_df.csv"
        results_json_path = results_dir / "results.json"

        if not trace_path.exists():
            logger.error(f"Trace file not found: {trace_path}")
            sys.exit(1)

        agent_trace = az.from_netcdf(trace_path)
        agent_df = pd.read_csv(agent_df_path)

        with results_json_path.open() as f:
            saved = json.load(f)
        var_result = saved["variance_decomposition"]

        env_names = sorted(agent_df["environment"].unique())
        scaffold_names = sorted(agent_df["scaffold"].unique())
        logger.success("Loaded MCMC trace and agent data")

    # Generate plots
    logger.info(f"\nGenerating plots to {output_dir}...")

    for plot_name in plots_to_generate:
        logger.info(f"Generating {plot_name}...")
        plot_info = PLOT_FUNCTIONS[plot_name]
        func = plot_info["func"]
        output_path = output_dir / f"{plot_name}.png"

        try:
            # Call appropriate plot function based on requirements
            if (
                plot_name
                in [
                    "capability_heatmaps",
                    "capability_profiles",
                    "capability_comparison",
                ]
                or plot_name == "knowledge_vs_reasoning"
            ):
                func(knowledge_theta_df, reasoning_theta_df, output_path)
            elif plot_name in ["lambda_forest", "psi_forest"]:
                func(agent_trace, env_names, output_path)
            elif plot_name == "variance_decomposition":
                func(var_result, output_path)
            elif plot_name == "scaffold_effects":
                func(agent_trace, scaffold_names, output_path)
            elif plot_name == "level_effects":
                func(agent_trace, output_path)
        except Exception as e:
            logger.error(f"Failed to generate {plot_name}: {e}")
            import traceback

            traceback.print_exc()

    logger.success(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire({"generate": generate, "list_plots": list_plots})
