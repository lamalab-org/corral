"""
Model 3 LOO Validation Plotting Script

Generates LOO (Leave-One-Out) validation plots for Model 3:
  1. LOO prediction vs observed (scatter)
  2. Environment-level LOO prediction vs observed (bar plot)
  3. Calibration curve
  4. Task-level residuals
  5. Task-level error distribution

All plots use ONE_COL format.

Usage:
  python plot_model3_loo.py --results_dir=../../analysis/results/model3_abilities_env
"""

import importlib.util
import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_CONFIG_PATH = REPO_ROOT / "analysis" / "plot_config.py"
plot_config_spec = importlib.util.spec_from_file_location(
    "plot_config", PLOT_CONFIG_PATH
)
if plot_config_spec is None or plot_config_spec.loader is None:
    raise ImportError(f"Could not load plot config from {PLOT_CONFIG_PATH}")
plot_config = importlib.util.module_from_spec(plot_config_spec)
plot_config_spec.loader.exec_module(plot_config)

ENVIRONMENT_NAMES = plot_config.ENVIRONMENT_NAMES

# Apply style
lama_aesthetics.get_style("main")


# =============================================================================
# LOO Plotting Functions
# =============================================================================


def plot_loo_prediction_vs_observed(agent_df, output_path):
    """
    LOO prediction vs observed scatter plot (TASK-LEVEL)

    Shows average observed success rate vs average predicted success rate per task
    """
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    # Aggregate by task
    task_stats = (
        agent_df.groupby(["task"])
        .agg({"success": "mean", "p_loo": "mean"})
        .reset_index()
    )

    # Create scatter plot with uniform styling
    ax.scatter(
        task_stats["p_loo"],
        task_stats["success"],
        alpha=0.6,
        s=10,
        color="#5f59d0",
        # edgecolors='black',
        # linewidth=0.5,
    )

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect prediction")

    # Calculate task-level correlation
    r, p = stats.pearsonr(task_stats["success"], task_stats["p_loo"])
    ax.text(
        0.05,
        0.95,
        f"Correlation:\nr = {r:.3f}\n(p < 0.001)"
        if p < 0.001
        else f"Correlation:\nr = {r:.3f}\n(p = {p:.3f})",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        # bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Predicted Success Rate", fontsize=8)
    ax.set_ylabel("Observed Success Rate", fontsize=8)
    ax.set_title(
        "Model 3: Task-Level LOO Prediction vs Observed", fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    range_frame(ax, np.array([0, 1]), np.array([0, 1]), pad=0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved LOO prediction vs observed to {output_path}")


def plot_environment_level_loo(agent_df, output_path):
    """
    Environment-level bar plot: LOO prediction vs observed success rate

    Shows calibration at the environment level
    """
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    # Map environment names
    agent_df = agent_df.copy()
    agent_df["environment"] = agent_df["environment"].map(ENVIRONMENT_NAMES)

    # Calculate environment-level means
    env_stats = (
        agent_df.groupby("environment")
        .agg({"success": "mean", "p_loo": "mean"})
        .reset_index()
    )

    env_stats = env_stats.sort_values("success", ascending=False)

    x = np.arange(len(env_stats))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        env_stats["success"],
        width,
        label="Observed",
        color="#5f59d0",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        env_stats["p_loo"],
        width,
        label="LOO Predicted",
        color="#e84ab5",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xlabel("Environment", fontsize=8)
    ax.set_ylabel("Success Rate", fontsize=8)
    ax.set_title("Environment-Level: LOO vs Observed", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(env_stats["environment"], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7)

    range_frame(ax, x, np.array([0, 1]), pad=0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved environment-level LOO to {output_path}")


def plot_calibration_curve(agent_df, output_path, n_bins=10):
    """
    Calibration curve: binned predicted probabilities vs observed frequencies

    Perfect calibration = diagonal line
    """
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate observed frequency in each bin
    bin_indices = np.digitize(agent_df["p_loo"], bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    observed_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            observed_freqs.append(agent_df.loc[mask, "success"].mean())
            bin_counts.append(mask.sum())
        else:
            observed_freqs.append(np.nan)
            bin_counts.append(0)

    # Plot calibration curve
    valid = ~np.isnan(observed_freqs)
    ax.plot(
        bin_centers[valid],
        np.array(observed_freqs)[valid],
        marker="o",
        markersize=6,
        linewidth=2,
        color="#5f59d0",
        label="Model calibration",
    )

    # Add perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Perfect calibration")

    # Add histogram of predictions at bottom
    ax_hist = ax.inset_axes([0.1, 0.02, 0.8, 0.15])
    ax_hist.hist(
        agent_df["p_loo"],
        bins=bins,
        color="#e84ab5",
        alpha=0.5,
        edgecolor="black",
    )
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.set_xlabel("Predicted P(success)", fontsize=6)

    ax.set_xlabel("Predicted P(success)", fontsize=8)
    ax.set_ylabel("Observed Frequency", fontsize=8)
    ax.set_title("Calibration Curve", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)

    range_frame(ax, np.array([0, 1]), np.array([0, 1]), pad=0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved calibration curve to {output_path}")


def plot_task_level_residuals(agent_df, output_path):
    """
    Task-level residuals: mean residual per task

    Shows which tasks have systematic over/under-prediction
    """
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    # Calculate task-level residuals
    task_stats = (
        agent_df.groupby("task").agg({"success": "mean", "p_loo": "mean"}).reset_index()
    )
    task_stats["residual"] = task_stats["success"] - task_stats["p_loo"]
    task_stats = task_stats.sort_values("residual")

    # Show top 20 positive and top 20 negative
    n_show = 20
    top_positive = task_stats.tail(n_show)
    top_negative = task_stats.head(n_show)
    to_plot = pd.concat([top_negative, top_positive])

    colors = ["#ff0677" if r < 0 else "#0051ff" for r in to_plot["residual"]]

    y_pos = np.arange(len(to_plot))
    ax.barh(y_pos, to_plot["residual"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(to_plot["task"], fontsize=5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Residual (Observed - Predicted)", fontsize=8)
    ax.set_title(
        f"Task-Level Residuals (Top {n_show} each direction)",
        fontsize=9,
        fontweight="bold",
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#ff0677", label="Under-predicted"),
        Patch(facecolor="#0051ff", label="Over-predicted"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    range_frame(ax, to_plot["residual"], y_pos, pad=0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved task-level residuals to {output_path}")


def plot_error_distribution(agent_df, output_path):
    """
    Task-level error distribution

    Shows distribution of absolute errors at task level
    """
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    # Calculate task-level absolute errors
    task_stats = (
        agent_df.groupby("task").agg({"success": "mean", "p_loo": "mean"}).reset_index()
    )
    task_stats["abs_error"] = np.abs(task_stats["success"] - task_stats["p_loo"])

    # Histogram
    ax.hist(
        task_stats["abs_error"],
        bins=30,
        color="#5f59d0",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add statistics
    mean_error = task_stats["abs_error"].mean()
    median_error = task_stats["abs_error"].median()

    ax.axvline(
        mean_error,
        color="#e84ab5",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_error:.3f}",
    )
    ax.axvline(
        median_error,
        color="#ff0677",
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_error:.3f}",
    )

    ax.set_xlabel("Absolute Error |Observed - Predicted|", fontsize=8)
    ax.set_ylabel("Number of Tasks", fontsize=8)
    ax.set_title("Task-Level Error Distribution", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)

    # Calculate MAE and RMSE
    mae = task_stats["abs_error"].mean()
    rmse = np.sqrt(((task_stats["success"] - task_stats["p_loo"]) ** 2).mean())

    ax.text(
        0.98,
        0.98,
        f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved error distribution to {output_path}")


# =============================================================================
# Main
# =============================================================================


def generate(
    results_dir: str = "../../analysis/results/model3_abilities_env",
    output_dir: str = "./output",
):
    """
    Generate Model 3 LOO validation plots.

    Args:
        results_dir: Directory with Model 3 results
        output_dir: Output directory for plots

    Examples:
        python plot_model3_loo.py generate
        python plot_model3_loo.py generate --results_dir=../../analysis/results/model3_abilities_env
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Model 3 results from {results_dir}")

    # Load agent data with LOO predictions
    agent_df_path = results_dir / "agent_df.csv"
    if not agent_df_path.exists():
        logger.error(f"Agent data not found: {agent_df_path}")
        logger.error("Make sure Model 3 has been fitted and saved LOO predictions")
        sys.exit(1)

    agent_df = pd.read_csv(agent_df_path)

    # Check for LOO predictions
    if "p_loo" not in agent_df.columns:
        logger.error("LOO predictions not found in agent_df.csv")
        logger.error(
            "Make sure the model was fitted with idata_kwargs={'log_likelihood': True}"
        )
        sys.exit(1)

    logger.success(f"Loaded agent data with {len(agent_df):,} observations")

    # Generate all plots
    logger.info("\nGenerating LOO validation plots...")

    plot_loo_prediction_vs_observed(
        agent_df, output_dir / "loo_prediction_vs_observed.png"
    )

    plot_environment_level_loo(agent_df, output_dir / "environment_level_loo.png")

    plot_calibration_curve(agent_df, output_dir / "calibration_curve.png")

    plot_task_level_residuals(agent_df, output_dir / "task_level_residuals.png")

    plot_error_distribution(agent_df, output_dir / "task_level_error_distribution.png")

    logger.success(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire({"generate": generate})
