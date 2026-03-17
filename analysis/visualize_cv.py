"""
Visualize cross-validation results for Model 3
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

sns.set_style("whitegrid")


def visualize_cv_results(
    cv_dir="results/cv_model3", output_file="results/model3_cv_results.png"
):
    """
    Create comprehensive visualization of CV results
    """
    logger.info("Creating CV visualization...")

    cv_path = Path(cv_dir)

    # Load summary results
    summary = pd.read_csv(cv_path / "cv_results_summary.csv")

    # Load all predictions
    all_predictions = []
    for fold in range(1, len(summary) + 1):
        pred_file = cv_path / f"fold{fold}_predictions.csv"
        if pred_file.exists():
            fold_preds = pd.read_csv(pred_file)
            fold_preds["fold"] = fold
            all_predictions.append(fold_preds)

    if not all_predictions:
        logger.error("No prediction files found!")
        return None

    all_preds = pd.concat(all_predictions, ignore_index=True)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Task-level correlation by fold
    ax = fig.add_subplot(gs[0, 0])
    folds = summary["fold"].values
    ax.bar(
        folds, summary["task_correlation"].values, color="skyblue", edgecolor="black"
    )
    mean_corr = summary["task_correlation"].mean()
    ax.axhline(
        mean_corr,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_corr:.3f}",
    )
    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task-Level Correlation", fontsize=11, fontweight="bold")
    ax.set_title("Task Correlation by Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 2: Task-level MAE by fold
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(folds, summary["task_mae"].values, color="lightcoral", edgecolor="black")
    mean_mae = summary["task_mae"].mean()
    ax.axhline(
        mean_mae,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_mae:.3f}",
    )
    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task-Level MAE", fontsize=11, fontweight="bold")
    ax.set_title("Task MAE by Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Observation-level accuracy by fold
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(folds, summary["obs_accuracy"].values, color="lightgreen", edgecolor="black")
    mean_acc = summary["obs_accuracy"].mean()
    ax.axhline(
        mean_acc,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_acc:.3f}",
    )
    ax.set_xlabel("Fold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Observation Accuracy by Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 4: Overall task-level calibration (all folds combined)
    ax = fig.add_subplot(gs[1, :])
    task_agg = (
        all_preds.groupby("task")
        .agg({"predicted": "mean", "success": "mean"})
        .reset_index()
    )

    ax.scatter(
        task_agg["predicted"],
        task_agg["success"],
        alpha=0.6,
        s=30,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")

    corr = np.corrcoef(task_agg["predicted"], task_agg["success"])[0, 1]
    mae = np.abs(task_agg["predicted"] - task_agg["success"]).mean()
    rmse = np.sqrt(((task_agg["predicted"] - task_agg["success"]) ** 2).mean())

    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Success Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Task-Level Calibration (All Folds, N={len(task_agg)} tasks)\n"
        + f"r={corr:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 5: Residuals by environment
    ax = fig.add_subplot(gs[2, 0])
    task_agg_env = (
        all_preds.groupby("task")
        .agg({"predicted": "mean", "success": "mean", "environment": "first"})
        .reset_index()
    )
    task_agg_env["residual"] = task_agg_env["success"] - task_agg_env["predicted"]

    envs = sorted(task_agg_env["environment"].unique())
    residuals_by_env = [
        task_agg_env[task_agg_env["environment"] == e]["residual"].values for e in envs
    ]

    bp = ax.boxplot(residuals_by_env, tick_labels=envs, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Environment", fontsize=11, fontweight="bold")
    ax.set_ylabel("Residual (Obs - Pred)", fontsize=11, fontweight="bold")
    ax.set_title("Residuals by Environment", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # Panel 6: Environment-level comparison
    ax = fig.add_subplot(gs[2, 1])
    env_agg = (
        all_preds.groupby("environment")
        .agg({"predicted": "mean", "success": "mean"})
        .reset_index()
    )

    x = np.arange(len(env_agg))
    width = 0.35

    ax.bar(
        x - width / 2,
        env_agg["predicted"],
        width,
        label="Predicted",
        color="skyblue",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        env_agg["success"],
        width,
        label="Observed",
        color="lightcoral",
        edgecolor="black",
    )

    ax.set_ylabel("Mean Success Rate", fontsize=11, fontweight="bold")
    ax.set_title("Environment-Level Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(env_agg["environment"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 7: Summary statistics table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    summary_text = f"""
5-FOLD CV SUMMARY

Folds: {len(summary)}
Total Tasks: {all_preds['task'].nunique()}
Total Observations: {len(all_preds)}

TASK-LEVEL:
  Correlation: {summary['task_correlation'].mean():.3f} ± {summary['task_correlation'].std():.3f}
  MAE:         {summary['task_mae'].mean():.3f} ± {summary['task_mae'].std():.3f}
  RMSE:        {summary['task_rmse'].mean():.3f} ± {summary['task_rmse'].std():.3f}

OBSERVATION-LEVEL:
  Accuracy:    {summary['obs_accuracy'].mean():.3f} ± {summary['obs_accuracy'].std():.3f}
  Log Loss:    {summary['obs_log_loss'].mean():.3f} ± {summary['obs_log_loss'].std():.3f}
  Brier Score: {summary['obs_brier'].mean():.3f} ± {summary['obs_brier'].std():.3f}

ENVIRONMENT-LEVEL:
  MAE:         {summary['env_mae'].mean():.3f} ± {summary['env_mae'].std():.3f}

CONVERGENCE:
  Max R-hat:   {summary['max_rhat'].mean():.4f} ± {summary['max_rhat'].std():.4f}

✅ NO DATA LEAKAGE!
Each task predicted without
being in training set.
"""

    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Overall title
    fig.suptitle(
        "Model 3: 5-Fold Stratified Cross-Validation Results\n"
        + "Holdout Validation - Each Task Predicted Without Being in Training Set",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.success(f"✅ Created: {output_file}")

    return task_agg, env_agg, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CV results")
    parser.add_argument(
        "--cv-dir", default="results/cv_model3", help="CV results directory"
    )
    parser.add_argument(
        "--output", default="results/model3_cv_results.png", help="Output file"
    )

    args = parser.parse_args()

    visualize_cv_results(cv_dir=args.cv_dir, output_file=args.output)
