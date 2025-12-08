"""
Comprehensive analysis of ML model results across all environments.

This script analyzes the results from example_ml_models_analysis.py and generates:
1. Model performance comparison plots
2. Feature importance analysis and plots
3. Overfitting analysis
4. Cross-environment comparisons
5. Detailed analysis report
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_all_results(environments: list[str]) -> dict:
    """Load all ML results from JSON files."""
    all_results = {}

    for env in environments:
        env_results = {}
        ml_dir = Path(f"output_{env}/ml_models")

        if not ml_dir.exists():
            logger.info(f"Warning: {ml_dir} not found, skipping {env}")
            continue

        # Load each model's metrics
        for model_name in ["logreg", "rf", "xgb"]:
            metrics_file = ml_dir / f"{model_name}_metrics.json"
            importance_file = ml_dir / f"{model_name}_feature_importance.csv"

            if metrics_file.exists():
                with metrics_file.open() as f:
                    env_results[model_name] = json.load(f)

            if importance_file.exists():
                env_results[f"{model_name}_importance"] = pd.read_csv(importance_file)

        if env_results:
            all_results[env] = env_results

    return all_results


def plot_model_comparison(all_results: dict, output_dir: Path):
    """Plot 1: Model Performance Comparison across environments."""
    logger.info("Generating Plot 1: Model Performance Comparison...")

    # Prepare data
    comparison_data = []

    for env, results in all_results.items():
        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            if model_name in results:
                metrics = results[model_name]
                comparison_data.append(
                    {
                        "Environment": env,
                        "Model": model_label,
                        "Test ROC-AUC": metrics.get("test_roc_auc", np.nan),
                        "Test F1": metrics.get("test_f1_score", np.nan),
                        "Test Accuracy": metrics.get("test_accuracy", np.nan),
                        "Test Precision": metrics.get("test_precision", np.nan),
                        "Test Recall": metrics.get("test_recall", np.nan),
                        "CV Mean": metrics.get("cv_mean", np.nan),
                    }
                )

    df = pd.DataFrame(comparison_data)

    # Create subplot for multiple metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Model Performance Comparison Across Environments",
        fontsize=16,
        fontweight="bold",
    )

    metrics_to_plot = [
        ("Test ROC-AUC", "ROC-AUC (Test Set)"),
        ("Test F1", "F1 Score (Test Set)"),
        ("Test Accuracy", "Accuracy (Test Set)"),
        ("CV Mean", "CV ROC-AUC Mean"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        # Pivot for grouped bar chart
        pivot_df = df.pivot(index="Environment", columns="Model", values=metric)
        pivot_df.plot(kind="bar", ax=ax, width=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment")
        ax.set_ylabel(metric)
        ax.legend(title="Model")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "plot1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ Saved plot1_model_comparison.png")

    # Save comparison table
    df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    logger.info("  ✓ Saved model_comparison_metrics.csv")


def plot_overfitting_analysis(all_results: dict, output_dir: Path):
    """Plot 2: Train vs Test Performance (Overfitting Detection)."""
    logger.info("Generating Plot 2: Overfitting Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Overfitting Analysis: Train vs Test ROC-AUC", fontsize=16, fontweight="bold"
    )

    for idx, (model_name, model_label) in enumerate(
        [("logreg", "Logistic Regression"), ("rf", "Random Forest"), ("xgb", "XGBoost")]
    ):
        ax = axes[idx]

        train_scores = []
        test_scores = []
        env_labels = []

        for env, results in all_results.items():
            if model_name in results:
                metrics = results[model_name]
                train_scores.append(metrics.get("train_roc_auc", np.nan))
                test_scores.append(metrics.get("test_roc_auc", np.nan))
                env_labels.append(env)

        # Scatter plot
        ax.scatter(train_scores, test_scores, s=100, alpha=0.7)

        # Add environment labels
        for i, env in enumerate(env_labels):
            ax.annotate(
                env,
                (train_scores[i], test_scores[i]),
                fontsize=8,
                ha="right",
                va="bottom",
            )

        # Add diagonal line (perfect fit)
        lims = [0, 1]
        ax.plot(lims, lims, "k--", alpha=0.5, zorder=0, label="No overfitting")

        # Add overfitting threshold
        ax.fill_between(
            [0, 1], [0, 1], [0, 0.9], alpha=0.1, color="red", label="Overfitting region"
        )

        ax.set_xlabel("Train ROC-AUC")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(model_label, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot2_overfitting_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("  ✓ Saved plot2_overfitting_analysis.png")


def plot_feature_importance_comparison(all_results: dict, output_dir: Path):
    """Plot 3: Feature Importance Comparison across models."""
    logger.info("Generating Plot 3: Feature Importance Comparison...")

    for env, results in all_results.items():

        # Get top 20 features from each model
        top_features = set()

        for model_name in ["logreg", "rf", "xgb"]:
            imp_key = f"{model_name}_importance"
            if imp_key in results:
                imp_df = results[imp_key]

                # Get appropriate column
                if model_name == "logreg":
                    imp_col = "abs_coefficient"
                else:
                    imp_col = "importance"

                top_features.update(imp_df.head(20)["feature"].tolist())

        if not top_features:
            continue

        # Create comparison dataframe
        feature_comparison = []

        for feature in top_features:
            row = {"feature": feature}

            for model_name in ["logreg", "rf", "xgb"]:
                imp_key = f"{model_name}_importance"
                if imp_key in results:
                    imp_df = results[imp_key]

                    if model_name == "logreg":
                        imp_col = "abs_coefficient"
                    else:
                        imp_col = "importance"

                    # Find feature importance
                    feat_row = imp_df[imp_df["feature"] == feature]
                    if not feat_row.empty:
                        row[model_name] = feat_row[imp_col].values[0]
                    else:
                        row[model_name] = 0

            feature_comparison.append(row)

        comp_df = pd.DataFrame(feature_comparison)

        # Normalize each model's importance to 0-1 scale
        for model in ["logreg", "rf", "xgb"]:
            if model in comp_df.columns:
                max_val = comp_df[model].max()
                if max_val > 0:
                    comp_df[model] = comp_df[model] / max_val

        # Sort by average importance
        comp_df["avg_importance"] = comp_df[["logreg", "rf", "xgb"]].mean(axis=1)
        comp_df = comp_df.sort_values("avg_importance", ascending=False).head(20)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))

        x = np.arange(len(comp_df))
        width = 0.25

        bars1 = ax.barh(x - width, comp_df["logreg"], width, label="LogReg", alpha=0.8)
        bars2 = ax.barh(x, comp_df["rf"], width, label="RF", alpha=0.8)
        bars3 = ax.barh(x + width, comp_df["xgb"], width, label="XGB", alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(comp_df["feature"])
        ax.set_xlabel("Normalized Importance", fontweight="bold")
        ax.set_title(
            f"Top 20 Feature Importance Comparison: {env.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"plot3_feature_importance_{env}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save comparison table
        comp_df.to_csv(
            output_dir / f"feature_importance_comparison_{env}.csv", index=False
        )

    logger.info("  ✓ Saved feature importance plots for all environments")


def plot_feature_agreement_heatmap(all_results: dict, output_dir: Path):
    """Plot 4: Feature Agreement Heatmap across models and environments."""
    logger.info("Generating Plot 4: Feature Agreement Analysis...")

    # Collect top 15 features from each model in each environment
    all_top_features = set()
    feature_ranks = {}

    for env in all_results:
    for env, results in all_results.items():

        for model_name in ["logreg", "rf", "xgb"]:
            imp_key = f"{model_name}_importance"
                imp_df = results[imp_key].head(15)

                for idx, row in imp_df.iterrows():
                    feature = row["feature"]
                    all_top_features.add(feature)

                    key = f"{env}_{model_name}"
                    if key not in feature_ranks:
                        feature_ranks[key] = {}

                    # Store rank (1-15)
                    feature_ranks[key][feature] = idx + 1

    # Create matrix
    feature_list = sorted(all_top_features)
    model_env_combos = sorted(feature_ranks.keys())

    matrix = np.zeros((len(feature_list), len(model_env_combos)))

    for i, feature in enumerate(feature_list):
        for j, combo in enumerate(model_env_combos):
            if feature in feature_ranks[combo]:
                # Lower rank = more important, so invert for better visualization
                matrix[i, j] = 16 - feature_ranks[combo][feature]
            else:
                matrix[i, j] = 0

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 20))

    sns.heatmap(
        matrix,
        xticklabels=[c.replace("_", "\n") for c in model_env_combos],
        yticklabels=feature_list,
        cmap="YlOrRd",
        cbar_kws={"label": "Importance Rank (15=highest)"},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        "Feature Importance Heatmap: All Models & Environments",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Environment_Model", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot4_feature_agreement_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("  ✓ Saved plot4_feature_agreement_heatmap.png")


def analyze_universal_features(all_results: dict, output_dir: Path):
    """Plot 5 & Analysis: Universal features important across all models/environments."""
    logger.info("Generating Plot 5: Universal Features Analysis...")

    # Count how many times each feature appears in top 10
    feature_counts = {}

    for env in all_results:
        results = all_results[env]

        for model_name in ["logreg", "rf", "xgb"]:
            imp_key = f"{model_name}_importance"
            if imp_key in results:
                imp_df = results[imp_key].head(10)

                for feature in imp_df["feature"]:
                    if feature not in feature_counts:
                        feature_counts[feature] = 0
                    feature_counts[feature] += 1

    # Convert to dataframe and sort
    universal_df = pd.DataFrame(
        [
            {
                "feature": feat,
                "count": count,
                "percentage": count / (len(all_results) * 3) * 100,
            }
            for feat, count in feature_counts.items()
        ]
    ).sort_values("count", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    top_universal = universal_df.head(30)

    bars = ax.barh(range(len(top_universal)), top_universal["count"], alpha=0.7)

    # Color by importance level
    colors = plt.cm.RdYlGn(top_universal["count"] / top_universal["count"].max())
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    ax.set_yticks(range(len(top_universal)))
    ax.set_yticklabels(top_universal["feature"])
    ax.set_xlabel(
        f"Times in Top 10 (out of {len(all_results) * 3} model-environment combinations)",
        fontweight="bold",
    )
    ax.set_title(
        "Universal Features: Consistency Across All Models & Environments",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add percentage labels
    for i, (idx, row) in enumerate(top_universal.iterrows()):
        ax.text(
            row["count"] + 0.2, i, f"{row['percentage']:.0f}%", va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot5_universal_features.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save table
    universal_df.to_csv(output_dir / "universal_features_analysis.csv", index=False)
    logger.info("  ✓ Saved plot5_universal_features.png")
    logger.info("  ✓ Saved universal_features_analysis.csv")


def plot_environment_difficulty(all_results: dict, output_dir: Path):
    """Plot 6: Environment Difficulty (which are hardest to predict?)."""
    logger.info("Generating Plot 6: Environment Difficulty Analysis...")

    env_performance = []

    for env, results in all_results.items():
        env_metrics = {"environment": env}

        # Get best model performance
        best_test_auc = 0
        avg_test_auc = []

        for model_name in ["logreg", "rf", "xgb"]:
            if model_name in results:
                test_auc = results[model_name].get("test_roc_auc", 0)
                avg_test_auc.append(test_auc)
                best_test_auc = max(best_test_auc, test_auc)

        env_metrics["best_test_auc"] = best_test_auc
        env_metrics["avg_test_auc"] = np.mean(avg_test_auc) if avg_test_auc else 0
        env_metrics["n_samples"] = results.get("logreg", {}).get("n_samples", 0)

        env_performance.append(env_metrics)

    env_df = pd.DataFrame(env_performance).sort_values("avg_test_auc")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Environment Difficulty Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Average performance
    ax = axes[0]
    bars = ax.barh(env_df["environment"], env_df["avg_test_auc"], alpha=0.7)

    # Color by difficulty (red = hard, green = easy)
    colors = plt.cm.RdYlGn(env_df["avg_test_auc"])
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    ax.set_xlabel("Average Test ROC-AUC", fontweight="bold")
    ax.set_title("Average Model Performance by Environment", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (idx, row) in enumerate(env_df.iterrows()):
        ax.text(
            row["avg_test_auc"] + 0.02,
            i,
            f"{row['avg_test_auc']:.3f}",
            va="center",
            fontsize=9,
        )

    # Plot 2: Performance vs Dataset Size
    ax = axes[1]
    scatter = ax.scatter(
        env_df["n_samples"],
        env_df["avg_test_auc"],
        s=200,
        alpha=0.6,
        c=env_df["avg_test_auc"],
        cmap="RdYlGn",
        edgecolors="black",
    )

    for idx, row in env_df.iterrows():
        ax.annotate(
            row["environment"],
            (row["n_samples"], row["avg_test_auc"]),
            fontsize=9,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Dataset Size (n_samples)", fontweight="bold")
    ax.set_ylabel("Average Test ROC-AUC", fontweight="bold")
    ax.set_title("Performance vs Dataset Size", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="ROC-AUC")

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot6_environment_difficulty.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save table
    env_df.to_csv(output_dir / "environment_difficulty_analysis.csv", index=False)
    logger.info("  ✓ Saved plot6_environment_difficulty.png")
    logger.info("  ✓ Saved environment_difficulty_analysis.csv")


def plot_cv_stability(all_results: dict, output_dir: Path):
    """Plot 7: Cross-Validation Stability Analysis."""
    logger.info("Generating Plot 7: CV Stability Analysis...")

    cv_data = []

    for env, results in all_results.items():
        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            if model_name in results:
                metrics = results[model_name]
                cv_data.append(
                    {
                        "Environment": env,
                        "Model": model_label,
                        "CV Mean": metrics.get("cv_mean", np.nan),
                        "CV Std": metrics.get("cv_std", np.nan),
                    }
                )

    cv_df = pd.DataFrame(cv_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cross-Validation Stability Analysis", fontsize=16, fontweight="bold")

    # Plot 1: CV Mean with error bars
    ax = axes[0]

    for model in ["LogReg", "RF", "XGB"]:
        model_data = cv_df[cv_df["Model"] == model].sort_values("Environment")
        x = range(len(model_data))
        ax.errorbar(
            x,
            model_data["CV Mean"],
            yerr=model_data["CV Std"],
            label=model,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
        )

    ax.set_xticks(range(len(model_data)))
    ax.set_xticklabels(model_data["Environment"], rotation=45, ha="right")
    ax.set_ylabel("CV ROC-AUC Mean ± Std", fontweight="bold")
    ax.set_xlabel("Environment", fontweight="bold")
    ax.set_title("Cross-Validation Scores by Environment", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 2: CV Std (stability measure - lower is better)
    ax = axes[1]

    pivot_df = cv_df.pivot(index="Environment", columns="Model", values="CV Std")
    pivot_df.plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylabel("CV Standard Deviation (lower = more stable)", fontweight="bold")
    ax.set_xlabel("Environment", fontweight="bold")
    ax.set_title("Model Stability Comparison", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "plot7_cv_stability.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ Saved plot7_cv_stability.png")


def plot_precision_recall_tradeoff(all_results: dict, output_dir: Path):
    """Plot 8: Precision vs Recall tradeoff analysis."""
    logger.info("Generating Plot 8: Precision-Recall Tradeoff...")

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = {"LogReg": "blue", "RF": "green", "XGB": "red"}
    markers = {"LogReg": "o", "RF": "s", "XGB": "^"}

    for env, results in all_results.items():
        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            if model_name in results:
                metrics = results[model_name]
                precision = metrics.get("test_precision", np.nan)
                recall = metrics.get("test_recall", np.nan)

                ax.scatter(
                    recall,
                    precision,
                    s=150,
                    alpha=0.6,
                    color=colors[model_label],
                    marker=markers[model_label],
                    edgecolors="black",
                    linewidth=1.5,
                )

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="LogReg",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="RF",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="XGB",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=12)

    ax.set_xlabel("Recall (Test Set)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Precision (Test Set)", fontweight="bold", fontsize=12)
    ax.set_title(
        "Precision vs Recall Tradeoff: All Models & Environments",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Add diagonal line for reference (F1 isolines could be added)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal P/R")

    plt.tight_layout()
    plt.savefig(
        output_dir / "plot8_precision_recall_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("  ✓ Saved plot8_precision_recall_tradeoff.png")


def generate_comprehensive_report(all_results: dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    logger.info("Generating comprehensive analysis report...")

    report = []
    report.append("# Comprehensive ML Models Analysis Report")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    report.append("## Executive Summary")
    report.append("")

    # Best performing model overall
    best_overall = {"model": "", "env": "", "auc": 0}
    worst_overall = {"model": "", "env": "", "auc": 1}

    for env, results in all_results.items():
        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            if model_name in results:
                auc = results[model_name].get("test_roc_auc", 0)
                if auc > best_overall["auc"]:
                    best_overall = {"model": model_label, "env": env, "auc": auc}
                if auc < worst_overall["auc"]:
                    worst_overall = {"model": model_label, "env": env, "auc": auc}

    report.append(
        f"- **Best Performance**: {best_overall['model']} on {best_overall['env']} "
        f"(Test ROC-AUC: {best_overall['auc']:.4f})"
    )
    report.append(
        f"- **Worst Performance**: {worst_overall['model']} on {worst_overall['env']} "
        f"(Test ROC-AUC: {worst_overall['auc']:.4f})"
    )
    report.append(f"- **Environments Analyzed**: {len(all_results)}")
    report.append("")

    # Model rankings
    report.append("## Model Rankings by Average Performance")
    report.append("")

    model_avg_scores = {"LogReg": [], "RF": [], "XGB": []}

    for env, results in all_results.items():
        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            if model_name in results:
                auc = results[model_name].get("test_roc_auc", np.nan)
                if not np.isnan(auc):
                    model_avg_scores[model_label].append(auc)

    model_rankings = []
    for model, scores in model_avg_scores.items():
        if scores:
            model_rankings.append(
                {
                    "Model": model,
                    "Avg ROC-AUC": np.mean(scores),
                    "Std": np.std(scores),
                    "Min": np.min(scores),
                    "Max": np.max(scores),
                }
            )

    ranking_df = pd.DataFrame(model_rankings).sort_values(
        "Avg ROC-AUC", ascending=False
    )
    report.append(ranking_df.to_markdown(index=False))
    report.append("")

    # Environment rankings
    report.append("## Environment Rankings by Predictability")
    report.append("")

    env_scores = []
    for env, results in all_results.items():
        scores = []
        for model_name in ["logreg", "rf", "xgb"]:
            if model_name in results:
                auc = results[model_name].get("test_roc_auc", np.nan)
                if not np.isnan(auc):
                    scores.append(auc)

        if scores:
            env_scores.append(
                {
                    "Environment": env,
                    "Avg ROC-AUC": np.mean(scores),
                    "Best Model AUC": np.max(scores),
                    "Worst Model AUC": np.min(scores),
                    "Dataset Size": results.get("logreg", {}).get("n_samples", 0),
                }
            )

    env_df = pd.DataFrame(env_scores).sort_values("Avg ROC-AUC", ascending=False)
    report.append(env_df.to_markdown(index=False))
    report.append("")

    # Key insights
    report.append("## Key Insights")
    report.append("")
    report.append("### 1. Model Performance")
    easiest_env = env_df.iloc[0]["Environment"]
    hardest_env = env_df.iloc[-1]["Environment"]
    report.append(
        f"- Easiest to predict: **{easiest_env}** "
        f"(Avg ROC-AUC: {env_df.iloc[0]['Avg ROC-AUC']:.4f})"
    )
    report.append(
        f"- Hardest to predict: **{hardest_env}** "
        f"(Avg ROC-AUC: {env_df.iloc[-1]['Avg ROC-AUC']:.4f})"
    )
    report.append("")

    report.append("### 2. Overfitting Analysis")
    report.append("Check Plot 2 for train-test gaps. Large gaps indicate overfitting.")
    report.append("")

    report.append("### 3. Feature Importance")
    report.append(
        "See Plot 4 and Plot 5 for universal features across models/environments."
    )
    report.append("")

    report.append("### 4. Model Stability")
    report.append(
        "See Plot 7 for cross-validation stability. Lower CV std = more stable model."
    )
    report.append("")

    # Save report
    report_path = output_dir / "comprehensive_analysis_report.md"
    with report_path.open("w") as f:
        f.write("\n".join(report))

    logger.info("  ✓ Saved comprehensive_analysis_report.md")


def main():
    """Main analysis pipeline."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ML RESULTS ANALYSIS")
    logger.info("=" * 80)

    # Define environments
    environments = ["md", "retrosynthesis", "afm", "catalyst", "ml", "resistor", "all"]

    # Create output directory
    output_dir = Path("ml_analysis_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}/")

    # Load all results
    logger.info("Loading ML results from all environments...")
    all_results = load_all_results(environments)
    logger.info(f"Loaded results for {len(all_results)} environments")

    if not all_results:
        logger.info(
            "Error: No results found. Make sure you've run example_ml_models_analysis.py first."
        )
        return

    # Generate all plots and analyses
    plot_model_comparison(all_results, output_dir)
    plot_overfitting_analysis(all_results, output_dir)
    plot_feature_importance_comparison(all_results, output_dir)
    plot_feature_agreement_heatmap(all_results, output_dir)
    analyze_universal_features(all_results, output_dir)
    plot_environment_difficulty(all_results, output_dir)
    plot_cv_stability(all_results, output_dir)
    plot_precision_recall_tradeoff(all_results, output_dir)
    generate_comprehensive_report(all_results, output_dir)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"All plots and reports saved to: {output_dir}/")
    logger.info("Generated files:")
    logger.info("  - plot1_model_comparison.png")
    logger.info("  - plot2_overfitting_analysis.png")
    logger.info("  - plot3_feature_importance_<env>.png (for each environment)")
    logger.info("  - plot4_feature_agreement_heatmap.png")
    logger.info("  - plot5_universal_features.png")
    logger.info("  - plot6_environment_difficulty.png")
    logger.info("  - plot7_cv_stability.png")
    logger.info("  - plot8_precision_recall_tradeoff.png")
    logger.info("  - comprehensive_analysis_report.md")
    logger.info("  - Various CSV files with detailed data")


if __name__ == "__main__":
    main()
