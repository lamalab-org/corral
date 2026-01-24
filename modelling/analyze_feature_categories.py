"""
Feature category analysis - understanding which types of features are most predictive.

This script categorizes features by type and analyzes:
1. Which feature categories are most important?
2. Do different models prefer different feature types?
3. Are there environment-specific feature patterns?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def categorize_features(feature_name: str) -> str:
    """Categorize features by name patterns."""

    # Define feature categories based on naming patterns
    if "error" in feature_name or "fail" in feature_name:
        return "Error Metrics"
    elif (
        "tool" in feature_name
        or "bash" in feature_name
        or "read" in feature_name
        or "write" in feature_name
    ):
        return "Tool Usage"
    elif "step" in feature_name or "action" in feature_name:
        return "Step Metrics"
    elif (
        "token" in feature_name
        or "input_tokens" in feature_name
        or "output_tokens" in feature_name
    ):
        return "Token Metrics"
    elif (
        "time" in feature_name
        or "duration" in feature_name
        or "latency" in feature_name
    ):
        return "Time Metrics"
    elif (
        "rate" in feature_name
        or "ratio" in feature_name
        or "pct" in feature_name
        or "percentage" in feature_name
    ):
        return "Rate/Ratio Metrics"
    elif (
        "count" in feature_name
        or "total" in feature_name
        or "num_" in feature_name
        or "n_" in feature_name
    ):
        return "Count Metrics"
    elif (
        "max" in feature_name
        or "min" in feature_name
        or "mean" in feature_name
        or "std" in feature_name
        or "median" in feature_name
    ):
        return "Statistical Metrics"
    elif (
        "complexity" in feature_name
        or "depth" in feature_name
        or "length" in feature_name
    ):
        return "Complexity Metrics"
    elif "balance" in feature_name or "distribution" in feature_name:
        return "Distribution Metrics"
    else:
        return "Other Metrics"


def load_all_feature_importance(environments: list) -> pd.DataFrame:
    """Load all feature importance data."""
    all_importance = []

    for env in environments:
        ml_dir = Path(f"output_{env}/ml_models")

        if not ml_dir.exists():
            continue

        for model_name, model_label in [
            ("logreg", "LogReg"),
            ("rf", "RF"),
            ("xgb", "XGB"),
        ]:
            importance_file = ml_dir / f"{model_name}_feature_importance.csv"

            if importance_file.exists():
                imp_df = pd.read_csv(importance_file)

                # Get importance column
                imp_col = "abs_coefficient" if model_name == "logreg" else "importance"

                # Add metadata
                imp_df["environment"] = env
                imp_df["model"] = model_label
                imp_df["importance_value"] = imp_df[imp_col]

                # Categorize features
                imp_df["category"] = imp_df["feature"].apply(categorize_features)

                all_importance.append(
                    imp_df[
                        [
                            "feature",
                            "importance_value",
                            "environment",
                            "model",
                            "category",
                        ]
                    ]
                )

    if all_importance:
        return pd.concat(all_importance, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_category_importance(imp_df: pd.DataFrame, output_dir: Path):
    """Plot 1: Feature category importance by model."""
    logger.info("Generating feature category importance plots...")

    # Aggregate importance by category
    category_stats = (
        imp_df.groupby(["model", "category"])["importance_value"]
        .agg(
            [
                ("mean_importance", "mean"),
                ("median_importance", "median"),
                ("count", "count"),
            ]
        )
        .reset_index()
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Feature Category Importance Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Average importance by category
    ax = axes[0]
    pivot_mean = category_stats.pivot_table(
        index="category", columns="model", values="mean_importance"
    )
    pivot_mean.plot(kind="barh", ax=ax, width=0.8)

    ax.set_xlabel("Mean Importance (normalized)", fontweight="bold")
    ax.set_ylabel("Feature Category", fontweight="bold")
    ax.set_title("Average Feature Importance by Category", fontweight="bold")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="x")

    # Plot 2: Feature count by category
    ax = axes[1]
    pivot_count = category_stats.pivot_table(
        index="category", columns="model", values="count"
    )
    pivot_count.plot(kind="barh", ax=ax, width=0.8)

    ax.set_xlabel("Number of Features", fontweight="bold")
    ax.set_ylabel("Feature Category", fontweight="bold")
    ax.set_title("Feature Count by Category", fontweight="bold")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        output_dir / "category_importance_by_model.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("  ✓ Saved category_importance_by_model.png")

    # Save stats
    category_stats.to_csv(output_dir / "category_importance_stats.csv", index=False)


def plot_category_by_environment(imp_df: pd.DataFrame, output_dir: Path):
    """Plot 2: Feature category importance by environment."""
    logger.info("Generating category by environment plots...")

    # Get top 20 features per environment
    top_features = (
        imp_df.groupby("environment")
        .apply(lambda x: x.nlargest(20, "importance_value"))
        .reset_index(drop=True)
    )

    # Count categories
    category_env_counts = (
        top_features.groupby(["environment", "category"])
        .size()
        .reset_index(name="count")
    )

    # Pivot for heatmap
    pivot = category_env_counts.pivot_table(
        index="category", columns="environment", values="count"
    ).fillna(0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(
        "Feature Category Distribution in Top 20 Features by Environment",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Environment", fontweight="bold")
    ax.set_ylabel("Feature Category", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        output_dir / "category_by_environment_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logger.info("  ✓ Saved category_by_environment_heatmap.png")


def plot_top_features_by_category(imp_df: pd.DataFrame, output_dir: Path):
    """Plot 3: Top features within each category."""
    logger.info("Generating top features by category...")

    categories = imp_df["category"].unique()

    for category in categories:
        category_data = imp_df[imp_df["category"] == category]

        # Get top features by average importance across all models/environments
        top_features = (
            category_data.groupby("feature")["importance_value"].mean().nlargest(15)
        )

        if len(top_features) == 0:
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        top_features.sort_values().plot(
            kind="barh", ax=ax, color="steelblue", alpha=0.7
        )

        ax.set_xlabel("Average Importance", fontweight="bold")
        ax.set_ylabel("Feature", fontweight="bold")
        ax.set_title(
            f"Top 15 Features in Category: {category}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        safe_name = category.replace("/", "_").replace(" ", "_").lower()
        plt.savefig(
            output_dir / f"top_features_{safe_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    logger.info("  ✓ Saved top features plots for all categories")


def analyze_category_consistency(imp_df: pd.DataFrame, output_dir: Path):
    """Analyze which categories are consistently important across models/environments."""
    logger.info("Analyzing category consistency...")

    # For each category, count in how many model-environment combos it appears in top 20
    consistency_data = []

    for category in imp_df["category"].unique():
        _category_data = imp_df[imp_df["category"] == category]

        # Count appearances in top 20 per model-environment combo
        appearances = 0
        total_combos = 0

        for env in imp_df["environment"].unique():
            for model in imp_df["model"].unique():
                combo_data = imp_df[
                    (imp_df["environment"] == env) & (imp_df["model"] == model)
                ]
                top_20 = combo_data.nlargest(20, "importance_value")

                total_combos += 1
                if category in top_20["category"].to_numpy():
                    appearances += 1

        consistency_data.append(
            {
                "category": category,
                "appearances": appearances,
                "total_combos": total_combos,
                "consistency_pct": (appearances / total_combos * 100)
                if total_combos > 0
                else 0,
            }
        )

    consistency_df = pd.DataFrame(consistency_data).sort_values(
        "consistency_pct", ascending=False
    )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        range(len(consistency_df)), consistency_df["consistency_pct"], alpha=0.7
    )

    # Color by consistency
    colors = plt.cm.RdYlGn(consistency_df["consistency_pct"] / 100)
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    ax.set_yticks(range(len(consistency_df)))
    ax.set_yticklabels(consistency_df["category"])
    ax.set_xlabel("Consistency % (appears in top 20)", fontweight="bold")
    ax.set_title(
        "Feature Category Consistency Across All Models & Environments",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add percentage labels
    for i, (_idx, row) in enumerate(consistency_df.iterrows()):
        ax.text(
            row["consistency_pct"] + 1,
            i,
            f"{row['consistency_pct']:.0f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "category_consistency.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save table
    consistency_df.to_csv(output_dir / "category_consistency_analysis.csv", index=False)
    logger.info("  ✓ Saved category_consistency.png")
    logger.info("  ✓ Saved category_consistency_analysis.csv")


def generate_category_report(imp_df: pd.DataFrame, output_dir: Path):
    """Generate feature category analysis report."""
    logger.info("Generating category analysis report...")

    report = []
    report.append("# Feature Category Analysis Report")
    report.append("=" * 80)
    report.append("")

    # Overall category statistics
    report.append("## Overall Category Statistics")
    report.append("")

    category_summary = (
        imp_df.groupby("category")
        .agg(
            {
                "feature": "nunique",
                "importance_value": ["mean", "median", "std", "max"],
            }
        )
        .round(4)
    )

    category_summary.columns = [
        "Unique Features",
        "Mean Importance",
        "Median Importance",
        "Std Importance",
        "Max Importance",
    ]
    category_summary = category_summary.sort_values("Mean Importance", ascending=False)

    report.append(category_summary.to_markdown())
    report.append("")

    # Top categories by model
    report.append("## Top Categories by Model")
    report.append("")

    for model in imp_df["model"].unique():
        model_data = imp_df[imp_df["model"] == model]
        top_categories = (
            model_data.groupby("category")["importance_value"].mean().nlargest(5)
        )

        report.append(f"### {model}")
        report.append("")
        for cat, score in top_categories.items():
            report.append(f"- {cat}: {score:.4f}")
        report.append("")

    # Top categories by environment
    report.append("## Top Categories by Environment")
    report.append("")

    for env in imp_df["environment"].unique():
        env_data = imp_df[imp_df["environment"] == env]
        top_20 = env_data.nlargest(20, "importance_value")
        category_counts = top_20["category"].value_counts()

        report.append(f"### {env.upper()}")
        report.append("")
        report.append(category_counts.to_markdown())
        report.append("")

    # Save report
    report_path = output_dir / "feature_category_report.md"
    with Path(report_path).open("w") as f:
        f.write("\n".join(report))

    logger.info("  ✓ Saved feature_category_report.md")


def main():
    """Main category analysis pipeline."""
    logger.info("=" * 80)
    logger.info("FEATURE CATEGORY ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    # Define environments
    environments = ["md", "retrosynthesis", "afm", "catalyst", "ml", "resistor", "all"]

    # Create output directory
    output_dir = Path("feature_category_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}/")

    # Load all feature importance data
    logger.info("Loading feature importance data...")
    imp_df = load_all_feature_importance(environments)

    if imp_df.empty:
        logger.info("Error: No feature importance data found.")
        logger.info("Make sure you've run example_ml_models_analysis.py first.")
        return

    logger.info(f"Loaded {len(imp_df)} feature importance records")
    logger.info(f"Categories found: {imp_df['category'].nunique()}")

    # Normalize importance within each model-environment combination
    logger.info("Normalizing importance scores...")
    imp_df["importance_normalized"] = imp_df.groupby(["environment", "model"])[
        "importance_value"
    ].transform(lambda x: x / x.max() if x.max() > 0 else 0)
    imp_df["importance_value"] = imp_df["importance_normalized"]

    # Generate all plots and analyses
    plot_category_importance(imp_df, output_dir)
    plot_category_by_environment(imp_df, output_dir)
    plot_top_features_by_category(imp_df, output_dir)
    analyze_category_consistency(imp_df, output_dir)
    generate_category_report(imp_df, output_dir)

    logger.info("=" * 80)
    logger.info("CATEGORY ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"All plots and reports saved to: {output_dir}/")


if __name__ == "__main__":
    main()
