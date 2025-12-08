"""
ML models (Logistic Regression, Random Forest, XGBoost) for binary outcome prediction.

This script:
1. Loads trace data and extracts features
2. Runs ML model analysis (LogReg, RF, XGBoost) with train/test split and cross-validation
3. Compares model performance
4. Generates PyCM HTML reports for each model
5. Saves feature importance and metrics to CSV/JSON
6. Creates a comprehensive markdown report comparing all models
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add the library to the path
sys.path.insert(0, str(Path(__file__).parent))

from corral_trace_analyzer import (
    BinaryOutcomeAnalyzer,
    FeatureExtractor,
    TraceDataLoader,
)
from loguru import logger


def format_metrics_table(metrics_dict: dict) -> str:
    """Format metrics into a markdown table."""
    rows = []
    rows.append("| Metric | Value |")
    rows.append("|--------|-------|")

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            rows.append(f"| {key} | {value:.4f} |")
        else:
            rows.append(f"| {key} | {value} |")

    return "\n".join(rows)


def generate_model_comparison_report(
    env_name: str,
    logreg_results: dict,
    rf_results: dict,
    xgb_results: dict,
) -> str:
    """Generate a comprehensive comparison report for all models."""
    report = []
    report.append(f"# ML Model Comparison Report: {env_name.upper()}")
    report.append("=" * 80)
    report.append("")

    # Model comparison table
    report.append("## Model Performance Comparison (Test Set)")
    report.append("")

    comparison_data = []

    for model_name, results in [
        ("Logistic Regression", logreg_results),
        ("Random Forest", rf_results),
        ("XGBoost", xgb_results),
    ]:
        if "error" not in results:
            comparison_data.append(
                {
                    "Model": model_name,
                    "Test Accuracy": f"{results.get('test_accuracy', 0):.4f}",
                    "Test Precision": f"{results.get('test_precision', 0):.4f}",
                    "Test Recall": f"{results.get('test_recall', 0):.4f}",
                    "Test F1": f"{results.get('test_f1_score', 0):.4f}",
                    "Test ROC-AUC": f"{results.get('test_roc_auc', 0):.4f}",
                    "CV Mean ± Std": f"{results.get('cv_mean', 0):.4f} ± {results.get('cv_std', 0):.4f}",
                }
            )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        report.append(comparison_df.to_markdown(index=False))
    else:
        report.append("No valid model results to compare.")

    report.append("")
    report.append("---")
    report.append("")

    # Logistic Regression Details
    report.append("## 1. Logistic Regression")
    if "error" not in logreg_results:
        report.append("")
        report.append("### Performance Metrics")
        report.append("")
        report.append("#### Test Set (Primary Metrics)")
        test_metrics = {
            "Accuracy": logreg_results["test_accuracy"],
            "Precision": logreg_results["test_precision"],
            "Recall": logreg_results["test_recall"],
            "F1 Score": logreg_results["test_f1_score"],
            "ROC-AUC": logreg_results["test_roc_auc"],
        }
        report.append(format_metrics_table(test_metrics))

        report.append("")
        report.append("#### Training Set")
        train_metrics = {
            "Accuracy": logreg_results["train_accuracy"],
            "Precision": logreg_results["train_precision"],
            "Recall": logreg_results["train_recall"],
            "F1 Score": logreg_results["train_f1_score"],
            "ROC-AUC": logreg_results["train_roc_auc"],
        }
        report.append(format_metrics_table(train_metrics))

        report.append("")
        report.append("#### Cross-Validation")
        cv_metrics = {
            "CV ROC-AUC (mean)": logreg_results["cv_mean"],
            "CV ROC-AUC (std)": logreg_results["cv_std"],
        }
        report.append(format_metrics_table(cv_metrics))

        # PyCM metrics if available
        if logreg_results.get("pycm_metrics"):
            report.append("")
            report.append("#### Additional PyCM Metrics (Test Set)")
            report.append(format_metrics_table(logreg_results["pycm_metrics"]))

        # Feature importance
        report.append("")
        report.append("### Top 15 Features (by coefficient magnitude)")
        feature_importance = logreg_results.get("feature_importance")
        if feature_importance is not None and not feature_importance.empty:
            report.append("")
            report.append(
                feature_importance.head(15)[
                    ["feature", "coefficient", "abs_coefficient"]
                ].to_markdown(index=False)
            )
    else:
        report.append(f"\nError: {logreg_results['error']}")

    report.append("")
    report.append("---")
    report.append("")

    # Random Forest Details
    report.append("## 2. Random Forest")
    if "error" not in rf_results:
        report.append("")
        report.append("### Performance Metrics")
        report.append("")
        report.append("#### Test Set (Primary Metrics)")
        test_metrics = {
            "Accuracy": rf_results["test_accuracy"],
            "Precision": rf_results["test_precision"],
            "Recall": rf_results["test_recall"],
            "F1 Score": rf_results["test_f1_score"],
            "ROC-AUC": rf_results["test_roc_auc"],
        }
        report.append(format_metrics_table(test_metrics))

        report.append("")
        report.append("#### Training Set")
        train_metrics = {
            "Accuracy": rf_results["train_accuracy"],
            "Precision": rf_results["train_precision"],
            "Recall": rf_results["train_recall"],
            "F1 Score": rf_results["train_f1_score"],
            "ROC-AUC": rf_results["train_roc_auc"],
        }
        report.append(format_metrics_table(train_metrics))

        report.append("")
        report.append("#### Cross-Validation")
        cv_metrics = {
            "CV ROC-AUC (mean)": rf_results["cv_mean"],
            "CV ROC-AUC (std)": rf_results["cv_std"],
        }
        report.append(format_metrics_table(cv_metrics))

        # PyCM metrics
        if rf_results.get("pycm_metrics"):
            report.append("")
            report.append("#### Additional PyCM Metrics (Test Set)")
            report.append(format_metrics_table(rf_results["pycm_metrics"]))

        # Feature importance
        report.append("")
        report.append("### Top 15 Features (by importance)")
        feature_importance = rf_results.get("feature_importance")
        if feature_importance is not None and not feature_importance.empty:
            report.append("")
            report.append(
                feature_importance.head(15)[["feature", "importance"]].to_markdown(
                    index=False
                )
            )
    else:
        report.append(f"\nError: {rf_results['error']}")

    report.append("")
    report.append("---")
    report.append("")

    # XGBoost Details
    report.append("## 3. XGBoost")
    if "error" not in xgb_results:
        report.append("")
        report.append("### Performance Metrics")
        report.append("")
        report.append("#### Test Set (Primary Metrics)")
        test_metrics = {
            "Accuracy": xgb_results["test_accuracy"],
            "Precision": xgb_results["test_precision"],
            "Recall": xgb_results["test_recall"],
            "F1 Score": xgb_results["test_f1_score"],
            "ROC-AUC": xgb_results["test_roc_auc"],
        }
        report.append(format_metrics_table(test_metrics))

        report.append("")
        report.append("#### Training Set")
        train_metrics = {
            "Accuracy": xgb_results["train_accuracy"],
            "Precision": xgb_results["train_precision"],
            "Recall": xgb_results["train_recall"],
            "F1 Score": xgb_results["train_f1_score"],
            "ROC-AUC": xgb_results["train_roc_auc"],
        }
        report.append(format_metrics_table(train_metrics))

        report.append("")
        report.append("#### Cross-Validation")
        cv_metrics = {
            "CV ROC-AUC (mean)": xgb_results["cv_mean"],
            "CV ROC-AUC (std)": xgb_results["cv_std"],
        }
        report.append(format_metrics_table(cv_metrics))

        # PyCM metrics
        if xgb_results.get("pycm_metrics"):
            report.append("")
            report.append("#### Additional PyCM Metrics (Test Set)")
            report.append(format_metrics_table(xgb_results["pycm_metrics"]))

        # Feature importance
        report.append("")
        report.append("### Top 15 Features (by importance)")
        feature_importance = xgb_results.get("feature_importance")
        if feature_importance is not None and not feature_importance.empty:
            report.append("")
            report.append(
                feature_importance.head(15)[["feature", "importance"]].to_markdown(
                    index=False
                )
            )
    else:
        report.append(f"\nError: {xgb_results['error']}")

    report.append("")
    report.append("=" * 80)
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("- **Logistic Regression**: Linear model, interpretable coefficients")
    report.append(
        "- **Random Forest**: Ensemble method, handles non-linear relationships"
    )
    report.append("- **XGBoost**: Gradient boosting, often highest performance")
    report.append("")
    report.append("**Note**: Test set metrics are the primary performance indicators.")
    report.append("Training metrics help identify overfitting (large train-test gap).")

    return "\n".join(report)


def run_ml_analysis_for_environment(features_df: pd.DataFrame, env_name: str):
    """
    Run all ML models for a given environment and generate reports.
    """
    if features_df.empty or len(features_df["score"].unique()) < 2:
        logger.info(
            f"\nSkipping environment '{env_name}' due to insufficient or uniform data."
        )
        return None

    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING ML MODEL ANALYSIS FOR: {env_name.upper()}")
    logger.info("=" * 80 + "\n")

    # Setup directories
    output_dir = Path(f"output_{env_name}")
    ml_dir = output_dir / "ml_models"
    data_dir = output_dir / "data"
    ml_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Outputs will be saved to: {ml_dir}/")
    logger.info(f"Dataset size: {len(features_df)} samples")
    logger.info(f"Class distribution: {features_df['score'].value_counts().to_dict()}")

    # Initialize analyzer
    analyzer = BinaryOutcomeAnalyzer(features_df, target_col="score")

    # Run Logistic Regression
    logger.info("--- Running Logistic Regression ---")
    logreg_results = analyzer.logistic_regression_analysis(
        use_all_features=True,
        test_size=0.2,
        cv_folds=5,
        save_html=True,
        output_dir=ml_dir,
    )

    if "error" not in logreg_results:
        # Save feature importance
        logreg_results["feature_importance"].to_csv(
            ml_dir / "logreg_feature_importance.csv", index=False
        )
        logger.info("  ✓ Saved logreg_feature_importance.csv")

        # Save SHAP importance
        if (
            "shap_importance" in logreg_results
            and not logreg_results["shap_importance"].empty
        ):
            logreg_results["shap_importance"].to_csv(
                ml_dir / "logreg_shap_importance.csv", index=False
            )
            logger.info("  ✓ Saved logreg_shap_importance.csv")

        # Save metrics summary
        metrics = {
            k: v
            for k, v in logreg_results.items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        with (ml_dir / "logreg_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("  ✓ Saved logreg_metrics.json")

    # Run Random Forest
    logger.info("--- Running Random Forest ---")
    rf_results = analyzer.random_forest_analysis(
        use_all_features=True,
        test_size=0.2,
        cv_folds=5,
        n_estimators=100,
        save_html=True,
        output_dir=ml_dir,
    )

    if "error" not in rf_results:
        rf_results["feature_importance"].to_csv(
            ml_dir / "rf_feature_importance.csv", index=False
        )
        logger.info("  ✓ Saved rf_feature_importance.csv")

        # Save SHAP importance
        if "shap_importance" in rf_results and not rf_results["shap_importance"].empty:
            rf_results["shap_importance"].to_csv(
                ml_dir / "rf_shap_importance.csv", index=False
            )
            logger.info("  ✓ Saved rf_shap_importance.csv")

        metrics = {
            k: v
            for k, v in rf_results.items()
            if isinstance(v, int | float | str) and not k.startswith("_")
        }
        with (ml_dir / "rf_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("  ✓ Saved rf_metrics.json")

    # Run XGBoost
    logger.info("--- Running XGBoost ---")
    xgb_results = analyzer.xgboost_analysis(
        use_all_features=True,
        test_size=0.2,
        cv_folds=5,
        save_html=True,
        output_dir=ml_dir,
    )

    if "error" not in xgb_results:
        xgb_results["feature_importance"].to_csv(
            ml_dir / "xgb_feature_importance.csv", index=False
        )
        logger.info("  ✓ Saved xgb_feature_importance.csv")

        # Save SHAP importance
        if (
            "shap_importance" in xgb_results
            and not xgb_results["shap_importance"].empty
        ):
            xgb_results["shap_importance"].to_csv(
                ml_dir / "xgb_shap_importance.csv", index=False
            )
            logger.info("  ✓ Saved xgb_shap_importance.csv")

        metrics = {
            k: v
            for k, v in xgb_results.items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }
        with (ml_dir / "xgb_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("  ✓ Saved xgb_metrics.json")

    # Generate comparison report
    logger.info("--- Generating Comparison Report ---")
    report = generate_model_comparison_report(
        env_name, logreg_results, rf_results, xgb_results
    )

    report_path = ml_dir / "ml_models_report.md"
    with report_path.open("w") as f:
        f.write(report)
    logger.info(f"  ✓ Saved ML models comparison report to: {report_path}")

    logger.info(f"\nML MODEL ANALYSIS FOR '{env_name.upper()}' COMPLETE")

    return {
        "logistic_regression": logreg_results,
        "random_forest": rf_results,
        "xgboost": xgb_results,
    }


def main():
    """Main ML analysis pipeline"""
    logger.info("=" * 80)
    logger.info("ML MODEL ANALYSIS PIPELINE")
    logger.info("=" * 80)

    # Load data
    logger.info("STEP 1: LOADING DATA & EXTRACTING FEATURES")
    data_path = "data/cleaned_corral_annotations.parquet"

    if not Path(data_path).exists():
        logger.info(f"Error: Data file not found at {data_path}")
        logger.info("Please ensure the data file exists or update the path.")
        return

    loader = TraceDataLoader(data_path)
    traces_df, steps_df, tools_df = loader.load_all()

    feature_extractor = FeatureExtractor()
    all_features_df = feature_extractor.extract_all(traces_df, steps_df, tools_df)
    logger.info(f"\nExtracted features for {len(all_features_df)} traces.")
    logger.info(f"Total features: {len(all_features_df.columns)}")

    # Analyze each environment
    environments_to_analyze = [
        "md",
        "retrosynthesis",
        "afm",
        "catalyst",
        "ml",
        "resistor",
        "all",
    ]

    all_results = {}

    for env in environments_to_analyze:
        if env == "all":
            env_features_df = all_features_df
        else:
            env_features_df = all_features_df[
                all_features_df["environment"] == env
            ].copy()

        results = run_ml_analysis_for_environment(env_features_df, env)

        if results:
            all_results[env] = results

    all_features_df.to_parquet(
        "output_all/data/features_all_environments.parquet", index=False
    )
    all_features_df.to_csv("output_all/data/features_all_environments.csv", index=False)
    logger.info(
        f"  ✓ Saved features_all_environments.parquet ({len(all_features_df)} traces)"
    )

    # Create master summary
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MASTER SUMMARY")
    logger.info("=" * 80)

    summary_lines = ["# ML Models Analysis - Master Summary\n"]
    summary_lines.append("## Environments Analyzed\n")

    for env, results in all_results.items():
        summary_lines.append(f"### {env.upper()}\n")
        summary_lines.append(f"- Report: `output_{env}/ml_models/ml_models_report.md`")
        summary_lines.append(f"- HTML Reports: `output_{env}/ml_models/*.html`")
        summary_lines.append(
            f"- Feature Importance: `output_{env}/ml_models/*_feature_importance.csv`"
        )
        summary_lines.append(f"- Metrics: `output_{env}/ml_models/*_metrics.json`")
        summary_lines.append("")

    with Path("ml_models_master_summary.md").open("w") as f:
        f.write("\n".join(summary_lines))

    logger.info("  ✓ Saved ml_models_master_summary.md")
    logger.info("=" * 80)
    logger.info("ALL ML ANALYSES COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
