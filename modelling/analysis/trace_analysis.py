"""
Comprehensive Trace Success Prediction Analysis
================================================
This script performs end-to-end ML analysis on trace data to predict success.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

CONFIG = {
    # File paths
    "input_file": "/Users/n0w0f/git/n0w0f_2026/corral_modeling/corral_modeling/modelling/data/absolute_agreement_annotations_features.csv",
    "output_dir": "/Users/n0w0f/git/n0w0f_2026/corral_modeling/corral_modeling/modelling/analysis/results",
    # Target variable
    "target_column": "success",
    # Columns to exclude from features
    "exclude_columns": [
        "trace_id",
        "file_id",
        "task_id",
        "timestamp",
        "trial_id",
        "success_rate",
        "score",
        "success",  # score and success are targets/outcomes
        "step_count",
        "steps_to_first_tool_call",
        "steps_to_first_error",
        "steps_to_first_positive_marker",
        "steps_to_first_negative_marker",
        "steps_to_first_planning",
        "steps_to_first_reasoning",
        "most_used_tool",
        "annotator",
        "tools_used_list",
    ],
    # Categorical columns to one-hot encode
    "categorical_columns": [
        # "annotator",
        "model",
        "environment",
        "agent_type",
        #  "most_used_tool",
    ],
    # Model parameters
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    # XGBoost parameters (install with: pip install xgboost)
    "use_xgboost": True,  # Set to False if xgboost is not installed
    # Correlation thresholds
    "high_correlation_threshold": 0.9,  # For removing redundant features
    "vif_threshold": 10,  # For multicollinearity detection
    # Visualization settings
    "figure_dpi": 300,
    "figure_size": (7, 5),
    "max_features_to_plot": 30,  # Max features in importance plots
    "shap_features_to_display": 20,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_output_directory(base_path):
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(base_path) / f"analysis_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)

    return output_path


def save_plot(filename, output_dir, dpi=300):
    """Save current plot to file."""
    filepath = output_dir / "plots" / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {filename}")


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================


class TracePredictionAnalysis:
    def __init__(self, config):
        self.config = config
        self.output_dir = create_output_directory(config["output_dir"])
        self.report = []

        print("\n" + "=" * 80)
        print("TRACE SUCCESS PREDICTION ANALYSIS")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")

    def log(self, message, level="INFO"):
        """Log message to console and report."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {level}: {message}"
        print(log_msg)
        self.report.append(message)

    def load_data(self):
        """Load and perform initial data exploration."""
        self.log("\n1. LOADING DATA", "STEP")
        self.log("-" * 40)

        self.df = pd.read_csv(self.config["input_file"])
        self.log(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")

        # Check target variable
        if self.config["target_column"] not in self.df.columns:
            raise ValueError(
                f"Target column '{self.config['target_column']}' not found!"
            )

        # Class distribution
        class_dist = self.df[self.config["target_column"]].value_counts()
        self.log(f"\nClass Distribution:")
        for cls, count in class_dist.items():
            pct = count / len(self.df) * 100
            self.log(f"  {cls}: {count} ({pct:.1f}%)")

        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            self.log(f"\nMissing values found in {(missing > 0).sum()} columns")
            # Fill missing values for categorical columns
            for col in self.config["categorical_columns"]:
                if col in self.df.columns and self.df[col].isnull().any():
                    self.df[col].fillna("UNKNOWN", inplace=True)

        # Save data summary
        with open(self.output_dir / "reports" / "data_summary.txt", "w") as f:
            f.write("DATA SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Total features: {len(self.df.columns)}\n\n")
            f.write("Class distribution:\n")
            f.write(str(class_dist) + "\n\n")
            f.write("Column types:\n")
            f.write(str(self.df.dtypes) + "\n\n")
            f.write("First few rows:\n")
            f.write(str(self.df.head()) + "\n")

    def exploratory_analysis(self):
        """Perform exploratory data analysis and visualizations."""
        self.log("\n2. EXPLORATORY DATA ANALYSIS", "STEP")
        self.log("-" * 40)

        # Numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            col for col in numeric_cols if col not in self.config["exclude_columns"]
        ]

        # Distribution plots for key numeric features
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.ravel()

            plot_cols = numeric_cols[:9]  # Plot first 9 numeric features
            for idx, col in enumerate(plot_cols):
                if self.df[col].nunique() > 1:
                    axes[idx].hist(
                        self.df[col].dropna(), bins=30, edgecolor="black", alpha=0.7
                    )
                    axes[idx].set_title(f"{col}")
                    axes[idx].set_xlabel("Value")
                    axes[idx].set_ylabel("Frequency")

            for idx in range(len(plot_cols), 9):
                fig.delaxes(axes[idx])

            plt.suptitle("Distribution of Key Numeric Features", fontsize=16, y=1.00)
            save_plot(
                "01_feature_distributions.png",
                self.output_dir,
                self.config["figure_dpi"],
            )

        # Target distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self.df[self.config["target_column"]].value_counts().plot(
            kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"]
        )
        ax.set_title(
            "Target Variable Distribution (Success)", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Success")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        save_plot(
            "02_target_distribution.png", self.output_dir, self.config["figure_dpi"]
        )

        self.log("Exploratory analysis completed")

    def feature_engineering(self):
        """Prepare features for modeling."""
        self.log("\n3. FEATURE ENGINEERING", "STEP")
        self.log("-" * 40)

        # Separate features and target
        self.y = self.df[self.config["target_column"]].copy()

        # Drop excluded columns
        feature_cols = [
            col for col in self.df.columns if col not in self.config["exclude_columns"]
        ]
        self.X = self.df[feature_cols].copy()

        self.log(f"Starting with {len(self.X.columns)} features")

        # Identify categorical and numeric columns
        self.categorical_features = [
            col for col in self.config["categorical_columns"] if col in self.X.columns
        ]

        # One-hot encode categorical features
        if self.categorical_features:
            self.log(
                f"One-hot encoding {len(self.categorical_features)} categorical features..."
            )
            self.X = pd.get_dummies(
                self.X, columns=self.categorical_features, drop_first=True
            )
            self.log(f"After encoding: {len(self.X.columns)} features")

        # Clean feature names for XGBoost compatibility
        self.X.columns = self.X.columns.str.replace(r"[\[\]<>]", "_", regex=True)
        self.log("Cleaned feature names for XGBoost compatibility")

        # Handle any remaining non-numeric columns
        non_numeric = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            self.log(
                f"Label encoding {len(non_numeric)} remaining non-numeric columns..."
            )
            le = LabelEncoder()
            for col in non_numeric:
                self.X[col] = le.fit_transform(self.X[col].astype(str))

        # Remove constant features
        constant_cols = [col for col in self.X.columns if self.X[col].nunique() <= 1]
        if constant_cols:
            self.log(f"Removing {len(constant_cols)} constant features")
            self.X = self.X.drop(columns=constant_cols)

        # Handle inf and nan
        self.X.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.X.fillna(self.X.median(), inplace=True)

        self.log(f"Final feature count: {len(self.X.columns)}")

        # Save feature names
        self.feature_names = self.X.columns.tolist()
        with open(self.output_dir / "reports" / "feature_names.txt", "w") as f:
            f.write("FEATURE NAMES\n")
            f.write("=" * 80 + "\n\n")
            for i, name in enumerate(self.feature_names, 1):
                f.write(f"{i}. {name}\n")

    def correlation_analysis(self):
        """Analyze feature correlations and multicollinearity."""
        self.log("\n4. CORRELATION ANALYSIS", "STEP")
        self.log("-" * 40)

        # Correlation matrix
        corr_matrix = self.X.corr()

        # Plot full correlation heatmap (limit to reasonable size)
        n_features = min(50, len(self.X.columns))
        if len(self.X.columns) > n_features:
            # Select features with highest variance for visualization
            variances = self.X.var().sort_values(ascending=False)
            top_features = variances.head(n_features).index.tolist()
            plot_corr = corr_matrix.loc[top_features, top_features]
        else:
            plot_corr = corr_matrix

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            plot_corr,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(
            "Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20
        )
        save_plot(
            "03_correlation_heatmap.png", self.output_dir, self.config["figure_dpi"]
        )

        # Find highly correlated features
        high_corr_pairs = []
        threshold = self.config["high_correlation_threshold"]
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append(
                        {
                            "Feature1": corr_matrix.columns[i],
                            "Feature2": corr_matrix.columns[j],
                            "Correlation": corr_matrix.iloc[i, j],
                        }
                    )

        if high_corr_pairs:
            self.log(
                f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {threshold})"
            )
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df.to_csv(
                self.output_dir / "reports" / "high_correlations.csv", index=False
            )
        else:
            self.log(f"No highly correlated pairs found (threshold: {threshold})")

        # Feature-Target correlation
        target_corr = pd.DataFrame(
            {
                "Feature": self.X.columns,
                "Correlation": [
                    pearsonr(self.X[col], self.y)[0] for col in self.X.columns
                ],
            }
        ).sort_values("Correlation", key=abs, ascending=False)

        # Plot top correlations with target
        top_n = min(30, len(target_corr))
        fig, ax = plt.subplots(figsize=(10, 8))
        target_corr.head(top_n).plot(
            x="Feature",
            y="Correlation",
            kind="barh",
            ax=ax,
            color=[
                "#2ecc71" if x > 0 else "#e74c3c"
                for x in target_corr.head(top_n)["Correlation"]
            ],
        )
        ax.set_title(
            f"Top {top_n} Features Correlated with Success",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Correlation Coefficient")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        save_plot(
            "04_target_correlations.png", self.output_dir, self.config["figure_dpi"]
        )

        target_corr.to_csv(
            self.output_dir / "reports" / "target_correlations.csv", index=False
        )

        # VIF Analysis (on subset to avoid computation issues)
        self.log("Computing VIF for multicollinearity detection...")
        vif_features = min(20, len(self.X.columns))
        X_vif = self.X[target_corr.head(vif_features)["Feature"]].copy()

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
        ]
        vif_data = vif_data.sort_values("VIF", ascending=False)

        # Save VIF results
        vif_data.to_csv(self.output_dir / "reports" / "vif_analysis.csv", index=False)

        high_vif = vif_data[vif_data["VIF"] > self.config["vif_threshold"]]
        if len(high_vif) > 0:
            self.log(
                f"Found {len(high_vif)} features with VIF > {self.config['vif_threshold']}"
            )

        # Plot VIF
        fig, ax = plt.subplots(figsize=(10, 8))
        vif_data.plot(
            x="Feature", y="VIF", kind="barh", ax=ax, color="steelblue", legend=False
        )
        ax.axvline(
            x=self.config["vif_threshold"],
            color="red",
            linestyle="--",
            label=f'Threshold ({self.config["vif_threshold"]})',
        )
        ax.set_title(
            "Variance Inflation Factor (VIF) - Multicollinearity Check",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("VIF Score")
        ax.legend()
        save_plot("05_vif_analysis.png", self.output_dir, self.config["figure_dpi"])

        self.log("Correlation analysis completed")

    def remove_high_vif_features(self):
        """Remove features with high VIF, preferring features correlated with target."""
        self.log("\n4.5. SMART VIF-BASED FEATURE REMOVAL", "STEP")
        self.log("-" * 40)

        max_vif_threshold = 10

        # Calculate target correlations first
        target_corr = {}
        for col in self.X.columns:
            try:
                corr = abs(pearsonr(self.X[col], self.y)[0])
                target_corr[col] = corr
            except:
                target_corr[col] = 0

        self.log(
            f"Removing features with VIF > {max_vif_threshold}, keeping features with highest target correlation..."
        )

        features_to_keep = self.X.columns.tolist()
        removed_features = []

        while len(features_to_keep) > 1:
            X_subset = self.X[features_to_keep]

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = features_to_keep
            vif_data["VIF"] = [
                variance_inflation_factor(X_subset.values, i)
                for i in range(len(features_to_keep))
            ]
            vif_data["Target_Corr"] = [target_corr[f] for f in features_to_keep]

            # Get features above threshold
            high_vif = vif_data[vif_data["VIF"] > max_vif_threshold]

            if len(high_vif) == 0:
                break

            # Among high VIF features, remove the one with LOWEST target correlation
            feature_to_remove = high_vif.loc[
                high_vif["Target_Corr"].idxmin(), "Feature"
            ]
            vif_value = high_vif.loc[high_vif["Target_Corr"].idxmin(), "VIF"]

            features_to_keep.remove(feature_to_remove)
            removed_features.append(
                {
                    "Feature": feature_to_remove,
                    "VIF": vif_value,
                    "Target_Corr": target_corr[feature_to_remove],
                }
            )
            self.log(
                f"  Removed: {feature_to_remove} (VIF: {vif_value:.2f}, Target Corr: {target_corr[feature_to_remove]:.4f})"
            )

        self.X = self.X[features_to_keep]

        self.log(f"\nRemoved {len(removed_features)} features with high VIF")
        self.log(f"Remaining features: {len(self.X.columns)}")

        if removed_features:
            removed_df = pd.DataFrame(removed_features).sort_values(
                "VIF", ascending=False
            )
            removed_df.to_csv(
                self.output_dir / "reports" / "removed_high_vif_features.csv",
                index=False,
            )

    def train_models(self):
        """Train multiple models and compare performance."""
        self.log("\n5. MODEL TRAINING", "STEP")
        self.log("-" * 40)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=self.y,
        )

        self.log(f"Train set: {len(self.X_train)} samples")
        self.log(f"Test set: {len(self.X_test)} samples")

        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Initialize models
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.config["random_state"],
                max_iter=1000,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.config["random_state"],
                class_weight="balanced",
                max_depth=10,
                n_jobs=-1,
            ),
        }

        # Add XGBoost if available
        if self.config["use_xgboost"]:
            try:
                import xgboost as xgb

                scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
                self.models["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=self.config["random_state"],
                    scale_pos_weight=scale_pos_weight,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                )
                self.log("XGBoost model added")
            except ImportError:
                self.log("XGBoost not available, skipping", "WARN")

        # Train and evaluate models
        self.results = {}
        cv = StratifiedKFold(
            n_splits=self.config["cv_folds"],
            shuffle=True,
            random_state=self.config["random_state"],
        )

        for name, model in self.models.items():
            self.log(f"\nTraining {name}...")

            # Use scaled data for Logistic Regression, original for tree-based
            X_train_use = self.X_train_scaled if "Logistic" in name else self.X_train
            X_test_use = self.X_test_scaled if "Logistic" in name else self.X_test

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_use, self.y_train, cv=cv, scoring="f1", n_jobs=-1
            )

            # Train on full training set
            model.fit(X_train_use, self.y_train)

            # Predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            # Metrics
            self.results[name] = {
                "model": model,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "accuracy": accuracy_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "X_test": X_test_use,
            }

            self.log(
                f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
            )
            self.log(f"  Test Accuracy: {self.results[name]['accuracy']:.4f}")
            self.log(f"  Test F1 Score: {self.results[name]['f1']:.4f}")
            self.log(f"  Test ROC-AUC: {self.results[name]['roc_auc']:.4f}")

        # Model comparison plot
        metrics_df = pd.DataFrame(
            {
                "Model": list(self.results.keys()),
                "CV F1": [self.results[m]["cv_mean"] for m in self.results],
                "Test F1": [self.results[m]["f1"] for m in self.results],
                "Test Accuracy": [self.results[m]["accuracy"] for m in self.results],
                "ROC-AUC": [self.results[m]["roc_auc"] for m in self.results],
            }
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(["CV F1", "Test F1", "Test Accuracy", "ROC-AUC"]):
            axes[idx].bar(
                metrics_df["Model"], metrics_df[metric], color="steelblue", alpha=0.7
            )
            axes[idx].set_title(f"{metric} Score", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("Score")
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for i, v in enumerate(metrics_df[metric]):
                axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

        plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
        save_plot("06_model_comparison.png", self.output_dir, self.config["figure_dpi"])

        # Save metrics
        metrics_df.to_csv(
            self.output_dir / "reports" / "model_metrics.csv", index=False
        )

        self.log("\nModel training completed")

    def evaluate_models(self):
        """Generate detailed evaluation plots for each model."""
        self.log("\n6. MODEL EVALUATION", "STEP")
        self.log("-" * 40)

        for name, result in self.results.items():
            self.log(f"\nEvaluating {name}...")

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, result["y_pred"])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Confusion matrix heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[0],
                xticklabels=["Fail", "Success"],
                yticklabels=["Fail", "Success"],
            )
            axes[0].set_title(f"Confusion Matrix - {name}", fontweight="bold")
            axes[0].set_ylabel("Actual")
            axes[0].set_xlabel("Predicted")

            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, result["y_pred_proba"])
            axes[1].plot(
                fpr, tpr, linewidth=2, label=f'ROC (AUC = {result["roc_auc"]:.3f})'
            )
            axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title(f"ROC Curve - {name}", fontweight="bold")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            save_plot(
                f'07_evaluation_{name.replace(" ", "_").lower()}.png',
                self.output_dir,
                self.config["figure_dpi"],
            )

            # Classification report
            report = classification_report(
                self.y_test,
                result["y_pred"],
                target_names=["Fail", "Success"],
                output_dict=True,
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(
                self.output_dir
                / "reports"
                / f'classification_report_{name.replace(" ", "_").lower()}.csv'
            )

        self.log("Model evaluation completed")

    def feature_importance_analysis(self):
        """Analyze and visualize feature importance."""
        self.log("\n7. FEATURE IMPORTANCE ANALYSIS", "STEP")
        self.log("-" * 40)

        for name, result in self.results.items():
            model = result["model"]

            # Get feature importance based on model type
            if hasattr(model, "feature_importances_"):
                # Tree-based models
                importances = model.feature_importances_
                feature_imp = pd.DataFrame(
                    {"Feature": self.feature_names, "Importance": importances}
                ).sort_values("Importance", ascending=False)

            elif hasattr(model, "coef_"):
                # Linear models
                importances = np.abs(model.coef_[0])
                feature_imp = pd.DataFrame(
                    {"Feature": self.feature_names, "Importance": importances}
                ).sort_values("Importance", ascending=False)
            else:
                continue

            # Save full importance scores
            feature_imp.to_csv(
                self.output_dir
                / "reports"
                / f'feature_importance_{name.replace(" ", "_").lower()}.csv',
                index=False,
            )

            # Plot top features
            top_n = min(self.config["max_features_to_plot"], len(feature_imp))
            fig, ax = plt.subplots(figsize=(10, 8))
            feature_imp.head(top_n).plot(
                x="Feature",
                y="Importance",
                kind="barh",
                ax=ax,
                color="steelblue",
                legend=False,
            )
            ax.set_title(
                f"Top {top_n} Most Important Features - {name}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Importance Score")
            save_plot(
                f'08_feature_importance_{name.replace(" ", "_").lower()}.png',
                self.output_dir,
                self.config["figure_dpi"],
            )

            self.log(f"  Top 5 features for {name}:")
            for idx, row in feature_imp.head(5).iterrows():
                self.log(f"    {row['Feature']}: {row['Importance']:.4f}")

        self.log("\nFeature importance analysis completed")

    def shap_analysis(self):
        """Perform SHAP analysis for model interpretation."""
        self.log("\n8. SHAP ANALYSIS", "STEP")
        self.log("-" * 40)

        try:
            import shap
        except ImportError:
            self.log("SHAP not installed. Skipping SHAP analysis.", "WARN")
            self.log("Install with: pip install shap", "INFO")
            return

        # Perform SHAP analysis on best model (highest F1)
        best_model_name = max(self.results, key=lambda x: self.results[x]["f1"])
        self.log(f"Running SHAP analysis on best model: {best_model_name}")

        model = self.results[best_model_name]["model"]

        # Use UNSCALED data for SHAP (important for interpretability)
        # Convert to DataFrame to preserve feature names
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)

        # Sample data if too large
        if len(X_test_df) > 100:
            sample_idx = np.random.choice(len(X_test_df), 100, replace=False)
            X_shap = X_test_df.iloc[sample_idx].copy()
        else:
            X_shap = X_test_df.copy()

        self.log(
            f"Computing SHAP values for {len(X_shap)} samples with {len(X_shap.columns)} features..."
        )

        try:
            # Create explainer based on model type
            if "XGBoost" in best_model_name or "Random Forest" in best_model_name:
                # TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)

                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                # LinearExplainer for Logistic Regression
                self.log("Using LinearExplainer for Logistic Regression...")

                # Need to scale the data for Logistic Regression
                X_shap_scaled = self.scaler.transform(X_shap)

                # Use LinearExplainer which is faster and more appropriate
                explainer = shap.LinearExplainer(
                    model, X_shap_scaled, feature_names=self.feature_names
                )
                shap_values = explainer.shap_values(X_shap_scaled)

                # For display, we still want to show original (unscaled) feature values
                # But use the SHAP values from the scaled model

            self.log(f"SHAP values computed. Shape: {shap_values.shape}")

            # Summary plot (beeswarm) - shows top 20 features
            fig = plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values,
                X_shap,  # Use unscaled data for display
                feature_names=self.feature_names,
                show=False,
                max_display=20,
                plot_size=(12, 10),
            )
            plt.title(
                f"SHAP Feature Importance - {best_model_name}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            save_plot("09_shap_summary.png", self.output_dir, self.config["figure_dpi"])
            plt.close()

            # Bar plot - shows mean absolute SHAP values
            fig = plt.figure(figsize=(10, 10))
            shap.summary_plot(
                shap_values,
                X_shap,  # Use unscaled data for display
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
                max_display=20,
                plot_size=(10, 10),
            )
            plt.title(
                f"SHAP Mean Absolute Value - {best_model_name}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            save_plot("10_shap_bar.png", self.output_dir, self.config["figure_dpi"])
            plt.close()

            # Save SHAP values to CSV for top features
            # Handle different SHAP value formats
            if len(shap_values.shape) == 1:
                mean_abs_shap = np.abs(shap_values)
            elif shap_values.shape[0] == 1:
                mean_abs_shap = np.abs(shap_values[0])
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Ensure it's 1D and matches feature count
            mean_abs_shap = np.ravel(mean_abs_shap)

            if len(mean_abs_shap) != len(self.feature_names):
                self.log(
                    f"WARNING: SHAP values length ({len(mean_abs_shap)}) doesn't match features ({len(self.feature_names)})",
                    "WARN",
                )
                # Truncate or pad as needed
                min_len = min(len(mean_abs_shap), len(self.feature_names))
                mean_abs_shap = mean_abs_shap[:min_len]
                feature_names_subset = self.feature_names[:min_len]
            else:
                feature_names_subset = self.feature_names

            shap_importance = pd.DataFrame(
                {"Feature": feature_names_subset, "Mean_Abs_SHAP": mean_abs_shap}
            ).sort_values("Mean_Abs_SHAP", ascending=False)

            shap_importance.to_csv(
                self.output_dir
                / "reports"
                / f'shap_importance_{best_model_name.replace(" ", "_").lower()}.csv',
                index=False,
            )

            self.log(f"Top 10 features by SHAP importance:")
            for idx, row in shap_importance.head(10).iterrows():
                self.log(f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

            self.log("SHAP analysis completed")

        except Exception as e:
            self.log(f"SHAP analysis failed: {str(e)}", "ERROR")
            import traceback

            self.log(traceback.format_exc(), "ERROR")

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        self.log("\n9. GENERATING SUMMARY REPORT", "STEP")
        self.log("-" * 40)

        report_path = self.output_dir / "reports" / "SUMMARY_REPORT.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TRACE SUCCESS PREDICTION - ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            f.write("-" * 80 + "\n")
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(
                f"Number of Features (after engineering): {len(self.feature_names)}\n"
            )
            f.write(f"Target Variable: {self.config['target_column']}\n")
            class_dist = self.y.value_counts()
            f.write(f"Class Distribution:\n")
            for cls, count in class_dist.items():
                f.write(f"  {cls}: {count} ({count/len(self.y)*100:.1f}%)\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            for name, result in self.results.items():
                f.write(f"{name}:\n")
                f.write(
                    f"  Cross-Validation F1: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n"
                )
                f.write(f"  Test Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Test F1 Score: {result['f1']:.4f}\n")
                f.write(f"  Test ROC-AUC: {result['roc_auc']:.4f}\n\n")

            # Best model
            best_model = max(self.results, key=lambda x: self.results[x]["f1"])
            f.write(f"BEST MODEL (by F1 Score): {best_model}\n")
            f.write(f"  F1 Score: {self.results[best_model]['f1']:.4f}\n")
            f.write(f"  ROC-AUC: {self.results[best_model]['roc_auc']:.4f}\n\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n\n")

            # Top features
            if "Random Forest" in self.results:
                model = self.results["Random Forest"]["model"]
                importances = model.feature_importances_
                top_features = (
                    pd.DataFrame(
                        {"Feature": self.feature_names, "Importance": importances}
                    )
                    .sort_values("Importance", ascending=False)
                    .head(10)
                )

                f.write("Top 10 Most Important Features (Random Forest):\n")
                for idx, row in top_features.iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n\n")
            f.write("Plots:\n")
            plot_files = sorted((self.output_dir / "plots").glob("*.png"))
            for pf in plot_files:
                f.write(f"  - {pf.name}\n")

            f.write("\nReports:\n")
            report_files = sorted((self.output_dir / "reports").glob("*.csv"))
            for rf in report_files:
                f.write(f"  - {rf.name}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        self.log(f"Summary report saved: {report_path}")

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        try:
            self.load_data()
            self.exploratory_analysis()
            self.feature_engineering()
            self.correlation_analysis()
            self.remove_high_vif_features()
            self.train_models()
            self.evaluate_models()
            self.feature_importance_analysis()
            self.shap_analysis()
            self.generate_summary_report()

            self.log("\n" + "=" * 80)
            self.log("ANALYSIS COMPLETE!", "SUCCESS")
            self.log("=" * 80)
            self.log(f"\nAll results saved to: {self.output_dir}")
            self.log("\nKey outputs:")
            self.log(f"  - Plots: {self.output_dir / 'plots'}")
            self.log(f"  - Reports: {self.output_dir / 'reports'}")
            self.log(
                f"  - Summary: {self.output_dir / 'reports' / 'SUMMARY_REPORT.txt'}"
            )

        except Exception as e:
            self.log(f"\nERROR: {str(e)}", "ERROR")
            import traceback

            traceback.print_exc()
            raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Install required packages if needed
    # print("\nChecking dependencies...")
    # required_packages = [
    #     "numpy",
    #     "pandas",
    #     "matplotlib",
    #     "seaborn",
    #     "scikit-learn",
    #     "scipy",
    #     "statsmodels",
    # ]

    # missing_packages = []
    # for package in required_packages:
    #     try:
    #         __import__(package)
    #     except ImportError:
    #         missing_packages.append(package)

    # if missing_packages:
    #     print(f"Missing packages: {', '.join(missing_packages)}")
    #     print("Please install with: pip install " + " ".join(missing_packages))
    #     exit(1)

    # Run analysis
    analyzer = TracePredictionAnalysis(CONFIG)
    analyzer.run_full_analysis()
