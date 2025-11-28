"""
Analysis methods specifically for binary outcomes (success/failure)
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pycm import ConfusionMatrix
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ..config import ALPHA
from loguru import logger


class BinaryOutcomeAnalyzer:
    """Analyze features with binary outcomes (0/1, success/failure)"""

    def __init__(self, features_df: pd.DataFrame, target_col: str = "score"):
        """
        Initialize binary outcome analyzer

        Args:
            features_df: DataFrame with features
            target_col: Binary target column (should be 0/1 or True/False)
        """
        self.features_df = features_df
        self.target_col = target_col

        # Validate binary outcome
        unique_values = set(features_df[target_col].dropna().unique())
        if not unique_values.issubset({0, 1, True, False}):
            logger.info(f"Warning: {target_col} has non-binary values: {unique_values}")

    def _prepare_features_with_encoding(
        self,
        feature_cols: list[str],
        include_categorical: bool = True,
        categorical_features: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Helper method to prepare features with optional categorical encoding

        Args:
            feature_cols: list of numeric feature columns
            include_categorical: Whether to include categorical features
            categorical_features: list of categorical feature names

        Returns:
            tuple of (X_df, cols_to_use, categorical_features_present)
        """
        if categorical_features is None:
            categorical_features = ["model", "agent_type"]

        cols_to_use = feature_cols.copy()
        categorical_features_present = []

        if include_categorical:
            # Add categorical features if they exist
            for cat_feat in categorical_features:
                if cat_feat in self.features_df.columns and cat_feat not in cols_to_use:
                    cols_to_use.append(cat_feat)
                    categorical_features_present.append(cat_feat)

        # Get data with no NaNs
        df = self.features_df[cols_to_use + [self.target_col]].dropna()
        X_df = df[cols_to_use]

        return X_df, cols_to_use, categorical_features_present

    def _create_preprocessor(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        scale_numeric: bool = True,
    ):
        """
        Create preprocessing pipeline for numeric and categorical features

        Args:
            numeric_features: list of numeric feature names
            categorical_features: list of categorical feature names
            scale_numeric: Whether to scale numeric features (True for LogReg, False for tree-based)

        Returns:
            tuple of (preprocessor, encoded_feature_names)
        """
        if categorical_features:
            logger.info(
                f"  One-hot encoding categorical features: {categorical_features}"
            )

            transformers = []

            if scale_numeric:
                transformers.append(("num", StandardScaler(), numeric_features))
            else:
                transformers.append(("num", "passthrough", numeric_features))

            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    categorical_features,
                )
            )

            preprocessor = ColumnTransformer(
                transformers=transformers, remainder="drop"
            )

            # We'll compute feature names after fitting
            encoded_feature_names = None

        else:
            # No categorical features
            if scale_numeric:
                preprocessor = StandardScaler()
            else:
                preprocessor = None  # Will just use raw data

            encoded_feature_names = numeric_features

        return preprocessor, encoded_feature_names

    def _get_encoded_feature_names(
        self,
        preprocessor,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> list[str]:
        """
        Extract feature names after encoding

        Args:
            preprocessor: Fitted preprocessor
            numeric_features: Original numeric feature names
            categorical_features: Original categorical feature names

        Returns:
            list of encoded feature names
        """
        encoded_feature_names = []

        # Numeric feature names (unchanged)
        encoded_feature_names.extend(numeric_features)

        # Categorical feature names (one-hot encoded)
        if categorical_features and hasattr(preprocessor, "named_transformers_"):
            cat_encoder = preprocessor.named_transformers_["cat"]
            for i, cat_feat in enumerate(categorical_features):
                categories = cat_encoder.categories_[i]
                # Drop first, so we get n-1 categories
                for cat in categories[1:]:  # Skip first due to drop='first'
                    encoded_feature_names.append(f"{cat_feat}_{cat}")

        logger.info(f"  Total features after encoding: {len(encoded_feature_names)}")

        return encoded_feature_names

    def point_biserial_correlation(
        self, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Compute point-biserial correlations (Pearson for binary outcome)

        This is the same as Pearson correlation but specifically for binary outcomes.
        Interpretation is clearer for binary cases.

        Args:
            feature_cols: list of feature columns

        Returns:
            DataFrame with correlations and statistics
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if self.target_col in feature_cols:
                feature_cols.remove(self.target_col)

        results = []

        target_values = self.features_df[self.target_col].values

        for col in feature_cols:
            if col not in self.features_df.columns:
                continue

            feature_values = self.features_df[col].values

            # Remove NaN values
            mask = ~(np.isnan(target_values) | np.isnan(feature_values))
            if mask.sum() < 3:
                continue

            clean_target = target_values[mask]
            clean_feature = feature_values[mask]

            # Point-biserial correlation
            corr, pval = stats.pointbiserialr(clean_target, clean_feature)

            # Calculate effect size (Cohen's d)
            success_vals = clean_feature[clean_target == 1]
            failure_vals = clean_feature[clean_target == 0]

            if len(success_vals) > 0 and len(failure_vals) > 0:
                pooled_std = np.sqrt(
                    (
                        (len(success_vals) - 1) * success_vals.std() ** 2
                        + (len(failure_vals) - 1) * failure_vals.std() ** 2
                    )
                    / (len(success_vals) + len(failure_vals) - 2)
                )
                cohens_d = (
                    (success_vals.mean() - failure_vals.mean()) / pooled_std
                    if pooled_std > 0
                    else 0
                )
            else:
                cohens_d = 0

            results.append(
                {
                    "feature": col,
                    "correlation": corr,
                    "p_value": pval,
                    "significant": pval < ALPHA,
                    "abs_correlation": abs(corr),
                    "cohens_d": cohens_d,
                    "success_mean": success_vals.mean()
                    if len(success_vals) > 0
                    else np.nan,
                    "failure_mean": failure_vals.mean()
                    if len(failure_vals) > 0
                    else np.nan,
                    "n_success": len(success_vals),
                    "n_failure": len(failure_vals),
                }
            )

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values("abs_correlation", ascending=False)

        return results_df

    def mann_whitney_tests(self, feature_cols: list[str] | None = None) -> pd.DataFrame:
        """
        Perform Mann-Whitney U tests for each feature

        Non-parametric test comparing feature distributions between success/failure.
        Better for non-normal distributions.

        Args:
            feature_cols: list of feature columns

        Returns:
            DataFrame with test results
        """
        if feature_cols is None:
            feature_cols = self.features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if self.target_col in feature_cols:
                feature_cols.remove(self.target_col)

        results = []

        for col in feature_cols:
            if col not in self.features_df.columns:
                continue

            success_vals = self.features_df[self.features_df[self.target_col] == 1][
                col
            ].dropna()
            failure_vals = self.features_df[self.features_df[self.target_col] == 0][
                col
            ].dropna()

            if len(success_vals) < 2 or len(failure_vals) < 2:
                continue

            u_stat, pval = stats.mannwhitneyu(
                success_vals, failure_vals, alternative="two-sided"
            )

            # Effect size (rank-biserial correlation)
            r = 1 - (2 * u_stat) / (len(success_vals) * len(failure_vals))

            results.append(
                {
                    "feature": col,
                    "u_statistic": u_stat,
                    "p_value": pval,
                    "significant": pval < ALPHA,
                    "effect_size_r": r,
                    "success_median": success_vals.median(),
                    "failure_median": failure_vals.median(),
                    "median_diff": success_vals.median() - failure_vals.median(),
                }
            )

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values("p_value")

        return results_df

    def logistic_regression_analysis(
        self,
        feature_cols: list[str] | None = None,
        use_all_features: bool = True,
        include_categorical: bool = True,
        categorical_features: list[str] | None = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        save_html: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Perform logistic regression with train/test split and cross-validation

        Args:
            feature_cols: list of features to include (if None and use_all_features=True, uses all)
            use_all_features: If True, uses all numeric features (default: True)
            include_categorical: If True, one-hot encodes categorical features (default: True)
            categorical_features: list of categorical feature names (default: ["model", "agent_type"])
            test_size: Proportion of data for test set (default: 0.2)
            cv_folds: Number of cross-validation folds (default: 5)
            save_html: Whether to save PyCM HTML report (default: True)
            output_dir: Directory to save HTML reports (default: current directory)

        Returns:
            dictionary with logistic regression results including train/test metrics
        """
        # Default categorical features
        if categorical_features is None:
            categorical_features = ["model", "agent_type"]

        if feature_cols is None:
            if use_all_features:
                # Use ALL numeric features
                feature_cols = self.features_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if self.target_col in feature_cols:
                    feature_cols.remove(self.target_col)
                logger.info(
                    f"Using all {len(feature_cols)} numeric features for logistic regression"
                )
            else:
                # Use top correlated features (backward compatibility)
                point_biserial = self.point_biserial_correlation()
                feature_cols = point_biserial.head(10)["feature"].tolist()
                logger.info(f"Using top 10 correlated features")

        # Prepare features with categorical encoding
        X_df, cols_to_use, categorical_features_present = (
            self._prepare_features_with_encoding(
                feature_cols, include_categorical, categorical_features
            )
        )

        if len(X_df) < 10:
            return {"error": "Insufficient data for logistic regression"}

        y = self.features_df.loc[X_df.index, self.target_col].values

        # Create preprocessor
        preprocessor, _ = self._create_preprocessor(
            feature_cols, categorical_features_present, scale_numeric=True
        )

        # Fit and transform
        X_processed = preprocessor.fit_transform(X_df)

        # Get encoded feature names
        encoded_feature_names = self._get_encoded_feature_names(
            preprocessor, feature_cols, categorical_features_present
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )

        # Fit logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        # Predictions on train and test sets
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Feature importance (coefficients) - use encoded feature names
        feature_importance = pd.DataFrame(
            {
                "feature": encoded_feature_names,
                "coefficient": model.coef_[0],
                "abs_coefficient": np.abs(model.coef_[0]),
            }
        ).sort_values("abs_coefficient", ascending=False)

        # Train metrics
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_pred_proba)
            if len(np.unique(y_train)) > 1
            else np.nan,
        }

        # Test metrics (the important ones!)
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_pred_proba)
            if len(np.unique(y_test)) > 1
            else np.nan,
        }

        # PyCM metrics and HTML report
        pycm_metrics = {}
        if PYCM_AVAILABLE and save_html:
            try:
                cm = ConfusionMatrix(
                    actual_vector=y_test.tolist(), predict_vector=y_test_pred.tolist()
                )

                # Extract important metrics
                pycm_metrics = {
                    "TPR": cm.TPR[1] if 1 in cm.TPR else np.nan,  # Sensitivity/Recall
                    "TNR": cm.TNR[1] if 1 in cm.TNR else np.nan,  # Specificity
                    "PPV": cm.PPV[1] if 1 in cm.PPV else np.nan,  # Precision
                    "NPV": cm.NPV[1]
                    if 1 in cm.NPV
                    else np.nan,  # Negative Predictive Value
                    "FPR": cm.FPR[1] if 1 in cm.FPR else np.nan,  # False Positive Rate
                    "FNR": cm.FNR[1] if 1 in cm.FNR else np.nan,  # False Negative Rate
                    "F1_Score": cm.F1[1] if 1 in cm.F1 else np.nan,
                    "MCC": cm.MCC,  # Matthews Correlation Coefficient
                    "Kappa": cm.Kappa,  # Cohen's Kappa
                }

                # Save HTML report
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    html_path = output_dir / "logistic_regression_confusion_matrix.html"
                else:
                    html_path = Path("logistic_regression_confusion_matrix.html")

                cm.save_html(str(html_path))
                logger.info(f"  ✓ Saved confusion matrix HTML report to: {html_path}")

            except Exception as e:
                logger.info(f"  ✗ Failed to generate PyCM report: {e}")

        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "encoded_feature_names": encoded_feature_names,
            "categorical_features_used": categorical_features_present
            if include_categorical
            else [],
            "feature_importance": feature_importance,
            # Cross-validation
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            # Train metrics
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1_score": train_metrics["f1_score"],
            "train_roc_auc": train_metrics["roc_auc"],
            # Test metrics (PRIMARY METRICS)
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            # PyCM metrics
            "pycm_metrics": pycm_metrics,
            # Data info
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": len(feature_cols),
            "baseline_accuracy": max(y.mean(), 1 - y.mean()),
            # Backward compatibility (use test metrics)
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "n_samples": len(y),
        }

    def success_rate_by_quantile(
        self, feature_col: str, n_bins: int = 5
    ) -> pd.DataFrame:
        """
        Calculate success rate for different quantiles of a feature

        Very interpretable: shows how success rate changes with feature value

        Args:
            feature_col: Feature to analyze
            n_bins: Number of quantile bins

        Returns:
            DataFrame with success rates by quantile
        """
        df = self.features_df[[feature_col, self.target_col]].dropna()

        if len(df) < n_bins:
            return pd.DataFrame()

        # Create quantile bins
        df["quantile"] = pd.qcut(
            df[feature_col], q=n_bins, labels=False, duplicates="drop"
        )

        # Calculate success rate per quantile
        quantile_stats = (
            df.groupby("quantile")
            .agg(
                {
                    feature_col: ["min", "max", "mean"],
                    self.target_col: ["sum", "count", "mean"],
                }
            )
            .reset_index()
        )

        quantile_stats.columns = [
            "quantile",
            "feature_min",
            "feature_max",
            "feature_mean",
            "n_success",
            "n_total",
            "success_rate",
        ]

        return quantile_stats

    def feature_distribution_by_outcome(self, feature_col: str) -> dict[str, Any]:
        """
        Get feature distribution statistics split by outcome

        Args:
            feature_col: Feature to analyze

        Returns:
            dictionary with distribution statistics
        """
        success_vals = self.features_df[self.features_df[self.target_col] == 1][
            feature_col
        ].dropna()
        failure_vals = self.features_df[self.features_df[self.target_col] == 0][
            feature_col
        ].dropna()

        return {
            "success": {
                "mean": success_vals.mean(),
                "median": success_vals.median(),
                "std": success_vals.std(),
                "min": success_vals.min(),
                "max": success_vals.max(),
                "q25": success_vals.quantile(0.25),
                "q75": success_vals.quantile(0.75),
                "n": len(success_vals),
            },
            "failure": {
                "mean": failure_vals.mean(),
                "median": failure_vals.median(),
                "std": failure_vals.std(),
                "min": failure_vals.min(),
                "max": failure_vals.max(),
                "q25": failure_vals.quantile(0.25),
                "q75": failure_vals.quantile(0.75),
                "n": len(failure_vals),
            },
        }

    def random_forest_analysis(
        self,
        feature_cols: list[str] | None = None,
        use_all_features: bool = True,
        include_categorical: bool = True,
        categorical_features: list[str] | None = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        n_estimators: int = 100,
        save_html: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Perform Random Forest classification with train/test split and cross-validation

        Args:
            feature_cols: list of features to include
            use_all_features: If True, uses all numeric features
            include_categorical: If True, one-hot encodes categorical features
            categorical_features: list of categorical feature names (default: ["model", "agent_type"])
            test_size: Proportion of data for test set
            cv_folds: Number of cross-validation folds
            n_estimators: Number of trees in the forest
            save_html: Whether to save PyCM HTML report
            output_dir: Directory to save HTML reports

        Returns:
            dictionary with Random Forest results
        """
        if categorical_features is None:
            categorical_features = ["model", "agent_type"]

        if feature_cols is None:
            if use_all_features:
                feature_cols = self.features_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if self.target_col in feature_cols:
                    feature_cols.remove(self.target_col)
                logger.info(
                    f"Using all {len(feature_cols)} numeric features for Random Forest"
                )
            else:
                point_biserial = self.point_biserial_correlation()
                feature_cols = point_biserial.head(10)["feature"].tolist()

        # Prepare features with categorical encoding
        X_df, cols_to_use, categorical_features_present = (
            self._prepare_features_with_encoding(
                feature_cols, include_categorical, categorical_features
            )
        )

        if len(X_df) < 10:
            return {"error": "Insufficient data for Random Forest"}

        y = self.features_df.loc[X_df.index, self.target_col].values

        # Create preprocessor (no scaling for Random Forest)
        preprocessor, _ = self._create_preprocessor(
            feature_cols, categorical_features_present, scale_numeric=False
        )

        # Fit and transform
        if preprocessor:
            X_processed = preprocessor.fit_transform(X_df)
            # Get encoded feature names
            encoded_feature_names = self._get_encoded_feature_names(
                preprocessor, feature_cols, categorical_features_present
            )
        else:
            X_processed = X_df.values
            encoded_feature_names = feature_cols

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )

        # Fit Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
        )
        model.fit(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Feature importance (based on impurity decrease) - use encoded feature names
        feature_importance = pd.DataFrame(
            {
                "feature": encoded_feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Train metrics
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_pred_proba)
            if len(np.unique(y_train)) > 1
            else np.nan,
        }

        # Test metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_pred_proba)
            if len(np.unique(y_test)) > 1
            else np.nan,
        }

        # PyCM metrics
        pycm_metrics = {}
        if PYCM_AVAILABLE and save_html:
            try:
                cm = ConfusionMatrix(
                    actual_vector=y_test.tolist(), predict_vector=y_test_pred.tolist()
                )

                pycm_metrics = {
                    "TPR": cm.TPR[1] if 1 in cm.TPR else np.nan,
                    "TNR": cm.TNR[1] if 1 in cm.TNR else np.nan,
                    "PPV": cm.PPV[1] if 1 in cm.PPV else np.nan,
                    "NPV": cm.NPV[1] if 1 in cm.NPV else np.nan,
                    "FPR": cm.FPR[1] if 1 in cm.FPR else np.nan,
                    "FNR": cm.FNR[1] if 1 in cm.FNR else np.nan,
                    "F1_Score": cm.F1[1] if 1 in cm.F1 else np.nan,
                    "MCC": cm.MCC,
                    "Kappa": cm.Kappa,
                }

                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    html_path = output_dir / "random_forest_confusion_matrix.html"
                else:
                    html_path = Path("random_forest_confusion_matrix.html")

                cm.save_html(str(html_path))
                logger.info(
                    f"  ✓ Saved Random Forest confusion matrix HTML to: {html_path}"
                )

            except Exception as e:
                logger.info(f"  ✗ Failed to generate PyCM report: {e}")

        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "encoded_feature_names": encoded_feature_names,
            "categorical_features_used": categorical_features_present
            if include_categorical
            else [],
            "feature_importance": feature_importance,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1_score": train_metrics["f1_score"],
            "train_roc_auc": train_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "pycm_metrics": pycm_metrics,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": len(encoded_feature_names),
            "baseline_accuracy": max(y.mean(), 1 - y.mean()),
            # Backward compatibility
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "n_samples": len(y),
        }

    def xgboost_analysis(
        self,
        feature_cols: list[str] | None = None,
        use_all_features: bool = True,
        include_categorical: bool = True,
        categorical_features: list[str] | None = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        save_html: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Perform XGBoost classification with train/test split and cross-validation

        Args:
            feature_cols: list of features to include
            use_all_features: If True, uses all numeric features
            include_categorical: If True, one-hot encodes categorical features
            categorical_features: list of categorical feature names (default: ["model", "agent_type"])
            test_size: Proportion of data for test set
            cv_folds: Number of cross-validation folds
            save_html: Whether to save PyCM HTML report
            output_dir: Directory to save HTML reports

        Returns:
            dictionary with XGBoost results
        """
        if not XGBOOST_AVAILABLE:
            return {"error": "XGBoost not available. Install with: pip install xgboost"}

        if categorical_features is None:
            categorical_features = ["model", "agent_type"]

        if feature_cols is None:
            if use_all_features:
                feature_cols = self.features_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                if self.target_col in feature_cols:
                    feature_cols.remove(self.target_col)
                logger.info(
                    f"Using all {len(feature_cols)} numeric features for XGBoost"
                )
            else:
                point_biserial = self.point_biserial_correlation()
                feature_cols = point_biserial.head(10)["feature"].tolist()

        # Prepare features with categorical encoding
        X_df, cols_to_use, categorical_features_present = (
            self._prepare_features_with_encoding(
                feature_cols, include_categorical, categorical_features
            )
        )

        if len(X_df) < 10:
            return {"error": "Insufficient data for XGBoost"}

        y = self.features_df.loc[X_df.index, self.target_col].values

        # Create preprocessor (no scaling for XGBoost)
        preprocessor, _ = self._create_preprocessor(
            feature_cols, categorical_features_present, scale_numeric=False
        )

        # Fit and transform
        if preprocessor:
            X_processed = preprocessor.fit_transform(X_df)
            # Get encoded feature names
            encoded_feature_names = self._get_encoded_feature_names(
                preprocessor, feature_cols, categorical_features_present
            )
        else:
            X_processed = X_df.values
            encoded_feature_names = feature_cols

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )

        # Fit XGBoost
        model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(X_train, y_train, verbose=False)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Feature importance - use encoded feature names
        feature_importance = pd.DataFrame(
            {
                "feature": encoded_feature_names,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Train metrics
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_pred_proba)
            if len(np.unique(y_train)) > 1
            else np.nan,
        }

        # Test metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_pred_proba)
            if len(np.unique(y_test)) > 1
            else np.nan,
        }

        # PyCM metrics
        pycm_metrics = {}
        if PYCM_AVAILABLE and save_html:
            try:
                cm = ConfusionMatrix(
                    actual_vector=y_test.tolist(), predict_vector=y_test_pred.tolist()
                )

                pycm_metrics = {
                    "TPR": cm.TPR[1] if 1 in cm.TPR else np.nan,
                    "TNR": cm.TNR[1] if 1 in cm.TNR else np.nan,
                    "PPV": cm.PPV[1] if 1 in cm.PPV else np.nan,
                    "NPV": cm.NPV[1] if 1 in cm.NPV else np.nan,
                    "FPR": cm.FPR[1] if 1 in cm.FPR else np.nan,
                    "FNR": cm.FNR[1] if 1 in cm.FNR else np.nan,
                    "F1_Score": cm.F1[1] if 1 in cm.F1 else np.nan,
                    "MCC": cm.MCC,
                    "Kappa": cm.Kappa,
                }

                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    html_path = output_dir / "xgboost_confusion_matrix.html"
                else:
                    html_path = Path("xgboost_confusion_matrix.html")

                cm.save_html(str(html_path))
                logger.info(f"  ✓ Saved XGBoost confusion matrix HTML to: {html_path}")

            except Exception as e:
                logger.info(f"  ✗ Failed to generate PyCM report: {e}")

        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_cols": feature_cols,
            "encoded_feature_names": encoded_feature_names,
            "categorical_features_used": categorical_features_present
            if include_categorical
            else [],
            "feature_importance": feature_importance,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1_score": train_metrics["f1_score"],
            "train_roc_auc": train_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "pycm_metrics": pycm_metrics,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": len(encoded_feature_names),
            "baseline_accuracy": max(y.mean(), 1 - y.mean()),
            # Backward compatibility
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "n_samples": len(y),
        }

    def run_all_binary_analyses(self) -> dict[str, Any]:
        """
        Run all binary outcome analyses

        Returns:
            dictionary with all analysis results
        """
        logger.info("Running binary outcome analyses...")

        results = {}

        # Point-biserial correlations
        logger.info("  Computing point-biserial correlations...")
        results["point_biserial"] = self.point_biserial_correlation()

        # Mann-Whitney tests
        logger.info("  Running Mann-Whitney U tests...")
        results["mann_whitney"] = self.mann_whitney_tests()

        # Logistic regression
        logger.info("  Fitting logistic regression...")
        results["logistic_regression"] = self.logistic_regression_analysis()

        return results
