"""
Example Configuration Template
===============================
Copy these settings to trace_analysis.py CONFIG section
Modify as needed for your specific analysis requirements
"""

EXAMPLE_CONFIG = {
    # =================================================================
    # FILE PATHS
    # =================================================================
    "input_file": "/Users/n0w0f/git/n0w0f_2026/corral_modeling/corral_modeling/modelling/data/absolute_agreement_annotations_features.csv",
    "output_dir": "/Users/n0w0f/git/n0w0f_2026/corral_modeling/corral_modeling/modelling/analysis/results",
    # =================================================================
    # TARGET VARIABLE
    # =================================================================
    "target_column": "success",  # Column to predict
    # =================================================================
    # COLUMNS TO EXCLUDE FROM FEATURES
    # =================================================================
    # These won't be used as predictors
    "exclude_columns": [
        # Identifiers (not predictive)
        "trace_id",
        "file_id",
        "task_id",
        "timestamp",
        "trial_id",
        # Target/outcome variables (would be cheating!)
        "score",
        "success",
        # Add any other columns you want to exclude
        # 'some_other_id_column',
    ],
    # =================================================================
    # CATEGORICAL FEATURES TO ONE-HOT ENCODE
    # =================================================================
    "categorical_columns": [
        "annotator",  # Person who annotated
        "model",  # Model used (e.g., gpt-4o, claude-3)
        "environment",  # Environment/setting
        "agent_type",  # Type of agent
        "most_used_tool",  # Most frequently used tool
        # Add more categorical columns as needed:
        # 'custom_category_column',
    ],
    # =================================================================
    # MODEL TRAINING PARAMETERS
    # =================================================================
    "test_size": 0.2,  # 20% of data for testing
    "random_state": 42,  # Random seed for reproducibility
    "cv_folds": 5,  # Number of cross-validation folds
    # =================================================================
    # ADVANCED MODEL OPTIONS
    # =================================================================
    "use_xgboost": True,  # Set False if XGBoost not installed
    # =================================================================
    # CORRELATION & MULTICOLLINEARITY THRESHOLDS
    # =================================================================
    "high_correlation_threshold": 0.8,  # Flag feature pairs with |r| > 0.9
    "vif_threshold": 10,  # Flag features with VIF > 10
    # =================================================================
    # VISUALIZATION SETTINGS
    # =================================================================
    "figure_dpi": 300,  # Resolution for saved plots
    "figure_size": (12, 8),  # Default figure size
    "max_features_to_plot": 30,  # Max features in importance plots
}

# =================================================================
# ADDITIONAL CONFIGURATION EXAMPLES
# =================================================================

# Example 1: Quick analysis (fewer folds, lower resolution)
QUICK_CONFIG = {
    "test_size": 0.3,
    "cv_folds": 3,
    "figure_dpi": 150,
    "max_features_to_plot": 20,
}

# Example 2: Detailed analysis (more folds, higher resolution)
DETAILED_CONFIG = {
    "test_size": 0.15,
    "cv_folds": 10,
    "figure_dpi": 600,
    "max_features_to_plot": 50,
}

# Example 3: Large dataset (more conservative)
LARGE_DATASET_CONFIG = {
    "high_correlation_threshold": 0.95,  # More aggressive feature removal
    "vif_threshold": 5,  # Stricter multicollinearity
    "max_features_to_plot": 20,  # Fewer features in plots
}

# Example 4: Imbalanced classes
IMBALANCED_CONFIG = {
    # Note: The script already uses class_weight='balanced'
    # But you can also consider SMOTE or adjusting thresholds
    "test_size": 0.2,
    "cv_folds": 5,  # Stratified CV handles imbalance
}

# =================================================================
# FEATURE ENGINEERING TIPS
# =================================================================

"""
For custom feature engineering, edit the feature_engineering() method:

1. CREATE INTERACTION FEATURES:
   self.X['feature1_x_feature2'] = self.X['feature1'] * self.X['feature2']

2. CREATE RATIO FEATURES:
   self.X['ratio'] = self.X['numerator'] / (self.X['denominator'] + 1e-10)

3. BINNING CONTINUOUS FEATURES:
   self.X['feature_binned'] = pd.cut(self.X['feature'], bins=5, labels=False)

4. POLYNOMIAL FEATURES:
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2, include_bias=False)
   X_poly = poly.fit_transform(self.X[['feat1', 'feat2']])

5. LOG TRANSFORM SKEWED FEATURES:
   self.X['feature_log'] = np.log1p(self.X['feature'])
"""

# =================================================================
# MODEL HYPERPARAMETER TUNING TIPS
# =================================================================

"""
To tune hyperparameters, modify the train_models() method:

RANDOM FOREST:
    RandomForestClassifier(
        n_estimators=200,      # More trees (slower but better)
        max_depth=15,          # Deeper trees (risk overfitting)
        min_samples_split=10,  # Min samples to split node
        min_samples_leaf=5,    # Min samples in leaf
        max_features='sqrt',   # Features per split
        class_weight='balanced'
    )

XGBOOST:
    xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,    # Slower learning (better generalization)
        subsample=0.8,         # Row sampling
        colsample_bytree=0.8,  # Column sampling
        gamma=1,               # Regularization
        scale_pos_weight=...,  # Auto-calculated for imbalance
    )

LOGISTIC REGRESSION:
    LogisticRegression(
        C=0.1,                 # Regularization strength (smaller = more)
        penalty='l1',          # L1 for feature selection, L2 for stability
        solver='saga',         # For L1
        max_iter=2000,
        class_weight='balanced'
    )
"""

# =================================================================
# INTERPRETING VIF VALUES
# =================================================================

"""
VIF (Variance Inflation Factor) Guide:

VIF = 1:     No correlation
VIF = 1-5:   Moderate correlation (acceptable)
VIF = 5-10:  High correlation (consider removing)
VIF > 10:    Very high correlation (definitely remove)

If two features have high VIF:
- Keep the one more correlated with target
- Or combine them into a single feature
"""

# =================================================================
# DEALING WITH CLASS IMBALANCE
# =================================================================

"""
If you have severe class imbalance (e.g., 95% fails, 5% success):

1. The script already uses class_weight='balanced' in models

2. For additional help, add SMOTE in feature_engineering():
   
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

3. Consider these metrics over accuracy:
   - F1 Score (harmonic mean of precision/recall)
   - ROC-AUC (discrimination ability)
   - Precision/Recall (depending on cost of errors)

4. Adjust classification threshold:
   Instead of 0.5, use threshold that optimizes F1
"""
