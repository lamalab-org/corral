"""
QUICK REFERENCE GUIDE
=====================
Handy reference for common tasks and interpretations
"""

# =================================================================
# QUICK START (3 STEPS)
# =================================================================

"""
1. Install dependencies:
   pip install -r requirements.txt

2. Edit trace_analysis.py:
   CONFIG['input_file'] = 'your_data.csv'

3. Run:
   python trace_analysis.py
"""

# =================================================================
# READING THE OUTPUT
# =================================================================

OUTPUT_FILES = """
START HERE: reports/SUMMARY_REPORT.txt
├── Dataset overview
├── Model performance comparison
├── Best model identification
└── Top 10 most important features

KEY PLOTS:
├── 01_feature_distributions.png     → Data distributions
├── 02_target_distribution.png       → Class balance check
├── 03_correlation_heatmap.png       → Feature relationships
├── 04_target_correlations.png       → What predicts success?
├── 05_vif_analysis.png             → Multicollinearity check
├── 06_model_comparison.png         → Which model is best?
├── 07_evaluation_*.png             → Confusion matrices & ROC
├── 08_feature_importance_*.png     → What matters most?
├── 09_shap_summary.png             → Feature effects (SHAP)
└── 10_shap_bar.png                 → Feature importance (SHAP)

KEY REPORTS:
├── model_metrics.csv               → All model scores
├── feature_importance_*.csv        → Complete rankings
├── target_correlations.csv         → Feature-target correlations
├── high_correlations.csv           → Redundant feature pairs
└── vif_analysis.csv               → Multicollinearity scores
"""

# =================================================================
# INTERPRETING MODEL METRICS
# =================================================================

METRICS_GUIDE = """
ACCURACY:
- (TP + TN) / Total
- Good for balanced classes
- Misleading for imbalanced data
- Example: 95% accuracy when 95% are fails means nothing!

F1 SCORE: ⭐ BEST FOR IMBALANCED DATA
- Harmonic mean of Precision and Recall
- Range: 0 (worst) to 1 (perfect)
- Balances false positives and false negatives
- Use this for model selection!

ROC-AUC:
- Area Under ROC Curve
- Range: 0.5 (random) to 1 (perfect)
- Measures discrimination ability
- Less sensitive to class imbalance than accuracy

PRECISION:
- TP / (TP + FP)
- "When I predict success, how often am I right?"
- Important when false positives are costly

RECALL (Sensitivity):
- TP / (TP + FN)
- "Of all actual successes, how many did I catch?"
- Important when false negatives are costly
"""

# =================================================================
# FEATURE IMPORTANCE INTERPRETATION
# =================================================================

FEATURE_IMPORTANCE_GUIDE = """
WHAT IT MEANS:
- Higher = more important for prediction
- Doesn't mean causation!
- Shows association, not necessarily the "cause"

TREE-BASED IMPORTANCE (RF, XGBoost):
- Based on how much each feature improves splits
- Can be biased toward high-cardinality features
- Good for initial screening

LOGISTIC REGRESSION COEFFICIENTS:
- Absolute value = importance
- Sign = direction (positive/negative effect)
- Only interpretable if features are standardized

SHAP VALUES: ⭐ BEST FOR INTERPRETATION
- Shows how each feature contributes to predictions
- Handles feature interactions
- Consistent with human intuition
- Red = increases success probability
- Blue = decreases success probability
"""

# =================================================================
# CORRELATION INTERPRETATION
# =================================================================

CORRELATION_GUIDE = """
CORRELATION STRENGTH:
0.0 - 0.1:  Negligible
0.1 - 0.3:  Weak
0.3 - 0.5:  Moderate
0.5 - 0.7:  Strong
0.7 - 0.9:  Very strong
0.9 - 1.0:  Extremely strong (possibly redundant!)

POSITIVE CORRELATION:
- Both increase together
- Example: more_planning → higher_success

NEGATIVE CORRELATION:
- One increases, other decreases
- Example: more_errors → lower_success

WATCH OUT FOR:
- Correlation ≠ Causation!
- High correlation between features (multicollinearity)
- Spurious correlations (random chance)
"""

# =================================================================
# COMMON QUESTIONS & ANSWERS
# =================================================================

FAQ = """
Q: Which model should I trust?
A: Use the one with highest F1 score. If similar, choose simpler model.

Q: My accuracy is high but F1 is low, why?
A: Class imbalance! Model just predicts majority class.

Q: Should I remove highly correlated features?
A: Yes if VIF > 10. Keep the one more correlated with target.

Q: Feature X is important but makes no sense?
A: Could be: (1) data leakage, (2) confounding variable, (3) spurious

Q: How do I improve model performance?
A: 
   1. Feature engineering (interactions, ratios)
   2. Hyperparameter tuning
   3. More/better quality data
   4. Different model architectures
   5. Ensemble methods

Q: What if I have very few samples?
A: 
   - Use simpler models (Logistic Regression)
   - More cross-validation folds
   - Be skeptical of complex patterns
   - Consider collecting more data

Q: How do I use this for new predictions?
A: See DEPLOYMENT section below
"""

# =================================================================
# COMMON ISSUES & SOLUTIONS
# =================================================================

TROUBLESHOOTING = """
ISSUE: All models perform poorly (F1 < 0.5)
SOLUTION:
  - Check if features are actually predictive
  - Look at feature distributions
  - May need better features or more data
  - Try feature engineering

ISSUE: Perfect accuracy (1.0)
SOLUTION:
  - Data leakage! Check excluded_columns
  - Remove features that contain the answer
  - Check for exact duplicates of target

ISSUE: Large gap between CV and test performance
SOLUTION:
  - Overfitting! Model too complex
  - Reduce max_depth in trees
  - Add regularization (Logistic Regression)
  - Get more training data

ISSUE: Model does well but wrong features are important
SOLUTION:
  - Possible confounding variables
  - Check business logic/domain knowledge
  - Look for data collection artifacts

ISSUE: Script crashes with memory error
SOLUTION:
  - Reduce max_features_to_plot
  - Sample data for VIF/SHAP analysis
  - Process in batches

ISSUE: XGBoost/SHAP not installed
SOLUTION:
  - pip install xgboost shap
  - Or set use_xgboost=False in config
  - Script continues without them
"""

# =================================================================
# DEPLOYMENT / USING THE MODEL
# =================================================================

DEPLOYMENT_GUIDE = """
After analysis, to make predictions on new data:

1. SAVE THE BEST MODEL (add to script):
   
   import joblib
   best_model = self.results[best_model_name]['model']
   joblib.dump(best_model, 'best_model.pkl')
   joblib.dump(self.scaler, 'scaler.pkl')  # If using scaling
   joblib.dump(self.feature_names, 'feature_names.pkl')

2. LOAD AND PREDICT (new script):
   
   import joblib
   import pandas as pd
   
   # Load model and preprocessing
   model = joblib.load('best_model.pkl')
   scaler = joblib.load('scaler.pkl')  # If needed
   feature_names = joblib.load('feature_names.pkl')
   
   # Load new data
   new_data = pd.read_csv('new_traces.csv')
   
   # Apply SAME preprocessing as training
   # (one-hot encoding, feature engineering, etc.)
   X_new = preprocess(new_data)  # You need to write this
   
   # Ensure same features
   X_new = X_new[feature_names]
   
   # Scale if needed
   X_new_scaled = scaler.transform(X_new)  # If model used scaling
   
   # Predict
   predictions = model.predict(X_new_scaled)
   probabilities = model.predict_proba(X_new_scaled)[:, 1]
   
   # Add to dataframe
   new_data['predicted_success'] = predictions
   new_data['success_probability'] = probabilities

3. MONITOR PERFORMANCE:
   - Track actual vs predicted
   - Retrain periodically with new data
   - Watch for distribution shift
"""

# =================================================================
# NEXT STEPS FOR ADVANCED USERS
# =================================================================

ADVANCED_TOPICS = """
1. HYPERPARAMETER TUNING:
   - Use GridSearchCV or RandomizedSearchCV
   - Optimize on F1 score
   - Use nested cross-validation

2. ENSEMBLE METHODS:
   - Combine multiple models (voting/stacking)
   - Often better than single model
   - More complex but more robust

3. FEATURE SELECTION:
   - Recursive Feature Elimination (RFE)
   - SelectKBest with chi2/f_classif
   - L1-regularized Logistic Regression

4. HANDLING IMBALANCE:
   - SMOTE oversampling
   - ADASYN adaptive sampling
   - Class weight adjustment
   - Threshold tuning

5. INTERPRETABILITY:
   - LIME for local explanations
   - Partial Dependence Plots
   - Individual Conditional Expectation (ICE)
   - Accumulated Local Effects (ALE)

6. MODEL VALIDATION:
   - Time-based split (if temporal data)
   - Stratified group K-fold
   - Leave-one-group-out CV
   - Calibration curves

7. PRODUCTION:
   - Model versioning (MLflow)
   - A/B testing
   - Monitoring drift
   - Retraining pipelines
"""

# =================================================================
# RECOMMENDED READING
# =================================================================

RESOURCES = """
BOOKS:
- "Hands-On Machine Learning" by Aurélien Géron
- "The Elements of Statistical Learning" (free PDF)
- "Interpretable Machine Learning" by Christoph Molnar (free)

ONLINE:
- scikit-learn documentation: https://scikit-learn.org
- SHAP documentation: https://shap.readthedocs.io
- Kaggle tutorials: https://www.kaggle.com/learn

PAPERS:
- "A Unified Approach to Interpreting Model Predictions" (SHAP)
- "XGBoost: A Scalable Tree Boosting System"
- "Random Forests" by Leo Breiman
"""

print(OUTPUT_FILES)
print(METRICS_GUIDE)
print(FEATURE_IMPORTANCE_GUIDE)
