# Trace Success Prediction Analysis

This script performs comprehensive machine learning analysis on trace data to predict success outcomes.

## Features

✅ **Exploratory Data Analysis**
- Data distribution visualizations
- Class balance analysis
- Missing value handling

✅ **Feature Engineering**
- One-hot encoding for categorical variables
- Automated feature selection
- Multicollinearity detection (VIF)

✅ **Correlation Analysis**
- Feature correlation heatmaps
- Feature-target correlation analysis
- High correlation pair detection

✅ **Multiple ML Models**
- Logistic Regression (baseline)
- Random Forest
- XGBoost (optional)

✅ **Model Evaluation**
- Cross-validation
- Confusion matrices
- ROC curves
- Classification reports

✅ **Feature Importance**
- Model-specific feature rankings
- SHAP analysis for interpretability
- Top feature visualizations

✅ **Comprehensive Reporting**
- Automated summary report
- All plots saved as high-resolution PNGs
- CSV exports of all analysis results

## Installation

### Step 1: Install Python packages

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost shap
```

### Step 2: Configure the script

Edit the `CONFIG` section at the top of `trace_analysis.py`:

```python
CONFIG = {
    'input_file': 'path/to/your/data.csv',  # CHANGE THIS
    'output_dir': './trace_analysis_results',
    'target_column': 'success',
    # ... other parameters
}
```

## Usage

### Basic usage:

```bash
python trace_analysis.py
```

### What happens:

1. Loads your CSV data
2. Performs exploratory analysis
3. Engineers features (one-hot encoding, etc.)
4. Analyzes correlations and multicollinearity
5. Trains multiple ML models
6. Evaluates model performance
7. Generates feature importance plots
8. Creates SHAP interpretability analysis
9. Saves everything to timestamped output directory

### Output Structure:

```
trace_analysis_results/
├── analysis_YYYYMMDD_HHMMSS/
│   ├── plots/
│   │   ├── 01_feature_distributions.png
│   │   ├── 02_target_distribution.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_target_correlations.png
│   │   ├── 05_vif_analysis.png
│   │   ├── 06_model_comparison.png
│   │   ├── 07_evaluation_*.png
│   │   ├── 08_feature_importance_*.png
│   │   ├── 09_shap_summary.png
│   │   └── 10_shap_bar.png
│   ├── reports/
│   │   ├── SUMMARY_REPORT.txt          ⭐ START HERE
│   │   ├── data_summary.txt
│   │   ├── feature_names.txt
│   │   ├── high_correlations.csv
│   │   ├── target_correlations.csv
│   │   ├── vif_analysis.csv
│   │   ├── model_metrics.csv
│   │   ├── classification_report_*.csv
│   │   └── feature_importance_*.csv
│   └── models/
```

## Configuration Options

### Key Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_file` | Path to your CSV file | Required |
| `output_dir` | Where to save results | './trace_analysis' |
| `target_column` | Column to predict | 'success' |
| `test_size` | Test set proportion | 0.2 |
| `random_state` | Random seed | 42 |
| `cv_folds` | Cross-validation folds | 5 |
| `use_xgboost` | Enable XGBoost | True |
| `high_correlation_threshold` | Correlation threshold | 0.9 |
| `vif_threshold` | VIF threshold | 10 |
| `max_features_to_plot` | Max features in plots | 30 |

### Categorical Columns:

By default, these columns are one-hot encoded:
- `annotator`
- `model`
- `environment`
- `agent_type`
- `most_used_tool`

Add more in the `categorical_columns` list if needed.

### Excluded Columns:

These columns are automatically excluded from features:
- `trace_id`, `file_id`, `task_id`, `trial_id` (identifiers)
- `timestamp` (temporal data)
- `score`, `success` (target variables)

## Understanding the Results

### 1. Summary Report (`SUMMARY_REPORT.txt`)
- Overall dataset statistics
- Model performance comparison
- Best model identification
- Top 10 most important features

### 2. Model Comparison (`model_metrics.csv`)
- Side-by-side comparison of all models
- F1 scores, accuracy, ROC-AUC

### 3. Feature Importance
- Multiple perspectives (each model's view)
- SHAP values for interpretability

### 4. Correlation Analysis
- `high_correlations.csv`: Redundant feature pairs
- `vif_analysis.csv`: Multicollinearity issues
- `target_correlations.csv`: Features most correlated with success

## Troubleshooting

### Issue: "XGBoost not available"
**Solution:** Install XGBoost or set `use_xgboost: False` in CONFIG

### Issue: "SHAP not installed"
**Solution:** `pip install shap` or analysis will continue without SHAP

### Issue: "Memory error"
**Solution:** Reduce data size or set `max_features_to_plot` to smaller number

### Issue: "Too many features for VIF"
**Solution:** VIF is computed on top 20 features only (automatic)

## Interpreting Results

### What makes traces successful?

1. **Check SUMMARY_REPORT.txt** for top features
2. **Review feature_importance plots** - which features matter most?
3. **Examine SHAP plots** - how do features interact?
4. **Look at target_correlations.csv** - positive/negative correlations

### Model Selection:

- **High accuracy but low F1?** → Class imbalance, check confusion matrix
- **Random Forest best?** → Non-linear patterns in data
- **Logistic Regression competitive?** → Linear patterns, simpler is better

### Red Flags:

⚠️ **High VIF (>10)**: Features are redundant, consider removing
⚠️ **Perfect accuracy**: Possible data leakage, check excluded columns
⚠️ **Low F1 on minority class**: Need better class balancing

## Extending the Analysis

### Add more models:
```python
from sklearn.svm import SVC
self.models['SVM'] = SVC(probability=True, kernel='rbf')
```

### Hyperparameter tuning:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
```

### Custom feature engineering:
Add your transformations in the `feature_engineering()` method.

## Support

For issues or questions:
1. Check configuration parameters
2. Review error messages in console
3. Examine data_summary.txt for data issues
4. Ensure all dependencies are installed

## License

MIT License - Feel free to modify and use!

---

**Happy Analyzing! 🚀**