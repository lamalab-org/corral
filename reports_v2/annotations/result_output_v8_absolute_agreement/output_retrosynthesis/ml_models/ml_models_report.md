# ML Model Comparison Report: RETROSYNTHESIS
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9167 |           0.9091 |             1 |    0.9524 |              1 | 0.9750 ± 0.0500 |
| Random Forest       |          1      |           1      |             1 |    1      |              1 | 0.8429 ± 0.1513 |
| XGBoost             |          1      |           1      |             1 |    1      |              1 | 0.8125 ± 0.1936 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9167 |
| Precision | 0.9091 |
| Recall | 1.0000 |
| F1 Score | 0.9524 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9750 |
| CV ROC-AUC (std) | 0.0500 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.5000 |
| PPV | 0.9091 |
| NPV | 1.0000 |
| FPR | 0.5000 |
| FNR | 0.0000 |
| F1_Score | 0.9524 |
| MCC | {0.0: 0.674199862463242, 1.0: 0.674199862463242} |
| Kappa | 0.6250 |

### Top 15 Features (by coefficient magnitude)

| feature                        |   coefficient |   abs_coefficient |
|:-------------------------------|--------------:|------------------:|
| give_up_count                  |     -1.06953  |          1.06953  |
| backtrack_after_error_rate     |     -0.618512 |          0.618512 |
| has_tool_error_in_first_step   |      0.320124 |          0.320124 |
| problem_marker_ratio           |     -0.245921 |          0.245921 |
| level                          |     -0.243327 |          0.243327 |
| prompt_tokens                  |     -0.226662 |          0.226662 |
| total_tokens                   |     -0.218311 |          0.218311 |
| syntax_error_count             |      0.20616  |          0.20616  |
| self_correction_marker_count   |     -0.206065 |          0.206065 |
| self_correction_marker_ratio   |     -0.197856 |          0.197856 |
| assistant_steps_to_first_error |     -0.196563 |          0.196563 |
| confidence_marker_count        |     -0.182229 |          0.182229 |
| tool_gini_coefficient          |     -0.175102 |          0.175102 |
| tool_latency_std               |      0.170266 |          0.170266 |
| max_consecutive_failures       |     -0.165301 |          0.165301 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8429 |
| CV ROC-AUC (std) | 0.1513 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 1.0000 |
| FPR | 0.0000 |
| FNR | 0.0000 |
| F1_Score | 1.0000 |
| MCC | {0.0: 1.0, 1.0: 1.0} |
| Kappa | 1.0000 |

### Top 15 Features (by importance)

| feature                 |   importance |
|:------------------------|-------------:|
| hallucination_count     |    0.0796174 |
| give_up_count           |    0.055612  |
| problem_marker_ratio    |    0.0546324 |
| confidence_marker_count |    0.0515414 |
| total_tokens            |    0.0503137 |
| tool_gini_coefficient   |    0.0430605 |
| prompt_tokens           |    0.0376826 |
| problem_marker_count    |    0.0361429 |
| marker_balance          |    0.0356864 |
| level                   |    0.0346827 |
| completion_tokens       |    0.0336564 |
| negative_marker_count   |    0.0326672 |
| error_rate              |    0.0258348 |
| total_markers           |    0.0257297 |
| steps_to_first_error    |    0.0220356 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8125 |
| CV ROC-AUC (std) | 0.1936 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | nan |
| TNR | nan |
| PPV | nan |
| NPV | nan |
| FPR | nan |
| FNR | nan |
| F1_Score | nan |
| MCC | {'0': 'None', '0.0': 'None', '1': 'None', '1.0': 'None'} |
| Kappa | 0.0000 |

### Top 15 Features (by importance)

| feature                   |   importance |
|:--------------------------|-------------:|
| hallucination_count       |    0.145169  |
| give_up_count             |    0.111016  |
| level                     |    0.0804013 |
| problem_marker_count      |    0.0775743 |
| positive_marker_ratio     |    0.0694431 |
| tool_gini_coefficient     |    0.0589775 |
| prompt_tokens             |    0.0439532 |
| confidence_marker_count   |    0.0402053 |
| problem_marker_ratio      |    0.03043   |
| max_consecutive_successes |    0.0288887 |
| step_count                |    0.028865  |
| annotatable_steps_count   |    0.0286101 |
| completion_tokens         |    0.0262126 |
| tool_latency_mean         |    0.0253933 |
| top3_tool_concentration   |    0.025082  |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
