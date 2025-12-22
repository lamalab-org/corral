# ML Model Comparison Report: CATALYST
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          1      |           1      |             1 |    1      |            1   | nan ± nan       |
| Random Forest       |          1      |           1      |             1 |    1      |            1   | nan ± nan       |
| XGBoost             |          0.9167 |           0.9167 |             1 |    0.9565 |            0.5 | nan ± nan       |

---

## 1. Logistic Regression

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
| CV ROC-AUC (mean) | nan |
| CV ROC-AUC (std) | nan |

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

### Top 15 Features (by coefficient magnitude)

| feature                   |   coefficient |   abs_coefficient |
|:--------------------------|--------------:|------------------:|
| reasoning_statement_count |      0.517459 |          0.517459 |
| reasoning_to_total_ratio  |      0.494609 |          0.494609 |
| positive_marker_ratio     |      0.378652 |          0.378652 |
| positive_marker_count     |      0.357999 |          0.357999 |
| prompt_tokens             |      0.306243 |          0.306243 |
| total_tokens              |      0.300838 |          0.300838 |
| marker_balance_ratio      |      0.287765 |          0.287765 |
| step_count                |      0.264615 |          0.264615 |
| tool_execution_duration   |      0.173523 |          0.173523 |
| marker_balance            |      0.173481 |          0.173481 |
| max_consecutive_successes |      0.167294 |          0.167294 |
| problem_marker_ratio      |     -0.164818 |          0.164818 |
| negative_marker_ratio     |     -0.157657 |          0.157657 |
| top3_tool_concentration   |     -0.154099 |          0.154099 |
| tools_used_count          |      0.150268 |          0.150268 |

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
| CV ROC-AUC (mean) | nan |
| CV ROC-AUC (std) | nan |

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

| feature                      |   importance |
|:-----------------------------|-------------:|
| positive_marker_ratio        |    0.116177  |
| neutral_count                |    0.068616  |
| avg_words_per_message        |    0.0495037 |
| message_length_std           |    0.0478261 |
| step_count                   |    0.0446156 |
| total_message_length         |    0.0428111 |
| avg_assistant_message_length |    0.0385343 |
| neutral_marker_ratio         |    0.0334695 |
| assistant_steps_count        |    0.031555  |
| longest_looping_sequence     |    0.0300955 |
| negative_marker_count        |    0.0294949 |
| total_calls                  |    0.0294686 |
| avg_message_length           |    0.0290499 |
| reasoning_statement_count    |    0.0246761 |
| total_tokens                 |    0.0242172 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9167 |
| Precision | 0.9167 |
| Recall | 1.0000 |
| F1 Score | 0.9565 |
| ROC-AUC | 0.5000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9583 |
| Precision | 0.9583 |
| Recall | 1.0000 |
| F1 Score | 0.9787 |
| ROC-AUC | 0.5000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | nan |
| CV ROC-AUC (std) | nan |

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
| MCC | {'0.0': 'None', '1': 'None', '1.0': 'None'} |
| Kappa | 0.0000 |

### Top 15 Features (by importance)

| feature                     |   importance |
|:----------------------------|-------------:|
| level                       |            0 |
| tool_execution_duration     |            0 |
| validation_after_error_rate |            0 |
| reasoning_to_total_ratio    |            0 |
| planning_to_total_ratio     |            0 |
| backtrack_to_total_ratio    |            0 |
| validation_to_total_ratio   |            0 |
| neutral_marker_ratio        |            0 |
| negative_marker_ratio       |            0 |
| positive_marker_ratio       |            0 |
| marker_balance_ratio        |            0 |
| marker_balance              |            0 |
| total_markers               |            0 |
| most_used_tool_ratio        |            0 |
| most_used_tool_count        |            0 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
