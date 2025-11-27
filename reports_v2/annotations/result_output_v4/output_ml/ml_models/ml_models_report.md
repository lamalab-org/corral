# ML Model Comparison Report: ML
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.75   |           0.8333 |        0.8333 |    0.8333 |         0.8704 | 0.7510 ± 0.1269 |
| Random Forest       |          0.8333 |           0.8889 |        0.8889 |    0.8889 |         0.963  | 0.8843 ± 0.0978 |
| XGBoost             |          0.9167 |           1      |        0.8889 |    0.9412 |         0.963  | 0.8171 ± 0.1243 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7500 |
| Precision | 0.8333 |
| Recall | 0.8333 |
| F1 Score | 0.8333 |
| ROC-AUC | 0.8704 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9271 |
| Precision | 0.9701 |
| Recall | 0.9286 |
| F1 Score | 0.9489 |
| ROC-AUC | 0.9824 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.7510 |
| CV ROC-AUC (std) | 0.1269 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8333 |
| TNR | 0.5000 |
| PPV | 0.8333 |
| NPV | 0.5000 |
| FPR | 0.5000 |
| FNR | 0.1667 |
| F1_Score | 0.8333 |
| MCC | {0.0: 0.3333333333333333, 1.0: 0.3333333333333333} |
| Kappa | 0.3333 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| problem_marker_count         |      0.907059 |          0.907059 |
| backtrack_after_error_rate   |     -0.688734 |          0.688734 |
| completion_tokens            |     -0.677391 |          0.677391 |
| validation_to_total_ratio    |      0.619129 |          0.619129 |
| negative_marker_ratio        |      0.593561 |          0.593561 |
| backtrack_to_total_ratio     |     -0.545003 |          0.545003 |
| annotatable_steps_count      |     -0.521306 |          0.521306 |
| problem_marker_ratio         |      0.489199 |          0.489199 |
| tool_latency_std             |     -0.486212 |          0.486212 |
| unique_tool_calls_ratio      |     -0.471108 |          0.471108 |
| tool_repetition_rate         |      0.471108 |          0.471108 |
| prompt_tokens                |      0.446892 |          0.446892 |
| avg_message_length           |      0.445888 |          0.445888 |
| avg_assistant_message_length |      0.445888 |          0.445888 |
| total_tokens                 |      0.429055 |          0.429055 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8333 |
| Precision | 0.8889 |
| Recall | 0.8889 |
| F1 Score | 0.8889 |
| ROC-AUC | 0.9630 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9896 |
| Precision | 0.9859 |
| Recall | 1.0000 |
| F1 Score | 0.9929 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8843 |
| CV ROC-AUC (std) | 0.0978 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8889 |
| TNR | 0.6667 |
| PPV | 0.8889 |
| NPV | 0.6667 |
| FPR | 0.3333 |
| FNR | 0.1111 |
| F1_Score | 0.8889 |
| MCC | {0.0: 0.5555555555555556, 1.0: 0.5555555555555556} |
| Kappa | 0.5556 |

### Top 15 Features (by importance)

| feature                      |   importance |
|:-----------------------------|-------------:|
| avg_message_length           |    0.0803803 |
| avg_assistant_message_length |    0.0750031 |
| total_tokens                 |    0.0749761 |
| message_length_std           |    0.0609384 |
| avg_words_per_message        |    0.057799  |
| tool_latency_mean            |    0.054159  |
| total_message_length         |    0.0529365 |
| tool_latency_std             |    0.0503093 |
| tool_latency_max             |    0.0496351 |
| prompt_tokens                |    0.0468177 |
| tool_execution_duration      |    0.0441517 |
| completion_tokens            |    0.0284862 |
| marker_balance_ratio         |    0.0240718 |
| tool_diversity_entropy       |    0.0219777 |
| marker_balance               |    0.0165674 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9167 |
| Precision | 1.0000 |
| Recall | 0.8889 |
| F1 Score | 0.9412 |
| ROC-AUC | 0.9630 |

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
| CV ROC-AUC (mean) | 0.8171 |
| CV ROC-AUC (std) | 0.1243 |

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

| feature                      |   importance |
|:-----------------------------|-------------:|
| avg_assistant_message_length |    0.148304  |
| avg_message_length           |    0.115819  |
| message_length_std           |    0.0723942 |
| error_rate                   |    0.0442931 |
| tool_execution_duration      |    0.0397001 |
| completion_tokens            |    0.0361183 |
| tool_latency_mean            |    0.0360728 |
| total_tokens                 |    0.032585  |
| neutral_count                |    0.0302796 |
| tool_repetition_rate         |    0.0274984 |
| neutral_marker_ratio         |    0.023616  |
| tools_used_count             |    0.0224005 |
| tool_latency_max             |    0.0223254 |
| positive_marker_ratio        |    0.0218191 |
| prompt_tokens                |    0.0212085 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).