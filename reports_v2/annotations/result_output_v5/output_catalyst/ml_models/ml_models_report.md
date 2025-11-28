# ML Model Comparison Report: CATALYST
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |
| Random Forest       |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |
| XGBoost             |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |

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
| CV ROC-AUC (mean) | 1.0000 |
| CV ROC-AUC (std) | 0.0000 |

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

| feature                            |   coefficient |   abs_coefficient |
|:-----------------------------------|--------------:|------------------:|
| steps_to_first_reasoning           |      0.74161  |          0.74161  |
| assistant_steps_to_first_reasoning |      0.74161  |          0.74161  |
| reasoning_to_total_ratio           |      0.663501 |          0.663501 |
| positive_marker_ratio              |      0.495677 |          0.495677 |
| backtrack_after_error_rate         |     -0.409585 |          0.409585 |
| marker_balance_ratio               |      0.336572 |          0.336572 |
| positive_marker_count              |      0.315796 |          0.315796 |
| tool_latency_max                   |      0.267935 |          0.267935 |
| neutral_marker_ratio               |     -0.255491 |          0.255491 |
| tool_execution_duration            |      0.240291 |          0.240291 |
| message_length_std                 |      0.238724 |          0.238724 |
| prompt_tokens                      |      0.230732 |          0.230732 |
| total_tokens                       |      0.226587 |          0.226587 |
| max_consecutive_successes          |      0.222335 |          0.222335 |
| annotatable_steps_count            |     -0.21578  |          0.21578  |

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
| CV ROC-AUC (mean) | 1.0000 |
| CV ROC-AUC (std) | 0.0000 |

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
| total_message_length         |    0.0815378 |
| avg_message_length           |    0.0569415 |
| positive_marker_ratio        |    0.0554244 |
| avg_words_per_message        |    0.0487734 |
| total_markers                |    0.0460439 |
| problem_marker_ratio         |    0.0451172 |
| loop_instance_count          |    0.0428231 |
| tool_diversity_entropy       |    0.0372205 |
| avg_assistant_message_length |    0.0366543 |
| problem_marker_count         |    0.0359141 |
| step_count                   |    0.0346515 |
| assistant_steps_count        |    0.0335013 |
| completion_tokens            |    0.0330511 |
| thought_count                |    0.0300748 |
| successful_calls             |    0.0290087 |

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
| CV ROC-AUC (mean) | 1.0000 |
| CV ROC-AUC (std) | 0.0000 |

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

| feature                  |   importance |
|:-------------------------|-------------:|
| marker_balance_ratio     |   0.273629   |
| marker_balance           |   0.20435    |
| steps_to_first_reasoning |   0.110504   |
| positive_marker_ratio    |   0.107839   |
| tool_diversity_entropy   |   0.0976423  |
| problem_marker_ratio     |   0.0564842  |
| avg_tools_per_step       |   0.0372725  |
| message_length_std       |   0.0261116  |
| tool_latency_max         |   0.0193959  |
| positive_marker_count    |   0.01572    |
| tool_execution_duration  |   0.0131974  |
| negative_marker_count    |   0.00706982 |
| avg_message_length       |   0.00670022 |
| retry_rate               |   0.00515882 |
| avg_thought_length       |   0.00386358 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
