# ML Model Comparison Report: ALL
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.8846 |           0.8723 |        0.8723 |    0.8723 |         0.9597 | 0.9337 ± 0.0165 |
| Random Forest       |          0.8942 |           0.8913 |        0.8723 |    0.8817 |         0.9571 | 0.9467 ± 0.0259 |
| XGBoost             |          0.8942 |           0.86   |        0.9149 |    0.8866 |         0.9623 | 0.9486 ± 0.0189 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8846 |
| Precision | 0.8723 |
| Recall | 0.8723 |
| F1 Score | 0.8723 |
| ROC-AUC | 0.9597 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9372 |
| Precision | 0.9215 |
| Recall | 0.9412 |
| F1 Score | 0.9312 |
| ROC-AUC | 0.9846 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9337 |
| CV ROC-AUC (std) | 0.0165 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8723 |
| TNR | 0.8947 |
| PPV | 0.8723 |
| NPV | 0.8947 |
| FPR | 0.1053 |
| FNR | 0.1277 |
| F1_Score | 0.8723 |
| MCC | {0.0: 0.7670772676371781, 1.0: 0.7670772676371781} |
| Kappa | 0.7671 |

### Top 15 Features (by coefficient magnitude)

| feature                        |   coefficient |   abs_coefficient |
|:-------------------------------|--------------:|------------------:|
| missing_validation_count       |      1.80395  |          1.80395  |
| level                          |     -1.50912  |          1.50912  |
| give_up_count                  |     -1.41404  |          1.41404  |
| planning_to_total_ratio        |     -0.978189 |          0.978189 |
| model_gpt-4o                   |     -0.944912 |          0.944912 |
| completion_tokens              |     -0.922726 |          0.922726 |
| assistant_steps_to_first_error |     -0.891476 |          0.891476 |
| thought_to_action_ratio        |     -0.856432 |          0.856432 |
| retry_rate                     |      0.83646  |          0.83646  |
| confidence_marker_ratio        |     -0.768927 |          0.768927 |
| problem_marker_ratio           |     -0.767033 |          0.767033 |
| total_message_length           |      0.748565 |          0.748565 |
| unnecessary_tool_use_count     |      0.703093 |          0.703093 |
| tool_switching_rate            |      0.696323 |          0.696323 |
| tools_used_count               |      0.68844  |          0.68844  |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8942 |
| Precision | 0.8913 |
| Recall | 0.8723 |
| F1 Score | 0.8817 |
| ROC-AUC | 0.9571 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9855 |
| Precision | 0.9738 |
| Recall | 0.9947 |
| F1 Score | 0.9841 |
| ROC-AUC | 0.9996 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9467 |
| CV ROC-AUC (std) | 0.0259 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8723 |
| TNR | 0.9123 |
| PPV | 0.8913 |
| NPV | 0.8966 |
| FPR | 0.0877 |
| FNR | 0.1277 |
| F1_Score | 0.8817 |
| MCC | {0.0: 0.7862369358684218, 1.0: 0.7862369358684218} |
| Kappa | 0.7861 |

### Top 15 Features (by importance)

| feature                      |   importance |
|:-----------------------------|-------------:|
| total_markers                |    0.0557099 |
| self_correction_marker_ratio |    0.0524678 |
| level                        |    0.0441228 |
| avg_tools_per_step           |    0.0377604 |
| self_correction_marker_count |    0.0336134 |
| tool_execution_duration      |    0.0323378 |
| most_used_tool_count         |    0.0319274 |
| qa_score                     |    0.0314644 |
| unique_tool_calls_ratio      |    0.0288042 |
| tool_latency_mean            |    0.0281746 |
| tool_gini_coefficient        |    0.0278002 |
| tool_latency_max             |    0.0253886 |
| step_count                   |    0.0230462 |
| total_calls                  |    0.0220379 |
| positive_marker_count        |    0.0209945 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8942 |
| Precision | 0.8600 |
| Recall | 0.9149 |
| F1 Score | 0.8866 |
| ROC-AUC | 0.9623 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9976 |
| Precision | 0.9947 |
| Recall | 1.0000 |
| F1 Score | 0.9973 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9486 |
| CV ROC-AUC (std) | 0.0189 |

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

| feature                           |   importance |
|:----------------------------------|-------------:|
| self_correction_marker_count      |    0.19222   |
| self_correction_marker_ratio      |    0.0963191 |
| level                             |    0.053823  |
| total_markers                     |    0.0241461 |
| unnecessary_tool_use_count        |    0.0231205 |
| assistant_steps_to_first_planning |    0.0195353 |
| tool_latency_mean                 |    0.0172655 |
| assistant_steps_count             |    0.0170068 |
| qa_score                          |    0.0166884 |
| tool_execution_duration           |    0.0162299 |
| planning_to_total_ratio           |    0.0159657 |
| confidence_marker_ratio           |    0.0159604 |
| give_up_count                     |    0.0158498 |
| problem_marker_ratio              |    0.0151684 |
| success_rate                      |    0.0140741 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
