# ML Model Comparison Report: ML
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.75   |           0.8333 |        0.8333 |    0.8333 |         0.8704 | 0.7481 ± 0.1294 |
| Random Forest       |          0.8333 |           0.8889 |        0.8889 |    0.8889 |         0.9537 | 0.8943 ± 0.0855 |
| XGBoost             |          0.9167 |           1      |        0.8889 |    0.9412 |         0.963  | 0.8143 ± 0.1252 |

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
| CV ROC-AUC (mean) | 0.7481 |
| CV ROC-AUC (std) | 0.1294 |

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
| problem_marker_count         |      0.905198 |          0.905198 |
| backtrack_after_error_rate   |     -0.691284 |          0.691284 |
| completion_tokens            |     -0.685967 |          0.685967 |
| validation_to_total_ratio    |      0.618084 |          0.618084 |
| negative_marker_ratio        |      0.593026 |          0.593026 |
| backtrack_to_total_ratio     |     -0.544765 |          0.544765 |
| annotatable_steps_count      |     -0.5192   |          0.5192   |
| tool_latency_std             |     -0.489674 |          0.489674 |
| problem_marker_ratio         |      0.488819 |          0.488819 |
| unique_tool_calls_ratio      |     -0.471367 |          0.471367 |
| tool_repetition_rate         |      0.471367 |          0.471367 |
| avg_message_length           |      0.446695 |          0.446695 |
| avg_assistant_message_length |      0.446695 |          0.446695 |
| prompt_tokens                |      0.436851 |          0.436851 |
| total_tokens                 |      0.418973 |          0.418973 |

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
| ROC-AUC | 0.9537 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9792 |
| Precision | 0.9722 |
| Recall | 1.0000 |
| F1 Score | 0.9859 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8943 |
| CV ROC-AUC (std) | 0.0855 |

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
| avg_message_length           |    0.0836926 |
| avg_assistant_message_length |    0.0667608 |
| avg_words_per_message        |    0.0639365 |
| tool_latency_std             |    0.0628302 |
| message_length_std           |    0.062636  |
| prompt_tokens                |    0.0569781 |
| tool_latency_max             |    0.053215  |
| total_tokens                 |    0.051014  |
| total_message_length         |    0.0495534 |
| tool_latency_mean            |    0.0427992 |
| tool_diversity_entropy       |    0.0332782 |
| tool_execution_duration      |    0.0325355 |
| completion_tokens            |    0.0236412 |
| positive_marker_ratio        |    0.022634  |
| max_consecutive_successes    |    0.0214889 |

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
| CV ROC-AUC (mean) | 0.8143 |
| CV ROC-AUC (std) | 0.1252 |

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
| avg_message_length           |    0.150556  |
| avg_assistant_message_length |    0.100466  |
| message_length_std           |    0.054744  |
| error_rate                   |    0.054243  |
| backtrack_trigger_count      |    0.0541992 |
| completion_tokens            |    0.0469923 |
| tool_latency_mean            |    0.0370243 |
| total_tokens                 |    0.0367096 |
| tool_latency_std             |    0.0357941 |
| neutral_marker_ratio         |    0.0314117 |
| tool_diversity_entropy       |    0.0284555 |
| tool_execution_duration      |    0.0283787 |
| marker_balance_ratio         |    0.0263918 |
| tool_latency_max             |    0.0247416 |
| prompt_tokens                |    0.0235043 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
