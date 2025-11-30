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
| assistant_steps_to_first_reasoning |      0.97703  |          0.97703  |
| steps_to_first_reasoning           |      0.97703  |          0.97703  |
| reasoning_to_total_ratio           |      0.858269 |          0.858269 |
| positive_marker_ratio              |      0.628997 |          0.628997 |
| backtrack_after_error_rate         |     -0.410062 |          0.410062 |
| marker_balance_ratio               |      0.399531 |          0.399531 |
| positive_marker_count              |      0.389327 |          0.389327 |
| neutral_marker_ratio               |     -0.349506 |          0.349506 |
| message_length_std                 |      0.288725 |          0.288725 |
| tool_latency_max                   |      0.260824 |          0.260824 |
| tool_execution_duration            |      0.256618 |          0.256618 |
| prompt_tokens                      |      0.255064 |          0.255064 |
| total_tokens                       |      0.249566 |          0.249566 |
| max_consecutive_successes          |      0.23802  |          0.23802  |
| annotatable_steps_count            |     -0.234926 |          0.234926 |

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
| positive_marker_ratio        |    0.0807637 |
| total_message_length         |    0.057309  |
| tools_used_count             |    0.0463043 |
| annotatable_steps_count      |    0.0457818 |
| total_calls                  |    0.0456063 |
| thought_count                |    0.044839  |
| avg_tools_per_step           |    0.0425228 |
| avg_assistant_message_length |    0.0370132 |
| longest_looping_sequence     |    0.0349565 |
| most_used_tool_ratio         |    0.0343335 |
| assistant_steps_count        |    0.0312266 |
| action_count                 |    0.0301979 |
| problem_marker_ratio         |    0.0297124 |
| negative_marker_count        |    0.0291308 |
| marker_balance               |    0.0288967 |

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
| positive_marker_ratio    |   0.227485   |
| marker_balance           |   0.213768   |
| avg_tools_per_step       |   0.133972   |
| successful_calls         |   0.078299   |
| completion_tokens        |   0.0685772  |
| avg_message_length       |   0.0635976  |
| steps_to_first_reasoning |   0.0478734  |
| negative_marker_ratio    |   0.0330722  |
| step_count               |   0.0236212  |
| total_calls              |   0.0229753  |
| message_length_std       |   0.0190394  |
| problem_marker_ratio     |   0.0183465  |
| tool_execution_duration  |   0.0156825  |
| tool_latency_max         |   0.00661906 |
| total_tokens             |   0.0064453  |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
