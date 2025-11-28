# ML Model Comparison Report: AFM
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          1      |                1 |           1   |    1      |         1      | 0.9584 ± 0.0233 |
| Random Forest       |          0.9118 |                1 |           0.5 |    0.6667 |         0.9821 | 1.0000 ± 0.0000 |
| XGBoost             |          0.9118 |                1 |           0.5 |    0.6667 |         1      | 0.9964 ± 0.0073 |

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
| Accuracy | 0.9848 |
| Precision | 0.9583 |
| Recall | 0.9583 |
| F1 Score | 0.9583 |
| ROC-AUC | 0.9996 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9584 |
| CV ROC-AUC (std) | 0.0233 |

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
| planning_to_total_ratio            |     -1.09777  |          1.09777  |
| level                              |     -1.01724  |          1.01724  |
| neutral_marker_ratio               |      0.863327 |          0.863327 |
| assistant_steps_to_first_tool_call |      0.722302 |          0.722302 |
| backtrack_trigger_count            |      0.69931  |          0.69931  |
| recovery_marker_count              |     -0.682055 |          0.682055 |
| reasoning_to_total_ratio           |     -0.671518 |          0.671518 |
| unique_tool_calls_ratio            |     -0.521267 |          0.521267 |
| planning_statement_count           |     -0.486517 |          0.486517 |
| backtrack_after_error_rate         |      0.460664 |          0.460664 |
| tools_used_count                   |     -0.456162 |          0.456162 |
| problem_marker_count               |     -0.434389 |          0.434389 |
| code_block_count                   |     -0.422391 |          0.422391 |
| validation_attempt_count           |     -0.406387 |          0.406387 |
| problem_marker_ratio               |     -0.400729 |          0.400729 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9118 |
| Precision | 1.0000 |
| Recall | 0.5000 |
| F1 Score | 0.6667 |
| ROC-AUC | 0.9821 |

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
| TPR | 0.5000 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9032 |
| FPR | 0.0000 |
| FNR | 0.5000 |
| F1_Score | 0.6667 |
| MCC | {0.0: 0.672021505032247, 1.0: 0.672021505032247} |
| Kappa | 0.6222 |

### Top 15 Features (by importance)

| feature                      |   importance |
|:-----------------------------|-------------:|
| level                        |    0.076379  |
| avg_words_per_message        |    0.0756947 |
| planning_to_total_ratio      |    0.0744627 |
| neutral_marker_ratio         |    0.0427559 |
| avg_assistant_message_length |    0.0366949 |
| avg_message_length           |    0.034577  |
| tool_latency_max             |    0.0342989 |
| tool_latency_std             |    0.0342497 |
| retry_rate                   |    0.0290597 |
| total_message_length         |    0.0288988 |
| tool_latency_mean            |    0.027257  |
| total_tokens                 |    0.0262888 |
| tool_execution_duration      |    0.0242508 |
| thought_to_action_ratio      |    0.0217825 |
| avg_tools_per_step           |    0.0198866 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9118 |
| Precision | 1.0000 |
| Recall | 0.5000 |
| F1 Score | 0.6667 |
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
| CV ROC-AUC (mean) | 0.9964 |
| CV ROC-AUC (std) | 0.0073 |

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

| feature                 |   importance |
|:------------------------|-------------:|
| positive_marker_ratio   |    0.141393  |
| level                   |    0.0941438 |
| retry_rate              |    0.0706015 |
| tool_execution_duration |    0.043795  |
| marker_balance_ratio    |    0.0399894 |
| completion_tokens       |    0.0389558 |
| planning_to_total_ratio |    0.0369223 |
| most_used_tool_count    |    0.0364478 |
| marker_balance          |    0.0359807 |
| success_rate            |    0.0338887 |
| avg_tools_per_step      |    0.0335544 |
| assistant_steps_count   |    0.028982  |
| neutral_marker_ratio    |    0.0260343 |
| tool_repetition_rate    |    0.0257806 |
| avg_words_per_message   |    0.0242982 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
