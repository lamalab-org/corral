# ML Model Comparison Report: RESISTOR
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9583 |           0.9048 |             1 |      0.95 |         0.9546 | 0.9949 ± 0.0042 |
| Random Forest       |          0.9583 |           0.9048 |             1 |      0.95 |         0.98   | 1.0000 ± 0.0000 |
| XGBoost             |          0.9583 |           0.9048 |             1 |      0.95 |         0.9891 | 0.9989 ± 0.0022 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9583 |
| Precision | 0.9048 |
| Recall | 1.0000 |
| F1 Score | 0.9500 |
| ROC-AUC | 0.9546 |

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
| CV ROC-AUC (mean) | 0.9949 |
| CV ROC-AUC (std) | 0.0042 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.9310 |
| PPV | 0.9048 |
| NPV | 1.0000 |
| FPR | 0.0690 |
| FNR | 0.0000 |
| F1_Score | 0.9500 |
| MCC | {0.0: 0.9178041904566052, 1.0: 0.9178041904566052} |
| Kappa | 0.9144 |

### Top 15 Features (by coefficient magnitude)

| feature                    |   coefficient |   abs_coefficient |
|:---------------------------|--------------:|------------------:|
| avg_thought_length         |      1.07     |          1.07     |
| planning_to_total_ratio    |     -1.03614  |          1.03614  |
| reasoning_to_total_ratio   |     -0.848261 |          0.848261 |
| message_length_std         |     -0.830816 |          0.830816 |
| give_up_count              |     -0.813446 |          0.813446 |
| annotatable_steps_count    |     -0.679261 |          0.679261 |
| has_planning_in_first_step |     -0.668382 |          0.668382 |
| negative_marker_ratio      |     -0.648405 |          0.648405 |
| neutral_marker_count       |     -0.527539 |          0.527539 |
| marker_balance_ratio       |      0.518785 |          0.518785 |
| success_rate               |      0.51024  |          0.51024  |
| question_count             |      0.492426 |          0.492426 |
| assistant_steps_count      |     -0.479967 |          0.479967 |
| neutral_count              |     -0.463364 |          0.463364 |
| total_calls                |     -0.45858  |          0.45858  |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9583 |
| Precision | 0.9048 |
| Recall | 1.0000 |
| F1 Score | 0.9500 |
| ROC-AUC | 0.9800 |

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
| TNR | 0.9310 |
| PPV | 0.9048 |
| NPV | 1.0000 |
| FPR | 0.0690 |
| FNR | 0.0000 |
| F1_Score | 0.9500 |
| MCC | {0.0: 0.9178041904566052, 1.0: 0.9178041904566052} |
| Kappa | 0.9144 |

### Top 15 Features (by importance)

| feature                  |   importance |
|:-------------------------|-------------:|
| total_markers            |    0.101709  |
| assistant_steps_count    |    0.0639222 |
| longest_looping_sequence |    0.0631886 |
| step_count               |    0.0539656 |
| marker_balance           |    0.0531386 |
| total_message_length     |    0.0504183 |
| planning_to_total_ratio  |    0.046027  |
| unique_tool_calls_ratio  |    0.0308023 |
| total_calls              |    0.0272736 |
| avg_thought_length       |    0.0239179 |
| recovery_marker_count    |    0.0236168 |
| planning_statement_count |    0.022831  |
| negative_marker_ratio    |    0.022579  |
| tool_repetition_rate     |    0.0202046 |
| problem_marker_count     |    0.0169689 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9583 |
| Precision | 0.9048 |
| Recall | 1.0000 |
| F1 Score | 0.9500 |
| ROC-AUC | 0.9891 |

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
| CV ROC-AUC (mean) | 0.9989 |
| CV ROC-AUC (std) | 0.0022 |

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
| longest_looping_sequence |    0.14164   |
| assistant_steps_count    |    0.116966  |
| successful_calls         |    0.107362  |
| marker_balance           |    0.0932637 |
| total_markers            |    0.0706832 |
| negative_marker_ratio    |    0.0602185 |
| code_block_count         |    0.0494973 |
| give_up_count            |    0.0353539 |
| problem_marker_ratio     |    0.0329945 |
| positive_marker_count    |    0.0290636 |
| planning_to_total_ratio  |    0.023993  |
| planning_statement_count |    0.019439  |
| total_message_length     |    0.0184345 |
| avg_tools_per_step       |    0.0177801 |
| neutral_marker_count     |    0.0153944 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
