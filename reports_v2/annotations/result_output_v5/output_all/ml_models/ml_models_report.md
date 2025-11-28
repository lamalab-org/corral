# ML Model Comparison Report: ALL
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9033 |           0.9083 |        0.879  |    0.8934 |         0.9485 | 0.9553 ± 0.0048 |
| Random Forest       |          0.9442 |           0.936  |        0.9435 |    0.9398 |         0.9798 | 0.9900 ± 0.0030 |
| XGBoost             |          0.9554 |           0.9444 |        0.9597 |    0.952  |         0.9838 | 0.9931 ± 0.0017 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9033 |
| Precision | 0.9083 |
| Recall | 0.8790 |
| F1 Score | 0.8934 |
| ROC-AUC | 0.9485 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9227 |
| Precision | 0.9124 |
| Recall | 0.9215 |
| F1 Score | 0.9169 |
| ROC-AUC | 0.9775 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9553 |
| CV ROC-AUC (std) | 0.0048 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8790 |
| TNR | 0.9241 |
| PPV | 0.9083 |
| NPV | 0.8993 |
| FPR | 0.0759 |
| FNR | 0.1210 |
| F1_Score | 0.8934 |
| MCC | {0.0: 0.805413059115004, 1.0: 0.805413059115004} |
| Kappa | 0.8051 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| give_up_count                |     -1.72111  |          1.72111  |
| model_gpt-4o                 |     -1.71559  |          1.71559  |
| assistant_steps_count        |     -1.69521  |          1.69521  |
| level                        |     -1.63173  |          1.63173  |
| tools_used_count             |      1.47866  |          1.47866  |
| total_markers                |     -1.14867  |          1.14867  |
| early_final_answer_count     |     -1.13614  |          1.13614  |
| reasoning_to_total_ratio     |      1.08187  |          1.08187  |
| avg_thought_length           |      1.06054  |          1.06054  |
| avg_message_length           |      0.945698 |          0.945698 |
| avg_assistant_message_length |      0.945698 |          0.945698 |
| uncertainty_marker_ratio     |     -0.935935 |          0.935935 |
| code_block_count             |     -0.903028 |          0.903028 |
| validation_after_error_rate  |     -0.899761 |          0.899761 |
| thought_to_action_ratio      |     -0.866249 |          0.866249 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9442 |
| Precision | 0.9360 |
| Recall | 0.9435 |
| F1 Score | 0.9398 |
| ROC-AUC | 0.9798 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9953 |
| Precision | 0.9900 |
| Recall | 1.0000 |
| F1 Score | 0.9950 |
| ROC-AUC | 0.9997 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9900 |
| CV ROC-AUC (std) | 0.0030 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9435 |
| TNR | 0.9448 |
| PPV | 0.9360 |
| NPV | 0.9514 |
| FPR | 0.0552 |
| FNR | 0.0565 |
| F1_Score | 0.9398 |
| MCC | {0.0: 0.8878822939250054, 1.0: 0.8878822939250054} |
| Kappa | 0.8879 |

### Top 15 Features (by importance)

| feature                  |   importance |
|:-------------------------|-------------:|
| total_markers            |    0.0631437 |
| level                    |    0.0538505 |
| uncertainty_marker_count |    0.0457496 |
| tool_latency_mean        |    0.0428858 |
| step_count               |    0.0392682 |
| tool_latency_std         |    0.0371364 |
| uncertainty_marker_ratio |    0.0370478 |
| tool_latency_max         |    0.0307908 |
| tool_execution_duration  |    0.0260508 |
| assistant_steps_count    |    0.0228916 |
| avg_message_length       |    0.0195151 |
| planning_to_total_ratio  |    0.0188593 |
| successful_calls         |    0.0179038 |
| message_length_std       |    0.0172876 |
| marker_balance           |    0.0171486 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9554 |
| Precision | 0.9444 |
| Recall | 0.9597 |
| F1 Score | 0.9520 |
| ROC-AUC | 0.9838 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9991 |
| Precision | 0.9980 |
| Recall | 1.0000 |
| F1 Score | 0.9990 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9931 |
| CV ROC-AUC (std) | 0.0017 |

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

| feature                            |   importance |
|:-----------------------------------|-------------:|
| total_markers                      |    0.0950215 |
| level                              |    0.0772301 |
| uncertainty_marker_count           |    0.0570612 |
| give_up_count                      |    0.0310292 |
| unnecessary_tool_use_count         |    0.0286817 |
| has_tool_error_in_first_step       |    0.0234284 |
| model_gpt-4o                       |    0.021406  |
| tool_latency_std                   |    0.0208996 |
| success_rate                       |    0.0199524 |
| qa_score                           |    0.0186967 |
| negative_marker_count              |    0.0182337 |
| avg_thought_length                 |    0.0181956 |
| tool_repetition_rate               |    0.0173875 |
| assistant_steps_to_first_tool_call |    0.0172613 |
| wrong_planning_count               |    0.0172192 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
