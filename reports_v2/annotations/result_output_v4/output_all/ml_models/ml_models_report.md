# ML Model Comparison Report: ALL
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.8848 |           0.8843 |        0.8629 |    0.8735 |         0.9462 | 0.9548 ± 0.0046 |
| Random Forest       |          0.9368 |           0.9213 |        0.9435 |    0.9323 |         0.9799 | 0.9903 ± 0.0021 |
| XGBoost             |          0.9591 |           0.952  |        0.9597 |    0.9558 |         0.9808 | 0.9934 ± 0.0022 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8848 |
| Precision | 0.8843 |
| Recall | 0.8629 |
| F1 Score | 0.8735 |
| ROC-AUC | 0.9462 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9209 |
| Precision | 0.9087 |
| Recall | 0.9215 |
| F1 Score | 0.9151 |
| ROC-AUC | 0.9758 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9548 |
| CV ROC-AUC (std) | 0.0046 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8629 |
| TNR | 0.9034 |
| PPV | 0.8843 |
| NPV | 0.8851 |
| FPR | 0.0966 |
| FNR | 0.1371 |
| F1_Score | 0.8735 |
| MCC | {0.0: 0.7678905333459233, 1.0: 0.7678905333459233} |
| Kappa | 0.7677 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| tools_used_count             |      1.79129  |          1.79129  |
| give_up_count                |     -1.73754  |          1.73754  |
| assistant_steps_count        |     -1.7206   |          1.7206   |
| level                        |     -1.64049  |          1.64049  |
| early_final_answer_count     |     -1.21361  |          1.21361  |
| avg_thought_length           |      1.17887  |          1.17887  |
| total_markers                |     -1.17405  |          1.17405  |
| thought_to_action_ratio      |     -1.17072  |          1.17072  |
| code_block_count             |     -1.12072  |          1.12072  |
| uncertainty_marker_ratio     |     -0.966721 |          0.966721 |
| marker_balance               |     -0.93822  |          0.93822  |
| validation_after_error_rate  |     -0.900564 |          0.900564 |
| avg_message_length           |      0.89187  |          0.89187  |
| avg_assistant_message_length |      0.89187  |          0.89187  |
| most_used_tool_ratio         |      0.828072 |          0.828072 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9368 |
| Precision | 0.9213 |
| Recall | 0.9435 |
| F1 Score | 0.9323 |
| ROC-AUC | 0.9799 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9926 |
| Precision | 0.9842 |
| Recall | 1.0000 |
| F1 Score | 0.9920 |
| ROC-AUC | 0.9997 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9903 |
| CV ROC-AUC (std) | 0.0021 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9435 |
| TNR | 0.9310 |
| PPV | 0.9213 |
| NPV | 0.9507 |
| FPR | 0.0690 |
| FNR | 0.0565 |
| F1_Score | 0.9323 |
| MCC | {0.0: 0.8732724871940608, 1.0: 0.8732724871940608} |
| Kappa | 0.8731 |

### Top 15 Features (by importance)

| feature                      |   importance |
|:-----------------------------|-------------:|
| total_markers                |    0.064248  |
| level                        |    0.0552    |
| step_count                   |    0.0453554 |
| uncertainty_marker_count     |    0.042757  |
| tool_latency_std             |    0.0359806 |
| uncertainty_marker_ratio     |    0.035336  |
| tool_latency_mean            |    0.0328699 |
| tool_execution_duration      |    0.0327081 |
| tool_latency_max             |    0.0313104 |
| assistant_steps_count        |    0.0285692 |
| longest_looping_sequence     |    0.0219213 |
| total_calls                  |    0.019245  |
| message_length_std           |    0.0188842 |
| unique_tool_calls_ratio      |    0.0184992 |
| avg_assistant_message_length |    0.0182816 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9591 |
| Precision | 0.9520 |
| Recall | 0.9597 |
| F1 Score | 0.9558 |
| ROC-AUC | 0.9808 |

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
| CV ROC-AUC (mean) | 0.9934 |
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

| feature                      |   importance |
|:-----------------------------|-------------:|
| uncertainty_marker_count     |    0.098514  |
| level                        |    0.0784287 |
| total_markers                |    0.0706242 |
| give_up_count                |    0.0319067 |
| loop_instance_count          |    0.0295935 |
| tool_latency_std             |    0.0278822 |
| has_tool_error_in_first_step |    0.0277896 |
| marker_balance               |    0.027447  |
| avg_thought_length           |    0.0206996 |
| steps_to_first_planning      |    0.0199323 |
| uncertainty_marker_ratio     |    0.0196468 |
| unnecessary_tool_use_count   |    0.0175608 |
| tool_latency_max             |    0.0174795 |
| success_rate                 |    0.0161042 |
| recovery_marker_ratio        |    0.0152908 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).