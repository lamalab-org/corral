# ML Model Comparison Report: ALL
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9282 |           0.9253 |        0.9314 |    0.9283 |         0.9835 | 0.9778 ± 0.0043 |
| Random Forest       |          0.9853 |           0.9837 |        0.9869 |    0.9853 |         0.9994 | 0.9991 ± 0.0005 |
| XGBoost             |          0.9935 |           0.9935 |        0.9935 |    0.9935 |         0.9998 | 0.9995 ± 0.0005 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9282 |
| Precision | 0.9253 |
| Recall | 0.9314 |
| F1 Score | 0.9283 |
| ROC-AUC | 0.9835 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9478 |
| Precision | 0.9477 |
| Recall | 0.9477 |
| F1 Score | 0.9477 |
| ROC-AUC | 0.9851 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9778 |
| CV ROC-AUC (std) | 0.0043 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9314 |
| TNR | 0.9251 |
| PPV | 0.9253 |
| NPV | 0.9311 |
| FPR | 0.0749 |
| FNR | 0.0686 |
| F1_Score | 0.9283 |
| MCC | {0.0: 0.8564630992277974, 1.0: 0.8564630992277974} |
| Kappa | 0.8564 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| assistant_steps_count        |      -3.18551 |           3.18551 |
| self_correction_marker_ratio |      -2.50476 |           2.50476 |
| self_correction_marker_count |      -2.15881 |           2.15881 |
| most_used_tool_ratio         |       1.89859 |           1.89859 |
| tools_used_count             |       1.84333 |           1.84333 |
| reasoning_to_total_ratio     |       1.73951 |           1.73951 |
| level                        |      -1.69461 |           1.69461 |
| tool_latency_max             |      -1.68959 |           1.68959 |
| avg_message_length           |       1.67658 |           1.67658 |
| avg_assistant_message_length |       1.67658 |           1.67658 |
| give_up_count                |      -1.62414 |           1.62414 |
| avg_words_per_message        |      -1.5506  |           1.5506  |
| max_consecutive_successes    |       1.54848 |           1.54848 |
| neutral_count                |       1.4959  |           1.4959  |
| confidence_marker_ratio      |      -1.43875 |           1.43875 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9853 |
| Precision | 0.9837 |
| Recall | 0.9869 |
| F1 Score | 0.9853 |
| ROC-AUC | 0.9994 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9980 |
| Precision | 0.9959 |
| Recall | 1.0000 |
| F1 Score | 0.9980 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9991 |
| CV ROC-AUC (std) | 0.0005 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9869 |
| TNR | 0.9837 |
| PPV | 0.9837 |
| NPV | 0.9869 |
| FPR | 0.0163 |
| FNR | 0.0131 |
| F1_Score | 0.9853 |
| MCC | {0.0: 0.9706414596240234, 1.0: 0.9706414596240234} |
| Kappa | 0.9706 |

### Top 15 Features (by importance)

| feature                  |   importance |
|:-------------------------|-------------:|
| total_markers            |    0.0501478 |
| level                    |    0.0421201 |
| uncertainty_marker_count |    0.0408062 |
| tool_execution_duration  |    0.0402464 |
| step_count               |    0.0336825 |
| confidence_marker_ratio  |    0.0335419 |
| tool_latency_mean        |    0.0311341 |
| uncertainty_marker_ratio |    0.0309847 |
| tool_latency_max         |    0.0278254 |
| planning_to_total_ratio  |    0.0277897 |
| tool_latency_std         |    0.0268751 |
| assistant_steps_count    |    0.0259877 |
| confidence_marker_count  |    0.0212964 |
| marker_balance           |    0.0181037 |
| tool_gini_coefficient    |    0.0177052 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9935 |
| Precision | 0.9935 |
| Recall | 0.9935 |
| F1 Score | 0.9935 |
| ROC-AUC | 0.9998 |

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
| CV ROC-AUC (mean) | 0.9995 |
| CV ROC-AUC (std) | 0.0005 |

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
| uncertainty_marker_count |    0.104155  |
| total_markers            |    0.0700441 |
| level                    |    0.0517548 |
| loop_instance_count      |    0.0493948 |
| uncertainty_marker_ratio |    0.0379173 |
| give_up_count            |    0.0375632 |
| longest_looping_sequence |    0.0286158 |
| marker_balance           |    0.026791  |
| tool_latency_std         |    0.0249468 |
| tool_execution_duration  |    0.018577  |
| confidence_marker_ratio  |    0.0165471 |
| qa_score                 |    0.0162454 |
| steps_to_first_planning  |    0.0149935 |
| tool_latency_mean        |    0.013758  |
| step_count               |    0.0136012 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
