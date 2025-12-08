# ML Model Comparison Report: MD
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9889 |                1 |        0.9412 |    0.9697 |         0.9968 | 0.9986 ± 0.0012 |
| Random Forest       |          1      |                1 |        1      |    1      |         1      | 1.0000 ± 0.0000 |
| XGBoost             |          1      |                1 |        1      |    1      |         1      | 1.0000 ± 0.0000 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9889 |
| Precision | 1.0000 |
| Recall | 0.9412 |
| F1 Score | 0.9697 |
| ROC-AUC | 0.9968 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9972 |
| Precision | 1.0000 |
| Recall | 0.9853 |
| F1 Score | 0.9926 |
| ROC-AUC | 0.9997 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9986 |
| CV ROC-AUC (std) | 0.0012 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9412 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9865 |
| FPR | 0.0000 |
| FNR | 0.0588 |
| F1_Score | 0.9697 |
| MCC | {0.0: 0.9635651870186781, 1.0: 0.9635651870186781} |
| Kappa | 0.9629 |

### Top 15 Features (by coefficient magnitude)

| feature                           |   coefficient |   abs_coefficient |
|:----------------------------------|--------------:|------------------:|
| avg_tools_per_step                |      2.0631   |          2.0631   |
| tool_latency_mean                 |     -1.80668  |          1.80668  |
| tool_latency_std                  |     -1.51605  |          1.51605  |
| tool_execution_duration           |     -1.49108  |          1.49108  |
| annotatable_steps_count           |      1.38298  |          1.38298  |
| tool_latency_max                  |     -1.35791  |          1.35791  |
| assistant_steps_to_first_error    |      1.20633  |          1.20633  |
| tools_used_count                  |      1.0698   |          1.0698   |
| total_message_length              |      1.04358  |          1.04358  |
| steps_to_first_error              |     -0.946963 |          0.946963 |
| prompt_tokens                     |     -0.844351 |          0.844351 |
| total_tokens                      |     -0.841588 |          0.841588 |
| assistant_steps_to_first_planning |     -0.781493 |          0.781493 |
| steps_to_first_planning           |     -0.781493 |          0.781493 |
| uncertainty_marker_count          |      0.76037  |          0.76037  |

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

| feature                   |   importance |
|:--------------------------|-------------:|
| retry_rate                |    0.083852  |
| tool_switching_rate       |    0.0722414 |
| tool_latency_mean         |    0.0438681 |
| tool_execution_duration   |    0.0405767 |
| annotatable_steps_count   |    0.0392115 |
| max_consecutive_successes |    0.0309391 |
| tool_latency_std          |    0.0303924 |
| avg_tools_per_step        |    0.0297648 |
| problem_marker_ratio      |    0.0245344 |
| uncertainty_marker_ratio  |    0.0236623 |
| total_tokens              |    0.0221392 |
| tool_latency_max          |    0.0197505 |
| max_tool_burst_length     |    0.0175276 |
| total_markers             |    0.0171989 |
| verification_marker_ratio |    0.0166736 |

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

| feature                           |   importance |
|:----------------------------------|-------------:|
| assistant_steps_to_first_error    |    0.0996777 |
| most_used_tool_count              |    0.0860597 |
| retry_rate                        |    0.0744937 |
| marker_balance                    |    0.0454653 |
| negative_marker_ratio             |    0.0435231 |
| neutral_marker_ratio              |    0.0376458 |
| tool_switching_rate               |    0.0334587 |
| message_length_std                |    0.0333739 |
| total_calls                       |    0.0322713 |
| error_burstiness                  |    0.0311589 |
| avg_tools_per_step                |    0.0286791 |
| uncertainty_marker_count          |    0.0281687 |
| total_message_length              |    0.0268168 |
| assistant_steps_to_first_planning |    0.0248807 |
| avg_words_per_message             |    0.024712  |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
