# ML Model Comparison Report: MD
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9333 |           0.8333 |        0.7692 |    0.8    |         0.9888 | 0.9831 ± 0.0134 |
| Random Forest       |          0.9733 |           1      |        0.8462 |    0.9167 |         0.9975 | 0.9988 ± 0.0016 |
| XGBoost             |          0.9733 |           1      |        0.8462 |    0.9167 |         0.9975 | 0.9952 ± 0.0068 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9333 |
| Precision | 0.8333 |
| Recall | 0.7692 |
| F1 Score | 0.8000 |
| ROC-AUC | 0.9888 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9798 |
| Precision | 0.9412 |
| Recall | 0.9412 |
| F1 Score | 0.9412 |
| ROC-AUC | 0.9989 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9831 |
| CV ROC-AUC (std) | 0.0134 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.7692 |
| TNR | 0.9677 |
| PPV | 0.8333 |
| NPV | 0.9524 |
| FPR | 0.0323 |
| FNR | 0.2308 |
| F1_Score | 0.8000 |
| MCC | {0.0: 0.7609533377782035, 1.0: 0.7609533377782035} |
| Kappa | 0.7601 |

### Top 15 Features (by coefficient magnitude)

| feature                           |   coefficient |   abs_coefficient |
|:----------------------------------|--------------:|------------------:|
| avg_tools_per_step                |      1.32503  |          1.32503  |
| tool_latency_mean                 |     -1.19343  |          1.19343  |
| tool_execution_duration           |     -1.10594  |          1.10594  |
| annotatable_steps_count           |      1.09771  |          1.09771  |
| tool_latency_std                  |     -1.09295  |          1.09295  |
| tool_latency_max                  |     -1.0612   |          1.0612   |
| tools_used_count                  |      0.70955  |          0.70955  |
| level                             |     -0.703493 |          0.703493 |
| assistant_steps_to_first_error    |      0.699181 |          0.699181 |
| retry_rate                        |      0.6857   |          0.6857   |
| assistant_steps_count             |     -0.669721 |          0.669721 |
| loop_instance_count               |     -0.60407  |          0.60407  |
| steps_to_first_planning           |     -0.56806  |          0.56806  |
| assistant_steps_to_first_planning |     -0.56806  |          0.56806  |
| message_length_std                |     -0.474223 |          0.474223 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9733 |
| Precision | 1.0000 |
| Recall | 0.8462 |
| F1 Score | 0.9167 |
| ROC-AUC | 0.9975 |

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
| CV ROC-AUC (mean) | 0.9988 |
| CV ROC-AUC (std) | 0.0016 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8462 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9688 |
| FPR | 0.0000 |
| FNR | 0.1538 |
| F1_Score | 0.9167 |
| MCC | {0.0: 0.9053792235641033, 1.0: 0.9053792235641033} |
| Kappa | 0.9009 |

### Top 15 Features (by importance)

| feature                   |   importance |
|:--------------------------|-------------:|
| retry_rate                |    0.112465  |
| tool_latency_mean         |    0.0613329 |
| tool_execution_duration   |    0.049439  |
| tool_latency_max          |    0.0474672 |
| avg_tools_per_step        |    0.0456295 |
| tool_latency_std          |    0.0398838 |
| annotatable_steps_count   |    0.0354652 |
| tool_diversity_entropy    |    0.0298922 |
| uncertainty_marker_ratio  |    0.0283968 |
| uncertainty_marker_count  |    0.0277658 |
| max_consecutive_successes |    0.0252968 |
| negative_marker_count     |    0.0240629 |
| total_markers             |    0.0202185 |
| planning_to_total_ratio   |    0.0185798 |
| avg_words_per_message     |    0.0184564 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9733 |
| Precision | 1.0000 |
| Recall | 0.8462 |
| F1 Score | 0.9167 |
| ROC-AUC | 0.9975 |

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
| CV ROC-AUC (mean) | 0.9952 |
| CV ROC-AUC (std) | 0.0068 |

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

| feature                        |   importance |
|:-------------------------------|-------------:|
| successful_calls               |    0.0884345 |
| success_rate                   |    0.0788306 |
| uncertainty_marker_count       |    0.0772519 |
| retry_rate                     |    0.074651  |
| most_used_tool_count           |    0.0607341 |
| problem_marker_count           |    0.0538337 |
| tool_execution_duration        |    0.0448971 |
| assistant_steps_to_first_error |    0.0354107 |
| avg_tools_per_step             |    0.0338768 |
| negative_marker_count          |    0.0315358 |
| tool_latency_mean              |    0.0272943 |
| most_used_tool_ratio           |    0.0231277 |
| total_markers                  |    0.0198607 |
| total_calls                    |    0.0187478 |
| marker_balance                 |    0.0187404 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
