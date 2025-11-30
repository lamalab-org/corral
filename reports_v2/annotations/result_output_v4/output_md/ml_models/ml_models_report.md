# ML Model Comparison Report: MD
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9333 |           0.8333 |        0.7692 |    0.8    |         0.9888 | 0.9835 ± 0.0138 |
| Random Forest       |          0.9733 |           1      |        0.8462 |    0.9167 |         0.9975 | 0.9972 ± 0.0047 |
| XGBoost             |          0.9733 |           1      |        0.8462 |    0.9167 |         0.995  | 0.9960 ± 0.0036 |

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
| CV ROC-AUC (mean) | 0.9835 |
| CV ROC-AUC (std) | 0.0138 |

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
| avg_tools_per_step                |      1.37738  |          1.37738  |
| tool_latency_mean                 |     -1.17984  |          1.17984  |
| annotatable_steps_count           |      1.10436  |          1.10436  |
| tool_execution_duration           |     -1.09766  |          1.09766  |
| tool_latency_std                  |     -1.07351  |          1.07351  |
| tool_latency_max                  |     -1.04454  |          1.04454  |
| tools_used_count                  |      0.735909 |          0.735909 |
| assistant_steps_to_first_error    |      0.731882 |          0.731882 |
| level                             |     -0.712683 |          0.712683 |
| retry_rate                        |      0.695538 |          0.695538 |
| assistant_steps_count             |     -0.678543 |          0.678543 |
| loop_instance_count               |     -0.609884 |          0.609884 |
| assistant_steps_to_first_planning |     -0.565793 |          0.565793 |
| steps_to_first_planning           |     -0.565793 |          0.565793 |
| error_burstiness                  |     -0.485521 |          0.485521 |

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
| CV ROC-AUC (mean) | 0.9972 |
| CV ROC-AUC (std) | 0.0047 |

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
| retry_rate                |    0.140659  |
| tool_execution_duration   |    0.0616131 |
| avg_tools_per_step        |    0.0469365 |
| tool_latency_mean         |    0.0466561 |
| annotatable_steps_count   |    0.044537  |
| tool_latency_std          |    0.040141  |
| tool_latency_max          |    0.0300823 |
| total_markers             |    0.0252542 |
| uncertainty_marker_ratio  |    0.0236714 |
| problem_marker_ratio      |    0.0224127 |
| marker_balance_ratio      |    0.0204798 |
| max_consecutive_successes |    0.0196496 |
| prompt_tokens             |    0.0187435 |
| negative_marker_count     |    0.0185879 |
| total_message_length      |    0.0183195 |

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
| ROC-AUC | 0.9950 |

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
| CV ROC-AUC (mean) | 0.9960 |
| CV ROC-AUC (std) | 0.0036 |

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
| has_planning_in_first_step     |    0.194289  |
| avg_words_per_message          |    0.127421  |
| retry_rate                     |    0.0654786 |
| tools_used_count               |    0.0479915 |
| assistant_steps_to_first_error |    0.0464911 |
| steps_to_first_planning        |    0.0423995 |
| tool_repetition_rate           |    0.0368658 |
| tool_execution_duration        |    0.0353362 |
| negative_marker_ratio          |    0.0255193 |
| marker_balance                 |    0.0254329 |
| total_message_length           |    0.0244033 |
| total_tokens                   |    0.0240599 |
| avg_tools_per_step             |    0.0223754 |
| uncertainty_marker_ratio       |    0.0219542 |
| annotatable_steps_count        |    0.0215274 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
