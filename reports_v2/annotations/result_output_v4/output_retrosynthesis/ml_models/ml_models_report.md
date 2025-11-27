# ML Model Comparison Report: RETROSYNTHESIS
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.8393 |           0.8372 |        0.9474 |    0.8889 |         0.9459 | 0.9405 ± 0.0453 |
| Random Forest       |          0.8929 |           0.9211 |        0.9211 |    0.9211 |         0.9664 | 0.9806 ± 0.0094 |
| XGBoost             |          0.8571 |           0.8947 |        0.8947 |    0.8947 |         0.9547 | 0.9883 ± 0.0065 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8393 |
| Precision | 0.8372 |
| Recall | 0.9474 |
| F1 Score | 0.8889 |
| ROC-AUC | 0.9459 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9777 |
| Precision | 0.9867 |
| Recall | 0.9801 |
| F1 Score | 0.9834 |
| ROC-AUC | 0.9986 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9405 |
| CV ROC-AUC (std) | 0.0453 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9474 |
| TNR | 0.6111 |
| PPV | 0.8372 |
| NPV | 0.8462 |
| FPR | 0.3889 |
| FNR | 0.0526 |
| F1_Score | 0.8889 |
| MCC | {0.0: 0.617773689518041, 1.0: 0.617773689518041} |
| Kappa | 0.6025 |

### Top 15 Features (by coefficient magnitude)

| feature                            |   coefficient |   abs_coefficient |
|:-----------------------------------|--------------:|------------------:|
| level                              |     -1.58313  |          1.58313  |
| give_up_count                      |     -1.26807  |          1.26807  |
| early_final_answer_count           |     -1.2511   |          1.2511   |
| assistant_steps_count              |     -0.938883 |          0.938883 |
| retry_rate                         |      0.800006 |          0.800006 |
| has_reasoning_in_first_step        |     -0.691058 |          0.691058 |
| neutral_count                      |     -0.681008 |          0.681008 |
| most_used_tool_count               |     -0.667045 |          0.667045 |
| tool_diversity_entropy             |     -0.655912 |          0.655912 |
| validation_after_error_rate        |     -0.620516 |          0.620516 |
| assistant_steps_to_first_tool_call |      0.598917 |          0.598917 |
| has_tool_error_in_first_step       |      0.579217 |          0.579217 |
| neutral_marker_count               |     -0.574862 |          0.574862 |
| message_length_std                 |      0.567124 |          0.567124 |
| max_consecutive_successes          |      0.539869 |          0.539869 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8929 |
| Precision | 0.9211 |
| Recall | 0.9211 |
| F1 Score | 0.9211 |
| ROC-AUC | 0.9664 |

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
| CV ROC-AUC (mean) | 0.9806 |
| CV ROC-AUC (std) | 0.0094 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9211 |
| TNR | 0.8333 |
| PPV | 0.9211 |
| NPV | 0.8333 |
| FPR | 0.1667 |
| FNR | 0.0789 |
| F1_Score | 0.9211 |
| MCC | {0.0: 0.7543859649122807, 1.0: 0.7543859649122807} |
| Kappa | 0.7544 |

### Top 15 Features (by importance)

| feature                            |   importance |
|:-----------------------------------|-------------:|
| assistant_steps_to_first_tool_call |    0.082241  |
| level                              |    0.0600669 |
| uncertainty_marker_ratio           |    0.0464391 |
| total_tokens                       |    0.0450478 |
| uncertainty_marker_count           |    0.0390386 |
| prompt_tokens                      |    0.035578  |
| problem_marker_ratio               |    0.0285552 |
| give_up_count                      |    0.0271556 |
| negative_marker_count              |    0.0261809 |
| tool_latency_std                   |    0.0261059 |
| annotatable_steps_count            |    0.0256954 |
| positive_marker_ratio              |    0.0247852 |
| completion_tokens                  |    0.0246657 |
| hallucination_count                |    0.0216472 |
| problem_marker_count               |    0.0210953 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8571 |
| Precision | 0.8947 |
| Recall | 0.8947 |
| F1 Score | 0.8947 |
| ROC-AUC | 0.9547 |

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
| CV ROC-AUC (mean) | 0.9883 |
| CV ROC-AUC (std) | 0.0065 |

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
| assistant_steps_to_first_tool_call |    0.102857  |
| level                              |    0.099412  |
| uncertainty_marker_count           |    0.0444914 |
| has_tool_error_in_first_step       |    0.0331827 |
| most_used_tool_count               |    0.0291469 |
| early_final_answer_count           |    0.0262776 |
| validation_attempt_count           |    0.0237235 |
| code_block_count                   |    0.0235488 |
| tool_latency_std                   |    0.0223984 |
| backtrack_to_total_ratio           |    0.0213118 |
| hallucination_count                |    0.0208744 |
| successful_calls                   |    0.0207543 |
| completion_tokens                  |    0.0206526 |
| positive_marker_ratio              |    0.0199233 |
| backtrack_trigger_count            |    0.0199155 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).