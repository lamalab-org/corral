# ML Model Comparison Report: RETROSYNTHESIS
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.8929 |           0.9    |        0.9474 |    0.9231 |         0.9518 | 0.9424 ± 0.0395 |
| Random Forest       |          0.875  |           0.8974 |        0.9211 |    0.9091 |         0.9605 | 0.9823 ± 0.0060 |
| XGBoost             |          0.8571 |           0.9167 |        0.8684 |    0.8919 |         0.9576 | 0.9833 ± 0.0071 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8929 |
| Precision | 0.9000 |
| Recall | 0.9474 |
| F1 Score | 0.9231 |
| ROC-AUC | 0.9518 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9821 |
| Precision | 0.9868 |
| Recall | 0.9868 |
| F1 Score | 0.9868 |
| ROC-AUC | 0.9990 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9424 |
| CV ROC-AUC (std) | 0.0395 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9474 |
| TNR | 0.7778 |
| PPV | 0.9000 |
| NPV | 0.8750 |
| FPR | 0.2222 |
| FNR | 0.0526 |
| F1_Score | 0.9231 |
| MCC | {0.0: 0.7496587917803454, 1.0: 0.7496587917803454} |
| Kappa | 0.7470 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| level                        |     -1.35891  |          1.35891  |
| early_final_answer_count     |     -1.23608  |          1.23608  |
| give_up_count                |     -1.22664  |          1.22664  |
| qa_score                     |      1.03441  |          1.03441  |
| assistant_steps_count        |     -0.867306 |          0.867306 |
| tool_diversity_entropy       |     -0.716947 |          0.716947 |
| validation_after_error_rate  |     -0.694587 |          0.694587 |
| retry_rate                   |      0.662337 |          0.662337 |
| has_reasoning_in_first_step  |     -0.646732 |          0.646732 |
| most_used_tool_ratio         |      0.615136 |          0.615136 |
| neutral_count                |     -0.614207 |          0.614207 |
| most_used_tool_count         |     -0.610475 |          0.610475 |
| has_tool_error_in_first_step |      0.606356 |          0.606356 |
| annotatable_steps_count      |     -0.600697 |          0.600697 |
| reasoning_to_total_ratio     |      0.547934 |          0.547934 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8750 |
| Precision | 0.8974 |
| Recall | 0.9211 |
| F1 Score | 0.9091 |
| ROC-AUC | 0.9605 |

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
| CV ROC-AUC (mean) | 0.9823 |
| CV ROC-AUC (std) | 0.0060 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9211 |
| TNR | 0.7778 |
| PPV | 0.8974 |
| NPV | 0.8235 |
| FPR | 0.2222 |
| FNR | 0.0789 |
| F1_Score | 0.9091 |
| MCC | {0.0: 0.7098115821544867, 1.0: 0.7098115821544867} |
| Kappa | 0.7092 |

### Top 15 Features (by importance)

| feature                            |   importance |
|:-----------------------------------|-------------:|
| level                              |    0.0764667 |
| assistant_steps_to_first_tool_call |    0.0568912 |
| uncertainty_marker_count           |    0.0530435 |
| uncertainty_marker_ratio           |    0.0468085 |
| total_tokens                       |    0.0407404 |
| problem_marker_count               |    0.0385102 |
| problem_marker_ratio               |    0.0345939 |
| give_up_count                      |    0.0336266 |
| prompt_tokens                      |    0.031358  |
| completion_tokens                  |    0.0226574 |
| positive_marker_ratio              |    0.0218658 |
| early_final_answer_count           |    0.0202572 |
| retry_rate                         |    0.0185359 |
| step_count                         |    0.01837   |
| hallucination_count                |    0.0183124 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8571 |
| Precision | 0.9167 |
| Recall | 0.8684 |
| F1 Score | 0.8919 |
| ROC-AUC | 0.9576 |

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
| CV ROC-AUC (mean) | 0.9833 |
| CV ROC-AUC (std) | 0.0071 |

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
| assistant_steps_to_first_tool_call |    0.208028  |
| level                              |    0.0884781 |
| uncertainty_marker_count           |    0.0542367 |
| has_tool_error_in_first_step       |    0.0392351 |
| total_markers                      |    0.0368599 |
| completion_tokens                  |    0.0238569 |
| early_final_answer_count           |    0.0228559 |
| positive_marker_ratio              |    0.0211393 |
| total_tokens                       |    0.019811  |
| backtrack_trigger_count            |    0.0185603 |
| unique_tool_calls_ratio            |    0.0182196 |
| qa_score                           |    0.0177559 |
| error_rate                         |    0.0177191 |
| total_calls                        |    0.0173667 |
| wrong_planning_count               |    0.017132  |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).