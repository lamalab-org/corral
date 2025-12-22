# ML Model Comparison Report: RESISTOR
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9091 |           0.8182 |             1 |    0.9    |         1      | 0.9944 ± 0.0069 |
| Random Forest       |          0.9545 |           0.9    |             1 |    0.9474 |         0.9744 | 0.9916 ± 0.0112 |
| XGBoost             |          0.9545 |           0.9    |             1 |    0.9474 |         1      | 0.9866 ± 0.0118 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9091 |
| Precision | 0.8182 |
| Recall | 1.0000 |
| F1 Score | 0.9000 |
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
| CV ROC-AUC (mean) | 0.9944 |
| CV ROC-AUC (std) | 0.0069 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.8462 |
| PPV | 0.8182 |
| NPV | 1.0000 |
| FPR | 0.1538 |
| FNR | 0.0000 |
| F1_Score | 0.9000 |
| MCC | {0.0: 0.8320502943378437, 1.0: 0.8320502943378437} |
| Kappa | 0.8182 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| qa_score                     |      0.943141 |          0.943141 |
| neutral_marker_count         |     -0.506951 |          0.506951 |
| confidence_marker_ratio      |     -0.496721 |          0.496721 |
| neutral_count                |     -0.475176 |          0.475176 |
| model_gpt-4o                 |     -0.468277 |          0.468277 |
| top3_tool_concentration      |      0.467732 |          0.467732 |
| rare_tool_ratio              |      0.405942 |          0.405942 |
| tool_latency_mean            |     -0.402712 |          0.402712 |
| self_correction_marker_ratio |     -0.376912 |          0.376912 |
| confidence_marker_count      |     -0.355809 |          0.355809 |
| tool_gini_coefficient        |      0.332621 |          0.332621 |
| success_rate                 |      0.325402 |          0.325402 |
| validation_to_total_ratio    |     -0.303339 |          0.303339 |
| total_calls                  |     -0.278539 |          0.278539 |
| successful_calls             |     -0.27623  |          0.27623  |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9545 |
| Precision | 0.9000 |
| Recall | 1.0000 |
| F1 Score | 0.9474 |
| ROC-AUC | 0.9744 |

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
| CV ROC-AUC (mean) | 0.9916 |
| CV ROC-AUC (std) | 0.0112 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.9231 |
| PPV | 0.9000 |
| NPV | 1.0000 |
| FPR | 0.0769 |
| FNR | 0.0000 |
| F1_Score | 0.9474 |
| MCC | {0.0: 0.9114654303752999, 1.0: 0.9114654303752999} |
| Kappa | 0.9076 |

### Top 15 Features (by importance)

| feature                      |   importance |
|:-----------------------------|-------------:|
| positive_marker_count        |    0.103147  |
| total_markers                |    0.0959191 |
| model_gpt-4o                 |    0.0502241 |
| qa_score                     |    0.0501782 |
| unique_tool_calls_ratio      |    0.0462222 |
| recovery_marker_ratio        |    0.0459004 |
| reasoning_statement_count    |    0.0428987 |
| marker_balance               |    0.0422498 |
| self_correction_marker_ratio |    0.039547  |
| step_count                   |    0.0332834 |
| total_calls                  |    0.0312521 |
| tool_repetition_rate         |    0.0268364 |
| most_used_tool_count         |    0.0260727 |
| max_consecutive_successes    |    0.025128  |
| prompt_tokens                |    0.0223398 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9545 |
| Precision | 0.9000 |
| Recall | 1.0000 |
| F1 Score | 0.9474 |
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
| CV ROC-AUC (mean) | 0.9866 |
| CV ROC-AUC (std) | 0.0118 |

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
| positive_marker_count        |    0.172011  |
| total_markers                |    0.153805  |
| total_calls                  |    0.0977948 |
| model_gpt-4o                 |    0.0793646 |
| qa_score                     |    0.0760848 |
| self_correction_marker_count |    0.0578285 |
| uncertainty_marker_ratio     |    0.0515263 |
| unique_tool_calls_ratio      |    0.0382366 |
| top3_tool_concentration      |    0.0367958 |
| recovery_marker_ratio        |    0.0347214 |
| assistant_steps_count        |    0.0191274 |
| tool_latency_std             |    0.0183078 |
| validation_attempt_count     |    0.0164565 |
| neutral_marker_count         |    0.0161741 |
| tool_diversity_entropy       |    0.0153475 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
