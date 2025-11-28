# ML Model Comparison Report: RESISTOR
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9583 |           0.9048 |             1 |      0.95 |         0.9673 | 0.9972 ± 0.0056 |
| Random Forest       |          0.9583 |           0.9048 |             1 |      0.95 |         0.9701 | 1.0000 ± 0.0000 |
| XGBoost             |          0.9583 |           0.9048 |             1 |      0.95 |         0.9655 | 0.9967 ± 0.0067 |

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
| ROC-AUC | 0.9673 |

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
| CV ROC-AUC (std) | 0.0056 |

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

| feature                   |   coefficient |   abs_coefficient |
|:--------------------------|--------------:|------------------:|
| qa_score                  |      1.71618  |          1.71618  |
| model_gpt-4o              |     -0.849597 |          0.849597 |
| planning_to_total_ratio   |     -0.615588 |          0.615588 |
| message_length_std        |     -0.584168 |          0.584168 |
| avg_thought_length        |      0.574208 |          0.574208 |
| give_up_count             |     -0.54619  |          0.54619  |
| success_rate              |      0.542659 |          0.542659 |
| annotatable_steps_count   |     -0.53845  |          0.53845  |
| neutral_marker_count      |     -0.450133 |          0.450133 |
| assistant_steps_count     |     -0.446634 |          0.446634 |
| successful_calls          |     -0.436542 |          0.436542 |
| total_calls               |     -0.436385 |          0.436385 |
| max_consecutive_successes |     -0.428726 |          0.428726 |
| neutral_count             |     -0.418438 |          0.418438 |
| negative_marker_ratio     |     -0.40277  |          0.40277  |

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
| ROC-AUC | 0.9701 |

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
| total_markers            |    0.0958355 |
| model_gpt-4o             |    0.0945372 |
| qa_score                 |    0.0777682 |
| longest_looping_sequence |    0.0498777 |
| assistant_steps_count    |    0.0441588 |
| planning_to_total_ratio  |    0.0402652 |
| step_count               |    0.0398582 |
| total_message_length     |    0.0373819 |
| successful_calls         |    0.0342214 |
| avg_thought_length       |    0.0275097 |
| unique_tool_calls_ratio  |    0.0258007 |
| marker_balance           |    0.0238612 |
| uncertainty_marker_ratio |    0.0198915 |
| most_used_tool_count     |    0.0198835 |
| tool_repetition_rate     |    0.0194201 |

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
| ROC-AUC | 0.9655 |

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
| CV ROC-AUC (mean) | 0.9967 |
| CV ROC-AUC (std) | 0.0067 |

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
| longest_looping_sequence |    0.295595  |
| qa_score                 |    0.113845  |
| total_calls              |    0.0808373 |
| model_gpt-4o             |    0.0734442 |
| total_markers            |    0.0728918 |
| unique_tool_calls_ratio  |    0.0474079 |
| avg_thought_length       |    0.033013  |
| successful_calls         |    0.0309761 |
| avg_tools_per_step       |    0.029853  |
| tool_latency_max         |    0.0265686 |
| tool_latency_mean        |    0.021765  |
| assistant_steps_count    |    0.020412  |
| most_used_tool_ratio     |    0.0199125 |
| planning_to_total_ratio  |    0.0183882 |
| uncertainty_marker_ratio |    0.017814  |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).