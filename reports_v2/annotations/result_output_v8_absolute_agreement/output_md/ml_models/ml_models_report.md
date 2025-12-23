# ML Model Comparison Report: MD
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9688 |                1 |           0.8 |    0.8889 |         0.9926 | 0.9182 ± 0.0714 |
| Random Forest       |          0.9375 |                1 |           0.6 |    0.75   |         1      | 0.9298 ± 0.0703 |
| XGBoost             |          0.9688 |                1 |           0.8 |    0.8889 |         1      | 0.8646 ± 0.1065 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9688 |
| Precision | 1.0000 |
| Recall | 0.8000 |
| F1 Score | 0.8889 |
| ROC-AUC | 0.9926 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9844 |
| Precision | 0.9500 |
| Recall | 0.9500 |
| F1 Score | 0.9500 |
| ROC-AUC | 0.9972 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9182 |
| CV ROC-AUC (std) | 0.0714 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8000 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9643 |
| FPR | 0.0000 |
| FNR | 0.2000 |
| F1_Score | 0.8889 |
| MCC | {0.0: 0.8783100656536799, 1.0: 0.8783100656536799} |
| Kappa | 0.8710 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| tool_latency_mean            |     -0.828506 |          0.828506 |
| has_planning_in_first_step   |      0.813081 |          0.813081 |
| level                        |     -0.785398 |          0.785398 |
| hallucination_count          |      0.709116 |          0.709116 |
| tool_latency_std             |     -0.671298 |          0.671298 |
| tools_used_count             |      0.637751 |          0.637751 |
| tool_gini_coefficient        |      0.533803 |          0.533803 |
| tool_latency_max             |     -0.52033  |          0.52033  |
| tool_execution_duration      |     -0.515969 |          0.515969 |
| self_correction_marker_count |     -0.496298 |          0.496298 |
| reasoning_statement_count    |     -0.473686 |          0.473686 |
| total_tokens                 |     -0.443527 |          0.443527 |
| prompt_tokens                |     -0.441755 |          0.441755 |
| model_gpt-4o                 |     -0.395485 |          0.395485 |
| self_correction_marker_ratio |     -0.393683 |          0.393683 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9375 |
| Precision | 1.0000 |
| Recall | 0.6000 |
| F1 Score | 0.7500 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9922 |
| Precision | 1.0000 |
| Recall | 0.9500 |
| F1 Score | 0.9744 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9298 |
| CV ROC-AUC (std) | 0.0703 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.6000 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9310 |
| FPR | 0.0000 |
| FNR | 0.4000 |
| F1_Score | 0.7500 |
| MCC | {0.0: 0.7474093186836597, 1.0: 0.7474093186836597} |
| Kappa | 0.7168 |

### Top 15 Features (by importance)

| feature                    |   importance |
|:---------------------------|-------------:|
| tool_switching_rate        |    0.0649256 |
| retry_rate                 |    0.05164   |
| tool_latency_std           |    0.0493905 |
| tool_execution_duration    |    0.0447809 |
| tool_latency_mean          |    0.0407908 |
| tool_latency_max           |    0.0393612 |
| total_markers              |    0.0270303 |
| unique_tool_calls_ratio    |    0.0267485 |
| top3_tool_concentration    |    0.0253335 |
| max_consecutive_successes  |    0.0224723 |
| most_used_tool_ratio       |    0.0218766 |
| tool_diversity_entropy     |    0.0216661 |
| unnecessary_tool_use_count |    0.0201397 |
| message_length_std         |    0.018299  |
| completion_tokens          |    0.0182229 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9688 |
| Precision | 1.0000 |
| Recall | 0.8000 |
| F1 Score | 0.8889 |
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
| CV ROC-AUC (mean) | 0.8646 |
| CV ROC-AUC (std) | 0.1065 |

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

| feature                    |   importance |
|:---------------------------|-------------:|
| most_used_tool_ratio       |    0.113417  |
| tool_switching_rate        |    0.0683406 |
| retry_rate                 |    0.0629008 |
| success_rate               |    0.0471432 |
| tool_execution_duration    |    0.0416879 |
| negative_marker_count      |    0.0408023 |
| total_markers              |    0.0407952 |
| steps_to_first_error       |    0.0347353 |
| model_gpt-4o               |    0.0331251 |
| tool_latency_mean          |    0.0324588 |
| total_message_length       |    0.0317809 |
| marker_balance             |    0.0311885 |
| top3_tool_concentration    |    0.0278907 |
| tool_latency_max           |    0.0249421 |
| has_planning_in_first_step |    0.0246638 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
