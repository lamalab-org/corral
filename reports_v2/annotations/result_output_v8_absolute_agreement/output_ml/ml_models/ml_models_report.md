# ML Model Comparison Report: ML
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.6667 |           0.7778 |        0.7778 |    0.7778 |         0.2593 | 0.8048 ± 0.1090 |
| Random Forest       |          0.75   |           0.75   |        1      |    0.8571 |         0.7407 | 0.7762 ± 0.2095 |
| XGBoost             |          0.75   |           0.75   |        1      |    0.8571 |         0.3704 | 0.7476 ± 0.1417 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.6667 |
| Precision | 0.7778 |
| Recall | 0.7778 |
| F1 Score | 0.7778 |
| ROC-AUC | 0.2593 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9167 |
| Precision | 0.9429 |
| Recall | 0.9429 |
| F1 Score | 0.9429 |
| ROC-AUC | 0.9714 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8048 |
| CV ROC-AUC (std) | 0.1090 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.7778 |
| TNR | 0.3333 |
| PPV | 0.7778 |
| NPV | 0.3333 |
| FPR | 0.6667 |
| FNR | 0.2222 |
| F1_Score | 0.7778 |
| MCC | {0.0: 0.1111111111111111, 1.0: 0.1111111111111111} |
| Kappa | 0.1111 |

### Top 15 Features (by coefficient magnitude)

| feature                           |   coefficient |   abs_coefficient |
|:----------------------------------|--------------:|------------------:|
| confidence_marker_count           |      0.592392 |          0.592392 |
| prompt_tokens                     |      0.566363 |          0.566363 |
| total_tokens                      |      0.55695  |          0.55695  |
| tool_latency_std                  |     -0.549621 |          0.549621 |
| tool_latency_max                  |     -0.52815  |          0.52815  |
| tool_latency_mean                 |     -0.502701 |          0.502701 |
| tool_execution_duration           |     -0.448334 |          0.448334 |
| thought_to_action_ratio           |     -0.430869 |          0.430869 |
| assistant_steps_to_first_planning |      0.405292 |          0.405292 |
| steps_to_first_planning           |      0.405292 |          0.405292 |
| rare_tool_ratio                   |     -0.359081 |          0.359081 |
| validation_attempt_count          |      0.26929  |          0.26929  |
| error_rate                        |     -0.256696 |          0.256696 |
| success_rate                      |      0.256696 |          0.256696 |
| validation_to_total_ratio         |      0.242241 |          0.242241 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7500 |
| Precision | 0.7500 |
| Recall | 1.0000 |
| F1 Score | 0.8571 |
| ROC-AUC | 0.7407 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9375 |
| Precision | 0.9211 |
| Recall | 1.0000 |
| F1 Score | 0.9589 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.7762 |
| CV ROC-AUC (std) | 0.2095 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.0000 |
| PPV | 0.7500 |
| NPV | None |
| FPR | 1.0000 |
| FNR | 0.0000 |
| F1_Score | 0.8571 |
| MCC | {0.0: 'None', 1.0: 'None'} |
| Kappa | 0.0000 |

### Top 15 Features (by importance)

| feature                 |   importance |
|:------------------------|-------------:|
| prompt_tokens           |    0.106925  |
| total_tokens            |    0.0634295 |
| tool_latency_max        |    0.0627799 |
| tool_diversity_entropy  |    0.0566414 |
| tool_execution_duration |    0.0501168 |
| avg_words_per_message   |    0.0473957 |
| tool_latency_std        |    0.0468431 |
| tool_latency_mean       |    0.0459673 |
| step_count              |    0.0304207 |
| successful_calls        |    0.026309  |
| message_length_std      |    0.0245798 |
| total_markers           |    0.0229831 |
| neutral_marker_ratio    |    0.022918  |
| thought_to_action_ratio |    0.0202253 |
| marker_balance          |    0.0182938 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7500 |
| Precision | 0.7500 |
| Recall | 1.0000 |
| F1 Score | 0.8571 |
| ROC-AUC | 0.3704 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9792 |
| Precision | 0.9722 |
| Recall | 1.0000 |
| F1 Score | 0.9859 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.7476 |
| CV ROC-AUC (std) | 0.1417 |

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
| MCC | {'0.0': 'None', '1': 'None', '1.0': 'None'} |
| Kappa | 0.0000 |

### Top 15 Features (by importance)

| feature                 |   importance |
|:------------------------|-------------:|
| neutral_marker_ratio    |    0.127349  |
| prompt_tokens           |    0.102427  |
| total_tokens            |    0.0678601 |
| marker_balance          |    0.064568  |
| positive_marker_count   |    0.0425531 |
| successful_calls        |    0.0407016 |
| tool_execution_duration |    0.0405894 |
| total_calls             |    0.0397505 |
| tool_latency_std        |    0.032936  |
| avg_words_per_message   |    0.0318373 |
| assistant_steps_count   |    0.0314223 |
| avg_thought_length      |    0.0312601 |
| tool_latency_mean       |    0.0310097 |
| tool_latency_max        |    0.0305911 |
| unique_tool_calls_ratio |    0.0286854 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).