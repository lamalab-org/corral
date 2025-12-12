# ML Model Comparison Report: RESISTOR
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9896 |           0.9744 |             1 |     0.987 |              1 | 0.9989 ± 0.0017 |
| Random Forest       |          1      |           1      |             1 |     1     |              1 | 1.0000 ± 0.0000 |
| XGBoost             |          1      |           1      |             1 |     1     |              1 | 1.0000 ± 0.0000 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9896 |
| Precision | 0.9744 |
| Recall | 1.0000 |
| F1 Score | 0.9870 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9922 |
| Precision | 0.9804 |
| Recall | 1.0000 |
| F1 Score | 0.9901 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9989 |
| CV ROC-AUC (std) | 0.0017 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 1.0000 |
| TNR | 0.9828 |
| PPV | 0.9744 |
| NPV | 1.0000 |
| FPR | 0.0172 |
| FNR | 0.0000 |
| F1_Score | 0.9870 |
| MCC | {0.0: 0.9785497849867489, 1.0: 0.9785497849867489} |
| Kappa | 0.9783 |

### Top 15 Features (by coefficient magnitude)

| feature                   |   coefficient |   abs_coefficient |
|:--------------------------|--------------:|------------------:|
| qa_score                  |      1.80098  |          1.80098  |
| confidence_marker_ratio   |     -1.32777  |          1.32777  |
| model_gpt-4o              |     -0.895727 |          0.895727 |
| validation_to_total_ratio |     -0.868856 |          0.868856 |
| recovery_marker_ratio     |     -0.738459 |          0.738459 |
| give_up_count             |     -0.583491 |          0.583491 |
| planning_to_total_ratio   |     -0.56158  |          0.56158  |
| marker_balance_ratio      |      0.554157 |          0.554157 |
| completion_tokens         |     -0.536042 |          0.536042 |
| avg_thought_length        |      0.519947 |          0.519947 |
| positive_marker_ratio     |      0.516921 |          0.516921 |
| success_rate              |      0.475798 |          0.475798 |
| successful_calls          |     -0.46423  |          0.46423  |
| total_calls               |     -0.462871 |          0.462871 |
| max_consecutive_successes |     -0.458147 |          0.458147 |

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

| feature                      |   importance |
|:-----------------------------|-------------:|
| confidence_marker_ratio      |    0.100505  |
| confidence_marker_count      |    0.0717406 |
| model_gpt-4o                 |    0.0685336 |
| longest_looping_sequence     |    0.061463  |
| qa_score                     |    0.0564711 |
| total_markers                |    0.0519368 |
| recovery_marker_count        |    0.0379815 |
| step_count                   |    0.0359159 |
| total_message_length         |    0.0333933 |
| assistant_steps_count        |    0.0326441 |
| marker_balance               |    0.0224446 |
| unique_tool_calls_ratio      |    0.0217257 |
| total_calls                  |    0.0215168 |
| self_correction_marker_count |    0.017861  |
| max_consecutive_successes    |    0.0178311 |

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

| feature                   |   importance |
|:--------------------------|-------------:|
| confidence_marker_count   |    0.166447  |
| model_gpt-4o              |    0.141284  |
| qa_score                  |    0.11752   |
| confidence_marker_ratio   |    0.0670506 |
| unique_tool_calls_ratio   |    0.0630231 |
| total_message_length      |    0.0585327 |
| total_calls               |    0.0514504 |
| neutral_marker_count      |    0.0335837 |
| top3_tool_concentration   |    0.0286955 |
| successful_calls          |    0.0282383 |
| tool_execution_duration   |    0.0268326 |
| tool_latency_mean         |    0.0255026 |
| question_count            |    0.0233548 |
| avg_message_length        |    0.0147396 |
| validation_to_total_ratio |    0.0146082 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
