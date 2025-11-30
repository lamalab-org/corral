# ML Model Comparison Report: AFM
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9726 |                1 |        0.8667 |    0.9286 |         0.9954 | 0.9971 ± 0.0049 |
| Random Forest       |          1      |                1 |        1      |    1      |         1      | 1.0000 ± 0.0000 |
| XGBoost             |          1      |                1 |        1      |    1      |         1      | 1.0000 ± 0.0000 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9726 |
| Precision | 1.0000 |
| Recall | 0.8667 |
| F1 Score | 0.9286 |
| ROC-AUC | 0.9954 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9931 |
| Precision | 1.0000 |
| Recall | 0.9649 |
| F1 Score | 0.9821 |
| ROC-AUC | 0.9997 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9971 |
| CV ROC-AUC (std) | 0.0049 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.8667 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9667 |
| FPR | 0.0000 |
| FNR | 0.1333 |
| F1_Score | 0.9286 |
| MCC | {0.0: 0.9153020145163988, 1.0: 0.9153020145163988} |
| Kappa | 0.9117 |

### Top 15 Features (by coefficient magnitude)

| feature                            |   coefficient |   abs_coefficient |
|:-----------------------------------|--------------:|------------------:|
| planning_to_total_ratio            |     -1.3182   |          1.3182   |
| reasoning_to_total_ratio           |     -1.13409  |          1.13409  |
| level                              |     -1.00998  |          1.00998  |
| assistant_steps_to_first_tool_call |      0.873105 |          0.873105 |
| backtrack_trigger_count            |      0.855885 |          0.855885 |
| recovery_marker_count              |     -0.825258 |          0.825258 |
| rare_tool_ratio                    |     -0.794581 |          0.794581 |
| neutral_marker_ratio               |      0.782613 |          0.782613 |
| code_block_count                   |     -0.716774 |          0.716774 |
| avg_thought_length                 |     -0.620994 |          0.620994 |
| validation_attempt_count           |     -0.595508 |          0.595508 |
| verification_marker_count          |     -0.589712 |          0.589712 |
| unique_tool_calls_ratio            |     -0.563886 |          0.563886 |
| give_up_count                      |     -0.557707 |          0.557707 |
| confidence_marker_ratio            |      0.540519 |          0.540519 |

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
| avg_words_per_message        |    0.085889  |
| planning_to_total_ratio      |    0.0583992 |
| level                        |    0.0574717 |
| neutral_marker_ratio         |    0.0531093 |
| avg_assistant_message_length |    0.0486607 |
| avg_message_length           |    0.0386425 |
| total_message_length         |    0.0275988 |
| tool_latency_max             |    0.0268681 |
| completion_tokens            |    0.0260963 |
| tool_latency_std             |    0.0259582 |
| tool_execution_duration      |    0.0197539 |
| tool_gini_coefficient        |    0.0193161 |
| has_planning_in_first_step   |    0.018884  |
| max_consecutive_successes    |    0.0178212 |
| total_tokens                 |    0.0161739 |

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
| validation_to_total_ratio |    0.207357  |
| max_consecutive_successes |    0.131451  |
| avg_words_per_message     |    0.0922015 |
| level                     |    0.075822  |
| planning_to_total_ratio   |    0.0583427 |
| tool_switching_rate       |    0.0383573 |
| error_burstiness          |    0.0354718 |
| error_rate                |    0.0270324 |
| failed_calls              |    0.0264136 |
| verification_marker_count |    0.0220657 |
| completion_tokens         |    0.0219425 |
| avg_thought_length        |    0.0182274 |
| marker_balance_ratio      |    0.0169847 |
| marker_balance            |    0.0152123 |
| tool_latency_max          |    0.0150931 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
