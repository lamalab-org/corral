# ML Model Comparison Report: AFM
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          1      |                1 |           1   |    1      |         1      | 0.9584 ± 0.0233 |
| Random Forest       |          0.9118 |                1 |           0.5 |    0.6667 |         0.9821 | 0.9927 ± 0.0089 |
| XGBoost             |          0.9118 |                1 |           0.5 |    0.6667 |         1      | 0.9982 ± 0.0036 |

---

## 1. Logistic Regression

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
| Accuracy | 0.9848 |
| Precision | 0.9583 |
| Recall | 0.9583 |
| F1 Score | 0.9583 |
| ROC-AUC | 0.9996 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9584 |
| CV ROC-AUC (std) | 0.0233 |

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

### Top 15 Features (by coefficient magnitude)

| feature                            |   coefficient |   abs_coefficient |
|:-----------------------------------|--------------:|------------------:|
| planning_to_total_ratio            |     -1.10763  |          1.10763  |
| level                              |     -1.02962  |          1.02962  |
| neutral_marker_ratio               |      0.877424 |          0.877424 |
| assistant_steps_to_first_tool_call |      0.702854 |          0.702854 |
| backtrack_trigger_count            |      0.698894 |          0.698894 |
| reasoning_to_total_ratio           |     -0.695343 |          0.695343 |
| recovery_marker_count              |     -0.680892 |          0.680892 |
| unique_tool_calls_ratio            |     -0.536526 |          0.536526 |
| planning_statement_count           |     -0.484914 |          0.484914 |
| tools_used_count                   |     -0.465175 |          0.465175 |
| backtrack_after_error_rate         |      0.458384 |          0.458384 |
| problem_marker_count               |     -0.438672 |          0.438672 |
| code_block_count                   |     -0.412397 |          0.412397 |
| validation_attempt_count           |     -0.40931  |          0.40931  |
| completion_tokens                  |     -0.401276 |          0.401276 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9118 |
| Precision | 1.0000 |
| Recall | 0.5000 |
| F1 Score | 0.6667 |
| ROC-AUC | 0.9821 |

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
| CV ROC-AUC (mean) | 0.9927 |
| CV ROC-AUC (std) | 0.0089 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.5000 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9032 |
| FPR | 0.0000 |
| FNR | 0.5000 |
| F1_Score | 0.6667 |
| MCC | {0.0: 0.672021505032247, 1.0: 0.672021505032247} |
| Kappa | 0.6222 |

### Top 15 Features (by importance)

| feature                           |   importance |
|:----------------------------------|-------------:|
| planning_to_total_ratio           |    0.0679826 |
| level                             |    0.0602854 |
| neutral_marker_ratio              |    0.0506774 |
| avg_words_per_message             |    0.0503421 |
| tool_latency_std                  |    0.0424529 |
| avg_assistant_message_length      |    0.0408858 |
| avg_message_length                |    0.0397544 |
| tool_latency_max                  |    0.0366978 |
| total_message_length              |    0.0295259 |
| total_tokens                      |    0.0290941 |
| avg_tools_per_step                |    0.0252314 |
| tool_execution_duration           |    0.0249336 |
| retry_rate                        |    0.0211631 |
| prompt_tokens                     |    0.0195432 |
| assistant_steps_to_first_planning |    0.0189784 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9118 |
| Precision | 1.0000 |
| Recall | 0.5000 |
| F1 Score | 0.6667 |
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
| CV ROC-AUC (mean) | 0.9982 |
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

| feature                   |   importance |
|:--------------------------|-------------:|
| marker_balance            |    0.128127  |
| level                     |    0.110743  |
| assistant_steps_count     |    0.103209  |
| planning_to_total_ratio   |    0.0645192 |
| retry_rate                |    0.0537458 |
| unique_tool_calls_ratio   |    0.0411709 |
| most_used_tool_count      |    0.0379628 |
| tool_execution_duration   |    0.0333162 |
| neutral_marker_ratio      |    0.0283412 |
| completion_tokens         |    0.0280425 |
| total_tokens              |    0.0256938 |
| max_consecutive_successes |    0.025291  |
| uncertainty_marker_ratio  |    0.0241978 |
| marker_balance_ratio      |    0.022546  |
| avg_tools_per_step        |    0.0206038 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).