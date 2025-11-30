# ML Model Comparison Report: RETROSYNTHESIS
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.9677 |           0.9796 |        0.9796 |    0.9796 |         0.991  | 0.9779 ± 0.0153 |
| Random Forest       |          0.9919 |           1      |        0.9898 |    0.9949 |         0.9984 | 0.9940 ± 0.0038 |
| XGBoost             |          0.9758 |           0.9798 |        0.9898 |    0.9848 |         0.9984 | 0.9920 ± 0.0066 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9677 |
| Precision | 0.9796 |
| Recall | 0.9796 |
| F1 Score | 0.9796 |
| ROC-AUC | 0.9910 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9919 |
| Precision | 0.9948 |
| Recall | 0.9948 |
| F1 Score | 0.9948 |
| ROC-AUC | 0.9998 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9779 |
| CV ROC-AUC (std) | 0.0153 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9796 |
| TNR | 0.9231 |
| PPV | 0.9796 |
| NPV | 0.9231 |
| FPR | 0.0769 |
| FNR | 0.0204 |
| F1_Score | 0.9796 |
| MCC | {0.0: 0.902668759811617, 1.0: 0.902668759811617} |
| Kappa | 0.9027 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| give_up_count                |     -1.19743  |          1.19743  |
| qa_score                     |      1.19735  |          1.19735  |
| assistant_steps_count        |     -1.12402  |          1.12402  |
| has_tool_error_in_first_step |      1.11804  |          1.11804  |
| code_block_count             |     -1.09739  |          1.09739  |
| annotatable_steps_count      |     -0.988431 |          0.988431 |
| level                        |     -0.987235 |          0.987235 |
| tool_switching_rate          |     -0.895573 |          0.895573 |
| early_final_answer_count     |     -0.849288 |          0.849288 |
| tool_diversity_entropy       |     -0.808679 |          0.808679 |
| avg_words_per_message        |     -0.79741  |          0.79741  |
| confidence_marker_ratio      |     -0.781692 |          0.781692 |
| neutral_marker_ratio         |      0.778364 |          0.778364 |
| max_consecutive_successes    |      0.762823 |          0.762823 |
| positive_marker_ratio        |      0.722624 |          0.722624 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9919 |
| Precision | 1.0000 |
| Recall | 0.9898 |
| F1 Score | 0.9949 |
| ROC-AUC | 0.9984 |

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
| CV ROC-AUC (mean) | 0.9940 |
| CV ROC-AUC (std) | 0.0038 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9898 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.9630 |
| FPR | 0.0000 |
| FNR | 0.0102 |
| F1_Score | 0.9949 |
| MCC | {0.0: 0.9762872580750338, 1.0: 0.9762872580750338} |
| Kappa | 0.9760 |

### Top 15 Features (by importance)

| feature                            |   importance |
|:-----------------------------------|-------------:|
| level                              |    0.0582046 |
| problem_marker_ratio               |    0.0549701 |
| assistant_steps_to_first_tool_call |    0.054589  |
| rare_tool_ratio                    |    0.0450551 |
| tool_gini_coefficient              |    0.044792  |
| tool_switching_rate                |    0.0442761 |
| total_tokens                       |    0.0413837 |
| problem_marker_count               |    0.029799  |
| completion_tokens                  |    0.0276942 |
| prompt_tokens                      |    0.0273141 |
| give_up_count                      |    0.0251651 |
| uncertainty_marker_count           |    0.0240672 |
| hallucination_count                |    0.0237834 |
| annotatable_steps_count            |    0.0219042 |
| confidence_marker_ratio            |    0.0181035 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.9758 |
| Precision | 0.9798 |
| Recall | 0.9898 |
| F1 Score | 0.9848 |
| ROC-AUC | 0.9984 |

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
| CV ROC-AUC (mean) | 0.9920 |
| CV ROC-AUC (std) | 0.0066 |

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
| level                              |    0.143788  |
| assistant_steps_to_first_tool_call |    0.131064  |
| rare_tool_ratio                    |    0.0967758 |
| has_tool_error_in_first_step       |    0.0419452 |
| backtrack_trigger_count            |    0.0407647 |
| action_count                       |    0.0365047 |
| completion_tokens                  |    0.031836  |
| give_up_count                      |    0.0310457 |
| negative_marker_ratio              |    0.0259853 |
| max_consecutive_successes          |    0.018318  |
| positive_marker_ratio              |    0.0156979 |
| successful_calls                   |    0.015667  |
| early_final_answer_count           |    0.0145941 |
| unique_tool_calls_ratio            |    0.0134942 |
| thought_to_action_ratio            |    0.0126392 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).