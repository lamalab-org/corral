# ML Model Comparison Report: AFM
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |          0.8571 |                1 |        0.3333 |       0.5 |         0.9091 | 0.7778 ± 0.2722 |
| Random Forest       |          0.7857 |                0 |        0      |       0   |         1      | 0.8222 ± 0.2288 |
| XGBoost             |          0.8571 |                1 |        0.3333 |       0.5 |         1      | 0.8233 ± 0.2321 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8571 |
| Precision | 1.0000 |
| Recall | 0.3333 |
| F1 Score | 0.5000 |
| ROC-AUC | 0.9091 |

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
| CV ROC-AUC (mean) | 0.7778 |
| CV ROC-AUC (std) | 0.2722 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.3333 |
| TNR | 1.0000 |
| PPV | 1.0000 |
| NPV | 0.8462 |
| FPR | 0.0000 |
| FNR | 0.6667 |
| F1_Score | 0.5000 |
| MCC | {0.0: 0.5310850045437943, 1.0: 0.5310850045437943} |
| Kappa | 0.4400 |

### Top 15 Features (by coefficient magnitude)

| feature                        |   coefficient |   abs_coefficient |
|:-------------------------------|--------------:|------------------:|
| level                          |     -0.865632 |          0.865632 |
| verification_marker_count      |     -0.595934 |          0.595934 |
| reasoning_statement_count      |     -0.4865   |          0.4865   |
| code_block_count               |     -0.472868 |          0.472868 |
| thought_to_action_ratio        |     -0.462857 |          0.462857 |
| message_length_std             |      0.440771 |          0.440771 |
| reasoning_to_total_ratio       |     -0.417744 |          0.417744 |
| error_burstiness               |      0.396119 |          0.396119 |
| max_consecutive_failures       |      0.396119 |          0.396119 |
| tool_latency_mean              |     -0.376161 |          0.376161 |
| total_markers                  |     -0.37499  |          0.37499  |
| avg_thought_length             |     -0.373682 |          0.373682 |
| planning_statement_count       |     -0.369548 |          0.369548 |
| assistant_steps_to_first_error |      0.365619 |          0.365619 |
| tool_latency_std               |     -0.342556 |          0.342556 |

---

## 2. Random Forest

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7857 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 Score | 0.0000 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9821 |
| Precision | 1.0000 |
| Recall | 0.9000 |
| F1 Score | 0.9474 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8222 |
| CV ROC-AUC (std) | 0.2288 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.0000 |
| TNR | 1.0000 |
| PPV | None |
| NPV | 0.7857 |
| FPR | 0.0000 |
| FNR | 1.0000 |
| F1_Score | 0.0000 |
| MCC | {0.0: 'None', 1.0: 'None'} |
| Kappa | 0.0000 |

### Top 15 Features (by importance)

| feature                            |   importance |
|:-----------------------------------|-------------:|
| message_length_std                 |    0.0571422 |
| level                              |    0.0557467 |
| avg_tools_per_step                 |    0.0460049 |
| planning_to_total_ratio            |    0.0419498 |
| annotatable_steps_count            |    0.0383059 |
| tool_switching_rate                |    0.035456  |
| avg_words_per_message              |    0.0349438 |
| total_message_length               |    0.0317509 |
| assistant_steps_to_first_tool_call |    0.0292663 |
| tool_latency_std                   |    0.0280786 |
| tool_gini_coefficient              |    0.0251118 |
| most_used_tool_ratio               |    0.0245489 |
| total_markers                      |    0.0216877 |
| tool_diversity_entropy             |    0.0206412 |
| neutral_marker_ratio               |    0.0197534 |

---

## 3. XGBoost

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8571 |
| Precision | 1.0000 |
| Recall | 0.3333 |
| F1 Score | 0.5000 |
| ROC-AUC | 1.0000 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9821 |
| Precision | 1.0000 |
| Recall | 0.9000 |
| F1 Score | 0.9474 |
| ROC-AUC | 1.0000 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.8233 |
| CV ROC-AUC (std) | 0.2321 |

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
| prompt_tokens                |    0.0774457 |
| planning_to_total_ratio      |    0.0759793 |
| level                        |    0.0733185 |
| avg_tools_per_step           |    0.0716335 |
| tool_execution_duration      |    0.0545023 |
| avg_message_length           |    0.0407441 |
| tool_latency_std             |    0.0394019 |
| retry_rate                   |    0.0363008 |
| steps_to_first_tool_call     |    0.0345459 |
| tool_latency_mean            |    0.0343843 |
| completion_tokens            |    0.0343737 |
| max_consecutive_successes    |    0.0326668 |
| thought_count                |    0.0278661 |
| avg_assistant_message_length |    0.0270068 |
| tool_gini_coefficient        |    0.0204009 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).