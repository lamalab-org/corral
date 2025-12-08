# ML Model Comparison Report: ML
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |           0.875 |           0.8718 |        0.9714 |    0.9189 |          0.989 | 0.9103 ± 0.0547 |
| Random Forest       |           1     |           1      |        1      |    1      |          1     | 0.9874 ± 0.0156 |
| XGBoost             |           1     |           1      |        1      |    1      |          1     | 0.9979 ± 0.0043 |

---

## 1. Logistic Regression

### Performance Metrics

#### Test Set (Primary Metrics)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8750 |
| Precision | 0.8718 |
| Recall | 0.9714 |
| F1 Score | 0.9189 |
| ROC-AUC | 0.9890 |

#### Training Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.9479 |
| Precision | 0.9517 |
| Recall | 0.9787 |
| F1 Score | 0.9650 |
| ROC-AUC | 0.9873 |

#### Cross-Validation
| Metric | Value |
|--------|-------|
| CV ROC-AUC (mean) | 0.9103 |
| CV ROC-AUC (std) | 0.0547 |

#### Additional PyCM Metrics (Test Set)
| Metric | Value |
|--------|-------|
| TPR | 0.9714 |
| TNR | 0.6154 |
| PPV | 0.8718 |
| NPV | 0.8889 |
| FPR | 0.3846 |
| FNR | 0.0286 |
| F1_Score | 0.9189 |
| MCC | {0.0: 0.668116203787842, 1.0: 0.668116203787842} |
| Kappa | 0.6496 |

### Top 15 Features (by coefficient magnitude)

| feature                      |   coefficient |   abs_coefficient |
|:-----------------------------|--------------:|------------------:|
| annotatable_steps_count      |     -1.02679  |          1.02679  |
| validation_to_total_ratio    |      1.02139  |          1.02139  |
| backtrack_after_error_rate   |     -0.88012  |          0.88012  |
| rare_tool_ratio              |     -0.864333 |          0.864333 |
| completion_tokens            |     -0.860557 |          0.860557 |
| planning_to_total_ratio      |      0.844534 |          0.844534 |
| self_correction_marker_ratio |     -0.795307 |          0.795307 |
| backtrack_to_total_ratio     |     -0.714429 |          0.714429 |
| prompt_tokens                |      0.685588 |          0.685588 |
| problem_marker_count         |      0.684883 |          0.684883 |
| tool_latency_std             |     -0.666986 |          0.666986 |
| total_tokens                 |      0.66154  |          0.66154  |
| thought_to_action_ratio      |     -0.647927 |          0.647927 |
| unique_tool_calls_ratio      |     -0.637102 |          0.637102 |
| tool_repetition_rate         |      0.637102 |          0.637102 |

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
| CV ROC-AUC (mean) | 0.9874 |
| CV ROC-AUC (std) | 0.0156 |

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
| avg_message_length           |    0.063738  |
| tool_latency_max             |    0.0616785 |
| avg_words_per_message        |    0.0594703 |
| tool_latency_std             |    0.0581416 |
| total_tokens                 |    0.0545921 |
| avg_assistant_message_length |    0.0525758 |
| prompt_tokens                |    0.0503476 |
| message_length_std           |    0.0485845 |
| tool_latency_mean            |    0.046251  |
| total_message_length         |    0.0445473 |
| tool_execution_duration      |    0.041308  |
| completion_tokens            |    0.0296618 |
| tool_diversity_entropy       |    0.0231953 |
| marker_balance_ratio         |    0.0190433 |
| tool_switching_rate          |    0.0187769 |

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
| CV ROC-AUC (mean) | 0.9979 |
| CV ROC-AUC (std) | 0.0043 |

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
| avg_message_length           |    0.159247  |
| self_correction_marker_count |    0.0885001 |
| tool_execution_duration      |    0.0536358 |
| completion_tokens            |    0.0491022 |
| tool_latency_max             |    0.0485007 |
| tool_diversity_entropy       |    0.0436827 |
| tool_latency_mean            |    0.0399165 |
| verification_marker_ratio    |    0.0341699 |
| avg_thought_length           |    0.0335176 |
| neutral_marker_ratio         |    0.0333749 |
| prompt_tokens                |    0.0284863 |
| marker_balance               |    0.0277903 |
| tool_latency_std             |    0.0246582 |
| tool_switching_rate          |    0.023801  |
| most_used_tool_ratio         |    0.0213357 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).