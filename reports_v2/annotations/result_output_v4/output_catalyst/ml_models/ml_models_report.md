# ML Model Comparison Report: CATALYST
================================================================================

## Model Performance Comparison (Test Set)

| Model               |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   Test ROC-AUC | CV Mean ± Std   |
|:--------------------|----------------:|-----------------:|--------------:|----------:|---------------:|:----------------|
| Logistic Regression |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |
| Random Forest       |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |
| XGBoost             |               1 |                1 |             1 |         1 |              1 | 1.0000 ± 0.0000 |

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

### Top 15 Features (by coefficient magnitude)

| feature                            |   coefficient |   abs_coefficient |
|:-----------------------------------|--------------:|------------------:|
| steps_to_first_reasoning           |      0.744601 |          0.744601 |
| assistant_steps_to_first_reasoning |      0.744601 |          0.744601 |
| reasoning_to_total_ratio           |      0.655583 |          0.655583 |
| positive_marker_ratio              |      0.499935 |          0.499935 |
| backtrack_after_error_rate         |     -0.435011 |          0.435011 |
| marker_balance_ratio               |      0.340588 |          0.340588 |
| positive_marker_count              |      0.325665 |          0.325665 |
| tool_latency_max                   |      0.268111 |          0.268111 |
| neutral_marker_ratio               |     -0.255855 |          0.255855 |
| message_length_std                 |      0.239527 |          0.239527 |
| tool_execution_duration            |      0.237999 |          0.237999 |
| prompt_tokens                      |      0.237243 |          0.237243 |
| total_tokens                       |      0.233351 |          0.233351 |
| max_consecutive_successes          |      0.223421 |          0.223421 |
| annotatable_steps_count            |     -0.220446 |          0.220446 |

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
| positive_marker_ratio        |    0.0831645 |
| avg_words_per_message        |    0.0752686 |
| total_markers                |    0.056142  |
| avg_assistant_message_length |    0.0471902 |
| total_message_length         |    0.0467696 |
| tools_used_count             |    0.0467096 |
| thought_count                |    0.0409335 |
| annotatable_steps_count      |    0.0391189 |
| syntax_error_count           |    0.0381873 |
| marker_balance               |    0.0324154 |
| avg_message_length           |    0.0311841 |
| most_used_tool_ratio         |    0.0307568 |
| step_count                   |    0.029982  |
| loop_instance_count          |    0.0289147 |
| negative_marker_count        |    0.0288687 |

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

| feature                            |   importance |
|:-----------------------------------|-------------:|
| marker_balance_ratio               |   0.158696   |
| negative_marker_ratio              |   0.13082    |
| marker_balance                     |   0.128983   |
| assistant_steps_to_first_reasoning |   0.107212   |
| steps_to_first_reasoning           |   0.10487    |
| positive_marker_ratio              |   0.0913882  |
| tool_diversity_entropy             |   0.0830812  |
| annotatable_steps_count            |   0.0422982  |
| message_length_std                 |   0.0277092  |
| problem_marker_ratio               |   0.0255838  |
| avg_tools_per_step                 |   0.0254812  |
| positive_marker_count              |   0.0167055  |
| tool_latency_max                   |   0.01511    |
| total_tokens                       |   0.0145251  |
| negative_marker_count              |   0.00537014 |

================================================================================

## Summary

- **Logistic Regression**: Linear model, interpretable coefficients
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, often highest performance

**Note**: Test set metrics are the primary performance indicators.
Training metrics help identify overfitting (large train-test gap).
