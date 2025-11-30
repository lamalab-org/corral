# Comprehensive ML Models Analysis Report
================================================================================

## Executive Summary

- **Best Performance**: XGB on md (Test ROC-AUC: 1.0000)
- **Worst Performance**: LogReg on all (Test ROC-AUC: 0.9835)
- **Environments Analyzed**: 7

## Model Rankings by Average Performance

| Model   |   Avg ROC-AUC |         Std |      Min |   Max |
|:--------|--------------:|------------:|---------:|------:|
| XGB     |      0.999751 | 0.000542595 | 0.99843  |     1 |
| RF      |      0.999694 | 0.000552628 | 0.99843  |     1 |
| LogReg  |      0.993672 | 0.00566095  | 0.983543 |     1 |

## Environment Rankings by Predictability

| Environment    |   Avg ROC-AUC |   Best Model AUC |   Worst Model AUC |   Dataset Size |
|:---------------|--------------:|-----------------:|------------------:|---------------:|
| catalyst       |      1        |          1       |          1        |            465 |
| resistor       |      1        |          1       |          1        |            480 |
| md             |      0.998926 |          1       |          0.996777 |            900 |
| afm            |      0.998467 |          1       |          0.995402 |            362 |
| ml             |      0.996337 |          1       |          0.989011 |            240 |
| retrosynthesis |      0.995945 |          0.99843 |          0.990973 |            616 |
| all            |      0.994266 |          0.99983 |          0.983543 |           3063 |

## Key Insights

### 1. Model Performance
- Easiest to predict: **catalyst** (Avg ROC-AUC: 1.0000)
- Hardest to predict: **all** (Avg ROC-AUC: 0.9943)

### 2. Overfitting Analysis
Check Plot 2 for train-test gaps. Large gaps indicate overfitting.

### 3. Feature Importance
See Plot 4 and Plot 5 for universal features across models/environments.

### 4. Model Stability
See Plot 7 for cross-validation stability. Lower CV std = more stable model.
