# Comprehensive ML Models Analysis Report
================================================================================

## Executive Summary

- **Best Performance**: LogReg on afm (Test ROC-AUC: 1.0000)
- **Worst Performance**: LogReg on ml (Test ROC-AUC: 0.8704)
- **Environments Analyzed**: 7

## Model Rankings by Average Performance

| Model   |   Avg ROC-AUC |       Std |      Min |   Max |
|:--------|--------------:|----------:|---------:|------:|
| XGB     |      0.983229 | 0.0167684 | 0.954678 |     1 |
| RF      |      0.98128  | 0.0129613 | 0.962963 |     1 |
| LogReg  |      0.957994 | 0.0421801 | 0.87037  |     1 |

## Environment Rankings by Predictability

| Environment    |   Avg ROC-AUC |   Best Model AUC |   Worst Model AUC |   Dataset Size |
|:---------------|--------------:|-----------------:|------------------:|---------------:|
| catalyst       |      1        |         1        |          1        |            165 |
| afm            |      0.994048 |         1        |          0.982143 |            166 |
| md             |      0.993797 |         0.997519 |          0.988834 |            372 |
| resistor       |      0.974592 |         0.989111 |          0.954628 |            240 |
| all            |      0.968984 |         0.980812 |          0.946218 |           1343 |
| retrosynthesis |      0.955653 |         0.966374 |          0.945906 |            280 |
| ml             |      0.932099 |         0.962963 |          0.87037  |            120 |

## Key Insights

### 1. Model Performance
- Easiest to predict: **catalyst** (Avg ROC-AUC: 1.0000)
- Hardest to predict: **ml** (Avg ROC-AUC: 0.9321)

### 2. Overfitting Analysis
Check Plot 2 for train-test gaps. Large gaps indicate overfitting.

### 3. Feature Importance
See Plot 4 and Plot 5 for universal features across models/environments.

### 4. Model Stability
See Plot 7 for cross-validation stability. Lower CV std = more stable model.
