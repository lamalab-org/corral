# Comprehensive ML Models Analysis Report
================================================================================

## Executive Summary

- **Best Performance**: LogReg on afm (Test ROC-AUC: 1.0000)
- **Worst Performance**: LogReg on ml (Test ROC-AUC: 0.8704)
- **Environments Analyzed**: 7

## Model Rankings by Average Performance

| Model   |   Avg ROC-AUC |       Std |      Min |   Max |
|:--------|--------------:|----------:|---------:|------:|
| XGB     |      0.981052 | 0.0173821 | 0.957602 |     1 |
| RF      |      0.97768  | 0.0162282 | 0.953704 |     1 |
| LogReg  |      0.96097  | 0.0419393 | 0.87037  |     1 |

## Environment Rankings by Predictability

| Environment    |   Avg ROC-AUC |   Best Model AUC |   Worst Model AUC |   Dataset Size |
|:---------------|--------------:|-----------------:|------------------:|---------------:|
| catalyst       |      1        |         1        |          1        |            165 |
| md             |      0.994624 |         0.997519 |          0.988834 |            372 |
| afm            |      0.994048 |         1        |          0.982143 |            166 |
| all            |      0.97069  |         0.98376  |          0.948498 |           1343 |
| resistor       |      0.967635 |         0.970054 |          0.965517 |            240 |
| retrosynthesis |      0.956628 |         0.960526 |          0.951754 |            280 |
| ml             |      0.929012 |         0.962963 |          0.87037  |            120 |

## Key Insights

### 1. Model Performance
- Easiest to predict: **catalyst** (Avg ROC-AUC: 1.0000)
- Hardest to predict: **ml** (Avg ROC-AUC: 0.9290)

### 2. Overfitting Analysis
Check Plot 2 for train-test gaps. Large gaps indicate overfitting.

### 3. Feature Importance
See Plot 4 and Plot 5 for universal features across models/environments.

### 4. Model Stability
See Plot 7 for cross-validation stability. Lower CV std = more stable model.
