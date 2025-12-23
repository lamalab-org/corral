# Comprehensive ML Models Analysis Report
================================================================================

## Executive Summary

- **Best Performance**: RF on md (Test ROC-AUC: 1.0000)
- **Worst Performance**: LogReg on ml (Test ROC-AUC: 0.2593)
- **Environments Analyzed**: 7

## Model Rankings by Average Performance

| Model   |   Avg ROC-AUC |       Std |      Min |   Max |
|:--------|--------------:|----------:|---------:|------:|
| RF      |      0.953168 | 0.0881254 | 0.740741 |     1 |
| LogReg  |      0.874376 | 0.25303   | 0.259259 |     1 |
| XGB     |      0.833239 | 0.254443  | 0.37037  |     1 |

## Environment Rankings by Predictability

| Environment    |   Avg ROC-AUC |   Best Model AUC |   Worst Model AUC |   Dataset Size |
|:---------------|--------------:|-----------------:|------------------:|---------------:|
| retrosynthesis |      1        |         1        |          1        |             60 |
| md             |      0.997531 |         1        |          0.992593 |            160 |
| resistor       |      0.991453 |         1        |          0.974359 |            108 |
| afm            |      0.969697 |         1        |          0.909091 |             70 |
| all            |      0.959686 |         0.962299 |          0.957074 |            518 |
| catalyst       |      0.833333 |         1        |          0.5      |             60 |
| ml             |      0.45679  |         0.740741 |          0.259259 |             60 |

## Key Insights

### 1. Model Performance
- Easiest to predict: **retrosynthesis** (Avg ROC-AUC: 1.0000)
- Hardest to predict: **ml** (Avg ROC-AUC: 0.4568)

### 2. Overfitting Analysis
Check Plot 2 for train-test gaps. Large gaps indicate overfitting.

### 3. Feature Importance
See Plot 4 and Plot 5 for universal features across models/environments.

### 4. Model Stability
See Plot 7 for cross-validation stability. Lower CV std = more stable model.
