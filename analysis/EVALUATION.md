# IRT Model Evaluation Guide

Rigorous evaluation of predictive performance using cross-validation.

## Two Evaluation Strategies

### 1. K-Fold CV (Leave-One-Environment-Out)
**What it tests:** Can the model generalize to a **new scientific domain**?

- Holds out all trials from one environment (e.g., all MD trials)
- Refits IRT models + Agent model on remaining 6 environments
- Predicts the held-out environment
- Repeats for all 7 environments

**Example:** Train on {AFM, Catalyst, ML, Resistor, Retro, Spectra} → Predict MD performance

### 2. Group LOO-CV
**What it tests:** Can the model generalize to **new (model, environment, scaffold) combinations**?

- 42 groups: 3 models × 7 environments × 2 scaffolds
- Holds out all trials from one group (e.g., all claude-4.5 + MD + react trials)
- Refits on remaining 41 groups
- Predicts the held-out group

**Example:** Train on all groups except (Claude-4.5, MD, react) → Predict that specific combination

## Usage

### Prerequisites
```bash
cd analysis

# Make sure data is prepared
snakemake results/data/knowledge_qa.csv results/data/reasoning_qa.csv results/data/overall_trace.csv
```

### Run K-Fold CV (7 environments in parallel)
```bash
python evaluate_irt_model.py kfold --n_jobs=7
```

**Expected runtime:** ~6-7 hours (7 folds × ~50 min per fold, parallelized)

**Output:** `results/eval/kfold_results.json`

### Run Group LOO-CV (42 groups in parallel)
```bash
python evaluate_irt_model.py group_loo --n_jobs=42
```

**Expected runtime:** ~3-4 hours (42 folds × ~50 min per fold, parallelized across 42 cores)

**Note:** Adjust `n_jobs` based on available cores. Using more cores = faster completion.

### Run Both Evaluations
```bash
python evaluate_irt_model.py all --n_jobs=20
```

## Metrics Computed

For each fold/group:
- **Accuracy**: Classification accuracy (threshold=0.5)
- **Log Loss**: Negative log-likelihood of predictions
- **ROC-AUC**: Area under ROC curve (if both classes present)
- **Mean Pred Prob**: Average predicted success probability
- **Mean True Rate**: Actual success rate

## Results Files

### kfold_results.json
```json
{
  "results": [
    {
      "fold": "LOEO-md",
      "metrics": {
        "accuracy": 0.523,
        "log_loss": 0.682,
        "roc_auc": 0.571,
        ...
      },
      "n_train": 52000,
      "n_test": 8550,
      ...
    },
    ...
  ],
  "summary": {
    "accuracy": 0.515,  // Average across 7 folds
    "log_loss": 0.691,
    "n_folds": 7
  }
}
```

### group_loo_results.json
Similar structure with 42 groups.

## Interpreting Results

### Good Model Performance
- **Accuracy > 0.60**: Model predicts better than chance
- **Log Loss < 0.65**: Well-calibrated probabilities
- **ROC-AUC > 0.65**: Good discrimination

### Comparing to Baseline
Compare K-Fold results to:
- Random baseline: 0.472 accuracy (mean success rate)
- Environment-only model: Predicts based on environment mean

### What to Look For
1. **K-Fold accuracy >> random**: Model generalizes across domains
2. **Group LOO accuracy high**: Model captures fine-grained (model, env, scaffold) interactions
3. **Low log loss**: Predicted probabilities are well-calibrated

## Computational Resources

### Recommended Setup
- **K-Fold**: 7 cores minimum (one per environment)
- **Group LOO**: 20-42 cores for reasonable runtime

### Memory Requirements
- Each fold: ~8-10 GB RAM (PyMC MCMC sampling)
- Total for 42 parallel jobs: ~400 GB RAM
- **Suggestion:** Run Group LOO with `n_jobs=10-20` if RAM limited

### Expected Runtimes
| Evaluation | n_jobs | Runtime |
|------------|--------|---------|
| K-Fold     | 7      | ~6h     |
| Group LOO  | 42     | ~3h     |
| Group LOO  | 20     | ~6h     |
| Group LOO  | 10     | ~12h    |

*Runtime per fold: ~50 minutes (2 IRT models + 1 agent model)*

## Troubleshooting

### "No valid predictions generated"
- Happens when test data has (model, environment) combinations not seen in training
- Expected for Group LOO - some groups may not be predictable
- Check `n_test_with_theta` in results to see how many test samples had valid theta estimates

### Memory Issues
- Reduce `n_jobs` to run fewer folds in parallel
- Each fold is independent, so you can run them sequentially if needed

### Failed Folds
- Check logs for specific fold failures
- Common cause: Not enough data for IRT model to fit properly
- Results JSON includes `null` metrics for failed folds

## Next Steps

After running evaluations:
1. Compare K-Fold results across the 4 model variants (baseline, category, task, category_task)
2. Identify which environments are hardest to predict (lowest accuracy in K-Fold)
3. Check if Group LOO shows systematic differences (e.g., some models harder to predict than others)
4. Use results to decide which model specification is best for your analysis

## Advanced: Running for All 4 Model Variants

Currently the script only evaluates the **category_task** model. To compare all 4 variants, you would need to:

1. Modify `fit_and_predict()` to accept `include_task_effects` and `include_category` flags
2. Run evaluations for each variant separately
3. Compare predictive performance to see which model specification generalizes best

This can help answer: "Does adding task effects improve out-of-sample prediction, or just in-sample fit?"
