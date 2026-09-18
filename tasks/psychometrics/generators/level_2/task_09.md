# Task 09 — which item bank should the adaptive test use?

## Task

A short adaptive test administers eight items per respondent and selects the
most informative available item at each step. The vendor recommends its full
20-item bank. Decide which bank to use and identify local dependence, unsupported
vendor flags, and items whose behaviour changes in the holdout sample.

## Files

- `data.csv`: 4,000 calibration responses;
- `holdout.csv`: 2,000 independent responses;
- `vendor_review.csv`, `codebook.md`, and `preliminary_analysis.md`.

The items are binary. Calibration discriminations range from `.70` to `1.65`.
The vendor's information assumes conditional independence and is not a test of
adaptive performance.

## Output

Submit the JSON format in the task definition: bank choice, dependent pairs,
unsupported vendor flags, unstable items, and a recommendation.

## What needs checking

High nominal information can count the same information twice when items are
locally dependent. A fixed-length analysis can miss the problem that appears
when an adaptive algorithm repeatedly selects the same items. The holdout is
needed to find item behaviour that does not transport.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy python generators/level_2/gen_l2_t09_adaptive_model_choice.py [--verify|--naive]
```
