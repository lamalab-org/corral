# Task 07 — does the HSNS predict behaviour beyond group membership?

## Task

Evaluate whether the HSNS predicts an independently measured behavioural outcome.
A memo reports an association in training data; a separate holdout contains the
same questionnaire, group variable, and outcome.

Use these rules:

- `supported`: the group-adjusted latent coefficient has absolute value at least
  `.10` in both samples;
- `replicates`: the coefficient has absolute value at least `.10` in both groups,
  and the two within-group estimates differ by no more than `.06`;
- `generalizes`: the group-adjusted coefficient changes by no more than `.06`
  between training and holdout;
- an item is unstable when its direct outcome coefficient is at least `.10` in
  training and no more than `.05` in holdout.

## Files

- `data.csv` and `holdout.csv`;
- `codebook.md`;
- `preliminary_analysis.md`.

## Output

```json
{"model_syntax": "F =~ HSNS1+...+HSNS10\nbehavior ~ F + gender", "association": {"group_adjusted": "supported|not_supported", "within_group": "replicates|does_not_replicate", "holdout": "generalizes|does_not_generalize"}, "unstable_items": ["item_name"]}
```

The model must cover all HSNS items and include `behavior` and `gender`. The
scorer re-estimates the claims from the submitted model.

## What needs checking

A pooled association can combine a within-group relationship with a difference
between groups. A training-only item effect can also inflate the latent result.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy python generators/level_2/gen_l2_t07_behavioral_validity.py [--verify|--naive]
```
