# Task 03 — HSNS dimensionality and local dependence

## Task

You are given responses from an online personality survey. Use the codebook to investigate whether the HSNS measures one trait or more than one.

The analysis should also identify pairs of items whose responses share additional association beyond the common trait. Report the standardised loading of every item and the latent correlation if the selected model has multiple traits. The Dirty Dozen items are present but are not part of this task.

Responses are five-point ratings and `0` denotes a missing response. The target population is respondents from the United States.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete lavaan/semopy model for the 10 HSNS items, including correlated residuals where justified, plus the item loadings and any factor correlation.

## Why this is non-trivial

Local dependence can imitate an additional factor. Raw item correlations can also highlight a pair simply because both items are strong indicators. Deleting items may hide the symptom without explaining it, so distinguish item wording effects from substantive dimensionality.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t03_local_dependence.py [--verify|--naive]
```
