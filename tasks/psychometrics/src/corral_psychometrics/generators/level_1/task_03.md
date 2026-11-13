# Task 03 — HSNS dimensionality and local dependence

## Task

You are given responses from an online personality survey. Use the codebook to investigate whether the HSNS measures one trait or more than one.

Use the presentation log and the covariance pattern to distinguish personality dimensions from a shared presentation effect. Report each item’s largest absolute standardised loading on a personality factor, and report the latent correlation if the selected model has multiple traits. The Dirty Dozen items are present but are not part of this task.

Responses are continuous and unbounded; zero is a valid neutral response and empty fields denote missing responses. The target population is respondents from the United States.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete semopy model for the 10 HSNS items, including correlated residuals or identified method factors where justified, plus the item loadings, any trait-factor correlation, and a `measurement_explanation` (`one_trait`, `multiple_traits`, `one_trait_with_method_effect`, or `multiple_traits_with_method_effect`). Report `trait_variance_share`: the proportion of variance in the unit-weighted ten-item sum attributable to personality factors, excluding any method variance.

## Why this is non-trivial

Local dependence can imitate an additional factor. Raw item correlations can also highlight a pair simply because both items are strong indicators. Deleting items may hide the symptom without explaining it, so distinguish presentation effects from substantive dimensionality.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t03_local_dependence.py [--verify|--naive]
```
