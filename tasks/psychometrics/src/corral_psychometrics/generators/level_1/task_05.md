# Task 05 — defensible comparisons between men and women

## Task

You are given responses from an online personality survey. Using US respondents who reported male or female, investigate how far the HSNS supports comparisons between the two groups.

Give a defensibility judgement for each of these comparisons:

- factor structure;
- item loadings;
- factor variances;
- association with the Dirty Dozen narcissism items;
- latent means;
- observed total-score means.

Also estimate the correlation between the HSNS trait and the Dirty Dozen narcissism trait separately for men and women. Responses are five-point ratings and `0` denotes a missing response.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete measurement model for the HSNS and Dirty Dozen narcissism items, a Boolean judgement for every comparison, and the two group-specific trait correlations.

## Why this is non-trivial

Different forms of comparability support different claims. Item response differences can make mean comparisons invalid even when the factor structure and associations are comparable. Bias in opposite directions may cancel in a total score, so the observed gap is not a reliable diagnostic.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t05_defensible_comparisons.py [--verify|--naive]
```
