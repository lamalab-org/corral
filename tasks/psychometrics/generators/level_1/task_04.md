# Task 04 — measurement invariance for a gender comparison

## Task

You are given responses from an online personality survey containing HSNS and Dirty Dozen items. Investigate which instrument, if either, supports a comparison between men and women: the instrument must measure the same construct in both groups.

For the instrument that supports the comparison, estimate the standardised latent difference, defined as the latent trait regressed on `gender`. For the other instrument, identify the items that function differently across gender.

Use US respondents who reported male or female only (`gender` 1 or 2). Responses are five-point ratings and `0` denotes a missing response.

## What to report

Submit the measurement model for the selected instrument, the standardised latent difference, and the biased item names in the other instrument.

## Why this is non-trivial

Testing items one at a time can use a contaminated baseline and make fair items look biased. Group differences and item-level measurement differences must be separated, and pooling countries can change the comparison itself. Consider joint invariance evidence and the substantive meaning of the model.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t04_invariant_combination.py [--verify|--naive]
```
