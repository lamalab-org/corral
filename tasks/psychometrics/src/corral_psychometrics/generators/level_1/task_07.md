# Task 07 — cross-country replication

## Task

You are given responses from an online personality survey collected in several countries. Calibrate a measurement model for each instrument using the US sample, then investigate how far each model carries over to every other country.

Classify each instrument in each non-US country as exactly one of:

- `exact`: the model holds with the same parameter values;
- `approximate`: the structure holds but parameter values differ;
- `substantive_only`: the structure fails, but the instrument still measures
the same broad construct or constructs;
- `none`: neither the structure nor the substantive interpretation carries
over.

Responses are continuous and unbounded; zero is a valid neutral response and empty fields denote missing responses.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit one complete model calibrated on the United States covering both instruments, plus the classification for both instruments in all six non-US countries. Identify the Dirty Dozen reference facet (`M` or `N`) in the US, GB, CA and AU.

## Why this is non-trivial

Replication is more than asking whether the original model still has acceptable fit. A model can fit well while its reference facet changes from Machiavellianism to narcissism. Parameter changes, construct changes, item reassignment, and different country sample sizes need to be considered separately.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t07_cross_country_replication.py [--verify|--naive]
```
