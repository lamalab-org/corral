# Task 01 — HSNS factor structure

## Task

You are given responses from an online personality survey. Use the codebook to understand the variables, then investigate the theoretically plausible psychometric models for the Hypersensitive Narcissism Scale (HSNS).

Your analysis should support a conclusion about how many traits the HSNS measures, which items belong to each trait, how strongly each item loads on its trait, and how the traits relate to one another. The Dirty Dozen items are present in the dataset but are not part of this task.

Responses are five-point ratings and `0` denotes a missing response. The target population is respondents from the United States.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete lavaan/semopy model, one standardised loading for every HSNS item, and the latent factor correlation when the selected model has correlated factors.

## Why this is non-trivial

Several plausible models may fit well. Model choice should consider interpretability, distinctness of the traits, item coverage, and the population being studied—not fit indices alone. A weak item or a small cross-loading can make the preferred structure less obvious.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t01_hsns_structure.py [--verify|--naive]
```
