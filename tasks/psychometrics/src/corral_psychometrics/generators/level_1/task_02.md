# Task 02 — Dirty Dozen factor structure

## Task

You are given responses from an online personality survey. Use the codebook to understand the variables, then investigate the theoretically plausible psychometric models for the Dirty Dozen.

Your analysis should support a conclusion about the number of traits, the items belonging to each narrower trait, and the standardised loading of every item. The HSNS items are present in the dataset but are not part of this task.

Responses are continuous and unbounded; zero is a valid neutral response and empty fields denote missing responses. The target population is respondents from the United States.

`codebook.md` gives the response coding and the text of every item.

## What to report

Submit a complete semopy model for the 12 Dirty Dozen items and one standardised loading for every item.

## Why this is non-trivial

Compare plausible structures using fit, parameter estimates, parsimony, and the meaning of the proposed scores. An acceptable fit cutoff alone does not establish the dimensional structure.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t02_dd_structure.py [--verify|--naive]
```
