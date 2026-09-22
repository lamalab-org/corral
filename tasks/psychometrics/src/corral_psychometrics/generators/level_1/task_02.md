# Task 02 — Dirty Dozen factor structure

## Task

You are given responses from an online personality survey. Use the codebook to understand the variables, then investigate the theoretically plausible psychometric models for the Dirty Dozen.

Your analysis should support a conclusion about the number of traits, the items belonging to each narrower trait, and the standardised loading of every item. The HSNS items are present in the dataset but are not part of this task.

Responses are five-point ratings and `0` denotes a missing response. The target population is respondents from the United States.

## What to report

Submit a complete lavaan/semopy model for the 12 Dirty Dozen items and one standardised loading for every item.

## Why this is non-trivial

A published subscale structure can fit adequately while still missing an important common dimension. Some apparently different model descriptions are also mathematically equivalent in this setting. Compare plausible models and their substantive interpretation rather than treating one fit cutoff as the answer.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t02_dd_structure.py [--verify|--naive]
```
