# Task 08 — what scores are justified?

## Task

You are given responses from an online personality survey containing HSNS and Dirty Dozen items. Using respondents from the United States, investigate what scores each instrument justifies.

Classify each instrument as exactly one of:

- `total_only`: one defensible score across all items;
- `subscales_only`: defensible separate subscale scores, but no defensible
total;
- `total_and_subscales`: both are defensible;
- `none`: neither is defensible.

Responses are five-point ratings and `0` denotes a missing response.

## What to report

Submit one complete model covering both instruments, one scoring classification for each instrument, and the item grouping implied by the model.

## Why this is non-trivial

Internal consistency is not the same as interpretability. A total can be reliable because several related dimensions share variance without representing one useful construct. Conversely, a sensible total can tolerate modest local item dependence. Use the dimensional structure and common-variance evidence, not reliability alone.

## Synthetic task command

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t08_score_justification.py [--verify|--naive]
```
