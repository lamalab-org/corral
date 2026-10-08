# Task 01 — can this questionnaire compare the two groups?

## Task

Assess a reported HSNS difference between two groups. The memo uses observed
scores, while the response file and behavioural indicators provide evidence
about measurement and the underlying traits. Decide whether the comparison is
defensible and submit a model, one recommendation, and the affected items.

## Files

- `data.csv`: HSNS responses and group information;
- `behavior.csv`: behavioural indicators for the same participants;
- `codebook.md`, `behavior_codebook.md`, and `preliminary_analysis.md`.

## Output

Use the JSON format in the task definition. The submitted model must support the
comparison behind the recommendation.

## What needs checking

Observed score differences can reflect item response behaviour rather than
trait differences. Competing measurement models and the external indicators
help distinguish those explanations.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy python generators/level_2/gen_l2_t01_group_comparability.py [--verify|--naive]
```
