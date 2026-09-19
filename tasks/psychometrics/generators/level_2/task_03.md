# Task 03 — which population did each case sample come from?

## Task

Six anonymised reference populations answered the Dirty Dozen. Five case samples
of 200 respondents came from those populations. For each case, report every
population compatible with the data.

A population is compatible when its case log-likelihood is no more than `.06`
units per respondent below the best-fitting population. Do not force a unique
answer when the rule leaves alternatives.

## Files

- six labelled reference datasets;
- `cases.csv`: five anonymised case samples;
- `codebook.md`: item and response definitions.

## Output

```json
{"classifications": {"case_1": ["population_a"], "case_2": ["population_b", "population_c"]}}
```

Include every case exactly once. Candidate order does not matter.

## What needs checking

The populations differ mainly in item relationships, not marginal response
distributions. A case sample can contain genuine overlap; one respondent or item
means alone cannot establish a population.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy python generators/level_2/gen_l2_t03_ddm_population_classification.py [--verify|--naive]
```
