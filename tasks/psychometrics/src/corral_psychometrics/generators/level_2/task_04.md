# Task 04 — which HSNS population fits each case sample?

## Task

Six anonymised reference populations answered the HSNS. Five case samples of 300
respondents came from those populations. For each case, report every compatible
population.

A population is compatible when its case log-likelihood is no more than `.04`
units per respondent below the best-fitting population. Some cases have more
than one compatible population.

## Files

- six labelled reference datasets;
- `cases.csv`: five anonymised case samples;
- `codebook.md`: item and response definitions.

## Output

```json
{"classifications": {"<case_id>": ["<population_id>", "..."]}}
```

Include every case exactly once. Candidate order does not matter.

## What needs checking

The populations differ mainly in relationships among items. Retain compatible
alternatives when the case sample does not separate them.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy python generators/level_2/gen_l2_t04_hsns_population_classification.py [--verify|--naive]
```
