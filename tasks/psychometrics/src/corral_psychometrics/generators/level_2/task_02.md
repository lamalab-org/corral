# Task 02 — does the HSNS model generalize out of sample?

## Task

Develop a measurement model in the labelled training sample. Refit it unchanged
in three anonymised holdouts and classify each result.

| decision | rule |
|---|---|
| `generalizes` | every loading and the factor correlation stays within `.10` of training |
| `measurement_structure_holds_relations_differ` | loadings stay within `.10`, but the factor correlation moves more |
| `measurement_structure_fails` | at least one loading moves more than `.10` |

## Files

- `data.csv`: labelled training responses;
- `holdout_a.csv`, `holdout_b.csv`, `holdout_c.csv`: anonymised holdouts;
- `codebook.md`: variable definitions.

Do not infer population identities from filenames or metadata.

## Output

Submit the frozen model and one decision for each holdout. Do not modify the
model separately for a holdout.

## What needs checking

A model can fit its development sample while failing to transport. Measurement
structure may remain stable even when the relationship between factors changes.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy python generators/level_2/gen_l2_t02_out_of_sample_generalization.py [--verify|--naive]
```
