# Task 10 — which model generated each dataset?

## Task

Six anonymised samples come from six candidate measurement models. Assign each
sample to the candidate supported by the data.

Fit every candidate to every dataset and compare BIC on usable responses. Report
every candidate within **2 BIC** of the best for that dataset. Do not break a
tie arbitrarily.

## Files

- `dataset_01.csv` through `dataset_06.csv`;
- `candidate_models.csv`: model descriptions and lavaan-style syntax;
- `codebook.md`: response and missing-value definitions.

The twelve items use five-point responses and `0` denotes missing data. For BIC,
use complete response rows and collapse exact duplicate response patterns. The
dataset-to-model mapping is permuted; catalogue order is not evidence.

## Output

```json
{"assignments": {"<dataset_id>": ["<model_id>", "..."]}}
```

Include every dataset exactly once and use only candidate IDs from the catalogue.

## What needs checking

Each sample has a different complication: missingness, residual dependence, a
subgroup threshold shift, duplicate rows, a weak item, or response-style shift.
These can distort fit without defining the generating model. Check that the
assignment follows the stated BIC rule rather than a filename, catalogue order,
or one unusual item.

## Rebuild

```bash
uv run python generators/level_2/gen_l2_t10_model_identification.py [--verify|--naive]
```
