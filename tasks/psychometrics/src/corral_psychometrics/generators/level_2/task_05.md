# Task 05 — what explains the apparent relationship?

## Task

Investigate an unexpectedly strong relationship between two HSNS dimensions in a
panel-delivered survey. Use the response file, preliminary memo, and delivery
audit to decide whether the relationship reflects repeated deliveries, shared
response profiles, or both. Identify a record key, collapse repeats with that
key, and refit the measurement model.

## Files

- `data.csv` and `codebook.md`;
- `preliminary_analysis.md`;
- `delivery_audit.csv` and `delivery_audit_codebook.md`.

The audit describes deliveries, not respondents. Different people may share its
values, and a repeated delivery may receive a new participant identifier.

## Output

```json
{"model_syntax": "F1 =~ ...\nF2 =~ ...\nF1 ~~ F2", "diagnosis": "repeated_delivery_only|shared_profiles_only|both_mechanisms", "duplicate_key": ["<column>", "..."]}
```

The key must use columns from `data.csv`, so duplicate records can be collapsed
consistently and the model fit and factor correlation can be reproduced.

## What needs checking

This synthetic panel samples a finite pool of continuous response profiles, so distinct people can have identical answers. Removing every match can
discard genuine respondents; retaining every row can count one respondent twice.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy python generators/level_2/gen_l2_t05_duplicate_records.py [--verify|--naive]
```
