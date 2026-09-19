# Task 06 — is the group difference real or an export artifact?

## Task

Investigate a reported HSNS difference between two groups. Decide which items
are usable, repair problems that can be repaired, fit a model that allows item
behaviour to differ by group, and decide whether the comparison is reportable.

## Files

- `data.csv`: item responses and group indicator;
- `codebook.md`: response coding and item descriptions;
- `preliminary_analysis.md`;
- `export_audit.csv` and `export_audit_codebook.md`.

## Output

Give every item exactly one label: `sound`, `mis_keyed`, `inserted_neutral`,
`gender_dif`, or `weak_item`. Then submit model syntax and either
`reportable_after_repair` or `not_reportable`.

```json
{"model_syntax": "F =~ HSNS1+...+HSNS10\nF ~ gender\nHSNS2 ~ gender", "item_diagnoses": {"HSNS1": "sound"}, "comparison": "reportable_after_repair|not_reportable"}
```

The model must make the group comparison explicit and represent supported item
differences. Repair a mis-keyed item before interpreting the latent difference.

## What needs checking

The audit describes transmission, not measurement. A frequent response category
may be an export fault, genuine DIF, or a weak item. Compare suspect respondents
with their remaining response profiles.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy python generators/level_2/gen_l2_t06_gender_item_integrity.py [--verify|--naive]
```
