# Task 08 — which model modifications survive replication?

## Task

Review a two-factor HSNS model in a development sample and an independent
replication sample. Decide which apparent problems are reproducible and whether
they represent local dependence, cross-loadings, DIF, or a weak item.

## Files

- `data.csv`: development responses and group variable;
- `replication.csv`: independent replication responses;
- `codebook.md` and `preliminary_analysis.md`.

## Output

Report a final model and, for each category below, the supported findings and
whether they replicate. Categories are not mutually exclusive.

- `local_dependence`: item pairs requiring residual covariance;
- `cross_loadings`: items loading on a second factor;
- `dif_items`: items with group-dependent response after accounting for factors;
- `poor_items`: items with a loading too weak to retain.

Also report development-supported modifications rejected after replication. Use
the JSON format in the task definition. A poor item may be omitted from the final
model; no other item may be omitted. The model must give `gender` a path to each
factor.

## What needs checking

A modification index from one sample is evidence to investigate, not evidence to
retain. A defensible modification has a clear interpretation and recurs in the
independent sample.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy python generators/level_2/gen_l2_t08_misfit_replication.py [--verify|--naive]
```
