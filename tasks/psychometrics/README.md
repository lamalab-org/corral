# Psychometrics tasks

Twenty tasks use simulated data with private ground truth. Questionnaire items
are continuous and centered around zero; zero is a valid response and empty
fields are missing. L2 Task 09 uses binary IRT responses.

## Layout

```text
src/corral_psychometrics/          generators, environment, tools, and scoring
artifacts/level_N/task_NN/         public task files
truth/level_N/task_NN.json         private answers and reference models
environments/level_N/tasks_json/   generated task definitions
build.py                           artifact generation and validation
```

Prompts and submission formats live in the generators. Generated definitions
are consumed by the environment. Setup copies only declared `public_inputs`.

## Run

From this directory:

```bash
uv sync --group dev
uv run python build.py --check                      # rebuild and validate all tasks
uv run python build.py --level 2 --tasks 3 7 --check # selected tasks
uv run python build.py --verify --naive              # validate without rebuilding
uv run pytest tests
uv run python -m corral_psychometrics.env --level 1
```

Editable installs use this directory as the data root. For other installations,
set `CORRAL_PSYCHOMETRICS_ROOT` or pass `data_root` to `create_environments`.
`build.py --data-root DIR` publishes generated files to another directory.

Agents receive file tools, a persistent `PythonREPL`, and a syntax validator.
The REPL includes NumPy, pandas, SciPy, factor-analyzer, and semopy. Docker runs
use Corral's restricted worker; checkpoint limits are configured by Corral.

## Models and scoring

L1 Tasks 02, 04, 07, and 08 use Dirty Dozen bifactor S−1 models. Task 02 facet
reliabilities are calibrated to .60–.70. Method effects in L1 Task 03 and L2 Tasks
03, 04, 08, and 10 are tied to documented presentation formats.

Continuous models use ML fitting, validity checks, positive degrees of freedom,
and comparisons of CFI, RMSEA, SRMR, and BIC. Comparisons use identical variables
and complete-case samples. The replication task restores omitted items for this
comparison. Interpretation, item diagnoses, and replication claims are assessed
separately. L2 Task 10 compares admissible catalogue models; L2 Task 09 uses IRT
and adaptive-test diagnostics.

The benchmark uses the binary all-checks-pass score. Partial scores are diagnostic;
comparable task difficulty requires agent pilot results.

## Tasks

| Number | Level 1 | Level 2 |
|---|---|---|
| 01 | HSNS structure | Group comparability |
| 02 | Dirty Dozen structure | Out-of-sample generalization |
| 03 | Trait and method variance | Dirty Dozen population classification |
| 04 | Measurement invariance | HSNS population classification |
| 05 | Defensible gender comparisons | Duplicate records |
| 06 | Latent relationships | Export artifacts and item integrity |
| 07 | Cross-country replication | Behavioral validity |
| 08 | Score interpretation | Replication of model modifications |
| 09 | Cross-instrument correlations | Adaptive item-bank selection |
| 10 | Item integrity | Model identification |
