# Psychometrics tasks

This environment contains psychometrics tasks. Each dataset is simulated from a known model, so the ground truth is available for evaluation.

Questionnaire tasks use synthetic continuous, unbounded item responses centered
around zero. Empty fields are missing; zero is a valid neutral response. Reverse
scoring negates a response. Level 2 Task 09 retains binary adaptive IRT data.
Scoring fits continuous SEMs by ML and compares BIC on the same items and sample.

Dirty Dozen tasks L1 02, 04, 07 and 08 use bifactor S−1 models. Task 02 targets
population omega of .60–.70 for each four-item facet score.

Method effects are explicit orthogonal factors tied to documented presentation
formats in L1 03 and L2 03, 04, 08 and 10. The main clusters have four items.
L2 08 also includes an identified two-item effect found only in development.
L1 03 asks agents to distinguish personality from method variance in the total
score; method variance is excluded from its reported trait variance share.
L1 01/08 and L2 01/02 omit incidental residual dependence to keep their questions
focused. L2 09 retains its binary IRT pair-dependence exercise.

## Scoring consistency

Continuous SEM tasks share validity checks (convergence, nonnegative variances,
positive degrees of freedom, finite standard errors, and nondegenerate factors)
and matched comparisons of CFI, RMSEA, SRMR and BIC. Comparisons use the same
observed variables and complete-case sample; the replication task restores
omitted items for this comparison. Scientific interpretation, item diagnoses,
and replication claims remain task-specific requirements.

L2 10 selects BIC-supported candidates only after checking model validity.
L2 09 uses binary IRT and adaptive-test diagnostics rather than continuous SEM
fit indices. Passing reference and adversarial checks establishes scoring
consistency; comparable agent difficulty requires pilot performance data.
The benchmark uses the binary all-checks-pass score; partial scores are diagnostic
and are not calibrated measures of difficulty.

## Layout

```text
src/corral_psychometrics/          # package, generators, scorer, environment
artifacts/level_N/task_NN/         # public files copied to the workspace (data and codebook)
truth/level_N/task_NN.json         evaluator-only answers
environments/level_N/tasks_json/   task definitions
build.py                           generate and validate data
```

Task definitions list `public_inputs`. Setup copies exactly those files and never copies `truth/`.

## Install

```bash
uv sync --group dev
```

Editable installs use the checkout as the default data root.
For a regular install, pass a data root or set `CORRAL_PSYCHOMETRICS_ROOT`:

```python
from corral_psychometrics.env import create_environments

envs = create_environments(
    level=1,
    data_root="/path/to/psychometrics-data",
    work_dir="/path/to/workspaces",
)
```


## Build and validate


```bash
uv run python build.py                         # rebuild all tasks
uv run python build.py --level 2 --tasks 3 7   # rebuild selected tasks
uv run python build.py --data-root DIR         # write to another data root
uv run python build.py --check                 # build and run all checks
uv run python build.py --verify                # validate intended answers
uv run python build.py --naive                 # validate naive analyses fail
```


## Agent environment

```bash
uv run python -m corral_psychometrics.env --level 1
uv run python -m corral_psychometrics.env --level 2
```

Each task receives its public files, workspace file tools, a persistent `PythonREPL`, and `validate_model_syntax`.
The REPL starts in the workspace; NumPy, pandas, SciPy, factor-analyzer, and semopy are available.

The scorer, generators, answer keys, and shell are not exposed to the agent.

### REPL checkpoints under Docker

Under Docker the REPL runs in a restricted worker that hands its whole session
back as a checkpoint on every call. Reading the survey into a dataframe
checkpoints to about 12 MiB, and roughly 25 MiB once a country subset and a
float matrix are derived from it.

Nothing needs configuring for that. The checkpoint travels on Corral's bulk
channel, which is spooled to disk rather than held in controller memory, and its
1 GiB default is far above what these tasks reach. A session is not bounded by
the smaller limit on a worker's JSON reply. Both are Corral-wide settings rather
than psychometrics ones, and either can be changed for a whole run:

```bash
CORRAL_MAX_WORKER_BULK_BYTES=$((2 * 1024 ** 3)) uv run python -m corral_psychometrics.env --level 1
```

A payload over either limit fails the call and names the setting to raise.

`corral bench` grants the trial container the capabilities the worker needs, so
these tasks need no Docker flags of their own. Running a trial image by hand
does; see "Restricted worker limits" in the Corral reference.

## Task list

| level | task | topic |
|---|---:|---|
| 1 | 1 | HSNS factor structure |
| 1 | 2 | Dirty Dozen factor structure |
| 1 | 3 | HSNS dimensionality and local dependence |
| 1 | 4 | Measurement invariance |
| 1 | 5 | Defensible gender comparisons |
| 1 | 6 | Latent relationships |
| 1 | 7 | Cross-country replication |
| 1 | 8 | Score justification |
| 1 | 9 | Relationships between instruments |
| 1 | 10 | Item integrity |
| 2 | 1 | Group comparability |
| 2 | 2 | Out-of-sample generalization |
| 2 | 3 | Population classification |
| 2 | 4 | Population classification |
| 2 | 5 | Duplicate records |
| 2 | 6 | Export artifacts |
| 2 | 7 | Behavioral validity |
| 2 | 8 | Replication of model modifications |
| 2 | 9 | Adaptive item-bank choice |
| 2 | 10 | Model identification |
