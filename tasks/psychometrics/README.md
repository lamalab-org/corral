# Psychometrics tasks

This environment contains psychometrics tasks. Each dataset is simulated from a known model, so the ground truth is available for evaluation.

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
