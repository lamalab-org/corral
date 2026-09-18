# Psychometrics Task Environment

This environment contains synthetic psychometrics tasks. Each dataset is
simulated from a known model, so the ground truth is available for evaluation.
The tasks are designed to require judgement: a simple or fit-only analysis may
give a plausible but misleading conclusion.

## Layout

```
pyproject.toml                    project dependencies and tool configuration
.pre-commit-config.yaml            repository quality checks
build.py                          rebuild and check the whole environment
generators/common.py              simulation, fitting and artifact code shared by all tasks
generators/level_*/gen_*.py       one script per task: the model its data come from
generators/level_*/task_*.md      short human-facing task descriptions
artifacts/level_*/task_*/         generated data and task inputs; truth.json is hidden
environments/level_*/tasks_json/  the task definitions
psychometrics/score.py            the scorer
tests/test_scoring.py             checks every task scores as intended
```

## Setup and building

The project uses `uv` and a local virtual environment. Install the runtime and
development dependencies with:

```bash
uv sync --group dev
uv run pre-commit install
```

Generators build everything under `artifacts/` and `environments/`.
Rebuild the environment with:

```bash
uv run python build.py --clean   # rebuild from nothing
uv run python build.py --check   # build, verify, run naive checks and scoring
uv run python build.py --tasks 3 7
```

`--check` verifies that the intended answer wins (`--verify`), the obvious
analysis fails (`--naive`), and the scorer gives the intended verdicts.

To run only the scoring integration test:

```bash
uv run python tests/test_scoring.py
```

## Scoring

Tasks use either a submitted model or a structured JSON conclusion. When a model
is submitted, the scorer refits it and checks its fit and reported claims. For
classification and decision tasks, it checks the required fields against
results derived from the generated evidence. Every scorer returns a binary
score and a partial diagnostic score.

## Model syntax

Submissions and reference models use lavaan-style syntax such as `=~`, `~~`,
and `~`. The Python implementation fits that syntax with `semopy`; R and the
R `lavaan` package are not required for this repository.

## Quality checks

Run all configured pre-commit checks manually with:

```bash
uv run pre-commit run --all-files
```

## Level 1

Level 1 tasks

| task | investigation |
|---|---|
| [1](generators/level_1/task_01.md) | HSNS factor structure |
| [2](generators/level_1/task_02.md) | Dirty Dozen factor structure |
| [3](generators/level_1/task_03.md) | HSNS dimensionality and local dependence |
| [4](generators/level_1/task_04.md) | Measurement invariance for a gender comparison |
| [5](generators/level_1/task_05.md) | Which gender comparisons are defensible? |
| [6](generators/level_1/task_06.md) | Relationships between latent dimensions |
| [7](generators/level_1/task_07.md) | Cross-country replication |
| [8](generators/level_1/task_08.md) | What scores are justified? |
| [9](generators/level_1/task_09.md) | Relationships between the two instruments |
| [10](generators/level_1/task_10.md) | Item integrity |

## Level 2

Level 2 tasks provide more than a dataset and a direct question. They may include
preliminary analyses, competing interpretations, or multiple files. The agent
must decide what evidence would distinguish the explanations, test the relevant
assumptions, and report a calibrated conclusion.

| task | investigation |
|---|---|
| [1](generators/level_2/task_01.md) | Can this questionnaire compare the two groups? |
| [2](generators/level_2/task_02.md) | Does the HSNS model generalize out of sample? |
| [3](generators/level_2/task_03.md) | Which population did each case sample come from? |
| [4](generators/level_2/task_04.md) | Which HSNS population fits each case sample? |
| [5](generators/level_2/task_05.md) | Duplicate records or chance collisions? |
| [6](generators/level_2/task_06.md) | Is the group difference real or an export artifact? |
| [7](generators/level_2/task_07.md) | Does the HSNS predict behaviour beyond group membership? |
| [8](generators/level_2/task_08.md) | Which model modifications survive replication? |
| [9](generators/level_2/task_09.md) | Which item bank should the adaptive test use? |
| [10](generators/level_2/task_10.md) | Which model generated each dataset? |
