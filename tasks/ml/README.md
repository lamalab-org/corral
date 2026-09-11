# ML Task Environment

This directory contains the machine learning task environment for Corral. It provides benchmark tasks around dataset preparation, materials-science data retrieval, model training, and evaluation workflows, including polymorph discovery and dataset curation tasks.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/ml
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Inspect The Environment Definitions

Build and list the ML environment definitions from this directory:

```bash
cd tasks/ml
source .venv/bin/activate
python src/ml/env.py --mode single
```

To run the chained subtask benchmark instead:

```bash
cd tasks/ml
source .venv/bin/activate
python src/ml/env.py --mode chained
```

The inspection command accepts these options:

- `tasks_json_path`: Optional path to a task JSON file or directory. If omitted, the command auto-discovers the level 1 benchmark.
- `--mode`: Use `single` to load `environments/level_1/tasks_json` or `chained` to load `environments/level_1/subtasks_json`.

## Notes

- Some tools query Materials Project and require `MP_API_KEY` to be set in the environment.
- Corral creates an isolated workspace per task under `CORRAL_WORK_DIR` if that environment variable is provided.
- Tasks can write files inside their task-specific work directory, so keeping a dedicated virtual environment per task is recommended.
