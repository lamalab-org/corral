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

## Run The Server

Start the ML environment server from this directory:

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

The server also accepts these options:

- `tasks_json_path`: Optional path to a task JSON file or directory. If omitted, the server auto-discovers the level 1 benchmark.
- `--host`: Bind host. Defaults to `CORRAL_HOST` or `0.0.0.0`.
- `--port`: Bind port. Defaults to `CORRAL_PORT` or `8000`.
- `--mode`: Use `single` to load `environments/level_1/tasks_json` or `chained` to load `environments/level_1/subtasks_json`.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Notes

- Some tools query Materials Project and require `MP_API_KEY` to be set in the environment.
- The server creates an isolated workspace per task under `CORRAL_WORK_DIR` if that environment variable is provided.
- Tasks can write files inside their task-specific work directory, so keeping a dedicated virtual environment per task is recommended.
