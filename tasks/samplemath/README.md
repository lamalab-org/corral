# SampleMath Task Environment

This directory contains the sample math task environment for Corral. It provides lightweight arithmetic and unit-conversion benchmarks that are useful both as a simple example task and as a small chained-task environment.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/samplemath
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Run The Server

Run the task-group benchmark using JSON task definitions from this directory:

```bash
cd tasks/samplemath
source .venv/bin/activate
python samplemath/env_subtask.py environments/level_1/tasks_json/task_1.json
```

To load the chained subtask version instead:

```bash
cd tasks/samplemath
source .venv/bin/activate
python samplemath/env_subtask.py environments/level_1/subtasks_json/task_1.json
```

The task-group server reads these environment variables:

- `CORRAL_HOST`: Bind host for the server. Defaults to `0.0.0.0`.
- `CORRAL_PORT`: Bind port for the server. Defaults to `8000`.
- `CORRAL_WORK_DIR`: Base working directory for per-task files.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Notes

- Available tools include arithmetic helpers, number conversion, and unit conversion.
- The JSON-backed task-group mode is the better reference if you want to understand chained tasks and task dependencies in a minimal environment.
