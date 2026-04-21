# Catalyst Task Environment

This directory contains the catalyst task environment for Corral. It provides computational catalysis benchmarks where agents retrieve bulk structures, generate slabs, identify adsorption sites, and construct adsorbate structures for catalyst design workflows.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/catalyst
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Run The Server

Start the catalyst environment server from this directory:

```bash
cd tasks/catalyst
source .venv/bin/activate
python src/catalyst/env.py --mode single
```

To run the chained subtask benchmark instead:

```bash
cd tasks/catalyst
source .venv/bin/activate
python src/catalyst/env.py --mode chained
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

- Many catalyst tools rely on Materials Project data and require `MP_API_KEY` to be available in the environment.
- The benchmark expects internet access for live materials retrieval.
- Generated structures and intermediate files are stored inside the task work directory configured through `CORRAL_WORK_DIR`.
