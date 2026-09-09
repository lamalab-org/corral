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

## Inspect The Environment Definitions

Build and list the catalyst environment definitions from this directory:

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

The inspection command accepts these options:

- `tasks_json_path`: Optional path to a task JSON file or directory. If omitted, the command auto-discovers the level 1 benchmark.
- `--mode`: Use `single` to load `environments/level_1/tasks_json` or `chained` to load `environments/level_1/subtasks_json`.

## Notes

- Many catalyst tools rely on Materials Project data and require `MP_API_KEY` to be available in the environment.
- The benchmark expects internet access for live materials retrieval.
- Generated structures and intermediate files are stored inside the task work directory configured through `CORRAL_WORK_DIR`.
