# Resistor Network Task Environment

This directory contains the resistor network task environment for Corral. It provides circuit-reasoning benchmarks where agents infer a hidden resistor topology from measurements and submit a network description with resistor values.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/resistor_network
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Inspect The Environment Definitions

Build and list the resistor-network environment definitions from this directory:

```bash
cd tasks/resistor_network
source .venv/bin/activate
python src/resistor_network/env.py --mode single
```

To run the chained subtask benchmark instead:

```bash
cd tasks/resistor_network
source .venv/bin/activate
python src/resistor_network/env.py --mode chained
```

The inspection command accepts these options:

- `tasks_json_path`: Optional path to a task JSON file or directory. If omitted, the command auto-discovers the level 1 benchmark.
- `--mode`: Use `single` to load `environments/level_1/tasks_json` or `chained` to load `environments/level_1/subtasks_json`.

## Notes

- Submissions are expected to describe the circuit as JSON with `resistors` and `connections` fields.
- Scoring checks both structural agreement and agreement with expected equivalent-resistance measurements.
- Internal resistance validation is based on nodal analysis, so the submitted topology should define a passive resistor network with positive resistance values.
