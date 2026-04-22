# Spectra Elucidation Task Environment

This directory contains the spectra elucidation task environment for Corral. It provides organic-structure elucidation benchmarks where agents interpret spectroscopic evidence, use chemistry tools, and submit molecular hypotheses that are scored against hidden target structures.

## Setup

Create and activate the virtual environment from this directory:

```bash
cd tasks/spectra_elucidation
uv venv --python 3.11.0
source .venv/bin/activate
uv sync
```

If you prefer not to activate the environment, use `uv run` to prefix the commands below.

## Run The Server

Start the spectra elucidation environment server from this directory:

```bash
cd tasks/spectra_elucidation
source .venv/bin/activate
python -m spectra_elucidation.env --level 1
```

To run the subtask benchmark instead:

```bash
cd tasks/spectra_elucidation
source .venv/bin/activate
python -m spectra_elucidation.env --level 1 --subtask_level True
```

The server also accepts these options:

- `--host`: Bind host. Defaults to `CORRAL_HOST` or `0.0.0.0`.
- `--port`: Bind port. Defaults to `CORRAL_PORT` or `8000`.
- `--level`: Benchmark level to load. Available task sets are stored under `environments/level_1` and `environments/level_2`.
- `--subtask_level`: Set to `True` to load `subtasks_json/` instead of `tasks_json/` for the selected level.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Notes

- The environment exposes tools for interpreting and simulating mass spectrometry, IR, and NMR data, as well as structure-validation utilities.
- Each task gets an isolated work directory under `CORRAL_WORK_DIR` if that environment variable is set.
- Levels provide different benchmark collections, while subtask mode exposes chained decomposition tasks rather than the main task set.
