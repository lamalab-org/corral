# WetLab Task Environment

This directory contains the WetLab task environment for Corral. It provides a qualitative inorganic analysis benchmark where agents interact with a simulated wet-lab server to identify ions by applying laboratory-style procedures.

The WetLab package was created by [Sadra Aghajani](https://github.com/aaaghajani).

## Setup

Create the Conda environment from this directory:

```bash
cd tasks/wetlab
conda env create -f environment.yml
conda activate wetlab
```

The environment file installs the local package in editable mode via `pip -e .`, so local code changes in this directory are picked up directly.

## Run The Server

Start the WetLab environment server from this directory:

```bash
cd tasks/wetlab
python -m wetlab.env --level 2 --port 8000
```

The server also accepts these options:

- `--host`: Bind host. Defaults to `CORRAL_HOST` or `0.0.0.0`.
- `--port`: Bind port. Defaults to `CORRAL_PORT` or `8000`.
- `--level`: Task difficulty level. Available task sets are stored under `level_1`, `level_2`, and `level_3`.
- `--subtask`: Set to `True` to load the subtask benchmark from `wetlab/subtasks_json/` instead of the main task set.

Example using subtasks:

```bash
cd tasks/wetlab
python -m wetlab.env --level 2 --subtask True --port 8000
```

## Notes

> **Warning:** Even if your shell prompt shows `(wetlab)`, `conda activate` may not correctly override a `python` alias set elsewhere. Always verify you are using the right interpreter with `which python` — it should point to `/opt/homebrew/Caskroom/miniconda/base/envs/wetlab/bin/python`. If it does not, either run `conda init zsh` and restart your terminal, or prefix commands with `conda run -n wetlab`, e.g.:
> ```bash
> conda run --no-capture-output -n wetlab python env.py --level 2
> ```

- The Conda environment depends on `reaktoro`, so using Conda or Mamba is the intended installation path.
- The server creates one environment per task and serves them through Corral's backend server.
- Inventory state and scoring are handled inside the WetLab package, so the standard Corral runner can connect to the exposed HTTP server once it is running.