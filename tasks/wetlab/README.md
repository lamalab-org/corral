# WetLab Task Environment

This directory contains the WetLab task environment for Corral. It provides a qualitative inorganic analysis benchmark where agents interact with a simulated wet lab to identify ions by applying laboratory-style procedures.

The WetLab package was created by [Sadra Aghajani](https://github.com/aaaghajani).

## Setup

Create the Conda environment from this directory:

```bash
cd tasks/wetlab
conda env create -f environment.yml
conda activate wetlab
```

The environment file installs the local package in editable mode via `pip -e .`, so local code changes in this directory are picked up directly.

## Inspect The Environment Definitions

Build and list the WetLab environment definitions from this directory:

```bash
cd tasks/wetlab
python -m wetlab.env --level 2
```

The inspection command accepts these options:

- `--level`: Task difficulty level. Available task sets are stored under `level_1`, `level_2`, and `level_3`.
- `--subtask`: Set to `True` to load the subtask benchmark from `wetlab/subtasks_json/` instead of the main task set.

Example using subtasks:

```bash
cd tasks/wetlab
python -m wetlab.env --level 2 --subtask True
```

## Notes

> **Warning:** Even if your shell prompt shows `(wetlab)`, `conda activate` may not correctly override a `python` alias set elsewhere. Always verify you are using the right interpreter with `which python` — it should point to `/opt/homebrew/Caskroom/miniconda/base/envs/wetlab/bin/python`. If it does not, either run `conda init zsh` and restart your terminal, or prefix commands with `conda run -n wetlab`, e.g.:
> ```bash
> conda run --no-capture-output -n wetlab python -m wetlab.env --level 2
> ```

- The Conda environment depends on `reaktoro`, so using Conda or Mamba is the intended installation path.
- The package constructs one definition per task for registration on a Temporal worker.
- Inventory state is captured in Corral State; scoring is performed separately by the evaluation layer.
