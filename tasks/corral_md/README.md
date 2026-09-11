# Corral MD Environment

LAMMPS molecular dynamics environment. The agent is tasked with setting up and running molecular dynamics simulations using LAMMPS, including structure preparation, interatomic potential selection, and analysis of simulation outputs such as logs and mean squared displacement.

## Install environment

```bash
cd tasks/corral_md
uv venv --python 3.11
uv sync
```

## Run the environment

```bash
cd tasks/corral_md
python src/corral_md/env.py
```

To run a specific level:

```bash
python src/corral_md/env.py --level 1
```

To run subtask mode:

```bash
python src/corral_md/env.py --level 1 --subtask_level True
```

The command constructs and lists the selected environment definitions.

## Local workspaces and Modal

Task files are stored in the local Corral workspace. `run_lammps` automatically
uploads that workspace to an isolated directory in the Modal `simulations`
Volume, runs LAMMPS, and downloads the complete resulting directory before the
tool returns. Redeploy `modal_app/lammps_app.py` after changing its worker code.

Set `CORRAL_MD_MODAL_APP` or `CORRAL_MD_MODAL_VOLUME` to override the default
`simagent` app and `simulations` Volume names.
