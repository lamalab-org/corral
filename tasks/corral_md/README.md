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

## See the tasks

```bash
curl http://localhost:8000/tasks/
```
