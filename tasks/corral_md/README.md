# Corral MD

Corral MD is a molecular dynamics benchmark built around LAMMPS, ASE, and
machine-learned interatomic potentials.

## Run it

From the repository root, install the environment:

```bash
cd tasks/corral_md
uv venv --python 3.11
uv sync
```

After completing the one-time Modal setup below, run a task with:

```bash
uv run corral bench \
  --sandbox local \
  --agent tool-calling \
  --environment corral_md \
  --task level_1_task_1 \
  --model openai/gpt-5.6
```

Use an agent and model for which you have credentials. To use Level 2, add
`--env-kwargs '{"level": 2}'` and select a `level_2_task_*` task. To list the
available tasks without running them, use:

```bash
uv run corral bench \
  --agent tool-calling \
  --environment corral_md \
  --list-tasks
```

When Modal-backed evaluation is enabled, the deployed evaluation functions are
hard-capped at 25 active containers per function. For a large benchmark, also
bound submissions from the Corral process so excess evaluations wait locally
instead of in Modal's input queue:

```bash
uv run corral bench \
  --sandbox local \
  --agent tool-calling \
  --environment corral_md \
  --env-kwargs '{"level": 2}' \
  --model openai/gpt-5.6 \
  --max-parallel-evaluations 25 \
  --max-parallel-evaluations-by-environment '{"corral_md": 25}'
```

The environment key must be the canonical runtime name `corral_md`.

## Modal setup and workspace sync

This environment is ready to run its simulation and analysis tools on
[Modal](https://modal.com). You need a Modal account configured on your machine:

```bash
uv run modal setup
uv run python modal_app/release.py
```

The second command deploys the workers and their assets. It is also the command
to run again after changing worker code. See the
[Modal notes](modal_app/README.md) for deployment and recovery details.

Each task still feels local: Corral uploads changed workspace files before a
remote tool runs and downloads the results when it finishes successfully.

## What the tasks cover

Level 1 prepares the structures, datasets, and initial simulation stages. Level
2 turns them into complete workflows with saved results and reproducible
evidence. The ten task pairs cover:

Each level has its own rubric. Level 1 evaluates only the preparation work in
its prompt and permits the Level-2-only fields in the shared submission example
to remain empty. Checks for work common to both levels use the same underlying
evidence validation; the later Level 2 stages are scored only at Level 2.

- silicon diffusion and sodium-silicate glass transition;
- silver model fine-tuning plus silicon and copper energy regression;
- palladium and strained-silicon phonons, and aluminum vibrational density of states;
- aluminum thermal expansion and heat capacity.
