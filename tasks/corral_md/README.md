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
uv run python modal_app/setup_simagent.py
```

The second command prepares the stable `simagent` app and its assets. Corral
discovers the active app automatically; no release environment variable is
needed. See the
[Modal notes](modal_app/README.md) for deployment and recovery details.

Each task still feels local: Corral uploads changed workspace files before a
remote tool runs and downloads the results when it finishes successfully.
LAMMPS, GPU Python, and independent verification use Modal. CPU Python and
terminal commands use Corral's shared tool executor. Docker trials run them in
an unprivileged worker with access only to the assigned workspace. With
`--sandbox local`, they run with the current user's OS permissions.

Tool path arguments use absolute POSIX paths under `/workspace`; controller
directories never belong in prompts or saved actions. Relative paths, parent
traversal, symlinks, and the reserved `/workspace/resources` namespace are
rejected. The virtual `/workspace/structures`, `/workspace/potentials`, and
`/workspace/models` catalogs are read-only. Use `copy_file` to copy assets into
`/workspace/input` for local analysis.

Remote dispatch retains the execution's pinned Modal build, storage volume,
workspace snapshot, and action ID. Interrupted remote calls remain resumable
under that same action ID instead of being recorded as completed failures.

## Bundled inputs

`structures.zip` contains the shared structure inputs, with its SHA-256 recorded
in `modal_app/assets.json`. Tasks 2 use the molten sodium-silicate state at
`/workspace/structures/melt/liq1300.dat`. Tasks 9 use the periodic 32-atom FCC
copper cell at `/workspace/structures/cu/cu32.extxyz`, with lattice parameter
3.615 Å. Tasks 1 retrieve mp-149 through Materials Project using `MP_API_KEY`;
that structure is not bundled.

Task definitions under `environments/` are included in the Python wheel along
with the submission templates and scoring structure archive. Deploying the
Modal app uses the repository's `modal_app/` scripts and both ZIP archives.

## What the tasks cover

Level 1 prepares the structures, datasets, and initial simulation stages. Level
2 turns them into complete workflows with saved results and reproducible
evidence. The ten task pairs cover:

- silicon diffusion and sodium-silicate glass transition;
- silver model fine-tuning plus silicon and copper energy regression;
- palladium and strained-silicon phonons, and aluminum vibrational density of states;
- aluminum thermal expansion and heat capacity.

Each level has its own rubric and submission examples. Level 1 examples contain
only the preparation work in their prompts. Checks for work common to both
levels use the same underlying evidence validation; the later Level 2 stages
are scored only at Level 2.

Level 1 scoring is binary. Tasks 3–10 require isolated Modal spot checks of
saved MACE energies and forces against the pinned teacher checkpoint. An
unavailable verifier leaves the score pending; it does not turn a completed
submission into a zero. Task 1 binary LAMMPS restarts are converted by the
deployed trusted reader by default; offline scoring leaves their state check
pending.

## Scoring

Start from the submission templates for
[Level 1](src/corral_md/submission_templates/level_1/) or
[Level 2](src/corral_md/submission_templates/level_2/).
Fill them with measured results and link the retained artifacts, settings,
scripts, and report. Blank examples are seeded into each task workspace.

To score a saved manifest and write a detailed check report:

```bash
uv run python -m corral_md.score 9 /path/to/workspace/output/manifest.json \
  --level 1 --output /tmp/corral-md-score.json
```

Level 1 automatically uses the trusted restart reader for Task 1 and independent
MACE verification for Tasks 3–10. Task 2 uses local evidence checks. At Level 2,
Task 3 defaults to Modal verification; other tasks default to offline checks.
Use `--verification modal` for independent Level 2 verification, or set
`CORRAL_MD_VERIFICATION=modal` when running Level 2 through Corral. This setting
requires the deployed `simagent` app described above.

`--verification offline` explicitly disables remote checks. Checks that require
independent evidence can remain pending; the CLI exits with code 2 for a
`pending_review` report. A trusted evaluator can supply `--review review.json`;
agent-submitted review files are not discovered automatically. For controlled MD
provenance, `--run-id` and `--action-id` bind verification to the recorded run.
