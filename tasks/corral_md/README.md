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

## Run simulations on this machine

Prepare the verified asset catalogs once, then select the local backend in the
environment arguments:

```bash
cd tasks/corral_md
uv run python modal_app/setup_assets.py --output-dir "$HOME/corral-md-assets"
uv run corral bench \
  --sandbox local \
  --agent tool-calling \
  --environment corral_md \
  --task level_1_task_1 \
  --env-kwargs "{\"local\": true, \"assets_dir\": \"$HOME/corral-md-assets\"}" \
  --model openai/gpt-5.6
```

Local mode uses the host `lmp` executable for `run_lammps` and the current
Python environment plus its CUDA GPU for `execute_python_script(use_gpu=True)`.
Install LAMMPS with the packages needed by your tasks, CUDA-enabled PyTorch,
MACE, ASE, and the other scientific dependencies on this machine. Set
`CORRAL_MD_LAMMPS_BIN` if `lmp` is not on `PATH`. The terminal and CPU Python
tools also run locally with `--sandbox local`. The asset catalogs stay outside
the writable task workspace; file tools expose them read-only. Local scripts
should use workspace-relative paths and copy needed assets into
`/workspace/input` with `copy_file` first. The LAMMPS tool maps absolute
`/workspace/...` paths in its input file to host paths before execution.

Local mode runs independent calculation checks in a separate Docker container
with the pinned models and potentials. Missing or invalid agent submissions
receive zero. The specialized `run_verified_md` tool and its execution
provenance receipt remain available only with Modal. Local scoring does not
send calculations to Modal.

## Bundled inputs

`structures.zip` contains the shared structure inputs, with its SHA-256 recorded
in `modal_app/assets.json`. Tasks 2 use the original sodium-silicate melt at
`/workspace/structures/melt/liq4000.dat`. Tasks 9 use the periodic 32-atom FCC
copper cell at `/workspace/structures/cu/cu32.extxyz`, with lattice parameter
3.615 Å. Tasks 1 retrieve mp-149 through Materials Project using `MP_API_KEY`;
that structure is not bundled.

Task definitions under `environments/` are included in the Python wheel along
with the submission templates and scoring structure archive. Deploying the
Modal app uses the repository's `modal_app/` scripts and both ZIP archives.

Task 2 protocol `task2-4000K-v3` starts from the supplied `liq4000.dat` and
restores the historical 4000→300 K cooling range, 10 K/ps rate, 10,000 fs
cooling pressure damping, and 100 ps hold at 300 K with 5000 fs pressure damping.
The integration timestep is 1 fs and thermostat damping is 100 fs, retained
from the audited implementation; the historical task prompt did not specify
those two parameters. Preserve the supplied velocities and start at step zero.

```lammps
units real
atom_style charge
boundary p p p
read_data /workspace/structures/melt/liq4000.dat
include /workspace/potentials/BKS/pot.mod
timestep 1.0
fix cooling all npt temp 4000 300 100 iso 0 0 10000
run 370000
unfix cooling
fix hold all npt temp 300 300 100 iso 0 0 5000
run 100000
```

Level 1 includes cooling and the hold: 470 ps and 470,000 steps. Level 2
adds reheating from 300 to 4000 K at 10 K/ps with 10,000 fs pressure damping,
for a complete 840 ps / 840,000-step cycle. The reheating stage retains the
Level 2 thermal-history extension; the historical task defined cooling and
the hold only. Save the initial state and both sides of every stage handoff.

Both scorers verify the pinned 4000 K source, temperature schedule, exact
step counts, damping, raw thermo samples, and state continuity. Level 1
requires the hold as well as cooling. Level 2 verifies the reported analysis.
The fit cutoff remains measured T ≤ 2600 K; it is not a simulation endpoint.
The scorer does not compare Tg with a fixed reference value. A straight
density line cannot establish a change point. Repeated analysis indices are
rejected, including overlaps between the two fitting regions.

Nearly linear noisy data are screened against a single-line fit: fewer than
six distinct measured temperatures or a BIC gain below 6 routes the transition
check to review. The gain is `n*log(SSE_single/SSE_segmented)-2*log(n)`, using
the same rows and weights and two versus four model parameters. This is an
ambiguity screen; time-correlated MD samples limit its statistical interpretation.
Level 2 also supports `method: "unresolved"`: retain the assessed indices,
reason, and reproducible single-line SSE; report null Tg and null difference.
That transition check remains pending for independent scientific review.
Reproducing a fitted breakpoint does not by itself validate a glass transition.

A full cooling/hold validation run did not establish a robust density-based
Tg. Its standalone audit artifacts were removed during report cleanup; final
benchmark results remain in the [run reports](../../reports/corral_md/README.md).
The author's input script and reference trajectory are unavailable, so
restoring the documented protocol does not establish exact reproduction of
the author's experiment or validate the historical numerical target.

Maintainers can generate full validation evidence with
`scripts/prepare_task2_original.py --lammps /path/to/pinned/lmp --output /tmp/task2`.
Use `--level 1` for cooling/hold only. With `corral_md` installed, verify it using
`scripts/score_task2_validation.py --run-dir /tmp/task2 --output /tmp/task2-score`
(and `--level 1` for a Level 1 run). These are maintainer checks, not benchmark
agent trials. The old checkpoint preparation/comparison scripts remain
historical audit tools for superseded protocol `task2-2600K-v2`; their generated
validation reports are no longer retained in `reports/corral_md/`.

Rebuild local assets or upload the new versioned structures volume and redeploy
the Modal runtime before launching trials against this protocol. Earlier
checkpoint trajectories cannot satisfy the restored full-range task by rescoring.

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

Submit the completed manifest path. A single unambiguous path in inline
backticks is also accepted, including when surrounded by explanatory text.
Answers naming different manifests remain ambiguous and are rejected.

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

Use `--verification local --assets-dir /path/to/corral-md-assets` to score a
saved manifest with independent calculations in the local Docker image.
