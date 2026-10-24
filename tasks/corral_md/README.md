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

Runs use local simulation and local Docker verification by default. Prepare the
pinned assets once and set their location before running a task:

```bash
uv run python modal_app/setup_assets.py --output-dir "$HOME/corral-md-assets"
export CORRAL_MD_ASSETS_DIR="$HOME/corral-md-assets"
```

Install the local runtime described below, then run:

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

When Modal is explicitly selected with `--env-kwargs '{"local": false}'`,
the deployed evaluation functions are
hard-capped at 25 active containers per function. For a large benchmark, also
bound submissions from the Corral process so excess evaluations wait locally
instead of in Modal's input queue:

```bash
uv run corral bench \
  --sandbox local \
  --agent tool-calling \
  --environment corral_md \
  --env-kwargs '{"level": 2, "local": false}' \
  --model openai/gpt-5.6 \
  --max-parallel-evaluations 25 \
  --max-parallel-evaluations-by-environment '{"corral_md": 25}'
```

The environment key must be the canonical runtime name `corral_md`.

## Optional Modal setup and workspace sync

To use [Modal](https://modal.com), configure a Modal account and deploy the worker:

```bash
uv run modal setup
uv run python modal_app/setup_simagent.py
```

The second command prepares the stable `simagent` app and its assets. Corral
discovers the active app automatically; no release environment variable is
needed. See the
[Modal notes](modal_app/README.md) for deployment and recovery details.

Select this backend explicitly with `--env-kwargs '{"local": false}'` (or
`'{"level": 2, "local": false}'` for Level 2). The `--sandbox` option selects
Corral's execution isolation separately; it does not select the MD backend.

With Modal selected, Corral uploads changed workspace files before a
remote tool runs and downloads the results when it finishes successfully.
LAMMPS, GPU Python, and independent verification use Modal. CPU Python and
terminal commands use Corral's shared tool executor. Docker trials run them in
an unprivileged worker with access only to the assigned workspace. With
`--sandbox local`, they run with the current user's OS permissions.

Tool path arguments use absolute POSIX paths under `/workspace`; controller
directories never belong in prompts or saved actions. Declared assets use
`/workspace/resources/<name>/<filename>` and are read-only in file tools and
sandboxed execution. Use `list_files("/workspace/resources", recursive=True)`
to discover them, and `copy_file` when an editable copy is needed. All sandboxed
execution starts in `/workspace`, including scripts stored in subdirectories.
Programs can use ordinary relative paths or canonical absolute paths; no
workspace environment variable or per-tool working-directory option is needed.
Resources stay outside mutable snapshots and workspace uploads.

The resource catalog and tool schema changed. Start new executions after this
update; incompatible saved runs are rejected by the existing catalog checks.
Redeploy Modal with the updated source before using the new resource paths.

Remote dispatch retains the execution's pinned Modal build, storage volume,
workspace snapshot, and action ID. Interrupted remote calls remain resumable
under that same action ID instead of being recorded as completed failures.

## Run simulations on this machine

The local backend is the default. Prepare the verified asset catalogs once and
provide their location through `assets_dir` or `CORRAL_MD_ASSETS_DIR`:

```bash
cd tasks/corral_md
uv run python modal_app/setup_assets.py --output-dir "$HOME/corral-md-assets"
uv run corral bench \
  --sandbox local \
  --agent tool-calling \
  --environment corral_md \
  --task level_1_task_1 \
  --env-kwargs "{\"assets_dir\": \"$HOME/corral-md-assets\"}" \
  --model openai/gpt-5.6
```

Local mode uses the host `lmp` executable for `run_lammps` and the current
Python environment plus its CUDA GPU for `execute_python_script(use_gpu=True)`.
Install LAMMPS with the packages needed by your tasks, CUDA-enabled PyTorch,
MACE, ASE, and the other scientific dependencies on this machine. Set
`CORRAL_MD_LAMMPS_BIN` if `lmp` is not on `PATH`. The terminal and CPU Python
tools also run locally with `--sandbox local`. The simulation backend and the
Corral sandbox are separate settings: a local backend inside the Docker sandbox
receives the same read-only resource mounts and `/workspace` cwd as other tools.
For a custom Docker image, prepare or copy the verified catalogs into
`/opt/corral/assets/corral_md`, inside the controller's private `/opt/corral` tree.
An existing explicit `assets_dir` or `CORRAL_MD_ASSETS_DIR` is preserved; this
image location is the fallback when no path is configured or the configured
host path is absent in Docker.
Rebuild legacy images without `/usr/local/share/corral-md-assets`: that location
is visible through the worker's system runtime, so sandboxed MD rejects those
images even when another asset source is configured.
Unconfined `--sandbox local` has no canonical mount guarantees; its existing host
path translation remains a compatibility fallback. It cannot provide the same
filesystem isolation or source-preservation guarantees as sandboxed execution.

Local mode runs independent calculation checks in a separate Docker container
with the pinned models and potentials. Missing or invalid agent submissions
receive zero. The specialized `run_verified_md` tool and its execution
provenance receipt remain available only with Modal. Local scoring does not
send calculations to Modal.

## Bundled inputs

`structures.zip` contains the shared structure inputs, with its SHA-256 recorded
in `modal_app/assets.json`. Tasks 2 use the original sodium-silicate melt at
`/workspace/resources/md-structures-melt-liq4000/liq4000.dat`. Tasks 9 use the periodic 32-atom FCC
copper cell at `/workspace/resources/md-structures-cu-cu32/cu32.extxyz`, with lattice parameter
3.615 Å. Tasks 1 retrieve mp-149 through Materials Project using `MP_API_KEY`;
that structure is not bundled.

The Task 2 melt is extracted from `structures/melt/liq4000.dat` in the archive;
setup does not generate it or run a melting simulation. Its 34,783 bytes match
the original archive in commit `f6c8c3e60` (SHA-256
`1674c66496b0bab9eda0ce79b439ad90f9ce2097c5d5450a86965442f5cdfa63`).
Both levels provide this file as a read-only resource before the agent starts
quenching. The former `/workspace/structures/melt/liq4000.dat` location is now
the `/workspace/resources/md-structures-melt-liq4000/liq4000.dat` path above.

Task definitions under `environments/` are included in the Python wheel.
Deploying the Modal app uses the repository's `modal_app/` scripts and both ZIP
archives.

Task 2 starts from the supplied `liq4000.dat` and
restores the historical 4000→300 K cooling range, 10 K/ps rate, 10,000 fs
cooling pressure damping, and 100 ps hold at 300 K with 5000 fs pressure damping.
The integration timestep is 1 fs and thermostat damping is 100 fs, retained
from the audited implementation; the historical task prompt did not specify
those two parameters. Preserve the supplied velocities and start at step zero.

```lammps
units real
atom_style charge
boundary p p p
read_data /workspace/resources/md-structures-melt-liq4000/liq4000.dat
include /workspace/resources/md-potentials-BKS-pot/pot.mod
timestep 1.0
fix cooling all npt temp 4000 300 100 iso 0 0 10000
run 370000
unfix cooling
fix hold all npt temp 300 300 100 iso 0 0 5000
run 100000
```

Both levels include cooling and the hold: 470 ps and 470,000 steps. The
simulation ends after the hold. Task 2 is scored by `scoring/task_2.py` using the
`task2-final-quantities-v1` policy and only `results` and `artifacts.raw_logs`.
Return that JSON directly as the final answer. The logs need native thermo
tables containing `Step` (using the prescribed 1 fs timestep) or `Time` (elapsed
fs), `Temp` in kelvin, and `Density` in g/cm3. Supply the log files containing
the samples used by the relevant level. Multiple files and repeated boundary
records are supported; file order does not affect the calculation.

The scorer reads numerical samples directly, without imposing a sampling
cadence or resampling onto a fixed grid. Setup commands, source and potential
files, pressure, run completion, trajectory continuity, and settling diagnostics
do not gate the score. The simulation instructions still define the requested
calculation; the scorer evaluates its final quantities from the saved data.

Level 1 calculates mean temperature and density over elapsed times
390,000–470,000 fs. The mean temperature must be within 15 K of 300 K, and the
temperature and density samples must be finite and positive. Optional reported
`hold_temperature_K` and `hold_density_g_cm3` must agree with the calculated
means. No Tg or separate stability check is required at this level.

Level 2 requires a finite numerical `tg` in kelvin and only the cooling samples
used to fit it; hold data and reported hold diagnostics do not affect its score.
The independent analysis excludes measured T > 2600 K and fits separate low- and
high-temperature lines to the cooling samples. It selects the partition with
the smallest combined squared residuals, with at least three samples and
distinct temperatures per regime, then calculates the lines' intersection.
To support a resolved transition, the slope difference must exceed three pooled
regression standard errors (and numerical roundoff), and the intersection must
lie inside the sampled temperature range. Reported Tg must agree within 100 K.
Both the independently fitted Tg and the reported Tg must also fall within
**100 K** of the historical **1576.88 K** reference: **1476.88–1676.88 K**, inclusive.
This uses an absolute tolerance around the target recorded in
[`632800bf`'s Level 2 task definition](https://github.com/lamalab-org/corral/blob/632800bf64a3f85ddcb003181f69ce12196a6244/tasks/corral_md/environments/level_2/tasks_json/task_2.json).
The reference does not enter the fit, restrict its search, or replace an
unresolved estimate. In particular, a self-consistent fit near the 2600 K
data cutoff cannot pass. Evaluation diagnostics retain the fitted value and
the reference range, including when a resolved estimate fails this check.
There is no prescribed 5 ps averaging, three-window agreement requirement,
single-line model comparison, BIC threshold, profile-width limit, or fixed
density-residual cutoff. These fitting choices do not constrain the agent's
analysis method; the evaluator independently derives its estimate from the logs.
An unresolved transition or a missing, null, or nonnumerical Tg receives zero
credit, even with adequately sampled, low-noise data. A reported number cannot
pass unless the retained data independently support a resolved transition.
Missing data or noise that obscures the slope change also fail. There is no
scientific-review queue for Task 2. Level 1 does not require a Tg or apply this
reference check.

The reference is the recorded benchmark result; its derivation has not been
independently reproduced from a reference trajectory. The fit-resolution screens
still need calibration against validated trajectories across seeds.
Saved logs supply the numerical evidence for these quantities, but do not
provide independent protection against fabricated logs. No simulation is rerun
during scoring.

Rebuild local assets or upload the new versioned structures volume and redeploy
the Modal runtime before launching trials against this protocol.

The Level 1 Task 6 scorer, `Task6Scorer` in `scoring/task_6.py`, reads sampled
temperatures and elapsed times from `artifacts.run_log`. Direct construction
and `check_level1_workflow(6)` use the same `level1-task6-temperature-v1` policy.
It requires at least 32 samples spanning 2 ps, with gaps no larger than 100 fs.
The final half must have a mean within 30 K of 300 K, a four-block standard
error no larger than 10 K, a block-mean span no larger than 45 K, and a final
sample within 90 K of that mean. An optional reported
`equilibration_temperature_K` must agree within 1 K plus 1% of the computed mean.

Kinetic energies or momenta can supply temperatures when temperature scalars
are absent. No ensemble or thermostat metadata, initialization seed, restart
artifact, or independent energy check is required. This policy assesses the
logged thermal outcome. Level 2 uses `check_level2_workflow(6)` and the
VDOS final-quantity policy described below.

## Level 1 final-quantity scoring

The default `check_level1_workflow` factory uses policy
`level1-final-quantities-v1` for tasks **5, 7, 9 and 10**. It calculates the
requested observables from numerical logs; it does not grade atomic geometry,
seeds, thermostat settings, restart completeness, trajectory continuity, or
effective-sampling/convergence diagnostics. The submission examples retain the
numerical inputs used by each calculation.
Level 2 tasks 6, 7, 9 and 10 use the final-quantity policy described below.

- Task 5 compares the highest positive phonon energy for each of the five
  prescribed strains with fixed-input teacher-model reference values (1% relative
  tolerance plus 1e-5 eV). Reference geometry is supplied by the evaluator;
  submissions need not serialize geometry, forces or Hessians. Other modes do
  not affect the score. Derivative-only logs remain readable.
- Task 7 computes production mean temperature, pressure and volume. It retains
  final mean targets of 300 K ±15 K and 1.01325 bar ±0.05 GPa, and checks any
  explicitly reported means against the log. There is no independent absolute
  reference-volume target: this is log-derived numerical scoring, not independent
  validation of the trajectory or potential.
- Task 9 calculates potential-energy correlation from chronological uniformly
  spaced production energies. A separately retained `energy_series` artifact is
  accepted when explicitly linked in `artifacts.energy_series`. Companion
  settings and analysis files must likewise be linked as `artifacts.settings`
  and `artifacts.analysis`; neighboring files are not discovered automatically.
  An alternate energy series needs its sampling interval in the linked settings
  or log header. Linked files are included in the evidence digest.
  Time spacing is required to interpret the result in fs. The default ACF is
  unbiased; the 1/e crossing and integrated time with a self-consistent window
  factor of 5 are calculated. Explicit normalization and integration cutoffs
  are supported. A first-lag crossing is an upper bound, and no crossing is a
  lower bound. Numerical correlation claims are checked against the saved data.
- Task 10 computes mean total energy per atom and overall/directional
  temperatures. Overall temperature is derived from the three directional
  temperatures; a separately reported overall temperature is optional and must
  agree. It retains final temperature allowances of ±30 K overall and
  ±45 K per direction around 300 K. Explicitly reported means are checked
  against the production data. No independent absolute mean-energy target is
  available; no single-frame model or restart verification is required.

For tasks 7 and 10, production is identified by phase labels or the logged
equilibration cutoff. Common scalar field spellings and nested stage logs are
accepted. Unit-bearing fields distinguish bar from GPa and total from per-atom
energy. Numerical answer comparisons use the existing 1% reporting allowance.
Non-finite or contradictory requested values fail. Unsupported layouts raise
`EvaluationError` with the unsupported check and reason.

## Level 2 final-quantity scoring

The default `check_level2_workflow` factory uses `level2-final-quantities-v1`
for tasks **6, 7, 9 and 10**, at every verification-backend setting. It grades
only the requested final numerical quantities from the retained calculation.
These tasks do not require a model-verification backend or an agent rerun.
Task 1 uses the separate `level2-diffusion-quantities-v1` policy described below.
Task 3 uses `level2-task3-temperature-v1`, described below, and also needs no
model-verification backend.

| Task | Scored quantity | Numerical source |
|---|---|---|
| 6 | VDOS weighted mean and standard deviation (THz) | Trapezoidal moments of the saved spectrum; production velocities with Hann-Welch settings are also supported |
| 7 | Volumetric thermal expansion coefficient (K⁻¹) | Unweighted fit of all heating-stage mean volumes against measured mean temperatures from 300–500 K, divided by the heating 300 K mean volume |
| 9 | MAE, RMSE and R² for both validation splits and all three independent runs | Logged energy labels and predictions; when predictions are absent, reconstruct SOAP and Ridge predictions from logged structures, partitions and fitting parameters |
| 10 | Heat-capacity slope (eV/atom/K) | Fit of mean total energy per atom against measured mean temperature for forward stages 0–4; OLS, declared WLS and Theil–Sen are supported |

Numerical answer comparisons use the shared 1% relative reporting allowance,
with absolute floors of 1e-6 for VDOS frequencies/prediction metrics and 1e-8
for expansion/heat-capacity coefficients. Poor predictive accuracy, including
negative R², can pass if the final metrics are correctly calculated. Task 9
requires all five evaluation groups; aggregate metrics alone are insufficient
without numerical inputs from which to recompute them. Its energy correlation
is included as a diagnostic and does not block prediction-metric scoring.
Task 7's local expansion, pressure-response and cooling comparisons, and Task 9's
production temperatures, remain agent analyses rather than evaluator outputs;
their raw inputs are not required by the submission format.

Flat JSONL records and nested stage/sample logs are supported, including
`part: prod`, `phase: production`, and declared equilibration cutoffs. Scalars
are sufficient where they determine the requested quantity. Task 10 can derive
temperature from retained momenta. Optional provenance, seeds, thermostat
settings, force arrays, restarts, convergence screens and additional diagnostic
claims do not affect these scores. Submitted scripts and model methods are
never executed. The policy establishes numerical consistency with the saved
logs; it does not independently validate the potential or trajectories.

Use `check_level1_workflow` and `check_level2_workflow` to select the scoring
policy. The superseded workflow policies for tasks 7, 9 and 10, Level 2 task 3,
and Level 1 task 5 have been removed. `WorkflowScorer` is limited to tasks 1,
4 and 8 at both levels, task 3 at Level 1, and task 5 at Level 2.
Existing data layouts remain readable; alternate companion files must be linked
explicitly. Historical reports
remain unchanged; rescoring writes a separate report with the policy version
and source/artifact provenance.

## What the tasks cover

Level 1 prepares the structures, datasets, and initial simulation stages. Level
2 turns them into complete workflows with saved results and reproducible
evidence. The ten task pairs cover:

- silicon diffusion and sodium-silicate glass transition;
- silver model fine-tuning plus silicon and copper energy regression;
- palladium and strained-silicon phonons, and aluminum vibrational density of states;
- aluminum thermal expansion and heat capacity.

Each level has its own rubric. Level 1 requires only the preparation artifacts
and calculation logs, with no numerical answers. An empty or absent `results`
object does not waive artifact requirements or scientific checks. Each policy validates the evidence needed for its requested quantities.

Level 1 scoring is binary. The default policies for tasks 3, 4 and 8 require
independent local or Modal checks against the pinned teacher checkpoint. Task 5
uses fixed-input phonon reference values. Tasks 6, 7, 9 and 10 calculate final
quantities directly from logs. An unavailable required verifier raises an
evaluation error. Task 1 is evaluated directly from native LAMMPS records.

## Scoring

Scorers, shared evidence checks, verification clients, and the evaluator CLI
live in [`src/corral_md/scoring/`](src/corral_md/scoring/). Task 2 uses
[`task_2.py`](src/corral_md/scoring/task_2.py) at both levels and can be scored
with `python -m corral_md.scoring 2 /path/to/answer.json --level 2`.

Submit the `artifacts` links and main scientific answers in `results` specified
by each task. Agents implement the calculations and save the evidence themselves.
Each task's submission format documents the final-answer JSON and artifact paths
under `/workspace/`. Task 4 (both levels) and Task 8 (Level 1) use native ASE
trajectories with short recording instructions. They share the existing
`Evidence.trajectory` reader; no new file format or recording framework is needed.
LAMMPS tasks retain their existing native logs and dumps where supported.
Each scoring policy reconstructs its requested quantities and tables from the
retained evidence. Optional numerical claims recognized by that policy must
agree with the evidence; exploratory diagnostics outside the policy are unscored.

Each Level 2 answer contains at most two required scalar quantities. Task 8
retains datasets and fitted pipelines for prediction reconstruction; Task 9
can supply labels and predictions directly. All Level 1 submissions contain
only artifacts.

| Task | Level 1 quantities | Level 2 quantities | Required Level 2 answer beyond artifacts |
|---|---:|---:|---|
| 1 | 0 | 1 | `diffusion` in m²/s |
| 2 | 0 | 1 | `tg` in K; unresolved Tg fails |
| 3 | 0 | 0 | `md_trajectory` with saved momenta |
| 4 | 0 | 1 | `zpe` in eV/atom (null for resolved instability) |
| 5 | 0 | 2 | `fit.isotropic`, `fit.uniaxial` in eV per unit strain |
| 6 | 0 | 2 | `mean_frequency`, `std_frequency` in THz |
| 7 | 0 | 1 | `beta` in K⁻¹ |
| 8 | 0 | 0 | Calculation log containing fitted pipelines and datasets |
| 9 | 0 | 0 | Four energy logs and regression log containing validation indices and predictions |
| 10 | 0 | 1 | `heat_capacity` in eV/(atom K) |

The evaluator accepts the previous result names for saved submissions. Conflicting
old and new names are rejected. Result renaming does not convert units.

Task 3 Level 1 needs only the labeled `dataset.extxyz`. The first frame may
record `info.default_dtype` as `float32` or `float64` (default `float64`) for the
independent teacher calculation. Model, dispersion and units are fixed by the
task, so no separate settings file or repeated assertions are required. An
explicitly linked legacy `run_log` remains supported; its settings must agree
with any corresponding dataset metadata.

Task 4 uses [its own scorer](src/corral_md/scoring/task_4.py) at both levels.
The agent writes one `forces.traj` using `ase.io.trajectory.Trajectory`, calling
`atoms.get_forces()` before `writer.write(atoms)` so ASE saves calculator results.
The first frame is the undisplaced reference; subsequent frames contain both
signs of each Cartesian displacement of atom 0 at each retained magnitude.
Atom order, cell and masses remain fixed. The scorer derives displacements and
central force constants from the saved geometry and forces. The smallest
retained displacement is the default; first-frame `info.selected_displacement_A`
can select another retained magnitude without repeating its force-constant tensor.
The submission formats also document a compact numerical NPZ alternative for
analytical Hessians and displaced-force checks. Its raw atom-zero tensor is
derived from the Hessian, so no duplicate tensor is needed. Legacy explicit
force-constant tensors remain readable and are checked against the derivatives.
Link it as `artifacts.force_log`; no settings documents, scripts, reports, or templates are
needed. Level 1 stops at usable force constants and reconstructed diagnostics.
Level 2 supplies only `results.zpe` (null for resolved instability).
Bands, DOS integration and imaginary-mode weights are reconstructed from the
force frames. Optional submitted bands/DOS and diagnostic values are checked
against the same evidence; they are not required artifacts.

The scorer reconstructs derivatives and atom indexing, checks numerical
sensitivity and raw symmetry/acoustic residuals, and independently computes
signed bands, three-mode DOS normalization, and ZPE. It refines the reciprocal
mesh using the retained constants. Small pinned-model spot checks establish
force identity; unavailable verification or unresolved reciprocal-space
integration raises an evaluation error. The derivative/correction uncertainty
bound distinguishes resolved imaginary modes from numerical residuals. Its
magnitude alone does not reject a submission: Level 2 calculates the ZPE from
the selected force constants, refines the mode integral, and compares the result
with the reported value using the existing numerical tolerance. Resolved
instability still requires a null ZPE, and independent force checks still apply.

Task 8 Level 1 saves all 100 configurations in `dataset.traj`, linked as
`artifacts.run_log`. Call `atoms.get_potential_energy()` before writing each frame.
The first frame's `info` contains `seed` (nonnegative integer);
`default_dtype` is optional and defaults to `float64`. The teacher checkpoint is
pinned by the evaluator, so its digest need not be repeated. Generate the
configurations with `numpy.random.default_rng(seed)` in order. Frame order is the
row identity, and ASE stores geometry and energy without a separate JSON log.
The evaluator checks the seed/displacement schedule and independently recalculates
all teacher energies. Legacy JSON logs remain readable.

Task 8 Level 2 retains datasets, candidate settings, validation folds, selected
pipelines and learning-curve training indices. The final fit identifies its
candidate by a zero-based index instead of repeating its settings. Teacher
precision is recorded once in the training record, defaulting to `float64`.
Each pipeline digest is recorded once at the freeze stage; training sizes and prediction metrics are
derived from the retained data. Learning-curve pipelines inherit the selected
SOAP and scaling settings. Separate test-metric summaries, training-size counts
and repeated digests are optional legacy claims and must agree if supplied.
The evaluator still checks training-only validation, fitted model state and
the order of fitting, freezing and test-label generation.

Task 1 uses [a separate scorer](src/corral_md/scoring/task_1.py)
for both levels. Level 1 returns only `artifacts.lammps_log`; Level 2 additionally
requires `artifacts.trajectory` and `results.diffusion`.
The Task 1 `run_lammps` tool records native `Step Temp` thermo output at Level 1,
without adding a trajectory dump. Level 2 retains its native log and trajectory
recording. The tool works from a copy of the input and preserves the original.

Level 1 uses policy `level1-task1-temperature-v1`: read the last row of the last
native thermo table and score 1 exactly when its finite `Temp` value satisfies
`abs(Temp - 2500) < 500 K`, otherwise 0. This preserves the former endpoint
allowance. A single final sample is sufficient; the only required column is
`Temp` in kelvin. Missing or malformed final data fails instead of falling back
to an earlier valid temperature. No separate numerical answer is required.

The simulation instructions still specify heating silicon from 300 to 2500 K
with the SW potential under zero-pressure NPT. Scoring assesses only the final
recorded temperature: protocol history, seed, elapsed time, atom geometry,
trajectory continuity, density, pressure and melting diagnostics do not gate
Level 1. Legacy extra trajectory links are ignored by its numerical check.

Level 2 grades the diffusion quantity, without requiring the echoed setup,
heating history, initial crystal, velocity initialization, thermostat or potential
commands, phase classification, or thermal/density stability checks. The saved
LAMMPS log needs only `Step Time` to associate trajectory steps with physical
times. Complete dump `ITEM: TIME` headers can supply the clock instead. Matching
timestamps must agree; a constant step/time scale established by the log permits
interpolation between its rows. The task specifies metal units (Angstrom and ps),
which apply when setup is absent; explicitly incompatible units are rejected.
Native dump `ITEM: UNITS` headers identifying metal units are supported.

Retain persistent atom IDs, periodic cell bounds and unwrapped positions or
wrapped positions with image counters. Velocities, temperature, pressure and
density columns are not required for Level 2. The scorer uses the full prescribed
500 ps diffusion interval: 1000–1500 ps on the complete-cycle clock, or 0–500 ps
for a production-only clock. A log switched after setup is sufficient; no
automatic search for other workspace logs is performed.

The scorer reconstructs diffusion using the Einstein relation and multi-origin
MSD after removing collective translation and homogeneous cell motion. It checks
coverage, atom identity, unwrapping continuity and regular physical-time sampling,
tests shorter analysis and fit windows, and estimates uncertainty from time blocks
or atom groups for sparse recordings. The reported diffusion coefficient is compared
with this independently reconstructed value using bounded uncertainty and window
spread. Both the reported and reconstructed coefficients must also lie within
10% of the fixed benchmark reference, 1.74e-8 m2/s: the inclusive acceptance range
is [1.566e-8, 1.914e-8] m2/s. This target and relative tolerance come from
Level 2 Task 1 in commit `9880925185b52e006212c41878fac70cfed31e57`.
Submission uncertainty and fitting-window spread cannot widen this reference
range. The MSD approach is described in the
[LAMMPS diffusion documentation](https://docs.lammps.org/Howto_diffusion.html).

Level 2 Task 1 requires usable timed positions, a resolved and stable MSD fit,
answer consistency, and reference agreement for a score of 1. It scores directly
through both the benchmark and CLI, using the reference embedded in the evaluator; submissions cannot
override it. Level 1 stops at the end of heating. The repository has no
validated calibration cohort; the inherited 10% reference tolerance and remaining
numerical screens still need validation on real runs before benchmarking.
Saved logs establish internal consistency, not tamper-proof execution provenance.

Task 3 Level 1 uses [the reference-dataset scorer](src/corral_md/scoring/task_3.py).
It requires only the labeled dimer dataset, with optional precision metadata,
and independently checks all 45 teacher energy/force labels.

Task 3 Level 2 defaults to
[the trajectory-temperature scorer](src/corral_md/scoring/task_3_temperature.py),
policy `level2-task3-temperature-v1`, at every verification-backend setting.
Submit only `artifacts.md_trajectory`, pointing to an ASE-readable trajectory
with saved momenta. Existing submissions with that key and additional artifacts
remain accepted. No dataset, checkpoint, calculation log, timestamps, or numerical
answer is required; no model is loaded and no simulation is rerun.

This restores the temperature rule from commit
`632800bf64a3f85ddcb003181f69ce12196a6244`: calculate `Atoms.get_temperature()`
for the last 10 frames (all available frames if fewer than 10), then award 1
when their arithmetic mean is in [270, 330] K inclusive, otherwise 0. Only
floating-point roundoff of 1e-10 K is allowed at the bounds. Empty, unreadable,
or missing trajectories, missing momenta, invalid masses, and non-finite
temperatures fail. The report retains the mean, allowed range, total frame count,
and number of averaged frames.

The averaging window counts frames, not elapsed time. Dataset generation,
fine-tuning, and the 5 ps bulk run remain workflow instructions; this policy
does not validate labels, training improvement, checkpoint identity, atom count,
run duration, or MD integration. Poor transfer or structural evolution does not
independently affect the score. The benchmark and CLI both use the temperature
policy; the former Level 2 fitting and bulk-workflow scorer has been removed.

Task 5 Level 1 uses [the final-quantity scorer](src/corral_md/scoring/final_quantities.py)
for the five isotropic-strain phonon maxima described above. Level 2 uses
[the log-based reconstruction scorer](src/corral_md/scoring/task_5.py). Both levels
submit `artifacts.calculation_log`. Level 2 requires all 25 isotropic/Cartesian-z
calculations but reports only `results.fit.isotropic`
and `results.fit.uniaxial`. The intercept, 25 energies
and fit residuals are derived;
optional `results.highest_positive_energies_eV` and `results.fit.residuals_eV`
are checked in calculation-log order.

For Level 2, the agent writes one JSON calculation log containing the actual
structures, masses, calculator settings, and raw Hessians, six Cartesian force pairs, or
the complete signed mode spectrum for independent comparison.
Finite differences also need half-step force checks at endpoint/reference
strains. Row containers such as `calculations`, `results`, and `data` are accepted;
equivalent quantity names are normalized without discarding contradictory values.
Submit the log path and required coefficients directly in the JSON answer.

The scorer validates strains and fixed geometry, checks raw symmetry/acoustic
errors and displacement sensitivity, symmetrizes and projects translations,
then mass-weights and diagonalizes the Hessians. It preserves imaginary modes
and independently fits the selected positive energies by unweighted OLS.
The two coefficients and any optional intercept or tables must match the evidence; there is
no minimum R². Cached evaluator-owned Hessians check numerical accuracy, with
an isolated calculator fallback. Backend unavailability raises an evaluation
error. Tolerances are numerical screens requiring calibration before
benchmarking; saved logs alone are not execution attestation.

Return the complete JSON object directly as the final answer, with the task's
`artifacts` and `results` fields. Do not return a manifest file path. Artifact paths
must start with `/workspace/` or be relative to `/workspace/`; they are not relative
to an answer file. For example:

```json
{"artifacts": {"run_log": "/workspace/output/run.jsonl"}, "results": {}}
```

Use the artifact roles and numerical results required by the selected task.
Construct the final JSON answer directly. No manifest file is required during
the task. The evaluator reads the answer and retained artifacts when scoring.

To score a saved manifest and write a detailed check report:

```bash
uv run python -m corral_md.scoring 9 /path/to/workspace/output/manifest.json \
  --level 1 --output /tmp/corral-md-score.json
```

The scoring CLI defaults to `--verification local` for both levels, using the
prepared assets and local Docker calculation worker. Independent MACE verification
is required for Level 1 tasks 3, 4, 5 and 8, and Level 2 tasks 4, 5 and 8.
The remaining policies score retained numerical data directly.
Use `--verification modal` explicitly for remote scoring through the deployed
`simagent` app. Benchmark runs select their simulation and scoring backends
together through `--env-kwargs '{"local": false}'`; the scoring CLI has its own
`--verification` option.

Scoring has two terminal outcomes: a numerical result with `status: "complete"`,
or a framework error with `status: "evaluation_error"`. Unsupported evidence or
an unavailable required calculation raises `EvaluationError`; its diagnostic
report identifies the task, unsupported checks and reasons, and contains no
`score` field. The CLI writes that error report and exits with code 2. Scientific
failures still score zero, and zero-point diagnostics do not block a score.
There is no pending-review result, manual credit override, `review=` argument,
or `--review` CLI option. Fixing a scorer error requires a supported calculation
and rescoring the saved artifacts.

`--verification offline` disables independent calculations, so tasks that need
them raise an evaluation error. This error policy does not add support for new
log layouts or change the quantities or tolerances in each task's scoring policy.

Use `--verification local --assets-dir /path/to/corral-md-assets` to score a
saved manifest with independent calculations in the local Docker image.

Task 9 Level 1 uses [the final-quantity scorer](src/corral_md/scoring/final_quantities.py)
for energy correlation and needs only `artifacts.run_logs.main`. Level 2 uses
[the Level 2 quantity scorer](src/corral_md/scoring/level2_quantities.py) for
prediction metrics and additionally needs the three independent run logs and
`artifacts.regression_log`. No `results` object is required. See the final-quantity
sections above for accepted numerical inputs, estimators and reporting tolerances.

### Silicon SOAP-Ridge regression (Task 8)

Both levels use [task_8.py](src/corral_md/scoring/task_8.py) and return a JSON
object with `artifacts.run_log`. Level 1 links its native ASE dataset trajectory;
Level 2 links the JSONL calculation log and saved datasets/pipelines described
in the submission format. No separate settings document, inference script or
report is required.

Legacy Level 1 JSON logs remain readable as grouped training-dataset records or
metadata followed by configuration rows. A seed and identified generator can
reconstruct omitted coordinates, with any retained coordinate hashes checked.
Level 2 records the candidate validation settings, selected final and
learning-curve pipelines, their freeze digests, and the two test datasets.
Event names are not a grading requirement: identifiable record contents,
equivalent event labels, individual fit records, and linked freeze manifests
are normalized before checking the workflow. Numeric and ISO timestamps are
accepted. A dataset without an explicit filename can be identified by a unique
match to its complete energy-label vector among sibling extxyz files; ambiguous
matches require explicit links. Saved SOAP and Ridge numerical state, including
uncompressed joblib pipelines, is read without executing submitted model methods.
Fit and select using training data only; freeze every pipeline before generating
test labels and use test results only for the final analysis.

Saved-pipeline selection uses the declared MAE, RMSE, or R² objective, minimizing
errors and maximizing R². A sole retained validation metric can identify the
objective; older multi-metric records without a declaration retain their RMSE
default. RMSE values are not mandatory for MAE or R² selection. All supplied
validation metrics must still agree with reconstructed training-only folds.
Once a generator and seed reproduce every retained structure, its full-precision
coordinates are replayed for the saved-pipeline calculations. This prevents text
coordinate rounding from changing sensitive standardized SOAP features, without
relaxing metric tolerances or replacing the saved energy labels.

The evaluator replays structure generation, checks each preprocessing fit and
Ridge solution against its own training rows, and recalculates validation and
all reported test metrics. A local or Modal backend independently recalculates
all teacher labels and SOAP descriptors; unavailable verification raises an
evaluation error. No reported metrics are required. Independent, pooled-strain, per-strain
and learning-curve calculations remain required in the log; their metrics and
tables are reconstructed and included in the evaluator report. Optional
`strain_groups` and `learning_curve` tables are checked in strain/subset order,
using `strain` or `size` plus the three metrics for each entry. Energy errors use total eV.
Reporting tolerance is 0.1% plus 1e-6 eV for MAE/RMSE and absolute 1e-4 for R².
There is no required minimum prediction quality and no MD convergence check.

The log records the declared fitting and test-generation order. Its editable
contents establish internal consistency, not an attestation that the agent
never ran a separate experiment or accessed teacher labels outside this workflow.

Task 10 uses [the Level 1 final-quantity scorer](src/corral_md/scoring/final_quantities.py)
and [the Level 2 quantity scorer](src/corral_md/scoring/level2_quantities.py).
Both levels submit `artifacts.run_log` directly as JSON. Level 1 evaluates the
initial 300 K production means; Level 2 additionally requires
`results.heat_capacity`, fitted from the five forward 300–700 K
stages. The full temperature cycle remains part of the task instructions.
The quantities, accepted fit methods and numerical checks are described in the
final-quantity sections above. Neither level requires a restart or independent
model-verification backend.
