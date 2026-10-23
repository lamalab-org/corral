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
simulation ends after the hold. Task 2 is scored by `scoring/task_2.py` using only
`results` and `artifacts.raw_logs` (an ordered list of the final calculation's
logs). Return that JSON directly as the final answer. No submission
examples, settings files, analysis documents, exported traces, or boundary
snapshots are required. The simulation tool already retains raw logs; keep
command echo enabled and record `Step Temp Press Density` at least every
1000 steps, including all run endpoints and completion lines.

The scorer checks the supplied melt, BKS coefficients, PPPM, schedule, damping,
completed step counts, and continuity of adjacent logged runs. It computes the
mean hold temperature, mean density, and density slope over steps 390,000–470,000
on a 1 ps grid. These hold diagnostics are derived at both levels; optional
reported values are checked. It also reconstructs measured hold pressure in
atmospheres (LAMMPS `real` units), requiring consistency with zero pressure,
bounded sampling error, and stable half-run means and fitted drift. Sampling
errors use the largest independent-sample, autocorrelation, or block-mean error.
The pressure error limit is 0.1 GPa converted to atmospheres; the target allowance
is three standard errors plus 0.05 GPa. The final 75% and 50% of the hold-analysis
window must also support the pressure target and precision, with error limits
scaled for their shorter lengths. Level 1 needs only the existing logs. Hold
checks assess settling, not glass equilibrium; no stationarity test is applied
to the cooling ramp.

Level 2 requires only a finite numerical `cooling_tg_K` in kelvin. The transition
status is derived; an optional `transition` claim must be `"resolved"`. The independent analysis excludes measured
T > 2600 K, averages in 5 ps blocks, and compares continuous two-slope and
single-line fits over 300–2600, 400–2500, and 500–2400 K. A resolved estimate
requires a BIC gain of at least 6 in each window, profile widths at most 200 K,
and window estimates within 150 K. Reported Tg must agree within 100 K.
An unresolved transition or a missing, null, or nonnumerical Tg receives zero
credit, even with adequately sampled, low-noise data. A reported number cannot
pass unless the retained data independently support a resolved transition.
Missing or excessively noisy data also fail. There is no fixed target Tg and
no scientific-review queue for Task 2.

These numerical screens are provisional evaluator policy. They still need
calibration against validated trajectories across seeds before benchmark use.
Saved logs establish the recorded protocol and continuity, but do not provide
independent protection against fabricated logs. No simulation is rerun during
scoring.

Rebuild local assets or upload the new versioned structures volume and redeploy
the Modal runtime before launching trials against this protocol. Earlier
checkpoint trajectories cannot satisfy the restored full-range task by rescoring.

Task 6 uses `scoring/task_6.py` for both levels. Submit `results` and
`artifacts.run_log`, pointing to the final calculation's JSON-lines log.
The agent implements the ASE dynamics and writes start, stage, sample, and end
records using the field names and units in the task's submission instructions.
Initialize velocities once and retain continuous equilibration followed by
optional VelocityVerlet NVE production, including restartable endpoints.

Level 1 needs only the equilibration log. Level 2 reports two scalar VDOS
summaries in THz: `mean_frequency_THz` and `std_frequency_THz`, the VDOS-weighted
mean and standard deviation. The evaluator also derives `frequency_95pct_THz`,
the first frequency enclosing 95% of the integrated spectral weight. No spectrum arrays
or separate spectrum JSON file are required.

The spectral convention is fixed: subtract the instantaneous mean atomic
velocity, then compute a one-sided Welch density for each atom/component using
a periodic Hann window, constant detrending, arithmetic segment averaging,
`nperseg = ceil(4000 / dt_fs)`, `noverlap = floor(nperseg/2)`, and
`nfft = nperseg`, with sampling frequency `1000/dt_fs` in THz. Here `dt_fs` is
the recorded production `interval * timestep_fs`, checked against saved times. Sum component
spectra and normalize to unit area by trapezoidal integration. Use no additional
smoothing or zero padding. Compute the weighted mean and variance by trapezoidal
integration; obtain the 95% frequency by linear interpolation at the first
crossing of the cumulative trapezoidal integral.

The evaluator reconstructs both temperatures, `sampling_interval_fs`,
`frequency_resolution_THz`, `energy_drift_eV_atom_ps`, and `energy_span_eV_atom`;
these scalar diagnostics are optional and checked when supplied.
Temperatures use 3N kinetic degrees of freedom. Energy drift is the least-squares
slope of total energy per atom against time in ps; the span is maximum minus
minimum. Summary answers must agree within 0.01 THz plus 1% of the recomputed value.

The scorer checks the recorded model digest, geometry, complete stage sequence,
boundary continuity, fixed cell, momentum/motion consistency, thermal and energy
settling, and NVE energy conservation. It reconstructs the Hann-Welch spectrum
and checks integrated frequency bands across four contiguous, nearly equal
trajectory blocks.
Policy screens require at least 2 ps and 32 equilibration samples; integration
steps at most 2 fs; production sampling at most 10 fs; and at least `4*nperseg`
production samples. The fixed effective resolution is `2000/(dt_fs*nperseg)` THz,
at most 0.5 THz. Energy span and cumulative drift must remain within 2% of mean
kinetic energy. Normalized spectral band L1 differences between each block and
the full run are limited to 0.25. Terminal equilibration block temperatures must
settle near 300 K, with standard error at most 10 K.

At both levels, the final half of equilibration must also settle in each kinetic
direction and in potential and total energy. Directional temperatures must be
within 60 K of 300 K, with standard errors at most 15 K and four-block mean spans
at most 90 K. Energy per atom must have standard error at most `0.15*kB*300 K`,
with four-block mean span and fitted change at most `0.6*kB*300 K`. Discarding the
first half of that analysis window may change its energy mean by at most
`0.3*kB*300 K`. These errors account for autocorrelation and block fluctuations.
The final equilibration geometry is screened for overlapping atoms. All these
diagnostics come from the existing log; no new submission fields are required.

The local or Modal verifier independently recalculates pinned-model potential
energies at both endpoints and up to three additional evaluator-selected saved
frames per stage. Mismatched energies fail; unavailable verification, including
offline mode, leaves the score pending. No agent-supplied program is executed.

These Task 6 thresholds need calibration against validated MACE runs before
benchmark use. Scoring analyzes saved data and performs single-frame model checks;
it does not rerun MD. These checks do not authenticate the entire trajectory.

## Task 7: aluminum thermal expansion

[Task7Scorer](src/corral_md/scoring/task_7.py) reads an answer JSON containing
`results` and `artifacts.run_log`. The agent writes one JSON-lines log containing
thermal samples and stage-boundary states. The task instructions specify the
event fields, units, and continuity requirements. Level 1 needs only the 300 K
reference stage; its mean temperature, pressure, and volume are derived from
all samples strictly after the declared equilibration. No new submission fields
or artifacts are needed for the reference convergence checks.

Level 2 requires only `results.beta_volume` in K⁻¹. It recomputes the interval
coefficient from all heating stages, the local
400 K coefficient from heating targets within 50 K, and a pressure plane from
fixed-cell NVT calculations near the heating 400 K mean volume V*. It derives
`B_T = -V* dP/dV` and `beta_pressure = (dP/dT)/B_T`, the signed local difference
and cooling-minus-heating volume differences, and reports
uncertainties and whether the local routes agree. Neither local-route agreement
nor zero heating/cooling differences are required for credit.

The scorer checks initial FCC geometry, model digest, NPT boundary continuity,
isotropic/fixed cells, complete sampling, and kinetic pressure. Each stage
requires at least 2 ps equilibration, 3 ps production and 32 production samples
at intervals no longer than 100 fs; timesteps are at most 2 fs. At both levels,
the scorer takes the largest mean-error estimate from independent-sample,
autocorrelation, and 4-, 8-, and 16-block analyses, using blocks with at least
two samples. The resulting precision must support at least 16 effective samples,
capped by the number of distinct thermal records (compared at 1e-9 in the logged
K, A³, and GPa units to ignore floating-point roundoff). Constant sampled observables
cannot establish precision. Fixed NVT volume is a controlled quantity and has
zero sampling error, with `null` effective sample count.

Standard errors are bounded by 2% of target temperature, 0.1% of volume and
0.1 GPa; production half differences must stay within four standard errors plus
these bounds. The last 75% and last 50% of production must also remain adequately
sampled: their effective-sample minimum scales with retained sample count, and
their error limits allow the corresponding square-root precision loss. Their
means must agree with the full-production means within the original precision
bounds plus twice the quadrature sum of the mean errors. The `computed.stages`
report includes sample counts, distinct thermal counts, block errors, and later
window means and errors. Optional reported reference means are checked against
the full-production means; agent-supplied uncertainties do not relax acceptance.

Temperature means must be within 5% of target, and NPT pressure within three
standard errors plus 0.05 GPa of target. Local perturbations must bracket
400 K by 5–50 K and V* by 0.2–3%. Derivative standard errors are limited to 25%;
fit residuals, local curvature and alternate production windows screen locality
and stability. Answer matching uses 1% rounding tolerance with quantity-specific
absolute floors, separately from sampling uncertainty.

These numerical screens need calibration on validated MACE runs before benchmark
use. Routine scoring parses logs only: it does not rerun MD or attest against
fabricated logs. The model hash and recorded state continuity are consistency
checks. The scorer returns derived values and errors in its `computed` report.

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
object does not waive artifact requirements or scientific checks. Work common
to both levels uses the same evidence validation; later stages are scored only
at Level 2.

Level 1 scoring is binary. Tasks 3–6 and 8–10 require independent local or Modal
checks of saved MACE energies and, where recorded, forces against the pinned
teacher checkpoint. An unavailable verifier leaves the score pending; it does not turn a completed
submission into a zero. Task 1 is evaluated directly from native LAMMPS records.

## Scoring

Scorers, shared evidence checks, verification clients, and the evaluator CLI
live in [`src/corral_md/scoring/`](src/corral_md/scoring/). Task 2 uses
[`task_2.py`](src/corral_md/scoring/task_2.py) at both levels and can be scored
with `python -m corral_md.scoring 2 /path/to/answer.json --level 2`.

Submit the `artifacts` links and main scientific answers in `results` specified
by each task. Agents implement the calculations and save the evidence themselves.
Each task's submission format contains only the final-answer JSON Schema and
instructions for artifact paths under `/workspace/`. Detailed simulation and
artifact-writing instructions are omitted; no simulation or recording helper
modules are supplied.
The evaluator always reconstructs diagnostics and repeated tables and applies
the scientific checks, even when those values are omitted from the answer.
Previously required diagnostic summaries remain optional; supplied values must
still agree with the evidence.

Each Level 2 answer contains at most two required scalar quantities. Model
training tasks submit their saved calculation records and model artifacts;
the evaluator reconstructs predictions and metrics. All Level 1 submissions
contain only artifacts. Optional diagnostics are checked when supplied.

| Task | Level 1 quantities | Level 2 quantities | Required Level 2 answer beyond artifacts |
|---|---:|---:|---|
| 1 | 0 | 1 | `diffusion_m2_s` |
| 2 | 0 | 1 | `cooling_tg_K`; unresolved Tg fails |
| 3 | 0 | 0 | Dataset, trained checkpoint, bulk trajectory, and log |
| 4 | 0 | 1 | `zpe_eV_per_atom` (null for resolved instability) |
| 5 | 0 | 2 | `fit.isotropic_coefficient_eV_per_strain`, `fit.uniaxial_coefficient_eV_per_strain` |
| 6 | 0 | 2 | `mean_frequency_THz`, `std_frequency_THz` |
| 7 | 0 | 1 | `beta_volume` in K⁻¹ |
| 8 | 0 | 0 | Calculation log containing fitted pipelines and datasets |
| 9 | 0 | 0 | Four trajectory logs and regression log containing fitted models |
| 10 | 0 | 1 | `heat_capacity.eV_per_atom_K` |

Task 4 uses [its own scorer](src/corral_md/scoring/task_4.py) at both levels.
The agent writes one `phonons.npz` archive containing the final geometry,
central force calculations at two displacement magnitudes, model digest,
masses, and raw force constants. Analytical Hessians can also be included.
The scorer expects specific array names, shapes, order, and units.
Link it as `artifacts.force_log`; no settings documents, scripts, reports, or templates are
needed. Level 1 stops at usable force constants and reconstructed diagnostics.
Level 2 supplies only `results.zpe_eV_per_atom` (null for resolved instability).
Bands, DOS integration and imaginary-mode weights are reconstructed from the
force archive. Optional submitted bands/DOS and diagnostic values are checked
against the same evidence; they are not required artifacts.

The scorer reconstructs derivatives and atom indexing, checks numerical
sensitivity and raw symmetry/acoustic residuals, and independently computes
signed bands, three-mode DOS normalization, and ZPE. It refines the reciprocal
mesh using the retained constants. Small pinned-model spot checks establish
force identity; unavailable verification or unresolved numerical accuracy leaves
the score pending. The numerical screens are conservative automatic-review
limits and still need calibration against validated MACE runs before benchmarking.

Task 1 uses [a separate scorer](src/corral_md/scoring/task_1.py)
for both levels. Return a JSON object directly with only `artifacts.lammps_log` and
`artifacts.trajectory`; Level 2 also requires `results.diffusion_m2_s`.
The Task 1 `run_lammps` tool adds native recording to a copy of the input and
returns the log and trajectory paths. Run the complete cycle in one input.
Direct LAMMPS runs can retain the same data: echoed commands, thermal columns
`Step Time Temp Press Density`, and an orthogonal periodic custom dump with
`id type xu yu zu vx vy vz`. Keep matching thermal and trajectory samples at
least every 2 ps, including initialization and both stage endpoints.
No settings, scripts, converted tables, analysis summaries, separate restart
files, or submission examples are needed.

The scorer checks the heating schedule and state continuity, reconstructs
temperature/density and neighbor exchange, and stops there for Level 1.
Level 2 reconstructs the diffusion coefficient using the Einstein relation and
multi-origin MSD from the full prescribed 500 ps interval at 2500 K (1000–1500 ps
after initialization), after removing collective translation and homogeneous
cell motion. It checks thermal stability, tests shorter analysis and fit windows,
and estimates uncertainty from time blocks, checking their duration against a
dynamical correlation estimate. The reported diffusion coefficient is compared
with this independently reconstructed value using bounded uncertainty and window
spread. The MSD approach is described in the
[LAMMPS diffusion documentation](https://docs.lammps.org/Howto_diffusion.html).

Task 1 scores directly from the retained evidence and reported answer, without
an external diffusion reference range. Valid submissions receive a completed
score through both the benchmark and CLI. The repository has no validated
calibration cohort; the remaining numerical screens still need validation on
real runs before benchmarking.
Saved logs establish internal consistency, not tamper-proof execution provenance.

Task 3 uses [a separate scorer](src/corral_md/scoring/task_3.py).
Its Level 1 submission needs only the labeled dimer dataset and calculation log.
Level 2 adds the trained checkpoint and bulk trajectory. No numerical answers
are required. Energy-per-atom and force-component RMSE before/after training
and bulk diagnostics are derived from the artifacts; optional supplied values
are checked.
The agent saves teacher labels in an ASE-readable dataset and writes a JSON
log containing calculator settings. Level 2 adds optimizer-step counts,
parameter and checkpoint digests, and bulk dynamics metadata. The task
instructions define these fields and the parameter-digest encoding.

The evaluator checks all 45 reference labels, evaluates the original and trained
models independently, and calculates the energy-per-atom and force-component
RMSEs. It verifies changed model parameters and checks the trained model against
selected bulk frames. Temperature statistics, energy range, force maximum,
collective-motion-corrected displacement and minimum pair distance are derived
from the trajectory and compared with every requested answer. Model loading
stays in the isolated verifier. Missing infrastructure leaves verification
pending; a failed model calculation or inconsistent records fail scoring.
Poor fitting or physical bulk evolution is allowed. Sparse sampling or possible
integration failure requires review rather than automatically establishing a
failure of transfer. These numerical screens are not calibrated model-quality
thresholds, and recorded optimizer/MD history is not tamper-proof execution
attestation. The parameter and model-output checks are independent.

Task 5 uses [its own log-based scorer](src/corral_md/scoring/task_5.py) for both
levels. Both levels submit `artifacts.calculation_log`. Level 1 needs only the
log for its five isotropic strains. Level 2 requires all 25 isotropic/Cartesian-z
calculations but reports only `results.fit.isotropic_coefficient_eV_per_strain`
and `results.fit.uniaxial_coefficient_eV_per_strain`. The intercept, 25 energies
and fit residuals are derived;
optional `results.highest_positive_energies_eV` and `results.fit.residuals_eV`
are checked in calculation-log order.

The agent writes one JSON calculation log containing the actual structures,
masses, calculator settings, and raw Hessians or six Cartesian force pairs.
Finite differences also need half-step force checks at endpoint/reference
strains. The task instructions specify the record keys and array shapes.
Submit the log path and required coefficients directly in the JSON answer.

The scorer validates strains and fixed geometry, checks raw symmetry/acoustic
errors and displacement sensitivity, symmetrizes and projects translations,
then mass-weights and diagonalizes the Hessians. It preserves imaginary modes
and independently fits the selected positive energies by unweighted OLS.
The two coefficients and any optional intercept or tables must match the evidence; there is
no minimum R². Cached evaluator-owned Hessians check numerical accuracy, with
an isolated calculator fallback. Backend unavailability leaves that check
pending. Tolerances are numerical screens requiring calibration before
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
prepared assets and local Docker calculation worker. Tasks 1 and 2 use local
evidence checks; Tasks 3–10 use independent MACE verification where required.
Use `--verification modal` explicitly for remote scoring through the deployed
`simagent` app. Benchmark runs select their simulation and scoring backends
together through `--env-kwargs '{"local": false}'`; the scoring CLI has its own
`--verification` option.

`--verification offline` explicitly disables independent calculations. Checks that require
independent evidence can remain pending; the CLI exits with code 2 for a
`pending_review` report. A trusted evaluator can supply `--review review.json`;
agent-submitted review files are not discovered automatically.

Use `--verification local --assets-dir /path/to/corral-md-assets` to score a
saved manifest with independent calculations in the local Docker image.

Task 9 uses [task_9.py](src/corral_md/scoring/task_9.py) at both levels.
The agent writes one JSONL log per trajectory, retaining initialization,
continuous equilibration and production, and saved energies and forces.
Level 1 needs only `artifacts.run_logs.main`. Level 2 adds the three independent
run logs and `artifacts.regression_log`. No `results` object is required.
The agent implements training and validation and saves preprocessing, fitted
parameters, features, and predictions in the regression log. The submission
instructions define both log layouts and the required fitting/evaluation order.

The scorer derives production temperatures and energy correlation from the
logs. Its default correlation estimator uses the biased ACF, first crossing of
exp(-1), and 50 lags; it labels sampling-limited results as bounds. Before
interpreting correlation, every production run must pass an energy-stationarity
screen: fitted change and four-block mean span at most twice the observed energy
standard deviation, and block fluctuation scales within a factor of four.
The main run's correlation estimator is repeated on its first half, last half,
and last 75%. A numerical estimate requires at least ten estimated correlation
times of observation and window estimates within the larger of one sampling
interval or 50% of the full-run estimate. Bounds must remain supported in every
window. The report includes window estimates, their spread, and correlated
energy-mean errors; the spread is a sensitivity diagnostic, not a confidence
interval. These checks use the existing logs and add no submission fields.

At Level 2, it verifies nested training-only scaling and Ridge fits, reproduces predictions, and
calculates MAE/RMSE/R² for both validation strategies and all independent tests.
Optional correlation claims (including nondefault estimator options), temperature
summaries and metrics are checked when supplied. The task prompt specifies the
artifact fields and record order; separate settings, scripts or reports are
not required.
Validation budgets are equal; independent features and all
holdout predictions follow the final freeze in the calculation log. Poor transfer
or negative R² can pass when correctly calculated. Three evaluator-selected
frames per run receive independent teacher and (at Level 2) SOAP checks through
the local or Modal backend. Missing verification leaves a score pending. The
logs establish consistency, not authenticated proof that no unrecorded work
occurred. Thermal and correlation-resolution screens still need calibration on
validated runs before benchmark cutoffs are frozen.

### Silicon SOAP-Ridge regression (Task 8)

Both levels use [task_8.py](src/corral_md/scoring/task_8.py). Return a JSON object directly
with only `artifacts.run_log` pointing to one JSONL calculation log. No settings document, separate checkpoint,
inference script, report, or submission templates are needed.

The agent generates and labels the 100 prescribed training structures and
writes their identities, seeds, geometry, and total energies to a JSONL log.
Level 1 contains only start, training-dataset, and end records. Level 2 also
retains candidate validation fits, a freeze record containing the final and
learning-curve pipelines, and the two test datasets and their descriptors.
The task instructions specify all record fields and fitted-state layouts.
Fit and select using training data only; freeze every pipeline before generating
test labels and use test results only for the final analysis.

The evaluator replays structure generation, checks each preprocessing fit and
Ridge solution against its own training rows, and recalculates validation and
all reported test metrics. A local or Modal backend independently recalculates
all teacher labels and SOAP descriptors; unavailable verification leaves scoring
pending. No reported metrics are required. Independent, pooled-strain, per-strain
and learning-curve calculations remain required in the log; their metrics and
tables are reconstructed and included in the evaluator report. Optional
`strain_groups` and `learning_curve` tables are checked in strain/subset order,
using `strain` or `size` plus the three metrics for each entry. Energy errors use total eV.
Reporting tolerance is 0.1% plus 1e-6 eV for MAE/RMSE and absolute 1e-4 for R².
There is no required minimum prediction quality and no MD convergence check.

The log records the declared fitting and test-generation order. Its editable
contents establish internal consistency, not an attestation that the agent
never ran a separate experiment or accessed teacher labels outside this workflow.

Task 10 uses `scoring/task_10.py` for both levels. Submit only `artifacts.run_log`
and `results`, directly as JSON. The agent implements the temperature cycle,
initializes momenta once, and writes one JSONL log containing the cumulative
clock, fixed cell, positions, momenta, potential energies, and stage endpoints.
The task instructions specify start, stage, sample, and end record fields.
To resume a calculation, restore the saved state and preserve its cumulative
time, model, and seed.

Level 1 stops after the initial 300 K stage and submits only the log. Temperature,
directional temperature and energy summaries are derived. Level 2 runs targets 300, 400, 500, 600, 700, 800, 900, 300 K
continuously; its stage table is derived from the log and need not be repeated.
An optional table must contain all eight stages in order and is checked.
Means use samples strictly after equilibration; kinetic
temperatures use 3N total and N directional degrees of freedom. The scorer checks
fixed-cell continuity, seeded initialization, thermal settling, energy drift,
and block sampling precision. Each stage needs at least 2 ps equilibration,
3 ps production, and 32 production samples at intervals no longer than 100 fs.
Integration steps may be at most 2 fs.

Level 2 reports only `results.heat_capacity.eV_per_atom_K`. The evaluator derives
`kB_per_atom`, high-temperature residuals (800/900 K), the return-state energy
and nearest-neighbor differences, the intercept and reference neighbor distances.
Optional claims under their existing keys are checked. The fit uses
only the five forward 300–700 K measured temperatures, with OLS by default;
`fit.method` also supports `wls` with five positive `weights`, or `theil_sen`.
The structural metric is the production mean per-atom nearest-neighbor distance
under periodic minimum images. Return differences are signed returning minus
initial; they need not vanish.

Scoring independently checks three saved potential energies per stage against
the supplied teacher model using the local or Modal backend. Offline analysis
reports the computed answers but remains pending until model verification.
Dynamics are never rerun. Sampling screens still require calibration on validated
MACE runs across seeds; recorded continuity and sampled energy checks do not
attest every integration step or detect every possible unrecorded intervention.
