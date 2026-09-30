# Stargazer Task Environment

Recover planetary systems from radial-velocity observations. The agent can use
`PythonREPL` and observation-only `validate_fit` diagnostics, then makes one final
`submit_answer(answer=<candidate JSON>)`. Hidden reference parameters, matching
scores, and planet counts are available only to the final scorer.

## Banks and historical results

The bundled upstream bank has two fixed levels of ten tasks each.
`create_environments` selects this bank by default. Level 1 contains two tasks
at each upstream difficulty from 1 through 5: eight single-planet systems at
difficulties 1–4 and two two-planet systems at difficulty 5. Level 2 retains its
ten tasks at difficulties 8–10. The selection version is
`stargazer-upstream-easy-l1-v3`; see
[data/selection_manifest.json](data/selection_manifest.json) for exact membership.

The revised Level 1 replaces eight tasks from the original difficulties 5–7
selection, lowering its mean planet count from 1.9 to 1.2. The new tasks come
from the same pinned upstream revision. All reference answers pass the existing
scorer, but that alone does not establish recovery from noisy data or a model's
pass rate. The completed 1/10 result applies to the original selection, not this
revision. Its records remain bundled; replay it with
`selector_path="tasks/stargazer/data/selectors/level_1_original.json"` from the
repository root. Record the selection and bank hash when comparing runs.

The separately versioned recovery bank uses the existing private RV generator,
longer observation baselines, and white measurement noise. Its difficulty labels
refer to this new distribution and do not imply equivalence to upstream tiers.
Scores from the two banks are not directly comparable. Record bank, source, and
protocol hashes with every comparison.

See [recovery-validation.md](recovery-validation.md) for the prepared bank's
calibration evidence, frozen identities, and Docker verification results.

## Explicit Level 1 comparison

Build the Docker image from the repository root:

```bash
docker build -f docker/stargazer.Dockerfile -t corral-stargazer:recovery-v2 .
```

The dedicated launcher requires a frozen evaluation bank mounted read-only at
`/corral-private/0`. Its configuration pins the evaluation and calibration
hashes, protocol, complete scoring criteria, and ten Level 1 tasks. It verifies
both bank contents and resolved execution definitions before model calls. A
bundled bank, old audit, changed manifest, or wrong configuration fails startup.

Verify the prepared bank without calling a model:

```bash
docker run --rm --network none \
  --mount type=bind,src="$PWD/.corral/stargazer-recovery-v2/evaluation",dst=/corral-private/0,readonly \
  corral-stargazer:recovery-v2 python -m stargazer.launch \
  --config /opt/corral/tasks/stargazer/recovery-l1.json --check-only
```

For a fresh comparison, use the same mount and configuration, a fresh workspace
volume, the existing worker capabilities, and the model/Langfuse environment:

```bash
docker run --rm --cap-add SYS_ADMIN --cap-add SYS_CHROOT \
  --security-opt apparmor=unconfined --cpus 5 --memory 7g --pids-limit 512 \
  --env-file .env --mount type=volume,src=stargazer-recovery-v2-run,dst=/workspace \
  --mount type=bind,src="$PWD/.corral/stargazer-recovery-v2/evaluation",dst=/corral-private/0,readonly \
  corral-stargazer:recovery-v2 python -m stargazer.launch \
  --config /opt/corral/tasks/stargazer/recovery-l1.json
```

The launcher wraps `run_scripts/run_tool_calling.py` inside Docker with restricted
workers enabled. Settings are `openai/gpt-5.6-sol`, medium reasoning, k=1, five
concurrent tasks, 50 iterations, verbose output, and Langfuse. It supplies
`data_root`, `development_mode=false`, the complete scorer, and public numerical
assistance explicitly. The startup summary includes the execution fingerprint.
Running the comparison is separate from generating and verifying the bank.

For local development, install the task environment with `uv sync --locked` from
`tasks/stargazer`, then use `uv run python -m stargazer.env --level 1` to inspect
the default task definitions. Local execution requires `development_mode=true`;
scored comparisons use Docker.

## Interaction and final scoring

`PythonREPL` keeps a per-trial numerical namespace with observations, stellar
mass, reference epoch, and public numerical helpers. `validate_fit` can be called
repeatedly within the agent iteration budget. It returns only fit diagnostics
computed from observations; it never reports hidden matching or count results.
The single final `submit_answer` ends the episode and determines its score.

Use canonical candidate fields:

```json
{
  "planets": [
    {"P_days": 23.5, "m_sin_i_mjup": 0.12, "e": 0.1, "omega_rad": 1.2, "l_rad": 3.4}
  ],
  "noise_jitter_ms": 0.2
}
```

Periods are in days, masses in Jupiter masses, and angles in radians. `l_rad` is
mean longitude at the first observation time. The optional `inc_rad` and
`Omega_rad` fields and legacy aliases remain accepted. The evaluator refits one
constant velocity offset per instrument. `stargazer_planet_from_fit` converts
fitted amplitude and phase into the native schema.

The complete scorer requires:

1. Positive BIC improvement per observation over the constant model.
2. Residual RMS at most 1.5 times the median measurement uncertainty.
3. Mean physical match score at least 0.8.
4. Correct planet count and complete matching, with no unmatched planets.

There is no additional individual-planet score threshold in the default scorer.
The legacy scorer is available for offline comparisons. Hidden-feedback
`submit_action` is no longer part of the agent interface.

The inherited jitter/BIC conventions remain: omitted jitter is estimated from
residuals, the null model uses zero jitter, and jitter is not charged an extra
BIC parameter. These conventions and the scoring formula are unchanged.

### Compact diagnostics and bounded periodograms

Print scalar values rather than complete diagnostic arrays:

```python
d = stargazer_diagnostics(planets)
print({'rms': d['residuals']['rms'], 'bic': d['bic']})
```

Large periodograms can exceed the worker address-space guard. Process the same
frequency grid in chunks:

```python
from scipy.signal import lombscargle
frequency = np.linspace(1 / 1000, 1 / 0.51, 200000)
time = times_days - times_days[0]
velocity = rvs_ms - np.mean(rvs_ms)
power = np.empty(frequency.size)
for start in range(0, frequency.size, 2000):
    stop = min(start + 2000, frequency.size)
    power[start:stop] = lombscargle(
        time, velocity, 2 * np.pi * frequency[start:stop], normalize=True
    )
print({'peak_period_days': float(1 / frequency[np.argmax(power)])})
```

The REPL has no per-call deadline. Its output is captured text, and matplotlib
is disallowed. Invalid optimizer bounds are analysis-code errors; the environment
does not repair model-generated code.

## Offline recovery calibration

`stargazer.identifiability` keeps coverage, amplitude, cadence, count separation,
and bootstrap checks. Candidates passing these checks receive 50 fresh noise
realizations at their original timestamps and uncertainties. Each is fitted
using observations only: neither injected periods nor the reference count
initialize or select fits. All candidates are graded privately with the exact
benchmark criteria; the lowest observed-data BIC selects the count.

Acceptance requires at least 45/50 resolved full passes. This is a screening
rule, not a 90% population guarantee. The audit records seeds, solver budgets,
count accuracy, match-score quantiles, per-start convergence, and rejection
reasons. An exhausted selected fit is unresolved and does not count as a full pass.
Exhausted competing count fits are also recorded as unresolved; this finite
search does not establish uniqueness.
Systems rejected by the initial checks skip the expensive noise audit.
Only RV-only white-noise generation is supported; correlated-noise calibration
is outside this patch.

Freeze rules on a separate calibration set before generating evaluation tasks:

```bash
python -m stargazer.bank calibrate /private/calibration --level 1 --per-level 3
python -m stargazer.bank generate /private/evaluation --level 1 --per-level 10 \
  --calibration /private/calibration
python -m stargazer.bank verify /private/evaluation
```

Run these offline commands inside the task image with a private output mount.
Use new directories: frozen banks are never overwritten. Evaluation imports
rules from a verified current calibration manifest. Attempt records are retained
if generation exhausts its budget. Private manifests, seeds, rejected candidates,
and reference systems must remain outside the public worker filesystem.

## Isolation and verification

Docker runs model-written code in Corral's unprivileged worker filesystem.
Only public observations and an opaque analysis checkpoint cross the boundary.
The controller never unpickles the checkpoint; decoding happens after privilege
dropping. Source under `/opt/corral`, private mounts, truth, controller checkpoints,
and other workspaces are excluded. Workers clear their environment and have no
network access. Their writable filesystem is limited to the trial workspace.
Previous-attempt context is disabled. These protections are unchanged.

Run the task suite inside the prepared Docker image with
`CORRAL_PERMISSION_TESTS=1`, the worker capabilities above, and the private bank
mount to include its isolation test:

```bash
python -m pytest tasks/stargazer/tests
```

The routine suite covers public/evaluator numerical parity, reference
substitution and exploit regressions, incomplete matching, bank integrity,
launch mismatches, and recovery decisions. Expensive scientific calibration and
five-worker memory measurements are retained separately as offline artifacts.
Use `python -m stargazer.experiment compare --data-root /corral-private/0
--run-root /workspace/runs --output /workspace/comparison.json` after a model run.
Include all recorded trials in the denominator, including unfinished trials.

Bundled reference verification remains available via
`python -m stargazer.audit --check`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for upstream attribution. REBOUND compatibility setup is in
[scripts/install_rebound.py](scripts/install_rebound.py); Docker builds exclude
credentials and local artifacts through the repository `.dockerignore`.
