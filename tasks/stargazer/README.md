# Stargazer Task Environment

This task adapts the Stargazer radial-velocity exoplanet benchmark to the
standard Corral lifecycle. An agent receives public observations and tools,
may request a bounded number of diagnostic candidate evaluations, and is
scored only on its final JSON answer.

## Task splits and candidate budgets

The three official levels contain 20 fixed, reference-valid synthetic tasks
each. Their explicit IDs are committed in `environments/level_*/tasks_json`,
so membership never changes at runtime.

| Split | Upstream difficulty | Synthetic tasks | Diagnostic evaluations | Final answer | Total candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| Level 1 | 1–2 | 20 | 2 | 1 | 3 |
| Level 2 | 3–6 | 20 | 4 | 1 | 5 |
| Level 3 | 7–10 | 20 | 9 | 1 | 10 |

The 20 archival tasks are available separately as the `real` challenge split
and are not included in official Levels 1–3. Their published systems do not
currently pass the unchanged observation model and thresholds, so real-split
results should be reported separately until that split is calibrated.

## Setup and server

Create and synchronize the task-local Python 3.12 environment:

```bash
cd tasks/stargazer
uv venv --python 3.12
uv sync --locked
```

Run an official level or the separate real-data split:

```bash
uv run python -m stargazer.env --level 1
uv run python -m stargazer.env --level real
```

Levels `2` and `3` are also available. The server honors `CORRAL_HOST`,
`CORRAL_PORT`, and `CORRAL_WORK_DIR`.

## Interaction and submission contract

Every trial exposes:

- `python_repl`, a persistent per-trial numerical namespace containing NumPy,
  SciPy, the observation arrays, instrument labels, stellar mass, and the
  reference epoch;
- `planet_from_fit`, which converts fitted semi-amplitude and phase values to
  Stargazer's native planet fields;
- `evaluate_candidate`, which evaluates canonical candidate arguments and
  returns redacted four-gate diagnostics without ending or scoring the trial.

Valid diagnostic evaluations consume the split allowance. Invalid JSON or
schema-invalid candidates do not. Once a diagnostic candidate passes all four
gates, further diagnostic calls are locked and the agent is instructed to
return that candidate as its final answer. Diagnostic history is never used as
a scored fallback.

The same canonical JSON works unchanged as `evaluate_candidate` arguments and
as the final answer:

```json
{
  "planets": [
    {
      "P_days": 23.5,
      "m_sin_i_mjup": 0.12,
      "e": 0.1,
      "omega_rad": 1.2,
      "l_rad": 3.4
    }
  ],
  "noise_jitter_ms": 0.2
}
```

`P_days` is in days, `m_sin_i_mjup` is in Jupiter masses, and angles are in
radians. `l_rad` is mean longitude at the first observation time. The optional
compatibility fields `inc_rad` and `Omega_rad` are accepted but are unnecessary
for the radial-velocity model. The evaluator fits one constant velocity offset
per instrument.

For migration only, the final scorer continues to accept the previously
supported field aliases and nested `noise.sigma_jitter_ms`. New candidates
should use the flat canonical schema above; aliases are not exposed by the
diagnostic tool or task prompt.

## Evaluation

Both `evaluate_candidate` and the final scorer call the same
`evaluate_submission()` implementation. A final answer scores `1.0` only when
all four gates pass:

1. BIC improvement over a per-instrument constant model is greater than zero
   per observation;
2. residual RMS is at most 1.5 times the median measurement uncertainty;
3. aggregate Hungarian-assigned physical match score is at least 0.8;
4. recovered planet count equals the hidden reference count.

Diagnostic feedback reports BIC and BIC per observation, residual RMS and MAE
with the RMS threshold, aggregate match score with its threshold, count
pass/fail, and remaining evaluations. It does not reveal hidden parameters,
truth indices, assignments, signed errors, or the true planet count.

## Task-bank audit

`python -m stargazer.audit` deterministically evaluates all 120 published
reference systems through the final scorer, validates official membership, and
regenerates `data/reference_audit.json`. CI-style verification uses:

```bash
uv run python -m stargazer.audit --check
```

The committed audit records invalid reference parameters and failures of the
BIC, RMS, physical-match, and count gates. At the current pinned revision,
76/100 synthetic references and 0/20 real references pass. The official levels
select 60 of the passing synthetic references, stratified by upstream
difficulty as evenly as the valid records allow; thresholds are not weakened
per task.

## Provenance and deliberate interface differences

The task records are copied without modification from Stargazer revision
`3f617667472061e253288c7b26f0e70f186f2dff`. Legacy synthetic records are
converted in memory from their original REBOUND signal to the current RV-only
Keplerian semantics while retaining their noise realization.

Compared with the upstream interaction loop, Corral owns the final submission:
the iterative submission action is named `evaluate_candidate`, the final
answer is the last candidate opportunity, and only that final answer affects
the benchmark score. See `THIRD_PARTY_NOTICES.md` for attribution and licenses.
