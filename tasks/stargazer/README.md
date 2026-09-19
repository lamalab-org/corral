# Stargazer Task Environment

The tasks come from the Stargazer paper's radial-velocity exoplanet benchmark,
adapted to the standard Corral lifecycle. An agent receives public observations
and tools, may evaluate candidates throughout its agent iteration budget,
and is scored on its committed `submit_action` submissions.

## Task splits

The two official levels contain 10 fixed, reference-valid synthetic tasks
each. Their explicit IDs are committed in `environments/level_*/tasks_json`,
so membership never changes at runtime.

| Split | Upstream difficulty | Synthetic tasks |
| --- | --- | ---: |
| Level 1 | 5–7 | 10 |
| Level 2 | 8–10 | 10 |

Both levels allow unlimited `submit_action` calls within Corral's configured
agent iteration budget.

Each level selects tasks from the Stargazer paper with passing reference
solutions, balanced across its difficulties (4/3/3). Tasks retain their
Stargazer IDs, with no duplicates across levels. `data/selection_manifest.json`
records the paper attribution, selection criteria, and every selected ID.
Only the 20 tasks used by Levels 1–2 are bundled.

## Setup and execution

Create and synchronize the task-local Python 3.12 environment:

```bash
cd tasks/stargazer
uv venv --python 3.12
uv sync --locked
```

Inspect either benchmark level:

```bash
uv run python -m stargazer.env --level 1
uv run python -m stargazer.env --level 2
```

These commands build and list environment definitions; Corral no longer uses
a separate task HTTP server.

Run one task through the current local runtime:

```bash
uv run corral run --agent tool-calling --environment stargazer \
  --task seed15_diff5 --model openai/gpt-4o
```

For scored trials, use `corral bench` with Docker as shown below. Select a split
with `--env-kwargs '{"level": 2}'`, and use
`task_config` or `selector_path` in the same object for a custom selector.
`CORRAL_WORK_DIR` controls the workspace root.
Use `--sandbox local` only for local debugging.

For Docker execution, build the task image from the repository root (build the
base image first if `corral-benchmark:latest` is unavailable):

```bash
docker build -f docker/benchmark.Dockerfile -t corral-benchmark:latest .
docker build -f docker/stargazer.Dockerfile -t corral-stargazer:latest .
tasks/stargazer/.venv/bin/python -m corral.cli bench \
  --agent tool-calling --environment stargazer --task seed15_diff5 \
  --model openai/gpt-5.6-terra --sandbox docker \
  --sandbox-image corral-stargazer:latest --trials 1 \
  --agent-kwargs '{"reasoning_effort": "medium", "additional_drop_params": ["temperature"]}'
```

Docker runs model-written analysis in Corral's unprivileged worker filesystem.
Only public observations and an opaque analysis checkpoint cross that boundary.
The checkpoint is decoded after privilege dropping; the controller never
unpickles it. Source under `/opt/corral`, task truth, private checkpoints,
other workspaces, and the host filesystem are not mounted into the worker.
The writable worker filesystem is limited to its trial workspace; installed
OS and scientific dependencies are available read-only. The REPL worker also
clears its environment before decoding state or running analysis code.

All task images built from the repository root use the shared `.dockerignore`
to exclude credentials, virtual environments, caches, and local workspaces.
The REBOUND compatibility fix lives in
[`scripts/install_rebound.py`](scripts/install_rebound.py), which installs the
locked source after verifying its checksum and corrects REBOUND 5.1.1's x86
architecture detection on ARM. The Dockerfile calls this task setup script;
it can also be run with a local environment's Python interpreter when a C
compiler is available.

## Execution state

Like Wetlab, `StargazerEnvironment` restores disposable sessions from
`ExecutionState.environment` and returns their changes for Corral to commit.
Diagnostic history, submission counts, and the success lock persist across
resumption and branch forks. The Python namespace is checkpointed inside its
public-data-only worker, preserving arrays, functions, and numerical random
state without replaying earlier code. Checkpoints are decoded only inside the
worker and should be resumed with the same Python and scientific environment.
Task truth stays in the task definition, outside the analysis checkpoint.
Values that cannot be checkpointed, such as live generators, cause the call to
fail without committing its namespace changes.

Run the task's regression suite with:

```bash
uv run pytest
```

## Interaction and submission contract

Every trial exposes:

- `PythonREPL`, a persistent per-trial numerical namespace containing the
  observations, stellar mass, reference epoch, numerical helpers, and the
  submission history;
- `stargazer_planet_from_fit`, a helper in that namespace which converts fitted
  semi-amplitude and phase values to Stargazer's native planet fields;
- `submit_action`, which evaluates a candidate, returns the Stargazer feedback
  described below, and commits the result to the trial trajectory.

There is no separate submission allowance or difficulty-based tool-call cap.
Agents can keep calling `submit_action` within Corral's configured
`max_iterations` budget. Valid submissions are counted for history only;
rejected candidates do not advance that count. The Stargazer interaction ends
when a candidate passes all four gates; otherwise, the agent can continue until
its iteration budget is exhausted or it closes the run. A
successful committed submission determines the scientific score; Corral's
final-answer action only closes the run and does not submit or rescore another
planetary system.

`submit_action` accepts canonical JSON fields directly:

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

For migration only, the scorer continues to accept the previously supported
field aliases and nested `noise.sigma_jitter_ms`. New candidates should use the
flat canonical schema above. The public `submit_action` schema continues to
expose the legacy aliases for compatibility.

## Evaluation

Both live `submit_action` calls and the reference audit call the same
`evaluate_submission()` implementation in [`score.py`](src/stargazer/score.py).
Corral scores the committed submission trajectory: a trial scores `1.0` when
at least one submitted candidate passes all four gates:

1. BIC improvement over a per-instrument constant model is greater than zero
   per observation;
2. residual RMS is at most 1.5 times the median measurement uncertainty;
3. aggregate Hungarian-assigned physical match score is at least 0.8;
4. recovered planet count equals the hidden reference count.

### Intentional evaluator feedback

Stargazer is an iterative model-fitting benchmark with evaluator feedback, not
a blind one-shot recovery task. Every valid `submit_action` reports BIC and BIC
per observation, residual RMS and MAE with the RMS threshold, aggregate match
score with its threshold, count pass/fail, and whether the interaction is done.

The feedback intentionally also contains:

- Hungarian assignment triples of `[truth_index, guess_index, distance]`;
- the indices of unmatched reference and submitted planets;
- for each accepted match, absolute error components for log-period,
  log-amplitude, eccentricity, phase, and the sampled RV curve.

Consequently, an agent can infer the hidden reference planet count from the
assignment and unmatched-reference lists. This information and the unsigned
distance components are deliberate optimization signals inherited from the
Stargazer interaction protocol. They are part of the benchmark task definition,
not an accidental disclosure. The feedback does not return the reference
orbital parameter values or signed parameter errors.

The aggregate physical match score deliberately averages the accepted matched
pairs. Assignments beyond the released distance cutoff are omitted from that
average; planet-count agreement remains a separate required gate. Requiring
strict recovery of every weak component made the benchmark collapse to an
all-zero regime in the original evaluation, eliminating useful discrimination
between agents. Upstream keeps the stricter alternative commented beside the
mean it chose, in `evaluator.py`:

```python
score = float(np.mean(s_list)) if len(s_list)>0 else 1.0
# For the strictest pass criterion (every planet must individually clear
# the threshold rather than only the mean), swap the line above for:
#     score = float(np.min(s_list)) if len(s_list)>0 else 1.0
```

## Deliberate design decisions

The choices below are recurring review findings. They are decisions, not
defects, and changing any of them changes what this environment measures.

### Which gate carries the discrimination

A submission must pass four gates: ΔBIC, residual RMS, physical match, and
planet count. The match and count gates are the discriminating ones. ΔBIC is a
floor check that a candidate explains the data better than a constant, and it
is easy to clear: an empty `{"planets": []}` submission reaches
`delta_bic_per_point = 38.17` on `seed101_diff9` and passes that gate alone,
while failing the other three.

### Jitter and the ΔBIC comparison

`normalize_submission` fits a jitter term to the residual RMS when an agent
omits `noise_jitter_ms`, whereas the null model is scored at exactly zero
jitter, and a candidate's jitter is not counted in the BIC parameter count.
Both follow upstream: `best_constant_fit` calls
`loglike_white_jitter(rv_obs, model, sigma_obs, 0.0)`, and the candidate uses
`k = len(guesses) * 5 + n_inst`. Corral preserves that comparison rather than
making the likelihood symmetric, because changing it changes every threshold
the bundled references were validated against. Charging one BIC parameter for
candidate jitter alone would lower `delta_bic_per_point` by `log(n)/n`,
about 0.055 to 0.092 across the bundled tasks; all 20 references still pass,
so this is a protocol choice rather than a correctness fix.

### Two difficulty levels

Upstream difficulties 5-7 and 8-10 collapse into Corral's two levels to match
the framework's level structure. The two levels are not a reproduction of the
upstream difficulty ladder.

### No REPL execution deadline

`PythonREPL` has no per-call time limit, and its tool description says so. The
agent's budget is Corral's configured `max_iterations`, not wall-clock time
inside one call. A long fit is therefore a legitimate use of the budget.

### No plotting

This task's executor refuses code containing `matplotlib`, inherited from the
upstream REPL. It is a task policy, not a property of Corral's shared REPL.
The binding constraint is not the filter but the return channel: the tool
returns captured stdout, so a rendered figure has no path back to the agent.
Text output, including character-plotted series, is unrestricted. Allowing
plots would need multimodal tool results to change anything.

### Upstream defects left in place

Three upstream behaviours are knowingly unpatched. In
`corral.runtime.python_repl`, now shared by any task adopting that REPL,
`sanitize_input`'s quote scanner mis-handles apostrophes inside comments and
its leading-`python` strip is not anchored to a word boundary. In this task's
`tools.py`, `detect_shadowing_callable_conflict` rejects a name rebound to a
callable in the same cell.

Replaying 240 REPL cells from recorded runs triggered neither the scanner nor
the prefix strip; the rebind check needs namespace replay and was not
measured. Rewriting the scanner carries the largest regression risk of any
change in that module, which is why it is tracked upstream rather than
patched here.

## Task-bank audit

`python -m stargazer.audit` deterministically evaluates all 20 benchmark
reference systems through the final scorer, validates official membership, and
regenerates `data/reference_audit.json`. CI-style verification uses:

```bash
uv run python -m stargazer.audit --check
```

The committed audit records reference scores and failures of the BIC, RMS,
physical-match, and count gates. All 20 bundled reference systems pass the
unchanged thresholds. Validation checks membership counts, uniqueness, source,
difficulty, and reference scores.

## Provenance and deliberate interface differences

All task records are adapted from the Stargazer paper's benchmark. Corral uses
Stargazer's RV-only Keplerian semantics: records marked as RV-only are loaded
directly, while REBOUND records are converted in memory with their noise
realization preserved.

Compared with the upstream interaction loop, Corral uses its configured agent
iteration budget instead of a difficulty-based submission cap. `submit_action`
evaluates candidates and determines the score; Corral's final-answer action
closes the run. See `THIRD_PARTY_NOTICES.md` for attribution and licenses.
