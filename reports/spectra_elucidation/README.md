# Spectra elucidation reports: corral-mini v1

12 runs (6 agent/model pairs × 2 levels), 3 tasks per level × k = 5,
180 trials. **Batch 1 only** (newest model per family); batches 2 and 3 will
be added here. Layout, names and index columns follow
[`../README.md`](../README.md).

The earlier full-benchmark runs (20 tasks per level, October) are archived in
[`archive/october-full/`](archive/october-full/); `build_index.py` ignores
`archive/`.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks (corral-mini v1)

| Level | Tasks |
|---|---|
| 1 | `10_15227_orgsyn_084_0011_sub2`, `10_15227_orgsyn_102_0086`, `33_33333_orgsyn_333_3333` |
| 2 | `10_15227_orgsyn_102_0001`, `10_15227_orgsyn_102_0086`, `55_55555_orgsyn_555_5555` |

## Results (passes out of 15 per level; pass@5 / pass^5)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 | Errors L1/L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 10 | 0.89 | 0.44 | 10 | 0.97 | 0.36 | 0/1 |
| 1 | react | kimi-k3.1-pro | 10 | 0.97 | 0.36 | 4 | 0.55 | 0.03 | 0/0 |
| 1 | claude-code | claude-opus-5-5 | 8 | 0.86 | 0.34 | 11 | 0.89 | 0.67 | 0/0 |
| 1 | react | claude-opus-5-5 | 8 | 0.86 | 0.34 | 6 | 0.56 | 0.33 | 0/0 |
| 1 | codex | gpt-6.1-sol | 12 | 1.00 | 0.47 | 15 | 1.00 | 1.00 | 0/0 |
| 1 | react | gpt-6.1-sol | 11 | 0.99 | 0.39 | 15 | 1.00 | 1.00 | 0/0 |

120 of 180 trials pass. Per task (all six routes, 30 trials each):

| Level | Task | Passes |
|---|---|---:|
| 1 | `10_15227_orgsyn_084_0011_sub2` | 17/30 |
| 1 | `10_15227_orgsyn_102_0086` | 30/30 |
| 1 | `33_33333_orgsyn_333_3333` | 12/30 |
| 2 | `10_15227_orgsyn_102_0001` | 18/30 |
| 2 | `10_15227_orgsyn_102_0086` | 28/30 |
| 2 | `55_55555_orgsyn_555_5555` | 15/30 |

Failures are wrong structures, mostly isomers of the target (for example an
oxime instead of the amide on `084_0011_sub2`, or swapped substituents on
`55_55555`). The one saved error is an OpenHands trial that ended without a
finish action.

## Settings

All runs: k = 5, 50 iterations, reasoning effort `medium`, surrender off,
`--verbose`, Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Kimi went through OpenRouter on Modal with `max_tokens` 32768 and
`allow_fallbacks=false`. Codex enforced the 50-call budget (#465).

Runs used Corral at `aa2602603` on branch `mini-1-psychom` (`origin/main`
`3f2eda1b8`, including #465, #468 and #472), with the task version at that
commit. Sandbox images were built from `docker/benchmark.Dockerfile` with two
additions:

- pip's dependencies were pinned to the host task venv's uv-locked versions
  (`PIP_CONSTRAINT`); unpinned, the build backtracked to source-only pandas and
  chroma-hnswlib and failed without a compiler;
- Node.js 22 with `isotopic-distribution`, `nmr-processing` and `openchemlib`
  in `/usr/local/share/corral-spectra-js`, which the MS and HSQC/NMR
  predictors need. No Dockerfile in the repository installs these.

## How the reports were made

Each trial ran as its own `corral bench --trials 1` run from a queue and was
merged with `scripts/merge_trial_reports.py`: one report per level, agent and
model, metrics recomputed with `BenchmarkResult`, and `provenance.json`
listing the 15 source runs. The runs were verbose; every trial has its
messages (13 per trial on average) and tool calls.

**Langfuse.** Unlike psychometrics and wetlab, these runs set
`CORRAL_LANGFUSE_SESSION_ID` (#472) to the merged run ID
(`spectra_elucidation-mini-v1-l<L>-<agent>-<model>`), so each run's 15 trials
share one session, and `langfuse_session_url` in `runs.csv` opens it.
`run-metadata.json` records the session ID for `build_index.py`. All 180
trace IDs were found in their sessions (`build_index.py --verify`).

## Caveats

- **27 gpt-6.1-sol trials (7 Codex, 20 ReAct) were rerun after an OpenAI
  outage.** The OpenAI account ran out of credit shortly after 14:00 UTC on
  2026-11-06. 14 trials failed with "You have no credits remaining" or were
  stopped mid-run when the OpenAI pools were paused. 13 ReAct trials ended
  instead with "a terminal agent run cannot record turns" after 0–3 model
  calls (all started 14:04–14:08 UTC); this is how a ReAct run reports the
  credit error, and it was first counted as a finished trial. All 27 were
  rerun after a top-up (14:23–14:40 UTC); the reports hold the reruns only.
  Kimi and Claude trials were not affected.
- **Rerun traces contain both attempts.** The reruns kept their single-trial
  run IDs, so each one wrote into the same Langfuse trace as its failed
  attempt (see "Known issue: some traces mix attempts" in `../README.md`).
  The report is authoritative.
- **Answer files are not visible to agents.** The task JSONs ship inside the
  image under `/opt/corral` (mode 0700); the agent user (uid 20001) cannot
  read them. Checked in the image.
- **Machine load.** Spectra ran alongside the resistor network and
  retrosynthesis batch 1 pools (up to 90 trials at once on 128 cores).
  Wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived runs cover all 20 tasks
  per level; compare per task only.
