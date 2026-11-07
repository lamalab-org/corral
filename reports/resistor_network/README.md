# Resistor network reports: corral-mini v1

8 runs (4 agent/model pairs × 2 levels), 3 tasks per level × k = 5,
120 trials. **Batch 1 only** (newest model per family), **without the Claude
routes**: Claude Code and ReAct with claude-opus-5-5 are pending (the
Anthropic account ran out of credit) and will be added here, followed by
batches 2 and 3. Layout, names and index columns follow
[`../README.md`](../README.md).

The earlier full-benchmark v4 runs (October) are archived in
[`archive/october-full/`](archive/october-full/); `build_index.py` ignores
`archive/`.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks (corral-mini v1)

| Level | Tasks |
|---|---|
| 1 | `task_2`, `task_3`, `task_5` |
| 2 | `task_1`, `task_2`, `task_9` |

## Results (passes out of 15 per level; pass@5 / pass^5)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 | Errors L1/L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 5 | 0.84 | 0.01 | 0 | 0.00 | 0.00 | 10/15 |
| 1 | react | kimi-k3.1-pro | 8 | 0.97 | 0.06 | 2 | 0.31 | 0.00 | 7/13 |
| 1 | codex | gpt-6.1-sol | 15 | 1.00 | 1.00 | 8 | 0.97 | 0.06 | 0/7 |
| 1 | react | gpt-6.1-sol | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 | 0/0 |
| 1 | claude-code | claude-opus-5-5 | pending | | | pending | | | |
| 1 | react | claude-opus-5-5 | pending | | | pending | | | |

67 of 120 trials pass. Per task (four routes, 20 trials each):

| Level | Task | Passes |
|---|---|---:|
| 1 | `task_2` | 15/20 |
| 1 | `task_3` | 15/20 |
| 1 | `task_5` | 13/20 |
| 2 | `task_1` | 7/20 |
| 2 | `task_2` | 10/20 |
| 2 | `task_9` | 7/20 |

Every saved error is an agent using its 50-step budget without submitting
(25 OpenHands "maximum iterations limit", 20 ReAct "interaction budget",
7 Codex "model-call budget"); they are kept as failures (score 0).

## Settings

All runs: k = 5, 50 iterations, reasoning effort `medium`, surrender off,
`--verbose`, Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Kimi went through OpenRouter on Modal with `max_tokens` 32768 and
`allow_fallbacks=false`. Codex enforced the 50-call budget (#465).

Runs used Corral at `aa2602603` on branch `mini-1-psychom` (`origin/main`
`3f2eda1b8`, including #465, #468 and #472), with the task version at that
commit. Sandbox images were built from `docker/benchmark.Dockerfile` with pip
dependencies pinned to the host task venv's uv-locked versions
(`PIP_CONSTRAINT`); unpinned, the build backtracked to source-only pandas and
chroma-hnswlib and failed without a compiler.

## How the reports were made

Each trial ran as its own `corral bench --trials 1` run from a queue and was
merged with `scripts/merge_trial_reports.py`: one report per level, agent and
model, metrics recomputed with `BenchmarkResult`, and `provenance.json`
listing the 15 source runs. The runs were verbose; every trial has its
messages and tool calls.

**Langfuse.** The runs set `CORRAL_LANGFUSE_SESSION_ID` (#472) to the merged
run ID (`resistor_network-mini-v1-l<L>-<agent>-<model>`), so each run's 15
trials share one session and `langfuse_session_url` in `runs.csv` opens it;
`run-metadata.json` records the session ID. All 120 trace IDs were found in
their sessions (`build_index.py --verify`).

## Caveats

- **45 trials were rerun after infrastructure failures**; the reports hold the
  reruns only.
  - 43 gpt-6.1-sol trials (28 Codex, 15 ReAct): the OpenAI account ran out
    of credit shortly after 14:00 UTC on 2026-11-06. Trials failed with "You
    have no credits remaining", were stopped when the OpenAI pools were
    paused, or (ReAct) ended with "a terminal agent run cannot record turns"
    after 0–3 model calls, which is how a ReAct run reports the credit error.
    Rerun after a top-up.
  - 1 Codex trial (L2 `task_1`, k = 2) hung for about five hours with no
    progress, network connections or CPU use; it was stopped and rerun.
  - 1 ReAct kimi-k3.1-pro trial (L2 `task_1`, k = 5) ended with an
    unspecified OpenRouter/Modal "Provider returned error" after 24 calls at
    about 60k tokens per call (not a context overflow); rerun.
- **Rerun traces contain both attempts.** The reruns kept their single-trial
  run IDs, so each one wrote into the same Langfuse trace as its failed
  attempt (see "Known issue: some traces mix attempts" in `../README.md`).
  The report is authoritative.
- **Tools.** No task-tool call returned an infrastructure error (connection,
  permission or missing-module errors) in any trial.
- **Machine load.** These runs shared the machine with the spectra
  elucidation and retrosynthesis batch-1 pools (up to 90 trials at once on
  128 cores). Wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived v4 runs cover all 10
  tasks per level; compare per task only.
