# Resistor network reports: corral-mini v2

28 of 36 runs (18 agent/model pairs × levels 1 and 2), 3 tasks per level ×
k = 5, 420 trials. Claude and OpenAI are complete; 8 Kimi runs wait for
OpenRouter credit and will be added.
These replace the corral-mini v1 batch-1 reports. Layout, names and index
columns follow [`../README.md`](../README.md). The earlier full-benchmark v4
runs (October) are archived in [`archive/october-full/`](archive/october-full/).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 1 | `task_3`, `task_5`, `task_7` |
| 2 | `task_0`, `task_3`, `task_5` |

Level 1 `task_3` and `task_5` are also in corral-mini v1; their batch-1
trials come from the v1 runs (same task, settings and code; 48 trials).

## Results (passes out of 15 per level)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 7 | 0.86 | 0.11 | 0 | 0.00 | 0.00 |
| 1 | react | kimi-k3.1-pro | 7 | 0.94 | 0.03 | pending |  |  |
| 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 |
| 1 | react | claude-opus-5-5 | 11 | 0.99 | 0.39 | 9 | 0.97 | 0.14 |
| 1 | codex | gpt-6.1-sol | 15 | 1.00 | 1.00 | 13 | 1.00 | 0.55 |
| 1 | react | gpt-6.1-sol | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 |
| 2 | openhands | kimi-k2.5 | 0 | 0.00 | 0.00 | pending |  |  |
| 2 | react | kimi-k2.5 | pending |  |  | pending |  |  |
| 2 | claude-code | claude-opus-4-7 | 7 | 0.94 | 0.03 | 3 | 0.67 | 0.00 |
| 2 | react | claude-opus-4-7 | 7 | 0.94 | 0.03 | 5 | 0.56 | 0.11 |
| 2 | codex | gpt-5.6-sol | 13 | 1.00 | 0.69 | 4 | 0.55 | 0.03 |
| 2 | react | gpt-5.6-sol | 15 | 1.00 | 1.00 | 13 | 1.00 | 0.69 |
| 3 | openhands | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | react | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | claude-code | claude-opus-4-5-20251101 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | claude-opus-4-5-20251101 | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

188 of 420 trials pass. Per task (merged runs only):

| Level | Task | Passes |
|---|---|---:|
| 1 | `task_3` | 38/75 |
| 1 | `task_5` | 33/75 |
| 1 | `task_7` | 42/75 |
| 2 | `task_0` | 32/65 |
| 2 | `task_3` | 25/65 |
| 2 | `task_5` | 18/65 |

141 trials end with an error, kept as failures (score 0). 138 are an agent
using its 50-step budget without submitting (Claude Code 45, ReAct 41,
OpenHands 37, Codex 15). One ReAct claude-opus-5-5 trial overflowed the context (3.4M tokens),
and OpenAI flagged one ReAct gpt-5.6-sol prompt as a possible policy
violation. One ReAct gpt-5 trial was scored 0 by hand (see Caveats).

## Settings

k = 5, 50 iterations, reasoning effort `medium` (provider default for
kimi-k2-0905), surrender off, `--verbose`, Docker sandbox with 2 CPUs, 6 GB
and a 4096 PID limit per trial, `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/
`MKL_NUM_THREADS` = 2, Langfuse on. Kimi via OpenRouter (`max_tokens` 32768,
no fallbacks): kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on Novita.
Codex enforced the 50-call budget (#465).

Corral at `ad800a2a5` (`origin/main` `2ef35fb3c`); the reused v1 trials ran at
`aa2602603`, with no change to the task code in between. Sandbox images were
built from `docker/benchmark.Dockerfile` with pip dependencies pinned to the
host task venv's uv-locked versions (`PIP_CONSTRAINT`).

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's v2 trials share one session named after the merged
run ID (`resistor_network-mini-v2-l<L>-<agent>-<model>`); the reused v1 trials
stay in their v1 sessions (`resistor_network-mini-v1-...`), recorded per trial
in `run-metadata.json` (`langfuse_session_ids`). All 420 traces were found in
their sessions (`build_index.py --verify`).

## Caveats

- **Reruns.** 45 v2 trials were rerun: 44 after the provider accounts ran
  out of credit (26 Kimi, 18 OpenAI) and 1 Codex gpt-6.1-sol trial whose
  run hung. The reports hold the reruns only. 20 of the reused v1 trials are
  themselves reruns from the v1 credit outages. Rerun traces share the trace
  of their failed attempt (see "Known issue: some traces mix attempts" in
  `../README.md`).
- **One trial scored 0 by hand.** In L1 ReAct gpt-5-2025-08-07 `task_5`
  k3 the model submitted prose (its plan) instead of a circuit. Corral tried
  to open the text as a file in the workspace and failed ("File name too
  long") before the scorer ran, so no score was recorded. It is a model
  failure and is reported as 0; the original evaluation error stays in the
  trial index, and the override is recorded in the run's `provenance.json`.
  The resistor tasks now skip file-path resolution (`resolve_answer=False`),
  so a prose answer goes straight to the scorer and scores 0.
- **Sandbox CPU count.** Each sandbox is capped at 2 CPUs, but `nproc` and
  `os.cpu_count()` report all 128 host cores. Agents size their own search
  scripts from that (`multiprocessing.Pool(24)`, `NUMBA_NUM_THREADS=32`), so
  24–32 workers share 2 CPUs. This was the same for every trial.
- **Long trials.** Agents often run their own optimisation scripts in the
  sandbox, in `terminal` calls of up to 60 minutes that they choose
  themselves. `terminal` took 232 of 345 trial-hours in the first 374
  finished trials; the task's own tools took under 2 hours. Trials ran from
  a few minutes to over 8 hours.
- **Machine load.** Runs shared the machine with the spectra, retro and
  wetlab pools; wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived v4 runs cover all 10
  tasks per level; compare per task only.
