# Resistor network reports: corral-mini v2

10 of 36 runs (18 agent/model pairs × levels 1 and 2), 3 tasks per level ×
k = 5, 150 trials. The other 26 runs are still running and will be added.
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
| 1 | react | kimi-k3.1-pro | 7 | 0.94 | 0.03 | pending | | |
| 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 |
| 1 | react | claude-opus-5-5 | pending | | | pending | | |
| 1 | codex | gpt-6.1-sol | 15 | 1.00 | 1.00 | pending | | |
| 1 | react | gpt-6.1-sol | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 |
| 2 | codex | gpt-5.6-sol | 13 | 1.00 | 0.69 | pending | | |
| 2 | react | gpt-5.6-sol | 15 | 1.00 | 1.00 | pending | | |

All other agent/model pairs (batch 2 Kimi and Claude, all of batch 3) are
pending. 115 of 150 trials pass. Per task (merged runs only):

| Level | Task | Passes |
|---|---|---:|
| 1 | `task_3` | 30/35 |
| 1 | `task_5` | 26/35 |
| 1 | `task_7` | 31/35 |
| 2 | `task_0` | 10/15 |
| 2 | `task_3` | 10/15 |
| 2 | `task_5` | 8/15 |

33 trials end with an error, all an agent using its 50-step budget without
submitting; they are kept as failures (score 0).

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
in `run-metadata.json` (`langfuse_session_ids`). All 150 traces were found in
their sessions (`build_index.py --verify`).

## Caveats

- **17 trials were rerun after the provider accounts ran out of credit**
  (12 Kimi, 5 OpenAI); the reports hold the reruns only. Some reused v1
  trials are themselves reruns from the v1 credit outage. Rerun traces
  share the trace of their failed attempt (see "Known issue: some traces mix
  attempts" in `../README.md`).
- **Long trials.** Some agents run their own optimisation scripts in the
  sandbox for hours; the trial waits for them, so durations vary widely.
- **Machine load.** Runs shared the machine with the spectra, retro and
  wetlab pools; wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived v4 runs cover all 10
  tasks per level; compare per task only.
