# Retrosynthesis reports: corral-mini v2

33 of 36 runs (18 agent/model pairs × levels 2 and 3), 3 tasks per level ×
k = 5, 495 trials. Three Kimi kimi-k2-0905 runs are still running and will be
added. Layout, names and index columns follow [`../README.md`](../README.md).
The October full-benchmark runs are archived in
[`archive/october-full/`](archive/october-full/).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 2 | `make_10_lvl2`, `make_1_lvl2`, `make_6_lvl2` |
| 3 | `make_1_lvl3`, `make_5_lvl3`, `make_6_lvl3` |

No task is shared with corral-mini v1, so every trial was run for v2.

## Results (passes out of 15 per level)

| Batch | Agent | Model | L2 passes | L2 pass@5 | L2 pass^5 | L3 passes | L3 pass@5 | L3 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | react | kimi-k3.1-pro | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | claude-code | claude-opus-5-5 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | react | claude-opus-5-5 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | codex | gpt-6.1-sol | 5 | 0.33 | 0.33 | 0 | 0.00 | 0.00 |
| 1 | react | gpt-6.1-sol | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | openhands | kimi-k2.5 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | react | kimi-k2.5 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | claude-code | claude-opus-4-7 | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | react | claude-opus-4-7 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | codex | gpt-5.6-sol | 2 | 0.31 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | react | gpt-5.6-sol | 4 | 0.33 | 0.11 | 0 | 0.00 | 0.00 |
| 3 | openhands | kimi-k2-0905 | pending | | | 0 | 0.00 | 0.00 |
| 3 | react | kimi-k2-0905 | pending | | | pending | | |
| 3 | claude-code | claude-opus-4-5-20251101 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | claude-opus-4-5-20251101 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

13 of 495 trials pass, all at level 2 (`make_6_lvl2` 9, `make_10_lvl2` 4).
Most failures are submitted routes that the checker rejects; 185 trials end
with an error, almost all an agent using its 50-step budget without
submitting. One ReAct claude-opus-4-5 trial overflowed the 200k context, and
OpenAI refused one ReAct gpt-5.6-sol prompt as a possible policy violation;
both are kept as failures.

## Settings

k = 5, 50 iterations, reasoning effort `medium` (provider default for
kimi-k2-0905), surrender off, Docker sandbox with 2 CPUs, 6 GB and a 4096 PID
limit per trial, `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` =
2, Langfuse on. Kimi via OpenRouter (`max_tokens` 32768, no fallbacks):
kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on Novita. Codex enforced
the 50-call budget (#465).

Corral at `ad800a2a5` (`origin/main` `2ef35fb3c`, including #475, which runs
`verify_route` in the controller so it reaches the reaction database). The
tools query the `corral-retro-db` Postgres container through the Docker
gateway (`RETRO_DB_HOST=172.17.0.1`). Images were built from
`docker/benchmark.Dockerfile`; the OpenHands image pins pip packages to the
versions of the earlier working image (unpinned, `agent-client-protocol` 1.0
breaks the OpenHands SDK).

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's trials share one session named after the merged run
ID (`retrosynthesis-mini-v2-l<L>-<agent>-<model>`, in `run-metadata.json`);
all 495 traces were found in their sessions (`build_index.py --verify`).

## Caveats

- **75 trials were rerun after the provider accounts ran out of credit**
  (63 Kimi, 12 OpenAI); the reports hold the reruns only. Rerun traces share
  the trace of their failed attempt (see "Known issue: some traces mix
  attempts" in `../README.md`).
- **Tools.** No tool call returned a database connection error; `verify_route`
  returned route checks in the trials that called it.
- **Machine load.** Runs shared the machine with the spectra, resistor and
  wetlab pools; wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived runs cover all 10
  tasks per level; compare per task only.
