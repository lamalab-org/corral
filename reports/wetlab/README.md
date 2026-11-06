# Wetlab reports: corral-mini v1

36 runs (18 agent/model pairs × 2 levels), 3 tasks per level × k = 5,
540 trials. Layout, names and index columns follow
[`../README.md`](../README.md).

The earlier full-benchmark runs (10 tasks per level, October) are archived in
[`archive/october-full/`](archive/october-full/); `build_index.py` ignores
`archive/`.

## Tasks (corral-mini v1)

Wetlab's subset uses levels 2 and 3.

| Level | Tasks |
|---|---|
| 2 | `qualysis_lvl2_02`, `qualysis_lvl2_05`, `qualysis_lvl2_10` |
| 3 | `qualysis_lvl3_01`, `qualysis_lvl3_06`, `qualysis_lvl3_10` |

## Results (passes out of 15 per level; pass@5 / pass^5)

| Batch | Agent | Model | L2 passes | L2 pass@5 | L2 pass^5 | L3 passes | L3 pass@5 | L3 pass^5 | Errors L2/L3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 8 | 0.97 | 0.06 | 2 | 0.45 | 0.00 | 0/0 |
| 1 | react | kimi-k3.1-pro | 13 | 1.00 | 0.69 | 7 | 0.66 | 0.14 | 0/0 |
| 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 | 12 | 1.00 | 0.47 | 0/0 |
| 1 | react | claude-opus-5-5 | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 | 0/0 |
| 1 | codex | gpt-6.1-sol | 15 | 1.00 | 1.00 | 8 | 0.66 | 0.36 | 0/0 |
| 1 | react | gpt-6.1-sol | 15 | 1.00 | 1.00 | 14 | 1.00 | 0.78 | 0/0 |
| 2 | openhands | kimi-k2.5 | 4 | 0.61 | 0.01 | 0 | 0.00 | 0.00 | 3/7 |
| 2 | react | kimi-k2.5 | 3 | 0.33 | 0.03 | 0 | 0.00 | 0.00 | 0/4 |
| 2 | claude-code | claude-opus-4-7 | 14 | 1.00 | 0.78 | 5 | 0.64 | 0.03 | 0/0 |
| 2 | react | claude-opus-4-7 | 9 | 0.95 | 0.34 | 2 | 0.31 | 0.00 | 0/0 |
| 2 | codex | gpt-5.6-sol | 14 | 1.00 | 0.78 | 8 | 0.89 | 0.14 | 0/0 |
| 2 | react | gpt-5.6-sol | 13 | 1.00 | 0.55 | 11 | 0.97 | 0.45 | 0/0 |
| 3 | openhands | kimi-k2-0905 | 2 | 0.31 | 0.00 | 0 | 0.00 | 0.00 | 6/11 |
| 3 | react | kimi-k2-0905 | 4 | 0.76 | 0.00 | 1 | 0.22 | 0.00 | 3/8 |
| 3 | claude-code | claude-opus-4-5-20251101 | 7 | 0.88 | 0.05 | 8 | 0.66 | 0.36 | 0/0 |
| 3 | react | claude-opus-4-5-20251101 | 8 | 0.95 | 0.12 | 7 | 0.78 | 0.33 | 0/1 |
| 3 | codex | gpt-5-2025-08-07 | 6 | 0.86 | 0.03 | 6 | 0.64 | 0.11 | 0/8 |
| 3 | react | gpt-5-2025-08-07 | 5 | 0.56 | 0.11 | 4 | 0.55 | 0.03 | 1/1 |

279 of 540 trials pass. Errors count trials with a saved error; all of them
are agents that used their 50-step budget without submitting (see Caveats).

## Settings

All runs: k = 5, 50 iterations, reasoning effort `medium`, surrender off,
Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Kimi went through OpenRouter with `max_tokens` 32768 and
`allow_fallbacks=false`: kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on
Novita. kimi-k2-0905 ran with the provider's default reasoning (no reasoning
setting sent). Codex enforced the 50-call budget (#465; approximate, counted
from token usage).

Runs used Corral at `origin/main` `d00fa60b9` plus #468 (thread limits) and
#465 (Codex iteration budget), on branch `mini-1-psychom` (`2bc75bc8e`), with
the task version at that commit. Sandbox images were built from
`docker/wetlab.Dockerfile` (reaktoro from conda-forge); the host controller ran
from a conda environment, since reaktoro is conda-only.

## How the reports were made

Each trial ran as its own `corral bench --trials 1` run from a queue, so the
reports here are merged with `scripts/merge_trial_reports.py`, as for
psychometrics: one report per level, agent and model, metrics recomputed with
`BenchmarkResult`, and `provenance.json` listing the 15 source runs.

The runs were not started with `--verbose`. The merge rebuilds `messages` and
`tool_calls` from each trial's final state snapshot with the same code
`--verbose` uses, and checks that rebuilt tool-call counts and outcomes match
the original reports. Every trial has its messages (72 per trial on average).

**Langfuse.** All 540 trace IDs in `trials.csv` were found in their sessions
(checked by paging the project's traces for the run window). Sessions are the
single-trial run IDs; the merged run IDs in `runs.csv`
(`wetlab-mini-v1-l<L>-<agent>-<model>`) are not Langfuse sessions.

## Caveats

- **78 trials were rerun after infrastructure failures.** The Anthropic
  account ran out of credit at about 01:22 UTC on 2026-11-06, partway through
  batch 2. This invalidated all 60 claude-opus-4-5-20251101 trials (Claude
  Code and ReAct) and 17 of 30 ReAct claude-opus-4-7 trials ("Your credit
  balance is too low"). One ReAct kimi-k2-0905 trial failed with a Novita
  "upstream generic invalid request". All 78 were rerun after a top-up
  (10:14–12:23 UTC); the reports hold the reruns only.
- **Rerun traces contain both attempts.** The reruns kept their single-trial
  run IDs, so each one wrote into the same Langfuse trace as its failed
  attempt (see "Some traces mix attempts" in `../README.md`). Each failed
  attempt ends at the model call that hit the credit error (for trials
  started after the outage, the first call). The report is authoritative.
- **Remaining errors are model failures.** Every saved error is an agent
  using its 50-step budget without calling `submit_answer`: OpenHands
  "maximum iterations limit", ReAct "interaction budget", Codex "model-call
  budget". These trials made no failed tool calls and spent their steps on lab
  actions; they are kept as failures (score 0).
- **Answer files are not visible to agents.** The task JSONs ship inside the
  image under `/opt/corral` (root, mode 0700); the agent user (uid 20001)
  cannot read them. Checked inside a running container.
- **Machine load.** Batch 1 ran alongside other users' jobs (~100 of 128
  cores busy); later batches and the reruns ran on a lighter load, and part of
  the batch 3 ReAct reruns used extra slots. Wall-clock durations are not
  comparable across batches.
- **Comparison with the archived runs.** The archived runs cover all 10 tasks
  per level; compare per task only.
