# Corral MD — corral-mini v1

Corral-mini v1 reports for Corral MD: 3 tasks per level, 5 trials per task,
three model generations per family (batches 1–3, newest first). The earlier
full benchmark (10 tasks per level) is archived in
[`../archive/october-v1/corral_md/`](../archive/october-v1/corral_md/).

**Status (2026-11-07): 26 of 36 runs.** Every OpenAI and Kimi run is here, plus
Claude batch 1 at level 1. The Claude runs below are pending because the
Anthropic credits ran out; they will be added when they finish.

| Batch | Agent | Model | Level 1 | Level 2 |
|---:|---|---|---|---|
| 1 | Claude Code | claude-opus-5-5 | ✅ | 14/15 trials; 1 to rerun |
| 1 | ReAct | claude-opus-5-5 | ✅ | 11/15 trials; 4 to rerun |
| 2 | Claude Code | claude-opus-4-7 | 6/15 trials | 4/15 trials |
| 2 | ReAct | claude-opus-4-7 | 5/15 trials | 3/15 trials |
| 3 | Claude Code | claude-opus-4-5-20251101 | not started | not started |
| 3 | ReAct | claude-opus-4-5-20251101 | not started | not started |

Trials that hit the credit error were discarded, not scored, and are queued
again (107 trials in all). A run is merged only when all 15 of its trials have
finished.

## Setup

| Setting | Value |
|---|---|
| Tasks | Level 1: `level_1_task_6`, `level_1_task_8`, `level_1_task_9`; Level 2: `level_2_task_2`, `level_2_task_5`, `level_2_task_10` |
| Models | Batch 1: claude-opus-5-5, gpt-6.1-sol, kimi-k3.1-pro. Batch 2: claude-opus-4-7, gpt-5.6-sol, kimi-k2.5. Batch 3: claude-opus-4-5-20251101, gpt-5-2025-08-07, kimi-k2-0905 |
| Agents | Each family's native agent (Claude Code, Codex, OpenHands) and ReAct |
| Trials | 5 per task |
| Iteration limit | 50 (agent steps, ReAct interactions, Claude Code turns, Codex model calls) |
| Reasoning effort | `medium` (kimi-k2-0905 has none) |
| Surrender | off |
| Sandbox | Docker, 2 CPUs, 6 GB, process limit 4096, BLAS/OpenMP threads = 2 |
| Image | `corral-corral_md:mini` (OpenHands, ReAct), `:mini-claude` (Claude Code), `:mini-codex` (Codex); digests in `provenance.json` |
| Kimi provider | OpenRouter, no fallbacks: Modal (kimi-k3.1-pro), Novita (kimi-k2.5, kimi-k2-0905) |

Each trial ran as its own 1-trial `corral bench` run, so free slots could start
the next trial at once. `reports/merge_trial_runs.py` merged them into one
report per agent, model and level, rebuilt with messages and tool calls from
each trial's saved state (most trials ran without `--verbose`). Metrics are
recomputed by Corral over the merged trials.

## Results

| Batch | Level | Agent | Model | Passes | Average | Pass@5 | Pass^5 | Per task (passes / 5) | Avg. duration | Input / output tokens |
|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 1 | Claude Code | claude-opus-5-5 | 14/15 | 0.93 | 1.00 | 0.78 | 6: 4, 8: 5, 9: 5 | 32 min | 1.3M / 59k |
| 1 | 1 | ReAct | claude-opus-5-5 | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 24 min | 1.9M / 196k |
| 1 | 1 | Codex | gpt-6.1-sol | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 18 min | 8.7M / 100k |
| 1 | 1 | ReAct | gpt-6.1-sol | 14/15 | 0.93 | 1.00 | 0.78 | 6: 4, 8: 5, 9: 5 | 16 min | 1.1M / 44k |
| 1 | 1 | OpenHands | kimi-k3.1-pro | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 75 min | 8.2M / 90k |
| 1 | 1 | ReAct | kimi-k3.1-pro | 15/15 | 1.00 | 1.00 | 1.00 | 6: 5, 8: 5, 9: 5 | 67 min | 2.6M / 82k |
| 2 | 1 | Codex | gpt-5.6-sol | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 62 min | 6.4M / 94k |
| 2 | 1 | ReAct | gpt-5.6-sol | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 29 min | 856k / 54k |
| 2 | 1 | OpenHands | kimi-k2.5 | 11/15 | 0.73 | 1.00 | 0.24 | 6: 4, 8: 4, 9: 3 | 76 min | 9.8M / 157k |
| 2 | 1 | ReAct | kimi-k2.5 | 8/15 | 0.53 | 0.97 | 0.06 | 6: 3, 8: 3, 9: 2 | 44 min | 5.5M / 207k |
| 3 | 1 | Codex | gpt-5-2025-08-07 | 14/15 | 0.93 | 1.00 | 0.78 | 6: 4, 8: 5, 9: 5 | 52 min | 8.7M / 196k |
| 3 | 1 | ReAct | gpt-5-2025-08-07 | 4/15 | 0.27 | 0.76 | 0.00 | 6: 1, 8: 2, 9: 1 | 12 min | 4.5M / 480k |
| 3 | 1 | OpenHands | kimi-k2-0905 | 7/15 | 0.47 | 0.94 | 0.03 | 6: 3, 8: 2, 9: 2 | 93 min | 12.2M / 175k |
| 3 | 1 | ReAct | kimi-k2-0905 | 11/15 | 0.73 | 0.99 | 0.39 | 6: 3, 8: 5, 9: 3 | 56 min | 6.1M / 188k |
| 1 | 2 | Codex | gpt-6.1-sol | 10/15 | 0.67 | 0.67 | 0.67 | 2: 0, 5: 5, 10: 5 | 48 min | 9.4M / 87k |
| 1 | 2 | ReAct | gpt-6.1-sol | 10/15 | 0.67 | 0.67 | 0.67 | 2: 0, 5: 5, 10: 5 | 30 min | 1.4M / 40k |
| 1 | 2 | OpenHands | kimi-k3.1-pro | 8/15 | 0.53 | 0.66 | 0.36 | 2: 0, 5: 5, 10: 3 | 103 min | 7.6M / 81k |
| 1 | 2 | ReAct | kimi-k3.1-pro | 8/15 | 0.53 | 0.66 | 0.36 | 2: 0, 5: 5, 10: 3 | 48 min | 4.8M / 87k |
| 2 | 2 | Codex | gpt-5.6-sol | 10/15 | 0.67 | 0.67 | 0.67 | 2: 0, 5: 5, 10: 5 | 65 min | 7.9M / 86k |
| 2 | 2 | ReAct | gpt-5.6-sol | 9/15 | 0.60 | 0.67 | 0.44 | 2: 0, 5: 5, 10: 4 | 42 min | 1.3M / 60k |
| 2 | 2 | OpenHands | kimi-k2.5 | 6/15 | 0.40 | 0.64 | 0.11 | 2: 0, 5: 2, 10: 4 | 118 min | 16.0M / 176k |
| 2 | 2 | ReAct | kimi-k2.5 | 7/15 | 0.47 | 0.66 | 0.14 | 2: 0, 5: 3, 10: 4 | 64 min | 11.9M / 316k |
| 3 | 2 | Codex | gpt-5-2025-08-07 | 7/15 | 0.47 | 0.66 | 0.14 | 2: 0, 5: 3, 10: 4 | 72 min | 8.2M / 190k |
| 3 | 2 | ReAct | gpt-5-2025-08-07 | 4/15 | 0.27 | 0.61 | 0.01 | 2: 0, 5: 2, 10: 2 | 52 min | 5.6M / 498k |
| 3 | 2 | OpenHands | kimi-k2-0905 | 5/15 | 0.33 | 0.64 | 0.03 | 2: 0, 5: 2, 10: 3 | 143 min | 21.9M / 208k |
| 3 | 2 | ReAct | kimi-k2-0905 | 6/15 | 0.40 | 0.64 | 0.11 | 2: 0, 5: 2, 10: 4 | 68 min | 11.6M / 233k |

Pass@k and Pass^k are Corral's estimators (`runs.csv`). No run solved
`level_2_task_2`. The newest models nearly saturate level 1; older models are
less consistent (lower Pass^5), most of all with ReAct.

## Caveats

- **Unscorable submissions count as failures.** Three Kimi trials on
  `level_2_task_10` (OpenHands kimi-k3.1-pro k1, ReAct kimi-k3.1-pro k5,
  ReAct kimi-k2-0905 k4) submitted output the grader could not parse.
  Corral leaves such trials unscored, which leaves every metric of the run
  undefined, so the merge scored them 0 and kept the evaluation error. They
  are listed under `unscorable_scored_as_zero` in `provenance.json`.
- **Codex iteration limit is approximate.** Corral counts Codex model calls
  from its token-usage updates and interrupts at 50; a tool started by the
  last call can still finish.
- **Langfuse sessions.** Each trial is its own session, named by its
  single-trial run ID (`<agent>__<model>__L<level>__<task>__k<k>`); the
  `langfuse_session_url` in `runs.csv` (the merged run ID) has no traces.
  Trace IDs follow the usual formula; use `trials.csv` for trace links.
- **Thread limit.** An earlier attempt hit the sandbox's thread limit
  (BLAS sized to the host's 64 cores under a 256-process cap) and was
  discarded. These runs use 2 BLAS/OpenMP threads, a 4096-process limit, and
  the `terminal` tool forwards the thread settings (#468).
- **API credit outages.** Anthropic credits ran out three times and OpenAI
  credits once (about 6 minutes). Trials with a credit error were discarded
  and rerun; this includes one Kimi ReAct trial, because `visualisation_tool`
  calls OpenAI (`gpt-4.1`) for every agent. The merged reports were scanned
  for credit errors in messages and tool outputs; none remain.
- **Durations** depend on host load: up to about 36 trials shared the host,
  and the GPUs were shared by each model family's trials.
