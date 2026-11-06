# Corral MD — corral-mini v1

Corral-mini v1 reports for Corral MD: 3 tasks per level, 5 trials per task.
The earlier full benchmark (10 tasks per level) is archived in
[`../archive/october-v1/corral_md/`](../archive/october-v1/corral_md/).

**Status (2026-11-06): batch 1, Kimi and OpenAI.** The Claude batch-1 runs
(Claude Code and ReAct with `claude-opus-5-5`) are still running and will be
added to this folder when they finish. Batches 2 and 3 (older models) follow.

## Setup

| Setting | Value |
|---|---|
| Tasks | Level 1: `level_1_task_6`, `level_1_task_8`, `level_1_task_9`; Level 2: `level_2_task_2`, `level_2_task_5`, `level_2_task_10` |
| Trials | 5 per task |
| Iteration limit | 50 (agent steps, ReAct interactions, Claude Code turns, Codex model calls) |
| Reasoning effort | `medium` |
| Surrender | off |
| Sandbox | Docker, 2 CPUs, 6 GB, process limit 4096, BLAS/OpenMP threads = 2, one GPU per model family |
| Image | `corral-corral_md:mini` (OpenHands, ReAct), `:mini-codex` (Codex); digests in `provenance.json` |
| Kimi provider | OpenRouter, pinned to Modal, no fallbacks |

Each trial ran as its own 1-trial `corral bench` run, so free slots could start
the next trial at once. `reports/merge_trial_runs.py` merged them into one
report per agent, model and level, rebuilt with messages and tool calls from
each trial's saved state (the runs did not pass `--verbose`). Metrics are
recomputed by Corral over the merged trials.

## Results

| Level | Agent | Model | Passes | Average | Pass@5 | Pass^5 | Per task (passes / 5) | Avg. duration | Input / output tokens |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | Codex | gpt-6.1-sol | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 18 min | 8.7M / 100k |
| 1 | ReAct | gpt-6.1-sol | 14/15 | 0.93 | 1.00 | 0.78 | 6: 4, 8: 5, 9: 5 | 16 min | 1.1M / 44k |
| 1 | OpenHands | kimi-k3.1-pro | 13/15 | 0.87 | 1.00 | 0.69 | 6: 3, 8: 5, 9: 5 | 75 min | 8.2M / 90k |
| 1 | ReAct | kimi-k3.1-pro | 15/15 | 1.00 | 1.00 | 1.00 | 6: 5, 8: 5, 9: 5 | 67 min | 2.6M / 82k |
| 2 | Codex | gpt-6.1-sol | 10/15 | 0.67 | 0.67 | 0.67 | 2: 0, 5: 5, 10: 5 | 48 min | 9.4M / 87k |
| 2 | ReAct | gpt-6.1-sol | 10/15 | 0.67 | 0.67 | 0.67 | 2: 0, 5: 5, 10: 5 | 30 min | 1.4M / 40k |
| 2 | OpenHands | kimi-k3.1-pro | 8/15 | 0.53 | 0.66 | 0.36 | 2: 0, 5: 5, 10: 3 | 103 min | 7.6M / 81k |
| 2 | ReAct | kimi-k3.1-pro | 8/15 | 0.53 | 0.66 | 0.36 | 2: 0, 5: 5, 10: 3 | 48 min | 4.8M / 87k |

Pass@k and Pass^k are Corral's estimators (`runs.csv`). No run solved
`level_2_task_2`. Level 1 is nearly saturated for these models; the
differences are on `level_1_task_6` and `level_2_task_10`.

## Caveats

- **Unscorable submissions count as failures.** Two Kimi trials on
  `level_2_task_10` (OpenHands k1, ReAct k5) submitted a run log the grader
  could not parse ("Cannot identify the production samples in this log").
  Corral leaves such trials unscored, which leaves every metric of the run
  undefined, so the merge scored them 0 and kept the evaluation error. They
  are listed under `unscorable_scored_as_zero` in `provenance.json`.
- One ReAct Kimi trial (`level_2_task_10`, k3) used all 50 interactions
  without submitting.
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
- **Anthropic credits** ran out during the first Claude batch-1 attempt;
  those trials were discarded and are being rerun. No Kimi or OpenAI trial
  was affected.
