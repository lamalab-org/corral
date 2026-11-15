# Psychometrics reports: corral-mini v1

28 runs (14 agent/model pairs × 2 levels), 3 tasks per level × k = 5,
420 trials. Layout, names and index columns follow
[`../README.md`](../README.md).

The earlier full-benchmark runs (10 tasks per level, October) are archived in
[`archive/october-full/`](archive/october-full/); `build_index.py` ignores
`archive/`.

## Tasks (corral-mini v1)

| Level | Tasks |
|---|---|
| 1 | `psy_l1_t02_dd_structure`, `psy_l1_t09_careless_responding`, `psy_l1_t10_item_integrity` |
| 2 | `psy_l2_t05_duplicate_records`, `psy_l2_t07_behavioral_validity`, `psy_l2_t09_adaptive_bank_choice` |

## Results (passes out of 15 per level; pass@5 / pass^5)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 6 | 0.86 | 0.03 | 2 | 0.45 | 0.00 |
| 1 | react | kimi-k3.1-pro | 5 | 0.56 | 0.11 | 1 | 0.22 | 0.00 |
| 1 | claude-code | claude-opus-5-5 | 8 | 0.66 | 0.36 | 3 | 0.53 | 0.00 |
| 1 | react | claude-opus-5-5 | 9 | 0.67 | 0.44 | 7 | 0.66 | 0.14 |
| 1 | codex | gpt-6.1-sol | 9 | 0.89 | 0.36 | 6 | 0.56 | 0.33 |
| 1 | react | gpt-6.1-sol | 5 | 0.56 | 0.11 | 7 | 0.64 | 0.34 |
| 2 | openhands | kimi-k2.5 | 4 | 0.33 | 0.11 | 0 | 0.00 | 0.00 |
| 2 | react | kimi-k2.5 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | codex | gpt-5.6-sol | 8 | 0.89 | 0.14 | 9 | 0.67 | 0.44 |
| 2 | react | gpt-5.6-sol | 9 | 0.67 | 0.44 | 11 | 0.89 | 0.67 |
| 3 | openhands | kimi-k2-0905 | 3 | 0.33 | 0.03 | 1 | 0.22 | 0.00 |
| 3 | react | kimi-k2-0905 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

**Not yet run:** claude-opus-4-7 (batch 2) and claude-opus-4-5-20251101
(batch 3), with Claude Code and ReAct. They stopped at the pre-launch smoke
test because the Anthropic account ran out of credit; they will be added in a
follow-up.

## Settings

All runs: k = 5, 50 iterations, reasoning effort `medium`, surrender off,
Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Kimi went through OpenRouter with `max_tokens` 32768 and
`allow_fallbacks=false`: kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on
Novita. Novita does not accept a reasoning setting for kimi-k2-0905, so it ran
with the provider default. Codex enforced the 50-call budget (#465;
approximate, counted from token usage).

Runs used Corral at `origin/main` `d00fa60b9` plus #468 (thread limits) and
#465 (Codex iteration budget), on branch `mini-1-psychom`, with the task
version at that commit.

## How the reports were made

Each trial ran as its own `corral bench --trials 1` run from a queue, so the
reports here are merged: `scripts/merge_trial_reports.py` groups the
single-trial runs by level, agent and model, recomputes the metrics with
`BenchmarkResult`, and writes one report per run. Each report's
`provenance.json` lists its 15 source runs and the wall-clock span.

The runs were not started with `--verbose`, so their own reports had no
`messages` or `tool_calls`. The merge rebuilds both from each trial's final
state snapshot with the same code `--verbose` uses, and checks that the
rebuilt tool-call counts and outcomes match the original reports.

**Langfuse.** Trials keep their original execution IDs, so each trial's
`langfuse_trace_url` and session (its own single-trial run ID) in
`trials.csv` resolve. The merged run IDs in `runs.csv`
(`psychometrics-mini-v1-l<L>-<agent>-<model>`) are not Langfuse sessions, so
`langfuse_session_url` there does not open a session.

## Caveats

- **No infrastructure failures, no reruns.** Every trial that ended in an
  error was a model failure and is kept:
  - ReAct with kimi-k2.5 and kimi-k2-0905 often read a whole CSV
    (`data.csv`, `holdout.csv`, `vendor_review.csv`; up to ~7 MB) with
    `read_file` and overflowed the 262k-token context. OpenRouter reported
    this either as a context-length error or only as "Provider returned
    error"; in every such case the last call before the error was a raw CSV
    read. One ReAct kimi-k3.1-pro trial (L1 t09 trial 1) failed the same way.
    The same failure mode is noted for ReAct Kimi K3 in the archived runs.
  - Several kimi-k2-0905 requests were rejected by the model ("Invalid
    request: model rejected the request").
  - Claude Code trials that ended without calling `submit_answer`, and
    exhausted iteration budgets.
- **gpt-5-2025-08-07 passed 0 of 60.** It submitted well-formed answers;
  they were wrong on substance (for example, no ordinal treatment of
  Likert items and wrong item-quality verdicts), so this is a model result.
- **Machine load.** Batch 1 ran while other users loaded the host to about
  140/128 load average; batches 2 and 3 ran on a lighter load. Wall-clock
  durations are therefore not comparable across batches.
- **Comparison with the archived runs.** The archived runs cover all 10 tasks
  per level and an earlier task version; compare per task only.
