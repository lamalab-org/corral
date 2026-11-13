# Corral MD reports: corral-mini v2

4 of 36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 60 trials: batch 1
level 1 for Claude and OpenAI. The other runs are in progress or wait for
credits (see [Pending](#pending)). These replace the corral-mini v1 reports
(#469). Layout, names and index columns follow
[`../../README.md`](../../README.md). The October full-benchmark runs are in
[`../`](../).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

corral-mini v2 as corrected in #492, which reselected corral MD from the
October reports rescored in #444:

| Level | Tasks |
|---|---|
| 1 | `level_1_task_3`, `level_1_task_7`, `level_1_task_9` |
| 2 | `level_2_task_1`, `level_2_task_2`, `level_2_task_7` |

`level_1_task_9` and `level_2_task_2` are also in corral-mini v1. Their v1
trials are reused (same task code and settings), so each run gets 5 trials per
reused task from v1. In the runs below, the 20 `level_1_task_9` trials come
from v1.

## Results

Passes are out of 15 (3 tasks × 5).

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 |
|---|---|---|---:|---:|---:|---|
| 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 | pending |
| 1 | react | claude-opus-5-5 | 15 | 1.00 | 1.00 | pending |
| 1 | codex | gpt-6.1-sol | 11 | 0.89 | 0.67 | pending |
| 1 | react | gpt-6.1-sol | 14 | 1.00 | 0.78 | pending |

Passes per task:

| Run | `level_1_task_3` | `level_1_task_7` | `level_1_task_9` |
|---|---:|---:|---:|
| claude-code / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| react / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| codex / gpt-6.1-sol | 1/5 | 5/5 | 5/5 |
| react / gpt-6.1-sol | 4/5 | 5/5 | 5/5 |

All 5 failures are on `level_1_task_3`. No trial ended with an error. Each
failure fails the scorer's `reference_settings` evidence check only: the
computed energies match the passing trials, but the reported settings carry
extra metadata, such as `dispersion=D3(BJ)-PBE` instead of `True`, or a
`teacher_model` entry. They are kept as failures. Whether the check is too
strict is a question for the task owners; the trials can be rescored without
rerunning.

## Pending

Progress of all 36 runs on 2026-11-13 at 11:45 UTC. Claude and OpenAI pools are
running; Kimi waits for an OpenRouter top-up.

| Batch | Agent | Model | Level 1 | Level 2 |
|---:|---|---|---|---|
| 1 | claude-code | claude-opus-5-5 | ✅ in this PR | 10/15 done, 5 running |
| 1 | react | claude-opus-5-5 | ✅ in this PR | 10/15 done, 5 running |
| 1 | codex | gpt-6.1-sol | ✅ in this PR | 10/15 done, 5 running |
| 1 | react | gpt-6.1-sol | ✅ in this PR | 6/15 done, 1 running |
| 1 | openhands | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 1 | react | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 2 | claude-code | claude-opus-4-7 | 7/15 done, 4 running | 2/15 done |
| 2 | react | claude-opus-4-7 | 0/15 | 2/15 done |
| 2 | codex | gpt-5.6-sol | 5/15 (v1) | 5/15 (v1) |
| 2 | react | gpt-5.6-sol | 5/15 (v1) | 3/15 (v1) |
| 2 | openhands | kimi-k2.5 | 5/15 (v1) | 5/15 (v1) |
| 2 | react | kimi-k2.5 | 5/15 (v1) | 5/15 (v1) |
| 3 | claude-code | claude-opus-4-5-20251101 | 0/15 | 0/15 |
| 3 | react | claude-opus-4-5-20251101 | 0/15 | 0/15 |
| 3 | codex | gpt-5-2025-08-07 | 5/15 (v1) | 5/15 (v1) |
| 3 | react | gpt-5-2025-08-07 | 5/15 (v1) | 5/15 (v1) |
| 3 | openhands | kimi-k2-0905 | 5/15 (v1) | 5/15 (v1) |
| 3 | react | kimi-k2-0905 | 5/15 (v1) | 5/15 (v1) |

That is 205 of 540 trials done (142 of them reused from v1): Claude 108
pending and 14 running, OpenAI 90 pending and 6 running, Kimi 120 pending.
ReAct gpt-5.6-sol level 2 has 3 v1 trials because its two v1 trials refused by
OpenAI's usage policy are rerun.

## Settings

k = 5, 50 iterations, reasoning effort `medium`, surrender off, `--verbose`,
Docker sandbox with 2 CPUs, 6 GB, a 4096 PID limit and one GPU per trial
(MD runs on the GPU), `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`
= 2, Langfuse on. Claude Code waited up to 5 min for the CLI to start
(`CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=300000`).

New trials ran at `origin/main` `cbc93d22b` (#492; `src/` and
`tasks/corral_md` are unchanged in `1e70d8793`). Reused v1 trials ran at
`3cef47eac`. Sandbox images were built from the v1 corral MD Dockerfile with
pip dependencies pinned to the v1 images' versions, so v1 and v2 images have
identical package lists: `corral-corral_md:mini-v2` (ReAct, OpenHands and the
verifier), `:mini-v2-claude` and `:mini-v2-codex`.

Each trial ran as its own `corral bench --trials 1` run. The reports are
merged with `scripts/merge_trial_reports.py`: metrics are recomputed, messages
and tool calls are rebuilt from each trial's final state, and the source runs
are listed in `provenance.json`. A run is added only when all 15 trials are
done.

**Langfuse.** Each trial is its own session, named by its single-trial run ID
(`<agent>__<model>__L<level>__<task>__k<k>`). Use the trial links in
`trials.csv`; the run-level `langfuse_session_url` in `runs.csv` has no
traces. All 60 traces were found (`build_index.py --verify`).

## Caveats

- **Reruns.** 6 trials in these runs are reruns after OpenAI refused calls for
  lack of credit (Codex gpt-6.1-sol `level_1_task_7` k3–k5; ReAct gpt-6.1-sol
  `level_1_task_3` k3 and `level_1_task_7` k1, k2). The reports hold the reruns
  only. The merged reports contain no credit or usage-policy errors.
- **Machine load.** Runs shared the machine and its GPUs with other pools, so
  wall-clock durations are not comparable with other runs.
- **Comparison with the October runs.** The October runs in [`../`](../) cover
  all tasks per level; compare per task only.
