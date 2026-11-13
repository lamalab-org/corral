# Corral MD reports: corral-mini v2

6 of 36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 90 trials: batch 1
level 1 for Claude and OpenAI, plus claude-code with claude-opus-4-7 on level 1
and with claude-opus-5-5 on level 2. The other runs are in progress or wait for
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
reused task from v1. In the runs below, the 20 `level_1_task_9` trials and the
5 `level_2_task_2` trials come from v1.

## Results

Passes are out of 15 (3 tasks × 5).

| Batch | Level | Agent | Model | Passes | pass@5 | pass^5 |
|---|---:|---|---|---:|---:|---:|
| 1 | 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 |
| 1 | 1 | react | claude-opus-5-5 | 15 | 1.00 | 1.00 |
| 1 | 1 | codex | gpt-6.1-sol | 11 | 0.89 | 0.67 |
| 1 | 1 | react | gpt-6.1-sol | 14 | 1.00 | 0.78 |
| 2 | 1 | claude-code | claude-opus-4-7 | 12 | 0.97 | 0.67 |
| 1 | 2 | claude-code | claude-opus-5-5 | 7 | 0.66 | 0.14 |

Passes per task:

| Run | `level_1_task_3` | `level_1_task_7` | `level_1_task_9` |
|---|---:|---:|---:|
| claude-code / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| react / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| codex / gpt-6.1-sol | 1/5 | 5/5 | 5/5 |
| react / gpt-6.1-sol | 4/5 | 5/5 | 5/5 |
| claude-code / claude-opus-4-7 | 5/5 | 2/5 | 5/5 |

| Run | `level_2_task_1` | `level_2_task_2` | `level_2_task_7` |
|---|---:|---:|---:|
| claude-code / claude-opus-5-5 | 3/5 | 0/5 (v1) | 4/5 |

No trial in these runs ended with an error. The failures:

- **`level_1_task_3`** (5, all GPT-6.1 Sol): each fails the scorer's
  `reference_settings` evidence check only. The computed energies match the
  passing trials, but the reported settings carry extra metadata, such as
  `dispersion=D3(BJ)-PBE` instead of `True`, or a `teacher_model` entry. They
  are kept as failures. Whether the check is too strict is a question for the
  task owners; the trials can be rescored without rerunning.
- **`level_1_task_7`** (3, claude-opus-4-7): mean pressure misses the NPT
  target.
- **`level_2_task_1`** (2): diffusion not stable across blocks and windows
  (k1); diffusion computed without unwrapped positions (k5).
- **`level_2_task_2`** (5, v1 trials): the reported cooling Tg differs from the
  independently fitted one (938 K) or misses the reference window.
- **`level_2_task_7`** (1): the recomputed volumetric expansion coefficient is
  outside ±10% of the reference.

## Pending

Progress of all 36 runs on 2026-11-13 at 14:40 UTC. Claude and OpenAI pools are
running; Kimi waits for an OpenRouter top-up. Counts include reused v1 trials.

| Batch | Agent | Model | Level 1 | Level 2 |
|---:|---|---|---|---|
| 1 | claude-code | claude-opus-5-5 | ✅ in this PR | ✅ in this PR |
| 1 | react | claude-opus-5-5 | ✅ in this PR | 13/15 done, 2 running |
| 1 | codex | gpt-6.1-sol | ✅ in this PR | 12/15 done, 3 running |
| 1 | react | gpt-6.1-sol | ✅ in this PR | 8/15 done, 3 running |
| 1 | openhands | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 1 | react | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 2 | claude-code | claude-opus-4-7 | ✅ in this PR | 9/15 done, 6 running |
| 2 | react | claude-opus-4-7 | 13/15 done, 2 running | 2/15 done, 4 running |
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

That is 245 of 540 trials done (142 of them reused from v1): Claude 69
pending and 14 running, OpenAI 86 pending and 6 running, Kimi 120 pending.
ReAct gpt-5.6-sol level 2 has 3 v1 trials because its two v1 trials refused by
OpenAI's usage policy are rerun. Two Codex gpt-6.1-sol `level_2_task_7` trials
(k1, k2) used up the 50-call budget; they count as failures.

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
traces. All 90 traces were found (`build_index.py --verify`).

## Caveats

- **Reruns.** 6 trials in these runs are reruns after OpenAI refused calls for
  lack of credit (Codex gpt-6.1-sol `level_1_task_7` k3–k5; ReAct gpt-6.1-sol
  `level_1_task_3` k3 and `level_1_task_7` k1, k2). The reports hold the reruns
  only. The merged reports contain no credit or usage-policy errors.
- **Machine load.** Runs shared the machine and its GPUs with other pools, so
  wall-clock durations are not comparable with other runs.
- **Comparison with the October runs.** The October runs in [`../`](../) cover
  all tasks per level; compare per task only.
