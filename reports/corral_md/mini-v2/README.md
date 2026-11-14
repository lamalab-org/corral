# Corral MD reports: corral-mini v2

12 of 36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 180 trials:
level 1 for claude-opus-5-5, claude-opus-4-7, claude-opus-4-5-20251101 and
gpt-6.1-sol, and level 2 for claude-opus-5-5 and claude-opus-4-7, each with the
native agent and ReAct. The other runs are in progress or wait for credits (see
[Pending](#pending)). These replace the corral-mini v1 reports (#469). Layout,
names and index columns follow [`../../README.md`](../../README.md). The
October full-benchmark runs are in [`../`](../).

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
reused task from v1. In the runs below, the 30 `level_1_task_9` trials of
batches 1 and 3 and the 14 `level_2_task_2` trials of batches 1 and 2 come from
v1 (claude-opus-4-7 has 2 v1 trials of `level_2_task_2` per agent).

## Results

Passes are out of 15 (3 tasks × 5).

| Batch | Level | Agent | Model | Passes | pass@5 | pass^5 |
|---|---:|---|---|---:|---:|---:|
| 1 | 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 |
| 1 | 1 | react | claude-opus-5-5 | 15 | 1.00 | 1.00 |
| 1 | 1 | codex | gpt-6.1-sol | 11 | 0.89 | 0.67 |
| 1 | 1 | react | gpt-6.1-sol | 14 | 1.00 | 0.78 |
| 2 | 1 | claude-code | claude-opus-4-7 | 12 | 0.97 | 0.67 |
| 2 | 1 | react | claude-opus-4-7 | 11 | 0.89 | 0.67 |
| 3 | 1 | claude-code | claude-opus-4-5-20251101 | 4 | 0.55 | 0.03 |
| 3 | 1 | react | claude-opus-4-5-20251101 | 3 | 0.33 | 0.03 |
| 1 | 2 | claude-code | claude-opus-5-5 | 7 | 0.66 | 0.14 |
| 1 | 2 | react | claude-opus-5-5 | 5 | 0.64 | 0.03 |
| 2 | 2 | claude-code | claude-opus-4-7 | 5 | 0.56 | 0.11 |
| 2 | 2 | react | claude-opus-4-7 | 5 | 0.56 | 0.11 |

Passes per task:

| Run | `level_1_task_3` | `level_1_task_7` | `level_1_task_9` |
|---|---:|---:|---:|
| claude-code / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| react / claude-opus-5-5 | 5/5 | 5/5 | 5/5 |
| codex / gpt-6.1-sol | 1/5 | 5/5 | 5/5 |
| react / gpt-6.1-sol | 4/5 | 5/5 | 5/5 |
| claude-code / claude-opus-4-7 | 5/5 | 2/5 | 5/5 |
| react / claude-opus-4-7 | 5/5 | 1/5 | 5/5 |
| claude-code / claude-opus-4-5-20251101 | 1/5 | 0/5 | 3/5 |
| react / claude-opus-4-5-20251101 | 0/5 | 0/5 | 3/5 |

| Run | `level_2_task_1` | `level_2_task_2` | `level_2_task_7` |
|---|---:|---:|---:|
| claude-code / claude-opus-5-5 | 3/5 | 0/5 | 4/5 |
| react / claude-opus-5-5 | 2/5 | 0/5 | 3/5 |
| claude-code / claude-opus-4-7 | 1/5 | 0/5 | 4/5 |
| react / claude-opus-4-7 | 1/5 | 0/5 | 4/5 |

One trial ended with an agent error: ReAct claude-opus-5-5 `level_2_task_7` k1
used up its 50-step budget. Two trials could not be scored and count as
failures (see below). The other failures fail the scorer's checks:

- **`level_1_task_3`**: GPT-6.1 Sol (5) fails only the `reference_settings`
  evidence check. The computed energies match the passing trials, but the
  reported settings carry extra metadata, such as `dispersion=D3(BJ)-PBE`
  instead of `True`, or a `teacher_model` entry. Whether the check is too
  strict is a question for the task owners; the trials can be rescored without
  rerunning. claude-opus-4-5 (9) fails the `independent_model_calculation`
  check.
- **`level_1_task_7`** (17, claude-opus-4-7 and claude-opus-4-5): mean pressure
  misses the NPT target.
- **`level_1_task_9`** (4, claude-opus-4-5): the energy series for the
  correlation is not chronological and uniformly spaced.
- **`level_2_task_1`** (13): positions and times not kept for the full 500 ps
  diffusion interval, unwrapped positions missing, diffusion not stable across
  blocks, non-chronological step records, a reported value that disagrees with
  the scorer's own MSD fit, or a value outside the reference range.
- **`level_2_task_2`** (20, 14 of them v1 trials): the cooling Tg differs from
  the independently fitted one (about 938 K), misses the reference window, is
  not a finite number, or the raw logs are missing.
- **`level_2_task_7`** (5): two expansion coefficients outside ±10% of the
  reference, the budget error above, and the two unscored trials.

**Unscored trials.** The scorer could not score claude-opus-4-7
`level_2_task_7` ReAct k1 and claude-code k4: "Cannot identify the production
samples in this log". Their `run.jsonl` logs label every sample
`phase: production` but also hold summary rows (types `nvt_summary`, `nvt`,
`diagnostics`) that the scorer neither uses nor skips (it skips rows labeled
exactly `summary`). This is a gap in the scorer, not a failed check: a failed
check would have scored 0. Both reported β ≈ 9.4–9.5e-5 K⁻¹ against the
reference 9.56e-5 K⁻¹. They are scored 0 here (`merge_trial_reports.py
--unscored-as-failure`; listed in each run's `provenance.json`, with the
scorer's message as the trial error) and can be rescored without rerunning once
the scorer handles these rows.

## Pending

Progress of all 36 runs on 2026-11-13 at 21:10 UTC. Counts include reused v1
trials. The Claude pool has 10 trials left (all running). OpenAI ran out of
credit at 16:09; after a top-up it resumed at 20:56. Kimi waits for an
OpenRouter top-up.

| Batch | Agent | Model | Level 1 | Level 2 |
|---:|---|---|---|---|
| 1 | claude-code | claude-opus-5-5 | ✅ in this PR | ✅ in this PR |
| 1 | react | claude-opus-5-5 | ✅ in this PR | ✅ in this PR |
| 1 | codex | gpt-6.1-sol | ✅ in this PR | 13/15, 2 running (credit reruns) |
| 1 | react | gpt-6.1-sol | ✅ in this PR | 8/15, 7 running (4 credit reruns) |
| 1 | openhands | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 1 | react | kimi-k3.1-pro | 5/15 (v1) | 5/15 (v1) |
| 2 | claude-code | claude-opus-4-7 | ✅ in this PR | ✅ in this PR |
| 2 | react | claude-opus-4-7 | ✅ in this PR | ✅ in this PR |
| 2 | codex | gpt-5.6-sol | 5/15 (v1), 1 running | 5/15 (v1) |
| 2 | react | gpt-5.6-sol | 5/15 (v1) | 3/15 (v1) |
| 2 | openhands | kimi-k2.5 | 5/15 (v1) | 5/15 (v1) |
| 2 | react | kimi-k2.5 | 5/15 (v1) | 5/15 (v1) |
| 3 | claude-code | claude-opus-4-5-20251101 | ✅ in this PR | 10/15, 5 running |
| 3 | react | claude-opus-4-5-20251101 | ✅ in this PR | 10/15, 5 running |
| 3 | codex | gpt-5-2025-08-07 | 5/15 (v1) | 5/15 (v1) |
| 3 | react | gpt-5-2025-08-07 | 5/15 (v1) | 5/15 (v1) |
| 3 | openhands | kimi-k2-0905 | 5/15 (v1) | 5/15 (v1) |
| 3 | react | kimi-k2-0905 | 5/15 (v1) | 5/15 (v1) |

That is 320 of 540 trials done (142 of them reused from v1); 20 running
(10 Claude, 10 OpenAI); 80 OpenAI and 120 Kimi pending. Six gpt-6.1-sol level 2
trials failed for lack of OpenAI credit and are being rerun; the reports will
hold the reruns only. ReAct gpt-5.6-sol level 2 has 3 v1 trials because its two
v1 trials refused by OpenAI's usage policy are rerun.

Codex gpt-6.1-sol `level_2_task_7` k1 and k2 used up the 50-call budget. k3
reached it too (51 model calls) but did not stop: the budget interrupt arrived
while a 24-minute tool call was running, and Codex never ended the turn. It was
stopped by hand after 4.5 idle hours, so its report shows "Codex process closed
stdout" instead of the budget error. All three count as failures.

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
traces. All 180 traces were found (`build_index.py --verify`).

## Caveats

- **Reruns.** 6 trials in these runs are reruns after OpenAI refused calls for
  lack of credit (Codex gpt-6.1-sol `level_1_task_7` k3–k5; ReAct gpt-6.1-sol
  `level_1_task_3` k3 and `level_1_task_7` k1, k2). The reports hold the reruns
  only. The merged reports contain no credit or usage-policy errors.
- **Machine load.** Runs shared the machine and its GPUs with other pools, so
  wall-clock durations are not comparable with other runs.
- **Comparison with the October runs.** The October runs in [`../`](../) cover
  all tasks per level; compare per task only.
