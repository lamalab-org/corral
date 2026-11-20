# Corral MD reports: corral-mini v2

36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 540 trials: Claude,
OpenAI and Kimi models, each with its native agent (Claude Code, Codex or
OpenHands) and ReAct. These replace the corral-mini v1 reports (#469). Layout,
names and index columns follow [`../../README.md`](../../README.md). The
October full-benchmark runs are in [`../`](..).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

corral-mini v2 as corrected in #492, which reselected corral MD from the
October reports rescored in #444:

| Level | Tasks |
|---|---|
| 1 | `level_1_task_3`, `level_1_task_7`, `level_1_task_9` |
| 2 | `level_2_task_1`, `level_2_task_2`, `level_2_task_7` |

`level_1_task_9` and `level_2_task_2` are also in corral-mini v1. Their v1
trials are reused (same task code and settings). In the runs below, 142 trials
come from v1: 5 `level_1_task_9` and 5 `level_2_task_2` trials for each
claude-opus-5-5, gpt-6.1-sol, gpt-5.6-sol, gpt-5-2025-08-07 and Kimi run (ReAct
gpt-5.6-sol has 3 `level_2_task_2` trials from v1; its other two v1 trials were
refused by OpenAI's usage policy and rerun), and 2 `level_2_task_2` trials for
each claude-opus-4-7 run. claude-opus-4-5-20251101 has no v1 trials.

## Results

Passes are out of 15 (3 tasks × 5). Native agent is Claude Code for Claude,
Codex for OpenAI and OpenHands for Kimi models.

| Batch | Model | L1 native | L1 ReAct | L2 native | L2 ReAct |
|---:|---|---:|---:|---:|---:|
| 1 | claude-opus-5-5 | 15 | 15 | 7 | 5 |
| 2 | claude-opus-4-7 | 12 | 11 | 5 | 5 |
| 3 | claude-opus-4-5-20251101 | 4 | 3 | 5 | 2 |
| 1 | gpt-6.1-sol | 11 | 14 | 1 | 2 |
| 2 | gpt-5.6-sol | 13 | 13 | 5 | 1 |
| 3 | gpt-5-2025-08-07 | 8 | 2 | 4 | 0 |
| 1 | kimi-k3.1-pro | 11 | 8 | 1 | 2 |
| 2 | kimi-k2.5 | 7 | 8 | 1 | 4 |
| 3 | kimi-k2-0905 | 6 | 7 | 1 | 2 |

pass@5 and pass^5 per run are in [`runs.csv`](runs.csv). Passes per task
(`task_3`, `task_7`, `task_9` on level 1; `task_1`, `task_2`, `task_7` on
level 2):

| Run | Level 1 | Level 2 |
|---|---|---|
| claude-code / claude-opus-5-5 | 5 · 5 · 5 | 3 · 0 · 4 |
| react / claude-opus-5-5 | 5 · 5 · 5 | 2 · 0 · 3 |
| claude-code / claude-opus-4-7 | 5 · 2 · 5 | 1 · 0 · 4 |
| react / claude-opus-4-7 | 5 · 1 · 5 | 1 · 0 · 4 |
| claude-code / claude-opus-4-5-20251101 | 1 · 0 · 3 | 2 · 0 · 3 |
| react / claude-opus-4-5-20251101 | 0 · 0 · 3 | 0 · 0 · 2 |
| codex / gpt-6.1-sol | 1 · 5 · 5 | 0 · 0 · 1 |
| react / gpt-6.1-sol | 4 · 5 · 5 | 1 · 0 · 1 |
| codex / gpt-5.6-sol | 4 · 4 · 5 | 2 · 0 · 3 |
| react / gpt-5.6-sol | 4 · 4 · 5 | 0 · 0 · 1 |
| codex / gpt-5-2025-08-07 | 1 · 2 · 5 | 1 · 0 · 3 |
| react / gpt-5-2025-08-07 | 1 · 0 · 1 | 0 · 0 · 0 |
| openhands / kimi-k3.1-pro | 2 · 4 · 5 | 0 · 0 · 1 |
| react / kimi-k3.1-pro | 2 · 1 · 5 | 1 · 0 · 1 |
| openhands / kimi-k2.5 | 3 · 1 · 3 | 0 · 0 · 1 |
| react / kimi-k2.5 | 5 · 1 · 2 | 0 · 0 · 4 |
| openhands / kimi-k2-0905 | 4 · 0 · 2 | 0 · 0 · 1 |
| react / kimi-k2-0905 | 4 · 0 · 3 | 1 · 0 · 1 |

No run passes `level_2_task_2` (0/90); the October runs did not pass it either
(0/35). Most of its failures report a cooling Tg that the scorer fits
independently at about 938 K from the agent's own data, while the reference
window is 1419–1735 K.

### Errors

42 of 540 trials ended with an error; all count as failures.

| Error | Trials |
|---|---:|
| Agent used up its 50-step (ReAct, OpenHands) or 50-model-call (Codex) budget | 29 |
| OpenAI usage-policy refusal (ReAct gpt-5.6-sol level 2) | 6 |
| OpenHands finished without calling `submit_answer` (Kimi) | 3 |
| Scorer could not score the trial | 2 |
| Prompt over the 262,144-token context (ReAct kimi-k2-0905 `level_2_task_1` k3) | 1 |
| Provider rejected the request as invalid (Novita; ReAct kimi-k2-0905 `level_2_task_1` k5) | 1 |

The budget count includes Codex gpt-6.1-sol `level_2_task_7` k3, which reached
the budget (51 model calls) but did not stop: the budget interrupt arrived
while a 24-minute tool call was running, and Codex never ended the turn. It was
stopped by hand after 4.5 idle hours, so its report shows "Codex process closed
stdout" instead of the budget error.

The six usage-policy refusals (`level_2_task_1` k2, k3, k5; `level_2_task_7`
k1, k3, k4) are kept as failures.

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

### Failed checks

Trials that fail a scorer check (errors above excluded), with the most common
failed checks:

- **`level_1_task_3`** (34): `independent_model_calculation` (28);
  `reference_settings` (5, all GPT-6.1 Sol). The `reference_settings` failures
  match the passing trials' energies but report settings with extra metadata,
  such as `dispersion=D3(BJ)-PBE` instead of `True`, or a `teacher_model`
  entry. Whether this check is too strict is a question for the task owners.
- **`level_1_task_7`** (45): mean pressure misses the NPT target (40).
- **`level_1_task_9`** (17): energies not chronological and uniformly spaced
  (10), or `main.jsonl` missing (7).
- **`level_2_task_1`** (67): unwrapped positions, or times and positions for
  the full 500 ps interval, not recorded (35); LAMMPS errors (9);
  non-chronological records (6); a reported diffusion coefficient that
  disagrees with the scorer's MSD fit (6).
- **`level_2_task_2`** (87): the cooling Tg differs from the scorer's own fit
  (51) or misses the reference window (25).
- **`level_2_task_7`** (27): the recomputed expansion coefficient is outside
  ±10% of the reference (17).

All trials can be rescored from their saved artifacts without rerunning.

## Settings

k = 5, 50 iterations, reasoning effort `medium`, surrender off, `--verbose`,
Docker sandbox with 2 CPUs, 6 GB, a 4096 PID limit and one GPU per trial
(MD runs on the GPU), `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`
= 2, Langfuse on. Kimi models ran through OpenRouter pinned to one provider
without fallbacks: Modal for kimi-k3.1-pro, Novita for kimi-k2.5 and
kimi-k2-0905 (no reasoning effort for kimi-k2-0905), with `max_tokens` 32768.
Claude Code waited up to 5 min for the CLI to start
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
traces. All 360 traces were found (`build_index.py --verify`).

## Caveats

- **Reruns.** 12 trials are reruns after OpenAI refused calls for lack of
  credit: Codex gpt-6.1-sol `level_1_task_7` k3–k5 and `level_2_task_7` k4, k5;
  ReAct gpt-6.1-sol `level_1_task_3` k3, `level_1_task_7` k1, k2,
  `level_2_task_1` k4 and `level_2_task_7` k1–k3. Two v1 trials of ReAct
  gpt-5.6-sol `level_2_task_2` (k3, k4) refused by OpenAI's usage policy were
  rerun. The reports hold the reruns only and contain no credit errors.
- **Langfuse traces of reruns.** A rerun uses the same trial name, so its
  Langfuse trace ID matches the failed attempt's; the trace holds both
  attempts.
- **Run order.** The 120 new Kimi trials ran on 2026-11-14 and 15, cheapest
  model first (kimi-k2.5, kimi-k2-0905, then kimi-k3.1-pro), 12 to 16 at a
  time. No Kimi trial was rerun.
- **Machine load.** Runs shared the machine and its GPUs with other pools, so
  wall-clock durations are not comparable with other runs.
- **Comparison with the October runs.** The October runs in [`../`](..) cover
  all tasks per level; compare per task only.
