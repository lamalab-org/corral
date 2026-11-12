# Inference optimization reports: corral-mini v2

15 of 36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 185 trials:
all 12 Claude runs, L1 Codex gpt-5.6-sol, and ReAct gpt-6.1-sol on both
levels. The other OpenAI runs wait for OpenAI credit; the Kimi runs have not
started (OpenRouter credit). **Level 1 has two tasks, not three:**
`math_number_theory_a` is excluded because its student model is no longer
served (see Caveats). These replace the corral-mini v1 batch-1 reports.
Layout, names and index columns follow [`../README.md`](../README.md). The
earlier full-benchmark runs (October) are archived in
[`archive/october-full/`](archive/october-full/).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Task | Student → reference |
|---|---|---|
| 1 | `math_counting_probability_b` | llama-3.1-8b → llama-3.1-70b |
| 1 | `math_geometry_d` | ministral-3b-2512 → ministral-14b-2512 |
| 1 | ~~`math_number_theory_a`~~ (excluded) | qwen-2.5-7b → qwen-2.5-72b |
| 2 | `gsm_p1_c`, `gsm_symbolic_c` | llama-3.2-3b → llama-3.1-70b |
| 2 | `gsm_p2_b` | llama-3.1-8b → llama-3.1-70b |

Level 2 `gsm_p1_c` is also in corral-mini v1; its batch-1 Kimi and OpenAI
trials come from the v1 runs (same task code). 5 of them are in the merged
ReAct gpt-6.1-sol L2 run.

## Results

L1: passes out of 10 (2 tasks × 5); L2: passes out of 15.

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | pending |  |  | pending |  |  |
| 1 | react | kimi-k3.1-pro | pending |  |  | pending |  |  |
| 1 | claude-code | claude-opus-5-5 | 10 | 1.00 | 1.00 | 7 | 0.86 | 0.11 |
| 1 | react | claude-opus-5-5 | 9 | 1.00 | 0.66 | 4 | 0.61 | 0.01 |
| 1 | codex | gpt-6.1-sol | pending |  |  | pending |  |  |
| 1 | react | gpt-6.1-sol | 7 | 0.96 | 0.51 | 0 | 0.00 | 0.00 |
| 2 | openhands | kimi-k2.5 | pending |  |  | pending |  |  |
| 2 | react | kimi-k2.5 | pending |  |  | pending |  |  |
| 2 | claude-code | claude-opus-4-7 | 6 | 0.84 | 0.50 | 0 | 0.00 | 0.00 |
| 2 | react | claude-opus-4-7 | 5 | 0.96 | 0.04 | 0 | 0.00 | 0.00 |
| 2 | codex | gpt-5.6-sol | 3 | 0.49 | 0.04 | pending |  |  |
| 2 | react | gpt-5.6-sol | pending |  |  | pending |  |  |
| 3 | openhands | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | react | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | claude-code | claude-opus-4-5-20251101 | 5 | 0.96 | 0.04 | 0 | 0.00 | 0.00 |
| 3 | react | claude-opus-4-5-20251101 | 1 | 0.34 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | pending |  |  | pending |  |  |
| 3 | react | gpt-5-2025-08-07 | pending |  |  | pending |  |  |

57 of 185 trials pass. Per task (merged runs only):

| Level | Task | Passes |
|---|---|---:|
| 1 | `math_counting_probability_b` | 18/40 |
| 1 | `math_geometry_d` | 28/40 |
| 2 | `gsm_p1_c` | 4/35 |
| 2 | `gsm_p2_b` | 1/35 |
| 2 | `gsm_symbolic_c` | 6/35 |

On level 2 only claude-opus-5-5 passes any trial (Claude Code 7, ReAct 4).
5 trials end with an error, kept as failures (score 0); all are an agent using
its 50-step budget (Codex 2, Claude Code 2, ReAct 1).

## Settings

k = 5, 50 iterations, reasoning effort `medium`, surrender off, `--verbose`,
Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Students were called through OpenRouter, each pinned to the provider in the
task JSON with fallbacks off. Codex enforced the 50-call budget (#465).

Corral at `origin/main` `2ef35fb3c` (corral-mini v2, #479); the reused v1
trials ran at `3cef47eac`, with the same `tasks/inference_opt` code. Sandbox
images were built from `docker/benchmark.Dockerfile` with
`CORRAL_TASK=inference_opt` and pip dependencies pinned to the v1 images'
versions (`PIP_CONSTRAINT`; unpinned, Playwright 1.65 needs `libatomic.so.1`
and agent-client-protocol 1.0 breaks OpenHands): `corral-inference_opt:mini-v2`
(OpenHands, ReAct), `:mini-v2-claude` and `:mini-v2-codex`. Claude Code trials
started after 2026-11-11 22:45 waited up to 5 min for the CLI to start
(`CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=300000`; default 60 s).

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** These runs do not use #472's shared sessions: each trial is its
own session, named by its single-trial run ID
(`<agent>__<model>__L<level>__<task>__k<k>`). Use the trial links in
`trials.csv`; the run-level `langfuse_session_url` in `runs.csv` has no
traces. All 185 traces were found (`build_index.py --verify`). Traces of
trials that finished before about 2026-11-11 21:10 have no `benchmark_score`
(the host venv lacked `langfuse`); the scores are in the reports.

## Caveats

- **`math_number_theory_a` excluded.** Its student,
  `qwen/qwen-2.5-7b-instruct@phala`, is no longer served by any OpenRouter
  provider ("404 … no provider serves this model anymore"). It still worked
  for the v1 runs (2026-11-07/08), which used it in two level-1 tasks. In the
  45 v2 trials of this task, 54 of 69 dry runs and final evaluations failed
  with the 404; 27 trials scored 0 without a single student answer and 18
  have no score. None of them measure the agent, so the task is left out of
  the reports and the queue. Three other tasks with this student
  (`math_geometry_a`, `math_counting_probability_a`, `gsm_p2_a`) are not in
  corral-mini v2 but are equally broken. A replacement needs an upstream
  decision; rerunning the v2 Rank + cost selection without the four
  unservable tasks picks `math_number_theory_b` (Spearman 1.00 against the
  reduced level).
- **Reruns.** 14 merged trials are reruns; the reports hold the reruns only.
  9 followed OpenRouter refusing student calls for credit (one of them
  twice; the second refusal was transient), and 5 Claude Code
  claude-opus-4-5 trials failed at start with "Control request timeout:
  initialize" when 20 trials started at once on a loaded host.
- **Machine load.** Runs shared the machine with other environments' pools;
  wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived October runs cover all
  tasks per level; compare per task only.
