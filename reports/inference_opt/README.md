# Inference optimization reports: corral-mini v2.1

24 of 36 runs (18 agent/model pairs × levels 1 and 2), k = 5, 360 trials:
all 12 Claude runs and all 12 OpenAI runs. The 12 Kimi runs have not started
(OpenRouter credit). These replace the corral-mini v1 batch-1 reports.
Layout, names and index columns follow [`../README.md`](../README.md). The
earlier full-benchmark runs (October) are archived in
[`archive/october-full/`](archive/october-full/).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Why v2.1

**corral-mini v2.1 is corral-mini v2 with a different task set:**
`math_number_theory_b` replaces `math_number_theory_a` on level 1. The
student model of `math_number_theory_a`, `qwen/qwen-2.5-7b-instruct@phala`, is
no longer served by any OpenRouter provider (see Caveats). Run IDs carry the
version: `inference_opt-mini-v2.1-l<level>-<agent>-<model>`.

## Tasks

| Level | Task | Student → reference |
|---|---|---|
| 1 | `math_counting_probability_b` | llama-3.1-8b → llama-3.1-70b |
| 1 | `math_geometry_d` | ministral-3b-2512 → ministral-14b-2512 |
| 1 | `math_number_theory_b` (v2.1; replaces `math_number_theory_a`) | llama-3.1-8b → llama-3.1-70b |
| 2 | `gsm_p1_c`, `gsm_symbolic_c` | llama-3.2-3b → llama-3.1-70b |
| 2 | `gsm_p2_b` | llama-3.1-8b → llama-3.1-70b |

Level 2 `gsm_p1_c` is also in corral-mini v1. Its 10 batch-1 OpenAI trials
come from the v1 runs, with the same task code, and are in the Codex and
ReAct gpt-6.1-sol level 2 runs. 10 batch-1 Kimi trials from v1 wait for the
Kimi runs.

## Results

Passes are out of 15 per level (3 tasks × 5).

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | pending |  |  | pending |  |  |
| 1 | react | kimi-k3.1-pro | pending |  |  | pending |  |  |
| 1 | claude-code | claude-opus-5-5 | 15 | 1.00 | 1.00 | 7 | 0.86 | 0.11 |
| 1 | react | claude-opus-5-5 | 13 | 1.00 | 0.55 | 4 | 0.61 | 0.01 |
| 1 | codex | gpt-6.1-sol | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | react | gpt-6.1-sol | 11 | 0.97 | 0.45 | 0 | 0.00 | 0.00 |
| 2 | openhands | kimi-k2.5 | pending |  |  | pending |  |  |
| 2 | react | kimi-k2.5 | pending |  |  | pending |  |  |
| 2 | claude-code | claude-opus-4-7 | 6 | 0.56 | 0.33 | 0 | 0.00 | 0.00 |
| 2 | react | claude-opus-4-7 | 5 | 0.64 | 0.03 | 0 | 0.00 | 0.00 |
| 2 | codex | gpt-5.6-sol | 4 | 0.55 | 0.03 | 0 | 0.00 | 0.00 |
| 2 | react | gpt-5.6-sol | 4 | 0.55 | 0.03 | 0 | 0.00 | 0.00 |
| 3 | openhands | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | react | kimi-k2-0905 | pending |  |  | pending |  |  |
| 3 | claude-code | claude-opus-4-5-20251101 | 5 | 0.64 | 0.03 | 0 | 0.00 | 0.00 |
| 3 | react | claude-opus-4-5-20251101 | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

76 of 360 trials pass. Passes per task:

| Level | Task | Passes |
|---|---|---:|
| 1 | `math_counting_probability_b` | 18/60 |
| 1 | `math_geometry_d` | 30/60 |
| 1 | `math_number_theory_b` | 17/60 |
| 2 | `gsm_p1_c` | 4/60 |
| 2 | `gsm_p2_b` | 1/60 |
| 2 | `gsm_symbolic_c` | 6/60 |

On level 2, only claude-opus-5-5 passes any trial: 7 with Claude Code and 4
with ReAct.

41 trials end with an error. They are kept as failures (score 0), because
each one is the agent's own outcome and not an infrastructure fault:

| Cause | Trials |
|---|---:|
| Used the 50-call or 50-step budget: Codex 33 (gpt-6.1-sol 24, gpt-5.6-sol 9), Claude Code 2, ReAct 2 | 37 |
| ReAct gpt-5.6-sol on `math_geometry_d`: input over OpenAI's 922k-token limit | 3 |
| ReAct gpt-5.6-sol: prompt flagged by OpenAI's usage policy | 1 |

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
versions (`PIP_CONSTRAINT`). Unpinned, Playwright 1.65 needs `libatomic.so.1`
and agent-client-protocol 1.0 breaks OpenHands. The images are
`corral-inference_opt:mini-v2` (OpenHands, ReAct), `:mini-v2-claude` and
`:mini-v2-codex`. Claude Code trials started after 2026-11-11 22:45 waited up
to 5 min for the CLI to start (`CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=300000`; the
default is 60 s).

Each trial ran as its own `corral bench --trials 1` run. The reports are
merged with `scripts/merge_trial_reports.py`: metrics are recomputed, messages
and tool calls are rebuilt from each trial's final state, and the source runs
are listed in `provenance.json`.

**Langfuse.** These runs do not use #472's shared sessions: each trial is its
own session, named by its single-trial run ID
(`<agent>__<model>__L<level>__<task>__k<k>`). Use the trial links in
`trials.csv`; the run-level `langfuse_session_url` in `runs.csv` has no
traces. The check that every trace exists (`build_index.py --verify`) is still running. Traces of
trials that finished before about 2026-11-11 21:10 have no `benchmark_score`
(the host venv lacked `langfuse`); the scores are in the reports.

## Caveats

- **`math_number_theory_a` replaced.** Its student,
  `qwen/qwen-2.5-7b-instruct@phala`, is no longer served by any OpenRouter
  provider ("404 … no provider serves this model anymore"). It still worked
  for the v1 runs (2026-11-07/08). In the 45 v2 trials of this task, 54 of 69
  dry runs and final evaluations failed with the 404; 27 trials scored 0
  without a single student answer and 18 have no score. None of them measure
  the agent, so they are not in the reports. Rerunning the v2 Rank + cost
  selection without the four tasks that use this student
  (`math_number_theory_a`, `math_geometry_a`, `math_counting_probability_a`,
  `gsm_p2_a`) picks `math_number_theory_b`: Spearman 1.00 against the reduced
  level, $605 for 7 agents × 5, against $619 for the released set.
- **Reruns.** 29 merged trials are reruns; the reports hold the reruns only.
  Some trials were rerun more than once, so the causes below add up to more
  than 29:
  - 9 after OpenRouter refused student calls for lack of credit (one of these
    refusals was transient);
  - 13 after OpenAI refused calls for lack of credit;
  - 5 Claude Code claude-opus-4-5 trials that failed at start with "Control
    request timeout: initialize" when 20 trials started at once on a loaded
    host;
  - 4 Codex trials that hung after an OpenAI "no credits remaining"
    disconnect.
- **Level 2 final-run timeouts.** 9 Claude trials on `gsm_p1_c` and
  `gsm_symbolic_c` score 0 because their final run hit the one-hour run limit
  (`run_time_limit_s`) before recording any answer. The two trials checked
  used 7–10 samples per question, with at most 8 requests in flight to
  Llama 3.2 3B; in one of them, a 150-question experiment with the same
  sampling took about 50 minutes. The limit is documented in the task, so the
  trials are kept, but whether a final run finishes depends on how fast the
  provider is at the time.
- **Machine load.** Runs shared the machine with other environments' pools,
  so wall-clock durations are not comparable with other runs.
- **Comparison with the archived runs.** The archived October runs cover all
  tasks per level; compare per task only.
