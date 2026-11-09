# Inference optimization reports: corral-mini v1

8 runs (4 agent/model pairs × 2 levels), 3 tasks per level × k = 5,
120 trials. **Batch 1 only** (newest model per family), **without the Claude
routes**: Claude Code and ReAct with claude-opus-5-5 are pending (the
Anthropic account ran out of credit) and will be added here, followed by
batches 2 and 3. Layout, names and index columns follow
[`../README.md`](../README.md).

The earlier full-benchmark runs (October, 10 tasks per level) are archived in
[`archive/october-full/`](archive/october-full/); `build_index.py` ignores
`archive/`.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks (corral-mini v1)

| Level | Task | Student | Must match |
|---|---|---|---|
| 1 | `math_algebra_b` | Llama 3.1 8B | Llama 3.1 70B accuracy |
| 1 | `math_counting_probability_a` | Qwen 2.5 7B | Qwen 2.5 72B accuracy |
| 1 | `math_geometry_a` | Qwen 2.5 7B | Qwen 2.5 72B accuracy |
| 2 | `gsm_p1_c` | Llama 3.2 3B | Llama 3.1 70B template consistency |
| 2 | `gsm_p2_d` | Ministral 3 3B | Ministral 3 14B template consistency |
| 2 | `gsm_symbolic_d` | Ministral 3 3B | Ministral 3 14B template consistency |

## Results (passes out of 15 per level; pass@5 / pass^5)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 | Errors L1/L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 10 | 0.89 | 0.44 | 4 | 0.33 | 0.10 | 1/7 |
| 1 | react | kimi-k3.1-pro | 3 | 0.53 | 0.00 | 5 | 0.55 | 0.10 | 7/6 |
| 1 | codex | gpt-6.1-sol | 5 | 0.63 | 0.02 | 3 | 0.53 | 0.00 | 10/11 |
| 1 | react | gpt-6.1-sol | 12 | 0.99 | 0.46 | 8 | 0.66 | 0.35 | 0/0 |
| 1 | claude-code | claude-opus-5-5 | pending | | | pending | | | |
| 1 | react | claude-opus-5-5 | pending | | | pending | | | |

50 of 120 trials pass. Per task (four routes, 20 trials each):

| Level | Task | Passes |
|---|---|---:|
| 1 | `math_algebra_b` | 7/20 |
| 1 | `math_counting_probability_a` | 14/20 |
| 1 | `math_geometry_a` | 9/20 |
| 2 | `gsm_p1_c` | 0/20 |
| 2 | `gsm_p2_d` | 5/20 |
| 2 | `gsm_symbolic_d` | 15/20 |

Every saved error is an agent using its 50-step budget without submitting
(21 Codex "model-call budget", 13 ReAct "interaction budget", 8 OpenHands
"maximum iterations limit"); they are kept as failures (score 0). Codex hit
its budget in 21 of 30 trials (it counts every model call, one per tool step);
ReAct with the same model never did.

## Settings

All runs: k = 5, 50 iterations, reasoning effort `medium`, surrender off,
`--verbose`, Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2 (passed by the
pool; the benchmark image sets none), Langfuse on, default bridge network.
Kimi went through OpenRouter on Modal with `max_tokens` 32768 and
`allow_fallbacks=false`. Codex enforced the 50-call budget (#465). Students
were called through OpenRouter, each pinned to one provider with fallbacks off
(the task JSONs' routes), so every trial received `OPENROUTER_API_KEY`.

Runs used Corral at `3cef47eac` on branch `mini-1` (with #465 and #468, not
#472), with the task version at that commit. Sandbox images were built from
`docker/benchmark.Dockerfile` with `CORRAL_TASK=inference_opt`:
`corral-inference_opt:mini` (OpenHands, ReAct), `:mini-claude` and
`:mini-codex`.

## How the reports were made

Each trial ran as its own `corral bench --trials 1` run from a queue and was
merged with `scripts/merge_trial_reports.py`: one report per level, agent and
model, metrics recomputed with `BenchmarkResult`, and `provenance.json`
listing the 15 source runs. The runs were verbose; every trial has its
messages and tool calls.

**Langfuse.** These runs predate #472, so each trial is its own Langfuse
session, named by its single-trial run ID
(`<agent>__<model>__L<level>__<task>__k<k>`). Use the trial links in
`trials.csv`; the run-level `langfuse_session_url` in `runs.csv` (the merged
run ID) has no traces. All 120 trace IDs were found in their sessions
(`build_index.py --verify`).

## Caveats

- **8 trials were rerun**; the reports hold the reruns only.
  - 7 ReAct gpt-6.1-sol trials: the OpenAI account ran out of credit at about
    21:45 UTC on 2026-11-07. Six failed with "You have no credits remaining";
    one (L2 `gsm_symbolic_d`, k = 4) ended with "a terminal agent run cannot
    record turns", which is how a ReAct run reports the credit error. Rerun
    after a top-up.
  - 1 ReAct gpt-6.1-sol trial (L1 `math_algebra_b`, k = 2): OpenAI refused a
    request ("your prompt was flagged as potentially violating our usage
    policy"); replaced by a completed rerun, as the October collection did
    for provider-blocked trials.
- **Rerun traces contain both attempts.** The reruns kept their single-trial
  run IDs, so each one wrote into the same Langfuse trace as its failed
  attempt (see "Known issue: some traces mix attempts" in `../README.md`).
  The report is authoritative.
- **No credit errors remain.** The merged reports, logs and final state
  snapshots were searched for Anthropic, OpenAI and OpenRouter credit errors.
- **`gsm_p1_c` was never solved** (0/20): Llama 3.2 3B has to reach Llama 3.1
  70B's template consistency (0.53 against its own 0.20 on test).
- **Comparison with the archived runs.** The archived runs cover all 10 tasks
  per level; compare per task only.
