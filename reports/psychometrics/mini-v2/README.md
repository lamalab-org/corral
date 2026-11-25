# Psychometrics reports: corral-mini v2

36 runs (18 agent/model pairs × 2 levels), three tasks per level × five trials:
540 trials, 159 passes. 180 trials reuse revised full-task reports; 360 are new.
Layout follows [`../../README.md`](../../README.md).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 1 | `psy_l1_t01_hsns_structure`, `psy_l1_t05_defensible_comparisons`, `psy_l1_t09_careless_responding` |
| 2 | `psy_l2_t01_group_comparability`, `psy_l2_t03_population_classification`, `psy_l2_t08_misfit_replication` |

## Results

Passes are out of 15 per level; Pass@5 uses the report estimator, averaged
over three tasks. Pass^5 is available in the run index.

| Model | Agent | L1 passes | L1 Pass@5 | L2 passes | L2 Pass@5 |
|---|---|---:|---:|---:|---:|
| claude-opus-5-5 | claude-code | 12 | 0.9965 | 8 | 0.8649 |
| claude-opus-5-5 | react | 8 | 0.8649 | 7 | 0.7815 |
| claude-opus-4-7 | claude-code | 7 | 0.6631 | 3 | 0.3299 |
| claude-opus-4-7 | react | 3 | 0.3299 | 2 | 0.3074 |
| claude-opus-4-5-20251101 | claude-code | 7 | 0.6407 | 0 | 0.0000 |
| claude-opus-4-5-20251101 | react | 7 | 0.6407 | 0 | 0.0000 |
| gpt-6.1-sol | codex | 14 | 0.9999 | 9 | 0.6666 |
| gpt-6.1-sol | react | 8 | 0.6633 | 7 | 0.6407 |
| gpt-5.6-sol | codex | 10 | 0.6667 | 5 | 0.3333 |
| gpt-5.6-sol | react | 9 | 0.6666 | 5 | 0.3333 |
| gpt-5-2025-08-07 | codex | 4 | 0.3332 | 0 | 0.0000 |
| gpt-5-2025-08-07 | react | 0 | 0.0000 | 0 | 0.0000 |
| kimi-k3.1-pro | openhands | 5 | 0.5573 | 4 | 0.3332 |
| kimi-k3.1-pro | react | 5 | 0.3333 | 3 | 0.3299 |
| kimi-k2.5 | openhands | 3 | 0.3299 | 0 | 0.0000 |
| kimi-k2.5 | react | 0 | 0.0000 | 0 | 0.0000 |
| kimi-k2-0905 | openhands | 4 | 0.3332 | 0 | 0.0000 |
| kimi-k2-0905 | react | 0 | 0.0000 | 0 | 0.0000 |

## Settings and provenance

50 iterations, reasoning medium (provider default for Kimi K2-0905), surrender
off, Docker 2 CPUs / 6 GiB / 4096 PIDs, two BLAS/OpenMP threads, verbose output
and Langfuse tracing. Kimi uses OpenRouter with 32768 max tokens and no
fallbacks: Novita for K2 models, Modal for the reused K3.1 Pro trials.
Kimi reasoning is set in the request body; OpenHands metadata retains its SDK default.

New trials ran at main `6773d5b1d`; reused trials ran at `e601ae03b`, with
identical Psychometrics tasks, datasets, and scoring code. Per-run provenance
retains source runs and execution IDs; aggregate subset metrics are recomputed.
The OpenHands image adds `libatomic1` and pins `agent-client-protocol==0.11.0`.
Thirty zero-token SDK import failures were excluded and replaced with distinct IDs.

## Errors and caveats

83 trials retain errors and score zero: 27 requests over 8 MB, 16 context
limits, 26 iteration limits, two missing submissions, two policy refusals,
five provider invalid-request errors, and five generic provider errors.
Large requests followed agents reading whole CSV files into context.

**Five ReAct/Kimi K2.5 L2 `psy_l2_t03_population_classification` trials remain
provisional:** the saved error is only “Provider returned error”; its cause
is unconfirmed. These are included as zero scores, with the original errors
preserved. No explicit credit/quota rejection appears in the saved trial records.

## Denario (claude-opus-4-7)

`denario__claude-opus-4-7` in each level folder (k = 5, 30 trials). The reports
are `report.json.xz`: gzip compressed Denario's repetitive sessions about 50
times worse.

| Level | Passes (of 15) | Hit the 100-request limit |
|---|---:|---:|
| 1 | 1 | 5 |
| 2 | 0 | 5 |

1 of 30 trials pass; 10 ended at the model-request limit before
submitting. One L1 trial ended with a Denario `UnboundLocalError`.

Settings: `--agent denario --model anthropic/claude-opus-4-7`, agent kwargs
`{"reasoning_effort": "medium"}`, `--max-iterations 100` (Denario's
model-request budget, including planning and review), surrender off, Docker
sandbox (2 CPUs, 6 GB, 4096 PIDs; guardrails on, evaluation notice off). Env kwarg `repl_timeout_minutes: 15`. Two `psy_l1_t01` trials ran with a 1 GiB worker-response limit (their sessions were about 281 MB).
Code: `main` 9269652a4 merged with PR #482 (b1a7783b8), not merged to `main`.
Worker-response limit 256 MiB (`CORRAL_MAX_WORKER_RESPONSE_BYTES`). Run
2026-11-24/25; trials that crashed on infrastructure (a full disk, connection
loss) were rerun. Single-trial runs merged with `scripts/merge_trial_reports.py`.
