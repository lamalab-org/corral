# Wetlab reports: corral-mini v2

36 runs (18 agent/model pairs × levels 2 and 3), 3 tasks per level × k = 5,
540 trials. Layout, names and index columns follow
[`../../README.md`](../../README.md). The October full-benchmark runs are in
[`../`](..).

## Tasks

| Level | Tasks |
|---|---|
| 2 | `qualysis_lvl2_01`, `qualysis_lvl2_08`, `qualysis_lvl2_10` |
| 3 | `qualysis_lvl3_05`, `qualysis_lvl3_09`, `qualysis_lvl3_10` |

`qualysis_lvl2_10` and `qualysis_lvl3_10` are also in corral-mini v1; their
trials come from the v1 runs (same task, settings and code).

## Results (passes out of 15 per level)

| Batch | Agent | Model | L2 passes | L2 pass@5 | L2 pass^5 | L3 passes | L3 pass@5 | L3 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 7 | 0.66 | 0.14 | 2 | 0.45 | 0.00 |
| 1 | react | kimi-k3.1-pro | 9 | 0.95 | 0.34 | 5 | 0.64 | 0.03 |
| 1 | claude-code | claude-opus-5-5 | 14 | 1.00 | 0.78 | 11 | 1.00 | 0.24 |
| 1 | react | claude-opus-5-5 | 15 | 1.00 | 1.00 | 13 | 1.00 | 0.55 |
| 1 | codex | gpt-6.1-sol | 15 | 1.00 | 1.00 | 15 | 1.00 | 1.00 |
| 1 | react | gpt-6.1-sol | 15 | 1.00 | 1.00 | 13 | 1.00 | 0.69 |
| 2 | openhands | kimi-k2.5 | 6 | 0.64 | 0.11 | 0 | 0.00 | 0.00 |
| 2 | react | kimi-k2.5 | 4 | 0.33 | 0.11 | 1 | 0.22 | 0.00 |
| 2 | claude-code | claude-opus-4-7 | 10 | 0.97 | 0.36 | 3 | 0.53 | 0.00 |
| 2 | react | claude-opus-4-7 | 2 | 0.31 | 0.00 | 1 | 0.22 | 0.00 |
| 2 | codex | gpt-5.6-sol | 15 | 1.00 | 1.00 | 9 | 0.97 | 0.14 |
| 2 | react | gpt-5.6-sol | 12 | 1.00 | 0.47 | 5 | 0.33 | 0.33 |
| 3 | openhands | kimi-k2-0905 | 1 | 0.22 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | kimi-k2-0905 | 4 | 0.55 | 0.03 | 1 | 0.22 | 0.00 |
| 3 | claude-code | claude-opus-4-5-20251101 | 5 | 0.84 | 0.01 | 4 | 0.55 | 0.03 |
| 3 | react | claude-opus-4-5-20251101 | 7 | 0.86 | 0.11 | 1 | 0.22 | 0.00 |
| 3 | codex | gpt-5-2025-08-07 | 8 | 0.86 | 0.34 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 9 | 0.89 | 0.22 | 0 | 0.00 | 0.00 |

242 of 540 trials pass.

## Settings

k = 5, 50 iterations, reasoning effort `medium` (provider default for
kimi-k2-0905), surrender off, Docker sandbox with 2 CPUs, 6 GB and a 4096 PID
limit per trial, Langfuse on. Kimi via OpenRouter (`max_tokens` 32768, no
fallbacks): kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on Novita.

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

## Denario (claude-opus-4-7)

`denario__claude-opus-4-7` in each level folder (k = 5, 30 trials). The reports
are `report.json.xz`: gzip compressed Denario's repetitive sessions about 50
times worse.

| Level | Passes (of 15) | Hit the 100-request limit |
|---|---:|---:|
| 2 | 3 | 5 |
| 3 | 0 | 6 |

3 of 30 trials pass; 11 ended at the model-request limit before
submitting.

Settings: `--agent denario --model anthropic/claude-opus-4-7`, agent kwargs
`{"reasoning_effort": "medium"}`, `--max-iterations 100` (Denario's
model-request budget, including planning and review), surrender off, Docker
sandbox (2 CPUs, 6 GB, 4096 PIDs; guardrails on, evaluation notice off). The original image adds a prebuilt fontconfig cache (`fc-cache -s`; on `main` since #519): without it, plotting code makes fontconfig write cache symlinks into the workspace, which Corral rejects.
Worker-response limit 256 MiB (`CORRAL_MAX_WORKER_RESPONSE_BYTES`).
Single-trial runs merged with `scripts/merge_trial_reports.py`.

Code and reruns: trials ran on `main` 9269652a4 merged with PR #482
(b1a7783b8) on 2026-11-24/25; trials that crashed on infrastructure (a full
disk, connection loss) were rerun then. 5 of the 30 trials hit Denario
or image bugs fixed since (broken MoLFormer from an unpinned `transformers`,
the `record_status_starter` loop, `KeyError: 'name'`, `UnboundLocalError`) and
were rerun on 2026-11-27/28 at `main` 2bd3114b5 (#482 merged, with #518 and
#519); their Langfuse sessions end in `-rerun`. The other trials are unchanged.
