# Retrosynthesis reports: corral-mini v2

32 of 36 runs (16 agent/model pairs × levels 2 and 3), 3 tasks per level ×
k = 5, 480 trials. The 4 kimi-k3.1-pro runs are still running and will be
added. Layout, names and index columns follow
[`../../README.md`](../../README.md). The October full-benchmark runs are in
[`../`](../).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 2 | `make_6_lvl2`, `make_9_lvl2`, `make_10_lvl2` |
| 3 | `make_6_lvl3`, `make_7_lvl3`, `make_8_lvl3` |

These are the retrosynthesis picks of the corrected corral-mini v2 manifest
(#492). The first v2 picks (`make_1`, `make_5_lvl3`) were selected from October
scores taken before the reaction-database and buyables fixes; the tasks were
reselected from the corrected October reports with the same method.

## Results (passes out of 15 per level)

| Batch | Agent | Model | L2 passes | L2 pass@5 | L2 pass^5 | L3 passes | L3 pass@5 | L3 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | claude-code | claude-opus-5-5 | 1 | 0.22 | 0.00 | 5 | 0.56 | 0.11 |
| 1 | react | claude-opus-5-5 | 0 | 0.00 | 0.00 | 6 | 0.92 | 0.01 |
| 1 | codex | gpt-6.1-sol | 4 | 0.33 | 0.11 | 14 | 1.00 | 0.78 |
| 1 | react | gpt-6.1-sol | 0 | 0.00 | 0.00 | 15 | 1.00 | 1.00 |
| 2 | claude-code | claude-opus-4-7 | 0 | 0.00 | 0.00 | 5 | 0.33 | 0.33 |
| 2 | react | claude-opus-4-7 | 0 | 0.00 | 0.00 | 4 | 0.33 | 0.11 |
| 2 | codex | gpt-5.6-sol | 1 | 0.22 | 0.00 | 7 | 0.64 | 0.34 |
| 2 | react | gpt-5.6-sol | 4 | 0.33 | 0.11 | 7 | 0.64 | 0.34 |
| 3 | claude-code | claude-opus-4-5-20251101 | 0 | 0.00 | 0.00 | 5 | 0.64 | 0.03 |
| 3 | react | claude-opus-4-5-20251101 | 0 | 0.00 | 0.00 | 4 | 0.33 | 0.11 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 1 | openhands, react | kimi-k3.1-pro | pending | | | pending | | |
| 2 | openhands | kimi-k2.5 | 0 | 0.00 | 0.00 | 2 | 0.45 | 0.00 |
| 2 | react | kimi-k2.5 | 0 | 0.00 | 0.00 | 1 | 0.22 | 0.00 |
| 3 | openhands | kimi-k2-0905 | 1 | 0.22 | 0.00 | 1 | 0.22 | 0.00 |
| 3 | react | kimi-k2-0905 | 0 | 0.00 | 0.00 | 2 | 0.31 | 0.00 |

89 of 480 trials pass (Claude and OpenAI 82 of 360, kimi-k2.5 and
kimi-k2-0905 7 of 120). Level 2 is almost unsolved: `make_9_lvl2` (a
16-reaction route) has no pass, `make_10_lvl2` 4 (react gpt-5.6-sol) and
`make_6_lvl2` 7. On level 3, gpt-6.1-sol solves 29 of 30 trials; the Claude
models mostly solve `make_7_lvl3`; gpt-5 solves none. The October runs found
the same level-2 tasks hard (`make_9_lvl2` 1 and `make_10_lvl2` 4 passes in 35
trials each).

172 trials end with an error, all kept as failures: 161 used up the 50-step
budget without a submission (Codex call budget, ReAct interaction budget,
Claude Agent SDK `error_max_turns`, OpenHands iteration limit), 5 OpenHands
runs finished without calling `submit_answer`, 5 prompts exceeded the model's
context limit, and the provider rejected 1 kimi-k2-0905 request as invalid. The other failures submitted a route the scorer rejects.

## Reused trials

- **`make_10_lvl2`** (90 trials) comes from the first v2 run (2026-11-10). The
  task is not affected by the database or buyables fixes.
- **80 trials** of claude-code/claude-opus-5-5, react/claude-opus-5-5,
  codex/gpt-5.6-sol and react/gpt-5.6-sol on `make_6_lvl2`, `make_6_lvl3`,
  `make_7_lvl3` and `make_8_lvl3` come from the 2026-11-12 rerun of the October
  full benchmark (same runner, images, database, settings and prompts; only the
  Langfuse session differs). They are part of the corrected October reports in
  [`../`](../).
- All other trials (290 so far) were run for these picks on 2026-11-13.

## Settings

k = 5, 50 iterations, reasoning effort `medium`, surrender off, Docker sandbox
with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Codex enforced the 50-call budget (#465). Kimi via OpenRouter (`max_tokens`
32768, no fallbacks): kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on
Novita; kimi-k2-0905 runs with the provider's default reasoning setting.

All trials used reaction database v2 (`ghcr.io/lamalab-org/corral-retro-db:v2`,
port 5432) and the regenerated `buyables.sqlite` with the #485 prices and
budgets (#486). The trials of 2026-11-12/13 ran on Corral at `c667e9b10`
(images built 2026-11-12); the reused `make_10_lvl2` trials on `ad800a2a5`
(including #475, which runs `verify_route` in the controller). The tools query
the `corral-retro-db` Postgres container through the Docker gateway
(`RETRO_DB_HOST=172.17.0.1`). The reference routes of all six tasks pass the
scorer on this database and buyables set.

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's trials share one session named after the merged run
ID (`retrosynthesis-mini-v2-l<L>-<agent>-<model>`, in `run-metadata.json`);
the 80 reused full-rerun trials keep their session
(`retrosynthesis-full-rerun-20261112-l<L>-<agent>-<model>`, listed under
`langfuse_session_ids`). All 480 traces were found (`build_index.py --verify`).

## Caveats

- **66 OpenAI trials were rerun** after the OpenAI account ran out of credit;
  the reports hold the reruns only.
- **Tools.** No trial log shows a database connection error.
- **Machine load.** Runs shared the machine with the spectra pools; wall-clock
  durations are not comparable with other runs.
- **Comparison with the October full runs.** The full runs cover all 10
  tasks per level; compare per task only.
