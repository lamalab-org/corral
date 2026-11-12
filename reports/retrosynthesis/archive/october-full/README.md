# Retrosynthesis reports (archived: October full runs)

Archived on 2026-11-10 when the corral-mini v2 runs replaced them in
[`reports/retrosynthesis/`](../../). These are the full-benchmark runs
(10 tasks per level) published in the October Hugging Face datasets.

14 runs, 700 trials. Layout and index columns follow
[`reports/README.md`](../../../README.md). Results include the supplied reports and
subsequent rescoring and reruns, including the 2026-11-12 reruns of the tasks
changed by the reaction-database and buyables fixes (see Corrections below).
Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../../../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 2 | 27/50 | 12 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 3 | 26/50 | 21 | 0 | [report](level_3/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 2 | 36/50 | 2 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 3 | 27/50 | 9 | 0 | [report](level_3/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 38/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 3 | 28/50 | 13 | 0 | [report](level_3/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 2 | 26/50 | 11 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 3 | 17/50 | 22 | 0 | [report](level_3/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 2 | 27/50 | 10 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 3 | 25/50 | 12 | 0 | [report](level_3/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 2 | 38/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 3 | 28/50 | 17 | 0 | [report](level_3/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 2 | 31/50 | 10 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 3 | 23/50 | 14 | 0 | [report](level_3/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Corrections (2026-11-12): tasks changed by the database and buyables fixes

Two fixes changed what some tasks ask for or how they can be solved:

- **Reaction database v2** (`ghcr.io/lamalab-org/corral-retro-db:v2`, now the
  default, #487). Template 1914384, the only template of `make_1`, had the wrong
  product in v1, so `make_1` could not be solved at either level.
- **Buyables and budgets** (#485, #486). Six reference-route leaves gained
  prices, and the level-3 budgets of `make_5` to `make_8` changed from 9999 to
  110% of the reference cost (stated in the prompt). The `is_buyable` tool now
  answers differently for those six compounds.

`make_1`, `make_5`, `make_6`, `make_7` and `make_8` were therefore rerun at
both levels with the October settings (same agents, models, reasoning effort
`medium`, 50 steps, k = 5) on current Corral, one `corral bench --trials 1` run
per trial (AI Scientist with 100 steps, as in October, after the Docker
checkpoint fix #491). One level-3 `make_9` trial (claude-code k2) that had asked
`is_buyable` about a newly priced compound was rerun too. Tasks `make_2`,
`make_3`, `make_4`, `make_10` and level-2 `make_9` are unaffected and keep their
October trials.

| Agent | Model | Rerun trials | Status |
|---|---|---:|---|
| claude-code | claude-opus-5-5 | 51 | done |
| codex | gpt-5.6-sol | 50 | done |
| react | claude-opus-5-5 | 50 | done |
| react | gpt-5.6-sol | 50 | done |
| openhands | kimi-k3 | 50 | done |
| react | kimi-k3 | 50 | done |
| ai-scientist | claude-opus-5-5 | 50 | done (100 steps, as in October) |

Each updated run report records the replaced trials and the rerun source runs
under `metadata.reruns_20261112`; rerun trials keep their own execution IDs and
Langfuse session (`retrosynthesis-full-rerun-20261112-l<L>-<agent>-<model>`,
recorded per trial in `run-metadata.json`). All 700 traces were found
(`build_index.py --verify`). The rerun trials ran on current Corral, which
differs from the October code in ways that can change agent behaviour (for
example #456 prompts, #465 Codex step budget, #475 `verify_route` in the
controller); kimi-k3 again ran on the Together provider, now with
`max_tokens` 32768; the unchanged tasks keep their October trials.

Aggregate passes: **315/700 before, 397/700 after**; all 351 rerun trials are in.

## Caveats

On 2026-10-23, the corrected catalog reaction validator was applied to the
**33 trials flagged in the retrosynthesis audit**. The 24 confirmed passes and
4 template-granularity passes changed from 1 to 0; the other 5 remained 0
(including 4 without a saved submission). Aggregate passes changed from
**343/700 (49%) to 315/700 (45%)**.

Only these 33 trials were selected. The other 667 trials retain their original
evaluations, so these are partially rescored results, not a full evaluation of
all 700 trials under the corrected scorer. The four template-granularity cases
remain classified as suspicious in the original audit: their submitted catalog
steps fail the strict check, which does not establish intentional cheating.

Rejection reasons are recorded in each selected trial's evaluation metadata.

Messages, submitted routes, execution IDs, timings, and token usage were
preserved. Langfuse links still point to the original executions. The local
reports and indexes contain the corrected scores; hosted copies have not been
republished.
