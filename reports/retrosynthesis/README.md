# Retrosynthesis reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 2 | 16/50 | 19 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 3 | 15/50 | 15 | 0 | [report](level_3/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 2 | 34/50 | 3 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 3 | 21/50 | 12 | 0 | [report](level_3/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 34/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 3 | 21/50 | 1 | 0 | [report](level_3/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 2 | 24/50 | 9 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 3 | 16/50 | 24 | 0 | [report](level_3/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 2 | 23/50 | 16 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 3 | 16/50 | 28 | 0 | [report](level_3/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 2 | 36/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 3 | 21/50 | 14 | 0 | [report](level_3/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 2 | 23/50 | 19 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 3 | 15/50 | 29 | 0 | [report](level_3/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

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
