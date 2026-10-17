# Retrosynthesis reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 2 | 17/50 | 19 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 3 | 15/50 | 15 | 0 | [report](level_3/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 2 | 39/50 | 3 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 3 | 21/50 | 12 | 0 | [report](level_3/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 39/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 3 | 21/50 | 1 | 0 | [report](level_3/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 2 | 28/50 | 9 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 3 | 16/50 | 24 | 0 | [report](level_3/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 2 | 30/50 | 16 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 3 | 17/50 | 28 | 0 | [report](level_3/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 2 | 38/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 3 | 22/50 | 14 | 0 | [report](level_3/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 2 | 25/50 | 19 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 3 | 15/50 | 29 | 0 | [report](level_3/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Caveats

None after retries.
