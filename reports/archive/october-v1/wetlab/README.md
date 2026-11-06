# Wetlab reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 2 | 20/50 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 3 | 9/50 | 0 | 0 | [report](level_3/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 2 | 48/50 | 0 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 3 | 44/50 | 0 | 0 | [report](level_3/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 42/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 3 | 31/50 | 0 | 0 | [report](level_3/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 2 | 14/50 | 5 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 3 | 11/50 | 9 | 0 | [report](level_3/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 2 | 48/50 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 3 | 37/50 | 0 | 0 | [report](level_3/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 2 | 41/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 3 | 26/50 | 0 | 0 | [report](level_3/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 2 | 16/50 | 13 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 3 | 7/50 | 23 | 0 | [report](level_3/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Caveats

None after retries.
