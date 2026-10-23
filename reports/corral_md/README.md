# Corral MD reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations and manual score assignments
are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 1 | 15/50 | 0 | 0 | [report](level_1/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 2 | 7/50 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 1 | 16/50 | 2 | 0 | [report](level_1/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 2 | 8/50 | 7 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 1 | 21/50 | 0 | 0 | [report](level_1/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 9/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 1 | 1/50 | 20 | 0 | [report](level_1/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 2 | 0/50 | 30 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 1 | 28/50 | 0 | 0 | [report](level_1/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 2 | 10/50 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 1 | 30/50 | 0 | 0 | [report](level_1/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 2 | 13/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 1 | 12/50 | 7 | 0 | [report](level_1/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 2 | 7/50 | 23 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Caveats

None after retries.
