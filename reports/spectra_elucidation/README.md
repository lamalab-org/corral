# Spectra elucidation reports

12 runs, 1,200 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 1 | 97/100 | 0 | 0 | [report](level_1/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 2 | 96/100 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 1 | 80/100 | 0 | 0 | [report](level_1/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 2 | 73/100 | 0 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 1 | 79/100 | 1 | 0 | [report](level_1/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 75/100 | 1 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| react | claude-opus-5-5 | 1 | 77/100 | 0 | 0 | [report](level_1/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 2 | 73/100 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 1 | 89/100 | 0 | 0 | [report](level_1/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 2 | 75/100 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 1 | 87/100 | 0 | 0 | [report](level_1/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 2 | 88/100 | 0 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Caveats

None after retries.
