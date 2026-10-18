# Spectra elucidation reports

14 runs, 1,400 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 1 | 97/100 | 0 | 0 | [report](level_1/ai-scientist__claude-opus-5-5/report.json.xz) |
| ai-scientist | claude-opus-5-5 | 2 | 96/100 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.xz) |
| claude-code | claude-opus-5-5 | 1 | 80/100 | 0 | 0 | [report](level_1/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 2 | 73/100 | 0 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 1 | 79/100 | 1 | 0 | [report](level_1/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 75/100 | 1 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 1 | 81/100 | 0 | 0 | [report](level_1/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 2 | 81/100 | 0 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 1 | 77/100 | 0 | 0 | [report](level_1/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 2 | 73/100 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 1 | 89/100 | 0 | 0 | [report](level_1/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 2 | 75/100 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 1 | 87/100 | 0 | 0 | [report](level_1/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 2 | 88/100 | 0 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |

## Caveats

The retained trials include IR tool failures from the underlying
spectral-prediction API. Affected calls returned `IR prediction failed` instead
of an IR spectrum; the same message can appear in the IR field of combined
spectral predictions. These failures remain in the reported traces after retries
and may have limited the spectral evidence available to the agent.

The tool returns this diagnostic as text, so a call can be recorded as completed
without producing a spectrum or setting a terminal trial error. Consequently,
the table's Errors column does not capture all IR API failures.
