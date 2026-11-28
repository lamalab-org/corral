# Stargazer reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 1 | 30/50 | 0 | 0 | [report](level_1/ai-scientist__claude-opus-5-5/report.json.gz) |
| ai-scientist | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 1 | 27/50 | 0 | 0 | [report](level_1/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 1 | 25/50 | 0 | 0 | [report](level_1/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 0/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 1 | 23/50 | 2 | 0 | [report](level_1/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 2 | 0/50 | 9 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 1 | 28/50 | 0 | 0 | [report](level_1/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 1 | 29/50 | 0 | 0 | [report](level_1/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 2 | 0/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 1 | 28/50 | 3 | 0 | [report](level_1/react__kimi-k3/report.json.gz) |
| react | kimi-k3 | 2 | 0/50 | 11 | 0 | [report](level_2/react__kimi-k3/report.json.gz) |

Errors count nonempty saved execution/evaluation error fields, including
iteration-limit or answer-format outcomes. CSV trial positions are zero-based.

## Caveats

The retained trials include 128 pre-execution guard rejections across 72 trials
(36 per level). Stargazer's `PythonREPL` tool wrapper included a safeguard
intended to catch accidental replacement of functions with data, such as passing
a number or array to an optimizer that expects a callable. It checked names
before executing a cell and returned a diagnostic when it detected a conflict.

The check used previously stored global values without correctly accounting for
local function parameters or new callable definitions in the submitted cell.
For example:

```python
# Previous cell
from scipy.optimize import minimize

f = 7

# Later cell: valid Python that replaces f with a callable
f = lambda x: (x[0] - 1) ** 2
minimize(f, [0.0])
```

Perhaps a warning would be more appropriate than an outright rejection, allowing the user to proceed with the cell execution while still being informed of potential conflicts.

## Technical rerun (2026-11-28)

One Level 2 trial ended on infrastructure and was replaced: OpenHands kimi-k3
`seed176_diff9`, trial 0. Its run stopped on an OpenRouter 429 ("kimi-k3 is
temporarily rate-limited upstream"). The replacement ran with the October
settings (OpenRouter pinned to Together, reasoning effort `medium`, 50
iterations, 8 GB) at `main` 9269652a4, and also scored 0, so the Level 2 numbers
are unchanged (0/50). The original trial's summary and the reason are kept in
the report metadata (`technical_reruns_20261128`); the replacement's Langfuse
session is set in `run-metadata.json` (`langfuse_session_ids`).

Two Codex gpt-5.6-sol Level 1 trials (`seed300025_diff1` trial 4, `seed8_diff4`
trial 1) contain an OpenAI "no credits" error in their history, but the harness
recovered and both passed. They are not infrastructure failures and are left
unchanged.
