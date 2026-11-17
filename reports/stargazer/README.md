# Stargazer reports

14 runs, 700 trials. Layout and index columns follow
[`../README.md`](../README.md). Results include the supplied reports and
subsequent rescoring and reruns. Final evaluations are retained in the run reports.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv) ·
[Import provenance](../IMPORT_NOTES.md)

| Agent | Model | Level | Passes | Errors | Unscored | Report |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 1 | 44/50 | 0 | 0 | [report](level_1/ai-scientist__claude-opus-5-5/report.json.gz) |
| ai-scientist | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/ai-scientist__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 1 | 41/50 | 0 | 0 | [report](level_1/claude-code__claude-opus-5-5/report.json.gz) |
| claude-code | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/claude-code__claude-opus-5-5/report.json.gz) |
| codex | gpt-5.6-sol | 1 | 36/50 | 0 | 0 | [report](level_1/codex__gpt-5.6-sol/report.json.gz) |
| codex | gpt-5.6-sol | 2 | 0/50 | 0 | 0 | [report](level_2/codex__gpt-5.6-sol/report.json.gz) |
| openhands | kimi-k3 | 1 | 31/50 | 2 | 0 | [report](level_1/openhands__kimi-k3/report.json.gz) |
| openhands | kimi-k3 | 2 | 0/50 | 9 | 0 | [report](level_2/openhands__kimi-k3/report.json.gz) |
| react | claude-opus-5-5 | 1 | 39/50 | 0 | 0 | [report](level_1/react__claude-opus-5-5/report.json.gz) |
| react | claude-opus-5-5 | 2 | 0/50 | 0 | 0 | [report](level_2/react__claude-opus-5-5/report.json.gz) |
| react | gpt-5.6-sol | 1 | 38/50 | 0 | 0 | [report](level_1/react__gpt-5.6-sol/report.json.gz) |
| react | gpt-5.6-sol | 2 | 0/50 | 0 | 0 | [report](level_2/react__gpt-5.6-sol/report.json.gz) |
| react | kimi-k3 | 1 | 39/50 | 3 | 0 | [report](level_1/react__kimi-k3/report.json.gz) |
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

## Calibrated complete scorer (PR #504, 2026-11-17)

Rescoring the retained October final answers with the complete scorer's new
0.60 mean match cutoff and 0.35 per-planet floor increases Level 1 from
190/350 (54.29%) to 268/350 (76.57%). Level 2 stays at 0/350. All 78 changes
are failures becoming passes. The previous complete criteria reproduce every
saved score for the 675 trials with answers; the 25 trials without a saved
answer retain their original execution outcomes. No agent was rerun.

| Agent | Model | L1 before | L1 after | L2 before → after |
|---|---|---:|---:|---:|
| ai-scientist | claude-opus-5-5 | 30/50 | 44/50 | 0 → 0/50 |
| claude-code | claude-opus-5-5 | 27/50 | 41/50 | 0 → 0/50 |
| codex | gpt-5.6-sol | 25/50 | 36/50 | 0 → 0/50 |
| openhands | kimi-k3 | 23/50 | 31/50 | 0 → 0/50 |
| react | claude-opus-5-5 | 28/50 | 39/50 | 0 → 0/50 |
| react | gpt-5.6-sol | 29/50 | 38/50 | 0 → 0/50 |
| react | kimi-k3 | 28/50 | 39/50 | 0 → 0/50 |

Each rescored trial retains its previous evaluation alongside the new one.
Report metadata records the source report SHA-256, scorer commit, and criteria;
execution IDs, outputs, messages, errors, timing and token usage are preserved.
The original execution version in run provenance still describes execution,
while the new evaluation fields identify the offline scorer.

The [trial audit](provenance/scorer-504/trials.csv) records all before/after
scores. The [mini selection audit](provenance/scorer-504/mini-selection.json)
records reselection with v2's existing Rank + cost, Pass^5, unsaturated-first,
k=3, seed=0 method. L1 changes to `seed15_diff5`, `seed21_diff5`, `seed503_diff2`
(Spearman 0.9273); L2 stays `seed122_diff10`, `seed176_diff9`, `seed1_diff8`.
The L2 constant zero score has no informative agent ranking.

Reproduce from the original reports (the rescoring script refuses to rescore
already updated reports):

```bash
PYTHONPATH=src:tasks/stargazer/src python scripts/rescore_stargazer_october.py \
  --write --audit-dir reports/stargazer/provenance/scorer-504
python scripts/reselect_stargazer_mini.py \
  --write --audit-dir reports/stargazer/provenance/scorer-504
```
