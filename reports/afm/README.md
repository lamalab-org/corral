# AFM benchmark reports

5 current runs, each with 10 tasks × 5 trials: 250 trials total.
Layout, names and index columns follow [`../README.md`](../README.md).

| Level | Agent | Model | Passes | Error trials | Average score |
|---|---|---|---:|---:|---:|
| 1 | codex | gpt-5.6-sol | 47/50 | 0 | 0.94 |
| 1 | react | claude-opus-5-5 | 37/50 | 0 | 0.74 |
| 1 | react | gpt-5.6-sol | 43/50 | 0 | 0.86 |
| 2 | react | claude-opus-5-5 | 42/50 | 0 | 0.84 |
| 2 | react | gpt-5.6-sol | 35/50 | 0 | 0.70 |

These are the scores and terminal error counts saved in the supplied reports.
Reports use medium reasoning effort and local execution. ReAct runs record a
50-iteration budget, one parallel task and one attempt per trial. Claude runs
use the direct Anthropic model ID; ReAct GPT runs use OpenRouter; Codex uses
the native harness. See each report and its `run-metadata.json` for
full configuration and original Windows output paths.

The compressed reports preserve the supplied JSON bytes, run IDs, trial IDs,
and metrics. Metadata contents are preserved with line endings normalized to LF.
No trials were removed, replaced or rescored during import.
A zero terminal error count does not establish the absence of infrastructure
issues or access to hidden benchmark files. These local runs have not received
an independent transcript or filesystem-isolation audit as part of this import.

The previous 4 runs (200 trials) are retained unchanged under [`old/`](old/),
including their historical README and indexes. They are excluded from the
current indexes and totals and should not be mixed with these newer runs.
Paths in the archived CSVs are relative to `old/`.

`runs.csv` and `trials.csv` were generated using `reports/build_index.py`.
Langfuse links use the repository's default project and deterministic ID
convention; trace availability and the destination project have not been
verified for these imports. The new reports have not been published to
Hugging Face as part of this update.

To regenerate and validate the current indexes:

```bash
python reports/build_index.py reports/afm
python reports/check_reports.py
```
