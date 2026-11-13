# AFM benchmark reports

14 runs, each covering 10 tasks × 5 trials: 700 trials total.
The Level 1 AI Scientist report combines the separate tasks 1–5 and 6–10 runs.

Layout and indexes follow [`../README.md`](../README.md).

| Level | Agent | Model | Tasks | Passes | Error trials | Average score |
|---|---|---|---:|---:|---:|---:|
| 1 | ai-scientist | claude-opus-5-5 | 10 | 48/50 | 1 | 0.96 |
| 1 | claude-code | claude-opus-5-5 | 10 | 47/50 | 0 | 0.94 |
| 1 | codex | gpt-5.6-sol | 10 | 47/50 | 0 | 0.94 |
| 1 | openhands | kimi-k3 | 10 | 43/50 | 1 | 0.86 |
| 1 | react | claude-opus-5-5 | 10 | 37/50 | 0 | 0.74 |
| 1 | react | gpt-5.6-sol | 10 | 43/50 | 0 | 0.86 |
| 1 | react | kimi-k3 | 10 | 38/50 | 6 | 0.76 |
| 2 | ai-scientist | claude-opus-5-5 | 10 | 49/50 | 0 | 0.98 |
| 2 | claude-code | claude-opus-5-5 | 10 | 50/50 | 0 | 1.0 |
| 2 | codex | gpt-5.6-sol | 10 | 49/50 | 0 | 0.98 |
| 2 | openhands | kimi-k3 | 10 | 39/50 | 4 | 0.78 |
| 2 | react | claude-opus-5-5 | 10 | 42/50 | 0 | 0.84 |
| 2 | react | gpt-5.6-sol | 10 | 35/50 | 0 | 0.7 |
| 2 | react | kimi-k3 | 10 | 42/50 | 1 | 0.84 |

## Provenance and limitations

These reports replace the previous AFM report collection. Compression is lossless:
original bytes are preserved for the thirteen unmerged reports. No trials were
rescored, removed or substituted. Their original `run-metadata.json` files and
`provenance.json` source hashes accompany them. Previous reports remain in Git history.

The Level 1 AI Scientist report merges `opus-ai-1-5` and `apus-ai-6-10`.
All task results and trial objects remain unchanged. Aggregate metrics were
recalculated with Corral’s metric registry after verifying it reproduces both
source reports. Its duration is the sum of the two run durations, excluding the
gap between runs. Source hashes, launcher metadata and original report metadata
are recorded in `provenance.json`; the combined launcher metadata is explicitly
marked as a composition.

`afm-opus-aiscientist-l1-combined-001` is a synthetic report ID, not a Langfuse
session. Trial execution IDs and their source session references are preserved.
The generated run-level session link for this synthetic ID does not represent
an uploaded session; use trial-level links for the two original sessions.

Saved failures remain included. Error counts describe saved terminal errors; they
do not establish whether a failure was caused by the agent or infrastructure.
The reports have not been audited for rerun eligibility.

`runs.csv` and `trials.csv` are generated with `reports/build_index.py`. Langfuse
links are derived from execution IDs using the shared report convention; their
existence and project membership have not been verified. Importing these reports
does not upload traces to Langfuse. This update has not been synchronized to
the Hugging Face datasets.

## Validation

```bash
python reports/build_index.py reports/afm
python reports/check_reports.py
```
