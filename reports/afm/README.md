# AFM benchmark reports

14 runs and 675 trials imported from the supplied `runs` folder. Thirteen runs
cover 10 tasks × 5 trials. The Level 1 AI Scientist run covers only tasks 1–5
(5 tasks × 5 trials); it is a partial benchmark and is not directly comparable
to the full-level runs.

Layout and indexes follow [`../README.md`](../README.md).

| Level | Agent | Model | Tasks | Passes | Error trials | Average score |
|---|---|---|---:|---:|---:|---:|
| 1 | ai-scientist | claude-opus-5-5 | 5 | 23/25 | 1 | 0.92 |
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
original report bytes, metrics, task results, execution IDs and run IDs are preserved.
No trials were rescored, removed or substituted. Each run includes its original
`run-metadata.json` and a `provenance.json` mapping to the source folder with SHA-256
hashes of both source files. Previous reports remain in Git history.

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
