# AFM benchmark reports

4 runs, each with 10 tasks × 5 trials: 200 trials total.
Layout, names and index columns follow [`../README.md`](../README.md).

| Level | Agent | Model | Passes | Error trials | Average score |
|---|---|---|---:|---:|---:|
| 1 | codex | gpt-5.6-sol | 24/50 | 0 | 0.48 |
| 1 | openhands | kimi-k3 | 10/50 | 30 | 0.20 |
| 1 | react | gpt-5.6-sol | 28/50 | 1 | 0.56 |
| 2 | react | gpt-5.6-sol | 20/50 | 0 | 0.40 |

Reports were imported from the supplied `afm-report` folder. Compressed files
preserve the original JSON bytes, including run IDs, trial IDs and metrics.
No trials were replaced or removed. Error trials remain in the reports;
their causes have not been audited for infrastructure reruns.

`runs.csv` and `trials.csv` were generated using `reports/build_index.py`.
Langfuse tracing was not enabled for these runs and no traces were uploaded.
The index script generates Langfuse links from execution IDs using the shared
report convention, but these links do not correspond to uploaded traces.
These reports have not been published to Hugging Face.

To regenerate the indexes and check the local report layout:

```bash
python reports/build_index.py reports/afm
python reports/check_reports.py
```
