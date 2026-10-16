# v4 terminal comparison reports

12 reports, 600 trials. Kimi ReAct terminal runs are excluded because provider
and credit failures remain unresolved. Shared OpenHands Kimi results are included.

| Agent/model | Level 1 passes | Level 2 passes |
|---|---:|---:|
| ReAct GPT | 44/50 | 28/50 |
| ReAct Opus | 38/50 | 26/50 |
| AI Scientist Opus | 49/50 | 32/50 |
| Codex GPT (shared) | 48/50 | 37/50 |
| Claude Code Opus (shared) | 50/50 | 39/50 |
| OpenHands Kimi (shared) | 9/50 | 3/50 |

`summary.csv` / `summary.json` list all run IDs and results. `final-trials.csv`
contains exactly the trials in these reports, including their scores, errors,
Langfuse session/trace IDs, local paths, and evaluation/ledger commit hashes.

Each `report.json.gz` is a lossless copy of the source report. Use
`gzip -dc report.json.gz` to read it. Summary SHA-256 values cover the original
uncompressed bytes. Run metadata is included beside each report.

Codex, Claude Code and OpenHands reports are byte-identical to the original
batch; their execution IDs and Langfuse IDs are intentionally the same. They
were not rerun for the terminal comparison. ReAct and AI Scientist used Docker
terminals. Resource and recovery differences are recorded in `provenance.json`.

Superseded attempts and test runs are archived externally. Final reported
failures remain in the reports and current trial directories. An archive-copy
count flags trace IDs also used by archived copies; some copies are interrupted
checkpoints or backups rather than separate model attempts. Langfuse traces
with shared IDs may mix observations. The indexed local checkpoint and report
are authoritative. Local delivery receipts are not a fresh remote availability
check. Some infrastructure-error trials have only an unfinished ledger head.
