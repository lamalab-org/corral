# Original v4 evaluation reports

Final reports for seven agent/model configurations, two levels, ten tasks per
level and five trials per task: **700 trials across 14 runs**.

`summary.csv` and `summary.json` contain the scores and Langfuse session IDs.
Each run directory contains the original `report.json.gz` (lossless gzip) and
its `run-metadata.json`. Read a report with `gzip -dc report.json.gz` or Python's
`gzip.open`. SHA-256 hashes in the summary cover the uncompressed source bytes.

The original batch is not uniformly without terminal access: Codex, Claude
Code and OpenHands use native tools; ReAct and AI Scientist do not have a
terminal in this batch. Docker terminal variants are excluded.

84 Codex/Claude Code trials originally ended at a 30-minute trial deadline.
These reports replace those outcomes with completed reruns without that
whole-trial deadline. Existing non-timeout outcomes were retained. Native
terminal defaults and remaining model/iteration settings were retained.

ReAct Kimi includes both original and updated settings, as recorded in
`provenance.json`; this should be considered when comparing scores.
Archived attempts are excluded from reports but remain in external storage.

`final-trials.csv` maps all 700 reported trials to Langfuse IDs, local trace
paths, report/evaluation hashes and archived-copy flags. The separate terminal
comparison is in `../v4_terminal/`; shared native trials retain the same IDs.
