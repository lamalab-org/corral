# Resistor v4 evaluation reports

- `v4_original/`: original 14-run batch (700 trials), including the corrected
  native-agent deadline-free replacement results. ReAct and AI Scientist had
  no terminal in this batch; coding agents used their native terminals.
- `v4_terminal/`: separate terminal comparison (12 runs, 600 trials), excluding
  the unresolved ReAct Kimi terminal runs. Codex, Claude Code and OpenHands
  reuse exactly the same 300 native trials and IDs from the original batch.
- `trace_index/`: archive inventory, relocation map, and verification results.

Each batch has `summary.csv`, compressed full reports, and `final-trials.csv`.
Use the final-trial table to find a reported trial's Langfuse trace/session ID,
local checkpoint, error category and evaluation commit hash. Archived copies
are indexed separately and never included as extra final-trial rows.

The two tables contain 1,000 unique executions after deduplicating the 300
shared native trials. Existing final-report errors are retained, not removed
from score denominators. Langfuse IDs can be shared with superseded attempts;
use the local report/checkpoint for exact attribution. Remote availability was
not independently checked and no Langfuse data was deleted.
