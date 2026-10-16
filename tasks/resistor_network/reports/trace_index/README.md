# Final versus archived traces

Use `../v4_original/final-trials.csv` or `../v4_terminal/final-trials.csv` to
select reported trials. Both map each row to a Langfuse session ID, trace ID,
local checkpoint, report and evaluation commit hash. Shared native rows appear
in both batches but refer to the same executions.

`archived-traces.csv.gz` is a separate inventory of superseded/test/excluded
trial copies. `archive-manifest.json` maps old external paths to their archive
locations. The archive retains full trace files; they are not committed here.

IDs alone cannot separate archived attempts that reused an execution ID.
Use the final local trial and its report/evaluation hash for exact attribution.
Remote Langfuse deletion or replay was not performed.
