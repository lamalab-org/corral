# Resistor network reports (v4, terminal)

14 runs (7 agent/model pairs × 2 levels), 10 tasks × k = 5, 700 trials.
Layout, names and index columns follow [`../README.md`](../README.md).

| Agent | Model | L1 passes | L1 errors | L2 passes | L2 errors | Terminal |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 49/50 | 0 | 33/50 | 0 | Docker |
| claude-code | claude-opus-5-5 | 50/50 | 0 | 39/50 | 11 | native |
| codex | gpt-5.6-sol | 48/50 | 0 | 38/50 | 0 | native |
| openhands | kimi-k3 | 9/50 | 41 | 3/50 | 47 | native |
| react | claude-opus-5-5 | 39/50 | 0 | 27/50 | 9 | Docker |
| react | gpt-5.6-sol | 45/50 | 1 | 28/50 | 11 | Docker |
| react | kimi-k3 | 10/50 | 11 | 1/50 | 19 | Docker |

Errors count as failures. The final selection has no remaining infrastructure
failures. Remaining errors are interaction/turn budgets and GPT context
exhaustion. OpenHands Kimi K3 hit its iteration limit in
88 of 100 trials.

Settings that differ between runs:

- Claude Code and Codex trials that hit a 30-minute trial deadline (84 in all)
  were rerun without it.
- react Kimi K3 ran through OpenRouter pinned to the Fireworks provider with low
  reasoning effort and a 32,000-token cap; 77 of its trials were restarted
  after failures with an earlier provider.
- Docker trials had 2 CPUs, 4 GiB RAM and 256 PIDs; native agents had no cap.

## Separate-ID technical reruns

Nine infrastructure failures received one fresh attempt each on 2026-10-16:
one Codex interruption, three ReAct Opus provider errors, three ReAct Opus
memory failures and two AI Scientist artifact-permission failures. Four passed,
four submitted incorrect answers, and one exhausted its interaction budget.
No provider, memory or artifact-permission failure recurred. Models, task bank,
budgets and resource limits were preserved; there was no whole-trial deadline.

The designated rerun replaces the failed trial in each affected report, with
all task/run metrics recomputed. Each report still has 50 trials. Reports keep
the original batch run ID, while substituted trials keep their new execution,
session and trace IDs. `technical-reruns.csv` maps all ten original slots to
those separate reruns. `trials.csv` points directly to their Langfuse traces.
Historical attempts remain in external run storage and Git history.

Assembled reports record replacement provenance in
`metadata.technical_recovery`. They omit `total_benchmark_duration`, because
these selected trials were executed in separate runs; original wall duration
is retained in recovery metadata. Trial durations and all aggregate metrics
are calculated from the final selection.

All nine new trace IDs were verified present in their separate Langfuse
sessions on 2026-10-16. All 700 CSV rows were checked against the selected
report trials, scores and execution IDs.

On 2026-10-20, ReAct GPT level 1 task_9 trial index 0 received one separate-ID
rerun after a harness robustness fix. The original model emitted a malformed
action name that the harness forwarded into API message metadata, ending the
trial without corrective feedback. The patched image validates action names
and returns feedback before dispatch. All other image contents, model options,
budgets and resource limits were unchanged. The rerun passed (score 1), so
ReAct GPT level 1 is now 45/50 with one interaction-budget error remaining.
The original 49 other trials are unchanged. The fix is covered by regression
tests; this fresh run did not itself emit a malformed action name.
