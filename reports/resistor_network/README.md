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
| react | gpt-5.6-sol | 44/50 | 2 | 28/50 | 11 | Docker |
| react | kimi-k3 | 10/50 | 11 | 1/50 | 19 | Docker |

Errors count as failures. The final selection has no remaining infrastructure
failures. Remaining errors are interaction/turn budgets, GPT context exhaustion,
and a malformed GPT tool name. OpenHands Kimi K3 hit its iteration limit in
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
session and trace IDs. `technical-reruns.csv` maps all nine original slots to
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
