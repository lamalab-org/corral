# Psychometrics reports

These full-task Psychometrics reports replace the October full-task results with
runs against the 20-task revision from PR #481. They are archived here because
the root Psychometrics directory now contains corral-mini v1 reports.

It contains 14 runs (7 agent/model pairs × 2 levels), with all 10 tasks × 5
trials at each level (700 trials total).
Kimi K3.1 Pro is excluded.

| Agent | Model | L1 passes /50 | L1 errors | L1 Pass@5 | L2 passes /50 | L2 errors | L2 Pass@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| ai-scientist | claude-opus-5-5 | 4 | 23 | 0.1662 | 1 | 46 | 0.0672 |
| claude-code | claude-opus-5-5 | 22 | 2 | 0.5661 | 25 | 0 | 0.7246 |
| codex | gpt-5.6-sol | 16 | 0 | 0.3672 | 16 | 0 | 0.4344 |
| openhands | kimi-k3 | 2 | 16 | 0.1345 | 6 | 7 | 0.2584 |
| react | claude-opus-5-5 | 16 | 0 | 0.4584 | 20 | 4 | 0.5334 |
| react | gpt-5.6-sol | 16 | 5 | 0.4344 | 20 | 6 | 0.5651 |
| react | kimi-k3 | 9 | 22 | 0.3334 | 18 | 12 | 0.4911 |

`passes /50` counts trials with a positive score. `errors` counts trials with a
saved terminal error; scores and pass@k are calculated from the report's saved
trial scores. Reports include verbose output and ran in Docker sandbox mode.
The adjacent `provenance.json` records the model settings and image digest;
`run-metadata.json` records the task limits and source commit.

**Source:** six agent/model pairs used `e601ae03b`; AI Scientist used merged main
`541230efa`, with identical Psychometrics tasks, datasets, and scoring code.
The AI Scientist reports replace the older-task runs from `4ec6f582`.

AI Scientist used Claude Opus 5.5, `max_iterations=100`,
`max_tool_calls=256`, `max_actions_per_node=256`, `max_search_nodes=8`, and
`max_validation_nodes=8`, reasoning effort `medium`, and temperature 1.0.
Both levels have no scoring errors and all 100 Langfuse traces were verified.
The 23 L1 and 46 L2 “LLM-call budget exhausted (100)” errors count as failures.

Kimi K3 runs used default reasoning settings (OpenHands records the provider
SDK default as high); no Kimi K3.1 Pro reports are included.
