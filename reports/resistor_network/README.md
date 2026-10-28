# Resistor network v4 Docker-sandboxed reports

All 14 runs (7 agent/model pairs × 2 levels; 10 tasks × 5 trials) in this report set used Docker sandboxing. The Codex, Claude Code, and OpenHands reports below replace their earlier outside-Docker runs. ReAct and AI Scientist reports are unchanged because those runs already used Docker.

| Agent | Model | L1 passes | L1 error trials | L2 passes | L2 error trials | Sandbox |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 49/50 | 0 | 33/50 | 0 | Docker |
| claude-code | claude-opus-5-5 | 40/50 | 10 | 12/50 | 37 | Docker |
| codex | gpt-5.6-sol | 47/50 | 0 | 19/50 | 0 | Docker |
| openhands | kimi-k3 | 10/50 | 40 | 0/50 | 50 | Docker |
| react | claude-opus-5-5 | 39/50 | 0 | 27/50 | 9 | Docker |
| react | gpt-5.6-sol | 45/50 | 1 | 28/50 | 11 | Docker |
| react | kimi-k3 | 10/50 | 11 | 1/50 | 19 | Docker |

Scores count exact circuit matches after simulation-based equivalence checking; a score below 1 is a failed trial. Error-trial counts include harness/model errors and iteration or turn limits, while incorrect submissions without a harness error can also score 0. OpenHands reached its 50-iteration limit in 87 of 100 selected trials; two other trials ended without a final submission.

The Claude Code, Codex, and OpenHands Docker results use 2 CPUs, 4 GiB RAM, and a 256-process limit per container. Their final reports select one result for each of 100 trial slots per agent after infrastructure recovery. `native-agent-attempts.csv` records the 380 Docker execution attempts for these three agents (300 initial trial runs and 80 retries). `technical-reruns.csv` maps selected replacements across this report set, including the earlier Docker ReAct and AI Scientist reruns.

The outside-Docker runs are not part of this report set. Their workspaces could expose benchmark files, and the audit found evidence of hidden-answer access in some traces.

Layout, names, and index columns follow [`../README.md`](../README.md). Rebuild `runs.csv` and `trials.csv` with `python reports/build_index.py reports/resistor_network`.
