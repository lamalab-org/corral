# Resistor network v4 results

All results in this report set come from agents running in Docker sandboxes. It contains 14 runs: seven agent/model combinations, each tested on levels 1 and 2. Every run has 50 trials per level.

| Agent | Model | Level 1 | Level 2 |
|---|---|---:|---:|
| AI Scientist | Claude Opus 5.5 | 49/50 | 33/50 |
| Claude Code | Claude Opus 5.5 | 40/50 | 12/50 |
| Codex | GPT-5.6-Sol | 47/50 | 19/50 |
| OpenHands | Kimi K3 | 10/50 | 0/50 |
| ReAct | Claude Opus 5.5 | 39/50 | 27/50 |
| ReAct | GPT-5.6-Sol | 45/50 | 28/50 |
| ReAct | Kimi K3 | 10/50 | 1/50 |

Each value is the number of trials that produced a correct circuit out of 50. The evaluator accepts circuits that match the measurements in simulation, including circuits that are electrically indistinguishable. An error or an incorrect answer receives a score of zero. Some incorrect answers do not produce a separate error, so error counts can be lower than the number of failed trials.

## Docker runs and retries

Claude Code, Codex, and OpenHands each had 100 final trial results: 50 per level. Their containers had 2 CPUs, 4 GiB of memory, and a 256-process limit. When infrastructure problems interrupted a trial, a retry could provide the final result for that same trial slot.

`native-agent-attempts.csv` records all 380 executions for those three agents: the original 300 attempts plus 80 retries. These are execution attempts, not 380 additional scored trials. `technical-reruns.csv` maps selected retries back to the trial slots they replace across the report set, including earlier ReAct and AI Scientist retries.

The OpenHands agent reached its 50-iteration limit in 87 of its 100 final trials. Two more ended without a final submission.

## Scope

The earlier runs made outside Docker are excluded. Their workspace access could expose benchmark files, and the audit found hidden-answer access in some of their traces. ReAct and AI Scientist results remain from their original runs because those runs already used Docker.

The `runs.csv` and `trials.csv` files list the final selected results and Langfuse trace IDs. Rebuild them with:

```bash
python reports/build_index.py reports/resistor_network
```

The report layout and CSV columns are described in [`../README.md`](../README.md).
