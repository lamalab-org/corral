# Resistor network reports (v4, terminal)

14 runs (7 agent/model pairs × 2 levels), 10 tasks × k = 5, 700 trials.
Layout, names and index columns follow [`../README.md`](../README.md).

| Agent | Model | L1 passes | L1 errors | L2 passes | L2 errors | Terminal |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 49/50 | 0 | 32/50 | 2 | Docker |
| claude-code | claude-opus-5-5 | 50/50 | 0 | 39/50 | 11 | native |
| codex | gpt-5.6-sol | 48/50 | 0 | 37/50 | 1 | native |
| openhands | kimi-k3 | 9/50 | 41 | 3/50 | 47 | native |
| react | claude-opus-5-5 | 38/50 | 3 | 26/50 | 11 | Docker |
| react | gpt-5.6-sol | 44/50 | 2 | 28/50 | 11 | Docker |
| react | kimi-k3 | 10/50 | 11 | 1/50 | 19 | Docker |

Errors are trials that ended with an agent or provider error; they count as
failures. Most are exhausted interaction or turn budgets; OpenHands Kimi K3 hit
its iteration limit in 88 of 100 trials. react Opus also has 3 provider 5xx
errors and 3 out-of-memory kills.

Settings that differ between runs:

- Claude Code and Codex trials that hit a 30-minute trial deadline (84 in all)
  were rerun without it.
- react Kimi K3 ran through OpenRouter pinned to the Fireworks provider with low
  reasoning effort and a 32,000-token cap; 77 of its trials were restarted
  after failures with an earlier provider.
- Docker trials had 2 CPUs, 4 GiB RAM and 256 PIDs; native agents had no cap.
