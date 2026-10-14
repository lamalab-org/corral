# Psychometrics run results

One archive per model and agent, k = 5, both levels. Each holds, per level,
`level_<L>/Psychometrics-<STAMP>/report.json` (the `corral bench` report with
every trial's messages, tool calls, submission and score) and `run.log`, plus
the launcher log. The run state (REPL session databases and snapshots, about
140 GB in total) stays local.

Unpack with `tar -xzf <archive>.tar.gz`. Langfuse session ids are
`psychometrics-l<L>-Psychometrics-<STAMP>`.

| archive | model / agent | L1 | L2 |
|---|---|---|---|
| `claudecode-opus55-20261007` | Opus 5.5 / Claude Code | 30 | 35 |
| `codex-gpt5sol-20261007` | gpt-5.6-sol / Codex | 24 | 29 |
| `react-gpt5sol-20261006` | gpt-5.6-sol / react | 18 (19 with rerun) | 32 (33 with rerun) |
| `react-opus55-20261007` | Opus 5.5 / react, reworked L2 t03/t04/t09/t10 | 22 | 27 |
| `react-kimik3-20261006` | Kimi K3 / react | 8 | 13 |

Scores are passed trials out of 50 per level. `react-gpt5sol-20261006` also
contains two reruns: `-rerun-refusals` (the 3 refused L1 trials) and
`-rerun-l2t04` (the crashed L2 t04 trial). Only the Opus react run used the
reworked Level 2 tasks (commit `fa56c111d`); the others ran before it.
