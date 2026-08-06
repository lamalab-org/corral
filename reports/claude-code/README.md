# Claude Code — corral-mini reports

Benchmark reports for the `ClaudeCodeAgent` (Claude Agent SDK / Claude Code
harness) run against the **corral-mini** task subset — see
`sampler/data/corral_mini_manifest_budget20.json` for the exact 20-task
selection (`stratified_proportional_nested`, budget=20, seed=0) and
`tasks/<env>/environment-mini/` for the symlinked task JSON that realizes it.

Layout mirrors [`lamalab-org/corral@kimi-k2.5-benchmark-reports`](https://github.com/lamalab-org/corral/tree/kimi-k2.5-benchmark-reports/reports/kimi-k2.5):

```
reports/claude-code/<environment>/level_<n>/
  run.py                                   # exact script used to produce this level's report
  <env>-l<n>-claude_code-<model>.json       # aggregated BenchmarkResult report (git-ignored, produced by run.py)
  agent_logs-ClaudeCodeAgent-<model>-<tool_verbosity>/
      <task_id>_<timestamp>.json           # per-trial transcript (only saved when run.py passes verbose=True)
  benchmark_checkpoints/                   # incremental per-trial checkpoints (resumable, git-ignored)
```

Unlike the kimi-k2.5 reports (which compare a `react` vs. `toolcalling`
*agent scope* per level), every level here uses a single agent —
`ClaudeCodeAgent` — since the Claude Code harness owns its own
planning/acting loop. There is therefore no `react/`/`toolcalling/`
subdirectory; `run.py` sits directly under `level_<n>/`.

## Environments

| Environment | Levels served by `environment-mini/` | Status |
|---|---|---|
| `spectra_elucidation` | `level_1`, `level_2` | ✅ set up, see `spectra_elucidation/level_*/run.py` |
| `wetlab` | `level_1`, `level_2`, `level_3` | 🔲 scaffolded, not yet wired up |
| `retrosynthesis` | `level_1`, `level_2`, `level_3` | 🔲 scaffolded, not yet wired up |

`resistor_network` is excluded from corral-mini's budget (only 6 items total,
kept whole rather than sampled) and is out of scope for this batch of runs.

## Running a level

Each level directory is self-contained: `cd` into it and run `run.py`. See
`spectra_elucidation/level_1/run.py` for the reference implementation —
it starts by loading `ANTHROPIC_API_KEY` (and other keys) from the repo-root
`.env`, expects the matching environment-mini server already running (see
that env's `serve_mini_env.py`), and writes its outputs into its own
directory.
