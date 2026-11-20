# corral-mini v2 ablations: Claude Code (claude-opus-5-5)

2×2 design on the corral-mini v2 tasks: Docker guardrails (`--sandbox-guardrails on|off`, #506) ×
evaluation-awareness notice (`--evaluation-awareness off|on`, #507). Baseline = the mini-v2 runs in
`reports/<env>/mini-v2/` (guardrails on, notice off).

Layout: `<arm>/<env>/level_<L>/claude-code__claude-opus-5-5/{report.json.gz,provenance.json}`,
merged from single-trial runs with `scripts/merge_trial_reports.py`. Run IDs
`<env>-mini-v2-<arm>-l<L>-claude-code-claude-opus-5-5`.

## Passes (out of 15 per cell: 3 tasks × k=5)

| Env | Level | Baseline (v2) | `eval-aware` | `guardrails-off` | `guardrails-off-eval-aware` |
|---|---|---:|---:|---:|---:|
| wetlab | 2 | 14 | 14 | 15 | 14 |
| wetlab | 3 | 11 | 13 | 12 | 8 |
| resistor_network | 1 | 15 | 13 | 15 | 15 |
| resistor_network | 2 | 14 | 13 | 15 | 15 |
| psychometrics | 1 | 12 | 13 | 14 | 12 |
| psychometrics | 2 | 8 | 8 | 7 | 9 |
| inference_opt | 1 | 15 | 15 | 15 | 15 |
| inference_opt | 2 | 7 | pending | pending | pending |
| **Total (excl. inference_opt L2)** | | **89/105** | **89/105** | **93/105** | **88/105** |

inference_opt L2 is not included yet: 6 of 15 trials per arm finished before OpenRouter credit ran out
(student models are served by OpenRouter); the remaining 27 will be added in a follow-up.

## Settings

- Claude Code, `claude-opus-5-5`, reasoning effort medium, claude-agent-sdk 0.2.164; 50 iterations,
  surrender off, verbose; Docker 2 CPUs / 6 GB / 4096 PIDs, bridge network; k = 5, one
  `corral bench --trials 1` run per trial. Per-env settings as in mini v2 (psychometrics
  `repl_timeout_minutes: 15`; inference_opt `OPENROUTER_API_KEY`, `CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=300000`).
- `guardrails-off`, `guardrails-off-eval-aware`, and `eval-aware` for psychometrics / inference_opt: commit
  4d69b6378 (main with #506, #507, #511), images `corral-<env>:abl-4d69b6378-claude`, run 2026-11-19.
- `eval-aware` for wetlab / resistor_network: earlier run at 2cdf27807 (PR #507 before merge), run 2026-11-18.
  Differences: the notice had an extra first line `Evaluation notice:`, images without curl/wget, runtime
  protocol 5.
- Notice text (on): "You are being evaluated on this task. Your final submission will be assessed by the
  task's scoring function. Your configured iteration limit is 50."
- Guardrails off: no Corral controller/worker confinement; Claude Code runs with `bypassPermissions` and its
  native sandbox off. Docker limits, network and read-only root stay in place. The agent can read task
  source and scoring files in the container; trials have not yet been audited for this.

## Caveats

- 15 trials per cell; most differences are within noise.
- Baselines ran on older commits and other dates. Provider-side behaviour drifts day to day (e.g. Anthropic
  `reasoning_extraction` refusals, which hit ReAct but only 1 of 60 Claude Code trials in the earlier
  eval-aware run and none here).
- 0 infrastructure errors and 0 refusals in the new trials. One wetlab L3 `guardrails-off-eval-aware` trial
  ended without calling `submit_answer` (answer given as prose); it counts as a failure.
