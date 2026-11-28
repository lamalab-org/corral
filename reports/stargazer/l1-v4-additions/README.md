# Stargazer Level 1 v4 additions

#513 replaced four Level 1 tasks (bank `stargazer-orbit-recovery-l1-v4`). These
runs cover only the four new tasks: `seed111_diff3`, `seed610696_diff5`,
`seed611761_diff5` and `seed630238_diff4`. With the six unchanged tasks in
[`../full/level_1`](../full/level_1), they give the v4 Level 1. The October
reports in `../full/` stay on the v3 bank. Level 2 is unchanged.

15 runs, 300 trials (4 tasks × k = 5 per run), **255 passes**, 1 error trial (an OpenHands kimi-k3
trial at the iteration limit), no infrastructure failures.

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Results

Passes out of 5 per task (`seed111_diff3`, `seed610696_diff5`,
`seed611761_diff5`, `seed630238_diff4`) and out of 20 in total.

| Agent | Model | Per task | Total |
|---|---|---|---:|
| claude-code | claude-opus-5-5 | 5 · 5 · 5 · 5 | 20 |
| claude-code | claude-opus-4-7 | 4 · 5 · 5 · 5 | 19 |
| claude-code | claude-opus-4-5-20251101 | 5 · 5 · 5 · 5 | 20 |
| codex | gpt-6.1-sol | 5 · 5 · 5 · 5 | 20 |
| codex | gpt-5.6-sol | 5 · 5 · 5 · 5 | 20 |
| codex | gpt-5-2025-08-07 | 1 · 0 · 1 · 0 | 2 |
| react | claude-opus-5-5 | 5 · 5 · 5 · 5 | 20 |
| react | claude-opus-4-7 | 5 · 5 · 5 · 5 | 20 |
| react | claude-opus-4-5-20251101 | 5 · 4 · 4 · 5 | 18 |
| react | gpt-6.1-sol | 5 · 5 · 5 · 5 | 20 |
| react | gpt-5.6-sol | 5 · 5 · 5 · 5 | 20 |
| react | gpt-5-2025-08-07 | 3 · 1 · 0 · 1 | 5 |
| ai-scientist | claude-opus-4-7 | 2 · 5 · 2 · 4 | 13 |
| openhands | kimi-k3 | 5 · 5 · 4 · 5 | 19 |
| react | kimi-k3 | 5 · 5 · 4 · 5 | 19 |

gpt-5 failures are model answers: most fit and submit two or three planets.
One Codex trial submitted its plan text instead of JSON.

ReAct claude-opus-5-5 had no empty or refused turns.

AI Scientist ran with claude-opus-4-7 instead of claude-opus-5-5 (refusal risk
on claude-opus-5-5). Its 7 failures are wrong answers; no trial errored or ran
out of budget. The October AI Scientist runs on the six unchanged tasks used
claude-opus-5-5, so a v4 Level 1 AI Scientist column built from both mixes the
two models.

Kimi: the OpenHands failure hit the 50-iteration limit; the ReAct failure submitted a wrong answer. One
OpenHands trial (`seed630238_diff4`, k3) stalled: a code-execution worker the agent started ran at 100%
CPU for over 8 hours and its next tool call queued behind it (the code tool has no wall-clock timeout).
It was replaced by a fresh trial (session `…-seed630238_diff4-k3-replacement`, passed); the rule that the
replacement counts was fixed before either trial finished. The stalled original is not in the report.

## Settings

- **Code:** `main` 9269652a4 (#513).
- **Images:** `corral-stargazer:l1v4-9269652a4-{claude,codex,base}`, built with the
  pinned stargazer recipe used for the mini-v2 images.
- **Common to all runs:** 50 iterations, reasoning effort `medium`, surrender off,
  Docker sandbox with 2 CPUs and bridge network, k = 5. Run 2026-11-23/24.
- **October combinations** (claude-code claude-opus-5-5, codex gpt-5.6-sol, react
  gpt-5.6-sol, react claude-opus-5-5): memory as in October, 12 GB, 4 GB, 16 GB
  and 4 GB respectively.
- **Other models:** mini-v2 settings, 6 GB.
- **AI Scientist claude-opus-4-7:** 100 iterations, agent kwargs
  `{"config": {"max_tool_calls": 256}, "reasoning_effort": "medium"}`, image
  `corral-stargazer:l1v4-9269652a4-base`, 6 GB. Run 2026-11-27.
- **Kimi (`openrouter/moonshotai/kimi-k3`), as in October:** OpenRouter pinned to Together
  (`{"provider": {"only": ["together"], "allow_fallbacks": false, "zdr": true, "data_collection": "deny"}}`,
  passed as `litellm_extra_body` for OpenHands and `extra_body` for ReAct), reasoning effort `medium`,
  8 GB. OpenHands image `corral-stargazer:l1v4-9269652a4-openhands` (same pinned recipe); ReAct uses the
  base image. ReAct also passes `"allowed_openai_params": ["reasoning_effort"]`: LiteLLM 1.102 rejects
  `reasoning_effort` for this route although OpenRouter supports it, and October's LiteLLM passed it
  through, so the request matches October's. Run 2026-11-27/28.
- **Merging:** each trial ran as its own `corral bench --trials 1` run; the reports
  are merged with `scripts/merge_trial_reports.py` (run IDs
  `stargazer-l1v4-l1-<agent>-<model>`, source runs in `provenance.json`).
