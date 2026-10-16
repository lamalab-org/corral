# Psychometrics reports

14 runs (7 agent/model pairs × 2 levels), 10 tasks × k = 5, 700 trials.
Layout, names and index columns follow [`../README.md`](../README.md).

| Agent | Model | L1 passes | L1 errors | L2 passes | L2 errors | Task version |
|---|---|---:|---:|---:|---:|---|
| ai-scientist | claude-opus-5-5 | 24/50 | 0 | 25/50 | 0 | current |
| claude-code | claude-opus-5-5 | 30/50 | 0 | 35/50 | 0 | earlier |
| codex | gpt-5.6-sol | 24/50 | 0 | 29/50 | 0 | earlier |
| openhands | kimi-k3 | 8/50 | 0 | 10/50 | 1 | current |
| react | claude-opus-5-5 | 22/50 | 0 | 27/50 | 0 | current |
| react | gpt-5.6-sol | 19/50 | 2 | 33/50 | 0 | earlier |
| react | kimi-k3 | 10/50 | 10 | 8/50 | 8 | current |

**Runs used two task versions.** Two commits changed the tasks while the runs
were under way:

- `f0f93e234` (2026-10-07 13:37 UTC) names semopy as the model syntax and adds
  a note on its factor-covariance default to every prompt that asks for a model
  (all of L1, L2 t01 and t05). Scoring is unchanged (re-scoring the submitted
  models with the missing correlations added changed no score), but agents in
  later runs saw the extra note.
- `fa56c111d` (2026-10-07 20:41 UTC) reworks L2 t03, t04, t09 and t10 so their
  prompts stop prescribing the method.

"earlier" runs started before both commits; "current" runs started after
both. This is inferred from run times, since reports do not record the task
commit. On L1 the two groups differ only by the semopy note. On L2, compare
them only on t02 and t06–t08, the tasks neither commit touched.

react GPT-5.6 Sol includes reruns of four trials that failed for infrastructure
reasons (three provider usage-policy rejections on L1, one agent bug on L2).
Its two remaining L1 errors are socket timeouts. react Kimi K3 errors are
mostly exhausted interaction budgets.

The runs were launched with [`../run_psychometrics.sh`](../run_psychometrics.sh).
