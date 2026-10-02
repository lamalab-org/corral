# Psychometrics runs

Launched with `run_psychometrics.sh` (this folder): `react` agent, `gpt-5.6-sol`,
reasoning effort medium, temperature 1, k = 1, 50 iterations, local sandbox.

| run | contents | settings |
|---|---|---|
| `20261001-235820` | both levels, all 20 tasks | verbose; no REPL time limit; LiteLLM default 600 s request timeout |
| `20261002-174227` | the 6 tasks that failed or never finished above | verbose; 15-minute REPL limit; 180 s request timeout; corrected workspace path descriptions |

The first run was cut off at its 2-hour launcher limit while three tasks sat in
semopy FIML fits, so `corral bench` never wrote `report.json` for it. Its
`results_offline.json` re-scores each saved submission with the task's own
scoring function, which is what the report would have used.

Earlier runs, made before the hint cleanup, are archived outside the repo in
`corral_runs_archive/psychometrics_2026-09-29_gpt-5.6-sol_pre-hint-cleanup/`.

## Per run folder

- `report.json`: `corral bench` report with per-trial agent messages and tool calls (rerun only).
- `results_offline.json`: offline scores (first run only).
- `trajectories/<task>.json`: prompt, full conversation, every tool call and
  output, submission, usage and runtime. The REPL session checkpoint, a pickled
  copy of the data, is left out.
- `run.log`: console log. `state/` (full run state, several GB) is not committed.

## Results (all 20 tasks)

Each task's score comes from the first run, or from the rerun for the six it repeated.
Partial scores in brackets; the old column is the 2026-09-29 run before the hint cleanup.

| task | score | from | old |
|---|---|---|---|
| L1 t01 hsns_structure | 1 | first | 1 |
| L1 t02 dd_structure | 0 (0.55, model fit) | first | 1 |
| L1 t03 local_dependence | 1 | first | 1 |
| L1 t04 invariant_combination | 0 (0.55, model fit) | rerun | 1 |
| L1 t05 defensible_comparisons | 1 | first | 1 |
| L1 t06 latent_relationships | 0 (0.55, model fit) | rerun | 0 |
| L1 t07 cross_country_replication | 0 (0.55, model fit) | rerun | 0 |
| L1 t08 score_justification | 0 (0.55, model fit) | first | 0 |
| L1 t09 careless_responding | 0 (0.92, one correlation) | rerun | policy flag |
| L1 t10 item_integrity | 1 | first | 1 |
| L2 t01 group_comparability | 1 | first | 1 |
| L2 t02 out_of_sample_generalization | 0 (0.55, model fit) | rerun | 0 |
| L2 t03 population_classification | 0 (0.80) | first | policy flag |
| L2 t04 hsns_population_classification | 1 | first | socket timeout |
| L2 t05 duplicate_records | 0 (0.55, model fit) | first | 1 |
| L2 t06 gender_item_integrity | 1 | first | 0 |
| L2 t07 behavioral_validity | 0 (0.71) | rerun | 0 |
| L2 t08 misfit_replication | 0 (0.57) | first | 1 |
| L2 t09 adaptive_bank_choice | 0 (0.80) | first | 1 |
| L2 t10 model_identification | 1 | first | policy flag |

Level 1: 4/10 (old 0.6). Level 2: 4/10 (old 0.4). The table mixes two runs with
different settings and has one trial per task, so treat single flips as noise.

## Why scores moved

- **Hints removed (intended).** L2 t05 lost the factor-ordered columns and grouped
  the items wrongly; L2 t08 and L2 t09 lost one-entry example lists and
  over-reported (an extra cross-loading or rejection; an extra unstable item).
- **Wording change.** L1 t04 now asks for biased items "in either instrument",
  so the planted Dirty Dozen trap (DDN4 looks biased under the nominal structure)
  counts. The agent also chose a second-order model over the bifactor one, which
  fails the fit check on its own.
- **Likely noise.** L1 t02 stopped at three correlated factors instead of the bifactor model.
- **Infrastructure fixed.** L2 t04 and L2 t10 now have scores; L2 t06 finished a
  55-minute FIML fit and passed.
- **L1 t09** was refused by OpenAI's policy filter in 2 of 3 runs, at different points.
