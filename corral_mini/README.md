# Corral mini

`corral-mini` is a reproducible selection of tasks from the final Corral
benchmark reports. The selection is a manifest of existing task IDs; it does
not modify the environments or their task banks.

## Build or refresh

Run this after publishing the final reports to the Hugging Face summary
dataset. Supply the **full commit hash** of the summary dataset so a later
refresh does not silently change the source data:

```bash
python scripts/build_corral_mini.py --revision <40-character-HF-commit>
```

The script downloads the `tasks` configuration from
`jablonkagroup/corral-reports-october-runs-summary-v1` and writes
[`manifest.json`](manifest.json) and [`assessment.json`](assessment.json).
For an already downloaded table, add `--tasks-parquet PATH`; the script still
records its SHA-256 and the supplied source revision. Re-run with a new
revision when reports change, then review the manifest diff.

## Selection protocol

1. Treat every `(environment, level)` as a stratum and require the **same
   number of tasks in all strata**. There are 18 strata in the current reports.
2. For each candidate count `k`, select the exact `k`-task combination within
   each stratum whose agent Pass@5 profile has the smallest mean squared error
   relative to all tasks in that stratum. Lexicographic task order breaks ties.
   This is exhaustive, deterministic selection; no task IDs are handpicked.
3. Use a common agent/model panel for ranking checks. A subject must have run
   at least 80% of strata; ranking uses the strata shared by all such subjects.
   Strata with fewer agents still receive the same `k` tasks.
4. For each `k`, compare the mini Pass@5 ranking with the full ranking. Repeat
   selection while withholding each subject in turn, then measure the held-out
   score error and whether any agent pair separated by at least five percentage
   points reverses order. Select the **smallest** `k` satisfying Spearman
   correlation ≥ 0.95, maximum held-out score error ≤ 0.05, and zero such
   reversals. These thresholds can be changed with CLI flags. The assessment
   also reports Pass@1 agreement.

The benchmark score for each subject is the mean of its task scores within
each stratum, then the mean across strata. This gives each environment and
level equal weight, including spectra elucidation's larger task pool.

## Current selection

Source: Hugging Face summary revision
`6d8dc71d6bcc3a3c6c76a28609f4f4a510ea2dd8` (6,500 final trials,
1,300 task-by-subject rows).

| Tasks per stratum | Total tasks | Pass@5 rank correlation | Maximum held-out score error | Meets criteria |
|---:|---:|---:|---:|:---:|
| 1 | 18 | 0.964 | 0.087 | No |
| 2 | 36 | 0.857 | 0.072 | No |
| 3 | 54 | 1.000 | 0.061 | No |
| **4** | **72** | **0.964** | **0.025** | **Yes** |

The selected 72 tasks preserve the order of all agent pairs whose full
Pass@5 scores differ by at least five percentage points in the leave-one-
subject-out check. Pass@1 rank correlation is 0.964. AFM has fewer completed
agent/model runs than the other environments, so its two levels are selected
but excluded from the common-panel ranking check. The validation measures
agreement with the observed agents; it does not guarantee rankings of future
models. Small score gaps among existing agents can reverse in a mini run. A
single agent's Pass@5 score in one environment/level can differ by as much as
13.4 percentage points. The selection targets the overall ranking; estimates
within individual strata can be less precise.

The `tasks_by_environment` mapping in `manifest.json` supplies task IDs for
each environment and level. Corral's `bench --task TASK_ID` option accepts
repeated task IDs, so the manifest can be used without copying task files.
