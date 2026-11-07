# Benchmark reports

Final benchmark reports, one folder per environment, for **corral-mini v1**:
3 tasks per level, 5 trials per task, three model generations per family.

| Collection | Folder | Hugging Face datasets |
|---|---|---|
| corral-mini v1 (current) | `reports/<environment>/` | `jablonkagroup/corral-reports-mini-runs-v1`, `jablonkagroup/corral-reports-mini-runs-summary-v1` |
| October full benchmark (archived) | [`archive/october-v1/`](archive/october-v1/) | `jablonkagroup/corral-reports-october-runs-v1`, `jablonkagroup/corral-reports-october-runs-summary-v1` |

The October reports (9 environments, 116 runs, 6,500 trials), their README,
the cheating-audit rules and flagged trials, and the cost estimate are
unchanged under `archive/october-v1/`. Its [README](archive/october-v1/README.md)
documents the shared conventions in full: layout, names, index columns and
Langfuse IDs.

| Environment | Folder | Runs | Trials |
|---|---|---:|---:|
| Corral MD | [`corral_md/`](corral_md/) | 26 | 390 |

## Layout

Same as the October collection:

```
reports/<environment>/
  README.md                         results and caveats
  runs.csv                          one row per run
  trials.csv                        one row per trial, with Langfuse links
  level_<L>/<agent>__<model>/
    report.json.gz                  final report, with messages and tool calls
    run-metadata.json
    provenance.json                 source runs and adjustments, where applicable
```

## Adding reports

Corral-mini runs each trial as its own 1-trial `corral bench` run, named
`<agent>__<model>__L<level>__<task>__k<k>`. Merge them into one verbose report
per agent, model and level, from the environment's host venv (it needs
Corral):

```bash
python reports/merge_trial_runs.py <runs dir> reports/<environment> \
    --model <model> [--model <model> ...] [--unscorable-as-zero]
python reports/build_index.py reports/<environment>
python reports/build_index.py reports/<environment> --verify   # Langfuse keys
python reports/check_reports.py
```

A group is merged only when every task has all its trials. The merge rebuilds
messages and tool calls from each trial's commit database, so runs launched
without `--verbose` are complete. `--unscorable-as-zero` scores trials whose
evaluation failed (Corral leaves them unscored, which makes every metric of the
run undefined) as 0 and lists them in `provenance.json`. New models must be
added to `MODELS` in `build_index.py`.

Push to Hugging Face (needs `HF_TOKEN`). The scripts push `reports/` to the
mini datasets; the October datasets are only written with
`CORRAL_REPORTS_COLLECTION=october`, from the archive:

```bash
python scripts/push_october_reports_to_hf.py    # → corral-reports-mini-runs-v1
python scripts/push_october_summary_to_hf.py    # → corral-reports-mini-runs-summary-v1
python reports/check_reports.py --hf
```

## Tools

- `serve_traces.py` browses the conversations: `python3 reports/serve_traces.py --open`
  (add `--reports reports/archive/october-v1` for the October collection).
- `scores.py` tabulates Pass@k and Pass^k from the `runs.csv` files.
- `check_reports.py [--collection mini|october] [--hf]` checks the
  convention and the Hugging Face copies.
- `find_infra_failures.py` lists rerun candidates.
- `dashboard/build_dashboard.py` builds the results dashboard; the published
  one is built from the October archive (`--reports reports/archive/october-v1`).
