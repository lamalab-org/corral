# Benchmark reports

Final benchmark reports, one folder per environment. Every environment uses the
same layout, names and index files, so runs can be compared across them.

| Environment | Folder | Branch | Runs | Trials |
|---|---|---|---:|---:|
| AFM | [`afm/`](afm/) | `report_audit` | 4 | 200 |
| Inference optimization | [`inference_opt/`](inference_opt/) | `ml_env_baseline` | 14 | 700 |
| Psychometrics | [`psychometrics/`](psychometrics/) | `psychometric_hint_cleanup_clean` | 14 | 700 |
| Resistor network (v4, terminal) | [`resistor_network/`](resistor_network/) | `resistor_level` | 14 | 700 |

The `report_audit` branch collects all environments. Add or update reports
there, following [Adding or updating reports](#adding-or-updating-reports).

## Layout

```
reports/<environment>/
  README.md                         results and caveats for this environment
  runs.csv                          one row per run
  trials.csv                        one row per trial, with Langfuse links
  level_<L>/<agent>__<model>/
    report.json.gz                  final `corral bench` report (or report.json.xz)
    run-metadata.json               launcher metadata, where it was recorded
```

Each report holds exactly the trials that count: 10 tasks × 5 trials. Where a
trial failed for an infrastructure reason and was rerun, the report holds the
rerun in that trial's place, and the report's metrics are computed from the
final trials. Superseded attempts are not kept here.

## Names

| Field | Values |
|---|---|
| `agent` | `react`, `claude-code`, `codex`, `openhands`, `ai-scientist` |
| `model` | `claude-opus-5-5`, `gpt-5.6-sol`, `kimi-k3` |
| `model_id` | `anthropic/claude-opus-5-5`, `openai/gpt-5.6-sol`, `openrouter/moonshotai/kimi-k3` |
| `level` | `1`, `2` |

Run folders are `<agent>__<model>`, e.g. `level_2/react__gpt-5.6-sol/`.
`run_id` is the report's `benchmark_run_id` and keeps the name the run was
launched under, because Langfuse sessions are keyed by it.

## Index files

`runs.csv`: `environment, level, agent, model, model_id, run_id, tasks,
trials_per_task, trials, passes, error_trials, average_score, pass_at_1,
pass_at_k, pass_hat_k, input_tokens, output_tokens, generated_at, report_file,
report_sha256, langfuse_session_url`. `report_sha256` is the hash of the
uncompressed JSON.

`trials.csv`: `environment, level, agent, model, run_id, task_id, trial,
execution_id, score, success, error, duration_s, input_tokens, output_tokens,
tool_calls, langfuse_session_id, langfuse_trace_id, langfuse_trace_url,
report_file`. `trial` is the trial's position within its task (0–4). `error`
is empty for clean trials; trials that still ended in an error count as
failures.

## Langfuse

Each trial is one trace in project `cmsio90yc092lad0ggc5lyo0x` on
`https://cloud.langfuse.com`:

- trace ID = `sha256("corral:" + execution_id)[:32]`, as set by
  `corral.observability.langfuse.deterministic_trace_id`
- session ID = the run part of `execution_id` (`<session>:<task_id>:<n>`).
  This is the `run_id`, except for a rerun substituted into a report, which
  keeps the session it was run in.

On 2026-10-16 every trace in the three `trials.csv` files (2,100 in total) was
found in Langfuse (`build_index.py --verify`).

**Known issue: some traces mix attempts.** Some trials were retried under the
same execution ID, so the retry wrote into the same trace as the attempt it
replaced. Such a trace can hold observations from both. This affects about 139
inference optimization trials and up to 319 resistor trials (Codex, Claude
Code, OpenHands, react Kimi K3 and react GPT L2 runs). The psychometrics reruns
and the nine resistor technical reruns (`resistor_network/technical-reruns.csv`)
used new run IDs and are not affected. The report is authoritative for what the
final attempt did. Future reruns should use a new run ID so each attempt gets
its own trace.

## Adding or updating reports

1. Put each final report at `reports/<environment>/level_<L>/<agent>__<model>/report.json.gz`
   (`report.json.xz` if gzip is much larger). Remove superseded reports; only
   final ones belong here. Track reports over ~50 MB in Git LFS.
2. Rebuild the index and check that every trace exists in Langfuse:

   ```bash
   python reports/build_index.py reports/<environment>
   python reports/build_index.py reports/<environment> --verify  # needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
   ```

3. Optionally, write or update `reports/<environment>/README.md` with results
   and caveats.
4. Push to Hugging Face (needs `HF_TOKEN`):

   ```bash
   python scripts/push_october_reports_to_hf.py   # jablonkagroup/corral-reports-october-runs-v1
   python scripts/push_october_summary_to_hf.py   # jablonkagroup/corral-reports-october-runs-summary-v1
   ```

5. Check everything before pushing to git:

   ```bash
   python reports/check_reports.py        # convention
   python reports/check_reports.py --hf   # Hugging Face datasets match (needs huggingface_hub, pyarrow)
   ```

The `Reports` workflow (`.github/workflows/reports.yaml`) runs both checks on
every push and pull request to `report_audit` that touches `reports/`. It fails
on the reports, the two CSVs, their Langfuse IDs, or a missing or outdated
Hugging Face copy; other files only produce warnings. It
checks out without Git LFS, so reports stored in LFS are verified through the
Hugging Face summary's `report_sha256` rather than rebuilt.

Read a report with `gzip -dc report.json.gz` or `xz -dc report.json.xz`. The
inference optimization AI Scientist reports are stored in Git LFS; run
`git lfs pull` first.
