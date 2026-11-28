# Benchmark reports

Final benchmark reports, one folder per environment. Every environment uses the
same layout, names and index files, so runs can be compared across them.

| Environment | Full runs | Runs | Trials |
|---|---|---:|---:|
| AFM | [afm/full/](afm/full/) | 14 | 700 |
| Inference optimization | [inference_opt/full/](inference_opt/full/) | 14 | 700 |
| Psychometrics | [psychometrics/full/](psychometrics/full/) | 14 | 700 |
| Resistor network | [resistor_network/full/](resistor_network/full/) | 14 | 700 |
| Corral MD | [corral_md/full/](corral_md/full/) | 14 | 700 |
| Retrosynthesis | [retrosynthesis/full/](retrosynthesis/full/) | 14 | 700 |
| Spectra elucidation | [spectra_elucidation/full/](spectra_elucidation/full/) | 14 | 1,400 |
| Stargazer | [stargazer/full/](stargazer/full/) | 14 | 700 |
| Wetlab | [wetlab/full/](wetlab/full/) | 14 | 700 |

The full benchmark has **126 runs and 7,000 trials** across nine environments.
Each environment has a `full/` collection with its own indexes; report paths
in each index are relative to that collection.

Mini versions identify subset releases, separate from the full benchmark:
`<environment>/mini-v1/` holds Psychometrics mini-v1 (28 runs / 420 trials).
`<environment>/mini-v2/` holds 36 runs / 540 trials for each of AFM, Corral MD,
inference optimization, Psychometrics, resistor network, retrosynthesis,
spectra elucidation and wetlab. AFM additionally includes two supplementary
AI Scientist runs (30 trials), bringing its mini-v2 collection to 38 runs /
570 trials. Stargazer mini-v2 reports are not yet present.
Run dates and source commits belong in metadata and provenance, not collection names.

Each mini version has its own Hugging Face datasets, not the October ones:
`jablonkagroup/corral-reports-mini-v1` (psychometrics) and
`jablonkagroup/corral-reports-mini-v2` (wetlab, retrosynthesis, spectra elucidation, resistor network,
corral MD, inference optimization), each with a `-summary`. The
push scripts refuse to put runs whose ID contains `-mini-` into a repo without
`mini` in its name (or the reverse), or runs of one mini version into another
version's repo:

```bash
python scripts/push_october_reports_to_hf.py --env=wetlab,retrosynthesis,spectra_elucidation,resistor_network,corral_md,inference_opt --only_env \
    --repo=jablonkagroup/corral-reports-mini-v2
python scripts/push_october_summary_to_hf.py --env=wetlab,retrosynthesis,spectra_elucidation,resistor_network,corral_md,inference_opt \
    --repo=jablonkagroup/corral-reports-mini-v2-summary
```

List every environment with runs of that mini version. The shared tables
(`runs`, and the summary's `runs`, `tasks` and `trials`) are rebuilt from the
listed environments and replaced as a whole, so the scripts refuse a mini push
that leaves one out.

The `report_audit` branch collects all environments. Add or update reports
there, following [Adding or updating reports](#adding-or-updating-reports).

## Browse agent conversations

The [published results dashboard](https://lamalab-org.github.io/corral/artifact/)
shows full-benchmark runs only. Its builder reads
`<environment>/full/trials.csv` for each environment.
Mini-v1 and mini-v2 results remain separate from this dashboard.

Start the local trace browser from the repository root (Python 3.9 or newer;
no extra dependencies):

```bash
git lfs pull --include='reports/**' --exclude=''
python3 reports/serve_traces.py --open
```

The download needs Git LFS and retrieves report artifacts, including the Corral MD
CSV indexes. It only needs to be repeated when new LFS artifacts are added.

Open **http://127.0.0.1:8765**. Filter and group trials by **agent, environment,
and model**, with additional filters for level, outcome, and saved errors.
Select a trial to read its messages, search the conversation,
filter by role or agent session, jump between tool calls and results, and
inspect the submitted output and evaluation. Download the saved trial as JSON
or copy a link to a trace or individual message. Links work when opened against
the same local report collection and server address. For legacy OpenHands
reports, the browser restores prompts ahead of tool calls and groups commentary
whose timing was not recorded separately. Its exact interleaving with tool
calls cannot be recovered. Message numbers follow the displayed order; JSON
downloads retain the original saved data and order.

AI Scientist trials open in **Experiments**. Select a stage and experiment node
to inspect its hypothesis, parent, steps, tool observations, evaluation, and
conclusion. Source links return to the original message; **Messages**, **Tool
log**, and **Result & details** remain available. The node view reads saved
graph metadata and snapshots in worker prompts, including older reports. It
keeps separate agent sessions separate and labels missing results as unavailable.
Copied links preserve the selected stage and node. JSON downloads retain the
original trial without adding or reordering data.

The browser reads `*/full/trials.csv` indexes by default at startup and decompresses only the
selected trial's report on demand. Large reports can take a moment to scan;
recently opened trials are cached in memory. Use `--collection mini-v2` (or
`--collection mini-v1`) to browse a subset collection separately. Both `.json.gz` and `.json.xz`
reports are supported. Git LFS pointers and missing reports appear in the list;
download LFS reports with `git lfs pull` to inspect their messages. Restart the
browser after rebuilding CSV indexes.

If an index is still an LFS pointer, the browser warns in the terminal and skips
that environment while serving the available indexes. Empty indexes or indexes
missing required columns are also skipped with a warning. If no readable trials
remain, startup exits with recovery instructions. Download or rebuild the affected
indexes and restart to include those environments.

Use `--port 9000` for a different port or `--reports /path/to/reports` for another
collection with the same layout. The server binds to `127.0.0.1` and serves only
the trace browser and indexed trial data. Conversations show exactly what was
saved: missing prompts, empty thinking messages, and unrecorded SDK activity
cannot be reconstructed.

## Layout

```
reports/<environment>/
  README.md                         links to available collections
  full/                             full benchmark (formerly October runs)
    README.md                       results and caveats
    runs.csv                        one row per run
    trials.csv                      one row per trial, with Langfuse links
    level_<L>/<agent>__<model>/
      report.json.gz                final report (or report.json.xz)
      run-metadata.json             run metadata, where available
      langfuse-overrides.json        recorded trace ID exceptions, if any
      langfuse.json                 verified session/trial IDs, where available
      provenance.json               source hashes and recovery details
  mini-v1/                          version 1 subset, where available
  mini-v2/                          version 2 subset, where available
```

Every collection uses the same index and level/run layout. Moving a collection
does not change report bytes, run IDs, execution IDs, scores, or trace IDs.

AFM’s Level 1 AI Scientist report combines two runs covering tasks 1–5 and 6–10;
see its full-collection README for provenance.

Each full report holds exactly the trials that count: 10 tasks × 5 trials, or
20 tasks × 5 trials for spectra elucidation. Where a
trial failed for an infrastructure reason and was rerun, the report holds the
rerun in that trial's place, and the report's metrics are computed from the
final trials. Superseded attempts are not kept here.

## Names

| Field | Values |
|---|---|
| `agent` | `react`, `claude-code`, `codex`, `openhands`, `ai-scientist` |
| `model` | `claude-opus-5-5`, `gpt-5.6-sol`, `kimi-k3` |
| `model_id` | `anthropic/claude-opus-5-5`, `openai/gpt-5.6-sol`, `openrouter/moonshotai/kimi-k3` |
| `level` | `1`, `2`; retrosynthesis and wetlab use `2`, `3` |

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
is empty when the saved terminal error fields are empty. These raw error
counts can include iteration-limit or answer-format outcomes. The saved
score and success flag determine the benchmark outcome.

## Langfuse

Each trial is one trace in project `cmsio90yc092lad0ggc5lyo0x` on
`https://cloud.langfuse.com`:

- trace ID defaults to `sha256("corral:" + execution_id)[:32]`, as set by
  `corral.observability.langfuse.deterministic_trace_id`. When a replacement keeps
  its execution ID but uses a fresh trace, `run-metadata.json` records that ID in
  `langfuse_trace_ids`, a mapping from execution ID to published trace ID.
- session ID = the run part of `execution_id` (`<session>:<task_id>:<n>`).
  This is the `run_id`, except for a rerun substituted into a report, which
  keeps the session it was run in by default. When finished replacements are
  moved into the original benchmark session, the report's `run-metadata.json`
  records `langfuse_session_id`; the index uses that session for all its trials.
  Session overrides do not change execution or trace IDs; trace overrides are
  applied separately through `langfuse_trace_ids`.

Recorded trace IDs in `langfuse-overrides.json` take precedence over the
deterministic formula. `build_index.py` reads these mappings before deriving
other IDs. Keep these files when rebuilding or copying a run.

On 2026-10-16 every trace in the original three environments' `trials.csv`
files (2,100 in total) was found in Langfuse (`build_index.py --verify`).

**Known issue: some traces mix attempts.** Some trials were retried under the
same execution ID, so the retry wrote into the same trace as the attempt it
replaced. Such a trace can hold observations from both. This affects about 139
inference optimization trials and some resistor trials (including native agent
Docker retries and react Kimi K3 and react GPT L2 runs). The psychometrics reruns
and separate-ID resistor technical reruns (`resistor_network/full/technical-reruns.csv`)
used new run IDs and are not affected. The report is authoritative for what the
final attempt did. Future reruns should use a new run ID so each attempt gets
its own trace.

## Adding or updating reports

1. Put each final report at `reports/<environment>/full/level_<L>/<agent>__<model>/report.json.gz`
   (`report.json.xz` if gzip is much larger). Remove superseded reports; only
   final ones belong here. Track reports over ~50 MB in Git LFS.
2. Rebuild the index and check that every trace exists in Langfuse:

   ```bash
   python reports/build_index.py reports/<environment>/full
   python reports/build_index.py reports/<environment>/full --verify  # needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
   ```

3. Optionally, write or update `reports/<environment>/full/README.md` with results
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

To find trials that failed for infrastructure or environment reasons (exhausted
credits, rate limits, provider errors, timeouts, environment crashes) and so
are rerun candidates:

```bash
python reports/find_infra_failures.py [<environment> ...] [--csv findings.csv]
```

Use `python reports/find_infra_failures.py afm --mini-v2` to scan mini-v2
reports with the expected 18 agent/model pairs per level. Full reports remain
the default. Git LFS index pointers are reported as unscanned, with unknown
trial counts; download them to include the affected environment.

Add the `check-infra-failure` label to a same-repository pull request to scan its
added and modified full, mini-v2, mini-v2 ablation, and Stargazer L1 additions
reports, including Git LFS reports. The workflow
updates its PR comment when new commits arrive and attaches a CSV of findings.
For a broader scan, comment `/infra-failures [lfs] [mini-v2] [<environment> ...]`
or run the workflow from the Actions tab. Use `lfs` to include Git LFS reports
in a comment-triggered scan.

The scan flags terminal refusals and provider failures as rerun candidates.
Refusals in the transcript and empty final assistant responses at an
iteration limit are flagged for review.

To tabulate pass@5 and pass^5 for every environment, agent, model and level
from the `runs.csv` files:

```bash
git lfs pull --include='reports/*/full/runs.csv' --exclude=''
python reports/scores.py [<environment> ...] [--markdown scores.md]
```

The `Scores` workflow (`.github/workflows/scores.yaml`) posts these tables on a
pull request whenever a `runs.csv` changes, and on the `scores` label or a
`/scores [<environment> ...]` comment. It updates its own comment in place.

The `Reports` workflow (`.github/workflows/reports.yaml`) runs both checks on
every push and pull request to `report_audit` that touches `reports/`. It fails
on the reports, the two CSVs, their Langfuse IDs, or a missing or outdated
Hugging Face copy; other files only produce warnings. It
checks out without Git LFS, so reports stored in LFS are verified through the
Hugging Face summary's `report_sha256` rather than rebuilt.

Read a report with `gzip -dc report.json.gz` or `xz -dc report.json.xz`. The
inference optimization AI Scientist reports are stored in Git LFS; run
`git lfs pull` first.
