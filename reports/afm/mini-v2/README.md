# AFM reports: corral-mini v2 — Claude and GPT batches

24 runs (six models × native agent/ReAct × two levels), with three tasks per
level × five trials: **360 trials, 274 passes**. This includes the Claude batch
(180 trials, 145 passes) and GPT batch (180 trials, 129 passes): 24 of the planned
36 runs. Kimi results are not included.
Layout follows [`../../README.md`](../../README.md), matching the other mini-v2 collections.

[Run index](runs.csv) · [Trial index with derived Langfuse links](trials.csv)

## Tasks

| Level | Tasks | Origin |
|---|---|---|
| 1 | 1, 4, 10 | New mini-v2 runs |
| 2 | 5, 6, 10 | Seven reused mini-v1 runs and five mini-v2 runs; the task IDs match |

## Results

Passes are out of 15 per level. Pass@5 is the saved report estimator averaged
across the three tasks; Pass^5 and token counts are available in the run index.

| Model | Agent | L1 passes | L1 Pass@5 | L2 passes | L2 Pass@5 |
|---|---|---:|---:|---:|---:|
| claude-opus-5-5 | claude-code | 9 | 0.6666 | 14 | 0.9999 |
| claude-opus-5-5 | react | 10 | 0.8907 | 13 | 0.9966 |
| claude-opus-4-7 | claude-code | 15 | 1.0000 | 7 | 0.7815 |
| claude-opus-4-7 | react | 15 | 1.0000 | 14 | 0.9999 |
| claude-opus-4-5-20251101 | claude-code | 14 | 0.9999 | 10 | 0.8907 |
| claude-opus-4-5-20251101 | react | 13 | 0.9966 | 11 | 0.9740 |

| gpt-6.1-sol | codex | 15 | 1.0000 | 15 | 1.0000 |
| gpt-6.1-sol | react | 15 | 1.0000 | 15 | 1.0000 |
| gpt-5.6-sol | codex | 15 | 1.0000 | 14 | 0.9999 |
| gpt-5.6-sol | react | 15 | 1.0000 | 14 | 0.9999 |
| gpt-5-2025-08-07 | codex | 4 | 0.5540 | 0 | 0.0000 |
| gpt-5-2025-08-07 | react | 3 | 0.3299 | 4 | 0.3332 |

## Settings and provenance

The saved configuration records 50 iterations, medium reasoning, surrender off,
verbose reports, five trials per task, one parallel trial, and a local sandbox.
These are Windows Nanosurf AFM runs using direct Anthropic and OpenAI routes.
**Local instrument access is an exception to the shared Docker benchmark setup:**
Docker CPU, memory and PID limits do not apply. Thread environment settings and
Langfuse export success cannot be established from the supplied report files.

Reports and launcher metadata were imported from the supplied
`corral-mini-result/data` (Claude) and `corral-mini-result/data_gpt` (GPT) folders without changing their bytes (reports use
lossless gzip compression). No trials were rescored, replaced or excluded.
Per-run `provenance.json` records source folder names, SHA-256 hashes, original
subset versions and the target v2 manifest hash. Source Git commits were not
recorded in the supplied files and are left unknown.

The 105 reused Level 2 trials (all six Claude runs plus GPT-6.1-Sol/Codex)
retain their original v1 launcher metadata and run IDs. Their task coverage
matches v2 exactly. The other 255 trials record v2 in launcher metadata. This reuse does not rename trials or invent new sessions.

Original execution IDs and run IDs are preserved. The index builder derives
Langfuse trace/session links using its configured host and project. These links
have **not been verified** against Langfuse; importing reports does not upload
traces. This GitHub update has not been uploaded to Hugging Face.

## Errors and caveats

One terminal error is preserved: Claude Code / Opus 5.5 / Level 1 task 10,
trial index 1, finished without calling `submit_answer`. Its score remains zero.
The other 85 zero scores have no saved terminal error. They are retained as
reported; individual scoring outcomes have not been independently audited.
GPT-5/Codex Level 2 has zero passes out of 15, with no saved terminal errors;
these outcomes are preserved, not treated as missing data.

## Validation

Verified source hashes and decompressed byte equality, 360 unique execution IDs,
three exact v2 task IDs per run, five trials per task, and index reconstruction
using the repository's report checker. Full AFM reports remain unchanged.

```bash
python reports/build_index.py reports/afm/mini-v2
python reports/check_reports.py
```
