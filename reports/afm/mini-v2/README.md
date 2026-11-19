# AFM reports: corral-mini v2 — Claude, GPT and Kimi

36 runs (nine models × native agent/ReAct × two levels), with three tasks per
level × five trials: **540 trials, 334 passes**. Claude contributes 145/180
passes, GPT 129/180 and Kimi 60/180. All 36 planned configurations are represented;
the generic provider failures described below remain provisional.
Layout follows [`../../README.md`](../../README.md), matching the other mini-v2 collections.

[Run index](runs.csv) · [Trial index with derived Langfuse links](trials.csv)

## Tasks

| Level | Tasks | Origin |
|---|---|---|
| 1 | 1, 4, 10 | New mini-v2 runs |
| 2 | 5, 6, 10 | Eight reused mini-v1 runs and ten mini-v2 runs; the task IDs match |

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

| kimi-k3.1-pro | openhands | 15 | 1.0000 | 15 | 1.0000 |
| kimi-k3.1-pro | react | 15 | 1.0000 | 4 | 0.7556 |
| kimi-k2.5 | openhands | 0 | 0.0000 | 2 | 0.4482 |
| kimi-k2.5 | react | 1 | 0.2241 | 1 | 0.2241 |
| kimi-k2-0905 | openhands | 1 | 0.2241 | 1 | 0.2241 |
| kimi-k2-0905 | react | 1 | 0.2241 | 4 | 0.6148 |

## Settings and provenance

The saved configuration records 50 iterations, surrender off,
verbose reports, five trials per task, one parallel trial, and a local sandbox.
These are Windows Nanosurf AFM runs using direct Anthropic/OpenAI routes and
OpenRouter for Kimi (Modal for K3.1 Pro, Novita for K2.5/K2-0905, no fallbacks).
Claude/GPT request medium reasoning. K2.5 and ReAct/K3.1 Pro request medium in
the extra request body; OpenHands/K3.1 Pro uses top-level reasoning_effort.
K2-0905 leaves reasoning at the provider/SDK default. The OpenHands/K3.1 Pro
reports do not record an explicit max_tokens; other Kimi reports redact that
field as `[REDACTED]`, so its numeric value cannot be verified from metadata.
**Local instrument access is an exception to the shared Docker benchmark setup:**
Docker CPU, memory and PID limits do not apply. Thread environment settings and
Langfuse export success cannot be established from the supplied report files.

Reports and launcher metadata were imported from the supplied
`corral-mini-result/data` (Claude, subsequently renamed `data_claude`),
`corral-mini-result/data_gpt` and `corral-mini-result/data_kimi` folders without changing their bytes (reports use
lossless gzip compression). No trials were rescored, replaced or excluded.
Per-run `provenance.json` records source folder names, SHA-256 hashes, original
subset versions and the target v2 manifest hash. Source Git commits were not
recorded in the supplied files and are left unknown.

The 120 reused Level 2 trials (all six Claude runs, GPT-6.1-Sol/Codex and
Kimi K3.1 Pro/OpenHands)
retain their original v1 launcher metadata and run IDs. Their task coverage
matches v2 exactly. The other 420 trials record v2 in launcher metadata. This reuse does not rename trials or invent new sessions.

Original execution IDs and run IDs are preserved. The index builder derives
Langfuse trace/session links using its configured host and project. These links
have **not been verified** against Langfuse; importing reports does not upload
traces. This GitHub update has not been uploaded to Hugging Face.

## Errors and caveats

33 terminal errors are preserved: 15 iteration limits, three context limits,
three missing submissions, one missing FinishAction and 11 generic provider errors.
All retain zero scores. The other 173 zero scores have no saved terminal error;
individual scoring outcomes have not been independently audited. GPT-5/Codex
Level 2 has zero passes out of 15 without saved terminal errors.

**Eleven Kimi K3.1 Pro/ReAct Level 2 trials remain provisional:** their stored
terminal message is only `Provider returned error`. Operator logs from this
configuration showed Modal shared-pool rate limiting, but the report does not
establish the cause of each failed trial. These are published unchanged for
transparency, not classified as confirmed model-performance failures.

## Validation

Verified source hashes and decompressed byte equality, 540 unique execution IDs,
three exact v2 task IDs per run, five trials per task, and index reconstruction
using the repository's report checker. Full AFM reports remain unchanged.

```bash
python reports/build_index.py reports/afm/mini-v2
python reports/check_reports.py
```
