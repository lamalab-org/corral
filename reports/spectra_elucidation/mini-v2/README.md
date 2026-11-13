# Spectra elucidation reports: corral-mini v2

30 of 36 runs (18 agent/model pairs × levels 1 and 2), 3 tasks per level ×
k = 5, 450 trials. Six runs are still running (Kimi and one Claude ReAct run)
and will be added. These replace the corral-mini v1 batch-1 reports. Layout,
names and index columns follow [`../../README.md`](../../README.md). The
October full-benchmark runs are in [`../`](../).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 1 | `55_55555_orgsyn_555_5555`, `66_66666_orgsyn_666_6666`, `88_88888_orgsyn_888_8888` |
| 2 | `10_15227_orgsyn_084_0011_sub2`, `10_15227_orgsyn_096_0036`, `88_88888_orgsyn_888_8888` |

No task is shared with corral-mini v1 at the same level, so every trial was
run for v2.

## Results (passes out of 15 per level)

| Batch | Agent | Model | L1 passes | L1 pass@5 | L1 pass^5 | L2 passes | L2 pass@5 | L2 pass^5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | openhands | kimi-k3.1-pro | 4 | 0.55 | 0.03 | 4 | 0.33 | 0.11 |
| 1 | react | kimi-k3.1-pro | 9 | 0.89 | 0.22 | 4 | 0.55 | 0.03 |
| 1 | claude-code | claude-opus-5-5 | 11 | 1.00 | 0.24 | 8 | 0.97 | 0.06 |
| 1 | react | claude-opus-5-5 | 2 | 0.31 | 0.00 | 5 | 0.64 | 0.03 |
| 1 | codex | gpt-6.1-sol | 12 | 0.97 | 0.67 | 12 | 1.00 | 0.47 |
| 1 | react | gpt-6.1-sol | 14 | 1.00 | 0.78 | 13 | 1.00 | 0.69 |
| 2 | openhands | kimi-k2.5 | 3 | 0.53 | 0.00 | 4 | 0.33 | 0.11 |
| 2 | react | kimi-k2.5 | 3 | 0.53 | 0.00 | pending | | |
| 2 | claude-code | claude-opus-4-7 | 9 | 0.89 | 0.36 | 3 | 0.53 | 0.00 |
| 2 | react | claude-opus-4-7 | 5 | 0.64 | 0.03 | 2 | 0.31 | 0.00 |
| 2 | codex | gpt-5.6-sol | 4 | 0.61 | 0.01 | 1 | 0.22 | 0.00 |
| 2 | react | gpt-5.6-sol | 4 | 0.55 | 0.03 | 3 | 0.33 | 0.03 |
| 3 | openhands | kimi-k2-0905 | pending | | | pending | | |
| 3 | react | kimi-k2-0905 | pending | | | pending | | |
| 3 | claude-code | claude-opus-4-5-20251101 | 4 | 0.61 | 0.01 | 9 | 0.67 | 0.44 |
| 3 | react | claude-opus-4-5-20251101 | 9 | 0.97 | 0.14 | pending | | |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 1 | 0.22 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

162 of 450 trials pass. Per task (merged runs only):

| Level | Task | Passes |
|---|---|---:|
| 1 | `55_55555_orgsyn_555_5555` | 40/80 |
| 1 | `66_66666_orgsyn_666_6666` | 28/80 |
| 1 | `88_88888_orgsyn_888_8888` | 25/80 |
| 2 | `10_15227_orgsyn_084_0011_sub2` | 39/70 |
| 2 | `10_15227_orgsyn_096_0036` | 22/70 |
| 2 | `88_88888_orgsyn_888_8888` | 8/70 |

Failures are mostly wrong structures. Six trials end with an error (five
Claude Code turn limits, one Claude Code run without a submission).

## Settings

k = 5, 50 iterations, reasoning effort `medium` (provider default for
kimi-k2-0905), surrender off, `--verbose`, Docker sandbox with 2 CPUs, 6 GB
and a 4096 PID limit per trial, `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/
`MKL_NUM_THREADS` = 2, Langfuse on. Kimi via OpenRouter (`max_tokens` 32768,
no fallbacks): kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on Novita.
Codex enforced the 50-call budget (#465).

Corral at `ad800a2a5` (`origin/main` `2ef35fb3c`). Sandbox images were built
from `docker/benchmark.Dockerfile` with pip dependencies pinned to the host
task venv's uv-locked versions (`PIP_CONSTRAINT`; unpinned, the build fails on
source-only pandas and chroma-hnswlib) and Node.js 22 with
`isotopic-distribution`, `nmr-processing` and `openchemlib` in
`/usr/local/share/corral-spectra-js`, which the MS and HSQC/NMR predictors
need.

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's trials share one session named after the merged run
ID (`spectra_elucidation-mini-v2-l<L>-<agent>-<model>`, in
`run-metadata.json`); all 450 traces were found in their sessions
(`build_index.py --verify`).

## Caveats

- **IR prediction service outages.** The spectrum predictors call
  `ir.cheminfo.org`, which timed out often during these runs: 310 of the 450
  trials got "IR prediction failed" for at least one candidate (NMR
  predictions failed far less often). The v1 runs saw the same service fail
  in 61 of 180 trials. Agents still had the task's measured spectra and the
  other predictors; how much the outages lowered scores is not measured.
- **50 trials were rerun after the provider accounts ran out of credit or hit
  a rate limit** (39 Kimi, 10 OpenAI, 1 Kimi rate limit); the reports hold the
  reruns only. Rerun traces share the trace of their failed attempt (see
  "Known issue: some traces mix attempts" in `../README.md`).
- **Answer files are not visible to agents.** The task JSONs ship inside the
  image under `/opt/corral` (mode 0700); the agent user cannot read them.
- **Machine load.** Runs shared the machine with the resistor, retro and
  wetlab pools; wall-clock durations are not comparable with other runs.
- **Comparison with the October full runs.** The full runs cover all 20 tasks
  per level; compare per task only.
