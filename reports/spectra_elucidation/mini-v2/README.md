# Spectra elucidation reports: corral-mini v2

36 runs (18 agent/model pairs × levels 1 and 2), 3 tasks per level × k = 5,
540 trials. These replace the corral-mini v1 batch-1 reports. Layout, names and
index columns follow [`../../README.md`](../../README.md). The October
full-benchmark runs are in [`../`](../).

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
| 1 | openhands | kimi-k3.1-pro | 3 | 0.53 | 0.00 | 2 | 0.31 | 0.00 |
| 1 | react | kimi-k3.1-pro | 4 | 0.76 | 0.00 | 3 | 0.33 | 0.03 |
| 1 | claude-code | claude-opus-5-5 | 11 | 1.00 | 0.24 | 6 | 0.86 | 0.03 |
| 1 | react | claude-opus-5-5 | 1 | 0.22 | 0.00 | 4 | 0.55 | 0.03 |
| 1 | codex | gpt-6.1-sol | 14 | 1.00 | 0.78 | 9 | 0.89 | 0.22 |
| 1 | react | gpt-6.1-sol | 14 | 1.00 | 0.78 | 11 | 0.97 | 0.45 |
| 2 | openhands | kimi-k2.5 | 2 | 0.45 | 0.00 | 0 | 0.00 | 0.00 |
| 2 | react | kimi-k2.5 | 3 | 0.53 | 0.00 | 2 | 0.31 | 0.00 |
| 2 | claude-code | claude-opus-4-7 | 7 | 0.78 | 0.33 | 3 | 0.33 | 0.03 |
| 2 | react | claude-opus-4-7 | 5 | 0.64 | 0.03 | 3 | 0.53 | 0.00 |
| 2 | codex | gpt-5.6-sol | 2 | 0.45 | 0.00 | 5 | 0.84 | 0.01 |
| 2 | react | gpt-5.6-sol | 6 | 0.64 | 0.11 | 1 | 0.22 | 0.00 |
| 3 | openhands | kimi-k2-0905 | 4 | 0.55 | 0.03 | 2 | 0.31 | 0.00 |
| 3 | react | kimi-k2-0905 | 6 | 0.64 | 0.11 | 2 | 0.45 | 0.00 |
| 3 | claude-code | claude-opus-4-5-20251101 | 6 | 0.92 | 0.01 | 9 | 0.67 | 0.44 |
| 3 | react | claude-opus-4-5-20251101 | 5 | 0.64 | 0.03 | 6 | 0.56 | 0.33 |
| 3 | codex | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 2 | 0.31 | 0.00 |
| 3 | react | gpt-5-2025-08-07 | 0 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |

163 of 540 trials pass (Claude and OpenAI 136 of 360, Kimi 27 of 180).
gpt-6.1-sol leads on both levels; gpt-5 solves 2 trials. Per task:

| Level | Task | Passes |
|---|---|---:|
| 1 | `55_55555_orgsyn_555_5555` | 41/90 |
| 1 | `66_66666_orgsyn_666_6666` | 31/90 |
| 1 | `88_88888_orgsyn_888_8888` | 21/90 |
| 2 | `10_15227_orgsyn_084_0011_sub2` | 39/90 |
| 2 | `10_15227_orgsyn_096_0036` | 25/90 |
| 2 | `88_88888_orgsyn_888_8888` | 6/90 |

Failures are mostly wrong structures. 44 trials end with an error, all kept as
failures: 41 used up the 50-step budget (ReAct interaction budget, OpenHands
iteration limit, Claude Agent SDK `error_max_turns`), 2 OpenHands runs and 1
Claude Code run finished without calling `submit_answer`.

## Settings

k = 5, 50 iterations, reasoning effort `medium` (provider default for
kimi-k2-0905), surrender off, `--verbose`, Docker sandbox with 2 CPUs, 6 GB
and a 4096 PID limit per trial, `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/
`MKL_NUM_THREADS` = 2, Langfuse on. Kimi via OpenRouter (`max_tokens` 32768,
no fallbacks): kimi-k3.1-pro on Modal, kimi-k2.5 and kimi-k2-0905 on Novita.
Codex enforced the 50-call budget (#465).

Sandbox images were built from `docker/benchmark.Dockerfile` with pip
dependencies pinned to the host task venv's uv-locked versions
(`PIP_CONSTRAINT`; unpinned, the build fails on source-only pandas and
chroma-hnswlib) and Node.js 22 with `isotopic-distribution`, `nmr-processing`
and `openchemlib` in `/usr/local/share/corral-spectra-js`, which the MS and
HSQC/NMR predictors need. The images were rebuilt as fixes landed; trials by
the Corral commit of the image they ran on: `ad800a2a5` 63, `8007b69e8` 35,
`6ca1b8505` 2, `c667e9b10` 31, `6ea2430da` 11, `d001507c8` 398. The spectra task
changed in that span only by the prompt fix and the IR changes described under
Caveats.

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's trials share one session named after the merged run
ID (`spectra_elucidation-mini-v2-l<L>-<agent>-<model>`, in
`run-metadata.json`); all 540 traces were found in their sessions
(`build_index.py --verify`).

## Caveats

- **IR predictor.** Until 2026-11-13 15:58 the IR predictions came from
  `ir.cheminfo.org`, which failed for hours on 2026-11-12/13. Later trials use a
  local copy of the same service ([cheminfo-py/xtbservice](https://github.com/cheminfo-py/xtbservice),
  `GFN2xTB`, through `SPECTRA_IR_URL`, #496). 131 trials used the public
  service, 409 the local one. On 30 random molecules both gave identical spectra
  for 20; the rest differ by conformer sampling (a few cm-1, at most 47 cm-1 on
  one N-H stretch). Until the images were rebuilt with #497 (2026-11-13 18:54),
  `simulate_spectra` ran in the sandbox, which does not see `SPECTRA_IR_URL`, and
  still called the public service; 11 kept trials ran in that window, none with
  an IR failure.
- **Clean-run rule.** Every reported trial finished without a prediction
  failure that points at infrastructure. Attempts with such a failure were set
  aside and rerun. Not counted as infrastructure, and kept as the trial's
  outcome: predictions for SMILES that RDKit cannot parse (22 trials), and
  predictions that fail for the same valid molecule every time while the
  predictor works for ethanol (22 trials, mostly on `88_88888`). The remote
  proton-NMR predictor that `nmr-processing` calls answers "Molecules are
  different" for 73 such molecules (mostly 2-aryl-thiazolines), and xtb finds no
  conformer for 7. An earlier version of the rule also reran those; 22 later
  reruns were discarded and the earliest valid attempt restored.
- **Reruns.** 475 of the 540 trials have at least one earlier attempt set
  aside: IR outages and the clean-run rule, provider credit (Kimi 103, OpenAI
  37), one Kimi rate limit, and the prompt fix below. The reports hold one
  valid attempt per trial. Rerun traces can share the trace of their failed
  attempt (see "Known issue: some traces mix attempts" in `../README.md`).
- **Prompt fix (2026-11-12 15:38).** The task prompt lost an always-empty
  inputs block. OpenAI and Kimi trials started before it were rerun (85); 94
  Claude trials from before it are kept.
- **Answer files are not visible to agents.** The task JSONs ship inside the
  image under `/opt/corral` (mode 0700); the agent user cannot read them.
- **Machine load.** Runs shared the machine with the resistor, retro, wetlab
  and stargazer pools; wall-clock durations are not comparable with other runs.
- **Comparison with the October full runs.** The full runs cover all 20 tasks
  per level; compare per task only.
