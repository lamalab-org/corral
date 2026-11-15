# Stargazer reports: corral-mini v2

24 of 36 runs (18 agent/model pairs × levels 1 and 2), 3 tasks per level ×
k = 5, 360 trials. Claude and OpenAI are complete; the 12 Kimi runs (180
trials) are not run yet. Layout, names and index columns follow
[`../../README.md`](../../README.md). The October full-benchmark runs are in
[`../`](../).

[Run index](runs.csv) · [Trial index with Langfuse links](trials.csv)

## Tasks

| Level | Tasks |
|---|---|
| 1 | `seed21_diff5`, `seed29_diff3`, `seed3_diff4` |
| 2 | `seed122_diff10`, `seed176_diff9`, `seed1_diff8` |

No task is shared with corral-mini v1 at the same level, so every trial was
run for v2. No October trial solved any of these six tasks (0 of 35 each).

## Results

**No trial passes** (0 of 360; pass@5 and pass^5 are 0 for every run).

| Batch | Agent | Model | L1 passes | L2 passes |
|---|---|---|---:|---:|
| 1 | claude-code | claude-opus-5-5 | 0/15 | 0/15 |
| 1 | react | claude-opus-5-5 | 0/15 | 0/15 |
| 1 | codex | gpt-6.1-sol | 0/15 | 0/15 |
| 1 | react | gpt-6.1-sol | 0/15 | 0/15 |
| 2 | claude-code | claude-opus-4-7 | 0/15 | 0/15 |
| 2 | react | claude-opus-4-7 | 0/15 | 0/15 |
| 2 | codex | gpt-5.6-sol | 0/15 | 0/15 |
| 2 | react | gpt-5.6-sol | 0/15 | 0/15 |
| 3 | claude-code | claude-opus-4-5-20251101 | 0/15 | 0/15 |
| 3 | react | claude-opus-4-5-20251101 | 0/15 | 0/15 |
| 3 | codex | gpt-5-2025-08-07 | 0/15 | 0/15 |
| 3 | react | gpt-5-2025-08-07 | 0/15 | 0/15 |
| 1 | openhands, react | kimi-k3.1-pro | pending | pending |
| 2 | openhands, react | kimi-k2.5 | pending | pending |
| 3 | openhands, react | kimi-k2-0905 | pending | pending |

A trial passes when the submitted planet solution reaches a match score of at
least 0.8 against the true system (plus the count, RMS and BIC criteria).
Agents mostly converge on the same solution per task, below that bar:

| Level | Task | Most common match score (trials) | Best match score |
|---|---|---|---:|
| 1 | `seed21_diff5` | 0.14 (16 of 60) | 0.56 |
| 1 | `seed29_diff3` | 0.64 (47 of 60) | 0.65 |
| 1 | `seed3_diff4` | 0.65 (50 of 60) | 0.67 |
| 2 | `seed122_diff10` | 0.46 (40 of 60) | 0.69 |
| 2 | `seed176_diff9` | −0.46 (30 of 60) | 0.05 |
| 2 | `seed1_diff8` | 0.26 (44 of 60) | 0.27 |

An identifiability check on the level 1 tasks found that the best fit to the
observed data alone reaches match 0.14, 0.64 and 0.65, close to what the
agents submit: the data may not determine the true system well enough to
pass. Whether to reselect these tasks is open (see the PR).

5 trials end with an error, kept as failures: 4 used up the 50-step budget
(ReAct 2, Claude Code 1, Codex 1) and OpenAI flagged one ReAct prompt as a
possible policy violation. Most trials submit within minutes (median 4.5
minutes, longest 4.4 hours).

## Settings

k = 5, 50 iterations, reasoning effort `medium`, surrender off, `--verbose`,
Docker sandbox with 2 CPUs, 6 GB and a 4096 PID limit per trial,
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 2, Langfuse on.
Codex enforced the 50-call budget (#465).

Corral at `d001507c8`; the stargazer task code is unchanged on `main` since.
Sandbox images were built from `docker/stargazer.Dockerfile` (locked REBOUND
first) with pip dependencies pinned to the host task venv's uv-locked versions
(unpinned, pip backtracks to a source-only pandas) and `agent-client-protocol`
0.12.1 (1.0.0 breaks the OpenHands SDK), as in the other environments.

Each trial ran as its own `corral bench --trials 1` run; the reports are
merged with `scripts/merge_trial_reports.py` (metrics recomputed, messages and
tool calls rebuilt from each trial's final state, source runs in
`provenance.json`).

**Langfuse.** Each run's trials share one session named after the merged run
ID (`stargazer-mini-v2-l<L>-<agent>-<model>`, in `run-metadata.json`);
all 360 traces were found in their sessions (`build_index.py --verify`).

## Caveats

- **Reruns.** 3 trials failed at startup and were rerun: 2 reused a workspace
  volume left by a smoke test ("Blind trials require an empty initial
  workspace") and 1 Codex trial ran out of memory at 6 GB. The reports hold
  the reruns.
- **astropy is not installed**, as in October; agents that try it fall back to
  scipy.
- **PythonREPL.** The pre-execution shadowing guard that rejected some October
  code is no longer applied, so those false rejections cannot recur.
- **Machine load.** Runs shared the machine with the resistor network, spectra
  and retrosynthesis pools; wall-clock durations are not comparable with other
  runs.
- **Comparison with the October full runs.** The full runs cover all tasks per
  level; compare per task only.
