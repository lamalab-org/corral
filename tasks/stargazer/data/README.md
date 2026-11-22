# Stargazer task bank

The 32 synthetic records include 20 current benchmark tasks and 12 tasks kept
for historical replay. The loader preserves RV-only records and converts
legacy REBOUND records to RV-only coordinates while preserving their noise
realizations. The [selection manifest](selection_manifest.json) records the
upstream revisions, generation provenance and membership of each selection.

## Level 1 v4

The default Level 1 has two single-planet tasks at each Stargazer difficulty
from 1 through 4 and two near-resonant double-planet tasks at difficulty 5.
Six tasks carry over from v3. The four replacements below passed all gates of
the complete scorer on their released observations, including mean orbital
match ≥ 0.80, planet count, complete pairing, RMS and BIC improvement.

| Previous task | New task | Original match | RMS / limit (m/s) | ΔBIC per point | Fresh noise passes |
|---|---|---:|---:|---:|---:|
| seed29_diff3 | seed111_diff3 | 0.939 | 1.29 / 1.96 | 74.4 | 97/100 |
| seed3_diff4 | seed630238_diff4 | 0.963 | 3.60 / 6.05 | 511.4 | 99/100 |
| seed21_diff5 | seed610696_diff5 | 0.937 | 2.67 / 4.44 | 1074.7 | 97/100 |
| seed15_diff5 | seed611761_diff5 | 0.946 | 0.58 / 0.94 | 1655.7 | 98/100 |

`seed111_diff3` comes from the released Stargazer bank. The other three records
use Stargazer's `generate_task` at revision
`7c59e572e15648aeb4965dd39d1f557036cb6b53`, with the seed in the ID, a
fixed multiplicity of one or two as appropriate, and the remaining defaults.

The table reports numerical recovery using Corral's locked dependencies.
`fit_counts` takes only the public observations and stellar mass, searches
candidate counts and selects the fit with the lowest BIC. Each fresh draw
keeps the timestamps and uncertainties and samples white noise plus jitter.
A pass requires the selected fit to converge and pass the complete scorer.
All ten v4 records use white noise and jitter. The assessment concerns RV fits
over the observation window. In particular, the original three-dimensional
configuration of `seed610696_diff5` exhibited a close encounter in extended
N-body integrations beyond its 16.2-day observation window.

Reproduce the table from `tasks/stargazer` with
`OPENBLAS_NUM_THREADS=1 uv run --frozen python scripts/check_level1_v4.py`.
Trial 0 uses the released observations. Trials 101–200 use fresh noise.
Agent performance on the four replacement tasks requires new runs.

## Replay and reference audit

Pass [the v3 selector](selectors/level_1_v3.json) as `selector_path` to replay
the October Level 1 membership. The [original selector](selectors/level_1_original.json)
preserves the earlier difficulties 5–7 selection. Level 2 retains its ten tasks
at difficulties 8–10.

[reference_audit.json](reference_audit.json) records how every bundled task's
canonical submission scores. Regenerate it with `python -m stargazer.audit`
and verify it with `python -m stargazer.audit --check`.
See the [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution.
