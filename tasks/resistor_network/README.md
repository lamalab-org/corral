# Resistor Network Environment — v4

Infer a full resistor network from sparse effective-resistance observations.
The runtime serves static JSON in `environments/level_*/tasks_json/`: ten
structurally distinct tasks per level, with five trials per task in the full
agent evaluation.

## Values, circuits, and public inputs

V4 uses **discrete resistance values from a published list of 30 choices**.
Values may repeat; their per-value counts are not disclosed. This makes exact
value recovery accessible while preserving the wiring search. Continuous
values remain a separate calibration variant.

```text
10, 11, 12, 14, 15, 17, 19, 21, 23, 25, 28, 31, 35, 38, 42,
47, 52, 58, 64, 71, 79, 88, 97, 108, 119, 132, 147, 163, 180, 200 ohms
```

| Property | Level 1 | Level 2 |
|---|---|---|
| Resistors | 12–16 | 18–24 |
| Private generation structure | Two coupled modules sharing an articulation node | One biconnected core, minimum node degree three |
| Construction | Random cycles with chords | Random cubic graphs with added chords |
| Observations | 18–24 | 27–36 |
| Default/saved seed | 202610141 | 202610142 |

Agents receive observations, resistor count, allowed values, and these rules:

- Exactly the nodes appearing in the readings; arbitrary resistor IDs.
- A connected simple graph with no self-loops or direct parallel edges.
- Any pair of named nodes can contain a resistor.
- Ideal resistors; values and effective resistances are in ohms.
- Readings rounded to three decimal places.
- Every reading must match within 0.000500001 ohms.

Circuit families, physical connections, and construction details are private.
All nodes appear in readings, so a separate node inventory is unnecessary.

## Measurement selection and feasibility

For `m` resistors, publish `m + ceil(m / 2)` observed pairs. This is a practical
calibration policy, not a theorem that every reading is essential or that the
original wiring is uniquely identifiable.

Pairs are selected across all visible nodes, including A–B, without seeding
physical resistor edges. Every accepted subset meets these checks:

- Each node appears at least twice; no node has a complete probe star.
- At most half of all possible pairs are observed.
- At least `ceil(m / 3)` physical edge pairs remain unobserved.
- The log-value measurement Jacobian has full column rank.
- First-order rounding stability passes the offline screening.
- Every resistor affects readings when opened or shorted.
- Fitting known wiring from a uniform 47-ohm start and snapping to the
  published pool reproduces all readings.

The generator checks up to 24 random single-edge rewires over all visible
pairs. It refits continuous values from two starts, snaps to the published
pool, and rejects demonstrated indistinguishable rewires. This bounded audit
does not establish global uniqueness. All observationally equivalent valid
circuits remain acceptable submissions.

`archives/resistor_network_v4_audit.json` records bank hashes, independent
scoring of ground truth, known-wiring value recovery, and an adaptation of the
old measured-pair fit/prune/snap shortcut. Shortcut failure does not establish
global hardness or practical recoverability for every agent.

## Tools and scoring

Tasks expose `simulate_circuit_resistance` for one supplied node pair and
`validate_circuit_topology`, plus standard workspace and submission tools.
These operate on the proposed circuit. There is no environment fitting,
topology-proposal, or batch validation tool. Native coding agents can still
write numerical solvers using their own execution tools.

Submit a JSON string, for example:

```json
{"resistors":{"R1":31,"R2":42},"connections":[["A","N1","R1"],["N1","B","R2"]]}
```

`sparse_circuit_solution` checks duplicate JSON keys, positive finite values,
pool membership, exact resistor count and node set, exactly-once resistor use,
unique node-pair edges, and connectivity. It independently simulates every
reading and requires each absolute error to be at most 0.000500001 ohms.
Submitted wiring is not compared with private wiring. An optional pool
parameter preserves continuous scoring for archived calibration tasks.

## Generation, plots, and tests

```bash
cd tasks/resistor_network
uv sync
uv run python environments/level_1/generate_tasks.py
uv run python environments/level_2/generate_tasks.py
uv run pytest tests
```

Generators require ten tasks, reject isomorphic circuits within a level, and
build the complete batch before replacing saved JSON. Seeds reproduce content
except UUIDs. Archive inputs before regeneration when reproducing reports.

Overviews, individual diagrams, and a 20-page PDF are saved in `figures/v4/`.
Diagrams include private values for human review, outside agent prompts and
trial workspaces.

The saved JSON bank matches the frozen benchmark inputs used for the v4 runs.
Its file hashes are recorded in `environments/task-bank-manifest.json`.

## Archives and calibration

`archives/resistor_network_v2_20261010.zip` preserves yesterday's benchmark.
`archives/resistor_network_v3_continuous_20261010.zip` preserves the continuous
bank, calibration code, inputs, diagnostics, and plots before v4. Both contain
verified SHA-256 manifests. External reports and traces remain in their
original versioned data directories.

`archives/resistor_network_calibration_20261014.zip` preserves the continuous
and 30-value calibration banks, their manifests, and the v3 diagrams. Its
`SHA256SUMS.json` verifies each archived file. These historical task banks are
excluded from the active environment directories; only the 20 v4 tasks remain.
