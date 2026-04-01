# Intervention Experiment

Measures whether injecting steps from successful traces helps agents succeed on tasks they otherwise struggle with.

## Flow

```
reports_v2 ──> select tasks ──> run baseline ──> pick traces from baseline ──> run interventions ──> analyze
```

1. **Select tasks** from existing `reports_v2/` (tasks with mixed success/failure)
2. **Run baseline** (no intervention, 15 trials) — this is the control AND the source of traces
3. **Build trace registry** from baseline output — picks one success + one failure trace per task (random, seeded)
4. **Run intervention conditions** using traces from baseline
5. **Analyze** recovery curves

## Step-by-Step

All commands run from `reports_v3/intervention/`.

### 1. Select tasks

```bash
uv run python scripts/select_traces.py
```

Outputs `task_selection.json` — task IDs per (env, agent) based on `reports_v2/` mixed results.

### 2. Launch baselines

```bash
# Preview
./scripts/launch_sweep.sh --baseline --dry-run

# Launch all 6 baselines (one per env/agent)
./scripts/launch_sweep.sh --baseline

# Or just one
./scripts/launch_sweep.sh --baseline --env spectra --agent react
```

Each baseline runs from `runs/{env}/{agent}/baseline/`. Traces, checkpoints, and reports land there.

```bash
# Monitor
find runs -name 'run.log' -path '*/baseline/*' -exec tail -1 {} +

# Check completion
find runs -name '*_report.json' -path '*/baseline/*'
```

### 3. Build trace registry

After **all baselines finish**:

```bash
uv run python scripts/build_trace_registry.py
```

Scans baseline run directories, matches traces to trial outcomes, picks one success + one failure trace per task (randomly, seed=42). Outputs `trace_registry.json`.

```bash
# Optional: validate traces
uv run python scripts/validate_traces.py
```

### 4. Launch interventions

```bash
# Preview
./scripts/launch_sweep.sh --intervention --dry-run

# Launch all
./scripts/launch_sweep.sh --intervention

# Or filter
./scripts/launch_sweep.sh --intervention --env wetlab --max-parallel 4
```

Each condition runs from `runs/{env}/{agent}/{condition}/`.

### 5. Analyze

```bash
uv run python analysis/aggregate_results.py      # -> analysis/results.csv
uv run python analysis/plot_recovery_curves.py    # -> analysis/figures/
uv run python analysis/statistical_tests.py       # -> analysis/statistical_tests.csv
```

## Conditions

| Condition | What's injected |
|-----------|----------------|
| `baseline` | Nothing (no intervention) — also the source of traces |
| `success_step1` | First step from a successful baseline trace |
| `success_step2` | First 2 steps |
| `success_stepn2` | All except last 2 steps |
| `success_stepn1` | All except last step (agent only does final submission) |
| `failed_step*` | Same as above, from a failed baseline trace (control) |

All interventions use `execute_tools=True` — tool calls from the trace are re-executed to get fresh observations.

## Directory Layout

```
reports_v3/intervention/
├── config.py                    # Shared constants
├── run_intervention.py          # Main runner (launched from run dirs)
├── task_selection.json          # Step 1 output: which tasks to run
├── trace_registry.json          # Step 3 output: traces from baseline
├── scripts/
│   ├── select_traces.py         # Step 1: pick tasks from reports_v2
│   ├── build_trace_registry.py  # Step 3: pick traces from baseline
│   ├── validate_traces.py       # Optional: sanity check traces
│   ├── sweep_config.py          # Generate condition matrix
│   └── launch_sweep.sh          # Headless launcher
├── runs/                        # Each run in its own dir
│   └── {env}/{agent}/{condition}/
│       ├── run.log
│       ├── run.pid
│       ├── agent_logs-*/
│       └── *_report.json
└── analysis/
    ├── aggregate_results.py
    ├── plot_recovery_curves.py
    └── statistical_tests.py
```

## Configuration

Edit `config.py` to change selected tasks, model, temperature, trials, WandB settings, etc.
