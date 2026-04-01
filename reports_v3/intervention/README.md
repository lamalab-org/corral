# Intervention Experiment

Measures whether injecting steps from successful traces helps agents succeed on tasks they otherwise struggle with.

## Flow

```
reports_v2 ──> select tasks ──> start servers ──> run baseline ──> pick traces ──> run interventions ──> analyze
```

1. **Set up environments** — create venvs for each benchmark environment
2. **Select tasks** from existing `reports_v2/` (tasks with mixed success/failure)
3. **Start servers** — each environment runs as a server on its own port
4. **Run baseline** (no intervention, 15 trials) — control + source of traces
5. **Build trace registry** from baseline — picks one success + one failure trace per task
6. **Run intervention conditions** using traces from baseline
7. **Analyze** recovery curves

## Prerequisites

- `uv` for spectra, resistor, and the main corral venv
- `micromamba` (or `conda`/`mamba`) for wetlab (reaktoro is conda-only)

If micromamba is not installed, `setup_envs.sh` will download it automatically.

## Step-by-Step

All commands run from `reports_v3/intervention/`.

### 0. Set up environment venvs (one-time)

```bash
./scripts/setup_envs.sh
```

This creates `.venv` in each environment directory:
- `tasks/spectra_elucidation/.venv` — uv, Python 3.11
- `tasks/resistor_network/.venv` — uv, Python 3.12
- `tasks/wetlab/.venv` — micromamba, Python 3.10 (osx-arm64) or 3.12 (linux-64)

You can also set up individual environments:
```bash
./scripts/setup_envs.sh spectra
./scripts/setup_envs.sh wetlab
```

### 1. Select tasks

```bash
uv run python scripts/select_traces.py
```

Outputs `task_selection.json` — task IDs per (env, agent) based on `reports_v2/` mixed results.

### 2. Start environment servers

```bash
# Start all three
./scripts/launch_sweep.sh --start-servers

# Or just one
./scripts/launch_sweep.sh --start-servers --env spectra

# Check status
./scripts/launch_sweep.sh --server-status

# View logs
tail -f servers/spectra.log
```

The environment server is **stateful** (per-task state mutated during trials), so
parallel agents on the same server would clash. We start **two server instances
per environment** — one per agent type — on different ports.

Servers:
| Environment | React Port | ToolCalling Port | Module | Args |
|-------------|-----------|-----------------|--------|------|
| Spectra | 8002 | 8012 | `spectra_elucidation.env` | `--level 2` |
| Resistor | 8001 | 8011 | `resistor_network.env` | `--mode single` |
| WetLab | 8003 | 8013 | `wetlab.env` | `--level 2` |

### 3. Launch baselines

```bash
# Preview
./scripts/launch_sweep.sh --baseline --dry-run

# Launch all 6 baselines (one per env/agent)
./scripts/launch_sweep.sh --baseline

# Or just one
./scripts/launch_sweep.sh --baseline --env spectra --agent react

# Limit parallelism
./scripts/launch_sweep.sh --baseline --max-parallel 2
```

Each baseline runs from `runs/{env}/{agent}/baseline/`. Traces, checkpoints, and reports land there.

```bash
# Monitor
find runs -name 'run.log' -path '*/baseline/*' -exec tail -1 {} +

# Check completion
find runs -name '*_report.json' -path '*/baseline/*'
```

### 4. Build trace registry

After **all baselines finish**:

```bash
uv run python scripts/build_trace_registry.py
```

Scans baseline run directories, matches traces to trial outcomes, picks one success + one failure trace per task (randomly, seed=42). Outputs `trace_registry.json`.

```bash
# Optional: validate traces
uv run python scripts/validate_traces.py
```

### 5. Launch interventions

```bash
# Preview
./scripts/launch_sweep.sh --intervention --dry-run

# Launch all
./scripts/launch_sweep.sh --intervention

# Or filter
./scripts/launch_sweep.sh --intervention --env wetlab --max-parallel 4
```

Each condition runs from `runs/{env}/{agent}/{condition}/`.

### 6. Analyze

```bash
uv run python analysis/aggregate_results.py      # -> analysis/results.csv
uv run python analysis/plot_recovery_curves.py    # -> analysis/figures/
uv run python analysis/statistical_tests.py       # -> analysis/statistical_tests.csv
```

### 7. Stop servers

```bash
./scripts/launch_sweep.sh --stop-servers
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
├── trace_registry.json          # Step 4 output: traces from baseline
├── servers/                     # Server PID files and logs
│   ├── spectra.pid / spectra.log
│   ├── resistor.pid / resistor.log
│   └── wetlab.pid / wetlab.log
├── scripts/
│   ├── setup_envs.sh           # One-time venv setup
│   ├── select_traces.py        # Step 1: pick tasks from reports_v2
│   ├── build_trace_registry.py # Step 4: pick traces from baseline
│   ├── validate_traces.py      # Optional: sanity check traces
│   ├── sweep_config.py         # Generate condition matrix
│   └── launch_sweep.sh         # Server + run launcher
├── runs/                       # Each run in its own dir
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
