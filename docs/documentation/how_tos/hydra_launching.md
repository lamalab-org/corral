# How to use Hydra Launching for Benchmarks

`Corral` supports [Hydra](https://hydra.cc/) for composable, sweepable benchmark configuration. This is especially useful when you need to run many combinations of agents, models, and parameters — a common pattern in benchmarking research.

## Installation

Install `Corral` with the `hydra` extra:

```bash
uv pip install -e ".[hydra]"
```

This adds `hydra-core`, `omegaconf`, and `hydra-joblib-launcher`.

## Quick Start

After installation, a new CLI command `corral-hydra` is available. By default it runs in **local mode** — this means the agent runs directly in your Python process, but you need an environment server already running (e.g. via `corral bench env` or manually with `uvicorn`).

```bash
# 1. Start the environment server first (in a separate terminal)
corral bench env --image ghcr.io/lamalab-org/corral-materials:latest

# 2. Run the agent against it (default: ToolCallingAgent, gpt-4o, 5 trials)
corral-hydra

# Choose a different agent
corral-hydra agent=react

# Override any parameter on the fly
corral-hydra agent=react agent.model=anthropic/claude-sonnet-4-5-20250929 agent.temperature=0.5
```

If you prefer Docker to handle everything (environment + agent containers), use `mode=docker`:

```bash
corral-hydra mode=docker docker.image=ghcr.io/lamalab-org/corral-materials:latest
```

No config files to edit — just override what you need from the command line.

## Configuration Structure

Hydra composes the final config from modular YAML files. The config groups are:

```
src/corral/conf/
├── config.yaml              # root config (defaults list)
├── agent/
│   ├── react.yaml           # ReActAgent
│   ├── tool_calling.yaml    # ToolCallingAgent
│   ├── llm_planner.yaml     # LLMPlanner
│   └── reflexion.yaml       # ReflexionAgent
├── runner/
│   ├── default.yaml         # 5 trials, k=[1,3,5]
│   ├── quick.yaml           # 1 trial (smoke test)
│   └── full.yaml            # 10 trials, k=[1,3,5,10], surrender enabled
└── docker/
    ├── default.yaml          # GHCR environment image
    └── local.yaml            # locally-built agent image
```

### Agent Configs

Each agent config only specifies the class (`_target_`) — everything else (model, temperature, iterations) is overridable from the CLI:

```yaml
# conf/agent/react.yaml
_target_: corral.agents.react.ReActAgent
model: openai/gpt-4o
max_iterations: 20
temperature: 1.0
api_endpoint: null
system_prompt: null
user_prompt: null
extractor_prompt: null
surrender_prompt: null
extra_kwargs: {}
```

### Runner Configs

Control trial count, pass@k values, and other runner settings:

```yaml
# conf/runner/default.yaml
base_url: "http://localhost:8000"
trials_per_task: 5
k_values: [1, 3, 5]
checkpoint_dir: ./benchmark_checkpoints
enable_surrender: false
verbose: true
tool_verbosity: brief
metrics_file: null
output: null
```

Use `runner=quick` for rapid iteration, `runner=full` for final evaluation.

## Common Patterns

### Changing the Model

```bash
corral-hydra agent=react agent.model=anthropic/claude-sonnet-4-5-20250929
```

### Adjusting Temperature and Iterations

```bash
corral-hydra agent=react agent.temperature=0.3 agent.max_iterations=50
```

### Quick Smoke Test

```bash
corral-hydra runner=quick
```

### Running Specific Tasks

```bash
corral-hydra tasks.ids='[task_1,task_3]'
```

### Using Docker Mode

```bash
corral-hydra mode=docker docker.image=ghcr.io/lamalab-org/corral-materials:latest
```

### Custom Metrics File

```bash
corral-hydra runner.metrics_file=./my_metrics.py
```

### Custom System Prompt

```bash
corral-hydra agent.system_prompt="You are a materials science expert."
```

## Multi-Run Sweeps

Hydra's killer feature for benchmarking — run all combinations automatically:

```bash
# Sweep agents × models
corral-hydra --multirun \
  agent=react,tool_calling \
  agent.model=openai/gpt-4o,anthropic/claude-sonnet-4-5-20250929

# Sweep temperature and trial count
corral-hydra --multirun \
  agent=react \
  agent.temperature=0.3,0.7,1.0 \
  runner.trials_per_task=1,5,10
```

Each combination gets its own timestamped output directory with the full resolved config saved for reproducibility.

### Parallel Sweeps with Joblib

Speed up sweeps by running combinations in parallel:

```bash
corral-hydra --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4 \
  agent=react,tool_calling \
  agent.model=openai/gpt-4o,anthropic/claude-sonnet-4-5-20250929
```

## Execution Modes

The `mode` setting controls how the benchmark runs:

| Mode | Description |
|------|-------------|
| `local` | Agent runs directly in the current Python process against a running environment server. You must start the environment separately. |
| `docker` | Uses the Docker two-container architecture (same as `corral bench run`). |

```bash
# Local mode (default) — environment must already be running
corral-hydra mode=local runner.base_url=http://localhost:8000

# Docker mode — spins up containers automatically
corral-hydra mode=docker docker.image=ghcr.io/lamalab-org/corral-materials:latest
```

## Hydra Output Directory

Hydra automatically creates a timestamped output directory for each run:

```
outputs/
└── 2026-02-21/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml       # resolved config
        │   ├── hydra.yaml        # hydra internal config
        │   └── overrides.yaml    # CLI overrides used
        └── ...                   # benchmark results
```

This is automatic and makes every run fully reproducible — you can always see exactly what config was used.

## Programmatic Usage

You can also use the structured configs directly in Python without the Hydra CLI:

```python
from corral.conf.config import CorralConfig, AgentConfig, RunnerConfig
from corral.run import CorralRunner

cfg = CorralConfig(
    agent=AgentConfig(
        _target_="corral.agents.react.ReActAgent",
        model="openai/gpt-4o",
        max_iterations=30,
        temperature=0.5,
    ),
    runner=RunnerConfig(
        trials_per_task=3,
        k_values=[1, 3],
        enable_surrender=True,
    ),
)

runner = CorralRunner.from_config(cfg)
result = runner.bench(
    trials_per_task=cfg.runner.trials_per_task,
    k_values=cfg.runner.k_values,
    verbose=cfg.runner.verbose,
)
```

## Creating Custom Config Groups

You can add your own presets by creating YAML files in the config groups.

For example, to add a custom agent preset:

```yaml
# Save as conf/agent/my_agent.yaml (or pass --config-dir)
_target_: corral.agents.react.ReActAgent
model: openai/gpt-4o-mini
max_iterations: 10
temperature: 0.3
api_endpoint: null
system_prompt: "You are a chemistry expert. Be precise and concise."
user_prompt: null
extractor_prompt: null
surrender_prompt: null
extra_kwargs: {}
```

Then use it:

```bash
corral-hydra agent=my_agent
```

You can also extend configs from an external directory:

```bash
corral-hydra --config-dir=./my_configs agent=my_custom_agent
```

## Relationship to `corral bench`

The existing Typer CLI (`corral bench run`) is **not replaced** — both CLIs coexist:

| Feature | `corral bench run` | `corral-hydra` |
|---------|-------------------|----------------|
| Config file | Flat YAML/JSON (`--config`) | Composable Hydra groups |
| Overrides | CLI flags (`--agent`, `--model`) | Hydra override grammar |
| Multi-run sweeps | Not supported | `--multirun` |
| Parallel sweeps | Not supported | `hydra/launcher=joblib` |
| Auto output dirs | No | Yes (timestamped) |
| Docker mode | Default | `mode=docker` |
| Local mode | `corral bench agent` | Default |

Use `corral bench` for quick interactive runs and `corral-hydra` for systematic benchmarking campaigns.
