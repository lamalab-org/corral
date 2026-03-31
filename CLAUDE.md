# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Corral is a benchmarking framework for evaluating AI agents on science tasks (materials science, chemistry, etc.). It provides standardized environments, tools, and metrics to test agent performance across diverse scientific challenges.

## Common Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests
uv run pytest tests

# Run a single test file
uv run pytest tests/agents/test_react.py

# Run a specific test
uv run pytest tests/agents/test_react.py::test_name -v

# Lint
ruff check .

# Format
ruff format .

# Install a task environment for development
cd tasks/<task_name> && uv pip install -e .

# Run task-specific tests
cd tasks/<task_name> && CORRAL_WORK_DIR=$(mktemp -d) uv run python -m pytest tests -v --rootdir=.

# Pre-commit (includes ruff, formatting, commitizen)
pre-commit run --all-files
```

## Architecture

### Core flow: Runner → Router → Server → Environment

- **CorralRunner** (`src/corral/run.py`) — Orchestrates benchmark execution: manages trials, checkpointing (auto-save/resume), budget tracking, and W&B logging.
- **CorralRouter** (`src/corral/router/routes.py`) — HTTP client that agents use to interact with the benchmark server (list tasks, get tools, execute tools, submit answers).
- **Benchmark Server** (`src/corral/backend/server.py`) — FastAPI server created via `create_benchmark_server()` that hosts task environments.
- **Environment** (`src/corral/backend/env.py`) — Abstract base class for task environments. Each task implements this with its own tools and scoring. `TaskState` tracks trial state (messages, tool calls, score).

### Agent types (`src/corral/agents/`)

All extend `BaseAgent` (`base_agent.py`). Four strategies:
- **ReActAgent** — Reasoning + Acting loop
- **ToolCallingAgent** — Native LLM function calling
- **LLMPlanner** — Hierarchical planning
- **ReflexionAgent** — Self-reflection and learning from mistakes

Agents use LiteLLM for unified LLM API access. Agent utilities (token counting, LLM calls) are in `agents/utils.py`.

### Tool system (`src/corral/backend/tool.py`)

Tools are defined with the `@tool` decorator. Supports MCP (Model Context Protocol) conversion and Modal cloud execution (`@modal_tool`). Tool verbosity levels (FULL, BRIEF, MINIMAL, NONE) control description detail via `src/corral/router/verbosity.py`.

### Metrics and reporting (`src/corral/report/`)

Registry-based metric system. `BenchmarkResult` aggregates results; `TaskTrialResult` holds individual trial data. Default metrics include average score, success rate, and pass@k. Custom metrics register via `MetricRegistry`.

### Task environments (`tasks/`)

Each task is its own Python package with `pyproject.toml`, `env.py`, `tools.py`, `tests/`, and `config/`. Tasks include: samplemath, spectra_elucidation, corral_md (LAMMPS MD), catalyst, afm, ml, retrosynthesis, resistor_network, wetlab.

## Key Conventions

- **Commits**: Uses commitizen for conventional commits (enforced by pre-commit hook on commit-msg stage). Install hooks with: `pre-commit install --hook-type commit-msg --hook-type pre-push`
- **Imports**: `src` layout — the package is `corral` under `src/corral/`. pytest is configured with `pythonpath = "src"`.
- **Ruff**: Extensive rule set enabled (see `pyproject.toml`). Notable ignores: E501 (line length), PLR (design pylint). Print statements (T20) and unused variables (F841) are unfixable (won't be auto-removed). `__init__.py` files allow unused imports (F401). Tests ignore ANN, ARG, D, E402, PTH, S101.
- **Pre-commit excludes**: Ruff, formatting, and whitespace hooks exclude `tasks/` (except `tasks/*/src/`) and `reports/afm/`.
- **Git LFS**: Used for large files (e.g., MD environment data).
- **Environment variable**: `CORRAL_WORK_DIR` — needed by some task tests for working directory.
