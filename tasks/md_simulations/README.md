# MD Simulations Environment

This package provides an environment about molecular dynamics (MD) simulations using for the `corral` framework.

## Overview

The MD Simulations environment is built on the [corral framework](https://github.com/lamalab-org/mat-agent-bench), which provides:

- Task environment management
- Tool integration and execution
- Benchmark interface for evaluation
- API-based interaction with agents

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup Environment

1. Navigate to the md_simulations directory:

```bash
cd tasks/md_simulations
```

2. Create and activate a virtual environment:

```bash
uv venv --python 3.11.0
source .venv/bin/activate
```

3. Install the package in development mode:

```bash
uv pip install -e .
```

## Running the Environment

Start the corral service to host the MD tutorial tasks:

```bash
cd tasks/md_tutorials/md_simulations
python env.py
```

The service will start on `http://localhost:8000` by default.

## Development

To extend this environment based on the MD tutorials, follow these steps:

1. Create new task classes inheriting from [`Environment`](https://github.com/lamalab-org/mat-agent-bench/blob/main/src/corral/base.py)
2. Add tools using the [`@tool`](https://github.com/lamalab-org/mat-agent-bench/blob/main/src/corral/utils.py) decorator
3. Implement scoring logic for educational objectives
4. Register tasks in the environment configuration

See the [API specifications](https://github.com/lamalab-org/mat-agent-bench/blob/main/docs/API_specs.md) for detailed guidance on tool and environment creation.
