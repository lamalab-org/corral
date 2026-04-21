# AFM Task Environment

This directory contains the atomic force microscopy task environment for Corral. It provides hardware-in-the-loop AFM benchmarks where agents configure scan parameters, capture `.nid` scans from a Nanosurf instrument, optimize image quality, and analyze the resulting images.

## Important Limitation

This environment can only be run on a Windows machine that has the required AFM equipment and vendor software installed.

- The task code uses the `nanosurf` Python API and Windows COM initialization.
- The task expects access to a connected Nanosurf AFM instrument.
- The current runner and helper scripts assume a lab workstation layout and local Windows paths.

This task is therefore not expected to run on macOS, Linux, or on machines that do not have the AFM hardware and supporting software stack already installed.

## Setup

Set up this environment only on the supported Windows workstation that has the AFM software stack installed.

Typical requirements include:

- A Windows Python environment for the AFM task.
- The `nanosurf` package and its dependencies.
- The vendor-side AFM control software and any required COM integrations.
- Access to the physical AFM instrument used by the lab setup.

The repository also includes Windows helper scripts:

- `env.bat`: Opens a command shell in the AFM task environment.
- `report.bat`: Opens a command shell in the AFM reports environment.

## Run The Server

Start the AFM task server from the supported Windows workstation after activating the correct environment:

```bat
cd tasks\afm\src
python env.py
```

The server reads these environment variables:

- `CORRAL_HOST`: Bind host for the server. Defaults to `0.0.0.0`.
- `CORRAL_PORT`: Bind port for the server. Defaults to `8000`.
- `LLM_MODEL`: Used to build the per-run output directory.

## Task Layout

- Benchmark task definitions are stored under `environments/level_1` through `environments/level_4`.
- Both `tasks_json` and `subtasks_json` variants are included in the repository.
- The current `src/env.py` entrypoint uses an internal workstation-specific configuration rather than a portable CLI for selecting level or mode.

## See The Tasks

```bash
curl http://localhost:8000/tasks/
```

## Notes

- AFM scans are saved as `.nid` files and are used directly for scoring and downstream image analysis.
- Available task tools include document retrieval for AFM control snippets, direct code execution against the instrument, scan optimization, grain-level rescanning, and image analysis.
- Because these tools can operate real hardware, this environment should only be used on the intended instrument workstation.
