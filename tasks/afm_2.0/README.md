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

## Inspect The Environment Definitions

Build and list the AFM environment definitions from the supported Windows workstation after activating the correct environment:

```bat
cd tasks\afm\src
python env.py
```

The script reads these environment variables:

- `LLM_MODEL`: Used to build the per-run output directory.

## Task Layout

- Benchmark task definitions are stored under `environments/level_1/tasks_json/` and `environments/level_2/tasks_json/`.
- The current `src/env.py` entrypoint uses an internal workstation-specific configuration rather than a portable CLI for selecting level or mode.

## Notes

- AFM scans are saved as `.nid` files and are used directly for scoring and downstream image analysis.
- Available task tools include document retrieval for AFM control snippets, direct code execution against the instrument, scan optimization, grain-level rescanning, and image analysis.
- Because these tools can operate real hardware, this environment should only be used on the intended instrument workstation.

## Two-level acquisition benchmarks

The 20 environment files implement questions 1–10 at both levels. Level 1
requires one raw NID image; level 2 requires exactly three submitted images in
order. `check_acquisition_function` in `src/score.py` is registered in `src/env.py`.
Each task's `submission_format` documents the required JSON report, and its
`scoring_params.acquisition_sequence` defines the expected per-image state.

Time is expressed in seconds per line, dimensions and relative centers in nm,
and physical setpoints use explicit unit-bearing keys. The initial center is
the coordinate origin. Nominal duration excludes overhead.

The supplied questions did not include their referenced common processing
rules. These tasks explicitly adopt full-image least-squares plane subtraction
for Ra/Rq (nm), and abs((trace - aligned retrace)/2) for friction (V), with
population spatial standard deviation. No filtering or masking is applied.
Between-image summaries use sample standard deviation, CV in percent, and null
for undefined ratios. Raw height channels must be in meters and raw lateral
channels in volts. Channel names and retrace orientation are declared in the
submission and must describe the instrument's exported data.

Scoring checks distinct readable artifacts, channel dimensions, each reported
parameter state, calculations, and summaries. It does not authenticate reported
settings, calibrations, channel declarations or acquisition history. Without a
trusted instrument log it cannot detect unreported acquisitions, prove that
calculations preceded acquisition, or independently verify interpretive claims.
Legacy scoring factories remain available for existing callers.

Hardware-independent regression tests use synthetic NID-reader outputs:

```sh
uv run --no-project --with numpy python tasks/afm_2.0/tests/test_acquisition_score.py
```
