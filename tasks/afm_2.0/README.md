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

The 20 tasks share four scoring factories in `src/score.py`:
`score_topography`, `score_roughness`, `score_friction`, and
`score_roughness_and_friction`. All four are registered in `src/env.py`.
Scoring runs without Nanosurf/COM or access to the live instrument.

Level 1 requires one raw NID image. Level 2 requires three distinct raw NID
files, in the order specified by the task. Keep the task's `submission_format`:
level 1 topography uses a bare absolute path; other submissions use a JSON
object containing numbered paths and the required numbered measurements.
Missing, extra, duplicate, non-finite, or incorrect measurements score zero.

Each scorer reads `HeaderDump/DataSet-Info` with NSFopen, validates the channel
headers and arrays, and compares extracted settings against `final_params`.
`final_params` means the expected recorded settings for each submitted artifact:

- Level 1: one parameter object for the single image.
- Level 2: an ordered list of three parameter objects. Entry 0 is checked against
  `path_1`, entry 1 against `path_2`, and entry 2 against `path_3`.

There is no separate `acquisition_params` or duplicate last-image target.
`initial_input.params` initializes a deliberately different reset baseline,
not the settings required for image 1. All tasks start at P/I gains 80/40,
0.125 s per line, 128×128 pixels, and a 1000×1000 nm scan. Tasks requiring
tapping reset to contact mode (2), D gain 0, and a 0.05 V setpoint. Tasks
requiring contact reset to tapping mode (4), D gain 5, and a 90% setpoint.
The reset mode and setpoint unit therefore both differ from every target.
The tip, origin, and rotation retain their configured values. Reset selects
the tip and mode before applying gains and the mode-appropriate setpoint. The agent must apply the task's requested settings before acquisition.
Every reset baseline falls outside the 1% tolerance for every acquisition
target; tests verify that images acquired at reset settings score zero.
`final_params` continues to describe only the required acquisition settings.
Settings are read from the files,
never trusted from a submitted parameter report or from current hardware state.
The unit contract is:

| Field or quantity | Task/configuration unit | Conversion from NID data |
| --- | --- | --- |
| `image_width`, `image_height`, `centre_x`, `centre_y` | nm | Convert header lengths to m, then multiply by 1e9 |
| `times_per_line` | s per line | Convert ms/us/ns to s |
| `rotation` | degrees | Read the angle in degrees |
| `points_per_line`, `lines_per_frame` | integer sample counts | No length conversion |
| `pgain`, `igain`, `dgain` | instrument gain settings | No length/voltage conversion |
| Tapping amplitude `setpoint` | % | Preserve percentage; 70 means 70%, not 0.7 |
| Contact deflection `setpoint` | V | Convert mV to V; reject mismatched physical units |
| `rms_roughness`, `mean_roughness` | nm | Convert the height channel to nm before calculation |
| `average_friction`, `rms_friction` | V | Convert lateral signal units to V; these are not forces in N |
| Roughness percentage changes | % | 100 × (value − reference) / reference; no length unit |
| `tolerance` | fraction | 0.01 means 1% |
| `mode`, `tip` | identifiers | Not physical quantities |

NSFopen applies each channel's scaling and offset but preserves its declared
`Dim2Unit`. A Z-Axis channel declaring `m` is in metres; the scorer and
`Image_Analyzer` both convert it to nm exactly once. `Image_Analyzer` now returns
roughness in nm and friction in V with a `metric_units` mapping; its raw
`image_data` retains reader units. For example, 2.4e-9 m becomes 2.4 nm.
A 5 µm scan is configured as 5000 nm and sent to the instrument as 5e-6 m.
A 70% amplitude setpoint cannot be replaced with 0.567 V without a known
amplitude reference/calibration.

Unit-bearing NID values are converted before comparison. Modes map to
SDK values (contact/static/lateral force = 2, dynamic force = 3, phase
contrast/tapping = 4). Setpoints preserve `%` versus `V`, including mV-to-V
conversion. Cantilever targets, if used, must be NID names rather than SDK GUIDs.
All 20 tasks set `tolerance: 0.01`. Continuous settings must fall within
`target ± 0.01 * abs(target)` using the corresponding `final_params` target.
There is no additional absolute allowance. Zero targets, counts, modes, and
setpoint units must match exactly.

`metrics` declares required measurements, preventing a submission from choosing
which checks to perform. Roughness uses the full forward Z-Axis image after
subtracting its mean: RMS deviation for `rms_roughness`, mean absolute deviation
for `mean_roughness`, both reported in nm. No plane fitting or filtering is
applied. Friction uses half the forward/backward `Friction force` difference,
retaining NSFopen's orientation, in V. Average friction is the signed mean;
`friction_absolute: true` selects mean magnitude where the task requests it.
RMS friction is the root mean square without mean subtraction. Channel units
are read from NID headers, and unsupported units fail scoring.

The same 1% tolerance applies to measurements relative to the value computed
from the image, including percentage-change fields. Zero measurements require
zero; undefined percentage changes still require null.
Every requested measurement and every required setting must pass to score 1;
otherwise the score is 0. Level 2 task 6 uses `percent_change_reference: 2` and
checks `100 * (value_i - value_2) / value_2` against computed measurements,
requiring JSON null when the reference is zero.

File scoring cannot prove how many unsubmitted scans occurred or whether a
calculation preceded acquisition. It validates the submitted artifacts, not
acquisition provenance. The acquisition tools still require the Windows
instrument workstation described above.

Run hardware-independent regressions (including a bundled NSFopen NID sample):

```sh
uv run --no-project --python 3.11 --with numpy --with pytest --with loguru --with NSFopen python -m pytest tasks/afm_2.0/tests -q
```
