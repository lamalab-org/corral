# Managed AFM API — managed-afm-v2

Factual documentation for Corral's instrument adapter. Document_Retrieval returns
only sections matching the question. This is a small local reference, not the
complete vendor manual. It contains no task-specific parameter choices or workflow.

## Connection and state

`nanosurf.SPM()` returns the operator-configured connection. Its `application`
exposes `Scan`, `OperatingMode`, `ZController`, and `Version`. Alternate application
constructors and raw COM access are unavailable in this adapter.
`Scan.IsScanning`, `Scan.IsPaused`, and `Scan.IsCapturing` report activity.
Each Code_Executor call has a fresh Python namespace and returns stdout, stderr,
and an observed parameter snapshot. The snapshot uses nm for geometry.

## Scan geometry and resolution settings

`application.Scan` exposes `ImageWidth`, `ImageHeight`, `CenterPosX`, and
`CenterPosY` in metres; `Rotation` in degrees; `Points` and `Lines` as counts;
and `Scantime` in seconds per line. Properties are readable and writable while idle.

## Feedback gain and setpoint settings

`application.ZController` exposes `PGain`, `IGain`, `DGain`, `SetPoint`, and
`SetPointForceUnitMode`. The setpoint unit depends on the operating mode. Amplitude
setpoints are percentages; for voltage setpoints, `SetPointForceUnitMode` is 0.

## Operating modes

`application.OperatingMode.OperatingMode` is a readable/writable mode selector.
The connection's `OperatingMode` enumeration includes `StaticAFM`, `DynamicAFM`,
`PhaseContrast`, and `LateralForce`. This reference does not select a mode for a task.

## Frame acquisition and completion

`application.Scan.StartFrameUp()` and `StartFrameDown()` each acquire one frame.
The adapter waits for a fresh, stable, parseable NID and an idle scanner, then
returns the absolute file path. `acquire_frame(direction="up")` is an equivalent
adapter entry point accepting `"up"` or `"down"`. Instrument calls are serialized.
`Scan.Stop()` stops activity. Continuous scanning, Pause, manual capture, and
changing the output directory are unavailable through this adapter.
`application.SetGalleryHistoryFilenameMask(prefix)` accepts a filename prefix,
not a directory. InstrumentUnavailable denotes failed infrastructure; ScriptError
reports a code error after successful cleanup and a health check.

## Image analysis and metrics

Image_Analyzer accepts an absolute NID `path`. Its `calculate_mean_roughness` and
`calculate_rms_roughness` flags return Ra and Rq in nm from unlevelled Forward/Z-Axis
height. `calculate_friction` returns the signed mean of half the forward-minus-
backward lateral signal in V; `friction_absolute` selects mean magnitude instead.
`calculate_rms_friction` returns the uncentered RMS of that signal. Lateral arrays
use their stored NSFopen orientation. `direction` and `channel` select only the
returned raw `image_data`; legacy `dynamic_code` also changes only this array,
not the metric definitions. Raw data retains its file unit.
