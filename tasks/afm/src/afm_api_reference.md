# Managed AFM API — managed-afm-v1

This is the complete bundled reference, returned for every nonempty query. It is
not a search of the vendor manual. It applies to Corral's managed Code_Executor;
raw Nanosurf scan methods outside this adapter can have different blocking behavior.

## Connection and state

```python
import nanosurf
spm = nanosurf.SPM()
application = spm.application
scan = application.Scan
opmode = application.OperatingMode
zcontrol = application.ZController
print(application.Version)
print(scan.IsScanning, scan.IsPaused, scan.IsCapturing)
```

SPM selects the one operator-configured machine. Do not select other application
types, instantiate COM, change the scan head/cantilever, or reconnect after an
infrastructure error. Each Code_Executor call has a fresh Python namespace. Define
your variables again in subsequent calls. Printed output and a parameter snapshot
are returned. Snapshot geometry is in nm; API geometry is in metres.

## Configure and acquire

Set only the values requested by the task, while idle. These names are available:

| Component | Properties and units |
| --- | --- |
| `scan` | `ImageWidth`, `ImageHeight`, `CenterPosX`, `CenterPosY` in metres; `Rotation` in degrees; `Points`, `Lines` as counts; `Scantime` in seconds per line |
| `zcontrol` | `PGain`, `IGain`, `DGain`; `SetPoint`; `SetPointForceUnitMode` |
| `opmode` | `OperatingMode` |

For tapping tasks, set `opmode.OperatingMode = spm.OperatingMode.PhaseContrast`
(4) and the percent amplitude setpoint directly, e.g. `zcontrol.SetPoint = 70`.
For contact tasks, use `spm.OperatingMode.LateralForce` (9), set
`zcontrol.SetPointForceUnitMode = 0`, then set the requested voltage.
Preserve the initial center and rotation unless the task specifies otherwise.

```python
# After applying ALL settings requested by the current task:
saved_path = scan.StartFrameUp()  # or StartFrameDown()
print(saved_path)
```

Alternatively, call `acquire_frame()` or `acquire_frame("down")` in Code_Executor.
Each blocks until a fresh NID is stable, parseable and the scanner is idle. Repeated
calls are serialized, including across concurrent tool requests. A successful return
includes the absolute path; preserve it when analyzing/submitting the image.
Do not use Start, Pause, continuous scans, manual capture, approach/retract or
external control programs. Corral manages manual-scan handover and trial cleanup.
`scan.Stop()` is available to stop activity, not to signal normal frame completion.
The output directory is fixed for the trial. An optional filename prefix can be set
with `application.SetGalleryHistoryFilenameMask("scan-")`; do not supply a path.

## Analyze and submit

Call Image_Analyzer as a separate tool (not from inside Code_Executor), using the
returned absolute NID path. Enable the requested flags:

| Flag | Metric |
| --- | --- |
| `calculate_mean_roughness=True` | Ra in nm, mean absolute height deviation from mean |
| `calculate_rms_roughness=True` | Rq in nm, RMS height deviation from mean |
| `calculate_friction=True` | Signed average of f in V |
| `calculate_friction=True, friction_absolute=True` | Mean magnitude of f in V |
| `calculate_rms_friction=True` | RMS of f in V, without mean subtraction |

Height is raw Forward/Z-Axis, without leveling. Friction is
`f = (Forward/Friction force - Backward/Friction force) / 2` after conversion to V;
use the stored NSFopen array orientation, without flipping. Raw `image_data` retains
its file unit. `direction` and `channel` select that returned array only. Legacy
`dynamic_code` also affects only that array; it cannot change metric definitions.

Submit the task's exact fields and units, with absolute paths to distinct raw NIDs.
Additional acquisitions are allowed on a healthy instrument; only the submitted
acquisitions are scored. Do not retry or restart after InstrumentUnavailable.
For an ordinary script error with successful cleanup, correct the code and continue.
