# AFM benchmarks with Corral

Run agents against a Nanosurf atomic force microscope and score their saved scans. The benchmark has 20 independent tasks: ten in [level 1](environments/level_1/tasks_json/) requiring one image each, and ten in [level 2](environments/level_2/tasks_json/) requiring three images each.

From the repository root, with the workstation's AFM Python environment activated and API credentials configured, run these commands sequentially in **Windows Command Prompt (`cmd.exe`)** to execute the entire benchmark:

```bat
corral bench --agent tool-calling --environment afm --model openai/gpt-4o --env-kwargs "{\"level\": 1}" --sandbox local --trials 1 --max-parallel 1 --max-attempts 1 --output-dir .corral\afm-runs
corral bench --agent tool-calling --environment afm --model openai/gpt-4o --env-kwargs "{\"level\": 2}" --sandbox local --trials 1 --max-parallel 1 --max-attempts 1 --output-dir .corral\afm-runs
```

Keep `--sandbox local` for access to Nanosurf/COM. AFM enforces one task at a time within a process; use only one Corral process per instrument. These commands run one trial per task without automatic retries.

Change `--agent`, `--model`, or `--trials` as needed. To run a specific task, add its full ID, such as `--task afm_experiment_level_1_task_1`, with the matching level.

## Setup

### Managed acquisition

The managed acquisition path serializes access and blocks further hardware commands
after uncertain failures. Run the fault-injection tests, then validate on the Windows
AFM workstation; mock tests cannot establish that a native Nanosurf crash is cured.

The implementation covers:

- A single COM owner thread for setup, code execution, scan helpers and cleanup.
  COM initialization/uninitialization is balanced on that thread, including errors.
- A shared OS lock and persistent quarantine marker. An unresolved timeout blocks
  later hardware commands and later processes using the same state directory.
- An explicit application ProgID, unique matching process, expected head ID,
  responsive subsystems, verified initial settings and verified output directory.
- Stopping inherited running/paused scans at trial handover before any configuration.
  Normal single-frame completion does not insert an unconditional Stop.
- Bounded acquisition waiting, fresh-file detection, stable file size/mtime and NID
  parsing, plus a configurable diagnostic delay between frames.
- Separate script/readback errors, captured script output, operation-level timestamps,
  process identity/version and read-only Windows crash-event collection.
- A bundled, versioned API reference checked before hardware setup. Retrieval has
  no embedding service, network dependency, or separate API credentials.

#### Machine configuration

Set these once on the Windows instrument host, replacing **all placeholders** with
the actual application's registered ProgID, executable name and scan-head ID. Do
not select an application class by trial and error. The matching Nanosurf process
must already be running, with scripting enabled and automatic gallery saving
configured. The driver uses `GetActiveObject` to attach to an instance registered
in the COM running-object table; it never falls back to Dispatch or starts an
application. If the installed version does not register an active object, setup
fails closed. Verify this on the workstation before a benchmark run.

```bat
set "AFM_MACHINE_ID=<stable lab machine label>"
set "AFM_PROGID=<installed-application>.Application"
set "AFM_PROCESS_NAME=<installed-application>.exe"
set "AFM_EXPECTED_HEAD_ID=<actual HeadID reported by this instrument>"
set "AFM_STATE_DIR=C:\AFM\instrument-state"
set "AFM_INTERFRAME_SECONDS=1"
set "AFM_SCAN_SECONDS=900"
set "AFM_STOP_SECONDS=30"
set "AFM_FILE_SECONDS=30"
set "AFM_COMMAND_SECONDS=3600"
```

Choose a stable lab label for `AFM_MACHINE_ID`, for example `afm-jena-01` (an example,
not an inferred name for your instrument). Keep it consistent across runs and use
different labels for different instrument stations. Every operation, health result,
parameter snapshot, trial configuration event and quarantine record contains this
label and the automatically detected workstation hostname. The diagnostic collector
saves both in `machine.json`. This labels the evidence; it does not rename Windows
or replace the process/head identity checks. A hostname change does not require
renaming the lab instrument label.

`AFM_STATE_DIR` must be the **same absolute local directory for every Corral process
using this instrument**, outside all trial workspaces. It contains `instrument.lock`,
`operations.jsonl`, and, after infrastructure failure, `QUARANTINED.json`. It must
be writable by the operator. Do not place separate runs in different lock directories.
The GUI does not honor this lock; hand over the instrument to Corral and avoid manual
control during a run. Stopping an inherited scan cancels any pending capture.

The scan timeout must accommodate the longest intended scan. The command timeout
must also accommodate scripts acquiring several frames. Configure a native agent's
`tool_timeout_s` to exceed `AFM_COMMAND_SECONDS` plus transport overhead (or reduce
the controller deadline to a validated smaller bound). The Codex adapter defaults
to 600 seconds, below the conservative 3600-second controller default. Keep the
trial budget long enough for setup, acquisitions and cleanup. A Python deadline cannot
interrupt a blocked native COM call: the owner retains its lock and is quarantined.
Do not start a replacement worker while the old process is alive. There is no
automatic application restart or blind replay of an uncertain acquisition.

#### Compatibility of agent code

The `Code_Executor` signature is unchanged. Use:

```python
import nanosurf
spm = nanosurf.SPM()
scan = spm.application.Scan
scan.ImageWidth = 5e-6
scan.ImageHeight = 5e-6
for _ in range(3):
    saved_path = scan.StartFrameUp()
    print(saved_path)
```

Within this tool, `StartFrameUp()` and `StartFrameDown()` now **block until a new
image is verified**, and return its absolute path. `acquire_frame()` is another
entry point to the same operation. Existing `while scan.IsScanning` loops after
these methods are harmless. `spm.OperatingMode` enums and scalar properties on
`Scan`, `ScanHead`, `ZController` and `OperatingMode` remain available.

Application probing (`nanosurf.C3000()`, `CoreAFM()`, etc.), direct COM imports,
continuous `Start()`, Pause, prescan, manual capture and unlisted application methods
are unsupported in the managed adapter. File naming can be changed; the trial output
directory cannot. Extend the adapter deliberately if another documented API is needed.
This changes the execution contract and must be reported when comparing benchmark
results with the historical run. It is **not a security sandbox**: arbitrary Python,
generic execution tools, external terminals or GUI actions can bypass a cooperative
adapter. Use the managed tool path for instrument access; do not present this patch
as enforcing isolation against arbitrary external code.

The grain-scan integration also corrects its geometry boundary: the existing image
helper consumes micrometers, while task/runtime parameters are nanometers. Geometry
is converted explicitly in both directions and covered by a regression test.

#### Checking and recovery

From the repository root, in the existing AFM Python environment:

```bat
python tasks\afm\src\afm_admin.py check
```

This checks the configured running application without issuing scan commands.
It does not clear a quarantine. After an infrastructure failure, stop the benchmark,
inspect the instrument and application, and have the operator restore the known
application/device state. Then explicitly take control and validate recovery:

```bat
python tasks\afm\src\afm_admin.py recover --take-control
```

Recovery stops inherited scans, validates identity/subsystems, and only then removes
the quarantine marker. It refuses to run while another worker holds the OS lock.
Start a new Corral process afterwards. Do not manually delete the marker to skip
these checks. A restarted application is not a continuation of the interrupted trial.

The controller rejects subsequent trials during quarantine. The existing benchmark
runner may still produce setup-error records for queued trials; this patch does not
change the generic runner's scheduling/fatal-error policy or retry failed acquisitions.

#### Validation before the full benchmark

Run the hardware-free tests from the repository root:

```bat
python -m pytest tasks\afm\tests\test_afm_runtime.py -q
```

For a hardware smoke run, create `validated-settings.json` containing a task-style
parameter object already validated for the actual mounted sample/tip. The command
below starts real scans, stops inherited scans, creates a new output directory, and
exercises ten separate trial resets with three frames each:

```bat
python tasks\afm\src\afm_admin.py smoke --take-control --params-json validated-settings.json --output-dir C:\AFM\smoke-new --repeats 10 --frames 3
```

Validate running and paused manual-scan handover in a controlled session as well.
Run the previously failing scan settings/sequences. Compare the default one-second
interframe delay with zero only after a successful baseline; the delay is a
diagnostic mitigation, not a proven vendor requirement. Mock tests cannot establish
real COM/GUI behavior, actual scan readiness, supported vendor properties or file
flush behavior. Confirm those on the installed Nanosurf version before scaling up.

Collect matching Windows application-crash events after a failure:

```powershell
.\tasks\afm\scripts\collect_diagnostics.ps1 -MachineId $env:AFM_MACHINE_ID -ProcessName $env:AFM_PROCESS_NAME -OutputDirectory C:\AFM\diagnostics -Hours 4
```

Keep these files with `operations.jsonl`, the quarantine marker and the Corral report.
The event collector is read-only; no events does not prove there was no native crash.

Reference API contracts:
[Nanosurf Script Programmers Manual](https://www.nanosurf.com/hubfs/downloads/Script%20Programmers%20Manual.pdf),
[Microsoft COM initialization](https://learn.microsoft.com/en-us/windows/win32/api/objbase/nf-objbase-coinitialize).

Use the Windows workstation's existing Python 3.11+ AFM environment and compatible Nanosurf/COM installation. AFM dependencies are declared in [pyproject.toml](pyproject.toml). Install the current Corral framework into that same environment and configure credentials and scan storage:

```bat
cd /d C:\path\to\corral
python -m pip install -e .
set "OPENAI_API_KEY=your-api-key"
set "CORRAL_WORK_DIR=C:\AFM\workspaces"
```

Replace the checkout path and configure the selected agent provider's credentials.
The AFM reference itself needs no API key. If `corral` is unavailable on PATH, use
`python -m corral.cli`.

## Agent and scoring contract

All agents receive the task requirements, machine ID and hostname, and a brief
notice that the tools control real hardware. The full initial-state snapshot is
retained in execution records for auditing; it is not injected into the prompt.
The common prompt contains no API recipe, enum values, analysis formulas, grading
tolerances, or recovery procedure. The model still chooses and executes its solution.
Native harness shells have their own workspaces, so their generic path guidance
must respect each task's absolute AFM-path requirement.

The task tool set is `Document_Retrieval`, `Code_Executor`, and `Image_Analyzer`.
The optimizer is excluded because these tasks prescribe exact gains. The local
[API reference](src/afm_api_reference.md) provides at most two relevant sections
per query, without network credentials. It replaces the four historical retrieval
snippets, including their misleading suggestion to try several application types.
It documents adapter semantics and tool capabilities, without choosing task settings
or providing a worked workflow. Legacy optimization/grain helpers remain available
to separately defined tasks.

Task text identifies the required scientific acquisition mode (phase-contrast or
lateral-force) because the existing scorer requires that specific mode. It does not
supply SDK enum identifiers or setter code. Level 2 also discloses the existing
center/rotation requirement and clarifies that acquisition counts concern submitted
files. Numerical targets, scoring tolerances, and grading logic are unchanged.
`Image_Analyzer` exposes RMS friction alongside the existing metrics. Its
`direction`/`channel` arguments select only the returned raw array; neither they nor
legacy `dynamic_code` change metric definitions. These definitions are documented
as tool behavior, not added as a solution to every task prompt.

Level 1 scores one submitted acquisition; level 2 scores three distinct submitted
acquisitions in the requested parameter order. Additional acquisitions are allowed
while healthy, but do not retry after an infrastructure failure. The scorer does
not claim to count all scans or verify chronology from submitted file contents.
The controller's operations journal records actual scan starts and completions.

Tasks carry execution_version=managed-afm-v2 so Corral rejects evaluation of
projections from a different execution contract. These execution and prompt changes
define a new benchmark condition. Do not mix
new runs with historical results. Report the contract version, machine identity,
agent/model configuration, successful trials, and infrastructure failures separately.
A quarantined or failed-setup trial is not evidence of agent incompetence. Current
aggregate report scores may still include those failures. For newly generated runs,
produce a separate summary without modifying the original report:

```bat
python tasks\afm\src\afm_report.py .corral\afm-runs\report.json > afm-summary.json
```

The helper also accepts `.json.gz`. It separates the explicit
`AFM_INFRASTRUCTURE_FAILURE:` markers in trial/tool errors, reports excluded and
unscored counts, and computes the mean over remaining scored trials. It does not
infer infrastructure failures from low scores, successful tool output or agent prose.
Historical runs lack this marker; other provider/service failures still require
separate review. Publish exclusions alongside accuracy, not just the filtered mean.

## Results and scans

Each benchmark command prints its output directory under `.corral\afm-runs`. Open `report.json` for scores and errors. Keep the complete run directory to retain execution histories and saved scan snapshots.

Scans go into a separate task workspace under `CORRAL_WORK_DIR`, whose full path appears in the agent's prompt. Full Windows and UNC paths remain supported in submissions. Follow each task's submission format and save `.nid` files inside that workspace; evaluation reads the saved snapshots even if the original files are moved or removed.
