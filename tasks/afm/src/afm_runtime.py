"""One COM owner, verified scan boundaries, and persistent instrument quarantine.

No Windows/vendor imports occur at module import time. The client adapter supports
existing Code_Executor scripts; it is deliberately not a Python security sandbox.
"""

# Driver imports remain lazy so validation works without Windows/vendor packages.
# ruff: noqa: PLC0415

from __future__ import annotations

import atexit
import builtins
import gc
import io
import json
import math
import os
import queue
import socket
import threading
import time
import traceback
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


class InstrumentUnavailable(RuntimeError):
    """Infrastructure failure: do not retry acquisition or start another trial."""

    def __str__(self):
        return f"AFM_INFRASTRUCTURE_FAILURE: {super().__str__()}"


class ScriptError(RuntimeError):
    """User code failed, but instrument cleanup and health checks succeeded."""


@dataclass(frozen=True)
class Config:
    progid: str
    process_name: str
    expected_head_id: str
    state_dir: Path
    machine_id: str
    poll_seconds: float = 0.25
    stop_seconds: float = 30.0
    scan_seconds: float = 900.0
    file_seconds: float = 30.0
    stable_seconds: float = 1.0
    interframe_seconds: float = 1.0
    command_seconds: float = 3600.0

    def __post_init__(self):
        if not self.machine_id.strip() or self.machine_id != self.machine_id.strip():
            raise ValueError(
                "AFM_MACHINE_ID must be a nonempty stable label without surrounding whitespace"
            )
        if not self.progid or self.progid.lower() == "spm.application":
            raise ValueError(
                "Configure a specific AFM_PROGID; generic SPM is ambiguous"
            )
        if not self.process_name or not self.expected_head_id:
            raise ValueError("Process name and expected head ID are required")
        if not self.state_dir.is_absolute():
            raise ValueError("AFM_STATE_DIR must be an absolute, shared local path")
        for name in (
            "poll_seconds",
            "stop_seconds",
            "scan_seconds",
            "file_seconds",
            "stable_seconds",
            "command_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.interframe_seconds) or self.interframe_seconds < 0:
            raise ValueError("interframe_seconds must be finite and nonnegative")

    @classmethod
    def from_env(cls):
        required = [
            "AFM_PROGID",
            "AFM_PROCESS_NAME",
            "AFM_EXPECTED_HEAD_ID",
            "AFM_STATE_DIR",
            "AFM_MACHINE_ID",
        ]
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise InstrumentUnavailable(
                "Missing machine configuration: " + ", ".join(missing)
            )
        options = {}
        for name in (
            "poll_seconds",
            "stop_seconds",
            "scan_seconds",
            "file_seconds",
            "stable_seconds",
            "interframe_seconds",
            "command_seconds",
        ):
            key = "AFM_" + name.upper()
            if key in os.environ:
                options[name] = float(os.environ[key])
        return cls(
            os.environ[required[0]],
            os.environ[required[1]],
            os.environ[required[2]],
            Path(os.environ[required[3]]),
            machine_id=os.environ["AFM_MACHINE_ID"],
            **options,
        )


class Journal:
    def __init__(self, config):
        config.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = config.state_dir / "operations.jsonl"
        self.marker = config.state_dir / "QUARANTINED.json"
        self.lock = threading.Lock()
        self.machine_id = config.machine_id
        self.hostname = socket.gethostname()

    def record(self, event, **fields):
        entry = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic": time.monotonic(),
            "client_pid": os.getpid(),
            "thread": threading.get_ident(),
            "event": event,
            **fields,
            "machine_id": self.machine_id,
            "hostname": self.hostname,
        }
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")

    def quarantine(self, reason):
        # Never clear this automatically. A new Corral process must also stop.
        with self.lock:
            self.marker.write_text(
                json.dumps(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "reason": reason,
                        "machine_id": self.machine_id,
                        "hostname": self.hostname,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        self.record("instrument.quarantined", reason=reason)


class Lease:
    """OS file lock held for the worker lifetime, including a hung COM call."""

    def __init__(self, path):
        self.stream = path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise InstrumentUnavailable(
                "Another process owns this AFM state directory"
            ) from exc

    def close(self):
        self.stream.close()  # OS releases the lock, including after process death.


class WindowsDriver:
    def __init__(self, config):
        self.config = config
        self.app = None
        self.initialized = False

    def identity(self):
        import psutil

        matches = [
            (process.info["pid"], process.info["create_time"])
            for process in psutil.process_iter(["name", "pid", "create_time"])
            if (process.info["name"] or "").casefold()
            == self.config.process_name.casefold()
        ]
        if len(matches) != 1:
            raise InstrumentUnavailable(
                "Expected exactly one configured Nanosurf process; "
                f"found {len(matches)}"
            )
        return matches[0]

    def open(self):
        if os.name != "nt":
            raise InstrumentUnavailable("The real AFM driver requires Windows")
        import pythoncom
        import win32com.client

        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        self.initialized = True
        before = self.identity()  # Do not auto-launch or probe other application types.
        # Attach only. Dispatch can launch another instance if the running
        # application is not registered in the COM running-object table.
        self.app = win32com.client.GetActiveObject(self.config.progid)
        if self.identity() != before:
            raise InstrumentUnavailable("Nanosurf process changed during connection")
        return self.app

    def pump(self):
        import pythoncom

        pythoncom.PumpWaitingMessages()

    def validate_file(self, path):
        from NSFopen.read import read

        image = read(str(path))
        # Parsing is stronger than file existence/unchanged size. Do not require
        # nonzero pixels: a valid flat/simulated acquisition can be all zero.
        if image.data is None or len(image.data) == 0:
            raise ValueError("No image datasets")

    def close(self):
        self.app = None
        gc.collect()
        if self.initialized:
            import pythoncom

            pythoncom.CoUninitialize()
            self.initialized = False


PARAMETERS = {
    "image_height": ("Scan", "ImageHeight", 1e-9),
    "image_width": ("Scan", "ImageWidth", 1e-9),
    "times_per_line": ("Scan", "Scantime", 1),
    "points_per_line": ("Scan", "Points", 1),
    "lines_per_frame": ("Scan", "Lines", 1),
    "rotation": ("Scan", "Rotation", 1),
    "centre_x": ("Scan", "CenterPosX", 1e-9),
    "centre_y": ("Scan", "CenterPosY", 1e-9),
    "pgain": ("ZController", "PGain", 1),
    "igain": ("ZController", "IGain", 1),
    "dgain": ("ZController", "DGain", 1),
}
MODES = {
    2: "StaticAFM",
    3: "DynamicAFM",
    4: "PhaseContrast",
    5: "ForceModulation",
    6: "SpreadingResistivity",
    7: "ContPhase",
    8: "DeltaF",
    9: "LateralForce",
}


class Session:
    """All methods and COM references remain on the worker thread."""

    def __init__(
        self, config, journal, driver, alive, clock=time.monotonic, sleep=time.sleep
    ):
        self.config, self.journal, self.driver, self.alive = (
            config,
            journal,
            driver,
            alive,
        )
        self.clock, self.sleep = clock, sleep
        self.app = None
        self.process = None
        self.owner_ident = threading.get_ident()
        self.workspace = None
        self.trial_id = None
        self.last_frame = float("-inf")

    def operation(self, label, fn):
        if threading.get_ident() != self.owner_ident:
            raise InstrumentUnavailable("COM access attempted outside its owner thread")
        self.alive()
        self.journal.record("operation.begin", operation=label, trial_id=self.trial_id)
        try:
            result = fn()
        except Exception as exc:
            detail = traceback.format_exc()
            self.journal.record(
                "operation.failed",
                operation=label,
                error=detail,
                trial_id=self.trial_id,
            )
            raise InstrumentUnavailable(f"{label}: {exc}") from exc
        self.alive()  # A watchdog may have quarantined this call while it was blocked.
        self.journal.record(
            "operation.end",
            operation=label,
            trial_id=self.trial_id,
            value=result
            if isinstance(result, str | int | float | bool | type(None))
            else None,
        )
        return result

    def user_property(self, callback):
        """Reject ordinary API mistakes without treating them as a lost instrument."""
        try:
            return callback()
        except InstrumentUnavailable as exc:
            cause = exc.__cause__
            hresult = getattr(cause, "hresult", None)
            bad_argument = isinstance(cause, AttributeError | TypeError | ValueError)
            if isinstance(hresult, int):
                bad_argument |= (hresult & 0xFFFFFFFF) in {
                    0x80020003,  # DISP_E_MEMBERNOTFOUND
                    0x80020005,  # DISP_E_TYPEMISMATCH
                    0x80020006,  # DISP_E_UNKNOWNNAME
                    0x80070057,  # E_INVALIDARG
                }
            if not bad_argument:
                raise
            self.health()
            self.idle()
            raise ValueError(f"Rejected API property request: {exc}") from exc

    def connect(self):
        if self.app is None:
            self.app = self.operation("connect", self.driver.open)
            self.process = self.operation("process.identity", self.driver.identity)
        self.health()

    def get(self, component, name):
        def read_value():
            obj = (
                self.app if component == "Application" else getattr(self.app, component)
            )
            return getattr(obj, name)

        return self.operation(f"{component}.{name}.read", read_value)

    def set(self, component, name, value):
        def write_value():
            obj = (
                self.app if component == "Application" else getattr(self.app, component)
            )
            setattr(obj, name, value)

        self.operation(f"{component}.{name}.write", write_value)

    def invoke(self, component, name, *args):
        def invoke_method():
            obj = (
                self.app if component == "Application" else getattr(self.app, component)
            )
            return getattr(obj, name)(*args)

        return self.operation(f"{component}.{name}", invoke_method)

    def health(self):
        current = self.operation("process.identity", self.driver.identity)
        if current != self.process:
            raise InstrumentUnavailable(
                "Nanosurf restarted; explicit recovery is required"
            )
        if self.get("Application", "IsStartingUp"):
            raise InstrumentUnavailable("Nanosurf is still starting")
        head = str(self.get("ScanHead", "HeadID"))
        if head != self.config.expected_head_id:
            raise InstrumentUnavailable(
                f"Wrong scan head: expected {self.config.expected_head_id}, got {head}"
            )
        values = {
            "machine_id": self.config.machine_id,
            "hostname": self.journal.hostname,
            "process": current,
            "version": str(self.get("Application", "Version")),
            "head_id": head,
            "tip": str(self.get("ScanHead", "CantileverByGUID")),
            "mode": self.get("OperatingMode", "OperatingMode"),
            "setpoint": self.get("ZController", "SetPoint"),
            **self.scan_state(),
        }
        self.journal.record("health", trial_id=self.trial_id, **values)
        return values

    def scan_state(self):
        state = {
            "scanning": bool(self.get("Scan", "IsScanning")),
            "paused": bool(self.get("Scan", "IsPaused")),
            "capturing": bool(self.get("Scan", "IsCapturing")),
        }
        self.journal.record("scan.state", trial_id=self.trial_id, **state)
        return state

    def tick(self):
        self.alive()
        self.driver.pump()
        self.sleep(self.config.poll_seconds)

    def stop(self):
        state = self.scan_state()
        self.journal.record("scan.stop_requested", trial_id=self.trial_id, **state)
        if state["scanning"] or state["paused"]:
            self.invoke("Scan", "Stop")
        if state["capturing"]:
            self.invoke("Scan", "StopCapture")
        deadline = self.clock() + self.config.stop_seconds
        while any(self.scan_state().values()):
            if self.clock() >= deadline:
                raise InstrumentUnavailable(
                    "Timed out waiting for stopped/idle instrument"
                )
            self.tick()

    def idle(self):
        state = self.scan_state()
        if any(state.values()):
            raise InstrumentUnavailable(f"Unexpected active scan/capture: {state}")

    def check_workspace(self):
        if self.workspace is None:
            raise InstrumentUnavailable("No active trial workspace")
        actual = Path(
            str(self.get("Application", "GetGalleryHistoryDirectoryPath"))
        ).resolve()
        if actual != self.workspace:
            raise InstrumentUnavailable(
                f"Output directory drift: {actual} != {self.workspace}"
            )

    def begin(self, params, workspace, trial_id):
        self.trial_id = trial_id
        self.connect()
        # Stop an inherited manual/abandoned scan BEFORE any settings/path writes.
        self.stop()
        self.workspace = Path(workspace).resolve(strict=True)
        self.invoke(
            "Application", "SetGalleryHistoryDirectoryPath", str(self.workspace)
        )
        self.check_workspace()
        if "tip" in params:
            self.set("ScanHead", "CantileverByGUID", params["tip"])
            if (
                str(self.get("ScanHead", "CantileverByGUID")).casefold()
                != str(params["tip"]).casefold()
            ):
                raise InstrumentUnavailable("Cantilever readback mismatch")
        if "mode" in params:
            self.set("OperatingMode", "OperatingMode", params["mode"])
            if self.get("OperatingMode", "OperatingMode") != params["mode"]:
                raise InstrumentUnavailable("Operating mode readback mismatch")
        for key, (component, attribute, scale) in PARAMETERS.items():
            if key in params:
                self.write_verified(component, attribute, params[key] * scale)
        if "setpoint" in params:
            setpoint = params["setpoint"]
            if setpoint["unit"] == "V" and params.get("mode") in (2, 9):
                self.write_verified("ZController", "SetPointForceUnitMode", 0)
            elif setpoint["unit"] != "%" or params.get("mode") not in (3, 4):
                raise ValueError("Expected V for static/lateral or % for dynamic/phase")
            self.write_verified("ZController", "SetPoint", setpoint["value"])
        elif "setpoint_v" in params:
            self.write_verified("ZController", "SetPointForceUnitMode", 0)
            self.write_verified("ZController", "SetPoint", params["setpoint_v"])
        elif "setpoint_p" in params:
            self.write_verified("ZController", "SetPoint", params["setpoint_p"])
        self.health()
        self.idle()
        return self.snapshot()

    def write_verified(self, component, name, expected):
        self.set(component, name, expected)
        actual = self.get(component, name)
        # Allow small vendor numeric rounding, not materially different settings.
        tolerance = max(abs(expected) * 1e-4, 1e-12)
        if not math.isclose(actual, expected, rel_tol=1e-4, abs_tol=tolerance):
            raise InstrumentUnavailable(
                f"{component}.{name}: wrote {expected}, read {actual}"
            )

    def snapshot(self):
        self.health()
        self.check_workspace()
        values = {
            key: self.get(component, attribute) / scale
            for key, (component, attribute, scale) in PARAMETERS.items()
        }
        values.update(
            setpoint=self.get("ZController", "SetPoint"),
            tip=self.get("ScanHead", "CantileverByGUID"),
        )
        mode = self.get("OperatingMode", "OperatingMode")
        values["mode"] = MODES.get(mode, f"Unknown({mode})")
        values["Current Working directory"] = str(self.workspace)
        values["machine_id"] = self.config.machine_id
        values["hostname"] = self.journal.hostname
        values["AFM image files in directory"] = sorted(
            p.name for p in self.workspace.glob("*.nid")
        )
        return values

    def acquire(self, direction="up"):
        if direction not in ("up", "down"):
            raise ValueError("direction must be up or down")
        self.health()
        self.check_workspace()
        self.idle()  # Do not silently stop an unexpected scan mid-trial.
        while self.clock() - self.last_frame < self.config.interframe_seconds:
            self.tick()
        before = set(self.workspace.glob("*.nid"))
        self.invoke("Scan", "StartFrameUp" if direction == "up" else "StartFrameDown")
        scan_deadline = self.clock() + self.config.scan_seconds
        file_deadline = None
        observed_active = False
        signatures = {}
        while True:
            self.check_workspace()
            state = self.scan_state()
            if state["paused"]:
                raise InstrumentUnavailable("Acquisition was unexpectedly paused")
            candidates = set(self.workspace.glob("*.nid")) - before
            if len(candidates) > 1:
                raise InstrumentUnavailable("Multiple new images from one acquisition")
            active = state["scanning"] or state["capturing"]
            if active:
                observed_active = True
                file_deadline = None
            elif observed_active or candidates:
                if file_deadline is None:
                    file_deadline = self.clock() + self.config.file_seconds
            # Do not mistake an initial false IsScanning read for completion.
            # A new stable, parseable file and idle are required together.
            for path in candidates:
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                previous, since = signatures.get(path, (None, self.clock()))
                if signature != previous:
                    signatures[path] = (signature, self.clock())
                    continue
                if (
                    stat.st_size
                    and self.clock() - since >= self.config.stable_seconds
                    and not any(state.values())
                ):
                    try:
                        self.driver.validate_file(path)
                    except Exception as exc:
                        self.journal.record(
                            "image.not_ready", path=str(path), error=str(exc)
                        )
                    else:
                        self.health()
                        self.last_frame = self.clock()
                        self.journal.record(
                            "image.completed",
                            trial_id=self.trial_id,
                            path=str(path),
                            bytes=stat.st_size,
                        )
                        return str(path)
            deadline = file_deadline if file_deadline is not None else scan_deadline
            if self.clock() >= deadline:
                self.stop()
                raise InstrumentUnavailable(
                    "Acquisition/output deadline exceeded; do not replay it"
                )
            self.tick()

    def finish(self):
        if self.app is not None:
            self.stop()
            self.health()
        self.workspace = None
        self.trial_id = None

    def update_and_acquire(self, params, mask=None):
        self.health()
        self.check_workspace()
        self.idle()
        for key, value in params.items():
            component, attribute, scale = PARAMETERS[key]
            self.write_verified(component, attribute, value * scale)
        if mask is not None:
            Application(self).SetGalleryHistoryFilenameMask(mask)
        return self.acquire()

    def execute(self, code):
        self.health()
        self.check_workspace()
        self.idle()
        stdout, stderr = io.StringIO(), io.StringIO()
        namespace = {"__name__": "__afm_script__", "__builtins__": dict(vars(builtins))}
        facade = NanosurfFacade(self)
        original_import = builtins.__import__

        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            if name == "nanosurf":
                return facade
            if name.startswith("nanosurf.") or name in (
                "pythoncom",
                "win32com",
                "win32com.client",
            ):
                raise ImportError(
                    "Use the managed nanosurf.SPM() connection; raw COM imports are unsupported"
                )
            return original_import(name, globals, locals, fromlist, level)

        namespace["__builtins__"]["__import__"] = controlled_import
        namespace.update(nanosurf=facade, spm=facade.SPM(), acquire_frame=self.acquire)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(compile(code, "<afm-agent>", "exec"), namespace, namespace)
        except InstrumentUnavailable:
            raise
        except Exception as exc:
            detail = traceback.format_exc()
            self.stop()
            self.health()
            self.journal.record("script.failed", trial_id=self.trial_id, error=detail)
            raise ScriptError(
                f"Script failed (instrument cleanup succeeded):\n{detail}"
                f"\nstdout:\n{stdout.getvalue()}\nstderr:\n{stderr.getvalue()}"
            ) from exc
        finally:
            namespace.clear()
            gc.collect()
        # Separately label this phase so a readback failure isn't blamed on exec.
        self.idle()
        params = self.operation("post_execution_readback", self.snapshot)
        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "parameters": params,
        }


class Component:
    def __init__(self, session, name):
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, name):
        session, component = self._session, self._name
        if name.startswith("_"):
            raise AttributeError(name)
        if component == "Scan":
            if name in ("StartFrameUp", "StartFrameDown"):
                return lambda: session.acquire(
                    "up" if name == "StartFrameUp" else "down"
                )
            if name == "Stop":
                return session.stop
            if name in (
                "Start",
                "Pause",
                "StartCapture",
                "StartPrescan",
                "StartSlopeCorrection",
            ):
                raise ValueError(
                    f"Scan.{name} is not supported in managed acquisition; use StartFrameUp/Down"
                )
            if name == "ImageSize":

                def image_size(width, height):
                    session.idle()
                    return session.invoke("Scan", "ImageSize", width, height)

                return image_size
        value = session.user_property(lambda: session.get(component, name))
        # Never return a callable or a COM object to agent code.
        if not isinstance(value, str | int | float | bool | type(None)):
            raise AttributeError(f"Unsupported managed property: {component}.{name}")
        return value

    def __setattr__(self, name, value):
        if name.startswith("_"):
            raise AttributeError(name)
        self._session.idle()
        self._session.user_property(lambda: self._session.set(self._name, name, value))


class Application:
    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        if name in ("Scan", "ScanHead", "ZController", "OperatingMode"):
            return Component(self._session, name)
        if name in ("Version", "IsStartingUp", "GetGalleryHistoryDirectoryPath"):
            return self._session.get("Application", name)
        if name == "SetGalleryHistoryFilenameMask":

            def set_mask(mask):
                if not isinstance(mask, str) or any(
                    x in mask for x in ("/", "\\", ":")
                ):
                    raise ValueError("Filename mask must not contain a path")
                self._session.idle()
                return self._session.invoke("Application", name, mask)

            return set_mask
        if name == "SetGalleryHistoryDirectoryPath":

            def set_directory(path):
                if Path(path).resolve() != self._session.workspace:
                    raise ValueError("Trial output directory is managed by Corral")
                self._session.check_workspace()

            return set_directory
        raise AttributeError(f"Unsupported managed Application member: {name}")


class NanosurfFacade:
    def __init__(self, session):
        self.session = session

    def SPM(self):
        self.session.health()
        # These documented enum values are also used by the existing AFM task schema.
        return SimpleNamespace(
            application=Application(self.session),
            OperatingMode=SimpleNamespace(**{v: k for k, v in MODES.items()}),
            is_connected=lambda: bool(self.session.health()),
            is_scripting_enabled=lambda: bool(self.session.health()),
        )

    def __getattr__(self, name):
        raise AttributeError(
            f"Application probing ({name}) is disabled; use nanosurf.SPM()"
        )


class Worker:
    def __init__(self, config, driver_factory=WindowsDriver, *, recovery=False):
        self.config = config
        self.journal = Journal(config)
        self.lease = Lease(config.state_dir / "instrument.lock")
        if self.journal.marker.exists() and not recovery:
            self.lease.close()
            raise InstrumentUnavailable(
                f"Instrument quarantined. Inspect {self.journal.marker}; run operator recovery"
            )
        self.poison = threading.Event()
        self.closed = False
        self.queue = queue.Queue()
        self.submit_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, args=(driver_factory,), name="afm-com-owner", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def check(self):
        if self.poison.is_set():
            raise InstrumentUnavailable(
                "Instrument quarantined; subsequent hardware commands are blocked"
            )

    def quarantine(self, reason):
        self.poison.set()
        self.journal.quarantine(reason)

    def _run(self, factory):
        driver = None
        session = None
        try:
            driver = factory(self.config)
            session = Session(self.config, self.journal, driver, self.check)
            while True:
                job = self.queue.get()
                if job is None:
                    break
                method, args, future = job
                try:
                    self.check()
                    result = getattr(session, method)(*args)
                    self.check()
                except BaseException as exc:
                    # Convert on owner thread: Future must not retain COM-bearing
                    # frames or exception tracebacks in a different apartment.
                    detail = traceback.format_exc()
                    script_error = isinstance(exc, ScriptError)
                    if not script_error:
                        self.quarantine(detail)
                    traceback.clear_frames(exc.__traceback__)
                    error = (
                        ScriptError(str(exc))
                        if script_error
                        else InstrumentUnavailable(detail)
                    )
                    future.set_exception(error)
                else:
                    future.set_result(result)
        finally:
            if session is not None:
                session.app = None
            if driver is not None:
                driver.close()
            self.lease.close()

    def call(self, method, *args):
        if threading.get_ident() == self.thread.ident:
            raise RuntimeError(
                "Do not call queued tools from inside Code_Executor; use acquire_frame()"
            )
        # The wait deadline includes neither competing callers nor a new command
        # on an old timed-out owner: poisoning is checked under the submit lock.
        with self.submit_lock:
            self.check()
            if self.closed:
                raise InstrumentUnavailable("AFM worker is closed")
            future = Future()
            self.queue.put((method, args, future))
            try:
                return future.result(timeout=self.config.command_seconds)
            except FutureTimeout as exc:
                self.quarantine(
                    f"Command {method} exceeded {self.config.command_seconds}s; "
                    "underlying call may still be running"
                )
                raise InstrumentUnavailable(
                    "AFM command timed out; stop the run and inspect the instrument"
                ) from exc

    def recover(self):
        # Only used by the separate operator CLI, never by an agent/tool retry.
        self.call("connect")
        self.call("stop")
        result = self.call("health")
        self.journal.marker.unlink(missing_ok=True)
        self.journal.record("operator.recovered", **result)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
        # Do not release the lease here: a hung owner still owns the instrument.
        self.thread.join(timeout=1)


_worker = None
_worker_lock = threading.Lock()


def runtime():
    global _worker  # noqa: PLW0603
    with _worker_lock:
        if _worker is None:
            _worker = Worker(Config.from_env())
        return _worker
