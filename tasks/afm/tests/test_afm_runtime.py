"""Fault-injection tests: no Windows, Nanosurf installation, or AFM required."""

import ast
import asyncio
import json
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import afm_runtime
from afm_contract import INSTRUCTIONS
from afm_runtime import (
    Config,
    InstrumentUnavailable,
    Journal,
    Lease,
    ScriptError,
    Session,
    WindowsDriver,
    Worker,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeScan:
    def __init__(self, driver):
        self.driver = driver
        self.IsScanning = False
        self.IsPaused = False
        self.IsCapturing = False
        self.ImageWidth = self.ImageHeight = 10e-6
        self.CenterPosX = self.CenterPosY = self.Rotation = 0
        self.Points = self.Lines = 256
        self.Scantime = 0.075
        self.remaining = 0
        self.delay = 0
        self.starts = 0
        self.stop_calls = 0

    def StartFrameUp(self):
        self.starts += 1
        if self.driver.start_error:
            raise RuntimeError("RPC server failed")
        self.remaining = 3
        self.delay = self.driver.start_delay
        self.IsScanning = not self.delay

    StartFrameDown = StartFrameUp

    def Stop(self):
        self.stop_calls += 1
        if not self.driver.stuck_stop:
            self.IsScanning = False
            self.IsPaused = False
            self.remaining = 0

    def StopCapture(self):
        self.IsCapturing = False

    def ImageSize(self, width, height):
        self.ImageWidth, self.ImageHeight = width, height


class FakeDriver:
    def __init__(self, config):
        self.config = config
        self.pid = (123, 456)
        self.start_error = False
        self.start_delay = 0
        self.stuck_stop = False
        self.write_image = True
        self.valid_image = True
        self.open_threads = []
        self.close_threads = []
        self.pump_threads = []
        self.app = SimpleNamespace(
            Version="test.1",
            IsStartingUp=False,
            ScanHead=SimpleNamespace(HeadID="test-head", CantileverByGUID="tip-a"),
            OperatingMode=SimpleNamespace(OperatingMode=9),
            ZController=SimpleNamespace(
                PGain=100, IGain=50, DGain=0, SetPoint=0.1, SetPointForceUnitMode=0
            ),
            GetGalleryHistoryDirectoryPath=str(config.state_dir),
        )
        self.app.Scan = FakeScan(self)
        self.app.SetGalleryHistoryDirectoryPath = self.set_path
        self.app.SetGalleryHistoryFilenameMask = lambda mask: None

    def set_path(self, path):
        self.app.GetGalleryHistoryDirectoryPath = path

    def identity(self):
        return self.pid

    def open(self):
        self.open_threads.append(threading.get_ident())
        return self.app

    def close(self):
        self.close_threads.append(threading.get_ident())

    def pump(self):
        self.pump_threads.append(threading.get_ident())
        scan = self.app.Scan
        if scan.delay:
            scan.delay -= 1
            if scan.delay == 0:
                scan.IsScanning = True
        elif scan.remaining:
            scan.remaining -= 1
            if scan.remaining == 0:
                scan.IsScanning = False
                if self.write_image:
                    path = (
                        Path(self.app.GetGalleryHistoryDirectoryPath)
                        / f"frame-{scan.starts}.nid"
                    )
                    path.write_bytes(b"complete image")

    def validate_file(self, path):
        if not self.valid_image:
            raise ValueError("incomplete NID")
        assert path.read_bytes() == b"complete image"


@pytest.fixture
def config(tmp_path):
    return Config(
        "TestAFM.Application",
        "TestAFM.exe",
        "test-head",
        tmp_path / "state",
        machine_id="afm-test-01",
        poll_seconds=0.01,
        stable_seconds=0.02,
        stop_seconds=0.1,
        scan_seconds=0.3,
        file_seconds=0.1,
        interframe_seconds=0.02,
        command_seconds=3,
    )


@pytest.fixture
def session(config, tmp_path):
    journal = Journal(config)
    clock = FakeClock()
    driver = FakeDriver(config)
    s = Session(config, journal, driver, lambda: None, clock.time, clock.sleep)
    workspace = tmp_path / "trial"
    workspace.mkdir()
    s.begin({}, workspace, "trial-1")
    return s


def events(session):
    return [json.loads(line) for line in session.journal.path.read_text().splitlines()]


@pytest.mark.parametrize("state", ["IsScanning", "IsPaused", "IsCapturing"])
def test_initial_manual_activity_stopped_before_settings(session, state):
    setattr(session.app.Scan, state, True)
    session.begin({"mode": 2, "pgain": 150}, session.workspace, "trial-2")
    assert not any(session.scan_state().values())
    ops = [
        e["operation"]
        for e in events(session)
        if e["event"] == "operation.begin" and e.get("trial_id") == "trial-2"
    ]
    stop = "Scan.StopCapture" if state == "IsCapturing" else "Scan.Stop"
    assert ops.index(stop) < ops.index("Application.SetGalleryHistoryDirectoryPath")
    assert ops.index(stop) < ops.index("OperatingMode.OperatingMode.write")


def test_stuck_manual_scan_prevents_any_configuration(session):
    session.driver.stuck_stop = True
    session.app.Scan.IsScanning = True
    with pytest.raises(InstrumentUnavailable, match="stopped/idle"):
        session.begin({"pgain": 150}, session.workspace, "trial-stuck")
    assert session.app.ZController.PGain == 100


def test_missing_head_fails_before_writes(session):
    session.app.ScanHead = None
    with pytest.raises(InstrumentUnavailable, match="HeadID"):
        session.begin({"pgain": 150}, session.workspace, "bad-head")
    assert session.app.ZController.PGain == 100


def test_wrong_head_fails(session):
    session.app.ScanHead.HeadID = "other"
    with pytest.raises(InstrumentUnavailable, match="Wrong scan head"):
        session.health()


def test_pid_restart_is_not_silently_accepted(session):
    session.driver.pid = (999, 1000)
    with pytest.raises(InstrumentUnavailable, match="restarted"):
        session.acquire()
    assert session.app.Scan.starts == 0


def test_output_directory_drift_prevents_scan(session, tmp_path):
    session.app.GetGalleryHistoryDirectoryPath = str(tmp_path)
    with pytest.raises(InstrumentUnavailable, match="directory drift"):
        session.acquire()
    assert session.app.Scan.starts == 0


def test_readback_mismatch_fails(session):
    class ClampingControl:
        PGain = property(lambda self: 50, lambda self, value: None)

    session.app.ZController = ClampingControl()
    with pytest.raises(InstrumentUnavailable, match="wrote 100, read 50"):
        session.write_verified("ZController", "PGain", 100)


def test_begin_applies_mode_tip_and_setpoint_units(session):
    params = {
        "mode": 2,
        "tip": "tip-b",
        "pgain": 150,
        "image_width": 5000,
        "setpoint": {"unit": "V", "value": 0.2},
        "centre_x": 3000,
    }
    result = session.begin(params, session.workspace, "new")
    assert result["tip"] == "tip-b"
    assert result["mode"] == "StaticAFM"
    assert result["image_width"] == pytest.approx(5000)
    assert result["centre_x"] == pytest.approx(3000)
    assert session.app.ZController.SetPointForceUnitMode == 0
    assert result["setpoint"] == 0.2


def test_acquisition_does_not_reuse_old_image(session):
    (session.workspace / "old.nid").write_bytes(b"complete image")
    result = session.acquire()
    assert Path(result).name == "frame-1.nid"
    assert len(list(session.workspace.glob("*.nid"))) == 2


def test_delayed_scan_start_is_not_false_completion(session):
    session.driver.start_delay = 3
    result = session.acquire()
    assert Path(result).exists()
    assert session.clock() >= 0.06


@pytest.mark.parametrize("attribute", ["write_image", "valid_image"])
def test_missing_or_unparseable_new_file_fails(session, attribute):
    setattr(session.driver, attribute, False)
    (session.workspace / "old.nid").write_bytes(b"complete image")
    with pytest.raises(InstrumentUnavailable, match="deadline"):
        session.acquire()


def test_unexpected_active_scan_during_trial_is_not_overwritten(session):
    session.app.Scan.IsScanning = True
    with pytest.raises(InstrumentUnavailable, match="Unexpected active"):
        session.acquire()
    assert session.app.Scan.starts == 0


def test_three_frame_script_waits_for_three_distinct_files(session):
    result = session.execute("""import nanosurf
spm = nanosurf.SPM()
scan = spm.application.Scan
for i in range(3):
    scan.StartFrameUp()
    while scan.IsScanning:
        raise AssertionError("Managed call must finish before return")
print("three completed")
""")
    assert result["stdout"] == "three completed\n"
    assert len(result["parameters"]["AFM image files in directory"]) == 3
    completed = [e for e in events(session) if e["event"] == "image.completed"]
    assert len(completed) == 3


def test_interframe_delay_is_enforced(session):
    session.acquire()
    finished = session.clock()
    session.acquire()
    assert session.clock() - finished >= session.config.interframe_seconds + 0.03


def test_exec_functions_and_comprehensions_share_namespace(session):
    result = session.execute("x=4\ndef f(): return x\nprint([f() for _ in range(2)])")
    assert result["stdout"] == "[4, 4]\n"


@pytest.mark.parametrize(
    "code",
    [
        "import nanosurf; nanosurf.C3000()",
        "import pythoncom",
        "spm.application.Scan.Start()",
    ],
)
def test_unsupported_control_paths_return_script_error(session, code):
    with pytest.raises(ScriptError):
        session.execute(code)
    assert session.app.Scan.starts == 0
    assert session.health()["head_id"] == "test-head"


def test_execution_error_does_not_poison_healthy_instrument(config, tmp_path):
    worker = Worker(config, FakeDriver)
    try:
        worker.call("begin", {}, tmp_path, "trial")
        with pytest.raises(ScriptError, match="ZeroDivisionError"):
            worker.call("execute", "1/0")
        assert (
            worker.call("execute", "print('still healthy')")["stdout"]
            == "still healthy\n"
        )
        assert not worker.journal.marker.exists()
    finally:
        worker.close()


def test_rpc_failure_quarantines_and_blocks_following_trial(config, tmp_path):
    driver = FakeDriver(config)
    worker = Worker(config, lambda cfg: driver)
    try:
        worker.call("begin", {}, tmp_path, "first")
        driver.start_error = True
        with pytest.raises(InstrumentUnavailable, match="RPC server failed"):
            worker.call("acquire")
        with pytest.raises(InstrumentUnavailable, match="quarantined"):
            worker.call("begin", {}, tmp_path, "next")
        assert driver.app.Scan.starts == 1
        assert worker.journal.marker.is_file()
    finally:
        worker.close()
    with pytest.raises(InstrumentUnavailable, match="quarantined"):
        Worker(config, FakeDriver)


def test_watchdog_blocks_next_command_even_if_timed_out_call_returns(config, tmp_path):
    entered, release = threading.Event(), threading.Event()
    driver = FakeDriver(config)

    def blocked_open():
        entered.set()
        release.wait(2)
        return driver.app

    driver.open = blocked_open
    worker = Worker(replace(config, command_seconds=0.03), lambda cfg: driver)
    try:
        with pytest.raises(InstrumentUnavailable, match="timed out"):
            worker.call("begin", {}, tmp_path, "hung")
        assert entered.is_set()
        with pytest.raises(InstrumentUnavailable, match="quarantined"):
            worker.call("begin", {}, tmp_path, "next")
        # The lease must remain held while the native call is still live.
        with pytest.raises(InstrumentUnavailable, match="Another process"):
            Lease(config.state_dir / "instrument.lock")
    finally:
        release.set()
        worker.close()
    assert driver.app.Scan.starts == 0
    assert worker.journal.marker.exists()


def test_all_driver_work_and_close_have_one_owner_thread(config, tmp_path):
    driver = FakeDriver(config)
    worker = Worker(config, lambda cfg: driver)
    try:
        worker.call("begin", {}, tmp_path, "first")
        worker.call("acquire")
        worker.call("finish")
    finally:
        worker.close()
    assert set(driver.open_threads + driver.pump_threads + driver.close_threads) == {
        worker.thread.ident
    }
    assert worker.thread.ident != threading.get_ident()


def test_recovery_requires_separate_owner_and_rechecks_health(config):
    journal = Journal(config)
    journal.quarantine("old failure")
    worker = Worker(config, FakeDriver, recovery=True)
    try:
        assert worker.recover()["head_id"] == "test-head"
        assert not journal.marker.exists()
    finally:
        worker.close()


def test_failed_recovery_keeps_marker(config):
    Journal(config).quarantine("old failure")
    driver = FakeDriver(config)
    driver.app.ScanHead.HeadID = "wrong"
    worker = Worker(config, lambda cfg: driver, recovery=True)
    try:
        with pytest.raises(InstrumentUnavailable):
            worker.recover()
        assert worker.journal.marker.exists()
    finally:
        worker.close()


def test_windows_com_initialization_balanced_even_when_connection_fails(
    config, monkeypatch
):
    calls = []
    com = SimpleNamespace(
        COINIT_APARTMENTTHREADED=2,
        CoInitializeEx=lambda flag: calls.append("init"),
        CoUninitialize=lambda: calls.append("uninit"),
    )
    monkeypatch.setitem(sys.modules, "pythoncom", com)
    monkeypatch.setitem(
        sys.modules, "win32com", SimpleNamespace(client=SimpleNamespace())
    )
    monkeypatch.setitem(sys.modules, "win32com.client", SimpleNamespace())
    monkeypatch.setattr(afm_runtime.os, "name", "nt")
    driver = WindowsDriver(config)
    monkeypatch.setattr(
        driver, "identity", lambda: (_ for _ in ()).throw(RuntimeError("no process"))
    )
    with pytest.raises(RuntimeError, match="no process"):
        driver.open()
    driver.close()
    driver.close()
    assert calls == ["init", "uninit"]


def load_function(filename, name, namespace):
    source = Path(__file__).resolve().parents[1] / "src" / filename
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name
    )
    node.decorator_list = []
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


def test_windows_driver_only_attaches_to_registered_running_instance(
    config, monkeypatch
):
    calls = []
    app = object()
    com = SimpleNamespace(
        COINIT_APARTMENTTHREADED=2,
        CoInitializeEx=lambda flag: calls.append("init"),
        CoUninitialize=lambda: calls.append("uninit"),
    )
    client = SimpleNamespace(
        GetActiveObject=lambda progid: (calls.append(progid), app)[1]
    )
    monkeypatch.setitem(sys.modules, "pythoncom", com)
    monkeypatch.setitem(sys.modules, "win32com", SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    monkeypatch.setattr(afm_runtime.os, "name", "nt")
    driver = WindowsDriver(config)
    monkeypatch.setattr(driver, "identity", lambda: (42, 100))
    assert driver.open() is app
    driver.close()
    assert calls == ["init", config.progid, "uninit"]


def test_code_executor_routes_entire_script_to_owner():
    calls = []

    def call(*args):
        calls.append(args)
        return {"stdout": "hello", "stderr": "", "parameters": {}}

    func = load_function(
        "tools.py", "Code_Executor", {"runtime": lambda: SimpleNamespace(call=call)}
    )
    assert func("pass")["stdout"] == "hello"
    assert calls == [("execute", "pass")]


def test_trial_configuration_checks_local_reference_before_hardware():
    calls = []

    def broken(query):
        calls.append(query)
        raise FileNotFoundError("bundled API reference")

    func = load_function(
        "env.py",
        "reset_params",
        {"reference": broken, "runtime": lambda: calls.append("hardware")},
    )
    with pytest.raises(FileNotFoundError, match="bundled API reference"):
        func(SimpleNamespace())
    assert calls == ["preflight"]


def test_prompt_identifies_machine_without_providing_solution_guidance():
    snapshot = {"machine_id": "afm-test-01", "hostname": "test-pc", "pgain": 300}
    environment = SimpleNamespace(
        current_task=SimpleNamespace(
            description="Set P gain to 150 and acquire.",
            submission_format="absolute NID path",
            input_map={},
            initial_input={"params": {"pgain": 999}, "work_dir": "hidden"},
        ),
        afm_dir="/workspace/trial",
        resolve_inputs=lambda state: {},
    )
    render = load_function(
        "env.py",
        "get_task_prompt",
        {"INSTRUCTIONS": INSTRUCTIONS, "json": json, "ExecutionState": object},
    )
    state = SimpleNamespace(
        environment=SimpleNamespace(values={"afm_initial_snapshot": snapshot})
    )
    prompt = render(environment, state)
    assert "afm-test-01" in prompt
    assert "test-pc" in prompt
    assert '"pgain": 300' not in prompt
    assert "999" not in prompt
    assert "Set P gain to 150" in prompt
    assert INSTRUCTIONS in prompt
    for hint in (
        "PhaseContrast",
        "LateralForce",
        "SetPointForceUnitMode",
        "StartFrameUp",
        "acquire_frame",
        "sqrt",
        "1%",
        "relative tolerance",
        "Verified initial",
    ):
        assert hint not in prompt
    # The same renderer is also used while constructing execution.started.
    provisional = render(
        environment, SimpleNamespace(environment=SimpleNamespace(values={}))
    )
    assert "no verified state is available" in provisional


def test_grain_adapter_converts_nm_to_um_and_back():
    calls = []

    def call(method, *args):
        calls.append((method, args))
        if method == "snapshot":
            return {
                "image_width": 5000,
                "image_height": 10000,
                "centre_x": 3000,
                "centre_y": 0,
            }
        return "new.nid"

    def geometry(**kwargs):
        assert kwargs["current_size"] == (5, 10)
        assert kwargs["current_center"] == (3, 0)
        return {"width": 1, "height": 2, "center_x": 3.5, "center_y": 1}

    fn = load_function(
        "tools.py",
        "scan_grain_area",
        {
            "runtime": lambda: SimpleNamespace(call=call),
            "Path": Path,
            "image_process": lambda path: (None, None, None, None),
            "get_subscan_parameters": geometry,
        },
    )
    assert fn("old.nid", 1) == "new.nid"
    assert calls[-1] == (
        "update_and_acquire",
        (
            {
                "image_width": 1000,
                "image_height": 2000,
                "centre_x": 3500,
                "centre_y": 1000,
            },
        ),
    )


def test_bad_property_does_not_report_a_native_crash(session):
    with pytest.raises(ScriptError, match="Rejected API property"):
        session.execute("print(spm.application.Scan.NoSuchProperty)")
    assert session.execute("print('working')")["stdout"] == "working\n"


def test_agent_spawned_thread_cannot_touch_com(session):
    errors = []

    def wrong_thread():
        try:
            session.get("Scan", "ImageWidth")
        except Exception as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert errors == [
        "AFM_INFRASTRUCTURE_FAILURE: COM access attempted outside its owner thread"
    ]


def test_post_execution_failure_is_labeled_separately(session, monkeypatch):
    def broken_snapshot():
        raise RuntimeError("readback disconnected")

    monkeypatch.setattr(session, "snapshot", broken_snapshot)
    with pytest.raises(InstrumentUnavailable, match="post_execution_readback"):
        session.execute("pass")
    assert any(
        e.get("operation") == "post_execution_readback"
        and e["event"] == "operation.failed"
        for e in events(session)
    )


def test_trial_guard_holds_ownership_during_cancelled_cleanup():
    lock = threading.Lock()
    entered, release = threading.Event(), threading.Event()

    def cleanup(method):
        assert method == "finish"
        entered.set()
        release.wait(2)

    namespace = {
        "asyncio": asyncio,
        "AsyncIterator": AsyncIterator,
        "_INSTRUMENT_LOCK": lock,
        "InstrumentUnavailable": InstrumentUnavailable,
        "runtime": lambda: SimpleNamespace(call=cleanup),
        "event": lambda *a, **kw: None,
    }
    guard = asynccontextmanager(
        load_function("env.py", "task_execution_guard", namespace)
    )

    async def run():
        async with guard(SimpleNamespace()):
            pass

    async def scenario():
        task = asyncio.create_task(run())
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0.02)
        assert lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_cleanup_failure_propagates_to_trial_result():
    lock = threading.Lock()

    def cleanup(method):
        raise InstrumentUnavailable("cannot stop")

    namespace = {
        "asyncio": asyncio,
        "AsyncIterator": AsyncIterator,
        "_INSTRUMENT_LOCK": lock,
        "InstrumentUnavailable": InstrumentUnavailable,
        "runtime": lambda: SimpleNamespace(call=cleanup),
        "event": lambda *a, **kw: None,
    }
    guard = asynccontextmanager(
        load_function("env.py", "task_execution_guard", namespace)
    )

    async def scenario():
        with pytest.raises(InstrumentUnavailable, match="cannot stop"):
            async with guard(SimpleNamespace()):
                pass
        assert not lock.locked()

    asyncio.run(scenario())


def test_nested_tool_dispatch_is_rejected_without_deadlock(
    config, tmp_path, monkeypatch
):
    worker = Worker(config, FakeDriver)
    monkeypatch.setattr(afm_runtime, "_worker", worker)
    try:
        worker.call("begin", {}, tmp_path, "first")
        with pytest.raises(ScriptError, match="Do not call queued tools"):
            worker.call(
                "execute", "from afm_runtime import runtime; runtime().call('snapshot')"
            )
        assert not worker.poison.is_set()
    finally:
        worker.close()


def test_machine_id_is_required_in_environment(config, monkeypatch):
    monkeypatch.setenv("AFM_PROGID", config.progid)
    monkeypatch.setenv("AFM_PROCESS_NAME", config.process_name)
    monkeypatch.setenv("AFM_EXPECTED_HEAD_ID", config.expected_head_id)
    monkeypatch.setenv("AFM_STATE_DIR", str(config.state_dir))
    monkeypatch.delenv("AFM_MACHINE_ID", raising=False)
    with pytest.raises(InstrumentUnavailable, match="AFM_MACHINE_ID"):
        Config.from_env()
    monkeypatch.setenv("AFM_MACHINE_ID", "afm-lab-02")
    assert Config.from_env().machine_id == "afm-lab-02"


@pytest.mark.parametrize("name", ["", " ", " afm-lab-01"])
def test_machine_id_cannot_be_blank_or_accidentally_padded(config, name):
    with pytest.raises(ValueError, match="AFM_MACHINE_ID"):
        replace(config, machine_id=name)


def test_machine_and_hostname_follow_health_snapshot_logs_and_quarantine(session):
    health = session.health()
    snapshot = session.snapshot()
    assert health["machine_id"] == snapshot["machine_id"] == "afm-test-01"
    assert health["hostname"] == snapshot["hostname"] == session.journal.hostname
    assert health["hostname"]
    session.journal.quarantine("test failure")
    marker = json.loads(session.journal.marker.read_text())
    assert marker["machine_id"] == "afm-test-01"
    assert marker["hostname"] == session.journal.hostname
    for entry in events(session):
        assert entry["machine_id"] == "afm-test-01"
        assert entry["hostname"] == session.journal.hostname


@pytest.mark.parametrize(
    ("mode", "setpoint"),
    [
        (2, {"value": 0.1, "unit": "V"}),
        (9, {"value": 0.1, "unit": "V"}),
        (4, {"value": 70, "unit": "%"}),
    ],
)
def test_initial_structured_setpoint_applied_after_mode(session, mode, setpoint):
    zcontrol = session.app.ZController

    class OperatingMode:
        mode = 9

        @property
        def OperatingMode(self):
            return self.mode

        @OperatingMode.setter
        def OperatingMode(self, value):
            self.mode = value
            zcontrol.SetPoint = -1
            zcontrol.PGain = zcontrol.IGain = zcontrol.DGain = -1

    session.app.OperatingMode = OperatingMode()
    params = {"mode": mode, "setpoint": setpoint, "pgain": 80, "igain": 40, "dgain": 5}
    session.begin(params, session.workspace, "setpoint-test")
    assert zcontrol.SetPoint == setpoint["value"]
    assert (zcontrol.PGain, zcontrol.IGain, zcontrol.DGain) == (80, 40, 5)
    if setpoint["unit"] == "V":
        assert zcontrol.SetPointForceUnitMode == 0
