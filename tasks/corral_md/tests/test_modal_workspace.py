from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Self

import pytest
from corral_md import modal_workspace as bridge
from corral_md import tools as md_tools
from corral_md.env import MolecularDynamicsEnvironment, _md_file_tools, _md_task_prompt

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.core.environment import Toolset
from corral.core.task import TaskDefinition
from corral.core.transition import ToolRecoveryPending
from corral.observability import NoOpObserver
from corral.persistence import SQLiteCommitStore
from corral.persistence.workspace import WorkspaceManager
from corral.runtime import TaskRuntime


def _ref(data: bytes) -> dict[str, str | int]:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


class _Upload:
    def __init__(self, volume: _Volume) -> None:
        self.volume = volume

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def put_file(self, source, remote: str) -> None:
        data = Path(source).read_bytes() if isinstance(source, str) else source.read()
        self.volume.files[remote] = data
        self.volume.uploaded.append(remote)


class _Volume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploaded: list[str] = []
        self.removed: list[str] = []

    def batch_upload(self, *, force: bool = False) -> _Upload:
        return _Upload(self)

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def remove_file(self, path: str, *, recursive: bool = False) -> None:
        self.removed.append(path)
        if recursive:
            self.files = {key: value for key, value in self.files.items() if not key.startswith(path.rstrip("/") + "/")}
        else:
            self.files.pop(path)

    def json(self, path: str) -> dict:
        return json.loads(self.files[path])

    def write_json(self, path: str, value: dict) -> None:
        self.files[path] = json.dumps(value).encode()


class _Initializer:
    def __init__(self, volume: _Volume) -> None:
        self.volume = volume
        self.calls = 0

    def remote(self, run_id: str, release: str) -> dict:
        self.calls += 1
        prefix = f"/corral/runs/{run_id}"
        state = self.volume.json(f"{prefix}/run.json") if f"{prefix}/run.json" in self.volume.files else {
            "schema": 1, "release_id": release, "head": "base", "files": {},
        }
        assert state["release_id"] == release
        working = f"{prefix}/workspace"
        self.volume.files = {
            key: value for key, value in self.volume.files.items()
            if not key.startswith(working + "/")
        }
        if state["head"] != "base":
            old = f"{prefix}/{state['head']}"
            for key, value in list(self.volume.files.items()):
                if key.startswith(old + "/"):
                    self.volume.files[key.replace(old, working, 1)] = value
        self.volume.write_json(f"{prefix}/run.json", state)
        return state


class _Call:
    def __init__(self, function: _Function, arguments: tuple) -> None:
        self.function = function
        self.arguments = arguments
        self.object_id = f"fc-{len(function.calls)}"

    def get(self) -> dict:
        if self.function.interrupt_once:
            self.function.interrupt_once = False
            raise KeyboardInterrupt
        volume = self.function.volume
        run_id, action, input_file, _local, _args, release, _head, _expected = self.arguments
        prefix = f"/corral/runs/{run_id}"
        attempt = f"{prefix}/attempts/{action}/workspace"
        working = f"{prefix}/workspace"
        for key, value in list(volume.files.items()):
            if key.startswith(working + "/"):
                volume.files[key.replace(working, attempt, 1)] = value
        if self.function.delete_file:
            volume.files.pop(f"{attempt}/{self.function.delete_file}", None)
        if self.function.fail:
            volume.write_json(f"{prefix}/actions/{action}.failure.json", {
                "error": "simulation failed", "attempt": attempt,
            })
            volume.files[f"{attempt}/diagnostic.log"] = b"failed\n"
            raise RuntimeError("simulation failed")
        if self.function.kind == "python":
            volume.files[f"{attempt}/gpu.out"] = b"GPU output\n"
        else:
            volume.files[f"{attempt}/output/{Path(input_file).with_suffix('.log').name}"] = b"LAMMPS log\n"
            volume.files[f"{attempt}/trajectory.dump"] = b"atoms\n"
        files = {
            key.removeprefix(attempt + "/"): _ref(value)
            for key, value in volume.files.items() if key.startswith(attempt + "/")
        }
        result = {
            "action_id": action, "kind": self.function.kind,
            "release_id": release, "workspace": "/results" + attempt,
            "files": files, "revision": 1,
        }
        volume.write_json(f"{prefix}/actions/{action}.json", result)
        state = volume.json(f"{prefix}/run.json")
        state.update(head=f"attempts/{action}/workspace", files=files)
        volume.write_json(f"{prefix}/run.json", state)
        if self.function.disconnect_after_commit:
            raise ConnectionError("caller disconnected after remote commit")
        return result


class _Function:
    def __init__(
        self, volume: _Volume, *, fail: bool = False,
        disconnect_after_commit: bool = False, interrupt_once: bool = False,
        kind: str = "lammps", delete_file: str | None = None,
    ) -> None:
        self.volume = volume
        self.fail = fail
        self.disconnect_after_commit = disconnect_after_commit
        self.interrupt_once = interrupt_once
        self.kind = kind
        self.delete_file = delete_file
        self.calls: list[_Call] = []

    def spawn(self, *args, execution_options=None) -> _Call:
        call = _Call(self, args)
        self.calls.append(call)
        self.volume.write_json(
            f"/corral/runs/{args[0]}/actions/{args[1]}.call.json",
            {"call_id": call.object_id},
        )
        return call


def _inputs(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "run.in").write_text("read_data structure.data\n")
    (workspace / "structure.data").write_text("atoms\n")
    return workspace


def test_persistent_modal_workspace_and_delta_sync(tmp_path: Path) -> None:
    workspace = _inputs(tmp_path)
    volume = _Volume()
    initializer = _Initializer(volume)
    function = _Function(volume)

    log, downloaded = bridge.run_lammps_in_modal(
        workspace, "run.in", action_id="action-1", release_id="release-1",
        volume=volume, remote_function=function, initializer=initializer,
    )
    assert log == workspace / "output/run.log"
    assert downloaded == 2
    assert log.read_bytes() == b"LAMMPS log\n"
    assert len(function.calls) == 1
    assert "/corral/runs/workspace/actions/action-1.json" in volume.files
    assert not volume.removed

    volume.uploaded.clear()
    (workspace / "structure.data").unlink()
    (workspace / "run.in").write_text("run 0\n")
    bridge.run_lammps_in_modal(
        workspace, "run.in", action_id="action-2", release_id="release-1",
        volume=volume, remote_function=function, initializer=initializer,
    )
    assert volume.uploaded == ["/corral/runs/workspace/workspace/run.in"]
    assert "/corral/runs/workspace/workspace/structure.data" in volume.removed
    assert len(function.calls) == 2
    assert (workspace / "structure.data").exists() is False


def test_gpu_python_uses_same_persistent_protocol(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "script.py").write_text("print('hello')\n")
    (workspace / "old.dat").write_text("delete me")
    volume = _Volume()
    function = _Function(volume, kind="python", delete_file="old.dat")
    downloaded = bridge.run_python_in_modal(
        workspace, "script.py", action_id="python-1", release_id="release-1",
        volume=volume, remote_function=function, initializer=_Initializer(volume),
    )
    assert downloaded == 1
    assert (workspace / "gpu.out").read_bytes() == b"GPU output\n"
    assert not (workspace / "old.dat").exists()
    assert volume.json("/corral/runs/workspace/actions/python-1.json")["kind"] == "python"


def test_recover_remote_completion_without_reexecuting(tmp_path: Path, monkeypatch) -> None:
    workspace = _inputs(tmp_path)
    volume = _Volume()
    initializer = _Initializer(volume)
    function = _Function(volume, disconnect_after_commit=True)
    original = bridge._sync_result
    monkeypatch.setattr(bridge, "_sync_result", lambda *_args: (_ for _ in ()).throw(OSError("sync stopped")))
    with pytest.raises(ToolRecoveryPending, match="sync stopped"):
        bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", release_id="release-1",
            volume=volume, remote_function=function, initializer=initializer,
        )
    assert not (workspace / "run.log").exists()
    monkeypatch.setattr(bridge, "_sync_result", original)
    _, downloaded = bridge.run_lammps_in_modal(
        workspace, "run.in", action_id="action-1", release_id="release-1",
        volume=volume, remote_function=function, initializer=initializer,
    )
    assert downloaded == 2
    assert len(function.calls) == 1
    assert initializer.calls == 1
    assert (workspace / "output/run.log").is_file()


def test_restart_reattaches_to_saved_modal_call(tmp_path: Path) -> None:
    workspace = _inputs(tmp_path)
    volume = _Volume()
    function = _Function(volume, interrupt_once=True)
    initializer = _Initializer(volume)
    with pytest.raises(KeyboardInterrupt):
        bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", release_id="release-1",
            volume=volume, remote_function=function, initializer=initializer,
        )
    assert len(function.calls) == 1
    _, downloaded = bridge.run_lammps_in_modal(
        workspace, "run.in", action_id="action-1", release_id="release-1",
        volume=volume, remote_function=function, initializer=initializer,
        call_factory=lambda call_id: function.calls[0],
    )
    assert downloaded == 2
    assert len(function.calls) == 1


def test_lost_remote_run_restarts_pending_action_from_local_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = _inputs(tmp_path)
    manager = WorkspaceManager(artifact_root=tmp_path / "artifacts")
    snapshot = asyncio.run(manager.snapshot(workspace))
    volume = _Volume()
    function = _Function(volume)
    initializer = _Initializer(volume)
    original = bridge._sync_result
    monkeypatch.setattr(
        bridge, "_sync_result",
        lambda *_args: (_ for _ in ()).throw(OSError("sync stopped")),
    )
    with pytest.raises(ToolRecoveryPending, match="sync stopped"):
        bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", release_id="release-1",
            volume=volume, remote_function=function, initializer=initializer,
        )
    volume.files.clear()  # Simulate loss of the remote execution directory.
    (workspace / "uncommitted.out").write_text("partial local sync")
    monkeypatch.setattr(bridge, "_sync_result", original)
    with bridge.recovery_snapshot(snapshot, manager):
        _, downloaded = bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", release_id="release-1",
            volume=volume, remote_function=function, initializer=initializer,
        )
    assert downloaded == 2
    assert len(function.calls) == 2
    assert (workspace / "output/run.log").is_file()
    assert not (workspace / "uncommitted.out").exists()


def test_failed_attempt_retains_diagnostics_without_publishing(tmp_path: Path) -> None:
    workspace = _inputs(tmp_path)
    volume = _Volume()
    function = _Function(volume, fail=True)
    with pytest.raises(RuntimeError, match="diagnostics retained"):
        bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", release_id="release-1",
            volume=volume, remote_function=function, initializer=_Initializer(volume),
        )
    assert not (workspace / "diagnostic.log").exists()
    assert not (workspace / "run.log").exists()
    assert "/corral/runs/workspace/attempts/action-1/workspace/diagnostic.log" in volume.files


def test_execution_release_pin_rejects_journal_drift(tmp_path: Path) -> None:
    workspace = _inputs(tmp_path)
    volume = _Volume()
    with bridge.pinned_release("release-1"):
        bridge.run_lammps_in_modal(
            workspace, "run.in", action_id="action-1", volume=volume,
            remote_function=_Function(volume), initializer=_Initializer(volume),
        )
    with bridge.pinned_release("release-2"):
        with pytest.raises(RuntimeError, match="pinned to a different"):
            bridge.run_lammps_in_modal(
                workspace, "run.in", action_id="action-2", volume=volume,
                remote_function=_Function(volume), initializer=_Initializer(volume),
            )


def test_modal_lammps_rejects_input_outside_workspace(tmp_path: Path) -> None:
    workspace = _inputs(tmp_path)
    outside = tmp_path / "outside.in"
    outside.write_text("run 1\n")
    with pytest.raises(ValueError, match="inside the current workspace"):
        bridge.run_lammps_in_modal(
            workspace, str(outside), release_id="release-1", volume=_Volume(),
            remote_function=_Function(_Volume()), initializer=_Initializer(_Volume()),
        )


def test_md_io_tools_are_bound_to_local_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = _md_file_tools(str(workspace))
    tools["write_file"].execute(path="/workspace/input/run.in", content="run 10\n")
    assert (workspace / "input/run.in").read_text() == "run 10\n"
    assert tools["read_file"].execute(path="/workspace/input/run.in") == "run 10\n"
    assert tools["run_lammps"].name == "run_lammps"
    assert "corral_action_id" in tools["run_lammps"].hidden_args
    assert "corral_action_id" in tools["execute_python_script"].hidden_args


def test_md_prompt_requires_absolute_workspace_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CORRAL_MD_RELEASE_ID", "release-1")
    task = TaskDefinition(
        name="md", description="md", tools=[], scoring_fn=lambda _answer: 1.0,
        submission_format={}, prompt_fn=_md_task_prompt, resolve_answer=False,
    )
    environment = MolecularDynamicsEnvironment(
        "md", task, base_work_dir=str(tmp_path), task_execution_id="execution",
        toolset=Toolset(workspace_factory=None),
    )
    started = environment.initial_event(execution_id="execution")
    prompt = started.task["prompt"]
    assert "accept only absolute POSIX paths under /workspace" in prompt
    assert str(environment.workspace_path) not in prompt
    assert started.runtime.metadata["corral_md_release_id"] == "release-1"
    assert (Path(environment.workspace_path) / "input").is_dir()
    assert (Path(environment.workspace_path) / "output").is_dir()


def test_md_domain_tool_paths_cannot_target_a_sibling_workspace(tmp_path: Path) -> None:
    task = TaskDefinition(
        name="md", description="md", tools=[], scoring_fn=lambda _answer: 1.0,
        submission_format={}, resolve_answer=False,
    )
    environment = MolecularDynamicsEnvironment(
        "md", task, base_work_dir=str(tmp_path), task_execution_id="execution",
        toolset=Toolset(workspace_factory=None),
    )
    workspace = Path(environment.workspace_path or "")
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    parsed = environment.preprocess_arguments(
        "convert_structure_to_lammps_data",
        {"structure_path": "/workspace/input.cif", "output_file": "/workspace/output.data"},
    )
    assert Path(parsed["structure_path"]).parent == workspace
    assert Path(parsed["output_file"]).parent == workspace
    for path in ("../sibling/secret.cif", str(sibling / "secret.cif")):
        with pytest.raises(ValueError, match="workspace"):
            environment.preprocess_arguments(
                "convert_structure_to_lammps_data",
                {"structure_path": path, "output_file": "output.data"},
            )


@pytest.mark.parametrize("error_type", [
    bridge.modal.exception.RemoteError,
    bridge.modal.exception.FunctionTimeoutError,
    bridge.modal.exception.InternalFailure,
    bridge.modal.exception.OutputExpiredError,
])
def test_terminal_call_restarts_same_action(tmp_path, monkeypatch, error_type):
    workspace = _inputs(tmp_path)
    volume = _Volume()
    function = _Function(volume)
    initializer = _Initializer(volume)
    get = _Call.get

    def terminated_once(call):
        if call is function.calls[0]:
            raise error_type("worker stopped")
        return get(call)

    monkeypatch.setattr(_Call, "get", terminated_once)
    kwargs = dict(action_id="action-1", release_id="release-1", volume=volume,
                  remote_function=function, initializer=initializer)
    with pytest.raises(ToolRecoveryPending, match="worker stopped"):
        bridge.run_lammps_in_modal(workspace, "run.in", **kwargs)
    journal = json.loads(bridge._journal_path(workspace).read_text())
    assert journal["actions"]["action-1"]["status"] == "retryable"
    assert not (workspace / "run.log").exists()

    bridge.run_lammps_in_modal(workspace, "run.in", **kwargs)
    assert len(function.calls) == 2
    assert [call.arguments[1] for call in function.calls] == ["action-1", "action-1"]
    assert function.calls[0].arguments == function.calls[1].arguments
    assert (workspace / "output/run.log").is_file()
    journal = json.loads(bridge._journal_path(workspace).read_text())
    assert journal["actions"]["action-1"]["superseded_call_ids"] == [function.calls[0].object_id]


@pytest.mark.parametrize("error_type", [
    ConnectionError, bridge.modal.exception.ConnectionError, bridge.modal.exception.TimeoutError,
])
def test_uncertain_call_reattaches_and_blocks_other_actions(tmp_path, monkeypatch, error_type):
    workspace = _inputs(tmp_path)
    volume = _Volume()
    function = _Function(volume)
    initializer = _Initializer(volume)
    get = _Call.get
    monkeypatch.setattr(_Call, "get", lambda _: (_ for _ in ()).throw(error_type("disconnected")))
    kwargs = dict(release_id="release-1", volume=volume,
                  remote_function=function, initializer=initializer)
    with pytest.raises(ToolRecoveryPending, match="disconnected"):
        bridge.run_lammps_in_modal(workspace, "run.in", action_id="action-1", **kwargs)
    with pytest.raises(ToolRecoveryPending, match="Resume Modal action action-1"):
        bridge.run_lammps_in_modal(workspace, "run.in", action_id="action-2", **kwargs)
    assert len(function.calls) == 1
    monkeypatch.setattr(_Call, "get", get)
    bridge.run_lammps_in_modal(
        workspace, "run.in", action_id="action-1", **kwargs,
        call_factory=lambda call_id: next(call for call in function.calls if call.object_id == call_id),
    )
    assert len(function.calls) == 1
    assert (workspace / "output/run.log").is_file()


def test_retry_dispatch_does_not_reattach_to_previous_call(tmp_path, monkeypatch):
    workspace = _inputs(tmp_path)
    volume = _Volume()
    function = _Function(volume)
    kwargs = dict(action_id="action-1", release_id="release-1", volume=volume,
                  remote_function=function, initializer=_Initializer(volume))
    get = _Call.get
    monkeypatch.setattr(_Call, "get", lambda _: (_ for _ in ()).throw(bridge.modal.exception.RemoteError("cancelled")))
    with pytest.raises(ToolRecoveryPending):
        bridge.run_lammps_in_modal(workspace, "run.in", **kwargs)
    # Dispatch fails before a new worker records its ID; the old call manifest
    # is still present. It must never be mistaken for the new dispatch.
    spawn = function.spawn
    monkeypatch.setattr(function, "spawn", lambda *_: (_ for _ in ()).throw(ConnectionError("ambiguous dispatch")))
    with pytest.raises(ToolRecoveryPending):
        bridge.run_lammps_in_modal(workspace, "run.in", **kwargs)
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    with pytest.raises(ToolRecoveryPending, match="dispatch outcome is unknown"):
        bridge.run_lammps_in_modal(
            workspace, "run.in", **kwargs,
            call_factory=lambda _: pytest.fail("must not reattach to cancelled call"),
        )
    # Once the newly dispatched worker reports its ID, recovery can attach.
    call = spawn(*function.calls[0].arguments)
    monkeypatch.setattr(_Call, "get", get)
    bridge.run_lammps_in_modal(workspace, "run.in", **kwargs, call_factory=lambda _: call)
    assert len(function.calls) == 2
    assert (workspace / "output/run.log").is_file()


@pytest.mark.parametrize("old_is_directory", [True, False])
def test_result_sync_handles_file_directory_replacements(tmp_path, old_is_directory):
    workspace = _inputs(tmp_path)
    if old_is_directory:
        (workspace / "output").mkdir()
        (workspace / "output/old.txt").write_bytes(b"old")
        name = "output"
    else:
        (workspace / "output").write_bytes(b"old")
        name = "output/new.txt"
    volume = _Volume()
    current = bridge._files(workspace)
    files = {key: value for key, value in current.items() if not key.startswith("output")}
    files[name] = _ref(b"new")
    attempt = "/corral/runs/workspace/attempts/action-1/attempt-1/workspace"
    volume.files[f"{attempt}/{name}"] = b"new"
    result = dict(action_id="action-1", attempt_id="attempt-1", files=files,
                  workspace="/results" + attempt)
    count = bridge._sync_result(volume, workspace, bridge._ROOT / "workspace", result, current)
    assert count == 1
    assert (workspace / name).read_bytes() == b"new"
    assert bridge._files(workspace) == files


@pytest.mark.parametrize("old_kind", ["directory", "empty-directory", "file"])
def test_input_sync_handles_file_directory_replacements(tmp_path, old_kind):
    workspace = _inputs(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    if old_kind == "file":
        (remote / "output").write_bytes(b"old")
        (workspace / "output").mkdir()
        (workspace / "output/new.txt").write_bytes(b"new")
        name = "output/new.txt"
    else:
        (remote / "output").mkdir()
        if old_kind == "directory":
            (remote / "output/old.txt").write_bytes(b"old")
        (workspace / "output").write_bytes(b"new")
        name = "output"
    previous = {"files": bridge._files(remote), "directories": ["output"] if old_kind != "file" else []}

    class FilesystemVolume:
        def remove_file(self, path, *, recursive=False):
            target = remote / path.removeprefix("/workspace/")
            if recursive:
                shutil.rmtree(target)
            else:
                target.unlink()

        def batch_upload(self, **_):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def put_file(self, source, path):
            target = remote / path.removeprefix("/workspace/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    bridge._sync_inputs(FilesystemVolume(), workspace, PurePosixPath("/workspace"), previous)
    assert (remote / name).read_bytes() == b"new"
    assert bridge._files(remote) == bridge._files(workspace)


@pytest.mark.parametrize("kind", ["lammps", "python"])
@pytest.mark.parametrize("interruption", ["download", "snapshot"])
def test_runtime_recovers_same_action_after_remote_success(tmp_path, monkeypatch, kind, interruption):
    monkeypatch.setenv("CORRAL_MD_RELEASE_ID", "release-1")
    task = TaskDefinition(name="md", description="md", tools=[],
                          scoring_fn=lambda _: 1.0, submission_format={}, resolve_answer=False)
    environment = MolecularDynamicsEnvironment(
        "md", task, base_work_dir=str(tmp_path), task_execution_id="execution",
        toolset=Toolset(workspace_factory=_md_file_tools),
    )
    workspace = Path(environment.workspace_path)
    (workspace / "run.in").write_text("run 0\n")
    (workspace / "script.py").write_text("print('hello')\n")
    volume = _Volume()
    function = _Function(volume, kind=kind)
    initializer = _Initializer(volume)
    bridge_function = bridge.run_lammps_in_modal if kind == "lammps" else bridge.run_python_in_modal

    def run(*args, **kwargs):
        return bridge_function(*args, **kwargs, volume=volume,
                               remote_function=function, initializer=initializer)

    monkeypatch.setattr(md_tools, "run_lammps_in_modal" if kind == "lammps" else "run_python_in_modal", run)
    if interruption == "download":
        original = bridge._sync_result
        monkeypatch.setattr(bridge, "_sync_result", lambda *_: (_ for _ in ()).throw(OSError("download interrupted")))
    else:
        original = environment.capture_workspace

        def capture(*args, **kwargs):
            if kwargs.get("created_by_action") == "action-1":
                raise OSError("snapshot interrupted")
            return original(*args, **kwargs)

        monkeypatch.setattr(environment, "capture_workspace", capture)

    action = Action(
        id="action-1", name="run_lammps" if kind == "lammps" else "execute_python_script",
        arguments={"input_file": "/workspace/run.in"} if kind == "lammps" else {"script_path": "/workspace/script.py", "use_gpu": True},
    )

    class RecoveringAgent:
        async def run_session(self, session):
            if "action-1" in session.state.actions:
                # TaskRuntime resumes pending tools before re-entering the agent.
                assert session.state.actions["action-1"].status == "completed"
            else:
                try:
                    response = await session.execute(action)
                except ToolRecoveryPending:
                    # An adapter must not issue another tool or submit while
                    # the original action is awaiting recovery.
                    with pytest.raises(ToolRecoveryPending, match="Resume pending"):
                        await session.execute(Action(id="premature", name="submit_answer", arguments={"answer": "bad"}))
                    raise
                assert response.success
            await session.execute(Action(id="submit", name="submit_answer", arguments={"answer": "done"}))
            return AgentOutcome(status="completed", answer="done")

    async def scenario():
        async with SQLiteCommitStore(tmp_path / "recovery.sqlite3", "execution") as store:
            runtime = TaskRuntime(store, NoOpObserver())
            kwargs = dict(execution_id="execution", started_at=datetime.now(timezone.utc), max_iterations=3)
            with pytest.raises(ToolRecoveryPending, match="interrupted"):
                await runtime.run(RecoveringAgent(), environment, **kwargs)
            state = await store.materialize("main")
            assert not state.is_terminal
            assert state.actions["action-1"].status == "running"
            invocation_id, = state.tool_invocations
            assert state.tool_invocations[invocation_id].status == "running"
            assert len(function.calls) == 1
            assert f"/corral/runs/{workspace.name}/actions/action-1.json" in volume.files
            if interruption == "download":
                monkeypatch.setattr(bridge, "_sync_result", original)
            else:
                monkeypatch.setattr(environment, "capture_workspace", original)
            recovered = await runtime.run(RecoveringAgent(), environment, **kwargs)
            assert recovered.actions["action-1"].status == "completed"
            assert recovered.tool_invocations[invocation_id].status == "completed"
            assert len(function.calls) == 1
            assert initializer.calls == 1
            output = "output/run.log" if kind == "lammps" else "gpu.out"
            assert output in recovered.workspace.files
            commits = [commit async for commit in store.iter_commits("main")]
            completions = [commit for commit in commits if commit.event.type == "tool.completed" and commit.event.action_id == "action-1"]
            assert len(completions) == 1

    asyncio.run(scenario())
