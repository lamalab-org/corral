import asyncio
import json
import os
import shutil
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter

from corral.observability import NoOpObserver
from corral.orchestration import (
    AgentRuntimeDefinition,
    BenchmarkInput,
    DockerSandboxSpec,
    EnvironmentRuntimeDefinition,
    SandboxMode,
    SandboxProfile,
    StateRef,
    internal,
    launchers,
)
from corral.orchestration.models import RunTaskInput
from corral.persistence import ShardedCommitStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_docker_benchmark_accepts_complete_runtime_definitions():
    task_id = "task-a"

    request = BenchmarkInput(
        benchmark_run_id="benchmark",
        task_ids=(task_id,),
        trials_per_task=1,
        agent_by_task={task_id: "agent"},
        environment_by_task={task_id: "environment"},
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(image="corral:test"),
        ),
        agent_runtime_by_task={
            task_id: AgentRuntimeDefinition(name="tool_calling", model="test-model")
        },
        environment_runtime_by_task={
            task_id: EnvironmentRuntimeDefinition(name="wetlab")
        },
    )

    assert request.sandbox.mode == SandboxMode.DOCKER.value


@pytest.mark.parametrize(
    ("parameters", "expected_effort"),
    [
        ({}, None),
        ({"options": {"reasoning_effort": None}}, None),
        ({"options": {"reasoning_effort": "none"}}, "none"),
        ({"reasoning_effort": "low"}, "low"),
        (
            {"reasoning_effort": "high", "options": {"reasoning_effort": "low"}},
            "high",
        ),
    ],
)
def test_saved_reasoning_effort_reconstructs_agent(
    monkeypatch, parameters, expected_effort
):
    from corral.runtime import environment_loader

    monkeypatch.setattr(
        environment_loader,
        "load_environment_group",
        lambda *args, **kwargs: {"task-a": Mock()},
    )
    # Older requests only carry options; newer ones may set the common field.
    request = TypeAdapter(RunTaskInput).validate_json(
        json.dumps(
            {
                "execution_id": "benchmark:task-a:0",
                "task_id": "task-a",
                "environment_id": "task-a",
                "agent_id": "agent",
                "started_at": "2026-01-01T00:00:00+00:00",
                "agent_runtime": {
                    "name": "tool-calling",
                    "model": "test-model",
                    "temperature": 0.2,
                    **parameters,
                },
                "environment_runtime": {"name": "samplemath"},
            }
        )
    )
    saved = json.loads(json.dumps(asdict(request)))
    assert saved["agent_runtime"]["reasoning_effort"] == expected_effort
    registry = internal._built_in_registry(request, workspace_manager=Mock())
    agent = registry.agent("agent")
    assert agent.temperature == 0.2
    assert agent.kwargs.get("reasoning_effort") == expected_effort
    if not parameters:
        assert "reasoning_effort" not in agent.kwargs


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_legacy_runtime_cannot_disable_worker_permissions(version):
    with pytest.raises(ValueError, match="rebuild the image"):
        DockerSandboxSpec(runtime_protocol_version=version)


def test_restore_host_ownership_uses_os_chown(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = checkpoint / "result.json"
    result.write_text("{}")
    ownership_changes = []

    monkeypatch.setattr(internal.os, "geteuid", lambda: 0)
    monkeypatch.setenv("CORRAL_HOST_UID", "501")
    monkeypatch.setenv("CORRAL_HOST_GID", "20")
    monkeypatch.setattr(
        internal.os,
        "chown",
        lambda path, uid, gid: ownership_changes.append((path, uid, gid)),
    )

    internal._restore_host_ownership(checkpoint)

    assert ownership_changes == [(result, 501, 20), (checkpoint, 501, 20)]


@pytest.mark.anyio
@pytest.mark.parametrize("cached", [False, True])
async def test_preflight_builds_selected_task_and_extra_only_when_image_is_missing(
    monkeypatch, tmp_path, cached
):
    commands = []
    digest = "sha256:" + "a" * 64

    async def command(*arguments, **_kwargs):
        commands.append(arguments)
        if len(commands) == 1 and not cached:
            return 1, "image missing"
        return 0, digest

    monkeypatch.setattr(launchers, "_command", command)
    dockerfile = tmp_path / "wetlab.Dockerfile"
    result = await launchers.DockerTaskLauncher.preflight(
        DockerSandboxSpec(image="corral-wetlab:claude"),
        build_context=tmp_path,
        dockerfile=dockerfile,
        build_args={"CORRAL_EXTRAS": "claude", "CORRAL_TASK": "wetlab"},
    )
    assert result.image_digest == digest
    if cached:
        assert len(commands) == 1
    else:
        assert commands[1] == (
            "docker",
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            "corral-wetlab:claude",
            "--build-arg",
            "CORRAL_EXTRAS=claude",
            "--build-arg",
            "CORRAL_TASK=wetlab",
            str(tmp_path),
        )
        assert commands[-1][1:3] == ("image", "inspect")


@pytest.mark.anyio
@pytest.mark.parametrize("private_data", [False, True])
async def test_docker_launcher_uses_private_volume_and_exports_host_checkpoint(
    monkeypatch, tmp_path, private_data
):
    store = ShardedCommitStore(tmp_path / ".corral")
    private_root = tmp_path / "private-bank"
    private_root.mkdir()
    execution_id = "benchmark:task-a:2"
    execution_dir = store.execution_dir(execution_id)
    container_state = tmp_path / "container-state"
    container_state.mkdir()
    commands = []
    langfuse_settings = {
        "CORRAL_LANGFUSE_ENABLED": "true",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test-public",
        "LANGFUSE_SECRET_KEY": "sk-lf-test-secret",
        "LANGFUSE_BASE_URL": "https://langfuse.example.test",
    }
    for name, value in langfuse_settings.items():
        monkeypatch.setenv(name, value)

    async def fake_command(*arguments, **_kwargs):
        commands.append(arguments)
        operation = arguments[1:3]
        if arguments[1] == "inspect":
            if any(command[1] == "create" for command in commands):
                return 0, json.dumps([{"Id": "container-id", "State": {"ExitCode": 0}}])
            return 1, "No such container"
        if arguments[1] == "create":
            assert execution_dir.stat().st_mode & 0o7777 == 0o700
            return 0, "container-id"
        if arguments[1] == "cp":
            source, destination = arguments[2:]
            if source.startswith("container-id:"):
                shutil.copytree(container_state, destination, dirs_exist_ok=True)
            else:
                shutil.copytree(source, container_state, dirs_exist_ok=True)
        if operation == ("start", "--attach"):
            assert (container_state / "request.json").is_file()
            assert not (execution_dir / "result.json").exists()
            connection = sqlite3.connect(container_state / "commits.sqlite3")
            connection.execute("CREATE TABLE checkpoint (answer INTEGER)")
            connection.execute("INSERT INTO checkpoint VALUES (42)")
            connection.commit()
            connection.close()
            # Sidecars left on the host must not survive a clean database export.
            for suffix in ("-wal", "-shm"):
                (execution_dir / f"commits.sqlite3{suffix}").touch()
            result = StateRef(
                commit_hash="a" * 64,
                execution_id=execution_id,
                branch_id="main",
                sequence=7,
                status="submitted",
                agent_steps=2,
                submission="42",
                output={"answer": "42"},
            )
            (container_state / "result.json").write_text(json.dumps(asdict(result)))
        return 0, ""

    monkeypatch.setattr(launchers, "_command", fake_command)
    request = RunTaskInput(
        execution_id=execution_id,
        task_id="task-a",
        environment_id="task-a",
        agent_id="agent",
        started_at="2026-01-01T00:00:00+00:00",
        benchmark_run_id="benchmark",
        trial_index=2,
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(
                image="corral:test",
                image_digest="sha256:" + "b" * 64,
                private_directories=(str(private_root),) if private_data else (),
            ),
        ),
        agent_runtime=AgentRuntimeDefinition(
            name="react", model="test-model", temperature=0.2, reasoning_effort="high"
        ),
        environment_runtime=EnvironmentRuntimeDefinition(
            name="samplemath",
            options={"data_root": str(private_root)} if private_data else {},
        ),
    )

    result = await launchers.DockerTaskLauncher(store).run(request)

    assert result.submission == "42"
    assert result.metadata["container_id"] == "container-id"
    assert result.metadata["recovered"] is False
    saved_request = json.loads((execution_dir / "request.json").read_text())
    assert saved_request["execution_id"] == execution_id
    assert saved_request["agent_runtime"]["temperature"] == 0.2
    assert saved_request["agent_runtime"]["reasoning_effort"] == "high"
    durable_metadata = json.loads((execution_dir / "metadata.json").read_text())
    assert durable_metadata["sandbox"]["container_id"] == "container-id"
    assert durable_metadata["sandbox"]["final_commit_hash"] == "a" * 64
    assert not (execution_dir / "checkpoint-pending.json").exists()
    for suffix in ("-wal", "-shm"):
        assert not (execution_dir / f"commits.sqlite3{suffix}").exists()
    connection = sqlite3.connect(execution_dir / "commits.sqlite3")
    assert connection.execute("SELECT answer FROM checkpoint").fetchone() == (42,)
    connection.close()
    create = next(command for command in commands if command[1] == "create")
    forwarded_env = {
        create[index + 1]
        for index, argument in enumerate(create)
        if argument == "--env"
    }
    assert langfuse_settings.keys() <= forwarded_env
    assert langfuse_settings["LANGFUSE_SECRET_KEY"] not in " ".join(create)
    assert (
        langfuse_settings["LANGFUSE_SECRET_KEY"]
        not in (execution_dir / "request.json").read_text()
    )
    assert "--read-only" in create
    assert "--init" in create
    assert create[create.index("--user") + 1] == "0:0"
    assert durable_metadata["sandbox"]["permissions_policy"] == "workspace-root-v1"
    assert "SYS_ADMIN" in create
    assert "SYS_CHROOT" in create
    security_options = {
        create[index + 1]
        for index, argument in enumerate(create)
        if argument == "--security-opt"
    }
    assert security_options == {"no-new-privileges", "apparmor=unconfined"}
    assert "--privileged" not in create
    assert create[create.index("--cap-drop") : create.index("--cap-drop") + 2] == (
        "--cap-drop",
        "ALL",
    )
    assert create[create.index("--network") : create.index("--network") + 2] == (
        "--network",
        "bridge",
    )
    state_volume = durable_metadata["sandbox"]["state_volume_name"]
    assert f"type=volume,src={state_volume},dst=/corral-state" in create
    if private_data:
        assert f"type=bind,src={private_root},dst=/corral-private/0,readonly" in create
        assert (
            saved_request["environment_runtime"]["options"]["data_root"]
            == "/corral-private/0"
        )
        assert request.environment_runtime.options["data_root"] == str(private_root)
    else:
        assert not any("type=bind" in value for value in create)
    operations = [command[1] for command in commands]
    assert operations.index("cp") < operations.index("start")
    assert operations.index("start") < operations.index("stop")
    assert operations[operations.index("stop") + 1] == "cp"
    assert state_volume in commands[-1]
    assert any(command[1:3] == ("rm", "--force") for command in commands)
    assert any(command[1:3] == ("volume", "rm") for command in commands)
    await store.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["exit", "cancel", "export"])
async def test_docker_launcher_preserves_checkpoints_after_failure(
    monkeypatch, tmp_path, failure
):
    store = ShardedCommitStore(tmp_path / ".corral")
    execution_id = "benchmark:interrupted:0"
    shard = store.execution_dir(execution_id)
    container_state = tmp_path / "container-state"
    container_state.mkdir()
    commands = []
    export_fails = failure == "export"
    starts = 0

    async def command(*arguments, **_kwargs):
        nonlocal starts
        commands.append(arguments)
        if arguments[1] == "inspect":
            return 1, "No such container"
        if arguments[1] == "create":
            return 0, "container-id"
        if arguments[1] == "cp":
            source, destination = arguments[2:]
            if source.startswith("container-id:"):
                if export_fails:
                    raise launchers.DockerInfrastructureError("copy failed")
                shutil.copytree(container_state, destination, dirs_exist_ok=True)
            else:
                if starts:
                    assert (Path(source) / "saved-checkpoint").read_text() == "saved"
                shutil.copytree(source, container_state, dirs_exist_ok=True)
        if arguments[1] == "start":
            starts += 1
            (container_state / "saved-checkpoint").write_text("saved")
            if failure == "cancel":
                raise asyncio.CancelledError
            raise launchers.DockerInfrastructureError("task exited 135")
        return 0, ""

    monkeypatch.setattr(launchers, "_command", command)
    request = RunTaskInput(
        execution_id=execution_id,
        task_id="interrupted",
        environment_id="interrupted",
        agent_id="agent",
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(
                image="corral:test",
                image_digest="sha256:" + "b" * 64,
                retention="never",
                registry_module="test:registry",
            ),
        ),
    )
    launcher = launchers.DockerTaskLauncher(store)
    error = (
        asyncio.CancelledError
        if failure == "cancel"
        else launchers.DockerInfrastructureError
    )
    with pytest.raises(error):
        await launcher.run(request)
    pending = shard / "checkpoint-pending.json"
    assert pending.exists() == export_fails
    if export_fails:
        # Only the initial stale-resource cleanup ran. The failed export's
        # checkpoint must remain available even with retention="never".
        assert sum(command[1] == "rm" for command in commands) == 1
        commands.clear()
        export_fails = False
        with pytest.raises(launchers.DockerInfrastructureError, match="exited 135"):
            await launcher.run(request)
        assert [command[1] for command in commands[:4]] == [
            "stop",
            "cp",
            "inspect",
            "rm",
        ]
    assert (shard / "saved-checkpoint").read_text() == "saved"
    assert not pending.exists()
    assert commands[-1][1:3] == ("volume", "rm")
    await store.aclose()


@pytest.fixture
def docker_lifecycle(monkeypatch, tmp_path):
    """A daemon double that retains container files and configuration on stop."""
    state = SimpleNamespace(
        commands=[],
        details=None,
        starts=0,
        creates=0,
        outcome="oom",
        export_fails=False,
        directory=tmp_path / "container-state",
    )
    state.directory.mkdir()

    async def command(*arguments, **_kwargs):
        state.commands.append(arguments)
        operation = arguments[1]
        if operation == "inspect":
            return (
                (0, json.dumps([state.details]))
                if state.details
                else (1, "Error: No such container")
            )
        if operation == "create":
            state.creates += 1
            labels = dict(
                arguments[index + 1].split("=", 1)
                for index, value in enumerate(arguments)
                if value == "--label"
            )
            environment = []
            for index, value in enumerate(arguments):
                if value == "--env":
                    item = arguments[index + 1]
                    environment.append(
                        item if "=" in item else f"{item}={os.environ[item]}"
                    )
            state.details = {
                "Id": f"container-{state.creates}",
                "Image": labels["corral.image_digest"],
                "Config": {"Labels": labels, "Env": environment},
                "State": {"Running": False, "ExitCode": 0, "OOMKilled": False},
            }
            return 0, state.details["Id"]
        if operation == "rm":
            state.details = None
        if arguments[1:3] == ("volume", "rm"):
            shutil.rmtree(state.directory)
            state.directory.mkdir()
        if operation == "cp":
            source, destination = arguments[2:]
            if source.startswith("container-"):
                if state.export_fails:
                    raise launchers.DockerInfrastructureError("export interrupted")
                shutil.copytree(state.directory, destination, dirs_exist_ok=True)
            else:
                shutil.copytree(source, state.directory, dirs_exist_ok=True)
        if operation == "start":
            state.starts += 1
            # Execute the real launch preamble without replacing this test process.
            with monkeypatch.context() as patch:
                patch.setattr(os, "execvp", lambda *_: None)
                exec(
                    launchers._DOCKER_BOOTSTRAP.replace(
                        "'/corral-state'", repr(str(state.directory))
                    )
                )
            request = json.loads((state.directory / "request.json").read_text())
            if state.starts > 1:
                assert (state.directory / "saved-checkpoint").read_text() == "saved"
            (state.directory / "saved-checkpoint").write_text("saved")
            oom = state.outcome == "oom"
            state.details["State"].update(ExitCode=137 if oom else 0, OOMKilled=oom)
            if state.outcome in ("submitted", "failed"):
                result = StateRef(
                    commit_hash="a" * 64,
                    execution_id=request["execution_id"],
                    branch_id="main",
                    sequence=1,
                    status=state.outcome,
                    agent_steps=1,
                )
                (state.directory / "result.json").write_text(json.dumps(asdict(result)))
            return (137, "Killed") if oom else (0, "")
        return 0, ""

    monkeypatch.setattr(launchers, "_command", command)
    return state


def _lifecycle_request(**docker_options):
    return RunTaskInput(
        execution_id="benchmark:resume:0",
        task_id="resume",
        environment_id="resume",
        agent_id="agent",
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(
                image_digest="sha256:" + "b" * 64,
                registry_module="test:registry",
                **docker_options,
            ),
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("interrupted_export", [False, True])
async def test_failed_container_is_reused_with_new_limits_then_removed(
    tmp_path, docker_lifecycle, interrupted_export
):
    daemon = docker_lifecycle
    daemon.export_fails = interrupted_export
    request = _lifecycle_request()
    store = ShardedCommitStore(tmp_path / "runs")
    launcher = launchers.DockerTaskLauncher(store)
    with pytest.raises(
        launchers.DockerInfrastructureError,
        match="Could not export" if interrupted_export else "memory limit",
    ):
        await launcher.run(request)
    original_id = daemon.details["Id"]
    shard = store.execution_dir(request.execution_id)
    if not interrupted_export:
        metadata = json.loads((shard / "sandbox.json").read_text())
        assert metadata["oom_killed"] is True
        assert metadata["exit_code"] == 137
    daemon.commands.clear()
    daemon.export_fails = False
    daemon.outcome = "submitted"
    request = replace(
        request,
        sandbox=replace(
            request.sandbox,
            docker=replace(
                request.sandbox.docker, memory="12g", cpus=4, pids_limit=512
            ),
        ),
    )

    result = await launcher.run(request)

    assert result.metadata["container_id"] == original_id
    assert result.metadata["reused"] is True
    assert daemon.creates == 1
    assert daemon.details is None
    assert (shard / "saved-checkpoint").read_text() == "saved"
    assert not (shard / "checkpoint-pending.json").exists()
    update = next(command for command in daemon.commands if command[1] == "update")
    assert update[update.index("--memory") + 1] == "12g"
    assert update[update.index("--memory-swap") + 1] == str(24 * 1024**3)
    operations = [command[1] for command in daemon.commands]
    assert "create" not in operations
    assert operations.index("rm") > operations.index("start")
    assert operations[-4:] == ["stop", "cp", "rm", "volume"]
    await store.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["image", "credentials", "network", "legacy"])
async def test_incompatible_container_is_replaced_from_exported_checkpoint(
    tmp_path, monkeypatch, docker_lifecycle, change
):
    monkeypatch.setenv("OPENAI_API_KEY", "original-secret")
    request = _lifecycle_request()
    store = ShardedCommitStore(tmp_path / "runs")
    launcher = launchers.DockerTaskLauncher(store)
    with pytest.raises(launchers.DockerInfrastructureError, match="memory limit"):
        await launcher.run(request)
    if change == "credentials":
        monkeypatch.setenv("OPENAI_API_KEY", "replacement-secret")
    elif change == "legacy":
        del docker_lifecycle.details["Config"]["Labels"]["corral.launcher_config"]
    else:
        options = (
            {"image_digest": "sha256:" + "c" * 64}
            if change == "image"
            else {"network": "none"}
        )
        request = replace(
            request,
            sandbox=replace(
                request.sandbox, docker=replace(request.sandbox.docker, **options)
            ),
        )
    docker_lifecycle.outcome = "submitted"

    result = await launcher.run(request)

    assert result.metadata["reused"] is False
    assert docker_lifecycle.creates == 2
    for path in store.execution_dir(request.execution_id).glob("*.json"):
        assert "original-secret" not in path.read_text()
        assert "replacement-secret" not in path.read_text()
    await store.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("conflict", ["owner", "running"])
async def test_resume_leaves_conflicting_container_untouched(
    tmp_path, docker_lifecycle, conflict
):
    store = ShardedCommitStore(tmp_path / "runs")
    launcher = launchers.DockerTaskLauncher(store)
    request = _lifecycle_request()
    with pytest.raises(launchers.DockerInfrastructureError, match="memory limit"):
        await launcher.run(request)
    if conflict == "owner":
        docker_lifecycle.details["Config"]["Labels"]["corral.execution_id"] = (
            "another-run"
        )
    else:
        docker_lifecycle.details["State"]["Running"] = True
    docker_lifecycle.commands.clear()
    with pytest.raises(
        launchers.DockerInfrastructureError, match="another execution|already running"
    ):
        await launcher.run(request)
    assert [command[1] for command in docker_lifecycle.commands] == ["inspect"]
    await store.aclose()


@pytest.mark.anyio
async def test_resume_cannot_return_stale_result_or_replay_stale_sqlite_sidecars(
    tmp_path, docker_lifecycle
):
    store = ShardedCommitStore(tmp_path / "runs")
    launcher = launchers.DockerTaskLauncher(store)
    request = _lifecycle_request(retention="always")
    docker_lifecycle.outcome = "submitted"
    await launcher.run(request)
    shard = store.execution_dir(request.execution_id)
    database = sqlite3.connect(shard / "commits.sqlite3")
    database.execute("CREATE TABLE checkpoint (answer INTEGER)")
    database.execute("INSERT INTO checkpoint VALUES (42)")
    database.commit()
    database.close()
    for suffix in ("-wal", "-shm"):
        (docker_lifecycle.directory / f"commits.sqlite3{suffix}").write_bytes(
            b"stale sidecar"
        )
    docker_lifecycle.outcome = "missing-result"
    with pytest.raises(
        launchers.DockerInfrastructureError, match="without publishing result"
    ):
        await launcher.run(request)
    assert docker_lifecycle.creates == 1
    assert not (shard / "result.json").exists()
    for suffix in ("-wal", "-shm"):
        assert not (shard / f"commits.sqlite3{suffix}").exists()
    database = sqlite3.connect(shard / "commits.sqlite3")
    assert database.execute("SELECT answer FROM checkpoint").fetchone() == (42,)
    database.close()
    await store.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("fails", [False, True])
async def test_container_uses_configured_observer_and_flushes_on_exit(
    monkeypatch, tmp_path, fails
):
    observer = Mock(spec=NoOpObserver)
    environment = SimpleNamespace(
        workspace_path="/workspace/test", tools={"terminal": None}
    )
    registry = SimpleNamespace(environment=lambda *_: environment, close=Mock())
    result = StateRef(
        commit_hash="a" * 64,
        execution_id="observed-task",
        branch_id="main",
        sequence=1,
        status="submitted",
        agent_steps=1,
    )

    class Launcher:
        def __init__(self, store, selected_registry, selected_observer, **kwargs):
            assert selected_registry is registry
            assert selected_observer is observer

        async def run(self, request, *, observation_context):
            assert observation_context.execution_id == request.execution_id
            if fails:
                raise RuntimeError("task failed")
            return result

    monkeypatch.setattr(internal.os, "chown", lambda *_: None)
    monkeypatch.setattr(internal.permissions, "configure", lambda *_: None)
    monkeypatch.setattr(internal, "_restore_host_ownership", lambda *_: None)
    monkeypatch.setattr(internal, "_registry_from_module", lambda *_: registry)
    monkeypatch.setattr(internal, "observer_from_env", lambda: observer)
    monkeypatch.setattr(internal, "LocalTaskLauncher", Launcher)
    request = RunTaskInput(
        execution_id=result.execution_id,
        task_id="task",
        environment_id="task",
        agent_id="tool-calling",
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(registry_module="test:registry"),
        ),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(asdict(request)))
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(asdict(result)))
    if fails:
        with pytest.raises(RuntimeError, match="task failed"):
            await internal.run_task_from_files(request_path, result_path)
        assert not result_path.exists()
    else:
        assert await internal.run_task_from_files(request_path, result_path) == 0
        assert (
            json.loads(result_path.read_text())["execution_id"] == result.execution_id
        )
    observer.flush.assert_called_once_with()
    registry.close.assert_called_once_with()
