import json
from dataclasses import asdict
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
async def test_docker_launcher_uses_one_hardened_container_and_host_shard(
    monkeypatch, tmp_path
):
    store = ShardedCommitStore(tmp_path / ".corral")
    execution_id = "benchmark:task-a:2"
    execution_dir = store.execution_dir(execution_id)
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
        if arguments[1] == "create":
            assert execution_dir.stat().st_mode & 0o7777 == 0o700
            return 0, "container-id"
        if operation == ("start", "--attach"):
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
            (execution_dir / "result.json").write_text(json.dumps(asdict(result)))
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
            ),
        ),
        agent_runtime=AgentRuntimeDefinition(
            name="react", model="test-model", temperature=0.2, reasoning_effort="high"
        ),
        environment_runtime=EnvironmentRuntimeDefinition(name="samplemath"),
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
    assert any("dst=/corral-state" in value for value in create)
    assert any(command[1:3] == ("rm", "--force") for command in commands)
    assert any(command[1:3] == ("volume", "rm") for command in commands)
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
    if fails:
        with pytest.raises(RuntimeError, match="task failed"):
            await internal.run_task_from_files(request_path, result_path)
    else:
        assert await internal.run_task_from_files(request_path, result_path) == 0
        assert (
            json.loads(result_path.read_text())["execution_id"] == result.execution_id
        )
    observer.flush.assert_called_once_with()
    registry.close.assert_called_once_with()
