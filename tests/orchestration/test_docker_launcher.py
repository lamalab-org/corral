import json
from dataclasses import asdict

import pytest

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


@pytest.fixture()
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


@pytest.mark.anyio()
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


@pytest.mark.anyio()
async def test_docker_launcher_uses_one_hardened_container_and_host_shard(
    monkeypatch, tmp_path
):
    store = ShardedCommitStore(tmp_path / ".corral")
    execution_id = "benchmark:task-a:2"
    execution_dir = store.execution_dir(execution_id)
    commands = []

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
        agent_runtime=AgentRuntimeDefinition(name="react", model="test-model"),
        environment_runtime=EnvironmentRuntimeDefinition(name="samplemath"),
    )

    result = await launchers.DockerTaskLauncher(store).run(request)

    assert result.submission == "42"
    assert result.metadata["container_id"] == "container-id"
    assert result.metadata["recovered"] is False
    assert (
        json.loads((execution_dir / "request.json").read_text())["execution_id"]
        == execution_id
    )
    durable_metadata = json.loads((execution_dir / "metadata.json").read_text())
    assert durable_metadata["sandbox"]["container_id"] == "container-id"
    assert durable_metadata["sandbox"]["final_commit_hash"] == "a" * 64
    create = next(command for command in commands if command[1] == "create")
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


def _docker_request(execution_id: str) -> RunTaskInput:
    return RunTaskInput(
        execution_id=execution_id,
        task_id="task-a",
        environment_id="task-a",
        agent_id="agent",
        started_at="2026-01-01T00:00:00+00:00",
        benchmark_run_id="benchmark",
        trial_index=1,
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(
                image="corral:test",
                image_digest="sha256:" + "b" * 64,
            ),
        ),
        agent_runtime=AgentRuntimeDefinition(name="react", model="test-model"),
        environment_runtime=EnvironmentRuntimeDefinition(name="samplemath"),
    )


@pytest.mark.anyio()
async def test_docker_launcher_forwards_langfuse_keys_by_default(monkeypatch, tmp_path):
    store = ShardedCommitStore(tmp_path / ".corral")
    execution_id = "benchmark:task-a:1"
    execution_dir = store.execution_dir(execution_id)
    commands = []

    async def fake_command(*arguments, **_kwargs):
        commands.append(arguments)
        if arguments[1] == "create":
            return 0, "container-id"
        if arguments[1:3] == ("start", "--attach"):
            result = StateRef(
                commit_hash="a" * 64,
                execution_id=execution_id,
                branch_id="main",
                sequence=1,
                status="submitted",
                agent_steps=1,
            )
            (execution_dir / "result.json").write_text(json.dumps(asdict(result)))
        return 0, ""

    monkeypatch.setattr(launchers, "_command", fake_command)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example")

    await launchers.DockerTaskLauncher(store).run(_docker_request(execution_id))

    create = next(command for command in commands if command[1] == "create")
    forwarded = {
        create[index + 1] for index, value in enumerate(create) if value == "--env"
    }
    # Names only: Docker reads the values from the launcher's environment.
    assert {"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"} <= (
        forwarded
    )
    await store.aclose()


@pytest.mark.anyio()
async def test_container_task_traces_through_the_environment_observer(
    monkeypatch, tmp_path
):
    flushed = []

    class Observer:
        def flush(self):
            flushed.append(True)

    def stop(*_args):
        raise RuntimeError("stop after the observer is created")

    monkeypatch.setattr(internal, "observer_from_env", Observer)
    monkeypatch.setattr(internal.os, "chown", stop)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    request_path = checkpoint / "request.json"
    request_path.write_text(json.dumps(asdict(_docker_request("benchmark:task-a:1"))))

    with pytest.raises(RuntimeError, match="stop after"):
        await internal.run_task_from_files(request_path, checkpoint / "result.json")

    assert flushed == [True]
