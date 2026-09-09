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
    assert create[
        create.index("--security-opt") : create.index("--security-opt") + 2
    ] == ("--security-opt", "no-new-privileges")
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
