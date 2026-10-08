from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from corral.backend.jobs import JobManager
from corral.core.environment import Environment, Toolset
from corral.core.resources import (
    RESOURCE_CATALOG_METADATA_KEY,
    ResourceCatalogMismatchError,
    ResourceHandle,
    UnmaterializedResourceError,
    declare_directory_resource,
    declare_file_resource,
    extracted_resource_archive,
)
from corral.core.state import EnvironmentState, ExecutionState, TaskState
from corral.core.task import TaskDefinition
from corral.core.tool import WorkspaceAccess, tool
from corral.core.tool_catalog import (
    TOOL_CATALOG_METADATA_KEY,
    TOOL_POLICY_METADATA_KEY,
    ToolPolicyMismatchError,
)
from corral.core.transition import ToolExecutionResult
from corral.runtime.tool_execution import PreparedToolCall, ToolExecutor
from corral.workspace import AbsoluteWorkspaceFilesystem


def _task(*names: str) -> TaskDefinition:
    return TaskDefinition(
        name="policy",
        description="policy",
        tools=list(names),
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )


def _state(environment: Environment, *, values: dict[str, Any] | None = None):
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="policy-test",
        branch_id="main",
        task=TaskState(
            environment={
                TOOL_CATALOG_METADATA_KEY: environment.tool_catalog_snapshot().model_dump(
                    mode="json"
                ),
                TOOL_POLICY_METADATA_KEY: environment.tool_policy_snapshot().model_dump(
                    mode="json"
                ),
                RESOURCE_CATALOG_METADATA_KEY: {
                    name: descriptor.model_dump(mode="json")
                    for name, descriptor in environment.file_resources.items()
                },
            }
        ),
        environment=EnvironmentState(values=values or {"hidden_arguments": {}}),
    )


def test_workspace_access_is_private_explicit_policy():
    @tool(workspace_access="read", resources=("database",))
    def inspect(value: int) -> str:
        """Inspect one value."""
        return str(value)

    assert inspect.workspace_access is WorkspaceAccess.READ
    assert inspect.resources == ("database",)
    assert inspect.params_json_schema["properties"] == {
        "value": {"title": "Value", "type": "integer"}
    }
    provider_schema = inspect.get_openai_tool_format()
    assert "workspace_access" not in str(provider_schema)
    assert "database" not in str(provider_schema)


def test_absolute_workspace_filesystem_accepts_only_canonical_paths(tmp_path):
    filesystem = AbsoluteWorkspaceFilesystem(tmp_path)
    filesystem.write_file("/workspace/output/result.txt", "result")

    assert filesystem.read_file("/workspace/output/result.txt") == "result"
    assert filesystem.list_files(recursive=True) == ["/workspace/output/result.txt"]
    assert filesystem.file_info("/workspace/output/result.txt")["path"] == (
        "/workspace/output/result.txt"
    )
    for invalid in (
        "output/result.txt",
        "/tmp/result.txt",
        "/workspace/../result.txt",
        "/workspace/resources/database/file",
    ):
        with pytest.raises(ValueError, match="Permission denied"):
            filesystem.read_file(invalid)


def test_executor_injects_and_returns_only_public_workspace_path(tmp_path):
    @tool(
        hidden_args=["work_dir"],
        workspace_args=("work_dir",),
        workspace_access="read",
    )
    def reveal_workspace(work_dir: str) -> str:
        """Reveal the injected workspace path."""
        return work_dir

    environment = Environment(
        "policy",
        _task("reveal_workspace"),
        toolset=Toolset(
            pool={"reveal_workspace": reveal_workspace}, workspace_factory=None
        ),
        workspace_path=str(tmp_path),
    )
    state = _state(environment)

    result = ToolExecutor(environment).execute(
        state, reveal_workspace, {}, action_id="action"
    )
    started = environment.initial_event(execution_id="public-path-test")

    assert result == "/workspace"
    assert str(tmp_path) not in result
    assert str(tmp_path) not in started.model_dump_json()
    assert "/workspace" in started.task["prompt"]


def test_local_executor_materializes_public_workspace_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("local data", encoding="utf-8")

    @tool(workspace_access="read")
    def read_workspace_file(path: str) -> str:
        """Read a file through its canonical workspace path."""
        return f"{Path(path).read_text(encoding='utf-8')} at {path}"

    environment = Environment(
        "policy",
        _task("read_workspace_file"),
        base_work_dir=str(tmp_path),
        toolset=Toolset(
            pool={"read_workspace_file": read_workspace_file},
            workspace_factory=None,
        ),
        workspace_path=str(workspace),
    )

    result = ToolExecutor(environment).execute(
        _state(environment),
        read_workspace_file,
        {"path": "/workspace/input.txt"},
        action_id="action",
    )

    assert result == "local data at /workspace/input.txt"
    assert str(workspace) not in result


def test_local_background_job_materializes_public_workspace_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("background data", encoding="utf-8")

    @tool(background_capable=True, workspace_access="read")
    def background_read(path: str) -> str:
        """Read a workspace file in a background worker."""
        return f"{Path(path).read_text(encoding='utf-8')} at {path}"

    environment = Environment(
        "policy",
        _task("background_read"),
        base_work_dir=str(tmp_path),
        toolset=Toolset(
            pool={"background_read": background_read}, workspace_factory=None
        ),
        workspace_path=str(workspace),
    )
    try:
        submitted = environment.submit_job(
            _state(environment),
            "background_read",
            {"path": "/workspace/input.txt"},
            action_id="background-action",
        )
        assert environment.job_manager is not None
        result = environment.job_manager.result(
            submitted["job_id"], wait=True, timeout=2
        )
    finally:
        environment.shutdown_jobs()

    assert result["status"] == "succeeded"
    assert result["result"] == "background data at /workspace/input.txt"
    assert result["arguments"] == {"path": "/workspace/input.txt"}
    assert str(workspace) not in str(result)


def test_background_start_uses_the_same_preparation_path():
    @tool(background_capable=True, hidden_args=["secret"])
    def background_echo(value: str, secret: str) -> str:
        """Echo a public value with a configured private suffix."""
        return f"{value}:{secret}"

    environment = Environment(
        "policy",
        _task("background_echo"),
        toolset=Toolset(
            pool={"background_echo": background_echo}, workspace_factory=None
        ),
    )
    state = _state(
        environment,
        values={"hidden_arguments": {"secret": "configured"}},
    )
    try:
        handle = json.loads(
            ToolExecutor(environment).execute(
                state,
                environment.tools["start_background_echo"],
                {"value": "public"},
                action_id="background-action",
            )
        )
        assert environment.job_manager is not None
        result = environment.job_manager.result(handle["job_id"], wait=True, timeout=2)
    finally:
        environment.shutdown_jobs()

    assert result["result"] == "public:configured"
    assert result["arguments"] == {"value": "public"}
    assert result["hidden_arg_names"] == ["secret"]


def test_public_path_normalization_is_recursive_and_normalizes_keys(tmp_path):
    environment = Environment(
        "policy",
        _task(),
        toolset=Toolset(pool={}, workspace_factory=None),
        workspace_path=str(tmp_path),
    )

    normalized = environment.normalize_public_paths(
        {str(tmp_path): [f"result at {tmp_path / 'output.txt'}"]}
    )

    assert normalized == {"/workspace": ["result at /workspace/output.txt"]}


def test_changed_private_policy_prevents_resumption():
    @tool
    def calculate(value: int) -> str:
        """Calculate one value."""
        return str(value)

    environment = Environment(
        "policy",
        _task("calculate"),
        toolset=Toolset(pool={"calculate": calculate}, workspace_factory=None),
    )
    state = _state(environment)
    calculate.workspace_access = WorkspaceAccess.READ

    with pytest.raises(ToolPolicyMismatchError, match="different permissions"):
        ToolExecutor(environment).execute(
            state, calculate, {"value": 1}, action_id="action"
        )


def test_changed_worker_operation_prevents_resumption():
    @tool
    def calculate(value: int) -> str:
        """Calculate one value."""
        return str(value)

    environment = Environment(
        "policy",
        _task("calculate"),
        toolset=Toolset(pool={"calculate": calculate}, workspace_factory=None),
    )
    state = _state(environment)
    calculate.worker_operation = "terminal"

    with pytest.raises(ToolPolicyMismatchError, match="different permissions"):
        ToolExecutor(environment).execute(
            state, calculate, {"value": 1}, action_id="action"
        )


def test_stateful_resource_is_restored_and_captured_atomically():
    class CounterAdapter:
        def restore(self, state):
            return {"value": state}

        def capture(self, runtime):
            return runtime["value"]

    @tool(trusted=True, hidden_args=["counter"], resources=("counter",))
    def increment(counter: Any) -> str:
        """Increment the durable counter resource."""
        counter["value"] += 1
        return str(counter["value"])

    environment = Environment(
        "policy",
        _task("increment"),
        toolset=Toolset(pool={"increment": increment}, workspace_factory=None),
        resource_adapters={"counter": CounterAdapter()},
        resource_states={"counter": 4},
    )
    state = _state(
        environment,
        values={"hidden_arguments": {}, "resources": {"counter": 4}},
    )

    result = ToolExecutor(environment).execute(state, increment, {}, action_id="action")

    assert isinstance(result, ToolExecutionResult)
    assert result.content == "5"
    assert result.environment["resources"] == {"counter": 5}
    assert state.environment.values["resources"] == {"counter": 4}


def test_immutable_file_resource_is_content_addressed_and_path_normalized(tmp_path):
    source = tmp_path / "source" / "database.txt"
    source.parent.mkdir()
    source.write_text("immutable data", encoding="utf-8")
    descriptor = declare_file_resource(
        "database", source, runtime_version="database-v1"
    )

    @tool(hidden_args=["database"], resources=("database",))
    def read_database(database: str) -> str:
        """Read the declared immutable database."""
        return f"{Path(database).read_text(encoding='utf-8')} at {database}"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = Environment(
        "policy",
        _task("read_database"),
        base_work_dir=str(tmp_path),
        toolset=Toolset(pool={"read_database": read_database}, workspace_factory=None),
        workspace_path=str(workspace),
        file_resources={"database": descriptor},
        file_resource_sources={"database": source},
    )
    result = ToolExecutor(environment).execute(
        _state(environment), read_database, {}, action_id="action"
    )

    assert result == ("immutable data at /workspace/resources/database/database.txt")
    persisted = descriptor.model_dump(mode="json")
    assert str(source) not in str(persisted)


def test_changed_immutable_resource_prevents_resumption(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    descriptor = declare_file_resource("database", first, runtime_version="database-v1")

    @tool(resources=("database",))
    def inspect_resource() -> str:
        """Inspect a declared resource."""
        return "ok"

    environment = Environment(
        "policy",
        _task("inspect_resource"),
        base_work_dir=str(tmp_path),
        toolset=Toolset(
            pool={"inspect_resource": inspect_resource}, workspace_factory=None
        ),
        workspace_path=str(tmp_path / "workspace"),
        file_resources={"database": descriptor},
        file_resource_sources={"database": first},
    )
    state = _state(environment)
    environment.file_resources["database"] = declare_file_resource(
        "database", second, runtime_version="database-v1"
    )

    with pytest.raises(ResourceCatalogMismatchError, match="different immutable"):
        ToolExecutor(environment).execute(
            state, inspect_resource, {}, action_id="action"
        )


def test_directory_resource_archive_is_safe_and_deterministic(tmp_path):
    source = tmp_path / "tree"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "data.txt").write_text("tree data", encoding="utf-8")
    descriptor, archive = declare_directory_resource(
        "dataset",
        source,
        cache_root=tmp_path / "resource-cache",
        runtime_version="dataset-v1",
    )
    repeated, repeated_archive = declare_directory_resource(
        "dataset",
        source,
        cache_root=tmp_path / "resource-cache",
        runtime_version="dataset-v1",
    )

    assert descriptor == repeated
    assert archive == repeated_archive
    handle = ResourceHandle(
        name="dataset",
        public_path=descriptor.public_path,
        descriptor=descriptor,
        controller_path=str(archive),
    )
    assert str(archive) not in repr(handle)
    with extracted_resource_archive(handle) as extracted:
        assert (extracted / "nested" / "data.txt").read_text() == "tree data"


def test_file_resource_rejects_unmaterialized_git_lfs_pointer(tmp_path):
    pointer = tmp_path / "database.sqlite3"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        "size 123456\n",
        encoding="utf-8",
    )

    with pytest.raises(UnmaterializedResourceError, match="git lfs pull"):
        declare_file_resource("database", pointer, runtime_version="database-v1")


def test_directory_resource_rejects_unmaterialized_git_lfs_pointer(tmp_path):
    source = tmp_path / "tree"
    source.mkdir()
    (source / "database.sqlite3").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        "size 123456\n",
        encoding="utf-8",
    )

    with pytest.raises(UnmaterializedResourceError, match="git lfs pull"):
        declare_directory_resource(
            "dataset",
            source,
            cache_root=tmp_path / "resource-cache",
            runtime_version="dataset-v1",
        )


def test_local_executor_extracts_declared_directory_resource(tmp_path):
    source = tmp_path / "tree"
    source.mkdir()
    (source / "data.txt").write_text("directory data", encoding="utf-8")
    descriptor, archive = declare_directory_resource(
        "dataset",
        source,
        cache_root=tmp_path / "resource-cache",
        runtime_version="dataset-v1",
    )

    @tool(hidden_args=["dataset"], resources=("dataset",))
    def read_dataset(dataset: str) -> str:
        """Read a file from a declared directory resource."""
        with extracted_resource_archive(dataset) as extracted:
            return (extracted / "data.txt").read_text(encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = Environment(
        "policy",
        _task("read_dataset"),
        base_work_dir=str(tmp_path),
        toolset=Toolset(pool={"read_dataset": read_dataset}, workspace_factory=None),
        workspace_path=str(workspace),
        file_resources={"dataset": descriptor},
        file_resource_sources={"dataset": archive},
    )

    result = ToolExecutor(environment).execute(
        _state(environment), read_dataset, {}, action_id="action"
    )

    assert result == "directory data"


def test_initial_event_makes_file_resource_durable(tmp_path):
    source = tmp_path / "database.txt"
    source.write_text("durable data", encoding="utf-8")
    descriptor = declare_file_resource(
        "database", source, runtime_version="database-v1"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = Environment(
        "policy",
        _task(),
        base_work_dir=str(tmp_path),
        toolset=Toolset(pool={}, workspace_factory=None),
        workspace_path=str(workspace),
        file_resources={"database": descriptor},
        file_resource_sources={"database": source},
    )

    started = environment.initial_event(execution_id="durable-resource-test")
    assert environment.workspace_manager is not None
    assert asyncio.run(
        environment.workspace_manager.artifact_store.contains(descriptor.blob_ref)
    )
    assert started.environment_metadata[RESOURCE_CATALOG_METADATA_KEY] == {
        "database": descriptor.model_dump(mode="json")
    }

    source.unlink()
    handle = environment.materialize_file_resources(("database",))["database"]
    assert Path(handle.controller_path).read_text(encoding="utf-8") == "durable data"


def test_job_records_expose_only_the_public_workspace(tmp_path):
    @tool
    def background_probe() -> str:
        """Return a background result."""
        return "ok"

    manager = JobManager()
    try:
        record = manager.submit(
            PreparedToolCall.capture(background_probe, {}, workspace=str(tmp_path))
        )
        view = manager.result(record.context.job_id, wait=True, timeout=2)
        assert view["workspace"] == "/workspace"
        assert str(tmp_path) not in repr(record.context)
        assert str(tmp_path) not in str(view)
    finally:
        manager.shutdown()
