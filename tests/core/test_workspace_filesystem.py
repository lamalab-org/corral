import json
from pathlib import Path

import pytest

from corral.core.environment import Environment, Toolset, default_file_tools
from corral.core.resources import declare_file_resource
from corral.core.state import EnvironmentState, ExecutionState, TaskState
from corral.core.task import TaskDefinition
from corral.orchestration.registry import RuntimeRegistry
from corral.runtime.tool_execution import (
    PreparedToolCall,
    ToolExecutor,
    execute_prepared_call,
)
from corral.workspace import (
    AbsoluteWorkspaceFilesystem,
    WorkspaceFilesystem,
    build_terminal_tool,
    build_workspace_tools,
    confine_workspace_path,
)


def test_workspace_filesystem_uses_logical_paths_and_stays_confined(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    filesystem = WorkspaceFilesystem(root)

    filesystem.write_file("nested/data.txt", "first\nmatch\nlast")
    filesystem.copy_file("nested/data.txt", "copy.txt")

    assert filesystem.list_files(recursive=True) == ["copy.txt", "nested/data.txt"]
    assert filesystem.read_file("copy.txt") == "first\nmatch\nlast"
    assert filesystem.file_info("copy.txt") == {
        "path": "copy.txt",
        "type": "file",
        "size": 16,
    }
    assert filesystem.grep("match", ".", recursive=True)[0]["line_number"] == 2
    with pytest.raises(ValueError):
        filesystem.write_file("../outside.txt", "escape")
    assert not (tmp_path / "outside.txt").exists()


def test_workspace_file_tools_bind_only_to_a_materialization(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    tools = build_workspace_tools(AbsoluteWorkspaceFilesystem(root))

    assert set(tools) == {
        "cat_files",
        "copy_file",
        "file_info",
        "grep",
        "list_files",
        "mkdir",
        "move_file",
        "read_file",
        "write_file",
    }
    assert all("FSManager" not in repr(item) for item in tools.values())


def test_workspace_file_tools_require_canonical_paths(tmp_path):
    # A restricted worker rebuilds these tools against /workspace, so a relative
    # binding would accept different paths with and without a sandbox.
    with pytest.raises(TypeError, match="AbsoluteWorkspaceFilesystem"):
        build_workspace_tools(WorkspaceFilesystem(tmp_path))


def test_workspace_file_tools_match_the_restricted_worker_rebuild(tmp_path):
    host_root = tmp_path / "host"
    worker_root = tmp_path / "worker"
    host_root.mkdir()
    worker_root.mkdir()
    host = build_workspace_tools(AbsoluteWorkspaceFilesystem(host_root))
    worker = build_workspace_tools(AbsoluteWorkspaceFilesystem(worker_root))

    # What the agent is shown cannot depend on where the workspace lives.
    for name, item in host.items():
        assert item.description == worker[name].description
        assert item.params_json_schema == worker[name].params_json_schema
        assert item.worker_operation == f"workspace:{name}"

    host["write_file"].execute(path="/workspace/data.csv", content="a,b\n")
    assert host["read_file"].execute(path="/workspace/data.csv") == "a,b\n"
    assert json.loads(host["list_files"].execute()) == {
        "files": ["/workspace/data.csv"]
    }
    with pytest.raises(ValueError, match="absolute under /workspace"):
        host["read_file"].execute(path="data.csv")


@pytest.fixture
def resource_file_tools(tmp_path):
    source = tmp_path / "assets" / "database.txt"
    source.parent.mkdir()
    source.write_text("immutable\nmatch\n", encoding="utf-8")
    (source.parent / "private.txt").write_text("controller sibling", encoding="utf-8")
    descriptor = declare_file_resource("database", source, runtime_version="v1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def workspace_factory(root):
        tools = build_workspace_tools(AbsoluteWorkspaceFilesystem(root))
        for item in tools.values():
            item.resources = (descriptor.name,)
        unprivileged = build_workspace_tools(AbsoluteWorkspaceFilesystem(root))[
            "read_file"
        ]
        unprivileged.name = "unprivileged_read"
        tools[unprivileged.name] = unprivileged
        return tools

    environment = Environment(
        "resource-files",
        TaskDefinition(
            name="resource-files",
            description="Read an immutable resource.",
            tools=[],
            scoring_fn=lambda _answer: 1.0,
            submission_format={},
            resolve_answer=False,
        ),
        base_work_dir=str(tmp_path),
        workspace_path=str(workspace),
        toolset=Toolset(workspace_factory=workspace_factory),
        file_resources={descriptor.name: descriptor},
        file_resource_sources={descriptor.name: source},
    )
    started = environment.initial_event(execution_id="resource-files")
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="resource-files",
        branch_id="main",
        task=TaskState(environment=started.environment_metadata),
        environment=EnvironmentState(values={"hidden_arguments": {}}),
    )

    def execute(name, **arguments):
        return ToolExecutor(environment).execute(
            state, environment.tools[name], arguments, action_id=f"action-{name}"
        )

    return execute, descriptor, workspace, environment


def test_prepared_file_tools_read_only_their_authorized_resources(resource_file_tools):
    execute, descriptor, workspace, environment = resource_file_tools
    path = descriptor.public_path

    assert execute("read_file", path=path) == "immutable\nmatch\n"
    assert (
        execute("cat_files", paths=[path, path], separator="|")
        == "immutable\nmatch\n|immutable\nmatch\n"
    )
    assert json.loads(
        execute("list_files", path="/workspace/resources", recursive=True)
    ) == {"files": [path]}
    assert json.loads(execute("file_info", path=path)) == {
        "path": path,
        "type": "file",
        "size": descriptor.size,
        "read_only": True,
    }
    assert json.loads(
        execute("grep", pattern="match", path="/workspace/resources", recursive=True)
    ) == [{"file": path, "line": "match", "line_number": 2}]
    execute("copy_file", source=path, destination="/workspace/editable.txt")
    assert (workspace / "editable.txt").read_bytes() == b"immutable\nmatch\n"
    assert not (workspace / "resources").exists()
    # Authorization belongs to each prepared tool, not to the environment or
    # a filesystem object left over from a previous authorized invocation.
    with pytest.raises(ValueError, match="Permission denied"):
        execute("unprivileged_read", path=path)
    with pytest.raises(ValueError, match="Permission denied"):
        environment.tools["read_file"].execute(path=path)


def test_prepared_resource_file_tools_require_an_assigned_workspace(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / "private.txt").write_text("controller-only data")
    source = tmp_path / "asset.txt"
    source.write_text("declared resource")
    read = build_workspace_tools(AbsoluteWorkspaceFilesystem(workspace))["read_file"]
    read.resources = ("asset",)
    prepared = PreparedToolCall.capture(
        read,
        {"path": "/workspace/private.txt"},
        resource_mounts={
            "asset": (str(source), "/workspace/resources/asset/asset.txt")
        },
        execution_kind="local",
    )
    monkeypatch.chdir(controller)

    with pytest.raises(ValueError, match="assigned workspace"):
        execute_prepared_call(prepared)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", {"path": "/workspace/resources/undeclared/database.txt"}),
        ("read_file", {"path": "/workspace/resources/database/private.txt"}),
        ("read_file", {"path": "/workspace/resources/database/../private.txt"}),
        ("read_file", {"path": "/workspace/resources/database/./database.txt"}),
        (
            "write_file",
            {
                "path": "/workspace/resources/database/database.txt",
                "content": "overwrite",
            },
        ),
        (
            "copy_file",
            {
                "source": "/workspace/resources/database/database.txt",
                "destination": "/workspace/resources/database/copy.txt",
            },
        ),
        (
            "move_file",
            {
                "source": "/workspace/resources/database/database.txt",
                "destination": "/workspace/moved.txt",
            },
        ),
        ("mkdir", {"path": "/workspace/resources/shadow"}),
    ],
)
def test_prepared_resource_file_tools_reject_unauthorized_paths(
    resource_file_tools, name, arguments
):
    execute, _, _, _ = resource_file_tools
    with pytest.raises(ValueError, match="Permission denied"):
        execute(name, **arguments)


@pytest.mark.parametrize("directory_link", [False, True])
def test_resource_file_tools_reject_symlink_bindings(tmp_path, directory_link):
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "file.txt").write_text("private", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(resources if directory_link else resources / "file.txt")
    public = "/workspace/resources/data/file.txt"
    filesystem = AbsoluteWorkspaceFilesystem(
        tmp_path,
        resource_paths={public: alias / "file.txt" if directory_link else alias},
    )
    with pytest.raises(ValueError, match="symbolic links"):
        filesystem.read_file(public)
    with pytest.raises(ValueError, match="symbolic links"):
        filesystem.copy_file(public, "/workspace/copied.txt")


def test_resource_listing_does_not_expose_mutable_shadows(tmp_path):
    shadow = tmp_path / "resources" / "undeclared"
    shadow.mkdir(parents=True)
    (shadow / "private.txt").write_text("shadow", encoding="utf-8")
    filesystem = AbsoluteWorkspaceFilesystem(tmp_path)

    assert filesystem.list_files(recursive=True) == []
    with pytest.raises(ValueError, match="Permission denied"):
        filesystem.read_file("/workspace/resources/undeclared/private.txt")


def test_default_file_tools_use_the_integrated_workspace(tmp_path):
    workspace = tmp_path / "not-created"
    tools = default_file_tools(str(workspace))

    assert set(tools) == {
        "cat_files",
        "copy_file",
        "file_info",
        "list_files",
        "read_file",
        "write_file",
    }
    tools["write_file"].execute(
        path="/workspace/nested/data.txt", content="workspace data"
    )
    assert (
        tools["read_file"].execute(path="/workspace/nested/data.txt")
        == "workspace data"
    )
    assert (workspace / "nested" / "data.txt").read_text() == "workspace data"


def test_environment_materializes_missing_root_before_resolving_workspace_tools(
    tmp_path,
):
    workspace = tmp_path / "not-created"
    observed_roots = []

    def workspace_factory(root: str):
        observed_roots.append(Path(root))
        assert Path(root).is_dir()
        return {}

    task = TaskDefinition(
        name="workspace",
        description="workspace",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )

    Environment(
        "workspace",
        task,
        base_work_dir=str(workspace),
        toolset=Toolset(workspace_factory=workspace_factory),
    )

    assert workspace.is_dir()
    assert observed_roots == [workspace.resolve()]


def test_execution_workspaces_do_not_require_workspace_tools(tmp_path):
    task = TaskDefinition(
        name="scratch",
        description="Use native agent tools for scratch files.",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    template = Environment(
        "scratch",
        task,
        base_work_dir=str(tmp_path),
        toolset=Toolset(workspace_factory=None),
    )
    registry = RuntimeRegistry(agents={}, environments={"scratch": template})
    try:
        first = registry.environment("scratch", "trial-1")
        second = registry.environment("scratch", "trial-2")
        assert registry.environment("scratch", "trial-1") is first
        assert first.workspace_path != second.workspace_path
        assert template.workspace_path is None
        for environment in (first, second):
            root = Path(environment.workspace_path)
            assert root.parent == tmp_path
            assert root.is_dir()
            assert environment.tools.keys() == template.tools.keys()
            assert environment.toolset.workspace_factory is None
        Path(first.workspace_path, "notes.txt").write_text("trial one")
        assert not Path(second.workspace_path, "notes.txt").exists()
    finally:
        registry.close()


def test_workspace_filesystem_rejects_symlinks_even_when_they_stay_inside_root(
    tmp_path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (root / "alias").symlink_to(real, target_is_directory=True)
    filesystem = WorkspaceFilesystem(root)

    with pytest.raises(ValueError, match="symbolic links"):
        filesystem.write_file("alias/data.txt", "not manifestable")


def test_parallel_task_workspaces_cannot_access_their_siblings(tmp_path):
    workspace_a = tmp_path / "task-a"
    workspace_b = tmp_path / "task-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    (workspace_a / "private.txt").write_text("A", encoding="utf-8")
    (workspace_b / "private.txt").write_text("B", encoding="utf-8")
    filesystem = WorkspaceFilesystem(workspace_a)

    assert filesystem.read_file("private.txt") == "A"
    for sibling_path in ("../task-b/private.txt", str(workspace_b / "private.txt")):
        with pytest.raises(ValueError):
            filesystem.read_file(sibling_path)
        with pytest.raises(ValueError):
            filesystem.write_file(sibling_path, "overwrite")
        with pytest.raises(ValueError):
            confine_workspace_path(workspace_a, sibling_path)

    (workspace_a / "sibling-link").symlink_to(workspace_b, target_is_directory=True)
    with pytest.raises(ValueError, match="workspace path"):
        filesystem.read_file("sibling-link/private.txt")
    assert (workspace_b / "private.txt").read_text(encoding="utf-8") == "B"


def test_terminal_is_bounded_to_workspace_and_does_not_inherit_secrets(
    monkeypatch, tmp_path
):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-shell")
    terminal = build_terminal_tool(WorkspaceFilesystem(root))

    raw = terminal.execute(
        command='printf \'%s\\n\' "$PWD" "$OPENAI_API_KEY"; printf done > result.txt'
    )
    result = json.loads(raw)

    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert str(root) in result["output"]
    assert "must-not-reach-shell" not in result["output"]
    assert (root / "result.txt").read_text() == "done"

    bounded = json.loads(
        terminal.execute(command="printf 1234567890", max_output_chars=5)
    )
    assert bounded["output"] == "67890"
    assert bounded["truncated"] is True


def test_terminal_timeout_kills_the_command_process_group(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    terminal = build_terminal_tool(WorkspaceFilesystem(root))

    result = json.loads(terminal.execute(command="sleep 30 & wait", timeout_seconds=1))

    assert result["timed_out"] is True
    assert result["exit_code"] != 0
