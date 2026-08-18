from pathlib import Path

import pytest

from corral.core.environment import Environment, Toolset, default_file_tools
from corral.core.task import TaskDefinition
from corral.workspace import (
    WorkspaceFilesystem,
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
    tools = build_workspace_tools(WorkspaceFilesystem(root))

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
    tools["write_file"].execute(path="nested/data.txt", content="workspace data")
    assert tools["read_file"].execute(path="nested/data.txt") == "workspace data"
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
