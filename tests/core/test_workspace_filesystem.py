import pytest

from corral.workspace import WorkspaceFilesystem, build_workspace_tools


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
