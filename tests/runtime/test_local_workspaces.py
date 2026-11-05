"""Exercise local workspace operations without Unix filesystem capabilities."""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from corral.runtime import permissions


@pytest.fixture
def portable_root(tmp_path, monkeypatch):
    # Resolve the host's temporary-directory aliases before testing link checks.
    root = tmp_path.resolve()
    for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "fchmod"):
        monkeypatch.delattr(os, name, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    monkeypatch.setattr(os, "supports_fd", set())
    monkeypatch.setattr(permissions, "_enabled", False)
    return root


def make_symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")


def test_node_creation_clone_and_promotion_without_unix_apis(portable_root):
    parent = portable_root / "trial"
    parent.mkdir()
    (parent / "data").mkdir()
    payload = bytes(range(256)) + b"\r\n\x00\xff"
    (parent / "data" / "input.bin").write_bytes(payload)
    (parent / "answer.txt").write_text("initial")

    first = Path(permissions.create_node_workspace(str(parent), str(parent)))
    second = Path(permissions.create_node_workspace(str(parent), str(parent)))
    assert first != second
    assert first.parent == second.parent == parent / permissions.NODE_WORKSPACE_DIR
    assert (first / "data" / "input.bin").read_bytes() == payload
    assert (second / "answer.txt").read_text() == "initial"

    (first / "answer.txt").write_text("selected answer")
    (first / "result.bin").write_bytes(payload[::-1])
    clone = Path(permissions.create_node_workspace(str(parent), str(first)))
    assert clone.parent == first.parent
    assert (clone / "answer.txt").read_text() == "selected answer"
    assert (clone / "result.bin").read_bytes() == payload[::-1]
    (clone / "answer.txt").write_text("clone only")
    assert (first / "answer.txt").read_text() == "selected answer"
    assert (parent / "answer.txt").read_text() == "initial"
    assert (second / "answer.txt").read_text() == "initial"

    (parent / "keep.txt").write_text("retain this existing file")
    copied = permissions.copy_workspace(first, parent)
    assert copied == ["answer.txt", "data/input.bin", "result.bin"]
    assert (parent / "answer.txt").read_text() == "selected answer"
    assert (parent / "result.bin").read_bytes() == payload[::-1]
    assert (parent / "keep.txt").read_text() == "retain this existing file"
    assert (second / "answer.txt").read_text() == "initial"


def test_copy_excludes_runtime_directories_only_at_workspace_root(portable_root):
    source = portable_root / "source"
    destination = portable_root / "destination"
    source.mkdir()
    destination.mkdir()
    reserved = (
        permissions.NODE_WORKSPACE_DIR,
        permissions.RESOURCE_WORKSPACE_DIR,
        f"{permissions.SCRATCH_PREFIX}sdk",
    )
    for name in reserved:
        top = source / name
        top.mkdir()
        (top / "private.txt").write_text("excluded")
        nested = source / "data" / name
        nested.mkdir(parents=True)
        (nested / "public.txt").write_text("included")

    copied = permissions.copy_workspace(source, destination)

    assert set(copied) == {f"data/{name}/public.txt" for name in reserved}
    for name in reserved:
        assert not (destination / name).exists()
        assert (destination / "data" / name / "public.txt").read_text() == "included"


def test_node_source_must_belong_to_parent(portable_root):
    parent = portable_root / "trial"
    outside = portable_root / "trial-other"
    parent.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")

    with pytest.raises(ValueError, match="node source must belong"):
        permissions.create_node_workspace(str(parent), str(outside))

    assert not (parent / permissions.NODE_WORKSPACE_DIR).exists()


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("location", ["root", "ancestor", "file", "directory"])
def test_copy_rejects_symlinks(portable_root, side, location):
    source = portable_root / "source"
    destination = portable_root / "destination"
    outside = portable_root / "outside"
    source.mkdir()
    destination.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")
    (source / "answer.txt").write_text("answer")
    selected = source if side == "source" else destination

    if location == "root":
        link = portable_root / "alias"
        make_symlink(link, selected, directory=True)
        if side == "source":
            source = link
        else:
            destination = link
    elif location == "ancestor":
        link = portable_root / "alias"
        make_symlink(link, portable_root, directory=True)
        if side == "source":
            source = link / "source"
        else:
            destination = link / "destination"
    elif location == "file":
        entry = selected / "answer.txt"
        entry.unlink(missing_ok=True)
        make_symlink(entry, outside / "secret.txt")
    else:
        make_symlink(selected / "data", outside, directory=True)
        if side == "destination":
            (source / "data").mkdir()
            (source / "data" / "secret.txt").write_text("overwrite attempt")

    with pytest.raises((OSError, ValueError)):
        permissions.copy_workspace(source, destination)

    assert (outside / "secret.txt").read_text() == "outside"


def test_node_creation_rejects_linked_nodes_directory(portable_root):
    parent = portable_root / "trial"
    outside = portable_root / "outside"
    parent.mkdir()
    outside.mkdir()
    make_symlink(parent / permissions.NODE_WORKSPACE_DIR, outside, directory=True)

    with pytest.raises((OSError, ValueError)):
        permissions.create_node_workspace(str(parent), str(parent))

    assert not list(outside.iterdir())


@pytest.mark.parametrize("side", ["source", "destination"])
def test_copy_rejects_windows_reparse_metadata(portable_root, monkeypatch, side):
    source = portable_root / "source"
    destination = portable_root / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "data").mkdir()
    (source / "data" / "answer.txt").write_text("answer")
    (destination / "data").mkdir()
    reparse = (source if side == "source" else destination) / "data"
    original_lstat = Path.lstat

    def report_reparse(path):
        status = original_lstat(path)
        if path == reparse:
            return SimpleNamespace(
                st_mode=status.st_mode,
                st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
            )
        return status

    monkeypatch.setattr(Path, "lstat", report_reparse)

    with pytest.raises(OSError, match="reparse"):
        permissions.copy_workspace(source, destination)

    assert not (destination / "data" / "answer.txt").exists()


def test_copy_replaces_destination_without_modifying_hardlink_target(portable_root):
    source = portable_root / "source"
    destination = portable_root / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "answer.txt").write_text("new answer")
    outside = portable_root / "outside.txt"
    outside.write_text("outside")
    try:
        os.link(outside, destination / "answer.txt")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    assert permissions.copy_workspace(source, destination) == ["answer.txt"]
    assert (destination / "answer.txt").read_text() == "new answer"
    assert outside.read_text() == "outside"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="host cannot create FIFOs")
def test_copy_rejects_special_files_but_ignores_open_mpi_debugger_fifo(portable_root):
    source = portable_root / "source"
    destination = portable_root / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "answer.txt").write_text("answer")
    mpi = source / "ompi.example" / "pid.123" / "0"
    mpi.mkdir(parents=True)
    os.mkfifo(mpi / "debugger_attach_fifo")

    assert permissions.copy_workspace(source, destination) == ["answer.txt"]
    assert not (destination / "ompi.example/pid.123/0/debugger_attach_fifo").exists()

    os.mkfifo(source / "other_fifo")
    with pytest.raises(ValueError, match="regular files"):
        permissions.copy_workspace(source, destination)


@pytest.mark.parametrize("operation", ["create", "copy"])
def test_enforced_workspaces_never_fall_back_to_portable_copy(
    portable_root, monkeypatch, operation
):
    parent = portable_root / "trial"
    destination = portable_root / "destination"
    parent.mkdir()
    destination.mkdir()
    (parent / "answer.txt").write_text("answer")
    monkeypatch.setattr(permissions, "_enabled", True)

    with pytest.raises(RuntimeError):
        if operation == "create":
            permissions.create_node_workspace(str(parent), str(parent))
        else:
            permissions.copy_workspace(parent, destination)

    assert not list(destination.iterdir())
    assert not (parent / permissions.NODE_WORKSPACE_DIR).exists()
