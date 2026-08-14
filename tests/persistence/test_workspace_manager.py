import asyncio
import hashlib
from pathlib import Path

import pytest

from corral.core import Artifact, State
from corral.persistence import (
    StoredBlob,
    WorkspaceDestinationError,
    WorkspaceManager,
    WorkspacePathError,
    WorkspaceStoreError,
)


def run(coro):
    return asyncio.run(coro)


def test_workspace_snapshot_restores_under_a_completely_different_path(tmp_path):
    source = tmp_path / "machine-a" / "trial"
    source.mkdir(parents=True)
    (source / "notes.txt").write_text("hello", encoding="utf-8")
    (source / "results").mkdir()
    (source / "results" / "data.bin").write_bytes(b"\x00\x01\x02")
    manager = WorkspaceManager(artifact_root=tmp_path / "shared-objects")

    workspace = run(
        manager.snapshot(
            source,
            created_by_action="action-1",
            artifacts={"dataset": Artifact(path="results/data.bin", kind="binary")},
        )
    )
    state = State(id="state-portable", workspace=workspace)
    restored_state = State.from_json(state.to_json())
    destination = tmp_path / "machine-b" / "unrelated-name"
    restored = run(manager.materialize(restored_state.workspace, destination))

    assert restored == destination
    assert (restored / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert (restored / "results" / "data.bin").read_bytes() == b"\x00\x01\x02"
    assert workspace.files["notes.txt"].created_by_action == "action-1"
    assert workspace.artifacts["dataset"].path == "results/data.bin"


def test_snapshot_tracks_changes_deletions_and_reuses_unchanged_blobs(tmp_path):
    source = tmp_path / "working"
    source.mkdir()
    (source / "keep.txt").write_text("same", encoding="utf-8")
    (source / "change.txt").write_text("old", encoding="utf-8")
    (source / "remove.txt").write_text("gone", encoding="utf-8")
    manager = WorkspaceManager(artifact_root=tmp_path / "objects")
    first = run(manager.snapshot(source, created_by_action="action-1"))

    unchanged = run(
        manager.snapshot(source, previous=first, created_by_action="action-2")
    )
    assert unchanged is first

    (source / "change.txt").write_text("new", encoding="utf-8")
    (source / "remove.txt").unlink()
    (source / "new.txt").write_text("added", encoding="utf-8")
    second = run(manager.snapshot(source, previous=first, created_by_action="action-2"))

    assert second.id == first.id
    assert second.revision == first.revision + 1
    assert second.files["keep.txt"] is first.files["keep.txt"]
    assert second.files["keep.txt"].created_by_action == "action-1"
    assert second.files["change.txt"].created_by_action == "action-2"
    assert second.files["new.txt"].created_by_action == "action-2"
    assert "remove.txt" not in second.files


class MemoryArtifactStore:
    """A remote-like store proving WorkspaceManager has no local-store coupling."""

    def __init__(self):
        self.objects = {}

    async def put_bytes(self, data):
        digest = hashlib.sha256(data).hexdigest()
        ref = f"sha256:{digest}"
        self.objects[ref] = data
        return StoredBlob(blob_ref=ref, sha256=digest, size=len(data))

    async def put_file(self, source):
        return await self.put_bytes(Path(source).read_bytes())

    async def get_bytes(self, blob_ref):
        return self.objects[blob_ref]

    async def materialize(self, blob_ref, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[blob_ref])

    async def contains(self, blob_ref):
        return blob_ref in self.objects


class CorruptingArtifactStore(MemoryArtifactStore):
    async def materialize(self, blob_ref, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"broken")


def test_workspace_manager_accepts_a_remote_style_artifact_store(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.txt").write_text("remote", encoding="utf-8")
    backend = MemoryArtifactStore()
    manager = WorkspaceManager(artifact_store=backend)

    workspace = run(manager.snapshot(source))
    restored = run(manager.materialize(workspace, tmp_path / "restored"))

    assert (restored / "result.txt").read_text(encoding="utf-8") == "remote"
    assert workspace.files["result.txt"].blob_ref in backend.objects


def test_workspace_manager_verifies_remote_materializations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.txt").write_text("remote", encoding="utf-8")
    backend = CorruptingArtifactStore()
    manager = WorkspaceManager(artifact_store=backend)
    workspace = run(manager.snapshot(source))

    with pytest.raises(WorkspaceStoreError, match="manifest check"):
        run(manager.materialize(workspace, tmp_path / "restored"))
    assert not (tmp_path / "restored").exists()


def test_materialization_rejects_stale_destination_and_snapshot_rejects_symlinks(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("data", encoding="utf-8")
    manager = WorkspaceManager(artifact_root=tmp_path / "objects")
    workspace = run(manager.snapshot(source))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(WorkspaceDestinationError, match="not empty"):
        run(manager.materialize(workspace, occupied))

    (source / "link.txt").symlink_to(source / "data.txt")
    with pytest.raises(WorkspacePathError, match="symbolic links"):
        run(manager.snapshot(source, previous=workspace))

    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(WorkspacePathError, match="symbolic link"):
        run(manager.snapshot(source_link, previous=workspace))


def test_temporary_materialization_is_isolated_and_cleaned_up(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("data", encoding="utf-8")
    manager = WorkspaceManager(artifact_root=tmp_path / "objects")
    workspace = run(manager.snapshot(source))

    async def use_temporary():
        async with manager.temporary_materialization(
            workspace, parent=tmp_path
        ) as path:
            assert (path / "data.txt").read_text(encoding="utf-8") == "data"
            return path

    materialized = run(use_temporary())
    assert not materialized.exists()
