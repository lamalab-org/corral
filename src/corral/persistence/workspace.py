"""Snapshot and materialization services for logical workspaces."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from corral.core.workspace import Artifact, FileRef, WorkspaceState
from corral.persistence.artifacts import ArtifactStore, LocalArtifactStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping


class WorkspaceStoreError(RuntimeError):
    """Base error for workspace snapshot or materialization failures."""


class WorkspacePathError(WorkspaceStoreError):
    """A local workspace contains a non-portable or unsafe filesystem entry."""


class WorkspaceDestinationError(WorkspaceStoreError):
    """A materialization destination cannot safely be populated."""


class WorkspaceManager:
    """Translate between local directories and immutable WorkspaceState.

    ``artifact_store`` is the only byte-storage dependency. It defaults to a
    local content-addressed store but can be replaced by an S3 implementation
    without changing manifests, State, or materialization callers.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        *,
        artifact_root: str | Path = ".corral/artifacts",
    ) -> None:
        self.artifact_store: ArtifactStore = (
            artifact_store
            if artifact_store is not None
            else LocalArtifactStore(artifact_root)
        )

    @staticmethod
    def _hash_local_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _local_files(root: Path) -> tuple[tuple[str, Path], ...]:
        if root.is_symlink() or not root.is_dir():
            raise WorkspacePathError(
                f"workspace source must be a regular directory: {root}"
            )

        files: list[tuple[str, Path]] = []
        for entry in sorted(root.rglob("*")):
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise WorkspacePathError(
                    f"workspace cannot contain symbolic links: {relative}"
                )
            if entry.is_dir():
                continue
            if not entry.is_file():
                raise WorkspacePathError(
                    f"workspace can contain regular files only: {relative}"
                )
            files.append((relative, entry))
        return tuple(files)

    async def snapshot(
        self,
        source: str | Path,
        *,
        previous: WorkspaceState | None = None,
        created_by_action: str | None = None,
        artifacts: Mapping[str, Artifact | Mapping[str, Any]] | None = None,
    ) -> WorkspaceState:
        """Upload a directory and return its complete portable manifest.

        An unchanged snapshot returns ``previous`` exactly. New or modified
        files are attributed to ``created_by_action``; unchanged files retain
        their original provenance.
        """
        if created_by_action is not None and not created_by_action.strip():
            raise ValueError("created_by_action cannot be empty")

        source_path = Path(source)
        if source_path.is_symlink():
            raise WorkspacePathError(
                f"workspace source cannot be a symbolic link: {source}"
            )
        root = source_path.resolve()
        local_files = self._local_files(root)
        file_refs: dict[str, FileRef] = {}
        for relative, local_path in local_files:
            stored = await self.artifact_store.put_file(local_path)
            old_ref = previous.files.get(relative) if previous is not None else None
            if old_ref is not None and old_ref.sha256 == stored.sha256:
                file_refs[relative] = old_ref
            else:
                file_refs[relative] = FileRef(
                    path=relative,
                    sha256=stored.sha256,
                    size=stored.size,
                    blob_ref=stored.blob_ref,
                    created_by_action=created_by_action,
                )

        if artifacts is None:
            artifact_manifest = {
                name: artifact
                for name, artifact in (previous.artifacts.items() if previous else ())
                if artifact.path in file_refs
            }
        else:
            artifact_manifest = dict(artifacts)

        if previous is None:
            return WorkspaceState(files=file_refs, artifacts=artifact_manifest)
        if previous.files == file_refs and previous.artifacts == artifact_manifest:
            return previous
        return previous.fork(files=file_refs, artifacts=artifact_manifest)

    async def materialize(
        self,
        workspace: WorkspaceState,
        destination: str | Path,
    ) -> Path:
        """Recreate a workspace at a new local path from its ArtifactStore.

        Materialization is staged beside the destination and published by one
        rename. A pre-existing non-empty destination is rejected so stale files
        can never silently enter the logical workspace.
        """
        if not isinstance(workspace, WorkspaceState):
            raise TypeError("materialize() requires WorkspaceState v2")

        destination_path = Path(destination)
        if destination_path.is_symlink():
            raise WorkspaceDestinationError(
                "workspace destination cannot be a symbolic link"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            if not destination_path.is_dir():
                raise WorkspaceDestinationError(
                    f"workspace destination is not a directory: {destination_path}"
                )
            if any(destination_path.iterdir()):
                raise WorkspaceDestinationError(
                    f"workspace destination is not empty: {destination_path}"
                )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_path.name}.corral-",
                dir=destination_path.parent,
            )
        )
        try:
            for relative, file_ref in sorted(workspace.files.items()):
                target = staging.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                await self.artifact_store.materialize(file_ref.blob_ref, target)
                actual_sha256, actual_size = await asyncio.to_thread(
                    self._hash_local_file,
                    target,
                )
                if actual_sha256 != file_ref.sha256 or actual_size != file_ref.size:
                    raise WorkspaceStoreError(
                        f"materialized file {relative!r} failed its manifest check"
                    )

            if destination_path.exists():
                destination_path.rmdir()
            staging.replace(destination_path)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination_path

    @asynccontextmanager
    async def temporary_materialization(
        self,
        workspace: WorkspaceState,
        *,
        parent: str | Path | None = None,
    ) -> AsyncIterator[Path]:
        """Yield an isolated materialization and remove it on context exit."""
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix="corral-workspace-",
                dir=parent,
            )
        )
        destination = temporary_root / "workspace"
        try:
            yield await self.materialize(workspace, destination)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
