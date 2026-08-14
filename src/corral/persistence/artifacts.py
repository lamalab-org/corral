"""ArtifactStore contract and local content-addressed implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from corral.core.workspace import blob_ref_for_sha256, sha256_from_blob_ref

_COPY_CHUNK_SIZE = 1024 * 1024


class ArtifactStoreError(RuntimeError):
    """Base error for content-addressed artifact persistence."""


class ArtifactNotFoundError(ArtifactStoreError):
    """The requested content-addressed blob does not exist."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Stored or transferred bytes do not match their content address."""


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """Metadata returned after ArtifactStore has durably accepted bytes."""

    blob_ref: str
    sha256: str
    size: int


@runtime_checkable
class ArtifactStore(Protocol):
    """Backend-neutral storage for immutable content-addressed bytes.

    A remote backend such as S3 implements this same upload/download boundary;
    WorkspaceState never contains provider-specific bucket names or local paths.
    """

    async def put_bytes(self, data: bytes) -> StoredBlob:
        """Persist bytes idempotently and return their portable content ref."""
        ...

    async def put_file(self, source: str | Path) -> StoredBlob:
        """Persist one local regular file without loading it all into memory."""
        ...

    async def get_bytes(self, blob_ref: str) -> bytes:
        """Load and integrity-check a blob."""
        ...

    async def materialize(self, blob_ref: str, destination: str | Path) -> None:
        """Download a blob atomically to a local path."""
        ...

    async def contains(self, blob_ref: str) -> bool:
        """Return whether an integrity-checked blob is available."""
        ...


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class LocalArtifactStore:
    """Filesystem-backed ArtifactStore using SHA-256 object keys.

    Blobs are laid out as ``<root>/<first two digest chars>/<digest>``. Writes
    use a temporary file and atomic replacement, so concurrent identical puts
    are idempotent and interrupted uploads never become visible as blobs.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"ArtifactStore root is not a directory: {self.root}")
        self.root = self.root.resolve()

    def _blob_path(self, blob_ref: str) -> Path:
        sha256 = sha256_from_blob_ref(blob_ref)
        return self.root / sha256[:2] / sha256

    @staticmethod
    def _stored_blob(sha256: str, size: int) -> StoredBlob:
        return StoredBlob(
            blob_ref=blob_ref_for_sha256(sha256),
            sha256=sha256,
            size=size,
        )

    @staticmethod
    def _verify(path: Path, *, expected_sha256: str) -> int:
        actual_sha256, size = _hash_file(path)
        if actual_sha256 != expected_sha256:
            raise ArtifactIntegrityError(
                f"blob {expected_sha256!r} failed its content-hash check"
            )
        return size

    def _commit_temporary_blob(self, temporary: Path, stored: StoredBlob) -> None:
        destination = self._blob_path(stored.blob_ref)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            existing_size = self._verify(
                destination,
                expected_sha256=stored.sha256,
            )
            if existing_size != stored.size:
                raise ArtifactIntegrityError(
                    f"blob {stored.blob_ref!r} has inconsistent size"
                )
            temporary.unlink()
            return

        temporary.replace(destination)

    def _put_bytes(self, data: bytes) -> StoredBlob:
        if not isinstance(data, bytes):
            raise TypeError("ArtifactStore.put_bytes() accepts bytes only")
        sha256 = hashlib.sha256(data).hexdigest()
        stored = self._stored_blob(sha256, len(data))

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".corral-upload-",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._commit_temporary_blob(temporary, stored)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return stored

    async def put_bytes(self, data: bytes) -> StoredBlob:
        return await asyncio.to_thread(self._put_bytes, data)

    def _put_file(self, source: str | Path) -> StoredBlob:
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"ArtifactStore source must be a regular file: {source}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".corral-upload-",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with (
                os.fdopen(descriptor, "wb") as target_stream,
                source_path.open("rb") as source_stream,
            ):
                while chunk := source_stream.read(_COPY_CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
                    target_stream.write(chunk)
                target_stream.flush()
                os.fsync(target_stream.fileno())

            stored = self._stored_blob(digest.hexdigest(), size)
            self._commit_temporary_blob(temporary, stored)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return stored

    async def put_file(self, source: str | Path) -> StoredBlob:
        return await asyncio.to_thread(self._put_file, source)

    def _get_bytes(self, blob_ref: str) -> bytes:
        path = self._blob_path(blob_ref)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact {blob_ref!r} was not found")
        data = path.read_bytes()
        expected_sha256 = sha256_from_blob_ref(blob_ref)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ArtifactIntegrityError(
                f"blob {blob_ref!r} failed its content-hash check"
            )
        return data

    async def get_bytes(self, blob_ref: str) -> bytes:
        return await asyncio.to_thread(self._get_bytes, blob_ref)

    def _materialize(self, blob_ref: str, destination: str | Path) -> None:
        source = self._blob_path(blob_ref)
        if not source.is_file():
            raise ArtifactNotFoundError(f"artifact {blob_ref!r} was not found")

        destination_path = Path(destination)
        if destination_path.is_symlink():
            raise ValueError("artifact destination cannot be a symbolic link")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.corral-",
            dir=destination_path.parent,
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with (
                source.open("rb") as source_stream,
                os.fdopen(descriptor, "wb") as target_stream,
            ):
                while chunk := source_stream.read(_COPY_CHUNK_SIZE):
                    digest.update(chunk)
                    target_stream.write(chunk)
                target_stream.flush()
                os.fsync(target_stream.fileno())

            expected_sha256 = sha256_from_blob_ref(blob_ref)
            if digest.hexdigest() != expected_sha256:
                raise ArtifactIntegrityError(
                    f"blob {blob_ref!r} failed its content-hash check"
                )
            temporary.replace(destination_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    async def materialize(self, blob_ref: str, destination: str | Path) -> None:
        await asyncio.to_thread(self._materialize, blob_ref, destination)

    def _contains(self, blob_ref: str) -> bool:
        path = self._blob_path(blob_ref)
        if not path.is_file():
            return False
        self._verify(path, expected_sha256=sha256_from_blob_ref(blob_ref))
        return True

    async def contains(self, blob_ref: str) -> bool:
        return await asyncio.to_thread(self._contains, blob_ref)
