"""Durable declarations for explicitly authorized environment resources."""

from __future__ import annotations

import collections.abc as collections_abc
import hashlib
import os
import re
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import Field, JsonValue, field_validator, model_validator

from corral.core._immutable import FrozenModel, validate_sha256_hex
from corral.core.workspace import blob_ref_for_sha256, sha256_from_blob_ref

if TYPE_CHECKING:
    from corral.persistence.artifacts import ArtifactStore

RESOURCE_STATE_NAMESPACE = "resources"
RESOURCE_SCHEMA_VERSION = 1
RESOURCE_CATALOG_METADATA_KEY = "resource_catalog"


class ResourceCatalogMismatchError(RuntimeError):
    """Persisted immutable resources differ from the bound Environment."""


class UnmaterializedResourceError(ValueError):
    """A declared resource is still a Git LFS pointer, not its real bytes."""


_GIT_LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:[0-9a-f]{64}\r?\n"
    rb"size [0-9]+\r?\n?"
)


def _reject_git_lfs_pointer(path: Path) -> None:
    """Fail closed instead of hashing or archiving an unresolved LFS pointer."""
    if path.stat().st_size > 1024:
        return
    if _GIT_LFS_POINTER.fullmatch(path.read_bytes()):
        raise UnmaterializedResourceError(
            f"resource source is an unresolved Git LFS pointer: {path}. "
            "Run `git lfs pull` before creating the environment."
        )


@runtime_checkable
class EnvironmentResourceAdapter(Protocol):
    """Restore and capture one JSON-backed trusted runtime resource."""

    def restore(self, state: JsonValue) -> object: ...

    def capture(self, runtime: object) -> JsonValue: ...


class FileResourceDescriptor(FrozenModel):
    """Portable descriptor for one immutable, content-addressed resource file.

    The source host path is intentionally absent. `blob_ref` is resolved by
    the execution's ArtifactStore and mounted read-only only for tools that
    declare this resource.
    """

    schema_version: int = RESOURCE_SCHEMA_VERSION
    name: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    blob_ref: str
    sha256: str
    size: int = Field(ge=0)
    runtime_version: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name != value
            or value in {".", ".."}
            or "\x00" in value
        ):
            raise ValueError("resource filename must be one portable basename")
        return value

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        validate_sha256_hex(value, field_name="resource sha256")
        return value

    @model_validator(mode="after")
    def _validate_descriptor(self) -> FileResourceDescriptor:
        if self.schema_version != RESOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"resource schema_version must be {RESOURCE_SCHEMA_VERSION}"
            )
        if sha256_from_blob_ref(self.blob_ref) != self.sha256:
            raise ValueError("resource blob_ref digest must match sha256")
        if "/" in self.name or self.name in {".", ".."} or "\x00" in self.name:
            raise ValueError("resource name must be one portable path component")
        return self

    @property
    def public_path(self) -> str:
        return f"/workspace/resources/{self.name}/{self.filename}"


class ResourceHandle(FrozenModel):
    """Ephemeral authorized handle given to trusted resource-aware code."""

    name: str
    public_path: str
    descriptor: FileResourceDescriptor
    # A controller path is intentionally not serializable or durable. Trusted
    # code can receive it as an explicit argument through the central executor.
    controller_path: Any = Field(exclude=True, repr=False, default=None)


class MaterializedResourcePath(str):
    """Ephemeral physical resource path authorized for local tool execution.

    A distinct string subtype lets resource helpers distinguish a path injected
    by the controller from an arbitrary host path supplied by application code.
    It is never persisted and behaves like `str` for existing file APIs.
    """

    __slots__ = ()


async def ingest_file_resource(
    name: str,
    source: str | Path,
    artifact_store: ArtifactStore,
    *,
    runtime_version: str,
    filename: str | None = None,
) -> FileResourceDescriptor:
    """Ingest a local file and return a durable descriptor without its path."""
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resource source must be a regular file: {source}")
    _reject_git_lfs_pointer(path)
    selected_name = filename or path.name
    stored = await artifact_store.put_file(source)
    return FileResourceDescriptor(
        name=name,
        filename=selected_name,
        blob_ref=stored.blob_ref,
        sha256=stored.sha256,
        size=stored.size,
        runtime_version=runtime_version,
    )


def declare_file_resource(
    name: str,
    source: str | Path,
    *,
    runtime_version: str,
    filename: str | None = None,
) -> FileResourceDescriptor:
    """Describe a local immutable file without retaining its controller path."""
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resource source must be a regular file: {source}")
    _reject_git_lfs_pointer(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()
    return FileResourceDescriptor(
        name=name,
        filename=filename or path.name,
        blob_ref=blob_ref_for_sha256(sha256),
        sha256=sha256,
        size=size,
        runtime_version=runtime_version,
    )


def resource_file_matches(path: str | Path, descriptor: FileResourceDescriptor) -> bool:
    """Return whether a regular local file matches an immutable descriptor."""
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        return False
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size == descriptor.size and digest.hexdigest() == descriptor.sha256


def declare_directory_resource(
    name: str,
    source: str | Path,
    *,
    cache_root: str | Path,
    runtime_version: str,
) -> tuple[FileResourceDescriptor, Path]:
    """Create a deterministic archive for an immutable directory resource.

    The returned source path is an ephemeral controller binding. Only the
    content-addressed descriptor belongs in durable execution state.
    """
    source_path = Path(source).expanduser().resolve()
    if source_path.is_symlink() or not source_path.is_dir():
        raise ValueError(f"resource source must be a regular directory: {source}")
    entries: list[tuple[str, Path]] = []
    digest = hashlib.sha256()
    for path in sorted(source_path.rglob("*")):
        relative = path.relative_to(source_path).as_posix()
        if path.is_symlink():
            raise ValueError(
                f"resource directories cannot contain symlinks: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"resource directories can contain regular files only: {relative}"
            )
        _reject_git_lfs_pointer(path)
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        entries.append((relative, path))
    tree_hash = digest.hexdigest()
    archive_root = Path(cache_root).expanduser().resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{tree_hash}.tar"
    if not archive_path.exists():
        descriptor_fd, temporary_name = tempfile.mkstemp(
            dir=archive_root,
            prefix=f".{tree_hash}.",
            suffix=".tmp",
        )
        os.close(descriptor_fd)
        temporary = Path(temporary_name)
        try:
            with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
                for relative, path in entries:
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o444
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
            temporary.replace(archive_path)
            archive_path.chmod(0o444)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = declare_file_resource(
        name,
        archive_path,
        runtime_version=runtime_version,
        filename=f"{name}.tar",
    )
    return descriptor, archive_path


@contextmanager
def extracted_resource_archive(
    archive_path: str | ResourceHandle | MaterializedResourcePath,
):
    """Materialize a declared directory archive into worker-private scratch."""
    if isinstance(archive_path, ResourceHandle):
        selected_path = str(archive_path.controller_path)
    elif isinstance(archive_path, MaterializedResourcePath):
        selected_path = str(archive_path)
    else:
        selected_path = archive_path
        archive = PurePosixPath(selected_path)
        expected_root = PurePosixPath("/workspace/resources")
        try:
            archive.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError(
                "resource archives must be below /workspace/resources"
            ) from exc
    with tempfile.TemporaryDirectory(prefix="corral-resource-") as temporary:
        destination = Path(temporary)
        with tarfile.open(selected_path, "r:") as bundle:
            for member in bundle.getmembers():
                candidate = PurePosixPath(member.name)
                if (
                    candidate.is_absolute()
                    or ".." in candidate.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError("resource archive contains an unsafe entry")
            bundle.extractall(destination, filter="data")
        yield destination


FileResourceDescriptor.model_rebuild(
    _types_namespace={"Mapping": collections_abc.Mapping}
)


__all__ = [
    "RESOURCE_CATALOG_METADATA_KEY",
    "RESOURCE_SCHEMA_VERSION",
    "RESOURCE_STATE_NAMESPACE",
    "EnvironmentResourceAdapter",
    "FileResourceDescriptor",
    "MaterializedResourcePath",
    "ResourceCatalogMismatchError",
    "ResourceHandle",
    "UnmaterializedResourceError",
    "declare_directory_resource",
    "declare_file_resource",
    "extracted_resource_archive",
    "ingest_file_resource",
    "resource_file_matches",
]
