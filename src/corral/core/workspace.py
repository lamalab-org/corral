"""Immutable logical workspaces for State v2.

A workspace contains only a portable manifest. File bytes are addressed by
content and live in an ArtifactStore, never in State or in a durable local path.
"""

from __future__ import annotations

import collections.abc as collections_abc
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    JsonValue as JSONValue,
)

from corral.core._immutable import FrozenModel, validate_sha256_hex

if TYPE_CHECKING:
    from collections.abc import Mapping

WORKSPACE_SCHEMA_VERSION = 2
BLOB_REF_PREFIX = "sha256:"


def new_workspace_id() -> str:
    """Mint a globally unique identity for a logical workspace."""
    return str(uuid4())


def normalize_workspace_path(path: str) -> str:
    """Return one portable relative POSIX path or reject an unsafe path."""
    if not isinstance(path, str) or not path:
        raise ValueError("workspace paths cannot be empty")
    if "\x00" in path:
        raise ValueError("workspace paths cannot contain NUL bytes")
    if "\\" in path:
        raise ValueError("workspace paths must use portable POSIX separators")

    candidate = PurePosixPath(path)
    if candidate == PurePosixPath("."):
        raise ValueError("workspace file paths cannot name the workspace root")
    if candidate.is_absolute():
        raise ValueError("workspace paths must be relative")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("workspace paths cannot contain '.', '..', or empty parts")

    normalized = candidate.as_posix()
    if normalized != path:
        raise ValueError(f"workspace path must be normalized as {normalized!r}")
    return normalized


def blob_ref_for_sha256(sha256: str) -> str:
    """Build the backend-independent content address for one blob."""
    validate_sha256_hex(sha256, field_name="sha256")
    return f"{BLOB_REF_PREFIX}{sha256}"


def sha256_from_blob_ref(blob_ref: str) -> str:
    """Extract and validate a SHA-256 digest from a content-addressed ref."""
    if not blob_ref.startswith(BLOB_REF_PREFIX):
        raise ValueError(f"blob_ref must start with {BLOB_REF_PREFIX!r}")
    sha256 = blob_ref.removeprefix(BLOB_REF_PREFIX)
    validate_sha256_hex(sha256, field_name="blob_ref digest")
    return sha256


class FileRef(FrozenModel):
    """Manifest entry for one regular file in a logical workspace."""

    path: str
    sha256: str
    size: int = Field(ge=0)
    blob_ref: str
    created_by_action: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return normalize_workspace_path(value)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        validate_sha256_hex(value, field_name="sha256")
        return value

    @model_validator(mode="after")
    def _validate_blob_ref(self) -> FileRef:
        if sha256_from_blob_ref(self.blob_ref) != self.sha256:
            raise ValueError("blob_ref digest must match sha256")
        if self.created_by_action is not None and not self.created_by_action.strip():
            raise ValueError("created_by_action cannot be empty")
        return self


class Artifact(FrozenModel):
    """A named, typed workspace output used by downstream consumers."""

    path: str
    kind: str = Field(min_length=1)
    metadata: Mapping[str, JSONValue] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return normalize_workspace_path(value)


class WorkspaceState(FrozenModel):
    """One immutable revision of a portable logical workspace."""

    schema_version: int = WORKSPACE_SCHEMA_VERSION
    id: str = Field(default_factory=new_workspace_id, min_length=1)
    revision: int = Field(default=0, ge=0)
    files: Mapping[str, FileRef] = Field(default_factory=dict)
    artifacts: Mapping[str, Artifact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> WorkspaceState:
        if self.schema_version != WORKSPACE_SCHEMA_VERSION:
            raise ValueError(
                f"Workspace schema_version must be {WORKSPACE_SCHEMA_VERSION}; "
                "other Workspace schemas are not supported"
            )
        for path, file_ref in self.files.items():
            normalized = normalize_workspace_path(path)
            if normalized != file_ref.path:
                raise ValueError(
                    f"workspace file key {path!r} must match FileRef.path "
                    f"{file_ref.path!r}"
                )
            parent = PurePosixPath(path).parent
            while parent != PurePosixPath("."):
                if parent.as_posix() in self.files:
                    raise ValueError(
                        f"workspace path {path!r} is nested below file "
                        f"{parent.as_posix()!r}"
                    )
                parent = parent.parent
        for name, artifact in self.artifacts.items():
            if not name.strip():
                raise ValueError("artifact names cannot be empty")
            if artifact.path not in self.files:
                raise ValueError(
                    f"artifact {name!r} references missing file {artifact.path!r}"
                )
        return self

    def fork(
        self,
        *,
        files: Mapping[str, FileRef | Mapping[str, Any]] | None = None,
        artifacts: Mapping[str, Artifact | Mapping[str, Any]] | None = None,
    ) -> WorkspaceState:
        """Create one child manifest while preserving workspace identity."""
        return WorkspaceState(
            id=self.id,
            revision=self.revision + 1,
            files=self.files if files is None else files,
            artifacts=self.artifacts if artifacts is None else artifacts,
        )


Artifact.model_rebuild(
    _types_namespace={"Mapping": collections_abc.Mapping},
)
WorkspaceState.model_rebuild(
    _types_namespace={"Mapping": collections_abc.Mapping},
)
