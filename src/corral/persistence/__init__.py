"""Persistence backends for immutable Corral State revisions."""

from corral.persistence.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    LocalArtifactStore,
    StoredBlob,
)
from corral.persistence.base import (
    StateIntegrityError,
    StateNotFoundError,
    StateStore,
    StateStoreError,
    StateTransitionConflictError,
)
from corral.persistence.sqlite import SQLiteStateStore
from corral.persistence.workspace import (
    WorkspaceDestinationError,
    WorkspaceManager,
    WorkspacePathError,
    WorkspaceStoreError,
)

__all__ = [
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactStore",
    "ArtifactStoreError",
    "LocalArtifactStore",
    "SQLiteStateStore",
    "StateIntegrityError",
    "StateNotFoundError",
    "StateStore",
    "StateStoreError",
    "StateTransitionConflictError",
    "StoredBlob",
    "WorkspaceDestinationError",
    "WorkspaceManager",
    "WorkspacePathError",
    "WorkspaceStoreError",
]
