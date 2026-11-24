"""Public workspace paths for MD tools."""

from pathlib import Path

from corral.workspace import (
    AbsoluteWorkspaceFilesystem,
    confine_workspace_path,
    workspace_relative_path,
)

ASSET_DIRECTORIES = frozenset({"models", "potentials", "structures"})
MDWorkspaceFilesystem = AbsoluteWorkspaceFilesystem


def local_path(root: str | Path, path: str, *, allow_root: bool = False) -> Path:
    """Translate a validated mutable public path to controller storage."""
    relative = workspace_relative_path(path, allow_root=allow_root)
    return confine_workspace_path(root, relative, allow_root=allow_root)
