"""Local views and tools for materialized execution workspaces."""

from corral.workspace.filesystem import (
    AbsoluteWorkspaceFilesystem,
    WorkspaceFilesystem,
    build_terminal_tool,
    build_workspace_tools,
    confine_workspace_path,
    workspace_relative_path,
)

__all__ = [
    "AbsoluteWorkspaceFilesystem",
    "WorkspaceFilesystem",
    "build_terminal_tool",
    "build_workspace_tools",
    "confine_workspace_path",
    "workspace_relative_path",
]
