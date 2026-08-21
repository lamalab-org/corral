"""Local views and tools for materialized execution workspaces."""

from corral.workspace.filesystem import (
    WorkspaceFilesystem,
    build_terminal_tool,
    build_workspace_tools,
    confine_workspace_path,
)

__all__ = [
    "WorkspaceFilesystem",
    "build_terminal_tool",
    "build_workspace_tools",
    "confine_workspace_path",
]
