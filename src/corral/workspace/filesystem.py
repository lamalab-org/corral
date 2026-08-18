"""Filesystem access confined to one materialized WorkspaceState."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from corral.core.tool import Tool, tool
from corral.core.workspace import normalize_workspace_path


def confine_workspace_path(
    root: str | Path,
    path: str | Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a local path while proving it remains below ``root``.

    This helper is for trusted server code that needs a physical path (for
    example, evaluation of a file submission). Agent-facing filesystem tools
    remain stricter and accept logical relative paths only. Symbolic links are
    rejected even when they currently point back inside the workspace so a
    task cannot turn one workspace path into ambient host-filesystem access.
    """
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError(f"workspace root must be a regular directory: {root}")
    root_path = root_path.resolve()

    raw_path = str(path)
    if not raw_path or "\x00" in raw_path:
        raise ValueError("workspace paths cannot be empty or contain NUL bytes")
    if "\\" in raw_path:
        raise ValueError("workspace paths must use portable POSIX separators")

    supplied = Path(path)
    if ".." in supplied.parts:
        raise ValueError(f"workspace path cannot traverse its parent: {path!r}")
    candidate = supplied if supplied.is_absolute() else root_path / supplied
    resolved = candidate.resolve(strict=False)
    if resolved != root_path and root_path not in resolved.parents:
        raise ValueError(f"workspace path escapes its materialization: {path!r}")
    if resolved == root_path and not allow_root:
        raise ValueError("workspace path cannot name the workspace root")

    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"workspace path escapes its materialization: {path!r}"
        ) from exc
    current = root_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"workspace paths cannot traverse symbolic links: {path!r}"
            )
    return resolved


class WorkspaceFilesystem:
    """Safe local file operations for a materialized v2 workspace.

    Paths accepted and returned by this class are logical workspace-relative
    POSIX paths. Provider protocols, bucket paths, and host-absolute paths are
    deliberately absent; remote bytes enter through ArtifactStore before this
    filesystem is constructed.
    """

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root)
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError(f"workspace root must be a regular directory: {root}")
        self.root = root_path.resolve()

    def _resolve(self, path: str, *, allow_root: bool = False) -> Path:
        if allow_root and path in {"", "."}:
            return self.root
        normalized = normalize_workspace_path(path)
        return confine_workspace_path(self.root, normalized, allow_root=allow_root)

    def _logical_path(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"filesystem entry escapes its workspace: {path}")
        return path.relative_to(self.root).as_posix()

    def list_files(self, path: str = ".", *, recursive: bool = False) -> list[str]:
        """List regular files using logical workspace paths."""
        directory = self._resolve(path, allow_root=True)
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"workspace directory was not found: {path}")
        entries = directory.rglob("*") if recursive else directory.iterdir()
        files: list[str] = []
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(
                    f"workspace cannot contain symbolic links: "
                    f"{self._logical_path(entry)}"
                )
            if entry.is_file():
                files.append(self._logical_path(entry))
        return sorted(files)

    def read_file(self, path: str, *, encoding: str = "utf-8") -> str:
        """Read one UTF-8 workspace file."""
        source = self._resolve(path)
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"workspace file was not found: {path}")
        return source.read_text(encoding=encoding)

    def write_file(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write one text file below the workspace root."""
        destination = self._resolve(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ValueError("workspace file cannot be a symbolic link")
        destination.write_text(content, encoding=encoding)

    def file_info(self, path: str) -> dict[str, Any]:
        """Return portable metadata for a regular file or directory."""
        target = self._resolve(path, allow_root=True)
        if target.is_symlink() or not target.exists():
            raise FileNotFoundError(f"workspace path was not found: {path}")
        info: dict[str, Any] = {
            "path": "." if target == self.root else self._logical_path(target),
            "type": "directory" if target.is_dir() else "file",
        }
        if target.is_file():
            info["size"] = target.stat().st_size
        return info

    def copy_file(self, source: str, destination: str) -> None:
        """Copy one regular file within the workspace."""
        source_path = self._resolve(source)
        destination_path = self._resolve(destination)
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"workspace file was not found: {source}")
        if destination_path.is_symlink():
            raise ValueError("workspace destination cannot be a symbolic link")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)

    def move_file(self, source: str, destination: str) -> None:
        """Move one regular file within the workspace."""
        source_path = self._resolve(source)
        destination_path = self._resolve(destination)
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(f"workspace file was not found: {source}")
        if destination_path.is_symlink():
            raise ValueError("workspace destination cannot be a symbolic link")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(destination_path)

    def mkdir(self, path: str, *, create_parents: bool = False) -> None:
        """Create a directory within the workspace."""
        directory = self._resolve(path)
        directory.mkdir(parents=create_parents, exist_ok=False)

    def cat_files(self, paths: list[str], *, separator: str = "\n") -> str:
        """Read and concatenate workspace files."""
        if not paths:
            raise ValueError("at least one workspace path is required")
        return separator.join(self.read_file(path) for path in paths)

    def grep(
        self,
        pattern: str,
        path: str,
        *,
        recursive: bool = False,
        ignore_case: bool = False,
        line_numbers: bool = True,
        context_lines: int = 0,
        max_matches: int = 0,
    ) -> list[dict[str, Any]]:
        """Search workspace text files using a Python regular expression."""
        if context_lines < 0 or max_matches < 0:
            raise ValueError("context_lines and max_matches cannot be negative")
        target = self._resolve(path, allow_root=True)
        if target.is_file():
            files = [self._logical_path(target)]
        elif target.is_dir():
            files = self.list_files(path, recursive=recursive)
        else:
            raise FileNotFoundError(f"workspace path was not found: {path}")

        expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        matches: list[dict[str, Any]] = []
        for file_path in files:
            lines = self.read_file(file_path).splitlines()
            for index, line in enumerate(lines):
                if expression.search(line) is None:
                    continue
                match: dict[str, Any] = {"file": file_path, "line": line.strip()}
                if line_numbers:
                    match["line_number"] = index + 1
                if context_lines:
                    start = max(0, index - context_lines)
                    end = min(len(lines), index + context_lines + 1)
                    match["context_before"] = [
                        {"line": lines[item].strip(), "line_number": item + 1}
                        for item in range(start, index)
                    ]
                    match["context_after"] = [
                        {"line": lines[item].strip(), "line_number": item + 1}
                        for item in range(index + 1, end)
                    ]
                matches.append(match)
                if max_matches and len(matches) >= max_matches:
                    return matches
        return matches


def build_workspace_tools(filesystem: WorkspaceFilesystem) -> dict[str, Tool]:
    """Build local file tools for one materialized v2 workspace."""

    @tool
    def list_files(path: str = ".", recursive: bool = False) -> str:
        """List regular files in the workspace.

        Args:
            path: Logical workspace-relative directory, or '.' for the root.
            recursive: Whether to include nested files.
        """
        return json.dumps(
            {"files": filesystem.list_files(path, recursive=recursive)},
            indent=2,
        )

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Logical workspace-relative file path.
        """
        return filesystem.read_file(path)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a text file to the workspace.

        Args:
            path: Logical workspace-relative file path.
            content: Text content to write.
        """
        filesystem.write_file(path, content)
        return f"Successfully wrote to {path}"

    @tool
    def file_info(path: str) -> str:
        """Inspect a workspace file or directory.

        Args:
            path: Logical workspace-relative path.
        """
        return json.dumps(filesystem.file_info(path), indent=2)

    @tool
    def copy_file(source: str, destination: str) -> str:
        """Copy a file within the workspace.

        Args:
            source: Logical source file path.
            destination: Logical destination file path.
        """
        filesystem.copy_file(source, destination)
        return f"Copied {source} to {destination}"

    @tool
    def move_file(source: str, destination: str) -> str:
        """Move a file within the workspace.

        Args:
            source: Logical source file path.
            destination: Logical destination file path.
        """
        filesystem.move_file(source, destination)
        return f"Moved {source} to {destination}"

    @tool
    def mkdir(path: str, create_parents: bool = False) -> str:
        """Create a directory within the workspace.

        Args:
            path: Logical workspace-relative directory path.
            create_parents: Whether to create missing parent directories.
        """
        filesystem.mkdir(path, create_parents=create_parents)
        return f"Directory {path} created successfully."

    @tool
    def cat_files(paths: list[str], separator: str = "\n") -> str:
        """Concatenate workspace text files.

        Args:
            paths: Logical workspace-relative file paths.
            separator: Text inserted between file contents.
        """
        return filesystem.cat_files(paths, separator=separator)

    @tool
    def grep(
        pattern: str,
        path: str,
        recursive: bool = False,
        ignore_case: bool = False,
        line_numbers: bool = True,
        context_lines: int = 0,
        max_matches: int = 0,
    ) -> str:
        """Search workspace text files using a regular expression.

        Args:
            pattern: Python regular expression to search for.
            path: Logical workspace-relative file or directory path.
            recursive: Whether to include nested files for a directory.
            ignore_case: Whether matching is case-insensitive.
            line_numbers: Whether results include one-based line numbers.
            context_lines: Number of adjacent lines to include.
            max_matches: Maximum matches, or zero for no limit.
        """
        return json.dumps(
            filesystem.grep(
                pattern,
                path,
                recursive=recursive,
                ignore_case=ignore_case,
                line_numbers=line_numbers,
                context_lines=context_lines,
                max_matches=max_matches,
            ),
            indent=2,
        )

    return {
        item.name: item
        for item in (
            list_files,
            read_file,
            write_file,
            file_info,
            copy_file,
            move_file,
            mkdir,
            cat_files,
            grep,
        )
    }
