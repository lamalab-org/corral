"""Filesystem access confined to one materialized WorkspaceState."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from corral.core.tool import Tool, tool
from corral.core.workspace import normalize_workspace_path

_TERMINAL_MAX_TIMEOUT_SECONDS = 3600
_TERMINAL_MAX_OUTPUT_CHARS = 100_000
PUBLIC_WORKSPACE_ROOT = "/workspace"


_PUBLIC_WORKSPACE_TOKEN = re.compile(r"/workspace(?=/|$|[\s'\"`;,):\]}&|<>])")


def materialize_public_workspace_paths(value: Any, root: str | Path) -> Any:
    """Translate public workspace tokens for the unconfined local fallback.

    Restricted workers receive a real `/workspace` mount. Host-side local
    execution cannot create that process-private mount, so tools with declared
    workspace access receive an ephemeral copy of their arguments in which
    canonical paths point at the assigned controller directory. Embedded
    occurrences cover shell and Python snippets as well as plain path fields.

    The reserved resource namespace is deliberately left alone; file resources
    have their own explicit controller binding and must never be confused with
    mutable task files.
    """
    physical = str(Path(root).expanduser().resolve())

    def replace(item: Any) -> Any:
        if isinstance(item, str):
            # Controller-issued path capabilities use `str` subclasses so
            # ordinary file APIs can consume them without exposing them as
            # agent-controlled workspace paths. Keep those bindings opaque.
            if type(item) is not str:
                return item

            def substitute(match: re.Match[str]) -> str:
                suffix = item[match.end() :]
                if suffix == "/resources" or suffix.startswith("/resources/"):
                    return match.group(0)
                return physical

            return _PUBLIC_WORKSPACE_TOKEN.sub(substitute, item)
        if isinstance(item, list):
            return [replace(child) for child in item]
        if isinstance(item, tuple):
            return tuple(replace(child) for child in item)
        if isinstance(item, dict):
            return {replace(key): replace(child) for key, child in item.items()}
        return item

    return replace(value)


def materialize_local_tool_arguments(
    tool: Any,
    arguments: dict[str, Any],
    workspace: str | Path | None,
    *,
    prepared: Any | None = None,
) -> dict[str, Any]:
    """Prepare host execution without pretending it is OS-isolated.

    Local execution translates canonical paths for compatibility, but unlike a
    restricted worker it cannot enforce read-only access at the kernel level.
    A prepared call supplies frozen policy so queued execution never consults
    mutable tool metadata.
    """
    trusted = (
        prepared.trusted if prepared is not None else getattr(tool, "trusted", False)
    )
    if workspace is None or trusted:
        return dict(arguments)
    access = (
        prepared.workspace_access
        if prepared is not None
        else getattr(tool, "workspace_access", "none")
    )
    access = access.value if hasattr(access, "value") else str(access)
    operation = (
        prepared.worker_operation
        if prepared is not None
        else getattr(tool, "worker_operation", None)
    )
    if access == "none" or (
        isinstance(operation, str) and operation.startswith("workspace:")
    ):
        return dict(arguments)
    return materialize_public_workspace_paths(arguments, workspace)


def normalize_public_workspace_path(path: str, *, allow_root: bool = False) -> str:
    """Validate an agent-visible path and return its portable relative form."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("workspace paths cannot be empty or contain NUL bytes")
    if "\\" in path:
        raise ValueError("workspace paths must use portable POSIX separators")
    candidate = PurePosixPath(path)
    if not candidate.is_absolute():
        raise ValueError("workspace paths must be absolute under /workspace")
    if any(part in {".", ".."} for part in candidate.parts):
        raise ValueError("workspace paths cannot contain '.' or '..'")
    normalized = candidate.as_posix()
    if normalized != path:
        raise ValueError(f"workspace path must be normalized as {normalized!r}")
    root = PurePosixPath(PUBLIC_WORKSPACE_ROOT)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("workspace paths must be under /workspace") from exc
    if relative == PurePosixPath("."):
        if not allow_root:
            raise ValueError("workspace path cannot name the workspace root")
        return "."
    if relative.parts[0] == "resources":
        raise ValueError(
            "/workspace/resources is reserved for declared read-only resources"
        )
    return normalize_workspace_path(relative.as_posix())


def resolve_public_workspace_path(
    root: str | Path, path: str, *, allow_root: bool = False
) -> Path:
    """Map canonical `/workspace` syntax to one controller materialization."""
    relative = normalize_public_workspace_path(path, allow_root=allow_root)
    return confine_workspace_path(root, relative, allow_root=allow_root)


def workspace_relative_path(path: str, *, allow_root: bool = False) -> str:
    """Validate an agent's absolute /workspace path before translating it.

    Snapshot keys and controller paths are separate, trusted representations.
    Never normalize a relative path or traversal into an allowed tool argument.
    """
    return normalize_public_workspace_path(path, allow_root=allow_root)


def confine_workspace_path(
    root: str | Path,
    path: str | Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a local path while proving it remains below `root`.

    This helper is for trusted server code that needs a physical path (for
    example, evaluation of a file submission). Agent-facing filesystem tools
    use canonical absolute paths rooted at `/workspace`. Symbolic links are
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

    path_root = "."

    def _resolve(self, path: str, *, allow_root: bool = False) -> Path:
        if allow_root and path in {"", "."}:
            return self.root
        try:
            normalized = normalize_workspace_path(path)
            return confine_workspace_path(self.root, normalized, allow_root=allow_root)
        except ValueError as exc:
            raise ValueError(
                f"Permission denied: operation is not permitted: {exc}"
            ) from exc

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
            "path": self.path_root
            if target == self.root
            else self._logical_path(target),
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
        info = self.file_info(path)
        if info["type"] == "file":
            files = [info["path"]]
        elif info["type"] == "directory":
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


class AbsoluteWorkspaceFilesystem(WorkspaceFilesystem):
    """Workspace filesystem with the canonical agent-facing `/workspace` API.

    The physical `root` remains controller-private. Every accepted and
    returned path is absolute in the worker namespace, which prevents prompts,
    observations, and persisted actions from depending on a host path.
    """

    path_root = PUBLIC_WORKSPACE_ROOT

    def _public_relative(self, path: str, *, allow_root: bool) -> str:
        return normalize_public_workspace_path(path, allow_root=allow_root)

    def _resolve(self, path: str, *, allow_root: bool = False) -> Path:
        try:
            relative = self._public_relative(path, allow_root=allow_root)
            return confine_workspace_path(self.root, relative, allow_root=allow_root)
        except ValueError as exc:
            raise ValueError(
                f"Permission denied: operation is not permitted: {exc}"
            ) from exc

    def _logical_path(self, path: Path) -> str:
        relative = super()._logical_path(path)
        return f"{PUBLIC_WORKSPACE_ROOT}/{relative}"

    def list_files(
        self, path: str = PUBLIC_WORKSPACE_ROOT, *, recursive: bool = False
    ) -> list[str]:
        return super().list_files(path, recursive=recursive)

    def file_info(self, path: str) -> dict[str, Any]:
        info = super().file_info(path)
        if info["path"] == ".":
            info["path"] = PUBLIC_WORKSPACE_ROOT
        return info


def build_workspace_tools(filesystem: WorkspaceFilesystem) -> dict[str, Tool]:
    """Build local file tools for one materialized v2 workspace."""

    default_path = (
        PUBLIC_WORKSPACE_ROOT
        if isinstance(filesystem, AbsoluteWorkspaceFilesystem)
        else "."
    )

    @tool(workspace_access="read")
    def list_files(path: str = default_path, recursive: bool = False) -> str:
        """List regular files in the workspace.

        Args:
            path: Workspace directory (canonically rooted at /workspace).
            recursive: Whether to include nested files.
        """
        return json.dumps(
            {"files": filesystem.list_files(path, recursive=recursive)},
            indent=2,
        )

    @tool(workspace_access="read")
    def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Workspace file path (canonically rooted at /workspace).
        """
        return filesystem.read_file(path)

    @tool(workspace_access="read_write")
    def write_file(path: str, content: str) -> str:
        """Write a text file to the workspace.

        Args:
            path: Workspace file path (canonically rooted at /workspace).
            content: Text content to write.
        """
        filesystem.write_file(path, content)
        return f"Successfully wrote to {path}"

    @tool(workspace_access="read")
    def file_info(path: str) -> str:
        """Inspect a workspace file or directory.

        Args:
            path: Workspace path (canonically rooted at /workspace).
        """
        return json.dumps(filesystem.file_info(path), indent=2)

    @tool(workspace_access="read_write")
    def copy_file(source: str, destination: str) -> str:
        """Copy a file within the workspace.

        Args:
            source: Source path in the workspace.
            destination: Destination path in the workspace.
        """
        filesystem.copy_file(source, destination)
        return f"Copied {source} to {destination}"

    @tool(workspace_access="read_write")
    def move_file(source: str, destination: str) -> str:
        """Move a file within the workspace.

        Args:
            source: Source path in the workspace.
            destination: Destination path in the workspace.
        """
        filesystem.move_file(source, destination)
        return f"Moved {source} to {destination}"

    @tool(workspace_access="read_write")
    def mkdir(path: str, create_parents: bool = False) -> str:
        """Create a directory within the workspace.

        Args:
            path: Workspace directory path (canonically rooted at /workspace).
            create_parents: Whether to create missing parent directories.
        """
        filesystem.mkdir(path, create_parents=create_parents)
        return f"Directory {path} created successfully."

    @tool(workspace_access="read")
    def cat_files(paths: list[str], separator: str = "\n") -> str:
        """Concatenate workspace text files.

        Args:
            paths: Workspace file paths (canonically rooted at /workspace).
            separator: Text inserted between file contents.
        """
        return filesystem.cat_files(paths, separator=separator)

    @tool(workspace_access="read")
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
            path: Workspace file or directory path.
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

    tools = {
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
    for name, item in tools.items():
        # Restricted workers rebuild these closures against their canonical
        # mount instead of serializing a controller-side physical root.
        item.worker_operation = f"workspace:{name}"
        if filesystem.path_root == PUBLIC_WORKSPACE_ROOT:
            item.description += " All paths must be absolute /workspace paths; relative paths are forbidden."
            for parameter_name, parameter in item.params_json_schema[
                "properties"
            ].items():
                if parameter_name in {"path", "paths", "source", "destination"}:
                    parameter["description"] = (
                        "Absolute POSIX path(s) under /workspace. Relative paths, "
                        "parent traversal and symbolic links are forbidden."
                    )
    return tools


def build_terminal_tool(filesystem: WorkspaceFilesystem) -> Tool:
    """Build a bounded shell tool rooted in one execution workspace."""

    @tool(workspace_access="read_write")
    def terminal(
        command: str,
        timeout_seconds: int = 120,
        max_output_chars: int = 20_000,
    ) -> str:
        """Run a foreground shell command inside the isolated workspace.

        Args:
            command: Shell program to run with the workspace as its cwd.
            timeout_seconds: Wall-clock limit, at most one hour.
            max_output_chars: Maximum combined stdout/stderr returned.
        """
        if not command.strip():
            raise ValueError("terminal command cannot be empty")
        if not 1 <= timeout_seconds <= _TERMINAL_MAX_TIMEOUT_SECONDS:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        if not 1 <= max_output_chars <= _TERMINAL_MAX_OUTPUT_CHARS:
            raise ValueError("max_output_chars must be between 1 and 100000")

        # The shell inherits its caller's identity; Docker tool workers have
        # already dropped privileges at the shared execution boundary.
        with tempfile.TemporaryFile() as output_stream:
            process = subprocess.Popen(
                ["/bin/sh", "-c", command],
                cwd=filesystem.root,
                env={
                    "HOME": str(filesystem.root),
                    "TMPDIR": str(filesystem.root),
                    "LANG": "C.UTF-8",
                    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                },
                stdout=output_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                # Commands may not leave descendants in their process group,
                # even when the foreground shell already exited cleanly.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)

            byte_count = output_stream.tell()
            output_stream.seek(max(0, byte_count - max_output_chars * 4))
            combined = output_stream.read().decode("utf-8", errors="replace")
        truncated = byte_count > max_output_chars or len(combined) > max_output_chars
        if len(combined) > max_output_chars:
            combined = combined[-max_output_chars:]
        return json.dumps(
            {
                "exit_code": process.returncode,
                "output": combined,
                "timed_out": timed_out,
                "truncated": truncated,
            },
            indent=2,
        )

    # The restricted bootstrap rebuilds this tool from the workspace path.
    # No controller-side callable or environment is serialized for shell input.
    terminal.worker_operation = "terminal"
    return terminal
