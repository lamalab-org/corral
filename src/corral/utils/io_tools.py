import json
import re
from pathlib import Path

import fsspec
import modal

from corral.backend.tool import Tool, tool


class FSManager:
    """A file-system abstraction layer using fsspec.
    This object is created with a given protocol (e.g., "file", "s3", "ftp")
    with optional Modal integration and base path resolution."""

    def __init__(
        self,
        protocol: str = "file",
        base_path: str = "./",
        app: str | None = None,
        **kwargs,
    ):
        self.protocol = protocol
        self.base_path = Path(base_path) if base_path else None
        self.fs = fsspec.filesystem(protocol, **kwargs)
        self.app = app

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path against the base_path"""
        if self.base_path is None:
            return path

        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        # Relative path - resolve against base_path
        resolved = self.base_path / path_obj
        return str(resolved)

    def list_files(self, path: str, recursive: bool = False) -> list[str]:
        """List files in a directory"""
        try:
            resolved_path = self._resolve_path(path)
            if self.app:
                list_files_ = modal.Function.from_name(self.app, "list_files")
                return list_files_.remote(resolved_path, recursive)
            return self.fs.ls(resolved_path, detail=False, recursive=recursive)
        except Exception as e:
            raise RuntimeError(f"Error listing files in {path}: {e}") from e

    def read_file(self, path: str) -> str:
        """Read contents of a file"""
        try:
            resolved_path = self._resolve_path(path)
            if self.app:
                read_file_ = modal.Function.from_name(self.app, "read_file")
                return read_file_.remote(resolved_path)
            with self.fs.open(resolved_path, "r") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}") from e

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file"""
        try:
            resolved_path = self._resolve_path(path)
            if self.app:
                write_file_ = modal.Function.from_name(self.app, "write_file")
                write_file_.remote(resolved_path, content)
            else:
                # Ensure directory exists
                resolved_path_obj = Path(resolved_path)
                resolved_path_obj.parent.mkdir(parents=True, exist_ok=True)

                with self.fs.open(resolved_path, "w") as f:
                    f.write(content)
        except Exception as e:
            raise RuntimeError(f"Error writing file {path}: {e}") from e

    def file_info(self, path: str) -> dict:
        """Get file information"""
        try:
            resolved_path = self._resolve_path(path)
            if self.app:
                file_info_ = modal.Function.from_name(self.app, "file_info")
                return file_info_.remote(resolved_path)
            return self.fs.info(resolved_path)
        except Exception as e:
            raise RuntimeError(f"Error getting file info for {path}: {e}") from e

    def copy_file(self, source: str, destination: str) -> None:
        try:
            resolved_source = self._resolve_path(source)
            resolved_dest = self._resolve_path(destination)
            if self.app:
                copy_file_ = modal.Function.from_name(self.app, "copy_file")
                copy_file_.remote(resolved_source, resolved_dest)
            else:
                self.fs.copy(resolved_source, resolved_dest)
        except Exception as e:
            raise RuntimeError(
                f"Error copying from {source} to {destination}: {e}"
            ) from e

    def move_file(self, source: str, destination: str) -> None:
        try:
            resolved_source = self._resolve_path(source)
            resolved_dest = self._resolve_path(destination)
            if self.app:
                move_file_ = modal.Function.from_name(self.app, "move_file")
                move_file_.remote(resolved_source, resolved_dest)
            else:
                # fsspec does not always provide a move method; if not, copy then remove.
                try:
                    self.fs.mv(resolved_source, resolved_dest)
                except Exception:
                    self.copy_file(source, destination)
                    self.fs.rm(resolved_source)
        except Exception as e:
            raise RuntimeError(
                f"Error moving from {source} to {destination}: {e}"
            ) from e

    def mkdir(self, path: str, create_parents: bool = False) -> None:
        """Create a directory at the given path."""
        try:
            resolved_path = self._resolve_path(path)
            if self.app:
                mkdir_ = modal.Function.from_name(self.app, "mkdir")
                mkdir_.remote(resolved_path, create_parents)
            else:
                if create_parents and hasattr(self.fs, "mkdirs"):
                    self.fs.mkdirs(resolved_path, exist_ok=True)
                else:
                    self.fs.mkdir(resolved_path)
        except Exception as e:
            raise RuntimeError(f"Error creating directory {path}: {e}") from e

    def cat_files(self, paths: list[str], separator: str = "\n") -> str:
        """Concatenate the contents of multiple files with the given separator."""
        resolved_paths = [self._resolve_path(path) for path in paths]
        if self.app:
            cat_files_ = modal.Function.from_name(self.app, "cat_files")
            return cat_files_.remote(resolved_paths, separator)

        contents = []
        for path in resolved_paths:
            try:
                with self.fs.open(path, "r") as f:
                    contents.append(f.read())
            except Exception as e:
                raise RuntimeError(f"Error reading file {path}: {e}") from e
        return separator.join(contents)


def _grep_file(
    fs_manager: FSManager,
    file_path: str,
    pattern: str,
    ignore_case: bool,
    line_numbers: bool,
    context_lines: int,
) -> list[dict]:
    """Search for `pattern` in a single file (helper for the `grep` tool)."""
    try:
        content = fs_manager.read_file(file_path)
        lines = content.splitlines()
        flags = re.IGNORECASE if ignore_case else 0
        matches = []

        for i, line in enumerate(lines, 1):
            if re.search(pattern, line, flags):
                match = {"file": file_path, "line": line.strip()}

                if line_numbers:
                    match["line_number"] = i

                if context_lines > 0:
                    start = max(0, i - context_lines - 1)
                    before = [
                        {"line": lines[j].strip(), "line_number": j + 1}
                        for j in range(start, i - 1)
                    ]

                    end = min(len(lines), i + context_lines)
                    after = [
                        {"line": lines[j].strip(), "line_number": j + 1}
                        for j in range(i, end)
                    ]

                    if before:
                        match["context_before"] = before
                    if after:
                        match["context_after"] = after

                matches.append(match)

        return matches
    except Exception as e:
        return [{"file": file_path, "error": f"Error processing file: {e!s}"}]


def build_file_tools(fs_manager: FSManager) -> dict[str, Tool]:
    """Build the filesystem tools bound to `fs_manager`.

    Each tool is created with the shared :func:`~corral.backend.tool.tool`
    decorator, so its JSON Schema is generated by the OpenAI Agents SDK from the
    function's type hints and docstring — exactly like the task ("env") tools —
    rather than from a hand-declared :class:`ToolArgument` list (which used
    Python type names such as `"str"` that are invalid JSON Schema types and
    were rejected by strict MCP clients like OpenHands). The tool functions close
    over `fs_manager` so each stays bound to one workspace. Callers pick the
    subset they expose from the returned name→tool mapping.
    """

    @tool
    def list_files(path: str, recursive: bool = False) -> str:
        """List files in a directory.

        Args:
            path: Path to the directory.
            recursive: Whether to list files recursively.
        """
        files = fs_manager.list_files(path, recursive)
        return json.dumps({"files": files}, indent=2)

    @tool
    def read_file(path: str) -> str:
        """Read the contents of a file into a string.

        Args:
            path: Path to the file to read.
        """
        return fs_manager.read_file(path)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file.

        Args:
            path: Path to the file to write.
            content: Content to write to the file.
        """
        fs_manager.write_file(path, content)
        return f"Successfully wrote to {path}"

    @tool
    def file_info(path: str) -> str:
        """Get information about a file or directory.

        Args:
            path: Path to the file or directory.
        """
        info = fs_manager.file_info(path)
        return json.dumps(info, indent=2)

    @tool
    def cat_files(paths: list[str], separator: str = "\n") -> str:
        """Concatenate and display contents of one or more files.

        Args:
            paths: List of file paths to concatenate.
            separator: Separator between file contents.
        """
        if not paths:
            raise ValueError("At least one file path must be provided")

        contents = []
        for path in paths:
            try:
                contents.append(fs_manager.read_file(path))
            except Exception as e:
                raise Exception(f"Error reading file {path}: {e!s}") from e

        return separator.join(contents)

    @tool
    def copy_file(source: str, destination: str) -> str:
        """Copy a file from a source to destination.

        Args:
            source: Source file path.
            destination: Destination file path.
        """
        try:
            fs_manager.copy_file(source, destination)
            return f"Copied {source} to {destination}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def mkdir(path: str, create_parents: bool = False) -> str:
        """Create a directory. Optionally create parent directories.

        Args:
            path: Path of the directory to create.
            create_parents: If true, create parent directories as needed.
        """
        try:
            fs_manager.mkdir(path, create_parents)
            return f"Directory {path} created successfully."
        except Exception as e:
            return f"Error: {e}"

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
        """Search for patterns in files using Python regular expressions.

        Args:
            pattern: Regular expression pattern to search for.
            path: Path to file or directory to search.
            recursive: Search recursively in directories (like grep -r).
            ignore_case: Perform case-insensitive matching (like grep -i).
            line_numbers: Show line numbers for matches (like grep -n).
            context_lines: Number of context lines to show before and after
                each match (like grep -C).
            max_matches: Maximum number of matches to return (0 for unlimited).
        """
        try:
            if recursive:
                files = fs_manager.list_files(path, recursive=True)
            else:
                info = fs_manager.file_info(path)
                files = (
                    [path] if info["type"] == "file" else fs_manager.list_files(path)
                )

            all_matches: list[dict] = []
            match_count = 0

            for file_path in files:
                try:
                    if fs_manager.file_info(file_path)["type"] != "file":
                        continue
                except Exception:
                    continue

                matches = _grep_file(
                    fs_manager,
                    file_path,
                    pattern,
                    ignore_case,
                    line_numbers,
                    context_lines,
                )

                all_matches.extend(matches)
                match_count += len(matches)

                if max_matches > 0 and match_count >= max_matches:
                    all_matches = all_matches[:max_matches]
                    break

            if not all_matches:
                return "No matches found"

            result_lines = []
            for match in all_matches:
                if "error" in match:
                    result_lines.append(f"ERROR: {match['file']}: {match['error']}")
                    continue

                file_path = match["file"]

                if "context_before" in match:
                    for ctx in match["context_before"]:
                        line_prefix = f"{ctx['line_number']}:" if line_numbers else ""
                        result_lines.append(
                            f"{file_path}:{line_prefix}{ctx['line']} (context)"
                        )

                line_prefix = (
                    f"{match['line_number']}:"
                    if line_numbers and "line_number" in match
                    else ""
                )
                result_lines.append(f"{file_path}:{line_prefix}{match['line']}")

                if "context_after" in match:
                    for ctx in match["context_after"]:
                        line_prefix = f"{ctx['line_number']}:" if line_numbers else ""
                        result_lines.append(
                            f"{file_path}:{line_prefix}{ctx['line']} (context)"
                        )

                result_lines.append("")

            return "\n".join(result_lines)

        except Exception as e:
            return f"Error searching for pattern: {e!s}"

    return {
        "list_files": list_files,
        "read_file": read_file,
        "write_file": write_file,
        "file_info": file_info,
        "cat_files": cat_files,
        "copy_file": copy_file,
        "mkdir": mkdir,
        "grep": grep,
    }
