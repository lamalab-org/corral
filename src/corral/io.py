import json
import re
import modal

import fsspec
from typing import Optional

from corral.base import Tool, ToolArgument

class FSManager:
    """A file-system abstraction layer using fsspec.
    This object is created with a given protocol (e.g., "file", "s3", "ftp")"""

    def __init__(self, protocol: str = "file", app: str | None = None, **kwargs):
        self.protocol = protocol
        self.fs = fsspec.filesystem(protocol, **kwargs)
        self.app = app

    def list_files(self, path: str, recursive: bool = False) -> list[str]:
        """List files in a directory"""
        try:
            if self.app:
                list_files_ = modal.Function.lookup(self.app, "list_files")
                return list_files_.remote(path, recursive)
            return self.fs.ls(path, detail=False, recursive=recursive)
        except Exception as e:
            raise RuntimeError(f"Error listing files: {e}") from e

    def read_file(self, path: str) -> str:
        """Read contents of a file"""
        try:
            if self.app:
                read_file_ = modal.Function.lookup(self.app, "read_file")
                return read_file_.remote(path)
            with self.fs.open(path, "r") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading files: {e}") from e

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file"""
        try:
            if self.app:
                write_file_ = modal.Function.lookup(self.app, "write_file") 
                write_file_.remote(path, content)
            else:
                with self.fs.open(path, "w") as f:
                    f.write(content)
        except Exception as e:
            raise RuntimeError(f"Error writing files: {e}") from e

    def file_info(self, path: str) -> dict:
        """Get file information"""
        try:
            if self.app:
                file_info_ = modal.Function.lookup(self.app, "file_info")
                return file_info_.remote(path) 
            return self.fs.info(path)
        except Exception as e:
            raise RuntimeError(f"Error getting file info: {e}") from e

    def copy_file(self, source: str, destination: str) -> None:
        try:
            if self.app:
                copy_file_ = modal.Function.lookup(self.app, "copy_file")
                copy_file_.remote(source, destination)
            else:
                self.fs.copy(source, destination)
        except Exception as e:
            raise RuntimeError(
                f"Error copying from {source} to {destination}: {e}"
            ) from e

    def move_file(self, source: str, destination: str) -> None:
        try:
            if self.app:
                move_file_ = modal.Function.lookup(self.app, "move_file")
                move_file_.remote(source, destination)
            else:
            # fsspec does not always provide a move method; if not, copy then remove.
                self.fs.mv(source, destination)
        except Exception:
            self.copy_file(source, destination)
            self.fs.rm(source)

    def mkdir(self, path: str, create_parents: bool = False) -> None:
        """Create a directory at the given path.

        If create_parents is True and the backend supports it, create all missing parent directories.
        """
        try:
            if self.app:
                mkdir_ = modal.Function.lookup(self.app, "mkdir")
                mkdir_.remote(path, create_parents)
            else:
                if create_parents and hasattr(self.fs, "mkdirs"):
                    # Many fsspec implementations support mkdirs.
                    # If not available, fall back to calling mkdir for each missing part.
                    self.fs.mkdirs(path, exist_ok=True)
                else:
                    self.fs.mkdir(path)
        except Exception as e:
            raise RuntimeError(f"Error creating directory {path}: {e}") from e

    def cat_files(self, paths: list[str], separator: str = "\n") -> str:
        """Concatenate the contents of multiple files with the given separator."""
        if self.app:
            cat_files_ = modal.Function.lookup(self.app, "cat_files")
            return cat_files_.remote(paths, separator)
        contents = []
        for path in paths:
            try:
                contents.append(self.read_file(path))
            except Exception as e:
                raise RuntimeError(f"Error reading file {path}: {e}") from e
        return separator.join(contents)


class ReadFileTool(Tool):
    """Tool for reading file contents"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="read_file",
            description="Read the contents of a file into a string",
            arguments=[
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path to the file to read",
                    required=True,
                )
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        return self.fs_manager.read_file(kwargs["path"])

class ListFilesTool(Tool):
    """Tool for listing files in a directory"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="list_files",
            description="List files in a directory",
            arguments=[
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path to the directory",
                    required=True,
                ),
                ToolArgument(
                    name="recursive",
                    type="bool",
                    description="Whether to list files recursively",
                    required=False,
                    default=False,
                ),
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        files = self.fs_manager.list_files(
            kwargs["path"], kwargs.get("recursive", False)
        )
        return json.dumps({"files": files}, indent=2)

class WriteFileTool(Tool):
    """Tool for writing content to a file"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="write_file",
            description="Write content to a file",
            arguments=[
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path to the file to write",
                    required=True,
                ),
                ToolArgument(
                    name="content",
                    type="str",
                    description="Content to write to the file",
                    required=True,
                ),
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        self.fs_manager.write_file(kwargs["path"], kwargs["content"])
        return f"Successfully wrote to {kwargs['path']}"


class FileInfoTool(Tool):
    """Tool for getting file information"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="file_info",
            description="Get information about a file or directory",
            arguments=[
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path to the file or directory",
                    required=True,
                )
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        info = self.fs_manager.file_info(kwargs["path"])
        return json.dumps(info, indent=2)


class CatFilesTool(Tool):
    """Tool for concatenating and displaying file contents"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="cat_files",
            description="Concatenate and display contents of one or more files",
            arguments=[
                ToolArgument(
                    name="paths",
                    type="list[str]",
                    description="List of file paths to concatenate",
                    required=True,
                ),
                ToolArgument(
                    name="separator",
                    type="str",
                    description="Separator between file contents",
                    required=False,
                    default="\n",
                ),
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        paths = kwargs["paths"]
        separator = kwargs.get("separator", "\n")

        if not paths:
            raise ValueError("At least one file path must be provided")

        contents = []
        for path in paths:
            try:
                content = self.fs_manager.read_file(path)
                contents.append(content)
            except Exception as e:
                raise Exception(f"Error reading file {path}: {e!s}") from e

        return separator.join(contents)


class CopyFileTool(Tool):
    """Tool for copying files."""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="copy_file",
            description="Copy a file from a source to destination",
            arguments=[
                ToolArgument(
                    name="source",
                    type="str",
                    description="Source file path",
                    required=True,
                ),
                ToolArgument(
                    name="destination",
                    type="str",
                    description="Destination file path",
                    required=True,
                ),
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        try:
            self.fs_manager.copy_file(kwargs["source"], kwargs["destination"])
            return f"Copied {kwargs['source']} to {kwargs['destination']}"
        except Exception as e:
            return f"Error: {e}"


class MkdirTool(Tool):
    """Tool for creating a directory (and optionally its parent dirs)."""

    def __init__(self, fs_manager):
        super().__init__(
            name="mkdir",
            description="Create a directory. Optionally create parent directories.",
            arguments=[
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path of the directory to create",
                    required=True,
                ),
                ToolArgument(
                    name="create_parents",
                    type="bool",
                    description="If true, create parent directories as needed.",
                    required=False,
                    default=False,
                ),
            ],
        )
        self.fs_manager = fs_manager

    def execute(self, **kwargs) -> str:
        try:
            self.fs_manager.mkdir(kwargs["path"], kwargs.get("create_parents", False))
            return f"Directory {kwargs['path']} created successfully."
        except Exception as e:
            return f"Error: {e}"


class GrepTool(Tool):
    """Tool for searching patterns in files, similar to bash grep"""

    def __init__(self, fs_manager: FSManager):
        super().__init__(
            name="grep",
            description="Search for patterns in files using Python regular expressions",
            arguments=[
                ToolArgument(
                    name="pattern",
                    type="str",
                    description="Regular expression pattern to search for",
                    required=True,
                ),
                ToolArgument(
                    name="path",
                    type="str",
                    description="Path to file or directory to search",
                    required=True,
                ),
                ToolArgument(
                    name="recursive",
                    type="bool",
                    description="Search recursively in directories (like grep -r)",
                    required=False,
                    default=False,
                ),
                ToolArgument(
                    name="ignore_case",
                    type="bool",
                    description="Perform case-insensitive matching (like grep -i)",
                    required=False,
                    default=False,
                ),
                ToolArgument(
                    name="line_numbers",
                    type="bool",
                    description="Show line numbers for matches (like grep -n)",
                    required=False,
                    default=True,
                ),
                ToolArgument(
                    name="context_lines",
                    type="int",
                    description="Number of context lines to show before and after each match (like grep -C)",
                    required=False,
                    default=0,
                ),
                ToolArgument(
                    name="max_matches",
                    type="int",
                    description="Maximum number of matches to return (0 for unlimited)",
                    required=False,
                    default=0,
                ),
            ],
        )
        self.fs_manager = fs_manager

    def _search_file(
        self,
        file_path: str,
        pattern: str,
        ignore_case: bool,
        line_numbers: bool,
        context_lines: int,
    ) -> list[dict]:
        """Search for pattern in a single file"""

        try:
            content = self.fs_manager.read_file(file_path)
            lines = content.splitlines()
            flags = re.IGNORECASE if ignore_case else 0
            matches = []

            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, flags):
                    match = {"file": file_path, "line": line.strip()}

                    if line_numbers:
                        match["line_number"] = i

                    if context_lines > 0:
                        before = []
                        after = []

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

    def execute(self, **kwargs) -> str:
        pattern = kwargs["pattern"]
        path = kwargs["path"]
        recursive = kwargs.get("recursive", False)
        ignore_case = kwargs.get("ignore_case", False)
        line_numbers = kwargs.get("line_numbers", True)
        context_lines = kwargs.get("context_lines", 0)
        max_matches = kwargs.get("max_matches", 0)

        try:
            if recursive:
                files = self.fs_manager.list_files(path, recursive=True)
            else:
                file_info = self.fs_manager.file_info(path)
                files = (
                    [path]
                    if file_info["type"] == "file"
                    else self.fs_manager.list_files(path)
                )

            all_matches = []
            match_count = 0

            for file_path in files:
                try:
                    if self.fs_manager.file_info(file_path)["type"] != "file":
                        continue
                except Exception:
                    continue

                matches = self._search_file(
                    file_path, pattern, ignore_case, line_numbers, context_lines
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
