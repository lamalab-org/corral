import json

import fsspec

from corral.base import Tool, ToolArgument


class FSManager:
    """A file-system abstraction layer using fsspec.
    This object is created with a given protocol (e.g., "file", "s3", "ftp")"""

    def __init__(self, protocol: str = "file", **kwargs):
        self.protocol = protocol
        self.fs = fsspec.filesystem(protocol, **kwargs)

    def list_files(self, path: str, recursive: bool = False) -> list[str]:
        """List files in a directory"""
        try:
            return self.fs.ls(path, detail=False, recursive=recursive)
        except Exception as e:
            raise RuntimeError(f"Error listing files: {e}") from e

    def read_file(self, path: str) -> str:
        """Read contents of a file"""
        try:
            with self.fs.open(path, "r") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading files: {e}") from e

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file"""
        try:
            with self.fs.open(path, "w") as f:
                f.write(content)
        except Exception as e:
            raise RuntimeError(f"Error writing files: {e}") from e

    def file_info(self, path: str) -> dict:
        """Get file information"""
        try:
            return self.fs.info(path)
        except Exception as e:
            raise RuntimeError(f"Error getting file info: {e}") from e

    def copy_file(self, source: str, destination: str) -> None:
        try:
            self.fs.copy(source, destination)
        except Exception as e:
            raise RuntimeError(
                f"Error copying from {source} to {destination}: {e}"
            ) from e

    def move_file(self, source: str, destination: str) -> None:
        try:
            # fsspec does not always provide a move method; if not, copy then remove.
            self.fs.mv(source, destination)
        except Exception:
            self.copy_file(source, destination)
            self.fs.rm(source)


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
    """Tool for copying a files."""

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
