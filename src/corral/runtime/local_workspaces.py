"""Portable workspace copies for local execution without permission enforcement.

Path checks reject links and Windows reparse points, including junctions. They
are not a race-proof privilege boundary; enforced execution must use the pinned
directory descriptors in permissions.py instead.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path
from uuid import uuid4

from corral.runtime import permissions


def _local_only() -> None:
    if permissions.enabled():
        raise RuntimeError("portable workspace copies require local execution")


def _checked_status(path: Path) -> os.stat_result:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or (
        getattr(status, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise OSError(f"workspace paths cannot contain links or reparse points: {path}")
    return status


def _directory(path: str | Path | int) -> Path:
    if isinstance(path, int):
        raise TypeError(
            "local workspace copies require paths, not directory descriptors"
        )
    absolute = Path(os.path.abspath(path))  # noqa: PTH100 - do not resolve links before checking
    for component in (*reversed(absolute.parents), absolute):
        if not stat.S_ISDIR(_checked_status(component).st_mode):
            raise NotADirectoryError(str(component))
    return absolute


def copy_local_workspace(
    source: str | Path | int, destination: str | Path | int
) -> list[str]:
    """Merge regular task files into a local workspace without following links."""
    _local_only()
    source_path = _directory(source)
    destination_path = _directory(destination)
    copied: list[str] = []

    def copy_directory(origin: Path, target: Path, prefix: str = "") -> None:
        for entry in sorted(origin.iterdir()):
            if not prefix and (
                entry.name.startswith(permissions.SCRATCH_PREFIX)
                or entry.name
                in {permissions.NODE_WORKSPACE_DIR, permissions.RESOURCE_WORKSPACE_DIR}
            ):
                continue
            status = _checked_status(entry)
            relative = prefix + entry.name
            if (
                entry.name == "debugger_attach_fifo"
                and stat.S_ISFIFO(status.st_mode)
                and prefix.split("/", 1)[0].startswith("ompi.")
            ):
                continue
            output = target / entry.name
            if stat.S_ISDIR(status.st_mode):
                output.mkdir(exist_ok=True)
                _directory(output)
                copy_directory(entry, output, relative + "/")
            elif stat.S_ISREG(status.st_mode):
                try:
                    output_status = _checked_status(output)
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(output_status.st_mode):
                        raise ValueError("workspace destination must be a regular file")
                # Publish by replacement: overwriting a hard-linked destination
                # must not truncate the file it aliases outside this workspace.
                descriptor, temporary_name = tempfile.mkstemp(dir=target)
                temporary = Path(temporary_name)
                try:
                    with (
                        os.fdopen(descriptor, "wb") as stream,
                        entry.open("rb") as data,
                    ):
                        shutil.copyfileobj(data, stream)
                    temporary.chmod((status.st_mode & 0o777) | 0o660)
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
                copied.append(relative)
            else:
                raise ValueError(
                    f"workspace can contain regular files only: {relative}"
                )

    copy_directory(source_path, destination_path)
    return copied


def create_local_node_workspace(parent: str, source: str) -> str:
    """Create a local AI Scientist branch without requiring Unix-only APIs."""
    _local_only()
    parent_path = _directory(parent)
    source_path = _directory(source)
    if not source_path.is_relative_to(parent_path):
        raise ValueError("node source must belong to its parent workspace")
    nodes = parent_path / permissions.NODE_WORKSPACE_DIR
    nodes.mkdir(exist_ok=True)
    _directory(nodes)
    node = nodes / uuid4().hex
    node.mkdir()
    copy_local_workspace(source_path, node)
    return str(node)
