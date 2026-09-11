"""Synchronize a local Corral workspace around a remote Modal LAMMPS run."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import modal

from corral.runtime.permissions import NODE_WORKSPACE_DIR, SCRATCH_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGER = logging.getLogger(__name__)
_DEFAULT_APP_NAME = "simagent"
_DEFAULT_VOLUME_NAME = "simulations"
_REMOTE_VOLUME_ROOT = PurePosixPath("/corral/jobs")
_REMOTE_MOUNT_ROOT = PurePosixPath("/results")


def _modal_app_name() -> str:
    configured = os.getenv("CORRAL_MD_MODAL_APP")
    if configured:
        return configured

    suffix = os.getenv("SIMAGENT_NAME", "")
    if not suffix:
        return _DEFAULT_APP_NAME
    return f"{_DEFAULT_APP_NAME}{suffix if suffix.startswith('-') else f'-{suffix}'}"


def _is_runtime_name(name: str) -> bool:
    return name.startswith(SCRATCH_PREFIX) or name == NODE_WORKSPACE_DIR


def _workspace_entries(workspace: Path, *, recursive: bool = False) -> Iterator[Path]:
    """Visit task files without entering private worker or node directories."""
    for entry in workspace.iterdir():
        if _is_runtime_name(entry.name):
            continue
        yield entry
        if recursive and entry.is_dir() and not entry.is_symlink():
            yield from entry.rglob("*")


def _validate_workspace(workspace: str | Path) -> Path:
    workspace_path = Path(workspace)
    if workspace_path.is_symlink() or not workspace_path.is_dir():
        raise ValueError(f"LAMMPS workspace must be a regular directory: {workspace}")

    root = workspace_path.resolve()
    for entry in _workspace_entries(root, recursive=True):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise ValueError(f"LAMMPS workspace cannot contain symlinks: {relative}")
        if not entry.is_dir() and not entry.is_file():
            raise ValueError(
                f"LAMMPS workspace can contain regular files only: {relative}"
            )
    return root


def _resolve_input(workspace: Path, input_file: str) -> tuple[Path, PurePosixPath]:
    supplied = Path(input_file)
    local_input = supplied if supplied.is_absolute() else workspace / supplied
    local_input = local_input.resolve()
    if workspace != local_input and workspace not in local_input.parents:
        raise ValueError(
            f"LAMMPS input must be inside the current workspace: {input_file}"
        )
    relative = local_input.relative_to(workspace)
    if relative.parts and _is_runtime_name(relative.parts[0]):
        raise ValueError(
            f"LAMMPS input cannot name a runtime workspace path: {input_file}"
        )
    if local_input.is_symlink() or not local_input.is_file():
        raise FileNotFoundError(f"LAMMPS input file was not found: {input_file}")
    return local_input, PurePosixPath(relative.as_posix())


def _entry_kind(entry: Any) -> str:
    entry_type = getattr(entry, "type", None)
    name = getattr(entry_type, "name", "")
    if name:
        return name.lower()
    if entry_type == 1:
        return "file"
    if entry_type == 2:
        return "directory"
    return str(entry_type).lower()


def _entry_relative_path(entry: Any, remote_directory: PurePosixPath) -> Path:
    entry_path = PurePosixPath(f"/{str(entry.path).lstrip('/')}")
    remote_root = PurePosixPath(f"/{str(remote_directory).lstrip('/')}")
    try:
        relative = entry_path.relative_to(remote_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Modal returned a file outside the job workspace: {entry.path!r}"
        ) from exc
    if str(relative) in {"", "."}:
        return Path()
    if any(part in {"", ".", ".."} for part in relative.parts) or _is_runtime_name(
        relative.parts[0]
    ):
        raise RuntimeError(f"Modal returned an unsafe workspace path: {entry.path!r}")
    return Path(*relative.parts)


def _download_workspace(
    volume: Any,
    remote_directory: PurePosixPath,
    destination: Path,
) -> int:
    entries = volume.listdir(str(remote_directory), recursive=True)
    downloaded = 0
    for entry in sorted(entries, key=lambda item: str(item.path)):
        relative = _entry_relative_path(entry, remote_directory)
        if relative == Path():
            continue

        target = destination / relative
        kind = _entry_kind(entry)
        if kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
            continue
        if kind != "file":
            raise RuntimeError(
                f"Modal job workspace contains unsupported {kind}: {entry.path!r}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with target.open("wb") as stream:
            for chunk in volume.read_file(str(entry.path)):
                stream.write(chunk)
                size += len(chunk)
        expected_size = getattr(entry, "size", None)
        if expected_size is not None and size != expected_size:
            raise RuntimeError(
                f"Downloaded {entry.path!r} with {size} bytes; expected {expected_size}"
            )
        downloaded += 1
    return downloaded


def _publish_workspace(staged_workspace: Path, workspace: Path) -> None:
    # The workspace may be a bind mount with a non-writable parent. Preserve its
    # inode, ownership and mode, and leave active worker scratch in place.
    backup = staged_workspace.parent / "backup"
    backup.mkdir()
    published: list[Path] = []
    try:
        for entry in list(_workspace_entries(workspace)):
            entry.replace(backup / entry.name)
        for entry in list(staged_workspace.iterdir()):
            target = entry.replace(workspace / entry.name)
            published.append(target)
    except Exception:
        for entry in reversed(published):
            entry.replace(staged_workspace / entry.name)
        for entry in list(backup.iterdir()):
            entry.replace(workspace / entry.name)
        raise


def run_lammps_in_modal(
    workspace: str | Path,
    input_file: str,
    *,
    volume: Any | None = None,
    remote_function: Any | None = None,
    job_id: str | None = None,
) -> tuple[Path, int]:
    """Run LAMMPS on Modal and synchronize its outputs into the local workspace.

    Local task files are uploaded to a unique directory in the
    `simulations` Volume. The deployed `run_lammps` function executes there,
    commits its writes, and this function downloads the complete directory into
    staging inside the workspace before replacing its task files in place.
    """

    local_workspace = _validate_workspace(workspace)
    local_input, relative_input = _resolve_input(local_workspace, input_file)
    identifier = job_id or uuid.uuid4().hex
    if not identifier or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
        for char in identifier.lower()
    ):
        raise ValueError(f"Invalid Modal job id: {identifier!r}")

    remote_volume_directory = _REMOTE_VOLUME_ROOT / identifier
    remote_workspace = _REMOTE_MOUNT_ROOT / remote_volume_directory.relative_to("/")
    remote_input = remote_workspace / relative_input
    log_name = f"{local_input.stem}.log"
    local_log = local_input.with_name(log_name)

    if volume is None:
        volume_name = os.getenv("CORRAL_MD_MODAL_VOLUME", _DEFAULT_VOLUME_NAME)
        volume = modal.Volume.from_name(volume_name)
    if remote_function is None:
        remote_function = modal.Function.from_name(_modal_app_name(), "run_lammps")

    with volume.batch_upload(force=True) as upload:
        for entry in _workspace_entries(local_workspace):
            remote_path = str(remote_volume_directory / entry.name)
            if entry.is_dir():
                upload.put_directory(str(entry), remote_path)
            else:
                upload.put_file(str(entry), remote_path)

    remote_error: Exception | None = None
    try:
        remote_function.remote(
            str(remote_input),
            log_name,
            str(local_workspace),
            str(remote_workspace),
        )
    except Exception as exc:  # preserve diagnostic files from failed LAMMPS runs
        remote_error = exc

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"{SCRATCH_PREFIX}modal-", dir=local_workspace)
    )
    staged_workspace = temporary_root / "workspace"
    staged_workspace.mkdir()
    try:
        downloaded = _download_workspace(
            volume,
            remote_volume_directory,
            staged_workspace,
        )
        _publish_workspace(staged_workspace, local_workspace)
    except Exception as sync_error:
        detail = (
            f" after the Modal LAMMPS call failed with {remote_error}"
            if remote_error is not None
            else ""
        )
        raise RuntimeError(
            f"Failed to synchronize Modal LAMMPS outputs back to "
            f"{str(local_workspace)!r}{detail}: {sync_error}"
        ) from sync_error
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    try:
        volume.remove_file(str(remote_volume_directory), recursive=True)
    except Exception as cleanup_error:  # outputs are already durable locally
        _LOGGER.warning(
            "Could not remove temporary Modal workspace %s: %s",
            remote_volume_directory,
            cleanup_error,
        )

    if remote_error is not None:
        raise RuntimeError(
            f"LAMMPS failed on Modal; remote diagnostics were synchronized to "
            f"{str(local_workspace)!r}: {remote_error}"
        ) from remote_error
    return local_log, downloaded


__all__ = ["run_lammps_in_modal"]
