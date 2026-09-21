"""Durable, action-keyed synchronization with the Modal MD worker.

The local journal is kept beside (not inside) the Corral workspace so its
protocol state does not become an agent-visible workspace artifact. The remote
run and its successful attempts are retained for recovery and later pruning.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import modal
from corral_md.workspace import ASSET_DIRECTORIES

from corral.core.transition import ToolRecoveryPending
from corral.runtime.permissions import NODE_WORKSPACE_DIR, SCRATCH_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterator

_ROOT = PurePosixPath("/corral/runs")
_MOUNT = PurePosixPath("/results")
_SCHEMA = 1
_PINNED_RELEASE: ContextVar[str | None] = ContextVar("corral_md_release", default=None)
_PINNED_APP: ContextVar[str | None] = ContextVar("corral_md_app", default=None)
_PINNED_VOLUME: ContextVar[str | None] = ContextVar("corral_md_volume", default=None)
_RECOVERY_SNAPSHOT: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "corral_md_recovery_snapshot", default=None
)


class _RemoteToolFailed(RuntimeError):
    """A completed simulation error, as opposed to interrupted recovery."""


def configured_release_id() -> str | None:
    """Read the selected release when an execution is first committed."""
    release = os.getenv("CORRAL_MD_RELEASE_ID")
    if not release:
        release_file = Path(__file__).with_name("release.json")
        if release_file.is_file():
            release = json.loads(release_file.read_text())["release_id"]
    return _identifier(release, "release ID") if release else None


def configured_app_name(release_id: str) -> str:
    release_file = Path(__file__).with_name("release.json")
    if release_file.is_file():
        record = json.loads(release_file.read_text())
        if record.get("release_id") == release_id and record.get("app_name"):
            return record["app_name"]
    return _app_name(release_id)


def configured_volume_name(release_id: str) -> str:
    release_file = Path(__file__).with_name("release.json")
    if release_file.is_file():
        record = json.loads(release_file.read_text())
        if record.get("release_id") == release_id and record.get("volume_name"):
            return record["volume_name"]
    return os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations")


def configured_asset_volume_name(name: str) -> str:
    """Resolve catalog validation against the selected release's asset Volumes."""
    release = _PINNED_RELEASE.get() or configured_release_id()
    release_file = Path(__file__).with_name("release.json")
    if release_file.is_file():
        record = json.loads(release_file.read_text())
        if record.get("release_id") == release:
            return record.get("asset_volumes", {}).get(name, name)
    if _PINNED_RELEASE.get():
        base = _read_json(
            modal.Volume.from_name("corral-md-bases"),
            PurePosixPath("/corral/releases")
            / _identifier(release, "release ID")
            / "base.json",
        )
        if base is None or base.get("release_id") != release:
            raise RuntimeError("Cannot resolve assets for the pinned MD release")
        return base.get("asset_volumes", {}).get(name, name)
    # Releases predating asset versioning used these literal Volume names.
    return name


@contextmanager
def pinned_release(
    release_id: str | None,
    app_name: str | None = None,
    volume_name: str | None = None,
):
    """Bind a release from Corral's durable ExecutionStarted metadata."""
    token = _PINNED_RELEASE.set(release_id)
    app_token = _PINNED_APP.set(app_name)
    volume_token = _PINNED_VOLUME.set(volume_name)
    try:
        yield
    finally:
        _PINNED_RELEASE.reset(token)
        _PINNED_APP.reset(app_token)
        _PINNED_VOLUME.reset(volume_token)


@contextmanager
def recovery_snapshot(workspace_state: Any, workspace_manager: Any):
    """Expose Corral's durable workspace revision to one tool invocation."""
    token = _RECOVERY_SNAPSHOT.set((workspace_state, workspace_manager))
    try:
        yield
    finally:
        _RECOVERY_SNAPSHOT.reset(token)


def _app_name(release_id: str) -> str:
    override = os.getenv("CORRAL_MD_MODAL_APP")
    suffix = os.getenv("SIMAGENT_NAME", "").strip("-")
    prefix = override or ("simagent" + (f"-{suffix}" if suffix else ""))
    return f"{prefix}-{release_id}"


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


def _identifier(value: str, label: str) -> str:
    if (
        not value
        or len(value) > 128
        or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in value
        )
    ):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _workspace(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"MD workspace must be a regular directory: {path}")
    root = source.resolve()
    for entry in _workspace_entries(root, recursive=True):
        if entry.relative_to(root).parts[0] in ASSET_DIRECTORIES:
            raise ValueError(
                "Shared asset directories cannot be stored in the writable workspace"
            )
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError(f"Unsafe workspace entry: {entry.relative_to(root)}")
    return root


def _input(root: Path, supplied: str, *, require_file: bool = True) -> PurePosixPath:
    candidate = Path(supplied)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        lexical_relative = candidate.relative_to(root)
    except ValueError:
        pass
    else:
        if lexical_relative.parts and _is_runtime_name(lexical_relative.parts[0]):
            raise ValueError(
                f"MD input cannot name a runtime workspace path: {supplied}"
            )
    if candidate.is_symlink():
        raise ValueError(f"MD input cannot be a symlink: {supplied}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"MD input must be inside the current workspace: {supplied}")
    relative = resolved.relative_to(root)
    if relative.parts and _is_runtime_name(relative.parts[0]):
        raise ValueError(f"MD input cannot name a runtime workspace path: {supplied}")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"MD input file was not found: {supplied}")
    return PurePosixPath(relative.as_posix())


def _hash_file(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _files(root: Path) -> dict[str, dict[str, str | int]]:
    _workspace(root)
    return {
        path.relative_to(root).as_posix(): _hash_file(path)
        for path in sorted(_workspace_entries(root, recursive=True))
        if path.is_file()
    }


def _safe_relative(path: str) -> Path:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError(f"Unsafe remote workspace path: {path!r}")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"Unsafe remote workspace path: {path!r}")
    if parsed.parts[0] in ASSET_DIRECTORIES:
        raise ValueError(f"Remote output cannot replace a read-only asset: {path!r}")
    if _is_runtime_name(parsed.parts[0]):
        raise ValueError(f"Remote output cannot replace a runtime path: {path!r}")
    return Path(*parsed.parts)


def _stage_workspace(workspace: Path, destination: Path) -> None:
    """Copy task files without copying private runtime directories."""
    destination.mkdir()
    for entry in _workspace_entries(workspace):
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


def _publish_workspace(staged_workspace: Path, workspace: Path) -> None:
    """Replace task files in place while preserving private runtime state."""
    _workspace(staged_workspace)
    staged_entries = list(staged_workspace.iterdir())
    if any(_is_runtime_name(entry.name) for entry in staged_entries):
        raise ValueError("Staged workspace contains a private runtime path")

    backup = staged_workspace.parent / "backup"
    backup.mkdir()
    published: list[Path] = []
    try:
        for entry in list(_workspace_entries(workspace)):
            entry.replace(backup / entry.name)
        for entry in staged_entries:
            target = entry.replace(workspace / entry.name)
            published.append(target)
    except Exception:
        for entry in reversed(published):
            entry.replace(staged_workspace / entry.name)
        for entry in list(backup.iterdir()):
            entry.replace(workspace / entry.name)
        raise


def _read_json(volume: Any, path: PurePosixPath) -> dict[str, Any] | None:
    try:
        payload = b"".join(volume.read_file(str(path)))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid Modal manifest at {path}")
    return value


def _journal_path(root: Path) -> Path:
    directory = root.parent / ".corral-modal"
    if directory.is_symlink():
        raise ValueError("Modal journal directory cannot be a symlink")
    directory.mkdir(exist_ok=True)
    return directory / f"{root.name}.json"


def _save_journal(path: Path, journal: dict[str, Any]) -> None:
    payload = (json.dumps(journal, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".modal-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _restore_snapshot(root: Path) -> None:
    bound = _RECOVERY_SNAPSHOT.get()
    if bound is None or bound[1] is None:
        raise RuntimeError(
            "Remote run was lost and no Corral workspace snapshot is available"
        )
    workspace_state, manager = bound
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"{SCRATCH_PREFIX}restore-", dir=root)
    )
    staged = temporary_root / "workspace"
    try:
        asyncio.run(manager.materialize(workspace_state, staged))
        _publish_workspace(staged, root)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _load_journal(
    root: Path,
    requested_release: str | None,
    requested_app: str | None,
    requested_volume: str | None,
) -> tuple[Path, dict[str, Any]]:
    path = _journal_path(root)
    if path.is_symlink():
        raise ValueError("Modal journal cannot be a symlink")
    if path.exists():
        journal = json.loads(path.read_text())
        if journal.get("schema") != _SCHEMA or journal.get("workspace") != str(root):
            raise RuntimeError(f"Invalid Modal journal: {path}")
        if requested_release and requested_release != journal["release_id"]:
            raise RuntimeError("Active execution is pinned to a different MD release")
        if requested_app and requested_app != journal.get("app_name"):
            raise RuntimeError("Active execution is pinned to a different MD app")
        if requested_volume and requested_volume != journal.get("volume_name"):
            raise RuntimeError("Active execution is pinned to a different MD Volume")
        return path, journal
    release_id = requested_release or _PINNED_RELEASE.get() or configured_release_id()
    if not release_id:
        raise RuntimeError("No MD release selected; deploy with modal_app/release.py")
    _identifier(release_id, "release ID")
    journal = {
        "schema": _SCHEMA,
        "workspace": str(root),
        "release_id": release_id,
        "app_name": requested_app or configured_app_name(release_id),
        "volume_name": requested_volume or configured_volume_name(release_id),
        "actions": {},
    }
    _save_journal(path, journal)
    return path, journal


def _sync_inputs(
    volume: Any,
    root: Path,
    remote_workspace: PurePosixPath,
    previous: dict[str, Any],
) -> dict[str, dict[str, str | int]]:
    local = _files(root)
    old = previous.get("files", {})
    for name in old:
        _safe_relative(name)
    for name in sorted(old.keys() - local.keys()):
        volume.remove_file(str(remote_workspace / name))
    directories = set(previous.get("directories", []))
    directories.update(
        parent.as_posix()
        for name in old
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    )
    for name in sorted(local.keys() & directories):
        # Removing files leaves their parent directories behind on a Volume.
        # Clear a directory that is being replaced with a regular file.
        volume.remove_file(str(remote_workspace / _safe_relative(name)), recursive=True)
    changed = [name for name in local if local[name] != old.get(name)]
    if changed:
        with volume.batch_upload(force=True) as upload:
            for name in sorted(changed):
                upload.put_file(
                    str(root / _safe_relative(name)), str(remote_workspace / name)
                )
    return local


def _sync_result(
    volume: Any,
    root: Path,
    remote_run: PurePosixPath,
    result: dict[str, Any],
    expected_local: dict[str, Any] | None,
) -> int:
    remote_workspace = PurePosixPath(result.get("workspace", ""))
    mounted_run = _MOUNT / remote_run.relative_to("/")
    attempt = mounted_run / "attempts" / result["action_id"]
    if result.get("attempt_id") is not None:
        attempt /= _identifier(result["attempt_id"], "attempt ID")
    expected_workspace = attempt / "workspace"
    if remote_workspace != expected_workspace:
        raise RuntimeError("Modal result points outside its action workspace")
    files = result.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Modal result has no file manifest")
    for name, ref in files.items():
        _safe_relative(name)
        if (
            not isinstance(ref, dict)
            or not isinstance(ref.get("size"), int)
            or ref["size"] < 0
            or not isinstance(ref.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        ):
            raise RuntimeError(f"Invalid Modal file manifest entry: {name}")
    current = _files(root)
    if current == files:
        return 0
    if expected_local is not None and current != expected_local:
        raise RuntimeError("Local workspace changed during the Modal call")
    temporary_root = Path(tempfile.mkdtemp(prefix=f"{SCRATCH_PREFIX}modal-", dir=root))
    staged = temporary_root / "workspace"
    downloaded = 0
    try:
        _stage_workspace(root, staged)
        for name in sorted(current.keys() - files.keys()):
            (staged / _safe_relative(name)).unlink()
        for name, ref in sorted(files.items()):
            if current.get(name) == ref:
                continue
            target = staged / _safe_relative(name)
            if target.is_dir():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                volume_path = (
                    PurePosixPath("/") / remote_workspace.relative_to(_MOUNT) / name
                )
                for chunk in volume.read_file(str(volume_path)):
                    stream.write(chunk)
            if _hash_file(target) != ref:
                raise RuntimeError(f"Modal output failed checksum: {name}")
            downloaded += 1
        if _files(root) != current:
            raise RuntimeError("Local workspace changed during Modal download")
        _publish_workspace(staged, root)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return downloaded


def _execute_impl(
    workspace: str | Path,
    input_file: str,
    *,
    kind: str,
    args: list[str],
    action_id: str | None,
    release_id: str | None,
    volume: Any | None,
    remote_function: Any | None,
    initializer: Any | None,
    call_factory: Any | None,
    execution_options: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    root = _workspace(workspace)
    action = _identifier(action_id or uuid.uuid4().hex, "action ID")
    journal_path, journal = _load_journal(
        root,
        release_id or _PINNED_RELEASE.get(),
        _PINNED_APP.get(),
        _PINNED_VOLUME.get(),
    )
    release = journal["release_id"]
    app_name = journal["app_name"]
    run_id = _identifier(root.name, "run ID")
    remote_run = _ROOT / run_id
    action_manifest = remote_run / "actions" / f"{action}.json"
    failure_manifest = remote_run / "actions" / f"{action}.failure.json"
    call_manifest = remote_run / "actions" / f"{action}.call.json"
    volume = volume or modal.Volume.from_name(journal["volume_name"])
    if remote_function is None:
        function_name = {
            "lammps": "run_lammps",
            "python": "run_python_gpu",
            "python_cpu": "run_python_cpu",
            "shell": "run_shell",
            "verified_md": "run_verified_md",
        }[kind]
        remote_function = modal.Function.from_name(app_name, function_name)
    if initializer is None:
        initializer = modal.Function.from_name(app_name, "prepare_workspace")
    call_factory = call_factory or modal.FunctionCall.from_id
    actions = journal["actions"]
    for other_action, other_record in actions.items():
        if other_action != action and other_record.get("status") not in {
            "completed",
            "failed",
        }:
            raise ToolRecoveryPending(
                f"Resume Modal action {other_action} before starting {action}"
            )
    record = actions.get(action)
    prior_arguments = record.get("arguments") if record else None
    if record is not None and _read_json(volume, remote_run / "run.json") is None:
        # A lost Volume directory cannot provide the old result or attempt.
        # Discard only this action's dispatch record and rebuild from Corral's
        # last committed workspace revision before starting the tool again.
        if _files(root) != record.get("local_before"):
            _restore_snapshot(root)
        record = None
        actions.pop(action)
        _save_journal(journal_path, journal)
    relative_input = _input(root, input_file, require_file=False)
    arguments = {"kind": kind, "input": relative_input.as_posix(), "args": args}
    if execution_options:
        arguments["execution_options"] = execution_options
    if prior_arguments is not None and prior_arguments != arguments:
        raise RuntimeError("An action ID was reused with different MD arguments")
    result = _read_json(volume, action_manifest)
    if result is None:
        failure = _read_json(volume, failure_manifest)
        superseded = record.get("superseded_call_ids", []) if record else []
        if failure is not None and failure.get("call_id") in superseded:
            failure = None
        if failure is not None:
            if record is not None:
                record["status"] = "failed"
                _save_journal(journal_path, journal)
            raise _RemoteToolFailed(
                f"Modal {kind} failed; diagnostics retained at {failure_manifest}: {failure.get('error')}"
            )
        if record is None or record.get("status") == "retryable":
            if record is not None and _files(root) != record["local_before"]:
                _restore_snapshot(root)
                if _files(root) != record["local_before"]:
                    raise ToolRecoveryPending(
                        "The saved workspace does not match this Modal action's inputs"
                    )
            _input(root, input_file, require_file=kind != "shell")
            run_state = initializer.remote(run_id, release)
            if (
                run_state.get("schema") != 1
                or run_state.get("release_id") != release
                or not isinstance(run_state.get("files"), dict)
                or not isinstance(run_state.get("head"), str)
            ):
                raise RuntimeError("Remote run has a different MD release")
            remote_workspace = remote_run / "workspace"
            local_before = _sync_inputs(volume, root, remote_workspace, run_state)
            if record is not None and record.get("call_id"):
                superseded = [*superseded, record["call_id"]]
            record = {
                "arguments": arguments,
                "local_before": local_before,
                "status": "dispatching",
                "superseded_call_ids": superseded,
            }
            actions[action] = record
            _save_journal(journal_path, journal)
            call = remote_function.spawn(
                run_id,
                action,
                relative_input.as_posix(),
                str(root),
                args,
                release,
                run_state.get("head"),
                local_before,
                **(
                    {"execution_options": execution_options}
                    if execution_options
                    else {}
                ),
            )
            record["call_id"] = call.object_id
            record["status"] = "running"
            _save_journal(journal_path, journal)
        else:
            call_id = record.get("call_id")
            if not call_id:
                # The worker records its call ID before doing any computation.
                # Never dispatch a second ambiguous call into the same action.
                for _ in range(30):
                    remote_call = _read_json(volume, call_manifest)
                    if (
                        remote_call is not None
                        and remote_call["call_id"] not in superseded
                    ):
                        call_id = remote_call["call_id"]
                        record["call_id"] = call_id
                        _save_journal(journal_path, journal)
                        break
                    time.sleep(1)
                if not call_id:
                    raise ToolRecoveryPending(
                        "Modal dispatch outcome is unknown; retry recovery after the worker starts"
                    )
            call = call_factory(call_id)
        try:
            call.get()
        except Exception as exc:
            result = _read_json(volume, action_manifest)
            failure = _read_json(volume, failure_manifest)
            if failure is not None and failure.get("call_id") in superseded:
                failure = None
            if result is None and failure is not None:
                record["status"] = "failed"
                _save_journal(journal_path, journal)
                raise _RemoteToolFailed(
                    f"Modal {kind} failed; diagnostics retained at {failure_manifest}: {failure.get('error')}"
                ) from exc
            if result is None:
                # These exceptions come from a settled FunctionCall output.
                # Transport errors and polling timeouts do not establish that
                # the worker stopped, so retain and reattach to those calls.
                if isinstance(
                    exc,
                    modal.exception.RemoteError
                    | modal.exception.FunctionTimeoutError
                    | modal.exception.InternalFailure
                    | modal.exception.OutputExpiredError,
                ):
                    record["status"] = "retryable"
                    record["last_error"] = str(exc)
                    _save_journal(journal_path, journal)
                raise ToolRecoveryPending(
                    f"Modal call {call.object_id} has no result manifest; resume action {action}: {exc}"
                ) from exc
    result = result or _read_json(volume, action_manifest)
    if (
        result is None
        or result.get("action_id") != action
        or result.get("release_id") != release
        or result.get("kind") != kind
    ):
        raise RuntimeError("Modal action result manifest is missing or mismatched")
    if not isinstance(result.get("revision"), int) or result["revision"] < 1:
        raise RuntimeError("Modal action result has no valid workspace revision")
    expected = record.get("local_before") if record else None
    try:
        count = _sync_result(volume, root, remote_run, result, expected)
    except Exception as exc:
        raise ToolRecoveryPending(
            f"Modal action {action} completed remotely; resume its output sync: {exc}"
        ) from exc
    if record is not None:
        record["status"] = "completed"
        record.pop("local_before", None)
        _save_journal(journal_path, journal)
    return root / "output" / relative_input.with_suffix(".log").name, count


def _execute(*args: Any, **kwargs: Any) -> tuple[Path, int]:
    try:
        return _execute_impl(*args, **kwargs)
    except (ToolRecoveryPending, _RemoteToolFailed, ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise ToolRecoveryPending(
            f"Modal workspace recovery is pending: {exc}"
        ) from exc


def run_lammps_in_modal(
    workspace: str | Path,
    input_file: str,
    *,
    action_id: str | None = None,
    release_id: str | None = None,
    volume: Any | None = None,
    remote_function: Any | None = None,
    initializer: Any | None = None,
    call_factory: Any | None = None,
    job_id: str | None = None,
) -> tuple[Path, int]:
    """Run LAMMPS once per Corral action and recover its committed outputs."""
    return _execute(
        workspace,
        input_file,
        kind="lammps",
        args=[],
        action_id=action_id or job_id,
        release_id=release_id,
        volume=volume,
        remote_function=remote_function,
        initializer=initializer,
        call_factory=call_factory,
    )


def run_python_in_modal(
    workspace: str | Path,
    script_file: str,
    args: list[str] | None = None,
    *,
    action_id: str | None = None,
    release_id: str | None = None,
    volume: Any | None = None,
    remote_function: Any | None = None,
    initializer: Any | None = None,
    call_factory: Any | None = None,
    job_id: str | None = None,
    use_gpu: bool = True,
    timeout: int = 600,
    working_dir: str = "/workspace",
) -> int:
    """Run isolated CPU/GPU Python and recover its committed outputs."""
    _, count = _execute(
        workspace,
        script_file,
        kind="python" if use_gpu else "python_cpu",
        args=[str(arg) for arg in args or []],
        action_id=action_id or job_id,
        release_id=release_id,
        volume=volume,
        remote_function=remote_function,
        initializer=initializer,
        call_factory=call_factory,
        execution_options={"timeout": timeout, "working_dir": working_dir},
    )
    return count


def run_shell_in_modal(
    workspace: str | Path,
    command: str,
    *,
    timeout: int = 120,
    max_output_chars: int = 20_000,
    action_id: str | None = None,
) -> int:
    """Run a shell action without creating or mutating an input script."""
    _, count = _execute(
        workspace,
        str(Path(workspace).resolve() / "__terminal__"),
        kind="shell",
        args=[command],
        action_id=action_id,
        release_id=None,
        volume=None,
        remote_function=None,
        initializer=None,
        call_factory=None,
        execution_options={"timeout": timeout, "max_output_chars": max_output_chars},
    )
    return count


def run_verified_md_in_modal(workspace, config_file, *, action_id=None):
    """Use normal action recovery and synchronization for controlled dynamics."""
    return _execute(
        workspace,
        config_file,
        kind="verified_md",
        args=[],
        action_id=action_id,
        release_id=None,
        volume=None,
        remote_function=None,
        initializer=None,
        call_factory=None,
    )


__all__ = [
    "run_lammps_in_modal",
    "run_python_in_modal",
    "run_shell_in_modal",
    "run_verified_md_in_modal",
]
