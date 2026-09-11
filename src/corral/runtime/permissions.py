"""Unix identities for model-controlled execution in Docker trials.

The root controller owns source and checkpoints. Workers permanently drop all
three user/group IDs before invoking agent or tool code. Only JSON is accepted
back from workers; untrusted pickle data must never be loaded by the controller.
"""

from __future__ import annotations

import ctypes
import io
import itertools
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import cloudpickle

DENIED = "Permission denied: access outside the trial workspace is not permitted"
POLICY_VERSION = "workspace-root-v1"
SCRATCH_PREFIX = ".corral-runtime-"
NODE_WORKSPACE_DIR = ".corral-nodes"
_enabled = False
_identities = itertools.count(20000)
_groups: dict[Path, int] = {}
_node_parents: dict[Path, Path] = {}
_lock = threading.Lock()
_root: Path | None = None


def _restore_prompt(attributes: dict[str, Any]):
    from promptstore import Prompt  # - load only during serialization

    return Prompt(**attributes)


def _reduce_prompt(prompt: Any):
    return _restore_prompt, (
        {
            key: value
            for key, value in vars(prompt).items()
            if key not in {"_template", "variables"}
        },
    )


def _restore_hooks(callbacks: Any):
    from corral.agents.hooks import (  # - avoid session import cycle
        AgentHooks,
    )

    hooks = AgentHooks()
    hooks._hooks = callbacks
    return hooks


def _reduce_hooks(hooks: Any):
    with hooks._lock:
        return _restore_hooks, (dict(hooks._hooks),)


@cache
def private_controller_types() -> tuple[type, ...]:
    """Preload serialization guards before a worker loses source access."""
    from corral.agents.session import AgentSession
    from corral.core.commit import Commit
    from corral.core.environment import Environment
    from corral.core.state import ExecutionState
    from corral.core.task import TaskDefinition

    return AgentSession, ExecutionState, Environment, TaskDefinition, Commit


def serialize(value: Any) -> bytes:
    """Transfer public configuration, refusing captured controller objects."""
    from promptstore import Prompt  # - load only during serialization

    from corral.agents.hooks import AgentHooks

    private_types = private_controller_types()

    class WorkerPickler(cloudpickle.CloudPickler):
        def persistent_id(self, obj: Any) -> None:
            if isinstance(obj, private_types):
                raise ValueError(
                    f"private controller object cannot enter a worker: {type(obj).__name__}"
                )

        dispatch_table: ClassVar[dict] = {
            **cloudpickle.CloudPickler.dispatch_table,
            Prompt: _reduce_prompt,
            AgentHooks: _reduce_hooks,
        }

    stream = io.BytesIO()
    pickler = WorkerPickler(stream)
    pickler.dump(value)
    return stream.getvalue()


def _open_directory(path: str | Path | int) -> int:
    """Pin a directory without following links in any path component."""
    if isinstance(path, int):
        return os.dup(path)
    absolute = Path(os.path.abspath(path))  # noqa: PTH100 - resolve would follow untrusted links
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _entries(directory: int, prefix: str = ""):
    """Open entries relative to pinned directory FDs; never follow a link."""
    for name in sorted(os.listdir(directory)):
        # SDK homes can contain sockets and executable aliases. They are private
        # scratch within the workspace, not task artifacts or shared task files.
        if not prefix and (
            name.startswith(SCRATCH_PREFIX) or name == NODE_WORKSPACE_DIR
        ):
            continue
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            status = os.fstat(descriptor)
            relative = prefix + name
            if not (stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)):
                raise ValueError(
                    f"workspace can contain regular files only: {relative}"
                )
            yield relative, descriptor, status
            if stat.S_ISDIR(status.st_mode):
                yield from _entries(descriptor, relative + "/")
        finally:
            os.close(descriptor)


def copy_workspace(
    source: str | Path | int, destination: str | Path | int
) -> list[str]:
    """Copy authorized workspace data without a privileged symlink race."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    source_fd = _open_directory(source)
    try:
        destination_fd = _open_directory(destination)
        try:
            destination_gid = os.fstat(destination_fd).st_gid
            copied = []
            for relative, descriptor, status in _entries(source_fd):
                parent = os.dup(destination_fd)
                try:
                    parts = relative.split("/")
                    for component in parts[:-1]:
                        next_parent = os.open(component, directory_flags, dir_fd=parent)
                        os.close(parent)
                        parent = next_parent
                    if stat.S_ISDIR(status.st_mode):
                        with suppress(FileExistsError):
                            os.mkdir(parts[-1], mode=0o770, dir_fd=parent)  # noqa: PTH102 - Path.mkdir does not support dir_fd
                        existing = os.open(parts[-1], directory_flags, dir_fd=parent)
                        try:
                            if enabled():
                                os.fchown(existing, 0, destination_gid)
                            os.fchmod(existing, (status.st_mode & 0o777) | 0o2770)
                        finally:
                            os.close(existing)
                    else:
                        target = os.open(
                            parts[-1],
                            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                            0o660,
                            dir_fd=parent,
                        )
                        try:
                            if not stat.S_ISREG(os.fstat(target).st_mode):
                                raise ValueError(
                                    "workspace destination must be a regular file"
                                )
                            # The controller's private umask must not hide newly
                            # promoted outputs from the owning workspace group.
                            # It lacks FSETID, so directory setgid inheritance
                            # cannot be relied on for controller-created files.
                            if enabled():
                                os.fchown(target, 0, destination_gid)
                            os.fchmod(target, (status.st_mode & 0o777) | 0o660)
                            os.ftruncate(target, 0)
                            while chunk := os.read(descriptor, 1024 * 1024):
                                view = memoryview(chunk)
                                while view:
                                    view = view[os.write(target, view) :]
                        finally:
                            os.close(target)
                        copied.append(relative)
                finally:
                    os.close(parent)
            return copied
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def create_node_workspace(parent: str, source: str) -> str:
    """Allocate a sibling node directory and copy only its parent's task files."""
    parent_path = Path(os.path.abspath(parent))  # noqa: PTH100 - do not follow links
    source_path = Path(os.path.abspath(source))  # noqa: PTH100 - do not follow links
    if not source_path.is_relative_to(parent_path):
        raise ValueError("node source must belong to its parent workspace")
    if enabled():
        # The main worker already has this group. Children use the same group
        # so it can inspect their files; their private mounts enforce isolation.
        workspace_identity(parent)
    parent_fd = _open_directory(parent)
    try:
        with suppress(FileExistsError):
            os.mkdir(NODE_WORKSPACE_DIR, mode=0o2770, dir_fd=parent_fd)  # noqa: PTH102 - Path.mkdir does not support dir_fd
        nodes_fd = os.open(
            NODE_WORKSPACE_DIR,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            if enabled():
                os.fchown(nodes_fd, 0, _groups[parent_path])
                os.fchmod(nodes_fd, 0o2770)
            name = uuid4().hex
            os.mkdir(name, mode=0o2770, dir_fd=nodes_fd)  # noqa: PTH102 - Path.mkdir does not support dir_fd
            node_fd = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=nodes_fd
            )
            try:
                copy_workspace(source, node_fd)
            finally:
                os.close(node_fd)
        finally:
            os.close(nodes_fd)
    finally:
        os.close(parent_fd)
    node = parent_path / NODE_WORKSPACE_DIR / name
    if enabled():
        with _lock:
            _node_parents[node] = parent_path
        workspace_identity(node)
    return str(node)


@contextmanager
def snapshot_source(source: str | Path):
    """Give artifact storage an immutable-to-agents, safely opened copy."""
    if _root is None:
        raise RuntimeError("Docker permission enforcement has not been configured")
    with tempfile.TemporaryDirectory(dir=_root) as directory:
        copy_workspace(source, directory)
        yield Path(directory)


def prepare_image() -> None:
    """Protect image data while retaining the public OS/language runtime."""
    for name in (
        "/opt/corral",
        "/root",
        "/home",
        "/var",
        "/run",
        "/srv",
        "/mnt",
        "/media",
        "/boot",
    ):
        directory = Path(name)
        if directory.is_dir() and not directory.is_symlink():
            os.chown(directory, 0, 0)
            directory.chmod(0o700)
    # DNS, certificates, and the dynamic loader need these particular resources;
    # other image configuration is private, including passwd and package config.
    public = (
        "ssl/certs",
        "pki/tls/certs",
        "ca-certificates",
        "alternatives",
        "resolv.conf",
        "hosts",
        "nsswitch.conf",
        "localtime",
        "ld.so.cache",
    )
    generated = {Path("/etc/hosts"), Path("/etc/resolv.conf"), Path("/etc/hostname")}
    for directory, directories, files in os.walk("/etc", followlinks=False):
        for name in (*directories, *files):
            entry = Path(directory) / name
            if not entry.is_symlink() and entry not in generated:
                entry.chmod(entry.stat().st_mode & 0o700)
    Path("/etc").chmod(0o711)
    for name in public:
        entry = Path("/etc") / name
        if not entry.exists():
            continue
        for parent in entry.parents:
            if parent == Path("/etc"):
                break
            parent.chmod(0o711)
        if not entry.is_symlink() and entry not in generated:
            entry.chmod(0o755 if entry.is_dir() else 0o644)
        if entry.is_dir():
            for child in entry.rglob("*"):
                if not child.is_symlink():
                    child.chmod(0o755 if child.is_dir() else 0o644)


def enabled() -> bool:
    return _enabled


def configure(checkpoints: Path) -> None:
    """Enable confinement in the trusted, container-only entry point."""
    global _enabled, _root  # noqa: PLW0603 - one trusted policy per trial process
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(
            "Docker permission enforcement requires a root Linux controller"
        )
    source = Path(__file__).resolve().parents[3]
    if source != Path("/opt/corral"):
        raise RuntimeError("Docker source must be installed privately at /opt/corral")
    for private in (source, checkpoints):
        status = private.stat()
        if status.st_uid != 0 or status.st_mode & 0o077:
            raise RuntimeError(
                f"Private Docker directory must be root-owned mode 0700: {private}"
            )
    if Path("/etc/passwd").stat().st_mode & 0o077:
        raise RuntimeError(
            "Docker image has not been prepared for Unix worker permissions"
        )
    # /tmp and shared memory otherwise provide writable paths outside workspace.
    for private in (Path("/tmp"), Path("/dev/shm")):
        os.chown(private, 0, 0)
        private.chmod(0o700)
    os.umask(0o077)
    Path("/workspace").chmod(0o711)
    _root = checkpoints / "workers"
    _root.mkdir(mode=0o700, exist_ok=True)
    _enabled = True


def workspace_identity(
    workspace: str | Path, *, descriptor: int | None = None
) -> tuple[int, int]:
    root = Path(os.path.abspath(workspace))  # noqa: PTH100 - do not follow links
    if root == Path("/workspace") or not root.is_relative_to("/workspace"):
        raise ValueError("workers require an assigned directory below /workspace")
    if root.is_relative_to("/workspace/.corral-runtime-system"):
        raise ValueError("the worker runtime path is reserved")
    with _lock:
        parent = _node_parents.get(root)
        if root not in _groups:
            if parent is not None:
                _groups[root] = _groups[parent]
            elif any(
                root.is_relative_to(existing) or existing.is_relative_to(root)
                for existing in _groups
            ):
                raise ValueError("nested workspaces must be assigned by the controller")
            else:
                _groups[root] = next(_identities)
        gid = _groups[root]
        uid = next(_identities)
    # Parent directories are traversable, but only the assigned group can enter
    # a materialization. The controller needs DAC_OVERRIDE for hostile file modes.
    descriptor = _open_directory(root if descriptor is None else descriptor)
    try:
        _allow_workspace_group(descriptor, gid)
    finally:
        os.close(descriptor)
    return uid, gid


def _allow_workspace_group(descriptor: int, gid: int) -> None:
    os.fchown(descriptor, 0, gid)
    os.fchmod(descriptor, 0o2770)
    for _name, entry, status in _entries(descriptor):
        os.fchown(entry, 0, gid)
        os.fchmod(
            entry,
            (status.st_mode & 0o777)
            | (0o2770 if stat.S_ISDIR(status.st_mode) else 0o660),
        )


def drop_privileges(
    uid: int, gid: int, workspace: str, *, scratch: str | None = None
) -> None:
    """Irreversibly enter the worker identity before running supplied code."""
    if os.geteuid() != 0 or uid <= 0 or gid <= 0:
        raise RuntimeError("restricted workers must start at the trusted bootstrap")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    if os.getresuid() != (uid, uid, uid) or os.getresgid() != (gid, gid, gid):
        raise RuntimeError("failed to drop worker privileges")
    status = Path("/proc/self/status").read_text()
    for field in ("CapInh", "CapPrm", "CapEff", "CapAmb"):
        if int(status.split(field + ":", 1)[1].splitlines()[0].strip(), 16):
            raise RuntimeError(f"worker retained Linux capabilities: {field}")
    os.umask(0o007)
    os.chdir(workspace)
    if scratch is None:
        scratch = tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=workspace)
    os.environ.update(HOME=scratch, TMPDIR=scratch, XDG_CACHE_HOME=scratch + "/.cache")
    tempfile.tempdir = scratch


def _kill_identity(uid: int) -> None:
    """Stop descendants, including detached processes and racing forks."""
    deadline = time.monotonic() + 5
    while True:
        found = False
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                text = status.read_text()
                fields = text.split("Uid:", 1)[1].splitlines()[0].split()
                if (
                    int(fields[0]) != uid
                    or text.split("State:", 1)[1].split()[0] == "Z"
                ):
                    continue
                found = True
                pid = int(status.parent.name)
                os.kill(pid, signal.SIGSTOP)
                os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not found:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("restricted worker descendants could not be stopped")
        time.sleep(0.01)


def run_worker(
    kind: str, payload: Any, workspace: str, *, cancel: threading.Event | None = None
) -> Any:
    """Run one trusted bootstrap, then accept an unprivileged JSON result."""
    if not enabled() or _root is None:
        raise RuntimeError("restricted workers require Docker permission enforcement")
    descriptor = _open_directory(workspace)
    try:
        return _run_worker(kind, payload, workspace, descriptor, cancel=cancel)
    finally:
        try:
            root = Path(workspace)
            if root in _node_parents:
                # A tool may create mode-0600 outputs. Once it and its children
                # have stopped, restore the main agent's access to node files.
                _allow_workspace_group(descriptor, _groups[root])
        finally:
            os.close(descriptor)


def _run_worker(
    kind: str,
    payload: Any,
    workspace: str,
    workspace_fd: int,
    *,
    cancel: threading.Event | None,
) -> Any:
    uid, gid = workspace_identity(workspace, descriptor=workspace_fd)
    with tempfile.TemporaryDirectory(dir=_root) as directory:
        request = Path(directory) / "input.pkl"
        request.write_bytes(
            serialize((kind, payload, uid, gid, workspace, workspace_fd))
        )
        request.chmod(0o600)
        reader, writer = os.pipe()
        output: list[bytes] = []

        def read_result() -> None:
            with os.fdopen(reader, "rb") as stream:
                output.append(stream.read(64 * 1024 * 1024 + 1))

        thread = threading.Thread(target=read_result, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryFile(dir=_root) as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "corral.runtime._permission_worker",
                        str(request),
                        str(writer),
                    ],
                    stdin=subprocess.DEVNULL,
                    # A write-only pipe is the only log handle inherited by the
                    # worker; never expose the controller's private log file FD.
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    # Environment factories may add private task import roots.
                    # The fresh trusted bootstrap needs the same module paths.
                    env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                    pass_fds=(writer, workspace_fd),
                    start_new_session=True,
                )

                def read_log() -> None:
                    with process.stdout as stream:
                        while chunk := stream.read(64 * 1024):
                            log.write(chunk)

                log_thread = threading.Thread(target=read_log, daemon=True)
                log_thread.start()
                os.close(writer)
                writer = -1
                try:
                    while True:
                        try:
                            process.wait(timeout=0.1)
                            break
                        except subprocess.TimeoutExpired:
                            if cancel is not None and cancel.is_set():
                                _kill_identity(uid)
                                process.kill()
                                process.wait()
                                raise RuntimeError(
                                    "restricted worker cancelled"
                                ) from None
                finally:
                    _kill_identity(uid)
                    thread.join(timeout=5)
                    log_thread.join(timeout=5)
                if log_thread.is_alive():
                    raise RuntimeError("restricted worker log stream did not close")
                if thread.is_alive() or not output or len(output[0]) > 64 * 1024 * 1024:
                    raise RuntimeError("restricted worker returned an invalid response")
                if process.returncode or not output[0]:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 16000))
                    raise RuntimeError(
                        "restricted worker failed: "
                        + log.read().decode(errors="replace")
                    )
                response = json.loads(output[0])
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["result"]
        finally:
            if writer >= 0:
                os.close(writer)


def execute_tool(
    environment: Any, state: Any, tool: Any, arguments: dict[str, Any]
) -> Any:
    """Keep private task logic here; give code workers only public arguments."""
    from corral.backend.background_tools import (  # - avoid tool import cycle
        _CallableTool,
    )

    # Job controls and explicitly trusted scientific functions never evaluate
    # model-controlled code. They alone may receive the execution projection.
    if isinstance(tool, _CallableTool) or getattr(tool, "trusted", False):
        return environment.execute_tool(state, tool, arguments)
    return _run_public_tool(tool, arguments, environment.workspace_path)


def visible_argument_error(tool: Any, arguments: dict[str, Any]) -> str | None:
    """Validate the public schema before preprocessing or injecting private data."""
    from jsonschema import Draft202012Validator

    schema = {
        **tool.params_json_schema,
        "required": [argument.name for argument in tool.arguments if argument.required],
        "additionalProperties": False,
    }
    error = next(Draft202012Validator(schema).iter_errors(arguments), None)
    return None if error is None else f"invalid visible tool arguments: {error.message}"


def _run_public_tool(
    tool: Any,
    arguments: dict[str, Any],
    workspace: str,
    *,
    cancel: threading.Event | None = None,
) -> Any:
    workspace_args = set(getattr(tool, "workspace_args", ()))
    if set(tool.hidden_args) - workspace_args:
        raise PermissionError(
            "restricted tools cannot receive hidden arguments; "
            "private inputs require explicitly trusted task logic"
        )
    public = {
        name: value for name, value in arguments.items() if name not in tool.hidden_args
    }
    if error := visible_argument_error(tool, public):
        raise ValueError(error)
    public.update(dict.fromkeys(workspace_args, workspace))
    operation = getattr(tool, "worker_operation", None)
    if operation == "terminal":
        result = run_worker("terminal", public, workspace, cancel=cancel)
    else:
        result = run_worker("tool", (tool, public), workspace, cancel=cancel)
    return result["content"]


def execute_job(
    tool: Any,
    arguments: dict[str, Any],
    workspace: str,
    *,
    cancel: threading.Event | None = None,
) -> Any:
    """Background jobs obey the same trust classification as foreground tools."""
    if getattr(tool, "trusted", False):
        from corral.core.transition import ToolExecutionResult

        result = tool.execute(**arguments)
        if isinstance(result, ToolExecutionResult):
            if result.environment is not None:
                raise ValueError("stateful task tools must execute in the foreground")
            return result.content
        return result
    return _run_public_tool(tool, arguments, workspace, cancel=cancel)


if __name__ == "__main__":
    if sys.argv[1:] != ["--prepare-image"]:
        raise SystemExit("expected --prepare-image")
    prepare_image()
