"""Build a worker-only filesystem before permanently dropping privileges.

The original container root is never mounted in the jail. Runtime mounts live
under /workspace; conventional runtime paths are aliases into that tree.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

RUNTIME_ROOT = Path("/workspace/.corral-runtime-system")

_CLONE_NEWNS = 0x00020000
_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REMOUNT = 32
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 1 << 18

_PUBLIC_ETC = (
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


def _directory(path: Path, mode: int = 0o755) -> None:
    if not path.parent.exists():
        _directory(path.parent)
    path.mkdir(exist_ok=True)
    # The controller uses a private umask; runtime ancestors must be traversable.
    path.chmod(mode)


def _runtime_roots() -> tuple[Path, ...]:
    roots = [Path(name) for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64")]
    for prefix in (sys.prefix, sys.base_prefix):
        path = Path(prefix).resolve()
        if any(path.is_relative_to(root) for root in roots):
            continue
        if not path.is_relative_to("/opt") or path.is_relative_to("/opt/corral"):
            raise RuntimeError(f"unsupported worker Python installation: {path}")
        roots.append(path)
    return tuple(path for path in roots if path.exists())


def bind_bootstrap_parent(parent_pid: int) -> None:
    """Kill this child if its trusted bootstrap is cancelled or exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot bind worker to its bootstrap")
    # The parent may have exited before prctl installed the death signal.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def enter_workspace(
    workspace: str,
    jail: Path,
    uid: int,
    gid: int,
    *,
    keep_fds: set[int],
    workspace_fd: int | None = None,
) -> str:
    """Relocate the runtime and workspace into a private, read-only root.

    Must run in a freshly forked bootstrap: unshare changes the calling thread's
    filesystem context, so pre-existing threads must not survive this boundary.
    No supplied callable is invoked here, and no workspace file is executed.
    """
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError(
            "worker filesystem setup requires a single-threaded bootstrap"
        )
    roots = _runtime_roots()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    )

    def mount(source, target, filesystem=None, flags=0, options=None):
        arguments = (
            None if source is None else os.fsencode(source),
            os.fsencode(target),
            None if filesystem is None else os.fsencode(filesystem),
            flags,
            None if options is None else os.fsencode(options),
        )
        if libc.mount(*arguments):
            raise OSError(
                ctypes.get_errno(), "cannot construct worker filesystem", str(target)
            )

    def bind(source: Path, target: Path, *, readonly: bool) -> None:
        if source.is_dir():
            _directory(target)
        else:
            _directory(target.parent)
            target.touch()
        # Deliberately do not import nested mounts from the original image.
        mount(source, target, flags=_MS_BIND)
        if readonly:
            mount(
                None,
                target,
                flags=_MS_BIND | _MS_REMOUNT | _MS_RDONLY | _MS_NOSUID | _MS_NODEV,
            )

    if libc.unshare(_CLONE_NEWNS):
        raise OSError(ctypes.get_errno(), "cannot create worker mount namespace")
    mount(None, "/", flags=_MS_REC | _MS_PRIVATE)
    _directory(jail, 0o700)
    mount("tmpfs", jail, "tmpfs", _MS_NOSUID | _MS_NODEV, "mode=0755,size=16m")
    runtime = jail / RUNTIME_ROOT.relative_to("/")
    _directory(runtime)
    _directory(jail / "workspace", 0o711)

    # These trees contain the installed OS/Python/SDK dependencies. Corral and
    # task packages are installed privately at /opt/corral, outside every mount.
    for source in roots:
        target = runtime / source.relative_to("/")
        _directory(target.parent)
        if source.is_symlink():
            target.symlink_to(RUNTIME_ROOT / source.resolve().relative_to("/"))
        else:
            bind(source, target, readonly=True)

    etc = runtime / "etc"
    _directory(etc)
    for name in _PUBLIC_ETC:
        source = Path("/etc") / name
        if not source.exists():
            continue
        target = etc / name
        # Only certificate and runtime configuration is published, never the
        # rest of /etc (including passwords, private keys, and application data).
        for parent in reversed(target.parents):
            if parent.is_relative_to(etc):
                _directory(parent)
        bind(source.resolve(), target, readonly=True)

    target_workspace = jail / Path(workspace).relative_to("/")
    for parent in reversed(target_workspace.parents):
        if parent.is_relative_to(jail / "workspace"):
            _directory(parent, 0o711)
    # Pin the controller-validated directory across concurrent renames by the
    # main agent. Mount from a verified descriptor, not a model-writable path.
    from corral.runtime.permissions import _open_directory

    # Bind sources must belong to the new mount namespace. Reopen with
    # O_NOFOLLOW and verify the inode against the controller's pinned handle.
    local_fd = _open_directory(workspace)
    try:
        if workspace_fd is not None:
            expected, actual = os.fstat(workspace_fd), os.fstat(local_fd)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise PermissionError(
                    "assigned workspace was replaced during bootstrap"
                )
        bind(Path(f"/proc/self/fd/{local_fd}"), target_workspace, readonly=False)
    finally:
        os.close(local_fd)
    scratch_path = Path(
        tempfile.mkdtemp(prefix=".corral-runtime-", dir=target_workspace)
    )
    os.chown(scratch_path, uid, gid)
    scratch = str(Path(workspace) / scratch_path.name)
    shared_memory = scratch_path / "shm"
    shared_memory.mkdir()
    os.chown(shared_memory, uid, gid)

    # A fresh procfs hides other worker/controller identities. In particular,
    # /proc/PID/root cannot act as a path back to the original container root.
    proc = runtime / "proc"
    _directory(proc)
    mount(
        "proc",
        proc,
        "proc",
        _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC,
        "hidepid=2",
    )
    dev = runtime / "dev"
    _directory(dev)
    for name in ("null", "zero", "random", "urandom", "tty"):
        bind(Path("/dev") / name, dev / name, readonly=False)
    _directory(dev / "pts")
    mount(
        "devpts",
        dev / "pts",
        "devpts",
        _MS_NOSUID | _MS_NOEXEC,
        "newinstance,ptmxmode=0666,mode=0620",
    )
    (dev / "ptmx").symlink_to("pts/ptmx")
    (dev / "shm").symlink_to(Path(scratch) / "shm")
    (dev / "fd").symlink_to("/proc/self/fd")
    for number, name in enumerate(("stdin", "stdout", "stderr")):
        (dev / name).symlink_to(f"/proc/self/fd/{number}")

    # Existing private paths fail with EACCES; other unpublished paths simply
    # do not exist in this filesystem. Neither can expose original image bytes.
    _directory(runtime / "opt" / "corral", 0)
    for name in (
        "root",
        "home",
        "corral-state",
        "sys",
        "srv",
        "mnt",
        "media",
        "boot",
        "run",
    ):
        _directory(jail / name, 0)
    for name in ("passwd", "shadow", "group", "gshadow"):
        (etc / name).touch(mode=0)

    for name in ("usr", "bin", "sbin", "lib", "lib64", "opt", "etc", "proc", "dev"):
        if (runtime / name).exists() or (runtime / name).is_symlink():
            (jail / name).symlink_to(RUNTIME_ROOT / name)
    (jail / "tmp").symlink_to(scratch)

    # Only IPC/stdio may survive the chroot; an inherited directory or regular
    # file descriptor would otherwise bypass the filesystem boundary.
    for descriptor in list(Path("/proc/self/fd").iterdir()):
        number = int(descriptor.name)
        if number not in keep_fds:
            with suppress(OSError):
                os.close(number)
    mount(None, jail, flags=_MS_REMOUNT | _MS_RDONLY | _MS_NOSUID | _MS_NODEV)
    os.chroot(jail)
    os.chdir(workspace)
    return scratch
