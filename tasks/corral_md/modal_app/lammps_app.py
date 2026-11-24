"""Versioned Modal workers for persistent Corral MD executions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import modal
from modal import App, Image
from modal.volume import FileEntryType

APP_DIR = (
    Path(__file__).resolve().parent if modal.is_local() else Path("/opt/corral-md")
)
sys.path.insert(0, str(APP_DIR))
from asset_versions import asset_volume_names  # noqa: E402 — image-local helper
from runtime_identity import runtime_id  # noqa: E402 — deployment-local helper

ASSETS = json.loads((APP_DIR / "assets.json").read_text())
# Pin this app's builder; older account defaults inject Python-3.12-incompatible
# dependencies. This also applies when a controller constructs a sandbox image.
os.environ["MODAL_IMAGE_BUILDER_VERSION"] = ASSETS["image_builder_version"]
ASSET_VOLUMES = asset_volume_names(ASSETS)
# The deployment itself supplies this internal fingerprint to its image. No
# operator-provided release setting is needed, including for a plain Modal deploy.
RELEASE_ID = (
    runtime_id(APP_DIR.parent)
    if modal.is_local()
    else os.environ["CORRAL_MD_RUNTIME_ID"]
)
APP_NAME = "simagent"
VOLUME_NAME = os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations")
SEED_FILE = (
    APP_DIR.parent / "src/corral_md/base_workspace.json"
    if modal.is_local()
    else APP_DIR / "base_workspace.json"
)
BASE_SEED = json.loads(SEED_FILE.read_text())

lammps_image = (
    # The account's legacy debian_slim builder uses retired Bullseye security
    # packages. Select Bookworm explicitly while preserving the Python pin.
    Image.from_registry(f"python:{ASSETS['python_version']}-slim-bookworm")
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "liblapack-dev",
        "libfftw3-dev",
        "libopenmpi-dev",
        "openmpi-bin",
        "libssl-dev",
    )
    .pip_install_from_requirements(str(APP_DIR / "requirements.txt"))
    .add_local_file(APP_DIR / "assets.json", "/opt/corral-md/assets.json", copy=True)
    .add_local_file(
        APP_DIR / "requirements.txt", "/opt/corral-md/requirements.txt", copy=True
    )
    .add_local_file(
        APP_DIR / "asset_versions.py", "/opt/corral-md/asset_versions.py", copy=True
    )
    .add_local_file(
        SEED_FILE,
        "/opt/corral-md/base_workspace.json",
        copy=True,
    )
    .run_commands(
        "git init /root/lammps",
        "git -C /root/lammps remote add origin https://github.com/lammps/lammps.git",
        f"git -C /root/lammps fetch --depth 1 origin {ASSETS['lammps_commit']}",
        f"git -C /root/lammps checkout --detach {ASSETS['lammps_commit']}",
        "cmake -S /root/lammps/cmake -B /root/lammps/build "
        "-C /root/lammps/cmake/presets/most.cmake "
        "-C /root/lammps/cmake/presets/nolib.cmake "
        "-DBUILD_MPI=ON -DPKG_MANYBODY=on -DPKG_ATC=yes",
        "cmake --build /root/lammps/build --parallel 4",
        "cmake --install /root/lammps/build",
        "python -m pip freeze > /opt/corral-md/installed-packages.txt",
        "groupadd --gid 10001 corral-md && "
        "useradd --uid 10001 --gid 10001 --no-create-home --home-dir /workspace corral-md",
    )
)

# Keep these after the expensive scientific image build so verifier edits reuse it.
# LAMMPS defaults to /root/.local; sandbox processes need a public runtime prefix.
lammps_image = lammps_image.run_commands(
    "cmake --install /root/lammps/build --prefix /usr/local"
)
lammps_image = lammps_image.pip_install_from_requirements(
    str(APP_DIR / "verification_requirements.txt")
).run_commands("python -m pip freeze > /opt/corral-md/installed-packages.txt")
lammps_image = lammps_image.env(
    {
        "CORRAL_MD_RUNTIME_ID": RELEASE_ID,
        "CORRAL_MD_MODAL_VOLUME": VOLUME_NAME,
        "MODAL_IMAGE_BUILDER_VERSION": ASSETS["image_builder_version"],
    }
)
lammps_image = lammps_image.add_local_file(
    APP_DIR.parent / "src/corral_md/resource_catalog.json",
    "/opt/corral-md/resource_catalog.json",
    copy=True,
)
for _source in (
    "runtime_identity.py",
    "verification_worker.py",
    "verification_runtime.py",
    "ground_truth.py",
    "trusted_md.py",
    "verification_requirements.txt",
):
    lammps_image = lammps_image.add_local_file(
        APP_DIR / _source, f"/opt/corral-md/{_source}", copy=True
    )

app = App(APP_NAME)


@app.function(image=lammps_image, cpu=1)
def runtime_info() -> dict:
    """Let clients discover the active SimAgent deployment without local pins."""
    return {
        "schema": 1,
        "app_name": APP_NAME,
        "release_id": RELEASE_ID,
        "volume_name": VOLUME_NAME,
        "asset_volumes": ASSET_VOLUMES,
    }


volume_potential = modal.Volume.from_name(ASSET_VOLUMES["potentials"])
volume_sim = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
volume_struct = modal.Volume.from_name(ASSET_VOLUMES["structures"])
volume_models = modal.Volume.from_name(ASSET_VOLUMES["models"])

CPUS = 2
GPU_TYPE = "A100"
MAX_EVALUATION_CONTAINERS = 25
RUNS = Path("/results/corral/runs")
ASSET_DIRECTORIES = frozenset({"models", "potentials", "structures"})
_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Invalid Modal workspace identifier: {value!r}")
    return value


def _relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid workspace path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Invalid workspace path: {value!r}")
    if path.parts[0] == "resources":
        raise ValueError(f"Writable workspace cannot replace shared assets: {value!r}")
    return Path(*path.parts)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid workspace manifest: {path}")
    return value


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _manifest(root: Path) -> dict[str, dict[str, str | int]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Modal workspace must be a regular directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Unsafe output in Modal workspace: {path}")
        _relative(path.relative_to(root).as_posix())
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files[path.relative_to(root).as_posix()] = {
            "sha256": digest.hexdigest(),
            "size": size,
        }
    return files


def _release_base(release_id: str) -> dict:
    _id(release_id)
    if release_id != RELEASE_ID:
        raise ValueError("Worker image does not match the pinned SimAgent build")
    if BASE_SEED.get("schema") != 1 or not isinstance(
        BASE_SEED.get("directories"), list
    ):
        raise ValueError("Invalid bundled SimAgent workspace seed")
    for name in BASE_SEED["directories"]:
        _relative(name)
    return {
        **BASE_SEED,
        "release_id": RELEASE_ID,
        "asset_volumes": ASSET_VOLUMES,
    }


def _prepare(run_id: str, release_id: str) -> dict:
    """Reset the mutable upload area from the last successful revision."""
    volume_sim.reload()
    base = _release_base(release_id)
    run = RUNS / _id(run_id)
    state_path = run / "run.json"
    if state_path.exists():
        state = _load(state_path)
        if state.get("release_id") != release_id:
            raise ValueError("Existing run is pinned to a different SimAgent build")
        if state.get("closed_at"):
            raise ValueError("MD run is closed")
    else:
        state = {
            "schema": 1,
            "release_id": release_id,
            "head": "base",
            "revision": 0,
            "files": {},
        }
    working = run / "workspace"
    if working.exists():
        shutil.rmtree(working)
    head = state["head"]
    if head == "base":
        working.mkdir(parents=True)
        for name in base["directories"]:
            (working / _relative(name)).mkdir(parents=True, exist_ok=True)
    else:
        source = run / _relative(head)
        if not source.is_relative_to(run / "attempts") or source.name != "workspace":
            raise ValueError("Invalid Modal run head")
        if _manifest(source) != state["files"]:
            raise ValueError("Committed Modal workspace failed its file manifest check")
        shutil.copytree(source, working)
    state["directories"] = sorted(
        path.relative_to(working).as_posix()
        for path in working.rglob("*")
        if path.is_dir()
    )
    _write(state_path, state)
    volume_sim.commit()
    return state


@app.function(
    image=lammps_image,
    cpu=1,
    timeout=600,
    volumes={"/results": volume_sim},
)
def prepare_workspace(run_id: str, release_id: str) -> dict:
    """Initialize or restore a run from the bundled SimAgent workspace seed."""
    return _prepare(run_id, release_id)


# This bootstrap runs before agent code. The sandbox has no controller mounts or
# credentials. Drop privileges after preparing only its private writable volume.
_BOOTSTRAP = """
import contextlib, ctypes, hashlib, json, os, shutil, signal, subprocess, sys
from pathlib import Path
root = Path('/workspace')
volume = Path('/task-volume')
# Modal Volume paths are symlinks. Stage mutable files into an ordinary
# directory so getcwd() and canonical paths agree inside every invocation.
root.mkdir(mode=0o755, exist_ok=True)
shutil.copytree(volume, root, dirs_exist_ok=True)
for name in json.loads(sys.argv[1]):
    (root / name).mkdir(parents=True, exist_ok=True)
for tree in (root, volume):
    for base, dirs, files in os.walk(tree):
        os.chown(base, 10001, 10001)
        for name in files:
            os.chown(Path(base) / name, 10001, 10001)
# Copy only authorized immutable files from the existing asset cache. Assets
# never enter controller uploads or successful mutable-workspace exports.
resources = root / 'resources'
resources.mkdir(mode=0o755)
for spec in json.loads(sys.argv[2]):
    source = Path('/assets') / spec['relative']
    if source.is_symlink() or not source.is_file():
        raise ValueError('Invalid declared resource source')
    destination = resources / spec['name'] / source.name
    destination.parent.mkdir(mode=0o755)
    digest = hashlib.sha256()
    size = 0
    with source.open('rb') as incoming, destination.open('wb') as outgoing:
        while chunk := incoming.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            outgoing.write(chunk)
    if digest.hexdigest() != spec['sha256'] or size != spec['size']:
        raise ValueError('Declared resource failed hash validation')
    destination.chmod(0o444)
    destination.parent.chmod(0o555)
resources.chmod(0o555)
# Prevent access to volume metadata or files outside the declared catalog.
os.chmod('/assets', 0o700)
# Modal exposes mounts through symlinks into a shared container-local parent.
# Hide those aliases too, retaining only a private handle for mutable export.
volume_fd = os.open(volume, os.O_RDONLY | os.O_DIRECTORY)
os.set_inheritable(volume_fd, False)
for mount in (volume, Path('/assets/potentials'), Path('/assets/models'),
              Path('/assets/structures')):
    if mount.is_symlink():
        backing_parent = mount.resolve().parent
        if backing_parent == Path('/'):
            raise ValueError('Unsafe Modal mount backing parent')
        backing_parent.chmod(0o700)
os.chown(root, 0, 0)
os.chmod(root, 0o1777)
os.setgroups([])
os.setgid(10001)
os.setuid(10001)
if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'Could not disable privilege escalation')
os.chdir('/workspace')
env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/workspace',
       'LANG': 'C.UTF-8', 'TMPDIR': '/workspace/.tmp'}
# CUDA needs the runtime paths injected by the GPU container runtime.
for key in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
            'NVIDIA_DRIVER_CAPABILITIES'):
    if key in os.environ:
        env[key] = os.environ[key]
process = subprocess.Popen(sys.argv[3:], env=env, start_new_session=True, close_fds=True)
try:
    return_code = process.wait()
finally:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
# Export as the same unprivileged identity as the program. Preserve symlinks
# for the controller to reject after container termination, never follow them
# with controller privileges. Declared resources stay outside mutable exports.
os.fchdir(volume_fd)
os.close(volume_fd)
volume = Path('.')
for entry in volume.iterdir():
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    else:
        entry.unlink()
shutil.copytree(
    root, volume, dirs_exist_ok=True, symlinks=True,
    ignore=lambda directory, names: {'resources'} if Path(directory) == root else set(),
)
sys.exit(return_code if return_code >= 0 else 128 - return_code)
"""


def _collect_workspace(volume, working: Path) -> None:
    """Validate the entire untrusted export before replacing the attempt files."""
    entries = list(volume.iterdir("/", recursive=True))
    checked = []
    for entry in entries:
        name = entry.path.removeprefix("/")
        # Root-owned resource copies are immutable and excluded from the
        # mutable workspace receipt and subsequent per-action uploads.
        if name == "resources" or name.startswith("resources/"):
            continue
        relative = _relative(name)
        if entry.type not in {FileEntryType.FILE, FileEntryType.DIRECTORY}:
            raise ValueError(f"Unsafe sandbox output: /workspace/{name}")
        checked.append((relative, entry))
    with tempfile.TemporaryDirectory(
        dir=working.parent, prefix=".export-"
    ) as temporary:
        staged = Path(temporary) / "workspace"
        staged.mkdir()
        for relative, entry in checked:
            target = staged / relative
            if entry.type == FileEntryType.DIRECTORY:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    for chunk in volume.read_file("/" + relative.as_posix()):
                        stream.write(chunk)
        _manifest(staged)
        shutil.rmtree(working)
        shutil.move(str(staged), working)


def _authorized_asset_files(resources: dict | None) -> list[dict]:
    catalog_path = Path("/opt/corral-md/resource_catalog.json")
    if not catalog_path.is_file():
        catalog_path = APP_DIR.parent / "src/corral_md/resource_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    declared = {
        "md-" + str(PurePosixPath(relative).with_suffix("")).replace("/", "-"): {
            "relative": relative,
            "sha256": spec["sha256"],
            "size": spec["size"],
        }
        for relative, spec in catalog.items()
    }
    if resources is None:
        # Trusted verifier code uses the same pinned scientific assets.
        return [{"name": name, **spec} for name, spec in declared.items()]
    selected = []
    for name, descriptor in resources.items():
        spec = declared.get(name)
        if spec is None or any(
            descriptor.get(key) != value
            for key, value in {
                "name": name,
                "filename": PurePosixPath(spec["relative"]).name,
                "sha256": spec["sha256"],
                "size": spec["size"],
            }.items()
        ):
            raise ValueError(f"Resource does not match the pinned MD catalog: {name}")
        selected.append({"name": name, **spec})
    return selected


def _run_sandbox(
    command: list[str],
    working: Path,
    *,
    gpu: bool = False,
    timeout: int = 7200,
    resources: dict | None = None,
    log_stem: str = "execution",
    check_exit: bool = True,
) -> dict:
    files = _manifest(working)
    directories = [
        p.relative_to(working).as_posix() for p in working.rglob("*") if p.is_dir()
    ]
    directories = sorted(set(directories) | {"input", "scripts", "output", ".tmp"})
    # A separate short-lived Volume avoids exposing even the names of other
    # runs or attempts. Assets are mounted independently and never uploaded.
    with modal.Volume.ephemeral() as task_volume:
        with task_volume.batch_upload() as upload:
            for name in files:
                upload.put_file(working / name, "/" + name)
        sandbox = modal.Sandbox.create(
            "python",
            "-c",
            _BOOTSTRAP,
            json.dumps(directories),
            json.dumps(_authorized_asset_files(resources)),
            *command,
            app=app,
            image=lammps_image,
            workdir="/",
            gpu=GPU_TYPE if gpu else None,
            cpu=CPUS,
            memory=10240 if gpu else 5120,
            timeout=timeout,
            volumes={
                "/task-volume": task_volume,
                "/assets/potentials": volume_potential.read_only(),
                "/assets/models": volume_models.read_only(),
                "/assets/structures": volume_struct.read_only(),
            },
        )
        timeout_error = None
        try:
            _write(working.parent / "sandbox.json", {"sandbox_id": sandbox.object_id})
            volume_sim.commit()
            sandbox.wait()
        except modal.exception.SandboxTimeoutError as exc:
            timeout_error = exc
        finally:
            # Wait for termination and the final Volume commit before reading
            # outputs. No background child may keep writing during collection.
            try:
                sandbox.terminate(wait=True)
            except modal.exception.SandboxTimeoutError as exc:
                # terminate(wait=True) waits again and re-raises a settled
                # timeout in the Modal SDK. Still retain the captured logs.
                timeout_error = exc
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
        if timeout_error is None:
            _collect_workspace(task_volume, working)
        (working / "output").mkdir(exist_ok=True)
        (working / "output" / f"{log_stem}.stdout.txt").write_text(stdout)
        (working / "output" / f"{log_stem}.stderr.txt").write_text(stderr)
        if timeout_error is not None:
            raise modal.exception.SandboxTimeoutError(
                f"Sandbox exceeded its {timeout}-second wall-clock timeout, "
                "including Python startup and model loading. Reduce the work per call "
                "and continue from a previously synchronized restart state. "
                "Outputs from this failed call were not synchronized. "
                "Captured logs are retained with the failed attempt.\n"
                f"stdout (last 4000 characters):\n{stdout[-4000:]}\n"
                f"stderr (last 4000 characters):\n{stderr[-4000:]}"
            ) from timeout_error
        if sandbox.returncode and check_exit:
            raise ValueError(
                f"Simulation failed (exit {sandbox.returncode}):\n"
                f"stdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
            )
        return {
            "exit_code": sandbox.returncode,
            "output": stdout + stderr,
            "logs": {
                "stdout": f"/workspace/output/{log_stem}.stdout.txt",
                "stderr": f"/workspace/output/{log_stem}.stderr.txt",
            },
        }


def _run_lammps(
    input_path: Path,
    log_name: str,
    _local_root: str,
    remote_root: Path,
    *,
    execution_options: dict | None = None,
) -> dict:
    # The sandbox supplies public mounts and cwd; preserve the original input.
    public_input = "/workspace/" + input_path.relative_to(remote_root).as_posix()
    result = _run_sandbox(
        [
            "mpirun",
            "--bind-to",
            "core",
            "--map-by",
            "core",
            "-np",
            str(CPUS),
            "/usr/local/bin/lmp",
            "-in",
            public_input,
            "-log",
            "/workspace/output/" + log_name,
        ],
        remote_root,
        log_stem=input_path.stem,
        resources=(execution_options or {}).get("resources", {}),
    )
    if (remote_root / "output" / log_name).is_file():
        result["logs"]["log"] = "/workspace/output/" + log_name
    return result


def _run_python(
    script_path: Path,
    args: list[str],
    _local_root: str,
    remote_root: Path,
    *,
    execution_options: dict | None = None,
) -> dict:
    options = execution_options or {}
    timeout = options.get("timeout", 900)
    if options.get("working_dir", "/workspace") != "/workspace":
        raise ValueError(
            "This MD runtime no longer supports working_dir overrides; start a new run"
        )
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Invalid Python timeout")
    return _run_sandbox(
        [
            "python",
            "-u",
            "/workspace/" + script_path.relative_to(remote_root).as_posix(),
            *args,
        ],
        remote_root,
        gpu=True,
        timeout=timeout,
        resources=options.get("resources", {}),
        log_stem=script_path.stem,
    )


def _execute(
    kind: str,
    run_id: str,
    action_id: str,
    relative_input: str,
    local_workspace: str,
    args: list[str],
    release_id: str,
    base_head: str,
    expected_files: dict,
    execution_options: dict | None = None,
) -> dict:
    volume_sim.reload()
    _release_base(release_id)
    run = RUNS / _id(run_id)
    action = _id(action_id)
    if not local_workspace or not Path(local_workspace).is_absolute():
        raise ValueError("Original MD workspace path must be absolute")
    relative = _relative(relative_input)
    action_dir = run / "actions"
    success_path = action_dir / f"{action}.json"
    failure_path = action_dir / f"{action}.failure.json"
    request = {
        "kind": kind,
        "input": relative_input,
        "args": args,
        "release_id": release_id,
        "base_head": base_head,
        "expected_files": expected_files,
        "execution_options": execution_options or {},
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    if success_path.exists():
        existing = _load(success_path)
        if existing.get("request_sha256") != request_hash:
            raise ValueError("Action ID was reused for a different Modal request")
        return existing
    state_path = run / "run.json"
    state = _load(state_path)
    if state.get("release_id") != release_id or state.get("head") != base_head:
        raise ValueError("Modal workspace advanced before this action started")
    if _manifest(run / "workspace") != expected_files:
        raise ValueError("Uploaded Modal workspace failed its file manifest check")
    call_id = modal.current_function_call_id()
    attempt_id = uuid.uuid4().hex
    _write(
        action_dir / f"{action}.call.json",
        {"call_id": call_id, "attempt_id": attempt_id},
    )
    volume_sim.commit()

    # Every invocation, including Modal's own retries, gets a new directory.
    # An interrupted attempt remains available for diagnostics.
    attempt = run / "attempts" / action / attempt_id
    working = attempt / "workspace"
    shutil.copytree(run / "workspace", working)
    input_path = working / relative
    trusted_md_record = None
    execution = None
    try:
        if not input_path.is_file() or input_path.is_symlink():
            raise FileNotFoundError(f"Modal input not found: {relative_input}")
        if kind == "lammps":
            execution = _run_lammps(
                input_path,
                relative.with_suffix(".log").name,
                local_workspace,
                working,
                **(
                    {"execution_options": execution_options}
                    if execution_options
                    else {}
                ),
            )
        elif kind == "python":
            execution = _run_python(
                input_path,
                args,
                local_workspace,
                working,
                execution_options=execution_options,
            )
        elif kind == "verified_md":
            if args:
                raise ValueError("Controlled MD accepts only a JSON configuration")
            trusted_md_record = _verification_runtime().dynamics(
                _load(input_path), working, run_id=run_id, action_id=action
            )
        else:
            raise ValueError(f"Unknown Modal worker kind: {kind}")
        files = _manifest(working)
    except Exception as exc:
        failure = {
            "action_id": action,
            "request_sha256": request_hash,
            "call_id": call_id,
            "attempt_id": attempt_id,
            "error": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
            "attempt": str(attempt),
        }
        _write(attempt / "failure.json", failure)
        _write(failure_path, failure)
        volume_sim.commit()
        raise

    result = {
        "schema": 1,
        "action_id": action,
        "kind": kind,
        "call_id": call_id,
        "attempt_id": attempt_id,
        "release_id": release_id,
        "request_sha256": request_hash,
        "workspace": str(working),
        "files": files,
        "revision": state["revision"] + 1,
    }
    if execution is not None:
        result["logs"] = execution.get("logs", {})
    if trusted_md_record is not None:
        result["trusted_md"] = trusted_md_record
        _write(
            working / "output/verified_md_receipt.json",
            {
                "run_id": run_id,
                "action_id": action,
                "release_id": release_id,
                "output_directory": "/workspace/" + trusted_md_record["prefix"],
            },
        )
        result["files"] = _manifest(working)
        files = result["files"]
    _write(success_path, result)
    state["head"] = f"attempts/{action}/{attempt_id}/workspace"
    state["revision"] = result["revision"]
    state["files"] = files
    _write(state_path, state)
    volume_sim.commit()
    return result


@app.function(
    image=lammps_image,
    cpu=1,
    timeout=7500,
    memory=2048,
    volumes={"/results": volume_sim},
)
def run_lammps(
    run_id: str,
    action_id: str,
    input_file: str,
    local_workspace: str,
    args: list[str],
    release_id: str,
    base_head: str,
    expected_files: dict,
    execution_options: dict | None = None,
) -> dict:
    """Coordinate one isolated LAMMPS sandbox and commit its successful output."""
    if args:
        raise ValueError("LAMMPS worker does not accept script arguments")
    return _execute(
        "lammps",
        run_id,
        action_id,
        input_file,
        local_workspace,
        args,
        release_id,
        base_head,
        expected_files,
        execution_options,
    )


@app.function(
    image=lammps_image,
    cpu=1,
    timeout=7500,
    memory=2048,
    volumes={"/results": volume_sim},
)
def run_python_gpu(
    run_id: str,
    action_id: str,
    script_file: str,
    local_workspace: str,
    args: list[str],
    release_id: str,
    base_head: str,
    expected_files: dict,
    execution_options: dict | None = None,
) -> dict:
    """Coordinate isolated GPU Python; the controller has no GPU reservation."""
    return _execute(
        "python",
        run_id,
        action_id,
        script_file,
        local_workspace,
        args,
        release_id,
        base_head,
        expected_files,
        execution_options,
    )


def _verification_runtime():
    from types import SimpleNamespace

    from verification_runtime import VerificationRuntime

    return VerificationRuntime(SimpleNamespace(**globals()))


@app.function(
    image=lammps_image,
    cpu=1,
    max_containers=MAX_EVALUATION_CONTAINERS,
    timeout=7500,
    memory=4096,
    volumes={"/results": volume_sim},
)
def verify_calculations(
    verification_id: str, release_id: str, expected_files: dict
) -> dict:
    """Evaluator-only independent single-point, descriptor and checkpoint checks."""
    return _verification_runtime().calculations(
        verification_id, release_id, expected_files
    )


@app.function(
    image=lammps_image,
    cpu=1,
    max_containers=MAX_EVALUATION_CONTAINERS,
    timeout=600,
    memory=2048,
    volumes={"/results": volume_sim},
)
def verify_md_provenance(
    run_id: str, action_id: str, release_id: str, expected_files: dict
) -> dict:
    """Resolve the controller's immutable record, never a submitted attestation."""
    return _verification_runtime().provenance(
        run_id, action_id, release_id, expected_files
    )


@app.function(
    image=lammps_image,
    cpu=1,
    timeout=7500,
    memory=2048,
    volumes={"/results": volume_sim},
)
def run_verified_md(
    run_id: str,
    action_id: str,
    config_file: str,
    local_workspace: str,
    args: list[str],
    release_id: str,
    base_head: str,
    expected_files: dict,
    execution_options: dict | None = None,
) -> dict:
    """Run a declarative task-10 protocol and retain actual execution evidence."""
    return _execute(
        "verified_md",
        run_id,
        action_id,
        config_file,
        local_workspace,
        args,
        release_id,
        base_head,
        expected_files,
        execution_options,
    )
