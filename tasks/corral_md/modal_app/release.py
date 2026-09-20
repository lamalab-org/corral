"""Verify assets, publish a versioned base, deploy the worker, and smoke test it.

Run from the task root with ``uv run python modal_app/release.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from contextlib import suppress
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import modal
from asset_versions import asset_volume_names
from ground_truth import publish_ground_truth
from setup_assets import prepare_assets, upload_assets
from verification_smoke import smoke_verification

TASK_ROOT = Path(__file__).resolve().parents[1]
SEED = TASK_ROOT / "src/corral_md/base_workspace.json"
RELEASE_RECORD = TASK_ROOT / "src/corral_md/release.json"
SOURCES = (
    TASK_ROOT / "modal_app/assets.json",
    TASK_ROOT / "modal_app/requirements.txt",
    TASK_ROOT / "modal_app/lammps_app.py",
    TASK_ROOT / "modal_app/asset_versions.py",
    TASK_ROOT / "modal_app/verification_worker.py",
    TASK_ROOT / "modal_app/verification_runtime.py",
    TASK_ROOT / "modal_app/ground_truth.py",
    TASK_ROOT / "modal_app/trusted_md.py",
    TASK_ROOT / "modal_app/verification_requirements.txt",
    TASK_ROOT / "src/corral_md/workflow_scoring/verification.py",
    TASK_ROOT / "src/corral_md/modal_workspace.py",
    TASK_ROOT / "src/corral_md/workspace.py",
    TASK_ROOT / "src/corral_md/tools.py",
    SEED,
)


def release_id() -> str:
    """Name the complete assets, worker image recipe, and workspace seed."""
    digest = hashlib.sha256()
    digest.update(
        b"simulations-volume\0"
        + os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations").encode()
        + b"\0"
    )
    for path in SOURCES:
        digest.update(path.relative_to(TASK_ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:24]


def _app_name(identifier: str) -> str:
    override = os.getenv("CORRAL_MD_MODAL_APP")
    suffix = os.getenv("SIMAGENT_NAME", "").strip("-")
    prefix = override or ("simagent" + (f"-{suffix}" if suffix else ""))
    return f"{prefix}-{identifier}"


def publish_base(volume: modal.Volume, identifier: str) -> None:
    seed = json.loads(SEED.read_text())
    if seed.get("schema") != 1 or not isinstance(seed.get("directories"), list):
        raise ValueError("Invalid MD base workspace seed")
    for name in seed["directories"]:
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError(f"Invalid MD base directory: {name!r}")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"Invalid MD base directory: {name!r}")
    manifest = {
        **seed,
        "release_id": identifier,
        "asset_volumes": asset_volume_names(
            json.loads((TASK_ROOT / "modal_app/assets.json").read_text())
        ),
        "source_sha256": {
            path.relative_to(TASK_ROOT).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in SOURCES
        },
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    remote = f"/corral/releases/{identifier}/base.json"
    try:
        existing = b"".join(volume.read_file(remote))
    except (FileNotFoundError, modal.exception.NotFoundError):
        existing = None
    if existing is not None:
        if existing != payload:
            raise ValueError(
                f"Refusing to replace a different MD base release: {remote}"
            )
        return
    with volume.batch_upload() as upload:
        upload.put_file(BytesIO(payload), remote)


def smoke_test(identifier: str, volume: modal.Volume) -> None:
    """Exercise LAMMPS, CPU/GPU Python, asset permissions and result recovery."""
    app_name = _app_name(identifier)
    initializer = modal.Function.from_name(app_name, "prepare_workspace")
    worker = modal.Function.from_name(app_name, "run_lammps")
    run_id = f"smoke-{uuid.uuid4().hex}"
    action_id = f"smoke-{uuid.uuid4().hex}"
    remote = f"/corral/runs/{run_id}"
    script = b"""units metal
atom_style atomic
lattice fcc 4.0
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
"""
    try:
        state = initializer.remote(run_id, identifier)
        with volume.batch_upload() as upload:
            upload.put_file(BytesIO(script), f"{remote}/workspace/smoke.in")
        call = worker.spawn(
            run_id,
            action_id,
            "smoke.in",
            "/smoke",
            [],
            identifier,
            state["head"],
            {
                "smoke.in": {
                    "sha256": hashlib.sha256(script).hexdigest(),
                    "size": len(script),
                }
            },
        )
        result = call.get()
        if result.get("action_id") != action_id or "output/smoke.log" not in result.get(
            "files", {}
        ):
            raise RuntimeError("MD worker smoke test did not publish its log")
        if not b"".join(volume.read_file(f"{remote}/actions/{action_id}.json")):
            raise RuntimeError(
                "MD worker smoke test did not commit its result manifest"
            )
        python_script = b"""import json, os, sys
from pathlib import Path
assert Path.cwd().samefile('/workspace'), str(Path.cwd())
assert os.getuid() == 10001
for hidden in ('/results', '/bases', '/corral-state', '/test_files'):
    assert not Path(hidden).exists(), hidden
for name in ('models', 'potentials', 'structures'):
    root = Path('/workspace') / name
    assert root.is_dir(), root
    assert root.is_symlink(), root
    try:
        root.unlink()
    except PermissionError:
        pass
    else:
        raise AssertionError(f'Asset link can be replaced: {root}')
    assert any(p.is_file() for p in root.rglob('*')), root
    probe = root / '.corral-read-only-smoke'
    try:
        with probe.open('xb'):
            pass
    except PermissionError:
        pass
    except OSError as exc:
        import errno
        assert exc.errno == errno.EROFS, exc
    else:
        probe.unlink()
        raise AssertionError(f'Asset mount is writable: {root}')
assert Path('/workspace/models/teacher.model').is_file()
assert Path('/workspace/models/student.model').is_file()
if sys.argv[1] == 'gpu':
    import torch
    assert torch.cuda.is_available()
    assert (torch.tensor([2.0], device='cuda') * 3).cpu().item() == 6
Path('/workspace/output/' + sys.argv[1] + '.json').write_text(json.dumps({'ok': True}))
"""
        for mode in ("cpu", "gpu"):
            state = initializer.remote(run_id, identifier)
            action = f"smoke-{uuid.uuid4().hex}"
            name = f"scripts/smoke_{mode}.py"
            with volume.batch_upload() as upload:
                upload.put_file(BytesIO(python_script), f"{remote}/workspace/{name}")
            worker = modal.Function.from_name(app_name, f"run_python_{mode}")
            result = worker.spawn(
                run_id,
                action,
                name,
                "/workspace",
                [mode],
                identifier,
                state["head"],
                {
                    **state["files"],
                    name: {
                        "sha256": hashlib.sha256(python_script).hexdigest(),
                        "size": len(python_script),
                    },
                },
                execution_options={"timeout": 300, "working_dir": "/workspace"},
            ).get()
            if f"output/{mode}.json" not in result.get("files", {}):
                raise RuntimeError(
                    f"MD {mode} sandbox smoke test did not publish its result"
                )
    finally:
        with suppress(FileNotFoundError, modal.exception.NotFoundError):
            volume.remove_file(remote, recursive=True)


def deploy() -> str:
    identifier = release_id()
    print(f"Preparing MD release {identifier}", flush=True)
    with TemporaryDirectory(prefix=".corral-md-assets-", dir=TASK_ROOT) as temporary:
        source = Path(temporary)
        prepare_assets(source)
        upload_assets(source)
        base_volume = modal.Volume.from_name("corral-md-bases", create_if_missing=True)
        publish_base(base_volume, identifier)
        print("Assets verified; deploying worker", flush=True)
        environment = {**os.environ, "CORRAL_MD_RELEASE_ID": identifier}
        subprocess.run(
            [sys.executable, "-m", "modal", "deploy", "modal_app/lammps_app.py"],
            cwd=TASK_ROOT,
            env=environment,
            check=True,
        )
        volume = modal.Volume.from_name(
            os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations"), create_if_missing=True
        )
        print("Running LAMMPS and CPU/GPU sandbox smoke tests", flush=True)
        smoke_test(identifier, volume)
        print(
            "Running independent calculator and controlled MD smoke tests", flush=True
        )
        verification_smoke = smoke_verification(
            _app_name(identifier), identifier, volume, source / "models"
        )
        print("Publishing fixed Task 3 and Task 5 numerical references", flush=True)
        ground_truth = publish_ground_truth(
            _app_name(identifier),
            identifier,
            volume,
            json.loads((TASK_ROOT / "modal_app/assets.json").read_text()),
        )
    temporary = RELEASE_RECORD.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "release_id": identifier,
                "app_name": _app_name(identifier),
                "volume_name": os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations"),
                "verification_smoke": verification_smoke,
                "ground_truth": ground_truth,
                "asset_volumes": asset_volume_names(
                    json.loads((TASK_ROOT / "modal_app/assets.json").read_text())
                ),
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(RELEASE_RECORD)
    return identifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-id", action="store_true", help="Print release ID without deploying"
    )
    args = parser.parse_args()
    identifier = release_id() if args.print_id else deploy()
    print(identifier)  # noqa: T201


if __name__ == "__main__":
    sys.exit(main())
