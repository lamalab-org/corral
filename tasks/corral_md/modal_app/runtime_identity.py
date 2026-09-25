"""Compute the internal identity of one SimAgent build from its exact inputs.

This value is generated automatically while deploying ``simagent``. It is used
for run recovery and verifier cache isolation; operators do not configure it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def source_paths(task_root: Path) -> tuple[Path, ...]:
    return (
        task_root / "modal_app/assets.json",
        task_root / "modal_app/requirements.txt",
        task_root / "modal_app/lammps_app.py",
        task_root / "modal_app/asset_versions.py",
        task_root / "modal_app/runtime_identity.py",
        task_root / "modal_app/verification_worker.py",
        task_root / "modal_app/verification_runtime.py",
        task_root / "modal_app/ground_truth.py",
        task_root / "modal_app/trusted_md.py",
        task_root / "modal_app/verification_requirements.txt",
        task_root / "src/corral_md/workflow_scoring/verification.py",
        task_root / "src/corral_md/modal_workspace.py",
        task_root / "src/corral_md/workspace.py",
        task_root / "src/corral_md/tools.py",
        task_root / "src/corral_md/base_workspace.json",
    )


def runtime_id(
    task_root: Path,
    paths: tuple[Path, ...] | None = None,
    *,
    volume_name: str | None = None,
) -> str:
    """Hash scientific assets, worker/client protocol, seed, and storage name."""
    digest = hashlib.sha256()
    volume = volume_name or os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations")
    digest.update(b"simulations-volume\0" + volume.encode() + b"\0")
    for path in paths if paths is not None else source_paths(task_root):
        digest.update(path.relative_to(task_root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:24]
