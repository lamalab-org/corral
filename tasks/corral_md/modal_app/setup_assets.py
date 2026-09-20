"""Verify benchmark inputs locally, then optionally populate the Modal volumes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_versions import asset_volume_names

TASK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(Path(__file__).with_name("assets.json").read_text())


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_assets(destination: Path) -> None:
    """Extract verified archives and fetch the exact, checksum-verified models."""
    destination.mkdir(parents=True, exist_ok=True)
    for volume, spec in MANIFEST["archives"].items():
        archive = TASK_ROOT / spec["file"]
        if file_sha256(archive) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {archive}")
        with ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                path = PurePosixPath(entry.filename)
                if entry.is_dir() or path.parts[0] == "__MACOSX":
                    continue
                if path.is_absolute() or ".." in path.parts or path.parts[0] != volume:
                    raise ValueError(f"Unexpected archive member: {entry.filename}")
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(entry))

    model_dir = destination / "models"
    model_dir.mkdir(exist_ok=True)
    for name, spec in MANIFEST["models"].items():
        target = model_dir / name
        if not target.exists():
            # A failed or interrupted download must never look like a model.
            with TemporaryDirectory(dir=model_dir) as temporary:
                download = Path(temporary) / name
                with (
                    urllib.request.urlopen(spec["url"], timeout=120) as response,
                    download.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                if file_sha256(download) != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch for downloaded model: {name}")
                download.replace(target)
        if file_sha256(target) != spec["sha256"]:
            raise ValueError(
                f"Checksum mismatch: {target}; remove the stale file and retry"
            )


def upload_assets(source: Path) -> None:
    """Add missing files; refuse to overwrite different existing volume contents."""
    import modal  # noqa: PLC0415 — local preparation needs only the standard library

    pending = []
    # Check every volume before uploading anything. Existing identical files
    # make setup repeatable; mismatches require an explicit operator decision.
    for name, volume_name in asset_volume_names(MANIFEST).items():
        volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        missing = []
        for path in sorted((source / name).rglob("*")):
            if not path.is_file():
                continue
            remote = "/" + path.relative_to(source / name).as_posix()
            digest = hashlib.sha256()
            try:
                for chunk in volume.read_file(remote):
                    digest.update(chunk)
            except (FileNotFoundError, modal.exception.NotFoundError):
                missing.append((path, remote))
            else:
                if digest.hexdigest() != file_sha256(path):
                    raise ValueError(
                        f"Refusing to overwrite different asset: {name}:{remote}"
                    )
        pending.append((volume, missing))
    for volume, missing in pending:
        if missing:
            with volume.batch_upload() as upload:
                for local, remote in missing:
                    upload.put_file(str(local), remote)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Keep verified assets locally")
    parser.add_argument(
        "--upload", action="store_true", help="Populate the Modal volumes"
    )
    args = parser.parse_args()
    if args.output_dir is None and not args.upload:
        parser.error(
            "specify --output-dir for local preparation or --upload for Modal setup"
        )
    with TemporaryDirectory(prefix="corral-md-assets-") as temporary:
        destination = args.output_dir or Path(temporary)
        prepare_assets(destination)
        if args.upload:
            upload_assets(destination)
        print("Corral MD assets verified" + (" and uploaded." if args.upload else "."))  # noqa: T201


if __name__ == "__main__":
    main()
