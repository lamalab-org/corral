"""Pinned MD assets in Corral's immutable resource namespace."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from corral.core.resources import FileResourceDescriptor
from corral.core.workspace import blob_ref_for_sha256

RUNTIME_VERSION = "corral-md-resources-v1"
# Load before workers enter their filesystem jail; the installed task package
# is controller-private, while these immutable declarations may be used inside.
_CATALOG = json.loads(Path(__file__).with_name("resource_catalog.json").read_text())


def asset_resource_name(relative: str) -> str:
    return "md-" + str(PurePosixPath(relative).with_suffix("")).replace("/", "-")


def asset_descriptors() -> dict[str, FileResourceDescriptor]:
    return {
        asset_resource_name(relative): FileResourceDescriptor(
            name=asset_resource_name(relative),
            filename=PurePosixPath(relative).name,
            blob_ref=blob_ref_for_sha256(spec["sha256"]),
            sha256=spec["sha256"],
            size=spec["size"],
            runtime_version=RUNTIME_VERSION,
        )
        for relative, spec in _CATALOG.items()
    }


def asset_relative_paths() -> dict[str, str]:
    return {asset_resource_name(relative): relative for relative in _CATALOG}


def asset_public_path(relative: str) -> str:
    return asset_descriptors()[asset_resource_name(relative)].public_path
