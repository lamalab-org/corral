"""Stable, content-addressed Volume names shared by setup and worker images."""

from __future__ import annotations

import hashlib
import json


def asset_volume_names(manifest: dict) -> dict[str, str]:
    specs = {**manifest["archives"], "models": manifest["models"]}
    return {
        name: "corral-md-" + name + "-" + hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        for name, spec in specs.items()
    }
