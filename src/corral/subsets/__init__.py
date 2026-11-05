"""Released corral-mini task subsets, written once by scripts/build_corral_mini.py."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_DIRECTORY = Path(__file__).parent


def mini_versions() -> tuple[int, ...]:
    """Return the released corral-mini versions in ascending order."""
    return tuple(
        sorted(
            int(path.stem.removeprefix("corral_mini_v"))
            for path in _DIRECTORY.glob("corral_mini_v*.json")
        )
    )


@cache
def load_mini(version: int | None = None) -> dict[str, Any]:
    """Return one corral-mini manifest; `None` selects the latest version."""
    versions = mini_versions()
    version = versions[-1] if version is None else version
    if version not in versions:
        known = ", ".join(map(str, versions))
        raise ValueError(
            f"unknown corral-mini version {version}; choose one of {known}"
        )
    return json.loads((_DIRECTORY / f"corral_mini_v{version}.json").read_text())


def mini_task_ids(
    environment: str, level: int, version: int | None = None
) -> tuple[str, ...] | None:
    """Return the corral-mini task IDs for one level, or `None` if it has none."""
    levels = load_mini(version)["tasks_by_environment"].get(environment, {})
    task_ids = levels.get(f"level_{level}")
    return None if task_ids is None else tuple(task_ids)


__all__ = ["load_mini", "mini_task_ids", "mini_versions"]
