"""Generate environments/level_1/tasks_json/task_N.json for discover_physics.

One task per (world, seed, noise_std) combination, mirroring the sweep in
upstream DiscoverPhysics's configs/bench.yml (seeds x noise_frac x worlds).
Re-run this after editing discover_physics.content / discover_physics.worlds
to regenerate the task JSONs; it has no dependency on scienceagent/physchool
(and therefore no jax requirement) since worlds.py only imports those lazily
inside build_world()/build_evaluator(), which this script never calls.

Usage:
    cd tasks/discover_physics
    PYTHONPATH=src python3 scripts/generate_tasks.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from discover_physics.content import WORLD_CONTENT, build_description
from discover_physics.worlds import ALL_WORLDS, WORLD_ENGINE

SEEDS = [0, 1]
NOISE_LEVELS = [0.0, 0.05]

OUT_DIR = Path(__file__).resolve().parents[1] / "environments" / "level_1" / "tasks_json"


def _submission_format(world: str) -> str:
    law_stub = WORLD_CONTENT[world].law_stub
    return (
        "A single Python source string defining `discovered_law(...)` with the exact "
        f"signature below (constants must be hardcoded, not fit at submission time):\n\n{law_stub}"
    )


def build_task(world: str, seed: int, noise_std: float) -> dict:
    task_id = f"{world}_seed{seed}_noise{noise_std}"
    return {
        "id": task_id,
        "name": task_id,
        "description": build_description(world),
        "tools": ["run_experiment"],
        "world_config": {
            "world": world,
            "engine": WORLD_ENGINE[world],
            "noise_std": noise_std,
            "noise_seed": seed,
        },
        "submission_format": _submission_format(world),
        "initial_input": {},
        "uuid": str(uuid.uuid4()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale generated files before regenerating.
    for stale in OUT_DIR.glob("task_*.json"):
        stale.unlink()

    index = 0
    for world in ALL_WORLDS:
        for seed in SEEDS:
            for noise_std in NOISE_LEVELS:
                index += 1
                task = build_task(world, seed, noise_std)
                out_path = OUT_DIR / f"task_{index}.json"
                out_path.write_text(json.dumps([task], indent=2) + "\n")

    print(f"Wrote {index} task JSON files to {OUT_DIR}")


if __name__ == "__main__":
    main()
