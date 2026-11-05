"""Write corral-mini release `--version`: k tasks per environment/level.

Levels with report data use `--method`; EXCLUDED_ENVIRONMENTS have none, so
their tasks are drawn at random with `--seed`. Releases are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from corral_mini_common import EXCLUDED_ENVIRONMENTS, METHODS, load, select
from loguru import logger

REPOSITORY = Path(__file__).resolve().parents[1]
RELEASES = REPOSITORY / "src" / "corral" / "subsets"


def task_ids_on_disk(environment: str) -> dict[int, list[str]]:
    """Task IDs per level from tasks/<environment>/environments/level_*/tasks_json."""
    root = REPOSITORY / "tasks" / environment / "environments"
    levels = {
        int(level_dir.name.removeprefix("level_")): sorted(
            entry["id"]
            for task_file in (level_dir / "tasks_json").glob("*.json")
            for entry in json.loads(task_file.read_text())
        )
        for level_dir in root.glob("level_*")
    }
    if not levels:
        raise FileNotFoundError(f"no task files under {root}")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="HF dataset commit hash")
    parser.add_argument("--tasks-parquet", type=Path, help="local copy of the table")
    parser.add_argument("--prices", type=Path, default=Path("corral_mini/prices.json"))
    parser.add_argument(
        "--method", choices=[m for m in METHODS if m != "random"], default="rank_cost"
    )
    parser.add_argument("--k", type=int, default=3, help="tasks per level")
    parser.add_argument(
        "--seed", type=int, default=0, help="breaks ties and draws random levels"
    )
    parser.add_argument("--version", type=int, required=True, help="release number")
    parser.add_argument(
        "--exclude-saturated",
        action="store_true",
        help="skip tasks every agent solves; top up short levels with them",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or RELEASES / f"corral_mini_v{args.version}.json"
    if output.exists():
        raise SystemExit(f"{output} exists; released versions are immutable")
    strata, agents, source = load(args.revision, args.tasks_parquet, args.prices)
    every_agent = np.arange(len(agents))
    tasks_by_environment = {}
    for (environment, level), stratum in strata.items():
        chosen, alternative = (
            select(
                stratum, args.k, args.method, every_agent,
                exclude_saturated=args.exclude_saturated, seed=seed,
            )
            for seed in (args.seed, args.seed + 1)
        )  # fmt: skip
        if not np.array_equal(chosen, alternative):
            logger.warning("{} level {}: decided by tie-break", environment, level)
        tasks_by_environment.setdefault(environment, {})[f"level_{level}"] = [
            str(task) for task in stratum.tasks[chosen]
        ]
    rng = np.random.default_rng(args.seed)
    for environment in EXCLUDED_ENVIRONMENTS:
        for level, ids in sorted(task_ids_on_disk(environment).items()):
            tasks_by_environment.setdefault(environment, {})[f"level_{level}"] = sorted(
                rng.choice(ids, size=args.k, replace=False).tolist()
            )
    manifest = {
        "name": "corral-mini",
        "version": args.version,
        "method": args.method,
        "tasks_per_level": args.k,
        "seed": args.seed,
        "exclude_saturated": args.exclude_saturated,
        "random_environments": list(EXCLUDED_ENVIRONMENTS),
        "total_tasks": sum(
            len(ids)
            for levels in tasks_by_environment.values()
            for ids in levels.values()
        ),
        "source": source,
        "tasks_by_environment": dict(sorted(tasks_by_environment.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote {} tasks to {}", manifest["total_tasks"], output)


if __name__ == "__main__":
    main()
