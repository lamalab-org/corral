"""Write corral_mini/manifest.json: k tasks per environment/level chosen by one method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from corral_mini_common import EXCLUDED_ENVIRONMENTS, METHODS, load, select
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="HF dataset commit hash")
    parser.add_argument("--tasks-parquet", type=Path, help="local copy of the table")
    parser.add_argument("--prices", type=Path, default=Path("corral_mini/prices.json"))
    parser.add_argument(
        "--method", choices=[m for m in METHODS if m != "random"], default="profile"
    )
    parser.add_argument("--k", type=int, default=4, help="tasks per level")
    parser.add_argument(
        "--exclude-saturated",
        action="store_true",
        help="skip tasks every agent solves; top up short levels with them",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    suffix = "_unsaturated" if args.exclude_saturated else ""
    output = args.output or Path(f"corral_mini/manifest{suffix}.json")
    strata, agents, source = load(args.revision, args.tasks_parquet, args.prices)
    every_agent = np.arange(len(agents))
    tasks_by_environment = {}
    for (environment, level), stratum in strata.items():
        chosen = select(
            stratum, args.k, args.method, every_agent,
            exclude_saturated=args.exclude_saturated,
        )  # fmt: skip
        tasks_by_environment.setdefault(environment, {})[f"level_{level}"] = [
            str(task) for task in stratum.tasks[chosen]
        ]
    manifest = {
        "name": "corral-mini",
        "status": "candidate",
        "method": args.method,
        "exclude_saturated": args.exclude_saturated,
        "tasks_per_level": args.k,
        "total_tasks": args.k * len(strata),
        "excluded_environments": list(EXCLUDED_ENVIRONMENTS),
        "source": source,
        "tasks_by_environment": tasks_by_environment,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote {} tasks to {}", manifest["total_tasks"], output)


if __name__ == "__main__":
    main()
