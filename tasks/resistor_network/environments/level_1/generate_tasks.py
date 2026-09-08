"""Generate level 1 resistor_network tasks_json/*.json files.

Level 1 circuits are simple: 8-12 resistors combined with nested series and
parallel composition (no bridge motifs), sampled by
`resistor_network.sampler`. Ground truth measurements come from actually
simulating each sampled circuit, not from hand-derived formulas.

Usage (from tasks/resistor_network, with the venv set up per README.md):
    uv run python environments/level_1/generate_tasks.py [--count 8] [--seed 1]
"""

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger
from resistor_network.sampler import generate_level_tasks

LEVEL = 1
DEFAULT_COUNT = 8
DEFAULT_SEED = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate level {LEVEL} resistor_network tasks")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help="Number of tasks to generate (6-10 recommended)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    tasks = generate_level_tasks(level=LEVEL, count=args.count, seed=args.seed)

    out_dir = Path(__file__).parent / "tasks_json"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for task in tasks:
        path = out_dir / f"{task['id']}.json"
        with path.open("w") as f:
            json.dump([task], f, indent=2)
        logger.info(f"Generated {path} ({task['num_resistors']} resistors)")

    logger.info(f"Wrote {len(tasks)} level {LEVEL} tasks to {out_dir}")


if __name__ == "__main__":
    main()
