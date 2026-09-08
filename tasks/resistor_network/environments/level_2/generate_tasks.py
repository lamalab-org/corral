"""Generate level 2 resistor_network tasks_json/*.json files.

Level 2 circuits are harder: 12-18 resistors combined with deeper random
series/parallel nesting and, with some probability, Wheatstone-bridge
motifs that cannot be solved by series/parallel reduction alone (real
nodal analysis, or a delta-wye transform, is required). Sampled by
`resistor_network.sampler`. Ground truth measurements come from actually
simulating each sampled circuit, not from hand-derived formulas.

Usage (from tasks/resistor_network, with the venv set up per README.md):
    uv run python environments/level_2/generate_tasks.py [--count 14] [--seed 2]
"""

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger
from resistor_network.sampler import generate_level_tasks

LEVEL = 2
DEFAULT_COUNT = 14
DEFAULT_SEED = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate level {LEVEL} resistor_network tasks")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help="Number of tasks to generate (10-18 recommended)"
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
