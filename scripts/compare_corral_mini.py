"""Compare selection methods for 1..max-k tasks per level.

Writes corral_mini/results.csv (results_unsaturated.csv with
--exclude-saturated) and corral_mini/saturation.csv.

Random rows report the median metric over seeded draws; their per-level
Spearman is averaged over draws before taking percentiles. All other methods
are deterministic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from corral_mini_common import (
    METHODS,
    evaluate,
    held_out_model_losses,
    load,
    select,
)
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="HF dataset commit hash")
    parser.add_argument("--tasks-parquet", type=Path, help="local copy of the table")
    parser.add_argument("--prices", type=Path, default=Path("corral_mini/prices.json"))
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--random-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-saturated", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--saturation-output", type=Path, default=Path("corral_mini/saturation.csv")
    )
    args = parser.parse_args()
    suffix = "_unsaturated" if args.exclude_saturated else ""
    output = args.output or Path(f"corral_mini/results{suffix}.csv")

    strata, agents, _ = load(args.revision, args.tasks_parquet, args.prices)
    every_agent = np.arange(len(agents))
    full_usd = sum(stratum.costs.sum() for stratum in strata.values())
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(1, args.max_k + 1):
        for method in METHODS:
            draws = args.random_draws if method == "random" else 1
            results = []
            for _ in range(draws):
                selection = {
                    key: select(
                        stratum, k, method, every_agent, rng, args.exclude_saturated
                    )
                    for key, stratum in strata.items()
                }
                result = evaluate(strata, selection)
                result["held_out_levels_with_loss"] = held_out_model_losses(
                    strata,
                    agents,
                    k,
                    method,
                    selection if method == "random" else None,
                    args.exclude_saturated,
                )
                results.append(result)
            # Per-level Spearman (averaged over random draws) for the spread bars.
            level_rho = np.mean([r["level_spearman"] for r in results], axis=0)
            row = {
                "tasks_per_level": k,
                "total_tasks": k * len(strata),
                "method": method,
            }
            for metric in (
                "mean_level_spearman",
                "levels_with_loss",
                "saturated_tasks",
                "held_out_levels_with_loss",
                "worst_level_error",
                "mean_score_shift",
                "overall_spearman",
                "usd",
            ):
                row[metric] = round(float(np.median([r[metric] for r in results])), 4)
            row["level_spearman_q25"] = round(float(np.percentile(level_rho, 25)), 4)
            row["level_spearman_q75"] = round(float(np.percentile(level_rho, 75)), 4)
            row["cost_fraction"] = round(row["usd"] / full_usd, 4)
            rows.append(row)
            logger.info("k={} {}: {}", k, method, row)
    write_csv(output, rows)
    logger.info("Full benchmark estimate: ${:,.0f}", full_usd)

    # How many agents solve each task (at least once in five trials), per level.
    saturation = []
    for (environment, level), stratum in strata.items():
        solvers = stratum.solved.sum(axis=1)
        saturation.append(
            {
                "environment": environment,
                "level": level,
                "tasks": len(solvers),
                "solved_by_all": int((solvers == len(agents)).sum()),
                "solved_by_majority_not_all": int(
                    ((solvers > len(agents) / 2) & (solvers < len(agents))).sum()
                ),
                "solved_by_minority": int(
                    ((solvers > 0) & (solvers <= len(agents) / 2)).sum()
                ),
                "solved_by_none": int((solvers == 0).sum()),
            }
        )
    write_csv(args.saturation_output, saturation)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
