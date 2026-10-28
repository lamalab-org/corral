"""Sweep task counts and compare within-level ranking across environments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from build_corral_mini import make_matrices, ranking_panel, read_tasks, spearman
from compare_corral_mini import add_cost, saturation_flags, select
from loguru import logger

METHODS = (
    ("random", "random", 0.0),
    ("unsaturated_random", "unsaturated_random", 0.0),
    ("score_profile", "profile", 0.0),
    ("unsaturated_score_profile", "unsaturated_profile", 0.0),
    ("spearman_rank", "spearman_rank", 0.0),
    ("guarded_spearman_rank", "guarded_spearman_rank", 0.0),
    ("cost_aware", "cost_tolerant", 0.02),
    ("unsaturated_cost_aware", "unsaturated_cost_tolerant", 0.02),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks-parquet", type=Path)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prices", type=Path, default=Path("corral_mini/prices.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("corral_mini"))
    return parser.parse_args()


def correlations(
    matrices: dict, subjects: list[str], selected: dict
) -> tuple[list[float], dict]:
    by_environment = defaultdict(list)
    all_levels = []
    for (environment, level), matrix in matrices.items():
        scored = matrix.loc[:, subjects]
        full = scored.mean(axis=0).to_numpy(dtype=float)
        if np.std(full) == 0:
            continue
        mini = (
            scored.loc[list(selected[(environment, level)])]
            .mean(axis=0)
            .to_numpy(dtype=float)
        )
        correlation = spearman(full, mini)
        all_levels.append(correlation)
        by_environment[environment].append(correlation)
    env_means = {env: float(np.mean(values)) for env, values in by_environment.items()}
    return all_levels, env_means


def main() -> None:
    args = parse_args()
    if args.random_repeats < 1 or args.min_k < 1 or args.max_k < args.min_k:
        raise ValueError("invalid sweep range or random repeat count")
    data, digest = read_tasks(args)
    data = data[data.environment != "afm"].copy()
    data, prices = add_cost(data, args.prices)
    matrices = make_matrices(data, "pass_at_5")
    costs = make_matrices(data, "estimated_cost_usd")
    saturated = saturation_flags(matrices)
    subjects, common = ranking_panel(matrices)
    if len(common) != len(matrices):
        raise ValueError("sweep requires a common subject panel across all strata")
    max_eligible = min(
        len(matrix) - len(saturated[key]) for key, matrix in matrices.items()
    )
    if args.max_k > max_eligible:
        raise ValueError(
            f"max-k exceeds the smallest unsaturated pool ({max_eligible})"
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    detail = {}
    for k in range(args.min_k, args.max_k + 1):
        for label, method, tolerance in METHODS:
            repeats = args.random_repeats if method.endswith("random") else 1
            env_samples = defaultdict(list)
            level_means = []
            for _ in range(repeats):
                selected = select(
                    matrices,
                    costs,
                    saturated,
                    k,
                    method,
                    rng=rng if method.endswith("random") else None,
                    cost_tolerance=tolerance,
                )
                levels, envs = correlations(matrices, subjects, selected)
                level_means.append(float(np.mean(levels)))
                for env, value in envs.items():
                    env_samples[env].append(value)
            env_means = {
                env: float(np.mean(values)) for env, values in env_samples.items()
            }
            environment_values = list(env_means.values())
            rows.append(
                {
                    "tasks_per_level": k,
                    "method": label,
                    "mean_level_spearman": float(np.mean(level_means)),
                    "env_q25": float(np.percentile(environment_values, 25)),
                    "env_q75": float(np.percentile(environment_values, 75)),
                    "env_min": float(np.min(environment_values)),
                    "env_max": float(np.max(environment_values)),
                    "random_repeats": repeats,
                }
            )
            detail[f"{label}/k_{k}"] = env_means
            logger.info("k={} {}: {:.3f}", k, label, rows[-1]["mean_level_spearman"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "sweep_environments.json").write_text(
        json.dumps(
            {
                "source": {"revision": args.revision, "sha256": digest},
                "prices": prices,
                "excluded_environments": ["afm"],
                "ranking_subjects": subjects,
                "informative_levels": int(
                    sum(
                        np.std(
                            matrix.loc[:, subjects].mean(axis=0).to_numpy(dtype=float)
                        )
                        > 0
                        for matrix in matrices.values()
                    )
                ),
                "environment_means": detail,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
