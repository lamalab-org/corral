"""Write corral-mini release `--version`: k tasks per environment/level.

Levels with report data use `--method`. AFM uses its newer local reports for
Pass^5 selection, as do the environments named with `--local-reports` (for
example retrosynthesis, whose October reports were corrected after the dataset
revision was published). Tasks named with `--exclude-task` are left out before
selection (for example, a task whose student model is no longer served).
Environments without reports are drawn with `--seed`.
Releases are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from corral_mini_common import (
    EXCLUDED_ENVIRONMENTS,
    METHODS,
    Stratum,
    load,
    load_afm_strata,
    load_report_strata,
    rank_cost_tie_counts,
    select,
)
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
    parser.add_argument(
        "--score-metric", choices=("pass_at_5", "pass_hat_5"), default="pass_hat_5"
    )
    parser.add_argument("--k", type=int, default=3, help="tasks per level")
    parser.add_argument(
        "--seed", type=int, default=0, help="breaks ties and draws random levels"
    )
    parser.add_argument("--version", type=int, required=True, help="release number")
    parser.add_argument(
        "--exclude-saturated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer tasks not solved by all agents (default: enabled)",
    )
    parser.add_argument(
        "--local-reports",
        action="append",
        default=[],
        metavar="ENVIRONMENT",
        help="take this environment's scores from reports/<environment>/trials.csv "
        "instead of the dataset revision (repeatable)",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        metavar="ENVIRONMENT/TASK",
        help="leave this task out of selection, e.g. "
        "inference_opt/math_number_theory_a (repeatable)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ties-output",
        type=Path,
        help="per-level counts of sets tied during Rank + cost selection",
    )
    args = parser.parse_args()

    output = args.output or RELEASES / f"corral_mini_v{args.version}.json"
    if output.exists():
        raise SystemExit(f"{output} exists; released versions are immutable")
    strata, agents, source = load(
        args.revision, args.tasks_parquet, args.prices, args.score_metric
    )
    afm_source = None
    if args.score_metric == "pass_hat_5":
        trials_csv = REPOSITORY / "reports/afm/trials.csv"
        runs_csv = REPOSITORY / "reports/afm/runs.csv"
        strata.update(load_afm_strata(trials_csv, args.prices, agents))
        afm_source = {
            "runs_csv": str(runs_csv.relative_to(REPOSITORY)),
            "runs_csv_sha256": hashlib.sha256(runs_csv.read_bytes()).hexdigest(),
            "trials_csv": str(trials_csv.relative_to(REPOSITORY)),
            "trials_csv_sha256": hashlib.sha256(trials_csv.read_bytes()).hexdigest(),
        }
    local_sources = {}
    for environment in args.local_reports:
        trials_csv = REPOSITORY / "reports" / environment / "trials.csv"
        runs_csv = trials_csv.with_name("runs.csv")
        replaced = {key for key in strata if key[0] == environment}
        loaded = load_report_strata(
            trials_csv, args.prices, agents, args.score_metric, environment
        )
        if set(loaded) != replaced:
            raise SystemExit(f"{environment}: local levels differ from the dataset")
        strata.update(loaded)
        local_sources[environment] = {
            "runs_csv": str(runs_csv.relative_to(REPOSITORY)),
            "runs_csv_sha256": hashlib.sha256(runs_csv.read_bytes()).hexdigest(),
            "trials_csv": str(trials_csv.relative_to(REPOSITORY)),
            "trials_csv_sha256": hashlib.sha256(trials_csv.read_bytes()).hexdigest(),
        }
    excluded_tasks: dict[str, list[str]] = {}
    for entry in args.exclude_task:
        environment, _, task = entry.partition("/")
        excluded_tasks.setdefault(environment, []).append(task)
    for environment, tasks in excluded_tasks.items():
        found = set()
        for key, stratum in strata.items():
            if key[0] != environment:
                continue
            keep = ~np.isin(stratum.tasks, tasks)
            found.update(stratum.tasks[~keep].tolist())
            strata[key] = Stratum(
                stratum.tasks[keep],
                stratum.passes[keep],
                stratum.costs[keep],
                stratum.solved[keep],
            )
        if missing := sorted(set(tasks) - found):
            raise SystemExit(f"--exclude-task: {environment} has no task {missing}")
        excluded_tasks[environment] = sorted(set(tasks))
    strata = dict(sorted(strata.items()))
    every_agent = np.arange(len(agents))
    tasks_by_environment = {}
    tie_rows = []
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
        if args.method == "rank_cost":
            tie_rows.append(
                {
                    "environment": environment,
                    "level": level,
                    **rank_cost_tie_counts(
                        stratum, args.k, every_agent, args.exclude_saturated
                    ),
                }
            )
    rng = np.random.default_rng(args.seed)
    random_environments = []
    for environment in EXCLUDED_ENVIRONMENTS:
        if environment in tasks_by_environment:
            continue
        random_environments.append(environment)
        for level, ids in sorted(task_ids_on_disk(environment).items()):
            tasks_by_environment.setdefault(environment, {})[f"level_{level}"] = sorted(
                rng.choice(ids, size=args.k, replace=False).tolist()
            )
    manifest = {
        "name": "corral-mini",
        "version": args.version,
        "method": args.method,
        "score_metric": args.score_metric,
        "tasks_per_level": args.k,
        "seed": args.seed,
        "exclude_saturated": args.exclude_saturated,
        "random_environments": random_environments,
        "total_tasks": sum(
            len(ids)
            for levels in tasks_by_environment.values()
            for ids in levels.values()
        ),
        "source": source,
        "tasks_by_environment": dict(sorted(tasks_by_environment.items())),
    }
    if afm_source is not None:
        manifest["afm_source"] = afm_source
    if local_sources:
        manifest["local_sources"] = local_sources
    if excluded_tasks:
        manifest["excluded_tasks"] = dict(sorted(excluded_tasks.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote {} tasks to {}", manifest["total_tasks"], output)
    if tie_rows:
        ties_output = args.ties_output or Path(
            f"corral_mini/rank_cost_{args.score_metric}_ties.csv"
        )
        ties_output.parent.mkdir(parents=True, exist_ok=True)
        with ties_output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(tie_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(tie_rows)
        logger.info("Wrote tie counts to {}", ties_output)


if __name__ == "__main__":
    main()
