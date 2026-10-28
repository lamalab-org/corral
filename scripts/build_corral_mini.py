"""Select equal-sized task subsets from a pinned Corral report summary.

The selector uses task Pass@5 scores from the final benchmark reports. It
chooses the smallest number of tasks per environment/level that meets the
rank and held-out score checks described in corral_mini/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPO = "jablonkagroup/corral-reports-october-runs-summary-v1"
TASKS_FILE = "tasks/train-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", required=True, help="40-character HF dataset commit"
    )
    parser.add_argument(
        "--tasks-parquet", type=Path, help="Use an already downloaded tasks table"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("corral_mini"))
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--min-spearman", type=float, default=0.95)
    parser.add_argument("--max-held-out-error", type=float, default=0.05)
    parser.add_argument("--material-gap", type=float, default=0.05)
    return parser.parse_args()


def read_tasks(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be a full 40-character commit hash")
    if args.tasks_parquet is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                REPO, filename=TASKS_FILE, repo_type="dataset", revision=args.revision
            )
        )
    else:
        path = args.tasks_parquet
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = pd.read_parquet(path)
    required = {
        "environment",
        "level",
        "agent",
        "model",
        "run_id",
        "task_id",
        "trials",
        "success_rate",
        "pass_at_5",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"tasks table is missing {sorted(missing)}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("tasks table is empty or has missing required values")
    if (data.trials != 5).any():
        raise ValueError("expected five final trials per task and agent")
    if not data.success_rate.between(0, 1).all():
        raise ValueError("success_rate must be between zero and one")
    if not data.pass_at_5.between(0, 1).all():
        raise ValueError("pass_at_5 must be between zero and one")
    data = data.copy()
    data["subject"] = data.agent + "__" + data.model
    key = ["environment", "level", "task_id", "subject"]
    if data.duplicated(key).any():
        raise ValueError("duplicate environment/level/task/subject rows")
    return data, digest


def make_matrices(
    data: pd.DataFrame, value: str
) -> dict[tuple[str, int], pd.DataFrame]:
    matrices = {}
    for (environment, level), group in data.groupby(["environment", "level"]):
        matrix = group.pivot_table(
            index="task_id", columns="subject", values=value, aggfunc="first"
        )
        matrix = matrix.sort_index().sort_index(axis=1)
        if matrix.isna().any().any():
            raise ValueError(f"incomplete task coverage in {environment} level {level}")
        matrices[(str(environment), int(level))] = matrix
    if not matrices:
        raise ValueError("no environment/level strata found")
    return dict(sorted(matrices.items()))


def ranking_panel(matrices: dict) -> tuple[list[str], list[tuple[str, int]]]:
    # A subject must have run at least 80% of the environment/level strata.
    # Their common strata define a like-for-like full-benchmark ranking.
    coverage = {}
    for matrix in matrices.values():
        for subject in matrix.columns:
            coverage[subject] = coverage.get(subject, 0) + 1
    subjects = sorted(
        s for s, count in coverage.items() if count / len(matrices) >= 0.8
    )
    common = [
        key for key, matrix in matrices.items() if set(subjects) <= set(matrix.columns)
    ]
    if len(subjects) < 3 or len(common) < 2:
        raise ValueError(
            "need at least three broadly observed subjects and two common strata"
        )
    return subjects, common


def best_subset(matrix: pd.DataFrame, k: int, subjects: list[str]) -> tuple[str, ...]:
    """Match the full stratum's agent success-rate profile by squared error.

    Enumerating combinations is exact for the current pools (10 or 20 tasks).
    Sorted task IDs give a deterministic tie break.
    """
    if k > len(matrix):
        raise ValueError("k exceeds stratum size")
    values = matrix.to_numpy(dtype=float)
    columns = [matrix.columns.get_loc(subject) for subject in subjects]
    target = values[:, columns].mean(axis=0)
    best_ids = None
    best_error = float("inf")
    for indexes in itertools.combinations(range(len(matrix)), k):
        error = float(
            np.mean((values[list(indexes)][:, columns].mean(axis=0) - target) ** 2)
        )
        if error < best_error - 1e-15:
            best_ids = indexes
            best_error = error
    assert best_ids is not None
    return tuple(str(matrix.index[index]) for index in best_ids)


def scores(
    matrices: dict,
    strata: list[tuple[str, int]],
    subjects: list[str],
    selection: dict[tuple[str, int], tuple[str, ...]] | None = None,
) -> np.ndarray:
    values = []
    for key in strata:
        matrix = matrices[key].loc[:, subjects]
        if selection is not None:
            matrix = matrix.loc[list(selection[key])]
        values.append(matrix.mean(axis=0).to_numpy(dtype=float))
    return np.mean(values, axis=0)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = pd.Series(left).rank().to_numpy()
    b = pd.Series(right).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def assess_k(
    matrices: dict,
    common: list[tuple[str, int]],
    subjects: list[str],
    full: np.ndarray,
    k: int,
    material_gap: float,
) -> tuple[dict, dict]:
    selected = {
        key: best_subset(matrix, k, list(matrix.columns))
        for key, matrix in matrices.items()
    }
    mini = scores(matrices, common, subjects, selected)
    stratum_errors = [
        abs(matrix.loc[list(selected[key])].mean(axis=0) - matrix.mean(axis=0))
        for key, matrix in matrices.items()
    ]
    all_stratum_errors = np.concatenate(
        [errors.to_numpy() for errors in stratum_errors]
    )
    held_out_errors = []
    material_reversals = 0
    all_reversals = 0
    for held_index, held_subject in enumerate(subjects):
        train = [subject for subject in subjects if subject != held_subject]
        held_selection = {
            key: best_subset(matrix, k, train)
            for key, matrix in matrices.items()
            if key in common
        }
        held_scores = scores(matrices, common, subjects, held_selection)
        held_out_errors.append(float(abs(held_scores[held_index] - full[held_index])))
        for other_index in range(len(subjects)):
            if held_index == other_index:
                continue
            full_gap = full[held_index] - full[other_index]
            mini_gap = held_scores[held_index] - held_scores[other_index]
            if full_gap * mini_gap < 0:
                all_reversals += 1
                if abs(full_gap) >= material_gap:
                    material_reversals += 1
    metrics = {
        "tasks_per_stratum": k,
        "total_tasks": k * len(matrices),
        "spearman": spearman(full, mini),
        "max_absolute_score_error": float(np.max(abs(full - mini))),
        "mean_absolute_stratum_error": float(np.mean(all_stratum_errors)),
        "max_absolute_stratum_error": float(np.max(all_stratum_errors)),
        "held_out_mean_absolute_score_error": float(np.mean(held_out_errors)),
        "held_out_max_absolute_score_error": float(np.max(held_out_errors)),
        "held_out_pair_reversals": all_reversals,
        "held_out_material_pair_reversals": material_reversals,
    }
    return metrics, selected


def main() -> None:
    args = parse_args()
    data, digest = read_tasks(args)
    matrices = make_matrices(data, "pass_at_5")
    subjects, common = ranking_panel(matrices)
    full = scores(matrices, common, subjects)
    maximum = min(args.max_k, *(len(matrix) for matrix in matrices.values()))
    if not 1 <= args.min_k <= maximum:
        raise ValueError("invalid k range")
    sweep = []
    chosen = None
    for k in range(args.min_k, maximum + 1):
        metrics, selection = assess_k(
            matrices, common, subjects, full, k, args.material_gap
        )
        passed = (
            metrics["spearman"] >= args.min_spearman
            and metrics["held_out_max_absolute_score_error"] <= args.max_held_out_error
            and metrics["held_out_material_pair_reversals"] == 0
        )
        metrics["passed"] = passed
        sweep.append(metrics)
        if passed:
            chosen = (metrics, selection)
            break
    if chosen is None:
        raise ValueError(f"no k met the validation thresholds: {sweep}")

    metrics, selection = chosen
    pass_at_1_matrices = make_matrices(data, "success_rate")
    pass_at_1_full = scores(pass_at_1_matrices, common, subjects)
    pass_at_1_mini = scores(pass_at_1_matrices, common, subjects, selection)
    pass_at_5_mini = scores(matrices, common, subjects, selection)
    by_environment = {}
    for (environment, level), task_ids in selection.items():
        by_environment.setdefault(environment, {})[f"level_{level}"] = list(task_ids)
    manifest = {
        "name": "corral-mini",
        "source": {
            "repo": REPO,
            "revision": args.revision,
            "file": TASKS_FILE,
            "sha256": digest,
        },
        "method": "equal tasks per environment/level; exact Pass@5 profile-matching subset",
        "tasks_per_stratum": metrics["tasks_per_stratum"],
        "total_tasks": metrics["total_tasks"],
        "tasks_by_environment": by_environment,
    }
    assessment = {
        "source_rows": len(data),
        "strata": [f"{env}/level_{level}" for env, level in matrices],
        "subjects_by_stratum": {
            f"{env}/level_{level}": list(matrix.columns)
            for (env, level), matrix in matrices.items()
        },
        "ranking_subjects": subjects,
        "ranking_strata": [f"{env}/level_{level}" for env, level in common],
        "full_pass_at_5_ranking": [subjects[i] for i in np.argsort(-full)],
        "mini_pass_at_5_ranking": [subjects[i] for i in np.argsort(-pass_at_5_mini)],
        "full_pass_at_5": dict(zip(subjects, map(float, full), strict=True)),
        "mini_pass_at_5": dict(zip(subjects, map(float, pass_at_5_mini), strict=True)),
        "full_pass_at_1": dict(zip(subjects, map(float, pass_at_1_full), strict=True)),
        "mini_pass_at_1": dict(zip(subjects, map(float, pass_at_1_mini), strict=True)),
        "pass_at_1_spearman": spearman(pass_at_1_full, pass_at_1_mini),
        "pass_at_1_max_absolute_score_error": float(
            np.max(abs(pass_at_1_full - pass_at_1_mini))
        ),
        "thresholds": {
            "min_spearman": args.min_spearman,
            "max_held_out_error": args.max_held_out_error,
            "material_gap": args.material_gap,
        },
        "sweep": sweep,
        "chosen": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_dir / "assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    logger.info(
        "Selected {} tasks ({} per stratum)",
        metrics["total_tasks"],
        metrics["tasks_per_stratum"],
    )
    logger.info("Ranking Spearman: {:.3f}", metrics["spearman"])
    logger.info(
        "Held-out maximum score error: {:.3f}",
        metrics["held_out_max_absolute_score_error"],
    )


if __name__ == "__main__":
    main()
