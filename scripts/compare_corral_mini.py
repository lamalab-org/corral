"""Compare equal-per-stratum Corral mini samplers on a pinned report table."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from build_corral_mini import make_matrices, ranking_panel, read_tasks, scores, spearman
from loguru import logger

if TYPE_CHECKING:
    import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks-parquet", type=Path)
    parser.add_argument("--tasks-per-stratum", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prices", type=Path, default=Path("corral_mini/prices.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("corral_mini/comparison.json")
    )
    return parser.parse_args()


def add_cost(data: pd.DataFrame, prices_path: Path) -> tuple[pd.DataFrame, dict]:
    prices = json.loads(prices_path.read_text())
    needed = {"total_input_tokens", "total_output_tokens"}
    if needed - set(data.columns):
        raise ValueError(
            f"task table lacks cost columns: {sorted(needed - set(data.columns))}"
        )
    if data[list(needed)].isna().any().any():
        raise ValueError("token totals contain missing values")
    missing = set(data.model) - set(prices)
    if missing:
        raise ValueError(f"missing model prices: {sorted(missing)}")
    data = data.copy()
    data["estimated_cost_usd"] = (
        data.total_input_tokens * data.model.map(lambda model: prices[model]["input"])
        + data.total_output_tokens
        * data.model.map(lambda model: prices[model]["output"])
    ) / 1_000_000
    return data, prices


def saturation_flags(matrices: dict) -> dict:
    return {
        key: set(matrix.index[(matrix == 1).all(axis=1)])
        for key, matrix in matrices.items()
    }


def select(
    matrices: dict,
    costs: dict,
    saturated: dict,
    k: int,
    method: str,
    subjects: list[str] | None = None,
    *,
    rng: np.random.Generator | None = None,
    cost_tolerance: float = 0.0,
) -> dict:
    result = {}
    for key, matrix in matrices.items():
        pool = (
            matrix.drop(index=list(saturated[key]))
            if method.startswith("unsaturated")
            else matrix
        )
        if len(pool) < k:
            raise ValueError(f"{key} has only {len(pool)} eligible tasks for k={k}")
        if method in {"random", "unsaturated_random"}:
            assert rng is not None
            result[key] = tuple(
                sorted(rng.choice(pool.index.to_numpy(), size=k, replace=False))
            )
            continue
        training = list(pool.columns) if subjects is None else subjects
        profile = pool.loc[:, training].to_numpy(dtype=float)
        target = matrix.loc[:, training].mean(axis=0).to_numpy(dtype=float)
        cost_by_task = (
            costs[key].loc[pool.index, training].sum(axis=1).to_numpy(dtype=float)
        )
        best = None
        candidates = []
        rank_method = method in {
            "level_rank",
            "unsaturated_level_rank",
            "guarded_level_rank",
            "unsaturated_guarded_level_rank",
        }
        for indexes in itertools.combinations(range(len(pool)), k):
            candidate_scores = profile[list(indexes)].mean(axis=0)
            rms = float(np.sqrt(np.mean((candidate_scores - target) ** 2)))
            max_error = float(np.max(abs(candidate_scores - target)))
            cost = float(cost_by_task[list(indexes)].sum())
            if rank_method:
                gaps = [
                    (
                        target[i] - target[j],
                        candidate_scores[i] - candidate_scores[j],
                    )
                    for i, j in itertools.combinations(range(len(training)), 2)
                ]
                material_losses = sum(
                    abs(full_gap) >= 0.05 and full_gap * mini_gap <= 0
                    for full_gap, mini_gap in gaps
                )
                all_losses = sum(
                    full_gap != 0 and full_gap * mini_gap <= 0
                    for full_gap, mini_gap in gaps
                )
            else:
                material_losses = all_losses = 0
            candidates.append(
                (rms, cost, indexes, material_losses, all_losses, max_error)
            )
            if best is None or (rms, indexes) < (best[0], best[2]):
                best = (rms, cost, indexes)
        assert best is not None
        if method in {"profile", "unsaturated_profile"}:
            chosen = best
        elif method in {"level_rank", "unsaturated_level_rank"}:
            chosen = min(
                candidates,
                key=lambda item: (item[3], item[4], item[0], item[1], item[2]),
            )
        elif method in {"guarded_level_rank", "unsaturated_guarded_level_rank"}:
            smallest_max_error = min(item[5] for item in candidates)
            eligible = [
                item
                for item in candidates
                if item[5] <= smallest_max_error + 0.02 + 1e-12
            ]
            chosen = min(
                eligible,
                key=lambda item: (item[3], item[4], item[0], item[1], item[2]),
            )
        elif method == "cheapest":
            chosen = min(candidates, key=lambda item: (item[1], item[0], item[2]))
        elif method in {"cost_tolerant", "unsaturated_cost_tolerant"}:
            eligible = [
                item
                for item in candidates
                if item[0] <= best[0] + cost_tolerance + 1e-12
            ]
            chosen = min(eligible, key=lambda item: (item[1], item[0], item[2]))
        else:
            raise ValueError(f"unknown method: {method}")
        result[key] = tuple(str(pool.index[index]) for index in chosen[2])
    return result


def evaluate(
    matrices: dict,
    costs: dict,
    saturated: dict,
    subjects: list[str],
    common: list,
    full: np.ndarray,
    selected: dict,
) -> dict:
    mini = scores(matrices, common, subjects, selected)
    full_usd = float(sum(cost.to_numpy().sum() for cost in costs.values()))
    selected_usd = float(
        sum(
            cost.loc[list(selected[key])].to_numpy().sum()
            for key, cost in costs.items()
        )
    )
    reversals = 0
    material_reversals = 0
    for i, j in itertools.combinations(range(len(subjects)), 2):
        full_gap = full[i] - full[j]
        if full_gap * (mini[i] - mini[j]) < 0:
            reversals += 1
            material_reversals += int(abs(full_gap) >= 0.05)
    level_correlations = []
    level_material_losses = 0
    levels_with_material_loss = 0
    worst_level_error = 0.0
    for key in common:
        matrix = matrices[key].loc[:, subjects]
        level_full = matrix.mean(axis=0).to_numpy(dtype=float)
        level_mini = matrix.loc[list(selected[key])].mean(axis=0).to_numpy(dtype=float)
        worst_level_error = max(
            worst_level_error, float(np.max(abs(level_full - level_mini)))
        )
        if np.std(level_full) > 0:
            level_correlations.append(spearman(level_full, level_mini))
        losses = int(
            sum(
                abs(level_full[i] - level_full[j]) >= 0.05
                and (level_full[i] - level_full[j]) * (level_mini[i] - level_mini[j])
                <= 0
                for i, j in itertools.combinations(range(len(subjects)), 2)
            )
        )
        level_material_losses += losses
        levels_with_material_loss += int(losses > 0)
    return {
        "spearman": spearman(full, mini),
        "max_absolute_score_error": float(np.max(abs(full - mini))),
        "pair_reversals": reversals,
        "material_pair_reversals": material_reversals,
        "estimated_usd": selected_usd,
        "cost_fraction_of_full": selected_usd / full_usd,
        "saturated_tasks": sum(
            len(set(ids) & saturated[key]) for key, ids in selected.items()
        ),
        "informative_levels": len(level_correlations),
        "mean_level_spearman": float(np.mean(level_correlations)),
        "worst_level_spearman": float(np.min(level_correlations)),
        "worst_level_score_error": worst_level_error,
        "level_material_pair_losses": level_material_losses,
        "levels_with_material_pair_loss": levels_with_material_loss,
    }


def level_details(matrices: dict, subjects: list[str], selected: dict) -> dict:
    details = {}
    for (environment, level), matrix in matrices.items():
        scored = matrix.loc[:, subjects]
        full = scored.mean(axis=0).to_numpy(dtype=float)
        mini = (
            scored.loc[list(selected[(environment, level)])]
            .mean(axis=0)
            .to_numpy(dtype=float)
        )
        losses = int(
            sum(
                abs(full[i] - full[j]) >= 0.05
                and (full[i] - full[j]) * (mini[i] - mini[j]) <= 0
                for i, j in itertools.combinations(range(len(subjects)), 2)
            )
        )
        details[f"{environment}/level_{level}"] = {
            "full_pass_at_5": dict(zip(subjects, map(float, full), strict=True)),
            "mini_pass_at_5": dict(zip(subjects, map(float, mini), strict=True)),
            "spearman": spearman(full, mini) if np.std(full) > 0 else None,
            "max_absolute_score_error": float(np.max(abs(full - mini))),
            "material_pair_losses": losses,
        }
    return details


def held_out(
    matrices: dict,
    costs: dict,
    saturated: dict,
    subjects: list[str],
    common: list,
    full: np.ndarray,
    k: int,
    method: str,
    cost_tolerance: float = 0.0,
) -> dict:
    errors = []
    reversals = 0
    material_reversals = 0
    level_errors = []
    level_material_losses = 0
    levels_with_loss = set()
    for held_index, held_subject in enumerate(subjects):
        train = [subject for subject in subjects if subject != held_subject]
        training_saturated = (
            saturation_flags(
                {key: matrix.loc[:, train] for key, matrix in matrices.items()}
            )
            if method.startswith("unsaturated")
            else saturated
        )
        selected = select(
            matrices,
            costs,
            training_saturated,
            k,
            method,
            train,
            cost_tolerance=cost_tolerance,
        )
        mini = scores(matrices, common, subjects, selected)
        errors.append(abs(mini[held_index] - full[held_index]))
        for key in common:
            matrix = matrices[key].loc[:, subjects]
            level_full = matrix.mean(axis=0).to_numpy(dtype=float)
            level_mini = (
                matrix.loc[list(selected[key])].mean(axis=0).to_numpy(dtype=float)
            )
            level_errors.append(abs(level_mini[held_index] - level_full[held_index]))
            for other_index in range(len(subjects)):
                if held_index == other_index:
                    continue
                gap = level_full[held_index] - level_full[other_index]
                if (
                    abs(gap) >= 0.05
                    and gap * (level_mini[held_index] - level_mini[other_index]) <= 0
                ):
                    level_material_losses += 1
                    levels_with_loss.add(key)
        for other_index in range(len(subjects)):
            if held_index == other_index:
                continue
            full_gap = full[held_index] - full[other_index]
            if full_gap * (mini[held_index] - mini[other_index]) < 0:
                reversals += 1
                material_reversals += int(abs(full_gap) >= 0.05)
    return {
        "held_out_mean_absolute_score_error": float(np.mean(errors)),
        "held_out_max_absolute_score_error": float(np.max(errors)),
        "held_out_pair_reversals": reversals,
        "held_out_material_pair_reversals": material_reversals,
        "held_out_max_level_score_error": float(np.max(level_errors)),
        "held_out_level_material_pair_losses": level_material_losses,
        "held_out_levels_with_material_pair_loss": len(levels_with_loss),
    }


def held_out_models(
    matrices: dict,
    costs: dict,
    saturated: dict,
    subjects: list[str],
    common: list,
    full: np.ndarray,
    k: int,
    method: str,
    cost_tolerance: float = 0.0,
) -> dict:
    errors = []
    reversals = 0
    material_reversals = 0
    level_errors = []
    level_material_losses = 0
    levels_with_loss = set()
    models = sorted({subject.split("__", 1)[1] for subject in subjects})
    for model in models:
        held = [
            i for i, subject in enumerate(subjects) if subject.endswith(f"__{model}")
        ]
        train = [subject for subject in subjects if not subject.endswith(f"__{model}")]
        training_saturated = (
            saturation_flags(
                {key: matrix.loc[:, train] for key, matrix in matrices.items()}
            )
            if method.startswith("unsaturated")
            else saturated
        )
        selected = select(
            matrices,
            costs,
            training_saturated,
            k,
            method,
            train,
            cost_tolerance=cost_tolerance,
        )
        mini = scores(matrices, common, subjects, selected)
        for held_index in held:
            errors.append(abs(mini[held_index] - full[held_index]))
            for key in common:
                matrix = matrices[key].loc[:, subjects]
                level_full = matrix.mean(axis=0).to_numpy(dtype=float)
                level_mini = (
                    matrix.loc[list(selected[key])].mean(axis=0).to_numpy(dtype=float)
                )
                level_errors.append(
                    abs(level_mini[held_index] - level_full[held_index])
                )
                for other_index in range(len(subjects)):
                    if held_index == other_index:
                        continue
                    gap = level_full[held_index] - level_full[other_index]
                    if (
                        abs(gap) >= 0.05
                        and gap * (level_mini[held_index] - level_mini[other_index])
                        <= 0
                    ):
                        level_material_losses += 1
                        levels_with_loss.add(key)
            for other_index in range(len(subjects)):
                if held_index == other_index:
                    continue
                full_gap = full[held_index] - full[other_index]
                if full_gap * (mini[held_index] - mini[other_index]) < 0:
                    reversals += 1
                    material_reversals += int(abs(full_gap) >= 0.05)
    return {
        "held_out_model_mean_absolute_score_error": float(np.mean(errors)),
        "held_out_model_max_absolute_score_error": float(np.max(errors)),
        "held_out_model_pair_reversals": reversals,
        "held_out_model_material_pair_reversals": material_reversals,
        "held_out_model_max_level_score_error": float(np.max(level_errors)),
        "held_out_model_level_material_pair_losses": level_material_losses,
        "held_out_model_levels_with_material_pair_loss": len(levels_with_loss),
    }


def summarize_random(results: list[dict]) -> dict:
    summary = {}
    for key in results[0]:
        values = [result[key] for result in results]
        summary[key] = {
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return summary


def main() -> None:
    args = parse_args()
    data, digest = read_tasks(args)
    data = data[data.environment != "afm"].copy()
    data, prices = add_cost(data, args.prices)
    matrices = make_matrices(data, "pass_at_5")
    costs = make_matrices(data, "estimated_cost_usd")
    saturated = saturation_flags(matrices)
    subjects, common = ranking_panel(matrices)
    if len(common) != len(matrices):
        raise ValueError("comparison requires every agent on every included stratum")
    k = args.tasks_per_stratum
    if k < 1 or k > min(
        len(matrix) - len(saturated[key]) for key, matrix in matrices.items()
    ):
        raise ValueError("k exceeds an unsaturated task pool")
    full = scores(matrices, common, subjects)
    rng = np.random.default_rng(args.seed)
    comparison = {}
    for method in ("random", "unsaturated_random"):
        results = [
            evaluate(
                matrices,
                costs,
                saturated,
                subjects,
                common,
                full,
                select(matrices, costs, saturated, k, method, rng=rng),
            )
            for _ in range(args.repeats)
        ]
        comparison[method] = {
            "repeats": args.repeats,
            "summary": summarize_random(results),
        }
    strategies = [
        ("profile", "profile", 0.0),
        ("unsaturated_profile", "unsaturated_profile", 0.0),
        ("level_rank", "level_rank", 0.0),
        ("unsaturated_level_rank", "unsaturated_level_rank", 0.0),
        ("guarded_level_rank", "guarded_level_rank", 0.0),
        ("unsaturated_guarded_level_rank", "unsaturated_guarded_level_rank", 0.0),
        ("cheapest", "cheapest", 0.0),
        ("cost_tolerant_0.02", "cost_tolerant", 0.02),
        ("cost_tolerant_0.05", "cost_tolerant", 0.05),
        ("unsaturated_cost_tolerant_0.02", "unsaturated_cost_tolerant", 0.02),
        ("unsaturated_cost_tolerant_0.05", "unsaturated_cost_tolerant", 0.05),
    ]
    for name, method, tolerance in strategies:
        selected = select(
            matrices, costs, saturated, k, method, cost_tolerance=tolerance
        )
        comparison[name] = {
            "metrics": evaluate(
                matrices, costs, saturated, subjects, common, full, selected
            ),
            "held_out": held_out(
                matrices, costs, saturated, subjects, common, full, k, method, tolerance
            ),
            "held_out_models": held_out_models(
                matrices, costs, saturated, subjects, common, full, k, method, tolerance
            ),
            "tasks_by_stratum": {
                f"{env}/level_{level}": list(ids)
                for (env, level), ids in selected.items()
            },
            "levels": level_details(matrices, subjects, selected),
        }
        logger.info("{}: {}", name, comparison[name]["metrics"])
    output = {
        "source": {
            "repo": "jablonkagroup/corral-reports-october-runs-summary-v1",
            "revision": args.revision,
            "sha256": digest,
        },
        "prices": prices,
        "excluded_environments": ["afm"],
        "tasks_per_stratum": k,
        "strata": len(matrices),
        "subjects": subjects,
        "saturated_tasks_by_stratum": {
            f"{env}/level_{level}": sorted(saturated[(env, level)])
            for env, level in matrices
        },
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("Saved {}", args.output)


if __name__ == "__main__":
    main()
