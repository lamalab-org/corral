"""Leave-one-subject-out (LOSO) cross-validation for the subsampling methods.

Everything evaluated so far scored subjects using item sets that, for the
IRT-based methods, were selected using a 2PL model FIT ON ALL 12 SUBJECTS —
including the one being scored. That's in-sample and doesn't tell us how the
method performs on a model it's never seen, which is the actual use case for
corral-mini (choose items now, evaluate a genuinely new model on them later).

This script re-runs the comparison honestly: for each of the 12 subjects,
hold it out, refit the 2PL model on only the remaining 11, build item
selection from THAT fit, then score the held-out subject with it (using its
real, already-collected full-benchmark data — we're not hiding data the
method needs to be evaluated, only data it's allowed to use while designing
the item set).

random / stratified_proportional don't need refitting — item selection never
looks at subject response data — so their "held-out" numbers are exactly
their ordinary sweep numbers by construction. That's a feature, not a
shortcut: it's the direct, concrete demonstration of the design-based
(nothing to leak) vs. model-based (something to leak) distinction.

Usage:
    python run_loso_cv.py
    python run_loso_cv.py --n_repeats_random=100
"""

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from fit_irt_2pl import build_trial_counts, fit_2pl  # noqa: E402
from subsampling_core import SAMPLERS, _greedy_info_order, allocate_budget, load_matrix  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def evaluate_held_out(matrix: pd.DataFrame, selected: list[str], held_out: str) -> tuple[float, float]:
    true_global = matrix.mean(axis=1)
    mini_global = matrix[selected].mean(axis=1)
    sq_err = float((mini_global[held_out] - true_global[held_out]) ** 2)
    rho = spearmanr(true_global, mini_global)[0]
    return sq_err, rho


def main(
    n_repeats_random: int = 50,
    budget_min: int = 5,
    budget_max: int = 90,
    budget_step: int = 5,
    exclude_environments: str = "resistor",
    seed: int = 0,
) -> None:
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    matrix, col_env = load_matrix(value="success", exclude_environments=excluded)
    all_items_list = list(matrix.columns)
    subjects = list(matrix.index)
    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= matrix.shape[1]]
    logger.info(f"{len(subjects)} subjects, {len(all_items_list)} items, budgets={budgets}")

    rng = np.random.default_rng(seed)
    rows = []

    for fold_i, held_out in enumerate(subjects):
        logger.info(f"[{fold_i + 1}/{len(subjects)}] held-out subject: {held_out}")

        # ---- refit 2PL excluding held_out, build IRT-based item selections ----
        fit_subjects, fit_items, K, N, item_env = build_trial_counts(exclude_environments=excluded)
        keep_idx = [i for i, s in enumerate(fit_subjects) if s != held_out]
        theta, a, b = fit_2pl(K[keep_idx], N[keep_idx], seed=seed)
        item_df = pd.DataFrame({"item": fit_items, "a": a, "b": b, "environment": item_env.values})
        grid = theta  # abilities of the 11 training subjects only

        irt_order = _greedy_info_order(
            item_df["a"].to_numpy(), item_df["b"].to_numpy(), item_df["item"].to_numpy(), grid, budget_max
        )
        per_env_order = {
            env: _greedy_info_order(sub["a"].to_numpy(), sub["b"].to_numpy(), sub["item"].to_numpy(), grid, len(sub))
            for env, sub in item_df.groupby("environment")
        }
        strata_sizes = {env: len(items) for env, items in per_env_order.items()}

        for budget in budgets:
            sq_err, rho = evaluate_held_out(matrix, irt_order[:budget], held_out)
            rows.append(dict(method="irt_maxinfo", held_out=held_out, budget=budget, mse=sq_err, spearman=rho))

            alloc = allocate_budget(strata_sizes, budget, mode="proportional")
            sel = [item for env, n in alloc.items() for item in per_env_order[env][:n]]
            sq_err, rho = evaluate_held_out(matrix, sel, held_out)
            rows.append(dict(method="stratified_irt", held_out=held_out, budget=budget, mse=sq_err, spearman=rho))

        # ---- design-based methods: no refit, average over repeated random draws ----
        for method_name in ["random", "stratified_proportional"]:
            sampler = SAMPLERS[method_name]
            for budget in budgets:
                errs, rhos = [], []
                for _ in range(n_repeats_random):
                    sel = sampler(all_items_list, budget, rng, col_env)
                    sq_err, rho = evaluate_held_out(matrix, sel, held_out)
                    errs.append(sq_err)
                    rhos.append(rho)
                rows.append(
                    dict(
                        method=method_name,
                        held_out=held_out,
                        budget=budget,
                        mse=float(np.mean(errs)),
                        spearman=float(np.mean(rhos)),
                    )
                )

    results = pd.DataFrame(rows)
    out = DATA_DIR / "loso_cv_results.csv"
    results.to_csv(out, index=False)
    logger.success(f"Saved {len(results)} rows -> {out}")

    summary = results.groupby(["method", "budget"]).agg(mse=("mse", "mean"), rho=("spearman", "mean")).reset_index()
    logger.info("\n" + summary.pivot(index="budget", columns="method", values="mse").round(4).to_string())
    logger.info("\n" + summary.pivot(index="budget", columns="method", values="rho").round(3).to_string())


if __name__ == "__main__":
    fire.Fire(main)
