"""Fit item selection on ONLY the 3 legacy models, evaluate on the 3 new ones.

This is a sharper, more honest generalization check than run_loso_cv.py.
LOSO holds out one of 12 subjects at a time, but the other 11 (including 2
of the 3 "new" models) are still in every fold's calibration set — it never
tests "design corral-mini with zero knowledge of any new-generation model."
This script does exactly that: fit the 2PL model using only
claude-4.5 / gpt-4o / gpt-oss-120b (x2 scaffolds = 6 subjects), build item
selection from THAT fit alone, then check how well it reconstructs scores
and preserves ranking for claude-opus-4.8 / deepseek-v3.2 / kimi-k2.5 — the
3 models it never saw during design.

The headline metric is rho_new_only: do the 3 new models rank correctly
AMONG THEMSELVES on the mini-set, using an item set chosen without ever
looking at their data. That's the most direct test of "will this work for a
new model" without ever needing to update the question set.

Usage:
    python run_legacy_to_new_cv.py
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
LEGACY_MODELS = ["claude-4.5", "gpt-4o", "gpt-oss-120b"]
NEW_MODELS = ["claude-opus-4.8", "deepseek-v3.2", "kimi-k2.5"]


def _rho(true: pd.Series, mini: pd.Series, subset: list[str] | None = None) -> float:
    if subset is not None:
        true, mini = true[subset], mini[subset]
    return spearmanr(true, mini)[0]


def main(
    n_repeats_random: int = 200,
    budget_min: int = 5,
    budget_max: int = 90,
    budget_step: int = 5,
    exclude_environments: str = "resistor",
    seed: int = 0,
) -> None:
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    matrix, col_env = load_matrix(value="success", exclude_environments=excluded)
    all_items_list = list(matrix.columns)
    true_global = matrix.mean(axis=1)

    subjects = list(matrix.index)
    group = {s: ("legacy" if s.split("__")[0] in LEGACY_MODELS else "new") for s in subjects}
    legacy_subjects = [s for s in subjects if group[s] == "legacy"]
    new_subjects = [s for s in subjects if group[s] == "new"]
    logger.info(f"legacy (fit on these): {legacy_subjects}")
    logger.info(f"new (never seen during fit): {new_subjects}")

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= matrix.shape[1]]

    # ---- fit 2PL using ONLY the 6 legacy subjects ----
    fit_subjects, fit_items, K, N, item_env = build_trial_counts(exclude_environments=excluded)
    keep_idx = [i for i, s in enumerate(fit_subjects) if s.split("__")[0] in LEGACY_MODELS]
    logger.info(f"Fitting 2PL on {len(keep_idx)}/{len(fit_subjects)} subjects (legacy only)")
    theta, a, b = fit_2pl(K[keep_idx], N[keep_idx], seed=seed)
    item_df = pd.DataFrame({"item": fit_items, "a": a, "b": b, "environment": item_env.values})
    grid = theta

    irt_order = _greedy_info_order(
        item_df["a"].to_numpy(), item_df["b"].to_numpy(), item_df["item"].to_numpy(), grid, budget_max
    )
    per_env_order = {
        env: _greedy_info_order(sub["a"].to_numpy(), sub["b"].to_numpy(), sub["item"].to_numpy(), grid, len(sub))
        for env, sub in item_df.groupby("environment")
    }
    strata_sizes = {env: len(items) for env, items in per_env_order.items()}

    rng = np.random.default_rng(seed)
    rows = []

    def record_deterministic(method: str, budget: int, mini: pd.Series) -> None:
        rho_all = _rho(true_global, mini)
        rho_new = _rho(true_global, mini, new_subjects)
        rho_legacy = _rho(true_global, mini, legacy_subjects)
        for s in subjects:
            rows.append(
                dict(
                    method=method,
                    subject=s,
                    group=group[s],
                    budget=budget,
                    sq_err=float((mini[s] - true_global[s]) ** 2),
                    rho_all=rho_all,
                    rho_new_only=rho_new,
                    rho_legacy_only=rho_legacy,
                )
            )

    def record_repeated(method: str, budget: int, sampler) -> None:
        # Correct MSE/rho requires computing the metric PER DRAW, then averaging the
        # metric — averaging the draws' scores first and computing one metric on that
        # average understates error by ~1/sqrt(n_repeats) (that was the bug: it measures
        # how well 200 mini-benchmarks average out, not how one actual deployment does).
        sq_errs = {s: [] for s in subjects}
        rho_all_list, rho_new_list, rho_legacy_list = [], [], []
        for _ in range(n_repeats_random):
            sel = sampler(all_items_list, budget, rng, col_env)
            mini = matrix[sel].mean(axis=1)
            for s in subjects:
                sq_errs[s].append(float((mini[s] - true_global[s]) ** 2))
            rho_all_list.append(_rho(true_global, mini))
            rho_new_list.append(_rho(true_global, mini, new_subjects))
            rho_legacy_list.append(_rho(true_global, mini, legacy_subjects))
        rho_all, rho_new, rho_legacy = np.mean(rho_all_list), np.mean(rho_new_list), np.mean(rho_legacy_list)
        for s in subjects:
            rows.append(
                dict(
                    method=method,
                    subject=s,
                    group=group[s],
                    budget=budget,
                    sq_err=float(np.mean(sq_errs[s])),
                    rho_all=rho_all,
                    rho_new_only=rho_new,
                    rho_legacy_only=rho_legacy,
                )
            )

    for budget in budgets:
        sel = irt_order[:budget]
        record_deterministic("irt_maxinfo", budget, matrix[sel].mean(axis=1))

        alloc = allocate_budget(strata_sizes, budget, mode="proportional")
        sel = [item for env, n in alloc.items() for item in per_env_order[env][:n]]
        record_deterministic("stratified_irt", budget, matrix[sel].mean(axis=1))

        for method_name in ["random", "stratified_proportional"]:
            record_repeated(method_name, budget, SAMPLERS[method_name])

    results = pd.DataFrame(rows)
    out = DATA_DIR / "legacy_to_new_cv_results.csv"
    results.to_csv(out, index=False)
    logger.success(f"Saved {len(results)} rows -> {out}")

    summary = (
        results.groupby(["method", "group", "budget"])
        .agg(mse=("sq_err", "mean"), rho_new_only=("rho_new_only", "first"), rho_all=("rho_all", "first"))
        .reset_index()
    )
    for grp in ["legacy", "new"]:
        logger.info(f"\n=== MSE, group={grp} ===")
        logger.info(
            "\n" + summary[summary.group == grp].pivot(index="budget", columns="method", values="mse").round(4).to_string()
        )
    logger.info("\n=== rho_new_only (ranking among the 3 unseen-at-design-time models) ===")
    logger.info(
        "\n" + summary[summary.group == "new"].pivot(index="budget", columns="method", values="rho_new_only").round(3).to_string()
    )


if __name__ == "__main__":
    fire.Fire(main)
