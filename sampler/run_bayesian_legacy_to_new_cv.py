"""Legacy-to-new CV for the Bayesian (posterior-averaged) item selection.

Same protocol as run_legacy_to_new_cv.py — item set built using ONLY the 3
legacy models, evaluated on the 3 models never seen during design — but
using bayes_maxinfo / stratified_bayes_irt (Fisher information averaged over
posterior draws, from fit_irt_2pl_bayesian.py's trace) instead of the
MAP-point-based irt_maxinfo / stratified_irt.

Requires irt_2pl_bayesian_trace_legacy.nc to already exist (run
fit_irt_2pl_bayesian.py --legacy_only=True first).

Output uses the identical schema as legacy_to_new_cv_results.csv, so the two
can be concatenated directly for a combined comparison plot.

Usage:
    python run_bayesian_legacy_to_new_cv.py
"""

import sys
from pathlib import Path

import arviz as az
import fire
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from bayesian_irt_selection import (  # noqa: E402
    build_bayes_item_order,
    build_bayes_item_order_per_stratum,
    make_stratified_bayes_sampler,
)
from fit_irt_2pl import build_trial_counts  # noqa: E402
from subsampling_core import allocate_budget, load_matrix  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
LEGACY_MODELS = ["claude-4.5", "gpt-4o", "gpt-oss-120b"]
NEW_MODELS = ["claude-opus-4.8", "deepseek-v3.2", "kimi-k2.5"]


def _rho(true: pd.Series, mini: pd.Series, subset: list[str] | None = None) -> float:
    if subset is not None:
        true, mini = true[subset], mini[subset]
    return spearmanr(true, mini)[0]


def main(
    trace_path: str | None = None,
    budget_min: int = 5,
    budget_max: int = 90,
    budget_step: int = 5,
    exclude_environments: str = "resistor",
    n_draws_used: int = 200,
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

    trace_p = Path(trace_path) if trace_path else DATA_DIR / "irt_2pl_bayesian_trace_legacy.nc"
    logger.info(f"Loading Bayesian trace from {trace_p}")
    trace = az.from_netcdf(trace_p)

    _, fit_items, _, _, item_env = build_trial_counts(exclude_environments=excluded)
    # fit_items (alphabetical) and matrix.columns (sorted by difficulty) can differ in
    # ORDER — fine, everything below indexes by item name, never by position.

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= matrix.shape[1]]

    logger.info(f"Building posterior-averaged item orders ({n_draws_used} draws)...")
    bayes_order = build_bayes_item_order(trace, fit_items, budget_max, n_draws_used=n_draws_used, seed=seed)
    per_env_order = build_bayes_item_order_per_stratum(trace, fit_items, item_env, n_draws_used=n_draws_used, seed=seed)
    strata_sizes = {env: len(items) for env, items in per_env_order.items()}

    rows = []

    def record(method: str, budget: int, mini: pd.Series) -> None:
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

    for budget in budgets:
        sel = bayes_order[:budget]
        record("bayes_maxinfo", budget, matrix[sel].mean(axis=1))

        alloc = allocate_budget(strata_sizes, budget, mode="proportional")
        sel = [item for env, n in alloc.items() for item in per_env_order[env][:n]]
        record("stratified_bayes_irt", budget, matrix[sel].mean(axis=1))

    results = pd.DataFrame(rows)
    out = DATA_DIR / "bayesian_legacy_to_new_cv_results.csv"
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
    logger.info("\n=== rho_all (ranking across all 12 subjects) ===")
    logger.info(
        "\n" + summary[summary.group == "new"].pivot(index="budget", columns="method", values="rho_all").round(3).to_string()
    )


if __name__ == "__main__":
    fire.Fire(main)
