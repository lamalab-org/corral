"""Final 3-method protocol comparison with proper error bars on all three.

random / stratified_proportional: error bars from repeated random draws (the
usual item-sampling variance, as in every earlier sweep).

stratified_bayes_irt: error bars from bootstrap-resampling the posterior
draws themselves — resample 4000 draws WITH replacement from the fitted
trace's 4000 available draws, rebuild the item order from that resampled
posterior average, repeat n_boot times. This reflects MCMC/posterior
estimation uncertainty (“how much would this design differ under a slightly
different posterior sample”), the correct analogue of sampling variance for
a method whose selection depends on fitted parameters rather than being
purely random. Using the full 4000 draws per resample (not a 200-draw
subsample) since there's no speed constraint here — each resample is fully
vectorized, no per-draw Python loop.

Usage:
    python run_final_protocol_comparison.py
"""

import sys
from pathlib import Path

import arviz as az
import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from fit_irt_2pl import build_trial_counts  # noqa: E402
from subsampling_core import SAMPLERS, allocate_budget, greedy_order_from_info_matrix, load_matrix  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NEW_MODELS = ["claude-opus-4.8", "deepseek-v3.2", "kimi-k2.5"]


def _rho(true: pd.Series, mini: pd.Series, subset: list[str] | None = None) -> float:
    if subset is not None:
        true, mini = true[subset], mini[subset]
    return spearmanr(true, mini)[0]


def vectorized_info(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fisher info for every (item, subject, draw), averaged over draws.

    theta: (n_subj, n_draws)   a, b: (n_item, n_draws)   ->   returns (n_item, n_subj)
    """
    logits = a[:, None, :] * (theta[None, :, :] - b[:, None, :])  # (n_item, n_subj, n_draws)
    p = 1 / (1 + np.exp(-logits))
    info = (a[:, None, :] ** 2) * p * (1 - p)
    return info.mean(axis=2)


def main(
    n_repeats: int = 200,
    n_boot: int = 50,
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
    new_subjects = [s for s in subjects if s.split("__")[0] in NEW_MODELS]

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= matrix.shape[1]]

    rng = np.random.default_rng(seed)
    rows = []

    for method_name in ["random", "stratified_proportional"]:
        sampler = SAMPLERS[method_name]
        for budget in budgets:
            for rep in range(n_repeats):
                sel = sampler(all_items_list, budget, rng, col_env)
                mini = matrix[sel].mean(axis=1)
                rows.append(
                    dict(
                        method=method_name,
                        budget=budget,
                        repeat=rep,
                        rho_all=_rho(true_global, mini),
                        mse_new=float(((mini[new_subjects] - true_global[new_subjects]) ** 2).mean()),
                    )
                )
        logger.info(f"{method_name}: {n_repeats} repeats x {len(budgets)} budgets done")

    logger.info(f"Loading Bayesian trace and bootstrap-resampling ({n_boot} resamples, full 4000 draws each)...")
    trace = az.from_netcdf(DATA_DIR / "irt_2pl_bayesian_trace_legacy.nc")
    _, fit_items, _, _, item_env = build_trial_counts(exclude_environments=excluded)
    theta_all = trace.posterior["theta"].stack(sample=("chain", "draw")).to_numpy()  # (6, 4000)
    a_all = trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()  # (94, 4000)
    b_all = trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()  # (94, 4000)
    n_total = theta_all.shape[1]
    items_arr = np.array(fit_items)
    env_arr = item_env.reindex(fit_items).to_numpy()

    for boot in range(n_boot):
        idx = rng.choice(n_total, size=n_total, replace=True)  # classic bootstrap resample
        avg_info = vectorized_info(theta_all[:, idx], a_all[:, idx], b_all[:, idx])  # (n_item, n_subj)

        per_env_order = {}
        for env in np.unique(env_arr):
            mask = env_arr == env
            per_env_order[env] = greedy_order_from_info_matrix(avg_info[mask], items_arr[mask], int(mask.sum()))
        strata_sizes = {env: len(v) for env, v in per_env_order.items()}

        for budget in budgets:
            alloc = allocate_budget(strata_sizes, budget, mode="proportional")
            sel = [it for env, n in alloc.items() for it in per_env_order[env][:n]]
            mini = matrix[sel].mean(axis=1)
            rows.append(
                dict(
                    method="stratified_bayes_irt",
                    budget=budget,
                    repeat=boot,
                    rho_all=_rho(true_global, mini),
                    mse_new=float(((mini[new_subjects] - true_global[new_subjects]) ** 2).mean()),
                )
            )
        if (boot + 1) % 10 == 0:
            logger.info(f"  bootstrap {boot + 1}/{n_boot} done")

    results = pd.DataFrame(rows)
    out = DATA_DIR / "final_protocol_comparison_results.csv"
    results.to_csv(out, index=False)
    logger.success(f"Saved {len(results)} rows -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
