"""Final ranking-preservation comparison across all corral-mini sampling
methods, with COMPARABLE error bars.

Methods compared:
  - random, stratified_equal, stratified_proportional,
    stratified_proportional_nested   (subsampling_core.SAMPLERS — item sets
    drawn fresh at random every replicate)
  - stratified_bayes_irt              (MCMC posterior-averaged Fisher info,
    bootstrap-resampled from fit_irt_2pl_bayesian.py's legacy-only trace.
    No MAP anywhere: a single point-estimate (a, b) fit turned out too
    noisy — see bayesian_irt_selection.py's docstring — so item selection
    always averages Fisher information over posterior draws instead.)

Ranking preservation ONLY (no MSE — corral-mini's job is to reproduce the
subject ranking, not the absolute scores). Reported at two subject scopes:
  - all: all 12 subjects (6 legacy + 6 new)
  - new: the 6 new-model subjects only (claude-opus-4.8 / deepseek-v3.2 /
    kimi-k2.5 x 2 scaffolds) — the actual generalization test, since the
    item sets are chosen without the new models' data influencing selection
    (stratified_bayes_irt fits on legacy only; the others never look at
    response data at all)
and two granularities:
  - global: one Spearman rho over item-pooled mini-set scores
  - per_env: Spearman rho computed separately within each environment, then
    averaged across environments (does ranking hold up per-environment, not
    just in aggregate)

Comparable error bars: every method gets exactly `n_draws` independent
replicates at every budget --
  - random/stratified_*  : n_draws independent random item draws
  - stratified_bayes_irt : n_draws independent bootstrap resamples (with
    replacement) of the 4000 posterior draws, each producing its own
    posterior-averaged item order
so the resulting std-dev bands measure the same thing (variability across
n_draws independently-constructed mini-sets) for every method — comparing a
200-draw random std against a 50-boot IRT std, as earlier versions of this
comparison did, understated the IRT method's actual uncertainty.

Usage:
    python run_final_ranking_comparison.py
    python run_final_ranking_comparison.py --n_draws=200 --budget_max=90
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
LEGACY_MODELS = ["claude-4.5", "gpt-4o", "gpt-oss-120b"]
NEW_MODELS = ["claude-opus-4.8", "deepseek-v3.2", "kimi-k2.5"]


def _rho(true: pd.Series, mini: pd.Series) -> float:
    if len(true) < 3:
        return np.nan
    return spearmanr(true, mini)[0]


def ranking_metrics(
    matrix: pd.DataFrame, col_env: pd.Series, selected: list[str], subject_groups: dict[str, list[str]]
) -> dict:
    """rho at (global, per_env) x each subject group, for one selected item set."""
    true_global = matrix.mean(axis=1)
    mini_global = matrix[selected].mean(axis=1)
    selected_set = set(selected)
    envs = sorted(col_env.unique())

    out = {}
    for group_name, subjects in subject_groups.items():
        out[f"rho_{group_name}_global"] = _rho(true_global[subjects], mini_global[subjects])

        per_env = []
        for env in envs:
            env_items = col_env[col_env == env].index
            env_items_sel = [c for c in env_items if c in selected_set]
            if not env_items_sel:
                continue
            true_env = matrix[env_items].mean(axis=1)
            mini_env = matrix[env_items_sel].mean(axis=1)
            per_env.append(_rho(true_env[subjects], mini_env[subjects]))
        out[f"rho_{group_name}_per_env"] = float(np.nanmean(per_env)) if per_env else np.nan
    return out


def vectorized_info(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fisher info for every (item, subject, draw), averaged over draws.

    theta: (n_subj, n_draws)   a, b: (n_item, n_draws)   ->   (n_item, n_subj)
    """
    logits = a[:, None, :] * (theta[None, :, :] - b[:, None, :])
    p = 1 / (1 + np.exp(-logits))
    info = (a[:, None, :] ** 2) * p * (1 - p)
    return info.mean(axis=2)


def main(
    n_draws: int = 200,
    budget_min: int = 5,
    budget_max: int = 90,
    budget_step: int = 5,
    exclude_environments: str = "resistor",
    trace_path: str | None = None,
    seed: int = 0,
) -> None:
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    matrix, col_env = load_matrix(value="success", exclude_environments=excluded)
    all_items = list(matrix.columns)
    subjects = list(matrix.index)
    subject_groups = {
        "all": subjects,
        "new": [s for s in subjects if s.split("__")[0] in NEW_MODELS],
    }
    logger.info(
        f"{len(all_items)} candidate items, {len(subjects)} subjects "
        f"({len(subject_groups['new'])} new-model subjects)"
    )

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= len(all_items)]

    rng = np.random.default_rng(seed)
    rows = []

    # ---- random-draw methods: n_draws independent draws per budget ----
    for method_name, sampler in SAMPLERS.items():
        for budget in budgets:
            for draw in range(n_draws):
                sel = sampler(all_items, budget, rng, col_env)
                m = ranking_metrics(matrix, col_env, sel, subject_groups)
                rows.append(dict(method=method_name, budget=budget, draw=draw, **m))
        logger.info(f"{method_name}: {n_draws} draws x {len(budgets)} budgets done")

    # ---- stratified_bayes_irt: n_draws bootstrap resamples of the MCMC posterior ----
    trace_p = Path(trace_path) if trace_path else DATA_DIR / "irt_2pl_bayesian_trace_legacy.nc"
    logger.info(f"Loading MCMC trace from {trace_p}")
    trace = az.from_netcdf(trace_p)
    _, fit_items, _, _, item_env = build_trial_counts(exclude_environments=excluded)
    theta_all = trace.posterior["theta"].stack(sample=("chain", "draw")).to_numpy()
    a_all = trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()
    b_all = trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()
    n_total = theta_all.shape[1]
    items_arr = np.array(fit_items)
    env_arr = item_env.reindex(fit_items).to_numpy()

    for draw in range(n_draws):
        idx = rng.choice(n_total, size=n_total, replace=True)  # classic bootstrap resample
        avg_info = vectorized_info(theta_all[:, idx], a_all[:, idx], b_all[:, idx])

        per_env_order = {}
        for env in np.unique(env_arr):
            mask = env_arr == env
            per_env_order[env] = greedy_order_from_info_matrix(avg_info[mask], items_arr[mask], int(mask.sum()))
        strata_sizes = {env: len(v) for env, v in per_env_order.items()}

        for budget in budgets:
            alloc = allocate_budget(strata_sizes, budget, mode="proportional")
            sel = [it for env, n in alloc.items() for it in per_env_order[env][:n]]
            m = ranking_metrics(matrix, col_env, sel, subject_groups)
            rows.append(dict(method="stratified_bayes_irt", budget=budget, draw=draw, **m))
        if (draw + 1) % 50 == 0:
            logger.info(f"  stratified_bayes_irt bootstrap {draw + 1}/{n_draws} done")

    results = pd.DataFrame(rows)
    out = DATA_DIR / "final_ranking_comparison_results.csv"
    results.to_csv(out, index=False)
    logger.success(f"Saved {len(results)} rows -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
