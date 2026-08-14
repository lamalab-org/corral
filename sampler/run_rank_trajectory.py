"""Per-new-model rank trajectory: does EACH new model's rank stay correct?

Same setup as run_legacy_to_new_cv.py — item set fit ONCE on the 3 legacy
models, never touching the 3 new models — but instead of collapsing ranking
preservation into one Spearman rho per budget, this tracks each of the 6 new
(model, scaffold) subjects' individual RANK (1=best...6=worst, among the 6
new subjects) as budget grows. rho=0.94 could mean "everyone's slightly
off" or "five are perfect and one is badly wrong"; this shows which.

random / stratified_proportional are stochastic, so their rank per budget is
the MEDIAN rank across repeated draws (not a single arbitrary draw).
stratified_irt is deterministic given the legacy-only fit.

Usage:
    python run_rank_trajectory.py
"""

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from fit_irt_2pl import build_trial_counts, fit_2pl  # noqa: E402
from subsampling_core import SAMPLERS, _greedy_info_order, allocate_budget, load_matrix  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
LEGACY_MODELS = ["claude-4.5", "gpt-4o", "gpt-oss-120b"]
NEW_MODELS = ["claude-opus-4.8", "deepseek-v3.2", "kimi-k2.5"]


def ranks_among(scores: pd.Series, subset: list[str]) -> pd.Series:
    """Rank 1 = highest score = best."""
    return scores[subset].rank(ascending=False, method="average")


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
    new_subjects = [s for s in subjects if s.split("__")[0] in NEW_MODELS]
    true_rank = ranks_among(true_global, new_subjects)
    logger.info("True rank among the 6 new (model, scaffold) subjects:\n" + true_rank.sort_values().to_string())

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    budgets = [b for b in budgets if b <= matrix.shape[1]]

    # fit 2PL on legacy models only (identical setup to run_legacy_to_new_cv.py)
    fit_subjects, fit_items, K, N, item_env = build_trial_counts(exclude_environments=excluded)
    keep_idx = [i for i, s in enumerate(fit_subjects) if s.split("__")[0] in LEGACY_MODELS]
    logger.info(f"Fitting 2PL on {len(keep_idx)}/{len(fit_subjects)} subjects (legacy only)")
    theta, a, b = fit_2pl(K[keep_idx], N[keep_idx], seed=seed)
    item_df = pd.DataFrame({"item": fit_items, "a": a, "b": b, "environment": item_env.values})
    grid = theta

    per_env_order = {
        env: _greedy_info_order(sub["a"].to_numpy(), sub["b"].to_numpy(), sub["item"].to_numpy(), grid, len(sub))
        for env, sub in item_df.groupby("environment")
    }
    strata_sizes = {env: len(items) for env, items in per_env_order.items()}

    rng = np.random.default_rng(seed)
    rows = []

    for budget in budgets:
        alloc = allocate_budget(strata_sizes, budget, mode="proportional")
        sel = [item for env, n in alloc.items() for item in per_env_order[env][:n]]
        mini_rank = ranks_among(matrix[sel].mean(axis=1), new_subjects)
        for s in new_subjects:
            rows.append(dict(method="stratified_irt", subject=s, budget=budget, rank=mini_rank[s], true_rank=true_rank[s]))

        for method_name in ["random", "stratified_proportional"]:
            sampler = SAMPLERS[method_name]
            rank_draws = {s: [] for s in new_subjects}
            for _ in range(n_repeats_random):
                sel = sampler(all_items_list, budget, rng, col_env)
                mr = ranks_among(matrix[sel].mean(axis=1), new_subjects)
                for s in new_subjects:
                    rank_draws[s].append(mr[s])
            for s in new_subjects:
                rows.append(
                    dict(
                        method=method_name,
                        subject=s,
                        budget=budget,
                        rank=float(np.median(rank_draws[s])),
                        true_rank=true_rank[s],
                    )
                )

    results = pd.DataFrame(rows)
    out = DATA_DIR / "rank_trajectory_results.csv"
    results.to_csv(out, index=False)
    logger.success(f"Saved {len(results)} rows -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
