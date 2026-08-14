"""Core reusable pieces for the corral-mini subsampling harness.

Shared machinery: load the (subject x item) matrix, evaluate a candidate
item subset against the full set, and run a budget x repeats sweep for a
given sampler. Samplers register into SAMPLERS — add IRT / active-selection
samplers here later; run_sweep and evaluate_subset don't need to change.

Item = (environment, level, task); cell = mean success rate over the 5
trials, same construction as plot_mini_item_response_matrix.py.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ConstantInputWarning, kendalltau, spearmanr

# Expected: a sampled subset can have identical scores across all 12 subjects
# (e.g. one item, uniformly solved/failed by everyone) -> correlation is
# undefined by definition, not a bug. We already convert it to NaN and
# exclude it via nanmean downstream.
warnings.filterwarnings("ignore", category=ConstantInputWarning)

sys.path.insert(0, str(Path(__file__).parent))
from plot_sorted_response_matrix import build_matrix  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "corral_mini_source.csv"
ITEM_KEY = ["environment", "level", "task"]


def load_matrix(
    value: str = "success",
    exclude_environments: list[str] | None = None,
    exclude_models: list[str] | None = None,
    data_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path or DATA_PATH)
    if exclude_environments:
        df = df[~df["environment"].isin(exclude_environments)]
        logger.info(f"Excluded environments from item pool: {exclude_environments}")
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]
        logger.info(f"Excluded models from subject pool: {exclude_models}")
    matrix, col_env = build_matrix(df, value=value, item_key=ITEM_KEY)
    return matrix, col_env


# ---------------- samplers ----------------
# A sampler is: (all_items, budget, rng, col_env) -> list[str]. col_env (item -> environment)
# is unused by samplers that don't need stratum info, but every sampler takes it for a
# consistent call signature in run_sweep.


def random_sampler(
    all_items: list[str], budget: int, rng: np.random.Generator, col_env: pd.Series
) -> list[str]:
    return list(rng.choice(all_items, size=budget, replace=False))


def allocate_budget(
    strata_sizes: dict[str, int], budget: int, mode: str, min_per_stratum: int = 1
) -> dict[str, int]:
    """Split `budget` items across strata.

    First guarantees `min_per_stratum` to every stratum (capped at its size) so
    no stratum is starved by construction. Distributes the rest via the
    "highest averages" apportionment method (same family as D'Hondt/Jefferson
    seat allocation): repeatedly give the next item to whichever stratum has
    the smallest (allocated+1)/weight, which converges to proportional-to-
    weight allocation while respecting each stratum's size cap.

    mode: "proportional" (weight = stratum size, matches SRS in expectation)
          or "equal" (weight = 1 for every stratum).
    """
    strata = list(strata_sizes)
    alloc = {s: 0 for s in strata}
    remaining = budget
    for s in strata:
        take = min(min_per_stratum, strata_sizes[s], remaining)
        alloc[s] += take
        remaining -= take
        if remaining <= 0:
            return alloc

    if mode == "proportional":
        weights = {s: strata_sizes[s] for s in strata}
    elif mode == "equal":
        weights = {s: 1 for s in strata}
    else:
        raise ValueError(f"unknown allocation mode: {mode}")

    for _ in range(remaining):
        candidates = [s for s in strata if alloc[s] < strata_sizes[s]]
        if not candidates:
            break
        chosen = min(candidates, key=lambda s: (alloc[s] + 1) / weights[s])
        alloc[chosen] += 1
    return alloc


def make_stratified_sampler(mode: str, min_per_stratum: int = 1):
    def sampler(
        all_items: list[str], budget: int, rng: np.random.Generator, col_env: pd.Series
    ) -> list[str]:
        strata_items = {
            env: list(col_env[col_env == env].index) for env in col_env.unique()
        }
        strata_sizes = {env: len(items) for env, items in strata_items.items()}
        alloc = allocate_budget(
            strata_sizes, budget, mode=mode, min_per_stratum=min_per_stratum
        )
        selected = []
        for env, n in alloc.items():
            if n > 0:
                selected.extend(rng.choice(strata_items[env], size=n, replace=False))
        return selected

    return sampler


def make_stratified_sampler_nested(mode: str, min_per_stratum: int = 1):
    """Proportional allocation nested two levels deep: environment, then level
    within environment. Item strings are "environment|level|task" (ITEM_KEY
    order), so the level is parsed straight out of the item id — no separate
    level lookup table needed.

    Two-stage `allocate_budget` call: first split the total budget across
    environments exactly like `make_stratified_sampler`, then split each
    environment's share across its levels using the same proportional (or
    equal) rule. `min_per_stratum` is applied at BOTH stages (same value),
    so every non-empty (environment, level) combo gets at least that many
    items as long as the budget stretches that far -- note the environment
    stage needs enough headroom to cover `min_per_stratum * n_levels` for
    each environment, or the level stage's own minimum guarantee will run
    short on whatever's left; check the actual per-stratum output rather
    than assuming the nested minimum always holds exactly at small budgets.
    """

    def sampler(
        all_items: list[str], budget: int, rng: np.random.Generator, col_env: pd.Series
    ) -> list[str]:
        strata_items = {
            env: list(col_env[col_env == env].index) for env in col_env.unique()
        }
        strata_sizes = {env: len(items) for env, items in strata_items.items()}
        env_alloc = allocate_budget(
            strata_sizes, budget, mode=mode, min_per_stratum=min_per_stratum
        )

        selected = []
        for env, n in env_alloc.items():
            if n == 0:
                continue
            level_items: dict[str, list[str]] = {}
            for item in strata_items[env]:
                level = item.split("|")[1]
                level_items.setdefault(level, []).append(item)
            level_sizes = {lv: len(items) for lv, items in level_items.items()}
            level_alloc = allocate_budget(
                level_sizes, n, mode=mode, min_per_stratum=min_per_stratum
            )
            for lv, k in level_alloc.items():
                if k > 0:
                    selected.extend(rng.choice(level_items[lv], size=k, replace=False))
        return selected

    return sampler


def greedy_order_from_info_matrix(
    info: np.ndarray, items: np.ndarray, budget_max: int
) -> list[str]:
    """Greedy sum-of-information item order given an already-computed (n_items,
    n_grid) Fisher-info matrix — used by bayesian_irt_selection.py's
    posterior-averaged path (info averaged across many MCMC draws; the
    MAP/point-estimate path that used to share this function was removed —
    a single point estimate of (a, b) was too noisy to trust, see the MCMC
    vs. MAP discussion this replaced).

    At each step, adds whichever remaining item most increases TOTAL Fisher
    information summed across the grid (D-optimal-like). Incremental, so the
    budget-k prefix of the full order is the optimal (by this criterion)
    k-item set for every budget at once.
    """
    selected_idx: list[int] = []
    cum_info = np.zeros(info.shape[1])
    remaining = set(range(len(items)))
    for _ in range(min(budget_max, len(items))):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            score = (cum_info + info[idx]).sum()
            if score > best_score:
                best_score, best_idx = score, idx
        selected_idx.append(best_idx)
        cum_info += info[best_idx]
        remaining.discard(best_idx)
    return [items[i] for i in selected_idx]


SAMPLERS = {
    "random": random_sampler,
    "stratified_proportional": make_stratified_sampler("proportional"),
    "stratified_equal": make_stratified_sampler("equal"),
    "stratified_proportional_nested": make_stratified_sampler_nested("proportional"),
    "stratified_proportional_nested_min2": make_stratified_sampler_nested(
        "proportional", min_per_stratum=2
    ),
}


# ---------------- evaluation ----------------


def evaluate_subset(
    matrix: pd.DataFrame, col_env: pd.Series, selected: list[str]
) -> dict:
    """Compare mini-set (over `selected` items) scores to full-set scores.

    Reports global (item-pooled) MSE/ranking, per-environment MSE/ranking
    (averaged over environments that got >=1 sampled item), and per-
    environment item coverage counts (to detect starved environments).
    """
    true_global = matrix.mean(axis=1)
    mini_global = matrix[selected].mean(axis=1)

    global_mse = float(((mini_global - true_global) ** 2).mean())
    global_spearman = spearmanr(true_global, mini_global)[0]
    global_kendall = kendalltau(true_global, mini_global)[0]

    selected_set = set(selected)
    envs = sorted(col_env.unique())
    per_env_mse, per_env_spearman, env_coverage = {}, {}, {}
    for env in envs:
        env_items_all = col_env[col_env == env].index
        env_items_sel = [c for c in env_items_all if c in selected_set]
        env_coverage[env] = len(env_items_sel)
        if len(env_items_sel) == 0:
            per_env_mse[env] = np.nan
            per_env_spearman[env] = np.nan
            continue
        true_env = matrix[env_items_all].mean(axis=1)
        mini_env = matrix[env_items_sel].mean(axis=1)
        per_env_mse[env] = float(((mini_env - true_env) ** 2).mean())
        per_env_spearman[env] = spearmanr(true_env, mini_env)[0]

    return {
        "global_mse": global_mse,
        "global_spearman": global_spearman,
        "global_kendall": global_kendall,
        "per_env_mse": per_env_mse,
        "per_env_spearman": per_env_spearman,
        "env_coverage": env_coverage,
    }


def run_sweep(
    sampler_name: str,
    budgets: list[int],
    n_repeats: int,
    seed: int = 0,
    value: str = "success",
    exclude_environments: list[str] | None = None,
    data_path: Path | None = None,
) -> pd.DataFrame:
    matrix, col_env = load_matrix(
        value=value, exclude_environments=exclude_environments, data_path=data_path
    )
    all_items = list(matrix.columns)
    logger.info(f"{len(all_items)} candidate items, {matrix.shape[0]} subjects")

    n_items = len(all_items)
    dropped = [b for b in budgets if b > n_items]
    budgets = [b for b in budgets if b <= n_items]
    if dropped:
        logger.warning(f"Dropping budgets > {n_items} available items: {dropped}")

    sampler = SAMPLERS[sampler_name]
    rng = np.random.default_rng(seed)

    rows = []
    for budget in budgets:
        for repeat in range(n_repeats):
            selected = sampler(all_items, budget, rng, col_env)
            m = evaluate_subset(matrix, col_env, selected)
            row = {
                "sampler": sampler_name,
                "budget": budget,
                "repeat": repeat,
                "global_mse": m["global_mse"],
                "global_spearman": m["global_spearman"],
                "global_kendall": m["global_kendall"],
                "mean_per_env_mse": np.nanmean(list(m["per_env_mse"].values())),
                "mean_per_env_spearman": np.nanmean(
                    list(m["per_env_spearman"].values())
                ),
            }
            for env, cov in m["env_coverage"].items():
                row[f"coverage__{env}"] = cov
            rows.append(row)
    return pd.DataFrame(rows)
