"""Uncertainty-aware item selection using the full Bayesian 2PL posterior."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from subsampling_core import (
    allocate_budget,
    greedy_order_from_info_matrix,
)


def posterior_averaged_info(
    trace, n_draws_used: int = 200, seed: int = 0
) -> np.ndarray:
    """Fisher information averaged over posterior draws.

    Returns an (n_items, n_subj) matrix: for each item, its Fisher
    information at each of the n_subj ability "grid points" — but unlike the
    MAP version, both the item parameters AND the grid points themselves
    vary draw-to-draw, so this integrates over uncertainty in both at once.
    """
    theta_samples = (
        trace.posterior["theta"].stack(sample=("chain", "draw")).to_numpy()
    )  # (n_subj, n_total)
    a_samples = (
        trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()
    )  # (n_item, n_total)
    b_samples = (
        trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()
    )  # (n_item, n_total)

    n_total = theta_samples.shape[1]
    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(n_total, size=min(n_draws_used, n_total), replace=False)

    n_item, n_subj = a_samples.shape[0], theta_samples.shape[0]
    cum_info = np.zeros((n_item, n_subj))
    for d in draw_idx:
        theta_d, a_d, b_d = theta_samples[:, d], a_samples[:, d], b_samples[:, d]
        logits = a_d[:, None] * (theta_d[None, :] - b_d[:, None])  # (n_item, n_subj)
        p = 1 / (1 + np.exp(-logits))
        cum_info += (a_d[:, None] ** 2) * p * (1 - p)
    return cum_info / len(draw_idx)


def build_bayes_item_order(
    trace, items: list[str], budget_max: int, n_draws_used: int = 200, seed: int = 0
) -> list[str]:
    """Global (unstratified) posterior-averaged item order."""
    info = posterior_averaged_info(trace, n_draws_used=n_draws_used, seed=seed)
    return greedy_order_from_info_matrix(info, np.array(items), budget_max)


def build_bayes_item_order_per_stratum(
    trace, items: list[str], item_env: pd.Series, n_draws_used: int = 200, seed: int = 0
) -> dict[str, list[str]]:
    """Per-environment posterior-averaged item order, for the stratified+Bayesian-IRT hybrid."""
    info = posterior_averaged_info(trace, n_draws_used=n_draws_used, seed=seed)
    items_arr = np.array(items)
    orders = {}
    for env in item_env.unique():
        env_mask = (item_env.reindex(items).to_numpy()) == env
        env_items = items_arr[env_mask]
        env_info = info[env_mask]
        orders[env] = greedy_order_from_info_matrix(env_info, env_items, env_mask.sum())
    return orders


def make_stratified_bayes_sampler(
    per_env_order: dict[str, list[str]], allocation_mode: str = "proportional"
):
    strata_sizes = {env: len(items) for env, items in per_env_order.items()}

    def sampler(all_items, budget, rng, col_env):
        alloc = allocate_budget(strata_sizes, budget, mode=allocation_mode)
        selected = []
        for env, n in alloc.items():
            if n > 0:
                selected.extend(per_env_order[env][:n])
        return selected

    return sampler


def build_bayes_item_order_per_env_level(
    trace, items: list[str], item_env: pd.Series, n_draws_used: int = 200, seed: int = 0
) -> dict[tuple[str, str], list[str]]:
    """Per-(environment, level) posterior-averaged item order -- the nested
    analogue of build_bayes_item_order_per_stratum, so the Bayes-IRT-greedy
    sampler can be compared head-to-head against
    subsampling_core.make_stratified_sampler_nested under the exact same
    (environment, level) allocation, isolating "how items are picked within
    a stratum" as the only thing that differs between the two methods.
    """
    info = posterior_averaged_info(trace, n_draws_used=n_draws_used, seed=seed)
    items_arr = np.array(items)
    env_arr = item_env.reindex(items).to_numpy()
    level_arr = np.array([it.split("|")[1] for it in items])

    orders: dict[tuple[str, str], list[str]] = {}
    for env in np.unique(env_arr):
        for level in np.unique(level_arr[env_arr == env]):
            mask = (env_arr == env) & (level_arr == level)
            orders[(env, level)] = greedy_order_from_info_matrix(
                info[mask], items_arr[mask], int(mask.sum())
            )
    return orders


def make_stratified_bayes_sampler_nested(
    orders: dict[tuple[str, str], list[str]],
    allocation_mode: str = "proportional",
    min_per_stratum: int = 1,
):
    """Nested (environment, level) allocation identical in structure to
    subsampling_core.make_stratified_sampler_nested -- same two-stage
    allocate_budget calls, same min_per_stratum floor -- but within each
    stratum, takes the top-k items by Fisher-information greedy order
    (from build_bayes_item_order_per_env_level) instead of a random draw.
    Deterministic: no rng use, despite the signature matching SAMPLERS'
    (all_items, budget, rng, col_env) shape for drop-in compatibility.
    """
    env_level_sizes = {k: len(v) for k, v in orders.items()}
    envs = sorted({env for env, _level in orders})
    env_sizes = {
        env: sum(n for (e, _lv), n in env_level_sizes.items() if e == env)
        for env in envs
    }

    def sampler(all_items, budget, rng, col_env):
        env_alloc = allocate_budget(
            env_sizes, budget, mode=allocation_mode, min_per_stratum=min_per_stratum
        )
        selected = []
        for env, n in env_alloc.items():
            if n == 0:
                continue
            level_sizes = {
                lv: sz for (e, lv), sz in env_level_sizes.items() if e == env
            }
            level_alloc = allocate_budget(
                level_sizes, n, mode=allocation_mode, min_per_stratum=min_per_stratum
            )
            for lv, k in level_alloc.items():
                if k > 0:
                    selected.extend(orders[(env, lv)][:k])
        return selected

    return sampler
