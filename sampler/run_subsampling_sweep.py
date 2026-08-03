"""Run a (budget x repeats) subsampling sweep for a given sampler.

Usage:
    python run_subsampling_sweep.py
    python run_subsampling_sweep.py --sampler=random --n_repeats=200
"""

from pathlib import Path

import fire
from loguru import logger

from subsampling_core import (
    SAMPLERS,
    build_irt_item_order,
    build_irt_item_order_per_stratum,
    make_irt_maxinfo_sampler,
    make_stratified_irt_sampler,
    run_sweep,
)

OUT_DIR = Path(__file__).parent / "data"

# Deterministic samplers (no randomness given a fixed budget) — repeats would
# just be identical rows, so force n_repeats=1 rather than waste compute /
# imply a variance band that doesn't exist.
DETERMINISTIC_SAMPLERS = {"irt_maxinfo", "stratified_irt"}


def main(
    sampler: str = "random",
    n_repeats: int = 200,
    budget_min: int = 5,
    budget_max: int = 100,
    budget_step: int = 5,
    seed: int = 0,
    exclude_environments: str = "resistor",
    output: str | None = None,
) -> None:
    """
    Args:
        exclude_environments: comma-separated environment names to drop from
            the candidate item pool before sampling (e.g. "resistor" — too
            few items/levels to be worth subsampling; keep it whole instead).
            Pass "" for no exclusions.
    """
    if sampler == "irt_maxinfo":
        item_order = build_irt_item_order(
            item_params_path=OUT_DIR / "irt_2pl_items.csv",
            subject_params_path=OUT_DIR / "irt_2pl_subjects.csv",
            budget_max=budget_max,
        )
        SAMPLERS["irt_maxinfo"] = make_irt_maxinfo_sampler(item_order)
    if sampler == "stratified_irt":
        per_env_order = build_irt_item_order_per_stratum(
            item_params_path=OUT_DIR / "irt_2pl_items.csv",
            subject_params_path=OUT_DIR / "irt_2pl_subjects.csv",
        )
        SAMPLERS["stratified_irt"] = make_stratified_irt_sampler(per_env_order, allocation_mode="proportional")
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}")
    if sampler in DETERMINISTIC_SAMPLERS and n_repeats != 1:
        logger.info(f"{sampler} is deterministic — forcing n_repeats=1 (was {n_repeats})")
        n_repeats = 1
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    logger.info(f"sampler={sampler} budgets={budgets} n_repeats={n_repeats} excluded={excluded}")

    results = run_sweep(sampler, budgets, n_repeats, seed=seed, exclude_environments=excluded)
    logger.info(f"{len(results)} (budget, repeat) rows")

    tag = f"_excl-{'-'.join(excluded)}" if excluded else ""
    out = Path(output) if output else OUT_DIR / f"subsampling_sweep_{sampler}{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    logger.success(f"Saved -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
