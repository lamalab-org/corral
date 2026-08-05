"""Run a (budget x repeats) subsampling sweep for a given sampler.

Usage:
    python run_subsampling_sweep.py
    python run_subsampling_sweep.py --sampler=random --n_repeats=200
"""

from pathlib import Path

import fire
from loguru import logger
from subsampling_core import SAMPLERS, run_sweep

OUT_DIR = Path(__file__).parent / "data"


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
        sampler: one of subsampling_core.SAMPLERS (random, stratified_equal,
            stratified_proportional, stratified_proportional_nested). The
            IRT-based samplers live in run_final_ranking_comparison.py
            instead — they need the MCMC trace and bootstrap-resample for
            comparable error bars, which doesn't fit this generic runner.
        exclude_environments: comma-separated environment names to drop from
            the candidate item pool before sampling (e.g. "resistor" — too
            few items/levels to be worth subsampling; keep it whole instead).
            Pass "" for no exclusions.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}")
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]

    budgets = sorted(set(range(budget_min, budget_max, budget_step)) | {budget_max})
    logger.info(
        f"sampler={sampler} budgets={budgets} n_repeats={n_repeats} excluded={excluded}"
    )

    results = run_sweep(
        sampler, budgets, n_repeats, seed=seed, exclude_environments=excluded
    )
    logger.info(f"{len(results)} (budget, repeat) rows")

    tag = f"_excl-{'-'.join(excluded)}" if excluded else ""
    out = Path(output) if output else OUT_DIR / f"subsampling_sweep_{sampler}{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    logger.success(f"Saved -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
