"""Apply the Track A cleaning decision and write the cleaned candidate pool.

Decision (see report_ceiling_check.html, report_two_track.html):
  - Track A (resistor/retro/spectra/wetlab): drop the 7 items with
    p_mean == 1.0 under the top-4-model pool (claude-opus-4.8, claude-4.5,
    kimi-k2.5, deepseek-v3.2) -- unanimous across all 8 (model x scaffold)
    subjects, every trial, and confirmed not to be high-discrimination items
    (corr(p_mean, a) = 0.06 on this pool). All 6 models' rows are kept for
    the *surviving* items -- only whole ceiling items are removed, not any
    subject's data.
  - Track B (afm/catalyst/md/ml): dropped entirely from corral-mini v2 (too
    few items -- 15 total -- and no coverage from the 3 newer models to
    justify inclusion in a benchmark meant to compare all 6). Not written
    here; corral_mini_source.csv never included them to begin with.

Usage:
    python build_cleaned_pool.py
"""

from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_PATH = DATA_DIR / "corral_mini_source.csv"
ITEM_STATS_TOP4_PATH = DATA_DIR / "item_stats_top4.csv"
OUT_POOL_PATH = DATA_DIR / "corral_mini_source_cleaned.csv"
OUT_REMOVED_PATH = DATA_DIR / "corral_mini_removed_items.csv"


def main() -> None:
    df = pd.read_csv(SOURCE_PATH)
    t4 = pd.read_csv(ITEM_STATS_TOP4_PATH)

    ceiling = t4[t4["p_mean"] == 1.0][["item", "environment", "level", "task"]].copy()
    ceiling["reason"] = "ceiling: p_mean=1.0 under top-4-model pool (8 subjects)"
    logger.info(
        f"Removing {len(ceiling)} ceiling items:\n{ceiling.to_string(index=False)}"
    )

    key = ["environment", "level", "task"]
    remove_keys = set(map(tuple, ceiling[key].to_numpy()))
    mask = df[key].apply(tuple, axis=1).isin(remove_keys)

    cleaned = df[~mask].copy()
    n_items_before = df.drop_duplicates(key).shape[0]
    n_items_after = cleaned.drop_duplicates(key).shape[0]
    logger.info(
        f"Pool: {n_items_before} -> {n_items_after} items "
        f"({df.shape[0]} -> {cleaned.shape[0]} rows, all 6 models' data kept)"
    )

    OUT_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_POOL_PATH, index=False)
    ceiling.to_csv(OUT_REMOVED_PATH, index=False)
    logger.success(f"Saved cleaned pool -> {OUT_POOL_PATH}")
    logger.success(f"Saved removal log -> {OUT_REMOVED_PATH}")


if __name__ == "__main__":
    main()
