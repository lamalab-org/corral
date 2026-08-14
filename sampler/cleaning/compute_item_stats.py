"""Per-item statistics for the corral-mini cleaning step.

Loads the same (subject x item) success matrix subsampling_core.py samples
from (corral_mini_source.csv: 6 models x 2 scaffolds, workflow verbosity,
100 items across resistor/retro/spectra/wetlab), and computes, for every
item, classical-test-theory statistics needed to decide which items are
"ceiling" (already solved by ~everyone) and therefore uninformative for
telling models apart in a mini-benchmark:

    p_mean       mean success rate across all 12 subjects (each subject's
                 cell is already the mean over its 5 trials)
    p_min/p_max  weakest / strongest subject's rate on this item
    n_perfect    # subjects with rate == 1.0 (solved every trial)
    n_zero       # subjects with rate == 0.0 (failed every trial)
    std          std of per-subject rates (0 => zero variance => item can't
                 discriminate between subjects at all, the CTT analogue of
                 "IRT can't fit a on this item")

Also left-joins the existing 2PL IRT (a, b) fit (irt_2pl_items.csv) where
available (94/100 items; resistor's 6 items were excluded from that fit) as
a cross-check, not as the filter basis itself (that's a raw-success-rate
threshold per the plan we agreed on).

Usage:
    python compute_item_stats.py
"""

import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from subsampling_core import load_matrix

DATA_DIR = Path(__file__).parent.parent / "data"
IRT_ITEMS_PATH = DATA_DIR / "irt_2pl_items.csv"


def compute_item_stats(
    exclude_models: list[str] | None = None,
    data_path: Path | None = None,
    irt_items_path: Path | None = None,
) -> pd.DataFrame:
    matrix, col_env = load_matrix(
        value="success",
        exclude_environments=None,
        exclude_models=exclude_models,
        data_path=data_path,
    )
    # matrix: rows=subjects (model__scaffold), cols=items "env|level|task"
    logger.info(f"Subject pool: {list(matrix.index)}")

    stats = pd.DataFrame(
        {
            "p_mean": matrix.mean(axis=0),
            "p_min": matrix.min(axis=0),
            "p_max": matrix.max(axis=0),
            "std": matrix.std(axis=0),
            "n_perfect": (matrix == 1.0).sum(axis=0),
            "n_zero": (matrix == 0.0).sum(axis=0),
        }
    )
    stats["n_subjects"] = matrix.shape[0]
    stats["environment"] = col_env.reindex(stats.index)

    parts = stats.index.to_series().str.split("|", expand=True)
    stats["level"] = parts[1].to_numpy()
    stats["task"] = parts[2].to_numpy()
    stats = stats.reset_index(names="item")

    # Default IRT join target: only the stock fit, and only when nothing about
    # the subject/item pool has been customized (a fit keyed just by "item"
    # could otherwise silently describe a different subject population, e.g.
    # after exclude_models, or a different item universe entirely, e.g. the
    # sparse environments via data_path).
    if irt_items_path is None and data_path is None and not exclude_models:
        irt_items_path = IRT_ITEMS_PATH
    if irt_items_path is None:
        logger.warning(
            "No irt_items_path resolved for this pool (custom data_path/"
            "exclude_models) -> skipping IRT a/b join; pass irt_items_path "
            "explicitly once a matching fit exists"
        )
    elif irt_items_path.exists():
        irt = pd.read_csv(irt_items_path)[["item", "a", "b"]]
        stats = stats.merge(irt, on="item", how="left")
        n_missing_irt = stats["a"].isna().sum()
        logger.info(
            f"Joined IRT a/b for {stats['a'].notna().sum()}/{len(stats)} items "
            f"({n_missing_irt} without an IRT fit, expected for excluded environments) "
            f"from {irt_items_path}"
        )
    else:
        logger.warning(f"No IRT items file at {irt_items_path}, skipping a/b join")

    stats = stats.sort_values(["environment", "level", "p_mean"]).reset_index(drop=True)
    cols = [
        "item",
        "environment",
        "level",
        "task",
        "p_mean",
        "p_min",
        "p_max",
        "std",
        "n_perfect",
        "n_zero",
        "n_subjects",
        "a",
        "b",
    ]
    cols = [c for c in cols if c in stats.columns]
    return stats[cols]


def main(
    exclude_models: str = "",
    data_path: str | None = None,
    irt_items_path: str | None = None,
    output: str | None = None,
    tag: str | None = None,
) -> None:
    """
    Args:
        exclude_models: comma-separated model names to drop from the subject
            pool before computing item stats (e.g. "gpt-oss-120b" — sensitivity
            check for a subject suspected of failing for reasons other than
            task difficulty, e.g. scaffold/tool-format issues rather than
            genuine reasoning failure; or "gpt-4o,gpt-oss-120b" for the
            top-4-model well-covered-track pool).
        data_path: override the source CSV (default corral_mini_source.csv).
            Use e.g. data/sparse_envs_source.csv for the sparse-environment
            track (afm/catalyst/md/ml).
        irt_items_path: explicit IRT items CSV to join a/b from. Only auto-
            resolved to the stock fit when data_path/exclude_models are both
            unset; pass explicitly once a matching refit exists.
        output: override output CSV path.
        tag: output filename suffix override, used literally as
            f"item_stats{tag}.csv" (include your own leading "_"; default
            derived from exclude_models, e.g. "_excl-gpt-oss-120b").
    """
    excluded = [m.strip() for m in exclude_models.split(",") if m.strip()]
    stats = compute_item_stats(
        exclude_models=excluded or None,
        data_path=Path(data_path) if data_path else None,
        irt_items_path=Path(irt_items_path) if irt_items_path else None,
    )
    logger.info(f"{len(stats)} items, subject pool excludes: {excluded or 'none'}")
    logger.info(
        "Ceiling counts (n_perfect == n_subjects, i.e. every subject solved every "
        f"trial): {(stats['n_perfect'] == stats['n_subjects']).sum()}"
    )
    logger.info(
        f"p_mean == 1.0 (all subjects, all trials): {(stats['p_mean'] == 1.0).sum()}"
    )
    tag = (
        tag if tag is not None else (f"_excl-{'-'.join(excluded)}" if excluded else "")
    )
    out = Path(output) if output else DATA_DIR / f"item_stats{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out, index=False)
    logger.success(f"Saved -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
