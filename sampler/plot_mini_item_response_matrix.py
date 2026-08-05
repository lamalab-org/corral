"""Item response matrix for the corral-mini source data (single verbosity).

Same construction as plot_sorted_response_matrix.py, but over
sampler/data/corral_mini_source.csv: 6 models x 2 scaffolds (12 subjects) x
100 items, where an item is (environment, level, task) — verbosity and
category are dropped from the item key here because both are constant
("workflow" / "task") in this dataset, unlike the full overall_trace.csv.

Also reports a Guttman scalogram diagnostic (coefficient of reproducibility),
which is the practical way to answer "can we get away with a simple Rasch
(1PL) model": Rasch assumes a single unidimensional ability/difficulty scale
with non-crossing item characteristic curves, which is the probabilistic
version of a Guttman scale. If the binarized response matrix already sorts
into a near-clean staircase (few errors relative to the ideal step pattern
for each subject's raw score), a 1PL fit is well justified. Heavy, structured
deviations (many errors, or errors concentrated in specific items/subjects)
point to multidimensionality or varying item discrimination, i.e. you'd want
at least a 2PL model instead.

Usage:
    python plot_mini_item_response_matrix.py
    python plot_mini_item_response_matrix.py --value=score
"""

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from plot_sorted_response_matrix import build_matrix, plot_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).parent / "data" / "corral_mini_source.csv"
OUT_DIR = Path(__file__).parent / "figures"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from plot_config import AGENT_NAMES, MODEL_COLOURS as BASE_MODEL_COLOURS, MODEL_NAMES as BASE_MODEL_NAMES  # noqa: E402

# corral_mini_source.csv has no subtask/verbosity variation, so item = (env, level, task)
ITEM_KEY = ["environment", "level", "task"]

# plot_config.py only knows the 3 legacy models; extend with the 3 new ones
# (colours match analysis/plot_new_models_heatmap.py for visual consistency)
MODEL_NAMES = {
    **BASE_MODEL_NAMES,
    "claude-opus-4.8": "Claude-Opus-4.8",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "kimi-k2.5": "Kimi-K2.5",
}
MODEL_COLOURS = {
    **BASE_MODEL_COLOURS,
    # claude-opus-4.8 deliberately NOT #7c3aed (plot_config's ENVIRONMENT_COLOURS["spectra"]) —
    # that hue is already used by the environment strip in this same figure.
    "claude-opus-4.8": "#ea580c",
    "deepseek-v3.2": "#0891b2",
    "kimi-k2.5": "#16a34a",
}


def guttman_reproducibility(matrix: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Coefficient of reproducibility for the row/col-sorted binary matrix.

    For each subject (row), reconstructs the "ideal" Guttman pattern given
    their raw score (pass the N easiest items, fail the rest, since columns
    are already sorted easiest -> hardest) and counts cells that disagree
    with it. CR = 1 - total_errors / (n_subjects * n_items).

    Conventionally CR >= 0.90 is considered a scalable (near-unidimensional,
    Rasch-compatible) item set.
    """
    binary = (matrix.to_numpy(dtype=float) >= threshold).astype(int)
    n_rows, n_cols = binary.shape
    per_row_errors = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        score = binary[i].sum()
        ideal = np.zeros(n_cols, dtype=int)
        ideal[:score] = 1
        per_row_errors[i] = np.abs(binary[i] - ideal).sum()

    total_errors = int(per_row_errors.sum())
    cr = 1 - total_errors / (n_rows * n_cols)
    return {
        "cr": cr,
        "total_errors": total_errors,
        "n_cells": n_rows * n_cols,
        "per_row_errors": dict(zip(matrix.index, per_row_errors.tolist())),
    }


def item_total_correlations(matrix: pd.DataFrame) -> pd.Series:
    """Item-rest point-biserial-style correlation (item vs. total-minus-item score).

    Rasch (1PL) assumes every item discriminates equally, i.e. roughly
    homogeneous item-total correlations. A wide spread (some items strongly
    tracking ability, others weakly or not at all) means a single
    discrimination parameter won't fit well and a 2PL model is more
    appropriate. With only 12 subjects these per-item estimates are noisy
    individually — read the spread, not any single item's value.
    """
    vals = matrix.to_numpy(dtype=float)
    total = vals.sum(axis=1)
    corrs = {}
    for j, item in enumerate(matrix.columns):
        rest = total - vals[:, j]
        if np.std(vals[:, j]) == 0 or np.std(rest) == 0:
            corrs[item] = np.nan
        else:
            corrs[item] = np.corrcoef(vals[:, j], rest)[0, 1]
    return pd.Series(corrs)


def main(
    value: str = "success", exclude_environments: str = "resistor", output: str | None = None
) -> None:
    """Plot the corral-mini item response matrix and report Guttman/Rasch diagnostics.

    Args:
        exclude_environments: comma-separated environments to drop before
            building the matrix (default "resistor" — too few items/levels
            to be worth subsampling; keep it whole instead). Pass "" for none.
    """
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    logger.info(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if excluded:
        df = df[~df["environment"].isin(excluded)]
        logger.info(f"Excluded environments: {excluded}")
    logger.info(f"{len(df)} trial rows, verbosity={sorted(df['verbosity'].unique())}")

    matrix, col_env = build_matrix(df, value=value, item_key=ITEM_KEY)
    logger.info(f"Matrix shape: {matrix.shape[0]} subjects x {matrix.shape[1]} items")
    logger.info(
        "Per-subject mean:\n"
        + matrix.mean(axis=1, skipna=True)
        .rename(lambda r: f"{MODEL_NAMES.get(r.split('__')[0], r.split('__')[0])} · {AGENT_NAMES.get(r.split('__')[1], r.split('__')[1])}")
        .round(3)
        .to_string()
    )

    diag = guttman_reproducibility(matrix)
    logger.info(f"Guttman coefficient of reproducibility (CR) = {diag['cr']:.3f}")
    logger.info(f"Total errors: {diag['total_errors']} / {diag['n_cells']} cells")
    logger.warning(
        "Note: cells are success rates averaged over only 5 trials, thresholded at 0.5 for this "
        "deterministic-scale check. Any item near ~50% true difficulty will look like binomial "
        "noise here even under a perfect Rasch model — so CR is a pessimistic lower bound, not a "
        "clean multidimensionality test on its own."
    )
    worst = sorted(diag["per_row_errors"].items(), key=lambda kv: -kv[1])[:3]
    logger.info(f"Subjects with the most Guttman errors: {worst}")

    item_corr = item_total_correlations(matrix)
    logger.info(
        f"Item-total correlation: mean={item_corr.mean():.3f}, std={item_corr.std():.3f}, "
        f"range=[{item_corr.min():.3f}, {item_corr.max():.3f}], "
        f"{(item_corr < 0).sum()} items negatively correlated with total score"
    )
    if diag["cr"] >= 0.90 and item_corr.std() < 0.15:
        verdict = "Both diagnostics support a simple Rasch (1PL) model as a reasonable starting point."
    elif item_corr.std() >= 0.15 or (item_corr < 0).sum() > 0:
        verdict = (
            "Item-total correlations are heterogeneous (or some are negative) -> items differ "
            "meaningfully in discrimination. A 2PL model is better justified than Rasch, which is "
            "also the more useful lens for corral-mini since you explicitly want to select for "
            "discriminative items, not just calibrate difficulty."
        )
    else:
        verdict = "Mixed signal — worth fitting both 1PL and 2PL and comparing held-out fit before deciding."
    logger.info(verdict)

    tag = f"_excl-{'-'.join(excluded)}" if excluded else ""
    out = Path(output) if output else OUT_DIR / f"mini_item_response_matrix_{value}{tag}.pdf"
    plot_matrix(
        matrix,
        col_env,
        value_label=f"mean {value} rate",
        output_path=out,
        model_names=MODEL_NAMES,
        model_colours=MODEL_COLOURS,
        title=f"corral-mini item response matrix ({matrix.shape[0]} subjects × {matrix.shape[1]} items, workflow verbosity only)",
        subtitle=f"Guttman coefficient of reproducibility = {diag['cr']:.3f} ({diag['total_errors']}/{diag['n_cells']} errors)",
    )


if __name__ == "__main__":
    fire.Fire(main)
