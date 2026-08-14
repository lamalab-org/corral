"""Extract the "sparse" environments (afm, catalyst, md, ml) into a
corral_mini_source.csv-shaped CSV for the cleaning/sampling pipeline.

These 4 environments were only ever benchmarked with the 3 original models
(claude-4.5, gpt-4o, gpt-oss-120b) — none of the 3 newer models (claude-
opus-4.8, deepseek-v3.2, kimi-k2.5) were run on them, unlike resistor/retro/
spectra/wetlab. Per the gpt-oss-120b subject-quality finding (see
report_ceiling_check.html), gpt-oss-120b is dropped here too, leaving
claude-4.5 + gpt-4o (2 models x 2 scaffolds = 4 subjects) — too few for a
reliable IRT fit, hence these environments get a separate, CTT-only cleaning
pass (see compute_item_stats.py's IRT-join skip whenever exclude_models is
set).

Source: analysis/results/data/overall_trace.csv (the full multi-environment
trace; corral_mini_source.csv itself only ever covered the other 4
environments). Filtered to category="task" (matches corral_mini_source,
which excludes "subtask") and verbosity="workflow" (matches the well-covered
track, so the whole mini-benchmark is built on one consistent measurement
condition).

Usage:
    python extract_sparse_envs.py
"""

from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_PATH = REPO_ROOT / "analysis" / "results" / "data" / "overall_trace.csv"
OUT_PATH = Path(__file__).parent.parent / "data" / "sparse_envs_source.csv"

SPARSE_ENVIRONMENTS = ["afm", "catalyst", "md", "ml"]
KEEP_MODELS = ["claude-4.5", "gpt-4o"]  # gpt-oss-120b dropped, see module docstring
VERBOSITY = "workflow"
CATEGORY = "task"


def main() -> None:
    df = pd.read_csv(SOURCE_PATH)
    logger.info(f"Loaded {len(df)} rows from {SOURCE_PATH}")

    out = df[
        df["environment"].isin(SPARSE_ENVIRONMENTS)
        & df["model"].isin(KEEP_MODELS)
        & (df["verbosity"] == VERBOSITY)
        & (df["category"] == CATEGORY)
    ].copy()

    logger.info(f"{len(out)} rows after filtering")
    logger.info(
        "tasks per environment:\n"
        + out.groupby("environment")["task"].nunique().to_string()
    )
    logger.info(
        "trials per (model, scaffold): \n"
        + out.groupby(["model", "scaffold"]).size().to_string()
    )

    n_items = out.groupby(["environment", "level", "task"]).ngroups
    n_subjects = out.groupby(["model", "scaffold"]).ngroups
    expected_rows = n_items * n_subjects * 5
    if len(out) != expected_rows:
        logger.warning(
            f"{len(out)} rows != {n_items} items x {n_subjects} subjects x 5 trials "
            f"= {expected_rows} — pool may not be a full rectangular grid"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.success(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
