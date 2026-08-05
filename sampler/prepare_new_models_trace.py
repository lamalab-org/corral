"""Build a trial-level trace CSV that adds the new models to the baseline.

The three new models (claude-opus-4.8, deepseek-v3.2, kimi-k2.5) were only
benchmarked on 4 of the 8 environments (resistor, retro, spectra, wetlab),
only on whole tasks (no subtasks), and only at "workflow" tool verbosity.
This script:

  1. Parses the new models' per-run summary JSONs under reports/ into the
     same trial-level schema as analysis/results/data/overall_trace.csv.
  2. Filters the baseline overall_trace.csv down to the matching slice
     (those 4 environments, category=task, verbosity=workflow) so the two
     sources are directly comparable.
  3. Concatenates them into one CSV — the source data for the corral-mini
     subsampling work.

Duplicate data note: some report dirs (e.g.
deepseek-v3.2/retrosynthesis/level_3/toolcalling/) contain both an aggregate
summary JSON and per-task "-fixcheck" re-exports of the *same* run. We reuse
plot_new_models_heatmap._pick_summary_json, which already excludes the
per-task files in favor of the aggregate, so trials aren't double-counted.

Usage:
    python prepare_new_models_trace.py
    python prepare_new_models_trace.py --output=data/corral_mini_source.csv
"""

import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
BASELINE_CSV = REPO_ROOT / "analysis" / "results" / "data" / "overall_trace.csv"
OUT_DIR = Path(__file__).parent / "data"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from plot_new_models_heatmap import (  # noqa: E402
    ENV_DIR_TO_KEY,
    MODEL_NAMES,
    _pick_summary_json,
)

SCAFFOLD_NORMALIZATION = {"react": "react", "toolcalling": "tool_calling"}

PASS_AT_KEYS = [f"Task Pass@{k}" for k in range(1, 6)]
PASS_HAT_KEYS = [f"Task Pass^{k}" for k in range(1, 6)]


def extract_new_model_trace() -> pd.DataFrame:
    """Walk reports/<model>/<env_dir>/level_N/<agent>/ and emit trial rows."""
    records = []
    for model_dir in sorted(REPORTS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name not in MODEL_NAMES:
            continue
        model = model_dir.name
        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir() or env_dir.name.startswith("_"):
                continue
            env_key = ENV_DIR_TO_KEY.get(env_dir.name)
            if env_key is None:
                continue
            for level_dir in sorted(env_dir.iterdir()):
                if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
                    continue
                for agent_dir in sorted(level_dir.iterdir()):
                    if (
                        not agent_dir.is_dir()
                        or agent_dir.name not in SCAFFOLD_NORMALIZATION
                    ):
                        continue
                    summary = _pick_summary_json(agent_dir)
                    if summary is None:
                        logger.warning(f"No summary JSON in {agent_dir}")
                        continue
                    scaffold = SCAFFOLD_NORMALIZATION[agent_dir.name]
                    records.extend(
                        _trial_rows_from_summary(
                            summary, model, env_key, level_dir.name, scaffold
                        )
                    )
    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} trial rows for new models")
    return df


def _trial_rows_from_summary(
    summary_path: Path, model: str, environment: str, level: str, scaffold: str
) -> list[dict]:
    import json

    with summary_path.open() as fh:
        data = json.load(fh)

    verbosity = data.get("metrics", {}).get("tool_verbosity", "unknown")
    rows = []
    for task_id, task_data in data.get("task_results", {}).items():
        pass_at_k = {f"pass@{k}": task_data.get(f"Task Pass@{k}") for k in range(1, 6)}
        pass_hat_k = {f"pass^{k}": task_data.get(f"Task Pass^{k}") for k in range(1, 6)}
        for trial in task_data.get("trials", []):
            # Not a plain list comprehension (PERF401): pass_at_k/pass_hat_k are
            # computed once per task_id above and reused across all its trials;
            # folding this into a comprehension would recompute them per-trial
            # or force an awkward nested-binding trick, both worse than the loop.
            rows.append(  # noqa: PERF401
                {
                    "model": model,
                    "environment": environment,
                    "scaffold": scaffold,
                    "level": level,
                    "category": "task",
                    "verbosity": verbosity,
                    "task": task_id,
                    "success": 1 if trial.get("success") else 0,
                    "score": trial.get("score", 0),
                    **pass_at_k,
                    **pass_hat_k,
                }
            )
    return rows


def load_baseline_slice(environments: list[str]) -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    sliced = df[
        df["environment"].isin(environments)
        & (df["category"] == "task")
        & (df["verbosity"] == "workflow")
    ].copy()
    logger.info(
        f"Baseline slice: {len(sliced)} trial rows "
        f"({sliced['model'].nunique()} models x {sliced['environment'].nunique()} envs)"
    )
    return sliced


def load_baseline_all_verbosity(environments: list[str]) -> pd.DataFrame:
    """Baseline models, same envs/category, but ALL 3 verbosities kept.

    Storage/sensitivity artifact only — NOT pooled into corral_mini_source.csv.
    New models only ever ran "workflow", so treating brief/comprehensive rows
    as extra IRT "subjects" would confound item discrimination with baseline-
    only verbosity sensitivity. Use this file for per-verbosity item-stability
    checks or as a shrinkage prior, not as raw pooled training data.
    """
    df = pd.read_csv(BASELINE_CSV)
    sliced = df[
        df["environment"].isin(environments) & (df["category"] == "task")
    ].copy()
    logger.info(
        f"Legacy all-verbosity slice: {len(sliced)} trial rows "
        f"(verbosities={sorted(sliced['verbosity'].unique())})"
    )
    return sliced


def main(output: str | None = None, legacy_output: str | None = None) -> None:
    """Build the combined (baseline + new models) trace CSV for corral-mini envs.

    Also writes a separate legacy-all-verbosity CSV (baseline models only, all
    3 verbosities) for sensitivity analysis / priors — see
    load_baseline_all_verbosity for why it's kept separate.
    """
    new_df = extract_new_model_trace()
    environments = sorted(new_df["environment"].unique())
    logger.info(f"New-model environments: {environments}")

    baseline_df = load_baseline_slice(environments)

    combined = pd.concat([baseline_df, new_df], ignore_index=True)
    columns = [
        "model",
        "environment",
        "scaffold",
        "level",
        "category",
        "verbosity",
        "task",
        "success",
        "score",
        "pass@1",
        "pass@2",
        "pass@3",
        "pass@4",
        "pass@5",
        "pass^1",
        "pass^2",
        "pass^3",
        "pass^4",
        "pass^5",
    ]
    combined = combined[columns]

    logger.info(
        "Trial counts by model:\n"
        + combined.groupby("model").size().sort_values(ascending=False).to_string()
    )
    coverage = combined.groupby(["model", "scaffold"])["environment"].nunique()
    logger.info(f"Environments covered per model x scaffold:\n{coverage.to_string()}")

    out = Path(output) if output else OUT_DIR / "corral_mini_source.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    logger.success(f"Saved {len(combined)} rows -> {out}")

    legacy_df = load_baseline_all_verbosity(environments)[columns]
    legacy_out = (
        Path(legacy_output) if legacy_output else OUT_DIR / "legacy_all_verbosity.csv"
    )
    legacy_df.to_csv(legacy_out, index=False)
    logger.success(f"Saved {len(legacy_df)} rows -> {legacy_out}")


if __name__ == "__main__":
    fire.Fire(main)
