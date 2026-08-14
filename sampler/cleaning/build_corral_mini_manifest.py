"""Build a corral-mini task manifest: sample from the (cleaned) candidate pool
and write it out in the same shape as the original corral_mini_manifest.

v1 (corral_mini_manifest_budget20.json) was produced ad hoc -- no script in
the repo built it, just subsampling_core.SAMPLERS["stratified_proportional_nested"]
called directly with budget=20, seed=0, resistor excluded from the budget and
appended whole afterward (noted in its own "note" field: "resistor is
excluded from the budget (only 6 items total) and kept whole, not sampled").
This script reproduces that exact method by default, generalized to take a
data_path -- so v2 is "v1's method, cleaned pool" with nothing else changed
at once, matching the earlier cleaned-vs-uncleaned sampling validation.

Usage:
    python build_corral_mini_manifest.py
    python build_corral_mini_manifest.py --data_path=../data/corral_mini_source_cleaned.csv --tag=_v2
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from subsampling_core import SAMPLERS, load_matrix

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "corral_mini_source_cleaned.csv"


def build_manifest(
    data_path: Path,
    sampler_name: str,
    budget: int,
    seed: int,
    exclude_from_budget: list[str],
) -> tuple[dict, pd.DataFrame]:
    matrix, col_env = load_matrix(
        value="success", exclude_environments=exclude_from_budget, data_path=data_path
    )
    all_items = list(matrix.columns)
    logger.info(
        f"{len(all_items)} candidate items for sampling (excl. {exclude_from_budget})"
    )

    sampler = SAMPLERS[sampler_name]
    rng = np.random.default_rng(seed)
    selected = sampler(all_items, budget, rng, col_env)
    logger.info(f"Sampled {len(selected)} items at budget={budget}")

    rows = []
    for item in selected:
        env, level, task = item.split("|")
        rows.append(
            {"environment": env, "level": level, "task": task, "status": "sampled"}
        )

    # kept-whole environments: every item in the source pool for that env
    df_full = pd.read_csv(data_path)
    for env in exclude_from_budget:
        whole = df_full[df_full["environment"] == env].drop_duplicates(
            ["level", "task"]
        )
        for _, r in whole.iterrows():
            rows.append(
                {
                    "environment": env,
                    "level": r["level"],
                    "task": r["task"],
                    "status": "kept_whole",
                }
            )
        logger.info(f"Kept {len(whole)} items whole for excluded environment '{env}'")

    selected_df = (
        pd.DataFrame(rows)
        .sort_values(["environment", "level", "task"])
        .reset_index(drop=True)
    )

    tasks_by_env: dict[str, dict[str, list[str]]] = {}
    for _, r in selected_df.iterrows():
        tasks_by_env.setdefault(r["environment"], {}).setdefault(r["level"], []).append(
            r["task"]
        )

    manifest = {
        "name": "corral-mini",
        "budget": budget,
        "sampler": sampler_name,
        "seed": seed,
        "source_script": "sampler/cleaning/build_corral_mini_manifest.py",
        "note": (
            f"{'/'.join(exclude_from_budget)} excluded from the budget and kept whole, not sampled. "
            f"Candidate pool: {data_path.name} (cleaned: 7 ceiling items removed under the "
            "top-4-model ceiling check, see report_ceiling_check.html / report_two_track.html)."
        ),
        "total_tasks": len(selected_df),
        "tasks_by_environment": tasks_by_env,
    }
    return manifest, selected_df


def main(
    data_path: str = str(DEFAULT_DATA_PATH),
    sampler: str = "stratified_proportional_nested",
    budget: int = 20,
    seed: int = 0,
    exclude_from_budget: str = "resistor",
    tag: str = "",
    output_dir: str | None = None,
) -> None:
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {list(SAMPLERS)}")
    excluded = [e.strip() for e in exclude_from_budget.split(",") if e.strip()]

    manifest, selected_df = build_manifest(
        Path(data_path), sampler, budget, seed, excluded
    )

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"corral_mini_manifest_budget{budget}{tag}.json"
    tasks_path = out_dir / f"corral_mini_selected_tasks_budget{budget}{tag}.csv"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    selected_df.to_csv(tasks_path, index=False)

    logger.success(f"Saved -> {manifest_path}")
    logger.success(f"Saved -> {tasks_path}")
    logger.info(f"Total tasks: {manifest['total_tasks']}")
    for env, levels in manifest["tasks_by_environment"].items():
        counts = {lv: len(tasks) for lv, tasks in levels.items()}
        logger.info(f"  {env}: {counts} (total {sum(counts.values())})")


if __name__ == "__main__":
    fire.Fire(main)
