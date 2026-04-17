"""Simple horizontal bar plot of average pass@5 per task category.

Averages across all models and agents. Subtasks only.

Usage:
    python plot_panel2_task_category_bar.py                          # comprehensive only
    python plot_panel2_task_category_bar.py --verbosity_strategy=average  # average all
"""

from collections import defaultdict
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import FONT_SIZES
from plot_utils import (
    classify_subtask,
    filter_by_verbosity,
    load_category_tags,
    load_reports_data,
)

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(verbosity_strategy: str = "comprehensive"):
    logger.info("Loading data...")
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    reports_df = load_reports_data()
    category_tags = load_category_tags()

    df_sub = reports_df[reports_df["category"] == "subtask"].copy()
    df_comp = filter_by_verbosity(
        df_sub, None if verbosity_strategy == "average" else verbosity_strategy
    )
    logger.info(f"Filtered to {len(df_comp)} rows (subtask + {verbosity_strategy})")

    category_order = ["retrieval", "execution", "reasoning", "validation"]
    scores_by_category = defaultdict(list)

    for _, row in df_comp.iterrows():
        environment = row["environment"]
        task_results = row["Task Results"]
        if not isinstance(task_results, dict):
            continue
        for subtask, result in task_results.items():
            if not isinstance(result, dict):
                continue
            pass_at_5 = result.get("Task Pass@5", None)
            if pass_at_5 is None:
                continue
            category = classify_subtask(subtask, environment, category_tags)
            if category is None:
                continue
            if category in ("code_execution", "experiment_execution"):
                category = "execution"
            scores_by_category[category].append(pass_at_5)

    # Compute means
    labels = [cat.replace("_", " ").title() for cat in category_order]
    values = [np.mean(scores_by_category.get(cat, [0])) for cat in category_order]

    for cat, val in zip(category_order, values, strict=False):
        n = len(scores_by_category.get(cat, []))
        logger.info(f"  {cat}: mean={val:.3f} (n={n})")

    # Horizontal bar plot
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    y_pos = np.arange(len(values))

    ax.barh(y_pos, values, color="#7150e0", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick_label"])
    ax.set_xlabel("Average Pass@5", fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])

    range_frame(ax, np.array([min(values), max(values)]), y_pos, pad=0.15)

    plt.tight_layout()

    suffix = f"_{verbosity_strategy}" if verbosity_strategy != "comprehensive" else ""
    for ext in ["pdf", "png"]:
        output_path = OUT_DIR / f"panel2_task_category_bar{suffix}.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
