"""Simple horizontal bar plot of average pass@5 per task category.

Averages across all models and agents. Only subtasks with comprehensive verbosity.

Usage:
    python 2c_task_category_bar.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import FONT_SIZES  # noqa: E402
from plot_utils import load_reports_data  # noqa: E402

lama_aesthetics.get_style("main")


def load_category_tags():
    tags_path = REPO_ROOT / "analysis" / "subtask_category_tags.json"
    with tags_path.open() as f:
        return json.load(f)


def get_env_key_mapping():
    return {
        "spectra": "sptectra",
        "retro": "retrosynthesis",
        "afm": "afm",
        "catalyst": "catalyst",
        "md": "md",
        "ml": "ml",
        "resistor": "resistor",
        "wetlab": "wetlab",
    }


def classify_subtask(subtask, environment, category_tags):
    env_mapping = get_env_key_mapping()
    tag_env_key = env_mapping.get(environment, environment)
    env_tags = category_tags.get(tag_env_key, {})
    if not env_tags:
        return None

    if environment == "afm":
        if "subtask_level_" in subtask:
            base_name = subtask.split("_level_")[0]
            if base_name + "_level_1" in env_tags:
                return env_tags[base_name + "_level_1"]
        return env_tags.get(subtask)

    if environment == "catalyst":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[1] in env_tags:
            return env_tags[parts[1]]
        return None

    if environment == "md":
        for task_type in ["melting", "quenching", "surface_energy", "surface"]:
            if task_type in subtask:
                task_tags = env_tags.get(task_type, {})
                if "subtask_" in subtask:
                    subtask_name = subtask.split("subtask_")[-1]
                    if subtask_name == "diffusion_coefficient":
                        subtask_name = "diffusivity"
                    elif subtask_name == "tg_calculation":
                        subtask_name = "tg_detection"
                    elif subtask_name == "equilibration":
                        subtask_name = "structure_retrieval"
                    if subtask_name in task_tags:
                        return task_tags[subtask_name]
        return None

    if environment == "ml":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            if (
                task_name.startswith("batch_retrieve_")
                and "batch_retrieve_*" in env_tags
            ):
                return env_tags["batch_retrieve_*"]
            if task_name in env_tags:
                return env_tags[task_name]
        return None

    if environment == "resistor":
        parts = subtask.split("_", 2)
        if len(parts) >= 3 and parts[0] == "task" and parts[1].isdigit():
            pattern = "_".join(parts[2:])
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "retro":
        if "-" in subtask:
            pattern = subtask.split("-", 1)[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "spectra":
        if "_subtask_" in subtask:
            subtask_num = "subtask_" + subtask.split("_subtask_")[-1]
            if subtask_num in env_tags:
                return env_tags[subtask_num]
        return None

    if environment == "wetlab":
        m = re.match(r"qualysis_lvl(\d+)_\d+_(sub\d+)", subtask)
        if m:
            level_key = f"level_{m.group(1)}"
            sub_key = f"qualysis_lvl{m.group(1)}_*_{m.group(2)}"
            level_tags = env_tags.get(level_key, {})
            if sub_key in level_tags:
                return level_tags[sub_key]
        return None

    return None


def main():
    logger.info("Loading data...")
    reports_df = load_reports_data()
    category_tags = load_category_tags()

    df_sub = reports_df[reports_df["category"] == "subtask"].copy()
    df_comp = df_sub[df_sub["Tool Verbosity"] == "comprehensive"].copy()
    logger.info(f"Filtered to {len(df_comp)} rows (subtask + comprehensive)")

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

    output_dir = Path(__file__).parent
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"2c_task_category_bar.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
