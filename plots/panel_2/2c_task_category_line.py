"""Task category line plot: two lines (ReAct solid, Tool calling dashed).

Models are averaged so we get one value per agent per category.
Grey vertical lines mark each category. Purple lines. 1-col width and height.

Usage:
    python 2c_task_category_line.py
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
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import AGENT_NAMES, FONT_SIZES  # noqa: E402
from plot_utils import load_reports_data  # noqa: E402

lama_aesthetics.get_style("main")

LINE_COLOR = "#7150e0"


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
    logger.info(f"Filtered to {len(df_sub)} rows (subtask, all verbosities)")

    category_order = ["retrieval", "execution", "reasoning", "validation"]

    # Collect scores by (agent_type, category) — average across models
    scores_by_agent_cat = defaultdict(lambda: defaultdict(list))

    for _, row in df_sub.iterrows():
        agent_type = row["agent_type"]
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
            scores_by_agent_cat[agent_type][category].append(pass_at_5)

    # Compute means per agent per category
    agent_avgs = {}
    for agent_type, cat_scores in scores_by_agent_cat.items():
        agent_avgs[agent_type] = {
            cat: np.mean(scores) for cat, scores in cat_scores.items()
        }

    for agent_type, avgs in agent_avgs.items():
        for cat in category_order:
            logger.info(
                f"  {AGENT_NAMES.get(agent_type, agent_type)} - {cat}: "
                f"{avgs.get(cat, 0):.3f}"
            )

    # Plot
    # GOLDEN_RATIO = 1.618
    # fig_w = TWO_COL_WIDTH / 2
    # fig_h = fig_w / GOLDEN_RATIO
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    x_values = np.arange(len(category_order))

    # Grey vertical lines at each category
    # for x in x_values:
    #     ax.axvline(
    #         x=x, color="gray", linewidth=0.5, alpha=0.2, linestyle="--", zorder=1
    #     )

    agent_linestyle = {"react": "-", "tool_calling": "--"}

    for agent_type, avgs in sorted(agent_avgs.items()):
        y_values = [avgs.get(cat, 0) for cat in category_order]
        linestyle = agent_linestyle.get(agent_type, "-")
        ax.plot(
            x_values,
            y_values,
            color=LINE_COLOR,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
        )

    ax.set_ylabel(
        "Average Pass@5", fontsize=FONT_SIZES["axis_label"], fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    category_labels = [cat.replace("_", " ").title() for cat in category_order]
    ax.set_xticks(x_values)
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZES["tick_label"])

    all_vals = [
        v for avgs in agent_avgs.values() for v in avgs.values() if not np.isnan(v)
    ]
    range_frame(
        ax,
        np.array([0, len(category_order) - 1]),
        np.array([min(all_vals), max(all_vals)]),
        pad=0.05,
    )

    # Legend
    handles = [
        Line2D([0], [0], color=LINE_COLOR, linestyle="-", lw=2),
        Line2D([0], [0], color=LINE_COLOR, linestyle="--", lw=2),
    ]
    labels = [
        AGENT_NAMES.get("react", "ReAct"),
        AGENT_NAMES.get("tool_calling", "Tool calling"),
    ]
    ax.legend(handles, labels, fontsize=FONT_SIZES["legend"], loc="upper right")

    plt.tight_layout()

    output_dir = Path(__file__).parent
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"2c_task_category_line.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
