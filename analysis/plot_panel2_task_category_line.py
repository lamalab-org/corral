"""Task category line plot: two lines (ReAct solid, Tool calling dashed).

Models are averaged so we get one value per agent per category.
Grey vertical lines mark each category. Purple lines. 1-col width and height.

Usage:
    python plot_panel2_task_category_line.py
"""

from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D
from plot_config import AGENT_NAMES, FONT_SIZES
from plot_utils import classify_subtask, load_category_tags, load_reports_data

lama_aesthetics.get_style("main")

LINE_COLOR = "#7150e0"

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    x_values = np.arange(len(category_order))

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

    for ext in ["pdf", "png"]:
        output_path = OUT_DIR / f"panel2_task_category_line.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
