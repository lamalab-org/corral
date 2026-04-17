"""
Plot performance (pass@5) across task categories for SUBTASKS ONLY.

This script:
1. Classifies subtasks into categories (retrieval, code_execution, experiment_execution, reasoning, validation)
2. Computes average pass@5 for each category across models and agent types
3. Creates a line plot showing performance comparison

IMPORTANT: Only analyzes rows where category='subtask' to exclude top-level tasks.
"""

from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D
from plot_config import AGENT_NAMES, FONT_SIZES, MODEL_COLOURS, MODEL_NAMES
from plot_utils import classify_subtask, load_category_tags, load_reports_data

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Generate the task category performance plot."""

    # Load data
    logger.info("Loading data...")
    reports_df = load_reports_data()
    category_tags = load_category_tags()

    # Filter to SUBTASKS ONLY
    reports_df = reports_df[reports_df["category"] == "subtask"].copy()
    logger.info(f"Filtered to {len(reports_df)} rows with category='subtask'")

    # Filter to comprehensive verbosity only
    df_comp = reports_df[reports_df["Tool Verbosity"] == "comprehensive"].copy()
    logger.info(f"Filtered to {len(df_comp)} rows with comprehensive verbosity")

    # Define category order (code_execution + experiment_execution merged into execution)
    category_order = [
        "retrieval",
        "execution",
        "reasoning",
        "validation",
    ]

    # Collect scores by (model, agent_type, category)
    scores_by_group = defaultdict(lambda: defaultdict(list))
    total_subtasks_processed = 0
    classified_count = 0

    for _, row in df_comp.iterrows():
        model = row["model"]
        agent_type = row["agent_type"]
        environment = row["environment"]
        task_results = row["Task Results"]

        if not isinstance(task_results, dict):
            continue

        for subtask, result in task_results.items():
            if not isinstance(result, dict):
                continue

            total_subtasks_processed += 1

            pass_at_5 = result.get("Task Pass@5", None)
            if pass_at_5 is None:
                continue

            category = classify_subtask(subtask, environment, category_tags)
            if category is None:
                continue

            if category in ("code_execution", "experiment_execution"):
                category = "execution"

            classified_count += 1
            scores_by_group[(model, agent_type)][category].append(pass_at_5)

    logger.info(f"Processed {total_subtasks_processed} subtask results")
    logger.info(
        f"Classified {classified_count} subtasks ({100 * classified_count / total_subtasks_processed:.1f}%)"
    )

    # Compute average scores
    avg_scores = {}
    for (model, agent_type), category_scores in scores_by_group.items():
        avg_scores[(model, agent_type)] = {
            cat: np.mean(scores) if scores else np.nan
            for cat, scores in category_scores.items()
        }

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    # Define visual mappings
    agent_display = dict(AGENT_NAMES)
    agent_markers = {
        "react": "o",
        "tool_calling": "D",
    }
    agent_linestyle = {
        "react": "-",
        "tool_calling": "--",
    }

    # Define x positions for categories
    x_values = np.arange(len(category_order))

    # Plot each model-agent combination
    for (model, agent_type), category_avgs in sorted(avg_scores.items()):
        color = MODEL_COLOURS.get(model, "gray")
        marker = agent_markers.get(agent_type, "o")
        linestyle = agent_linestyle.get(agent_type, "-")
        model_display = MODEL_NAMES.get(model, model)
        agent_display_name = agent_display.get(agent_type, agent_type)

        y_values = [category_avgs.get(cat, np.nan) for cat in category_order]

        label = f"{model_display} ({agent_display_name})"
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6,
            label=label,
            alpha=0.8,
            fillstyle="none",
            linewidth=2,
        )

    # Customize axes
    ax.set_ylabel("Average Pass@5", fontsize=FONT_SIZES["tick_label"])
    ax.set_xlabel("Task Category", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])

    category_labels = [cat.replace("_", "\n").title() for cat in category_order]
    ax.set_xticks(x_values)
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZES["tick_label"])

    range_frame(ax, np.array([0, len(category_order) - 1]), np.array([0, 1]), pad=0.05)

    # Create legend
    handles = []
    labels = []

    for model in sorted({m for m, _ in avg_scores}):
        handles.append(Line2D([0], [0], color=MODEL_COLOURS.get(model, "gray"), lw=4))
        labels.append(MODEL_NAMES.get(model, model))

    for agent_type in sorted({a for _, a in avg_scores}):
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker=agent_markers.get(agent_type, "o"),
                markersize=8,
                linestyle="None",
                fillstyle="none",
            )
        )
        labels.append(f"{agent_display.get(agent_type, agent_type)} Agent")

    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=FONT_SIZES["legend"],
    )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        output_path = OUT_DIR / f"panel2_task_category.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
