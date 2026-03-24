"""
Plot performance (pass@5) across task categories for SUBTASKS ONLY.

This script:
1. Classifies subtasks into categories (retrieval, code_execution, experiment_execution, reasoning, validation)
2. Computes average pass@5 for each category across models and agent types
3. Creates a line plot showing performance comparison

IMPORTANT: Only analyzes rows where category='subtask' to exclude top-level tasks.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from matplotlib.lines import Line2D

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from loguru import logger  # noqa: E402
from plot_config import (  # noqa: E402
    AGENT_NAMES,
    FONT_SIZES,
    MODEL_COLOURS,
    MODEL_NAMES,
)
from plot_utils import load_reports_data  # noqa: E402

lama_aesthetics.get_style("main")


def load_category_tags():
    """Load the subtask category tags from JSON file."""
    tags_path = REPO_ROOT / "analysis" / "subtask_category_tags.json"
    with tags_path.open() as f:
        return json.load(f)


def get_env_key_mapping():
    """Map environment names in data to keys in category tags."""
    return {
        "spectra": "sptectra",
        "retro": "retrosynthesis",
        "afm": "afm",
        "catalyst": "catalyst",
        "md": "md",
        "ml": "ml",
        "resistor": "resistor",
    }


def classify_subtask(subtask, environment, category_tags):
    """
    Classify a subtask based on category tags with pattern matching.

    Args:
        subtask: Full subtask name
        environment: Environment name from data
        category_tags: Category tags dictionary

    Returns:
        Category string or None if not found
    """
    # Map environment name to tag key
    env_mapping = get_env_key_mapping()
    tag_env_key = env_mapping.get(environment, environment)

    # Get tags for this environment
    env_tags = category_tags.get(tag_env_key, {})
    if not env_tags:
        return None

    # AFM: pattern matching for subtasks
    if environment == "afm":
        # Match level 1-4 subtasks
        if "subtask_level_" in subtask:
            # Extract base name
            base_name = subtask.split("_level_")[0]
            if base_name + "_level_1" in env_tags:
                return env_tags[base_name + "_level_1"]

        # Direct match
        return env_tags.get(subtask)

    # Catalyst: extract generic pattern
    if environment == "catalyst":
        # Extract pattern (e.g., cu20_retrieve_structure -> retrieve_structure)  or si_retrieve_structure -> retrieve_structure)
        parts = subtask.split("_", 1)
        if len(parts) > 1:
            pattern = parts[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # MD: nested structure by task type
    if environment == "md":
        # Extract task type and subtask name
        for task_type in ["melting", "quenching", "surface_energy", "surface"]:
            if task_type in subtask:
                task_tags = env_tags.get(task_type, {})
                if "subtask_" in subtask:
                    subtask_name = subtask.split("subtask_")[-1]
                    # Map variations
                    if subtask_name == "diffusion_coefficient":
                        subtask_name = "diffusivity"
                    elif subtask_name == "tg_calculation":
                        subtask_name = "tg_detection"
                    elif subtask_name == "equilibration":
                        subtask_name = "structure_retrieval"

                    if subtask_name in task_tags:
                        return task_tags[subtask_name]

        return None

    # ML: pattern matching
    if environment == "ml":
        # Extract generic pattern
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            # Match batch_retrieve
            if (
                task_name.startswith("batch_retrieve_")
                and "batch_retrieve_*" in env_tags
            ):
                return env_tags["batch_retrieve_*"]
            # Direct match
            if task_name in env_tags:
                return env_tags[task_name]
        return None

    # Resistor: pattern matching
    if environment == "resistor":
        # Extract pattern (e.g., task_0_subnet_1 -> subnet_1)
        parts = subtask.split("_", 2)
        if len(parts) >= 3 and parts[0] == "task" and parts[1].isdigit():
            pattern = "_".join(parts[2:])
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # Retrosynthesis: pattern matching
    if environment == "retro":
        # Extract pattern (e.g., make_1_lvl1-apply_template-1 -> apply_template-1)
        if "-" in subtask:
            pattern = subtask.split("-", 1)[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    # Spectra: pattern matching
    if environment == "spectra":
        # Extract subtask number (e.g., ..._subtask_1 -> subtask_1)
        if "_subtask_" in subtask:
            subtask_num = "subtask_" + subtask.split("_subtask_")[-1]
            if subtask_num in env_tags:
                return env_tags[subtask_num]
        return None

    return None


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

        # For each subtask, get score and classify
        for subtask, result in task_results.items():
            if not isinstance(result, dict):
                continue

            total_subtasks_processed += 1

            # Get pass@5 score (note: key is "Task Pass@5" in the data)
            pass_at_5 = result.get("Task Pass@5", None)
            if pass_at_5 is None:
                continue

            # Classify subtask
            category = classify_subtask(subtask, environment, category_tags)
            if category is None:
                continue

            # Merge code_execution and experiment_execution into execution
            if category in ("code_execution", "experiment_execution"):
                category = "execution"

            classified_count += 1

            # Store score
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
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH * 2 / 3, ONE_COL_HEIGHT))

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
        # Get visual properties
        color = MODEL_COLOURS.get(model, "gray")
        marker = agent_markers.get(agent_type, "o")
        linestyle = agent_linestyle.get(agent_type, "-")
        model_display = MODEL_NAMES.get(model, model)
        agent_display_name = agent_display.get(agent_type, agent_type)

        # Build score array in category order
        y_values = [category_avgs.get(cat, np.nan) for cat in category_order]

        # Plot
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

    # Set x-axis labels
    category_labels = [cat.replace("_", "\n").title() for cat in category_order]
    ax.set_xticks(x_values)
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZES["tick_label"])

    # Apply range frame
    range_frame(ax, np.array([0, len(category_order) - 1]), np.array([0, 1]), pad=0.05)

    # Create legend
    handles = []
    labels = []

    # Model colors
    for model in sorted({m for m, _ in avg_scores}):
        handles.append(Line2D([0], [0], color=MODEL_COLOURS.get(model, "gray"), lw=4))
        labels.append(MODEL_NAMES.get(model, model))

    # Agent markers
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

    # Adjust layout and save
    plt.tight_layout()

    output_dir = Path(__file__).parent
    for ext in ["pdf", "png"]:
        output_path = output_dir / f"2c_task_category.{ext}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
