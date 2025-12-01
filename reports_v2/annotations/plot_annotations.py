import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

task_data = Path(__file__).parent / "data" / "all_tasks_info.json"
main_results = Path(__file__).parent / "data" / "main_results.json"
annotation_data = Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"

LABELS = {
    "Positive": [
        "validation_attempt",
        "backtrack_trigger",
        "planning_statement",
        "reasoning_statement",
        "correct_submission",
        "todo_list",
    ],
    "Neutral": ["neutral", "iteration_limit"],
    "Negative": [
        "missing_validation",
        "unnecessary_tool_use",
        "non_sense",
        "loop_instance",
        "hallucination",
        "wrong_planning",
        "wrong_reasoning",
        "syntax_error",
        "early_final_answer",
        "give_up",
        "inefficient_tool_call",
        "misunderstood_tool",
    ],
}


def get_tasks_per_environment(tasks):
    env_tasks = {}
    for task in tasks:
        env = task["environment"]
        if env not in env_tasks:
            env_tasks[env] = []
        env_tasks[env].append(task)
    return env_tasks


def build_filename_to_env_map(tasks):
    """Build a mapping from filename patterns to environments based on actual task data."""
    filename_to_env = {}

    for task in tasks:
        env = task["environment"]
        task_id = task["task_id"]

        # Store the task_id to environment mapping
        # Filenames typically start with task_id
        filename_to_env[task_id] = env

    return filename_to_env


def get_environment_from_filename(filename, filename_to_env):
    """Extract environment from filename using the mapping from task data."""
    # Remove .json extension and timestamp
    base_name = filename.replace(".json", "")

    # Try to match against task_ids in the mapping
    for task_id, env in filename_to_env.items():
        if base_name.startswith(task_id):
            return env

    # If no match found, return "unknown"
    return "unknown"


def count_markers_per_environment(annotations, filename_to_env):
    """Count each marker type per environment."""
    env_marker_counts = {}

    for annotation in annotations:
        filename = annotation["fileId"]
        env = get_environment_from_filename(filename, filename_to_env)

        if env not in env_marker_counts:
            env_marker_counts[env] = {}

        for node in annotation["fileNodes"]:
            if node.get("annotatable", False) and "markers" in node:
                for marker in node["markers"]:
                    if marker not in env_marker_counts[env]:
                        env_marker_counts[env][marker] = 0
                    env_marker_counts[env][marker] += 1

    return env_marker_counts


def count_sentiment_per_environment(annotations, filename_to_env):
    """Count positive, negative, and neutral markers per environment."""
    env_sentiment_counts = {}

    for annotation in annotations:
        filename = annotation["fileId"]
        env = get_environment_from_filename(filename, filename_to_env)

        if env not in env_sentiment_counts:
            env_sentiment_counts[env] = {"positive": 0, "negative": 0, "neutral": 0}

        for node in annotation["fileNodes"]:
            if node.get("annotatable", False) and "markers" in node:
                for marker in node["markers"]:
                    if marker in LABELS["Positive"]:
                        env_sentiment_counts[env]["positive"] += 1
                    elif marker in LABELS["Negative"]:
                        env_sentiment_counts[env]["negative"] += 1
                    elif marker in LABELS["Neutral"]:
                        env_sentiment_counts[env]["neutral"] += 1

    return env_sentiment_counts


def count_annotations_per_environment(annotations, filename_to_env):
    """Count number of annotated files per environment."""
    env_annotation_counts = {}

    for annotation in annotations:
        filename = annotation["fileId"]
        env = get_environment_from_filename(filename, filename_to_env)

        if env not in env_annotation_counts:
            env_annotation_counts[env] = 0
        env_annotation_counts[env] += 1

    return env_annotation_counts


def calculate_averages(sentiment_counts, annotation_counts):
    """Calculate average sentiment markers per annotation."""
    averages = {}

    for env in sentiment_counts:
        if env in annotation_counts and annotation_counts[env] > 0:
            averages[env] = {
                "avg_positive": sentiment_counts[env]["positive"]
                / annotation_counts[env],
                "avg_negative": sentiment_counts[env]["negative"]
                / annotation_counts[env],
                "avg_neutral": sentiment_counts[env]["neutral"]
                / annotation_counts[env],
            }

    return averages


def plot_sentiment_counts_per_environment(sentiment_counts, output_path=None):
    """
    Create a cumulative bar plot showing sentiment counts per environment.

    Args:
        sentiment_counts: Dictionary with environment as keys and sentiment counts as values
        output_path: Optional path to save the figure. If None, the plot is displayed.
    """
    # Sort environments for consistent ordering
    environments = sorted(sentiment_counts.keys())

    # Extract counts for each sentiment
    positive_counts = [sentiment_counts[env]["positive"] for env in environments]
    neutral_counts = [sentiment_counts[env]["neutral"] for env in environments]
    negative_counts = [sentiment_counts[env]["negative"] for env in environments]

    # Create the figure
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.TWO_COL_WIDTH, lama_aesthetics.TWO_COL_HEIGHT)
    )

    # Set up the x positions
    x = range(len(environments))

    # Draw vertical lines for each sentiment type
    for i, _env in enumerate(environments):
        # Positive line (from 0 to positive_counts)
        ax.vlines(
            i,
            0,
            positive_counts[i],
            color="lightblue",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            positive_counts[i],
            "o",
            markersize=6,
            color="darkblue",
            alpha=1.0,
        )

        # Neutral line (from positive to positive + neutral)
        neutral_start = positive_counts[i]
        neutral_end = positive_counts[i] + neutral_counts[i]
        ax.vlines(
            i,
            neutral_start,
            neutral_end,
            color="grey",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            neutral_end,
            "o",
            markersize=6,
            color="dimgrey",
            alpha=1.0,
        )

        # Negative line (from positive + neutral to positive + neutral + negative)
        negative_start = positive_counts[i] + neutral_counts[i]
        negative_end = positive_counts[i] + neutral_counts[i] + negative_counts[i]
        ax.vlines(
            i,
            negative_start,
            negative_end,
            color="lightcoral",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            negative_end,
            "o",
            markersize=6,
            color="darkred",
            alpha=1.0,
        )

    # Create legend entries manually
    legend_elements = [
        Line2D([0], [0], color="lightblue", linewidth=5, alpha=0.5, label="Positive"),
        Line2D([0], [0], color="grey", linewidth=5, alpha=0.5, label="Neutral"),
        Line2D([0], [0], color="lightcoral", linewidth=5, alpha=0.5, label="Negative"),
    ]

    # Customize the plot
    ax.set_xlabel("Environment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(environments, rotation=45, ha="right")
    range_frame(
        ax,
        np.array([0, len(environments) - 1]),
        np.array(
            [
                0,
                max(
                    [
                        positive_counts[i] + neutral_counts[i] + negative_counts[i]
                        for i in range(len(environments))
                    ]
                ),
            ]
        ),
    )
    ax.legend(handles=legend_elements)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or display
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")

    plt.close()


def plot_average_markers_per_environment(averages, output_path=None):
    """
    Create a cumulative bar plot showing average sentiment markers per environment.

    Args:
        averages: Dictionary with environment as keys and average sentiment counts as values
        output_path: Optional path to save the figure. If None, the plot is displayed.
    """
    # Sort environments for consistent ordering
    environments = sorted(averages.keys())

    # Extract averages for each sentiment
    avg_positive = [averages[env]["avg_positive"] for env in environments]
    avg_neutral = [averages[env]["avg_neutral"] for env in environments]
    avg_negative = [averages[env]["avg_negative"] for env in environments]

    # Create the figure
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.TWO_COL_WIDTH, lama_aesthetics.TWO_COL_HEIGHT)
    )

    # Set up the x positions
    x = range(len(environments))

    # Draw vertical lines for each sentiment type
    for i, _env in enumerate(environments):
        # Positive line (from 0 to avg_positive)
        ax.vlines(
            i,
            0,
            avg_positive[i],
            color="lightblue",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            avg_positive[i],
            "o",
            markersize=6,
            color="darkblue",
            alpha=1.0,
        )

        # Neutral line (from positive to positive + neutral)
        neutral_start = avg_positive[i]
        neutral_end = avg_positive[i] + avg_neutral[i]
        ax.vlines(
            i,
            neutral_start,
            neutral_end,
            color="grey",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            neutral_end,
            "o",
            markersize=6,
            color="dimgrey",
            alpha=1.0,
        )

        # Negative line (from positive + neutral to positive + neutral + negative)
        negative_start = avg_positive[i] + avg_neutral[i]
        negative_end = avg_positive[i] + avg_neutral[i] + avg_negative[i]
        ax.vlines(
            i,
            negative_start,
            negative_end,
            color="lightcoral",
            alpha=0.5,
            linewidth=5,
        )
        # Add marker at the end
        ax.plot(
            i,
            negative_end,
            "o",
            markersize=6,
            color="darkred",
            alpha=1.0,
        )

    # Create legend entries manually
    legend_elements = [
        Line2D([0], [0], color="lightblue", linewidth=5, alpha=0.5, label="Positive"),
        Line2D([0], [0], color="grey", linewidth=5, alpha=0.5, label="Neutral"),
        Line2D([0], [0], color="lightcoral", linewidth=5, alpha=0.5, label="Negative"),
    ]

    # Customize the plot
    ax.set_xlabel("Environment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Count per File", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(environments, rotation=45, ha="right")
    range_frame(
        ax,
        np.array([0, len(environments) - 1]),
        np.array(
            [
                0,
                max(
                    [
                        avg_positive[i] + avg_neutral[i] + avg_negative[i]
                        for i in range(len(environments))
                    ]
                ),
            ]
        ),
    )
    ax.legend(handles=legend_elements)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or display
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")

    plt.close()


def plot_marker_counts_per_environment(marker_counts, output_dir=None):
    """
    Create bar plots showing the count of each marker per environment.
    One plot is created per environment.

    Args:
        marker_counts: Dictionary with environment as keys and marker counts as values
        output_dir: Optional directory path to save the figures. If None, plots are displayed.
    """
    # Collect all unique markers from LABELS
    all_markers = []
    for category in LABELS.values():
        all_markers.extend(category)

    # Sort environments for consistent ordering
    environments = sorted(marker_counts.keys())

    # Create a plot for each environment
    for env in environments:
        # Get counts for all markers, using 0 if marker not present
        counts = [marker_counts[env].get(marker, 0) for marker in all_markers]

        # Create the figure
        fig, ax = plt.subplots(
            figsize=(lama_aesthetics.TWO_COL_WIDTH, lama_aesthetics.TWO_COL_HEIGHT)
        )

        # Set up the x positions
        x = range(len(all_markers))

        # Draw vertical lines for each marker with colors based on sentiment category
        for i, marker in enumerate(all_markers):
            if marker in LABELS["Positive"]:
                line_color = "lightblue"
                marker_color = "darkblue"
            elif marker in LABELS["Negative"]:
                line_color = "lightcoral"
                marker_color = "darkred"
            else:  # Neutral
                line_color = "grey"
                marker_color = "dimgrey"

            # Draw vertical line from 0 to count
            ax.vlines(
                i,
                0,
                counts[i],
                color=line_color,
                alpha=0.5,
                linewidth=5,
            )
            # Add marker at the end
            ax.plot(
                i,
                counts[i],
                "o",
                markersize=6,
                color=marker_color,
                alpha=1.0,
            )

        # Create legend entries manually
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="lightblue",
                linewidth=5,
                alpha=0.7,
                label="Positive",
            ),
            Line2D([0], [0], color="grey", linewidth=5, alpha=0.7, label="Neutral"),
            Line2D(
                [0],
                [0],
                color="lightcoral",
                linewidth=5,
                alpha=0.7,
                label="Negative",
            ),
        ]

        # Customize the plot
        ax.set_xlabel("Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        range_frame(
            ax,
            np.array([0, len(all_markers) - 1]),
            np.array([0, max(counts) if max(counts) > 0 else 1]),
        )
        ax.set_xticklabels(all_markers, rotation="vertical")
        ax.legend(handles=legend_elements)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save or display
        if output_dir:
            output_path = output_dir / f"marker_counts_{env}.pdf"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {output_path}")

        plt.close()


def plot_specific_marker_counts(marker_type, annotations, output_path=None):
    """
    Plot counts of a specific marker type (positive or negative) across all models, environments, and agents.

    Args:
        marker_type: Type of markers to plot ("positive" or "negative")
        annotations: List of annotation data
        output_path: Optional path to save the figure
    """
    # Validate marker_type
    if marker_type.lower() not in ["positive", "negative"]:
        logger.error(
            f"Invalid marker_type: {marker_type}. Must be 'positive' or 'negative'"
        )
        return

    # Determine which markers to count
    if marker_type.lower() == "positive":
        target_markers = LABELS["Positive"]
    else:
        target_markers = LABELS["Negative"]

    # Collect counts per (model, environment, agent_type)
    counts_by_combo = defaultdict(int)

    for annotation in annotations:
        model = annotation.get("model")
        env = annotation.get("environment")
        agent_type = annotation.get("agent_type")

        # Skip if metadata is missing
        if not all([model, env, agent_type]):
            continue

        # Count markers of the specified type
        for node in annotation["fileNodes"]:
            if node.get("annotatable", False) and "markers" in node:
                for marker in node["markers"]:
                    if marker in target_markers:
                        counts_by_combo[(model, env, agent_type)] += 1

    if not counts_by_combo:
        logger.warning(f"No data found for {marker_type} markers")
        return

    # Organize data for plotting
    models = sorted({combo[0] for combo in counts_by_combo})
    environments = sorted({combo[1] for combo in counts_by_combo})
    agent_types = sorted({combo[2] for combo in counts_by_combo})

    # Create a grouped bar plot
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.TWO_COL_WIDTH * 1.5, lama_aesthetics.TWO_COL_HEIGHT)
    )

    # Set up colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = {model: colors[i] for i, model in enumerate(models)}

    # Set up hatching patterns for agent types
    hatch_patterns = ["", "//", "\\\\", "xx", "++", "||", "--", "oo"]
    agent_hatch_map = {
        agent: hatch_patterns[i % len(hatch_patterns)]
        for i, agent in enumerate(agent_types)
    }

    # Create x positions for environments
    x = np.arange(len(environments))
    width = 0.8 / (len(models) * len(agent_types))  # Width of each bar

    # Plot bars
    bar_offset = 0
    for model in models:
        for agent in agent_types:
            counts = [
                counts_by_combo.get((model, env, agent), 0) for env in environments
            ]

            ax.bar(
                x + bar_offset * width,
                counts,
                width,
                label=f"{model} - {agent}",
                color=model_color_map[model],
                hatch=agent_hatch_map[agent],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )

            bar_offset += 1

    # Customize the plot
    ax.set_xlabel("Environment", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        f"{marker_type.capitalize()} Marker Count", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"{marker_type.capitalize()} Markers by Model, Environment, and Agent Type",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + width * (len(models) * len(agent_types) - 1) / 2)
    ax.set_xticklabels(environments, rotation=45, ha="right")

    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9, fontsize=9)

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    plt.close()


def plot_dispersion_of_marker_performance(
    marker_type, annotations, main_results_data, output_path=None
):
    """
    Plot the average counts of markers per environment vs the overall score in the environment.

    Args:
        marker_type: Type of markers to plot ("positive", "negative", or "neutral")
        annotations: List of annotation data
        main_results_data: List of main results data
        output_path: Optional path to save the figure
    """
    # Map agent types between annotations and results
    agent_type_map = {
        "ReActAgent": "react",
        "ToolCallingAgent": "tool",
    }

    # Filter main_results_data for tool_verbosity == "workflow"
    workflow_results = [
        r for r in main_results_data if r.get("tool_verbosity") == "workflow"
    ]

    # Aggregate marker counts per (environment, level, agent_type, model)
    marker_counts_by_combo = defaultdict(list)

    for annotation in annotations:
        # Get metadata from annotation (already included in the data)
        env = annotation.get("environment")
        level = annotation.get("level")
        agent_type = annotation.get("agent_type")
        model = annotation.get("model")

        # Skip if any metadata is missing
        if not all([env, level is not None, agent_type, model]):
            continue

        # Count markers of the specified type for this annotation
        marker_count = 0
        for node in annotation["fileNodes"]:
            if node.get("annotatable", False) and "markers" in node:
                for marker in node["markers"]:
                    if (
                        (
                            marker_type.lower() == "positive"
                            and marker in LABELS["Positive"]
                        )
                        or (
                            marker_type.lower() == "negative"
                            and marker in LABELS["Negative"]
                        )
                        or (
                            marker_type.lower() == "neutral"
                            and marker in LABELS["Neutral"]
                        )
                    ):
                        marker_count += 1

        # Aggregate counts by combination
        key = (env, level, agent_type, model)
        marker_counts_by_combo[key].append(marker_count)

    # Now create data points with averaged marker counts
    data_points = []
    for (env, level, agent_type, model), counts in marker_counts_by_combo.items():
        # Map agent_type to match main_results format
        agent_type_mapped = agent_type_map.get(agent_type, agent_type)

        # Calculate average marker count for this combination
        avg_marker_count = sum(counts) / len(counts) if counts else 0

        # Find matching result in workflow_results
        matching_results = [
            r
            for r in workflow_results
            if r["environment"] == env
            and r["level"] == level
            and r["agent_type"] == agent_type_mapped
            and r["model"] == model
        ]

        # Add one data point per combination
        data_points.extend(
            [
                {
                    "environment": env,
                    "level": level,
                    "agent_type": agent_type,
                    "model": model,
                    "marker_count": avg_marker_count,
                    "average_score": result["average_score"],
                }
                for result in matching_results
            ]
        )

    if not data_points:
        logger.warning(f"No data points found for {marker_type} markers")
        return

    # Group data by environment and level for plotting
    # Colors per environment
    environments = sorted({dp["environment"] for dp in data_points})
    colors = plt.cm.tab10(np.linspace(0, 1, len(environments)))
    env_color_map = {env: colors[i] for i, env in enumerate(environments)}

    # Markers per level
    levels = sorted({dp["level"] for dp in data_points})
    marker_styles = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    level_marker_map = {
        level: marker_styles[i % len(marker_styles)] for i, level in enumerate(levels)
    }

    # Create the plot
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.TWO_COL_WIDTH, lama_aesthetics.TWO_COL_HEIGHT)
    )

    # Plot each data point
    for dp in data_points:
        ax.scatter(
            dp["marker_count"],
            dp["average_score"],
            color=env_color_map[dp["environment"]],
            marker=level_marker_map[dp["level"]],
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

    # Create legend for environments (colors)
    env_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=env,
            markerfacecolor=env_color_map[env],
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for env in environments
    ]

    # Create legend for levels (markers)
    level_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=level_marker_map[level],
            color="w",
            label=f"Level {level}",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for level in levels
    ]

    # Add both legends
    first_legend = ax.legend(
        handles=env_legend_elements,
        title="Environment",
        loc="upper left",
        framealpha=0.9,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=level_legend_elements, title="Level", loc="upper right", framealpha=0.9
    )

    # Customize the plot
    ax.set_xlabel(
        f"Average {marker_type.capitalize()} Marker Count",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Average Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{marker_type.capitalize()} Markers vs Performance",
        fontsize=14,
        fontweight="bold",
    )
    range_frame(
        ax,
        np.array(
            [
                0,
                max(dp["marker_count"] for dp in data_points),
            ]
        ),
        np.array([0, 1]),
    )

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    plt.close()


def main():
    annotations_df = pd.read_parquet(annotation_data)

    # Parse JSON strings in fileNodes column
    annotations_df["fileNodes"] = annotations_df["fileNodes"].apply(json.loads)

    annotations = annotations_df.to_dict("records")

    with task_data.open() as f:
        tasks = json.load(f)

    # Get task-to-environment mapping
    env_tasks = get_tasks_per_environment(tasks)
    logger.debug("Task-to-environment mapping:")
    for env, task_list in env_tasks.items():
        logger.debug(f"{env}: {len(task_list)} tasks")
    logger.debug(" ")

    # Build filename to environment mapping
    filename_to_env = build_filename_to_env_map(tasks)

    logger.debug("=" * 80)
    logger.debug("ANNOTATION STATISTICS BY ENVIRONMENT")
    logger.debug("=" * 80)

    # Count markers per environment
    marker_counts = count_markers_per_environment(annotations, filename_to_env)
    logger.debug("\n1. MARKER COUNTS PER ENVIRONMENT:")
    logger.debug("-" * 80)
    for env in sorted(marker_counts.keys()):
        logger.debug(f"\n{env.upper()}:")
        for marker, count in sorted(
            marker_counts[env].items(), key=lambda x: x[1], reverse=True
        ):
            logger.debug(f"  {marker}: {count}")

    # Count sentiment per environment
    sentiment_counts = count_sentiment_per_environment(annotations, filename_to_env)
    logger.debug("\n\n2. SENTIMENT COUNTS PER ENVIRONMENT:")
    logger.debug("-" * 80)
    for env in sorted(sentiment_counts.keys()):
        logger.debug(f"\n{env.upper()}:")
        logger.debug(f"  Positive: {sentiment_counts[env]['positive']}")
        logger.debug(f"  Negative: {sentiment_counts[env]['negative']}")
        logger.debug(f"  Neutral: {sentiment_counts[env]['neutral']}")

    # Count annotations per environment
    annotation_counts = count_annotations_per_environment(annotations, filename_to_env)
    logger.debug("\n\n3. NUMBER OF ANNOTATED FILES PER ENVIRONMENT:")
    logger.debug("-" * 80)
    for env in sorted(annotation_counts.keys()):
        logger.debug(f"{env}: {annotation_counts[env]} files")

    # Calculate averages
    averages = calculate_averages(sentiment_counts, annotation_counts)
    logger.debug("\n\n4. AVERAGE MARKERS PER ANNOTATION BY ENVIRONMENT:")
    logger.debug("-" * 80)
    for env in sorted(averages.keys()):
        logger.debug(f"\n{env.upper()}:")
        logger.debug(f"  Average Positive: {averages[env]['avg_positive']:.2f}")
        logger.debug(f"  Average Negative: {averages[env]['avg_negative']:.2f}")
        logger.debug(f"  Average Neutral: {averages[env]['avg_neutral']:.2f}")

    # Prepare results dictionary
    results = {
        "task_to_environment_mapping": {
            env: len(task_list) for env, task_list in env_tasks.items()
        },
        "marker_counts_per_environment": marker_counts,
        "sentiment_counts_per_environment": sentiment_counts,
        "annotation_counts_per_environment": annotation_counts,
        "average_markers_per_annotation": averages,
    }

    # Write results to JSON file
    output_file = Path(__file__).parent / "data" / "annotation_main_results.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results written to {output_file}")

    # Create sentiment counts bar plot
    plot_output_path = (
        Path(__file__).parent / "plots" / "sentiment_counts_per_environment.pdf"
    )
    plot_sentiment_counts_per_environment(
        sentiment_counts, output_path=plot_output_path
    )

    # Create average markers bar plot
    avg_plot_output_path = (
        Path(__file__).parent / "plots" / "average_markers_per_environment.pdf"
    )
    plot_average_markers_per_environment(averages, output_path=avg_plot_output_path)

    # Create marker counts bar plots (one per environment)
    marker_plots_output_dir = Path(__file__).parent / "plots"
    plot_marker_counts_per_environment(
        marker_counts, output_dir=marker_plots_output_dir
    )


if __name__ == "__main__":
    main()
