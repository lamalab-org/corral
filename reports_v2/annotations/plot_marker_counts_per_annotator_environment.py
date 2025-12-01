"""
Plot fine-grained marker counts for each annotator and environment.

This script creates two sets of plots:
1. All annotations: One figure per environment with subplots for each annotator
2. Multi-annotator only: Same plots but excluding entries annotated by only one annotator
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

# Data paths
DATA_FILE = Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots" / "marker_counts_per_annotator_environment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define all markers based on the labels
MARKERS = {
    "Positive": [
        "validation_attempt",
        "backtrack_trigger",
        "planning_statement",
        "reasoning_statement",
        "correct_submission",
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

# Flatten all markers
ALL_MARKERS = MARKERS["Positive"] + MARKERS["Neutral"] + MARKERS["Negative"]


def load_data():
    """Load the annotation data from parquet file."""
    logger.info(f"Loading data from {DATA_FILE}")
    annotations_df = pd.read_parquet(DATA_FILE)
    logger.info(f"Loaded {len(annotations_df)} annotations")
    return annotations_df


def get_marker_counts(df, environment, annotator):
    """
    Get counts for all markers for a specific environment and annotator.

    Returns a dictionary with marker names as keys and counts as values.
    Markers with 0 count are included.
    """
    # Filter for specific environment and annotator
    filtered = df[(df["environment"] == environment) & (df["annotator"] == annotator)]

    # Initialize counts dictionary with all markers set to 0
    counts = dict.fromkeys(ALL_MARKERS, 0)

    # Sum up counts for each marker
    for marker in ALL_MARKERS:
        marker_col = f"{marker}_count"
        if marker_col in filtered.columns:
            counts[marker] = filtered[marker_col].sum()

    return counts


def plot_environment_annotators(
    df, environment, output_suffix="", use_percentages=False
):
    """
    Create a figure with subplots for each annotator showing marker counts or percentages.

    Args:
        df: DataFrame with annotation data
        environment: The environment to plot
        output_suffix: Suffix to add to output filename (e.g., "_multi_annotator")
        use_percentages: If True, plot percentages instead of absolute counts
    """
    # Get all annotators and filter those with at least one marker in this environment
    all_annotators = sorted(df["annotator"].unique())
    annotators_with_data = []

    for annotator in all_annotators:
        counts = get_marker_counts(df, environment, annotator)
        total_counts = sum(counts.values())
        if total_counts > 0:
            annotators_with_data.append(annotator)

    annotators = annotators_with_data
    n_annotators = len(annotators)

    if n_annotators == 0:
        logger.warning(f"No annotators with data for environment: {environment}")
        return  # Create figure with subplots
    fig, axes = plt.subplots(
        n_annotators, 1, figsize=(14, 4 * n_annotators), squeeze=False
    )
    axes = axes.flatten()

    for idx, annotator in enumerate(annotators):
        ax = axes[idx]

        # Get marker counts for this annotator and environment
        counts = get_marker_counts(df, environment, annotator)

        # Prepare data for plotting
        markers = list(counts.keys())
        values = list(counts.values())

        # Convert to percentages if requested
        if use_percentages:
            total = sum(values)
            if total > 0:
                values = [(v / total) * 100 for v in values]
            else:
                values = [0] * len(values)

        # Assign colors based on marker category
        colors = []
        marker_colors = []
        for marker in markers:
            if marker in MARKERS["Positive"]:
                colors.append("lightblue")
                marker_colors.append("darkblue")
            elif marker in MARKERS["Neutral"]:
                colors.append("grey")
                marker_colors.append("dimgrey")
            elif marker in MARKERS["Negative"]:
                colors.append("lightcoral")
                marker_colors.append("darkred")
            else:
                colors.append("#34495e")
                marker_colors.append("#2c3e50")

        # Create plot using vlines with markers
        x_pos = np.arange(len(markers))

        for _i, (pos, val, color, mcolor) in enumerate(
            zip(x_pos, values, colors, marker_colors, strict=True)
        ):
            if val > 0:
                # Draw vertical line
                ax.vlines(pos, 0, val, color=color, alpha=0.5, linewidth=5)
                # Add marker at the end
                ax.plot(pos, val, "o", markersize=6, color=mcolor, alpha=1.0)

        # Set labels and title
        ylabel = "Percentage (%)" if use_percentages else "Count"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{annotator}", fontsize=12, fontweight="bold")

        # Use range_frame instead of grid
        y_max = (
            100 if use_percentages else max(values) if values and max(values) > 0 else 1
        )
        range_frame(ax, np.array([0, len(markers) - 1]), np.array([0, y_max]))

        # Set x-axis labels after range_frame
        ax.set_xticks(x_pos)
        ax.set_xticklabels(markers, rotation=45, ha="right", fontsize=8)

    # Add overall title
    title_prefix = "Marker Percentages" if use_percentages else "Marker Counts"
    fig.suptitle(
        f"{title_prefix} per Annotator - {environment.upper()}{output_suffix}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Add legend
    legend_elements = [
        Line2D([0], [0], color="lightblue", linewidth=5, alpha=0.5, label="Positive"),
        Line2D([0], [0], color="grey", linewidth=5, alpha=0.5, label="Neutral"),
        Line2D([0], [0], color="lightcoral", linewidth=5, alpha=0.5, label="Negative"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.99))

    # Save figure as PDF
    prefix = "marker_percentages" if use_percentages else "marker_counts"
    output_file = OUTPUT_DIR / f"{prefix}_{environment}{output_suffix}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format="pdf")
    logger.info(f"Saved figure to {output_file}")
    plt.close()


def filter_multi_annotator_entries(df):
    """
    Filter dataframe to keep only entries (fileIds) that have multiple annotators.

    Returns:
        Filtered dataframe
    """
    # Count annotators per fileId
    annotator_counts = df.groupby("fileId")["annotator"].nunique()

    # Get fileIds with more than one annotator
    multi_annotator_files = annotator_counts[annotator_counts > 1].index

    # Filter dataframe
    filtered_df = df[df["fileId"].isin(multi_annotator_files)]

    logger.info(f"Original entries: {len(df)}")
    logger.info(f"Entries with multiple annotators: {len(filtered_df)}")
    logger.info(
        f"Unique fileIds with multiple annotators: {len(multi_annotator_files)}"
    )

    return filtered_df


def main():
    """Main function to generate all plots."""
    # Load data
    annotations_df = load_data()

    # Get unique environments
    environments = sorted(annotations_df["environment"].unique())
    logger.info(f"Environments: {environments}")

    # Plot 1: All annotations - Counts
    logger.info("\n=== Generating count plots for all annotations ===")
    for environment in environments:
        logger.info(f"Processing environment: {environment}")
        plot_environment_annotators(
            annotations_df, environment, output_suffix="", use_percentages=False
        )

    # Plot 2: All annotations - Percentages
    logger.info("\n=== Generating percentage plots for all annotations ===")
    for environment in environments:
        logger.info(f"Processing environment: {environment}")
        plot_environment_annotators(
            annotations_df, environment, output_suffix="", use_percentages=True
        )

    # Plot 3: Multi-annotator only - Counts
    logger.info("\n=== Generating count plots for multi-annotator entries only ===")
    multi_annotator_df = filter_multi_annotator_entries(annotations_df)

    for environment in environments:
        logger.info(f"Processing environment: {environment}")
        plot_environment_annotators(
            multi_annotator_df,
            environment,
            output_suffix="_multi_annotator",
            use_percentages=False,
        )

    # Plot 4: Multi-annotator only - Percentages
    logger.info(
        "\n=== Generating percentage plots for multi-annotator entries only ==="
    )
    for environment in environments:
        logger.info(f"Processing environment: {environment}")
        plot_environment_annotators(
            multi_annotator_df,
            environment,
            output_suffix="_multi_annotator",
            use_percentages=True,
        )

    logger.info("\n=== All plots generated successfully ===")


if __name__ == "__main__":
    main()
