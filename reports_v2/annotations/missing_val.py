"""
Script to identify missing annotations per environment and annotator.

For each environment, this script:
1. Finds all files that should be labeled (from accepted_questions)
2. Checks which annotators have labeled files in that environment
3. For each annotator that has labeled at least one file in that environment,
   calculates which files are still missing
4. Saves a report per environment in the missing_files directory
5. Generates summary bar plots showing annotation progress
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics.plotutils import range_frame
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lama_aesthetics.get_style("main")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ACCEPTED_QUESTIONS_PATH = BASE_DIR / "accepted_questions"
ANNOTATIONS_PATH = (
    BASE_DIR
    / "reports_v2"
    / "annotations"
    / "data"
    / "cleaned_corral_annotations.parquet"
)
OUTPUT_DIR = BASE_DIR / "reports_v2" / "annotations" / "missing_val"

# Define which annotators are assigned to each environment
# This overrides the automatic detection based on who has already labeled files
ASSIGNED_ANNOTATORS = {
    "afm": ["Indrajeet", "Nawaf", "Sajid"],
    "catalyst": ["Chandan", "Martino", "Nawaf"],
    "md": ["Chandan", "Nawaf", "Sajid"],
    "ml": ["Chandan", "Indrajeet", "Nawaf"],
    "resistor": ["Chandan", "Indrajeet", "Nawaf"],
    "retrosynthesis": ["Kevin", "Martino", "Sadra"],
}

# Colors for the plots
COLOR_ANNOTATED = "#2ecc71"  # Green
COLOR_MISSING = "#e74c3c"  # Red


def get_files_to_label_per_env():
    """Get all JSON files that should be labeled per environment."""
    files_per_env = defaultdict(set)

    for model_path in ACCEPTED_QUESTIONS_PATH.iterdir():
        if not model_path.is_dir():
            continue

        for env_path in model_path.iterdir():
            if not env_path.is_dir():
                continue

            env = env_path.name
            # Find all json files in this environment
            for json_file in env_path.rglob("*.json"):
                files_per_env[env].add(json_file.name)

    return {k: sorted(v) for k, v in files_per_env.items()}


def get_annotated_files_per_env_annotator(df):
    """Get files annotated per environment and annotator from the dataframe."""
    annotated = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        env = row["environment"]
        annotator = row["annotator"]
        file_id = row["fileId"]
        annotated[env][annotator].add(file_id)

    return annotated


def get_annotators_per_env():
    """Get annotators assigned to each environment (from ASSIGNED_ANNOTATORS config)."""
    # Use the configured assignments instead of inferring from data
    return {env: set(annotators) for env, annotators in ASSIGNED_ANNOTATORS.items()}


def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotation data
    logger.info("Loading annotations from %s", ANNOTATIONS_PATH)
    annotations_df = pd.read_parquet(ANNOTATIONS_PATH)
    logger.info("Loaded %d annotations", len(annotations_df))

    # Get files to label per environment
    files_to_label = get_files_to_label_per_env()
    logger.info("Files to label per environment:")
    for env, files in files_to_label.items():
        logger.info("  %s: %d files", env, len(files))

    # Get annotated files per env/annotator
    annotated = get_annotated_files_per_env_annotator(annotations_df)

    # Get annotators per environment (those who have labeled at least one file)
    annotators_per_env = get_annotators_per_env()

    logger.info("Annotators per environment:")
    for env, annotators in sorted(annotators_per_env.items()):
        logger.info("  %s: %s", env, sorted(annotators))

    # For each environment, create a report
    for env in files_to_label:
        if env not in annotators_per_env:
            logger.warning("Skipping %s: No annotators found", env)
            continue

        files_needed = set(files_to_label[env])
        annotators = annotators_per_env[env]

        report = {
            "environment": env,
            "total_files_to_label": len(files_needed),
            "files_to_label": sorted(files_needed),
            "annotators": {},
        }

        for annotator in sorted(annotators):
            annotated_files = annotated[env][annotator]
            missing_files = files_needed - annotated_files

            report["annotators"][annotator] = {
                "files_annotated_count": len(annotated_files),
                "files_annotated": sorted(annotated_files),
                "missing_files_count": len(missing_files),
                "missing_files": sorted(missing_files),
            }

        # Save report
        output_path = OUTPUT_DIR / f"{env}_missing_report.json"
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved report for %s to %s", env, output_path)

        # Log summary
        logger.info("  Total files to label: %d", len(files_needed))
        for annotator in sorted(annotators):
            annotated_count = len(annotated[env][annotator])
            missing_count = len(files_needed - annotated[env][annotator])
            logger.info(
                "    %s: %d annotated, %d missing",
                annotator,
                annotated_count,
                missing_count,
            )

    # Generate summary plots
    generate_summary_plots(files_to_label, annotated, annotators_per_env)


def generate_summary_plots(files_to_label, annotated, annotators_per_env):
    """Generate summary bar plots showing annotation progress per environment."""

    # Prepare data for plotting
    environments = sorted(files_to_label.keys())

    # Create a plot for each environment showing annotator progress
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            lama_aesthetics.TWO_COL_WIDTH * 1.5,
            lama_aesthetics.TWO_COL_HEIGHT * 2,
        ),
    )
    axes = axes.flatten()

    for idx, env in enumerate(environments):
        ax = axes[idx]

        if env not in annotators_per_env:
            ax.set_title(f"{env}\n(No annotators)")
            ax.axis("off")
            continue

        files_needed = set(files_to_label[env])
        total_files = len(files_needed)
        annotators = sorted(annotators_per_env[env])

        annotated_counts = []
        missing_counts = []

        for annotator in annotators:
            annotated_files = annotated[env][annotator]
            missing_files = files_needed - annotated_files
            annotated_counts.append(len(annotated_files))
            missing_counts.append(len(missing_files))

        x = np.arange(len(annotators))

        # Use vlines style like plot_marker_counts_per_annotator_environment.py
        for i, (ann_count, miss_count) in enumerate(
            zip(annotated_counts, missing_counts, strict=True)
        ):
            # Draw annotated portion (green vline from 0 to annotated count)
            if ann_count > 0:
                ax.vlines(i, 0, ann_count, color="lightgreen", alpha=0.6, linewidth=8)
                ax.plot(i, ann_count, "o", markersize=8, color="darkgreen", alpha=1.0)

            # Draw missing portion (red vline from annotated count to total)
            if miss_count > 0:
                ax.vlines(
                    i,
                    ann_count,
                    ann_count + miss_count,
                    color="lightcoral",
                    alpha=0.6,
                    linewidth=8,
                )
                ax.plot(
                    i,
                    ann_count + miss_count,
                    "o",
                    markersize=8,
                    color="darkred",
                    alpha=1.0,
                )

        ax.set_ylabel("Number of Files")
        ax.set_title(f"{env.upper()}\n(Total: {total_files} files)")
        ax.set_xticks(x)
        ax.set_xticklabels(annotators, rotation=45, ha="right")

        # Apply range_frame
        range_frame(ax, x, np.array([0, total_files]), pad=0.05)

        if idx == 0:
            # Add legend
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="lightgreen",
                    linewidth=5,
                    alpha=0.6,
                    label="Annotated",
                ),
                Line2D(
                    [0],
                    [0],
                    color="lightcoral",
                    linewidth=5,
                    alpha=0.6,
                    label="Missing",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    plot_path_pdf = OUTPUT_DIR / "annotation_progress_by_environment.pdf"
    plt.savefig(plot_path_pdf, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path_pdf)
    plt.close()


if __name__ == "__main__":
    main()
