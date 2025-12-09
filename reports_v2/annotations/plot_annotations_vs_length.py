"""
Plot marker counts against number of nodes for each environment.

This script creates plots showing the relationship between the number of nodes
in a trace and the count of different marker types (positive, negative, neutral).
For each environment, it creates 3 plots (one per marker type), with each plot
containing 4 subplots for the different agent-model combinations.
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")


def load_data(filepath):
    """Load the parquet file and calculate the number of nodes.

    Args:
        filepath: Path to the parquet file to load.

    Returns:
        DataFrame with an additional 'num_nodes' column.
    """
    df_data = pd.read_parquet(filepath)

    # Calculate number of nodes from fileNodes JSON
    df_data["num_nodes"] = df_data["fileNodes"].apply(
        lambda x: len(json.loads(x)) if isinstance(x, str) else 0
    )

    return df_data


def create_marker_plot(df, environment, marker_type, output_dir):
    """Create a plot for a specific environment and marker type.

    Args:
        df: The filtered dataframe for the specific environment.
        environment: The environment name.
        marker_type: One of 'positive', 'negative', or 'neutral'.
        output_dir: Directory to save the plot.
    """
    # Map marker type to column name
    marker_column_map = {
        "positive": "positive_marker_count",
        "negative": "negative_marker_count",
        "neutral": "neutral_marker_count",
    }

    column_name = marker_column_map[marker_type]

    # Get unique agents and models
    agents = sorted(df["agent_type"].unique())
    models = sorted(df["model"].unique())

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(lama_aesthetics.TWO_COL_WIDTH * 2, lama_aesthetics.TWO_COL_HEIGHT * 2),
    )
    fig.suptitle(
        f"{environment.upper()} - {marker_type.capitalize()} Marker Count vs Number of Nodes",
        fontsize=16,
        fontweight="bold",
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Create a subplot for each agent-model combination
    plot_idx = 0
    for agent in agents:
        for model in models:
            if plot_idx >= 4:
                break

            # Filter data for this combination
            mask = (df["agent_type"] == agent) & (df["model"] == model)
            subset = df[mask]

            if len(subset) == 0:
                # If no data, create empty plot with message
                axes_flat[plot_idx].text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=axes_flat[plot_idx].transAxes,
                )
                axes_flat[plot_idx].set_title(f"{agent} + {model}")
                plot_idx += 1
                continue

            # Create scatter plot
            ax = axes_flat[plot_idx]
            ax.scatter(subset["num_nodes"], subset[column_name], alpha=0.6, s=50)

            # Add trend line if there's enough data
            if len(subset) > 1:
                z = np.polyfit(subset["num_nodes"], subset[column_name], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(
                    subset["num_nodes"].min(), subset["num_nodes"].max(), 100
                )
                ax.plot(
                    x_trend,
                    p(x_trend),
                    "--",
                    alpha=0.8,
                    linewidth=2,
                    label="Trend line",
                )
                ax.legend()

            # Calculate correlation if there's variation
            if subset["num_nodes"].std() > 0 and subset[column_name].std() > 0:
                corr = subset["num_nodes"].corr(subset[column_name])
                ax.text(
                    0.05,
                    0.95,
                    f"Corr: {corr:.3f}",
                    transform=ax.transAxes,
                    bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
                    verticalalignment="top",
                )

            # Labels and title
            ax.set_xlabel("Number of Nodes", fontweight="bold")
            ax.set_ylabel(f"{marker_type.capitalize()} Marker Count", fontweight="bold")
            ax.set_title(
                f"{agent} + {model} (n={len(subset)})", fontsize=12, fontweight="bold"
            )

            # Apply range_frame for cleaner borders
            range_frame(
                ax,
                np.array([subset["num_nodes"].min(), subset["num_nodes"].max()]),
                np.array([subset[column_name].min(), subset[column_name].max()]),
            )

            plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, 4):
        axes_flat[idx].axis("off")

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"{environment}_{marker_type}_markers_vs_nodes.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {output_file}")


def main():
    """Main function to generate all plots.

    Reads data from filtered_corral_annotations.parquet (or falls back to
    cleaned_corral_annotations.parquet), creates plots for each environment
    and marker type combination, and prints summary statistics.
    """
    # Define input and output paths
    input_file = Path("filtered_corral_annotations.parquet")
    output_dir = Path(__file__).parent / "plots" / "marker_counts_vs_length"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Check if input file exists
    if not input_file.exists():
        logger.warning(f"{input_file} does not exist yet.")
        logger.info("Using data/cleaned_corral_annotations.parquet for demonstration.")
        input_file = (
            Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"
        )

    # Load data
    logger.info("Loading data...")
    df_file = load_data(input_file)
    logger.info(f"Loaded {len(df_file)} records")

    # Get unique environments
    environments = sorted(df_file["environment"].unique())
    logger.info(f"Environments found: {environments}")

    # Marker types to plot
    marker_types = ["positive", "negative", "neutral"]

    # Generate plots for each environment and marker type
    total_plots = len(environments) * len(marker_types)
    current_plot = 0

    for environment in environments:
        logger.info(f"Processing environment: {environment}")
        env_df = df_file[df_file["environment"] == environment].copy()
        logger.info(f"  Records: {len(env_df)}")

        for marker_type in marker_types:
            current_plot += 1
            logger.info(
                f"  Creating {marker_type} marker plot ({current_plot}/{total_plots})..."
            )
            create_marker_plot(env_df, environment, marker_type, output_dir)

    logger.info(f"All plots saved to: {output_dir}/")
    logger.info(f"Total plots created: {total_plots}")

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    for env in environments:
        env_df = df_file[df_file["environment"] == env]
        logger.info(f"\n{env.upper()}:")
        logger.info(f"  Total records: {len(env_df)}")
        logger.info(f"  Avg nodes: {env_df['num_nodes'].mean():.1f}")
        logger.info(
            f"  Avg positive markers: {env_df['positive_marker_count'].mean():.1f}"
        )
        logger.info(
            f"  Avg negative markers: {env_df['negative_marker_count'].mean():.1f}"
        )
        logger.info(
            f"  Avg neutral markers: {env_df['neutral_marker_count'].mean():.1f}"
        )


if __name__ == "__main__":
    main()
