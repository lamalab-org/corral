"""
Plot the total completion tokens for single and chained tasks.
It creates four subplots, one for each environment with the different agents and models.
The data is loaded from the processed_results.json file.
The figure is saved to tools_usage_plot.pdf.
"""

import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from scipy.constants import golden

# Figure dimensions
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

lama_aesthetics.get_style("main")

# Define model colors
MODEL_COLORS = {
    "Claude 3.5 Sonnet": "#768eab",
    "GPT-4o": "#a285a6",
}

# Define agent marker map
AGENT_MARKERS = {
    "ReAct Agent": "o",  # circle
    "Tool-Calling Agent": "D",  # diamond
}

# Environment name mapping
ENVIRONMENT_NAMES = {
    "MD": "MD",
    "Catalyst": "OpenCatalyst",
    "ML": "ML",
    "Spectra": "Spectra",
}

# Load processed results
data_path = Path("processed_results.json")
with data_path.open() as f:
    results = json.load(f)

# Filter for comprehensive verbosity only
filtered = [r for r in results if r.get("verbosity_level") == "comprehensive"]

# Group by env, agent_type, model, and chained status
plot_data = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: {"single": None, "chained": None}))
)  # env -> agent_type -> model -> {single: tokens, chained: tokens}

for r in filtered:
    env = r["env"]
    agent_type = r["agent_type"]
    model = r["model"]
    completion_tokens = r["completion_tokens"]
    is_chained = r.get("chained", False)

    task_type = "chained" if is_chained else "single"
    plot_data[env][agent_type][model][task_type] = completion_tokens

# Define domain expertise order (from lower to higher expertise)
domain_order = ["ML", "OpenCatalyst", "MD", "Spectra"]

# Extract environments from the data, excluding spectra_ablations_env
environments = set()
for env in plot_data:
    if env != "spectra_ablations_env":
        environments.add(env)

environments = sorted(environments)
logger.info(f"Found environments: {environments}")

# Create figure with subplots
n_envs = len(environments)
n_cols = 2
n_rows = (n_envs + 1) // 2  # Ceiling division

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),
    sharex=True,
)

# Handle case where we have only one row
if n_rows == 1:
    axes = axes.reshape(1, -1)
# Handle case where we have only one subplot
elif n_envs == 1:
    axes = np.array([[axes]])

axes = axes.flatten()

# Collect all y-values for consistent y-axis scaling
all_tokens = []

# Process each environment
for env_idx, environment in enumerate(environments):
    ax = axes[env_idx]

    logger.info(f"\nProcessing environment: {environment}")

    # Collect data for this environment
    env_tokens = []

    # Plot for each agent-model combination
    for agent_type in plot_data[environment]:
        for model in plot_data[environment][agent_type]:
            data_point = plot_data[environment][agent_type][model]

            # Check if we have both single and chained data
            if data_point["single"] is None or data_point["chained"] is None:
                logger.warning(
                    f"  Skipping {model}-{agent_type}: missing single/chained data"
                )
                continue

            single_tokens = data_point["single"]
            chained_tokens = data_point["chained"]

            logger.info(f"  {model}-{agent_type}: {single_tokens} -> {chained_tokens}")

            # Store tokens for range calculations
            env_tokens.extend([single_tokens, chained_tokens])
            all_tokens.extend([single_tokens, chained_tokens])

            # Get visual properties
            # Model detection with fallback
            if "claude" in model.lower():
                color = MODEL_COLORS["Claude 3.5 Sonnet"]
            elif "gpt" in model.lower():
                color = MODEL_COLORS["GPT-4o"]
            else:
                color = "#333333"  # Default color

            marker = "D" if "tool" in agent_type.lower() else "o"
            logger.info(f"  Agent: '{agent_type}' -> Marker: '{marker}'")

            # Plot the line connecting single to chained
            ax.plot(
                [0, 1],  # x-positions for single and chained
                [single_tokens, chained_tokens],
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=f"{model} - {agent_type}" if env_idx == 0 else "",
            )

            # Plot markers separately to ensure correct shapes
            ax.scatter(
                0,  # x-position for single
                single_tokens,
                facecolors="white",
                edgecolors=color,
                marker=marker,
                s=40,
                linewidth=1.5,
                zorder=3,
                alpha=0.8,
            )

            # Chained task: filled marker
            ax.scatter(
                1,  # x-position for chained
                chained_tokens,
                color=color,
                marker=marker,
                s=40,
                zorder=3,
                alpha=0.8,
            )

    # Customize subplot
    ax.set_title(ENVIRONMENT_NAMES.get(environment, environment), fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single", "Chained"])
    # Only set ylabel for subplots on the left side (column 0)
    if env_idx % 2 == 0:  # For first column
        ax.set_ylabel("Completion Tokens")
    else:
        ax.set_ylabel("")  # Empty for right column

    # Apply range frame if we have data
    if env_tokens:
        range_frame(ax, np.array([0, 1]), np.array(env_tokens))

# Remove empty subplots
for idx in range(len(environments), len(axes)):
    fig.delaxes(axes[idx])

# Create legends
# Model legend
model_legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=model,
        markerfacecolor=color,
        markersize=8,
        linestyle="-",
        linewidth=2,
        markeredgecolor=color,
    )
    for model, color in MODEL_COLORS.items()
]

# Agent legend
agent_legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="ReAct Agent",
        markerfacecolor="gray",
        markersize=8,
    ),
    plt.Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Tool-Calling Agent",
        markerfacecolor="gray",
        markersize=8,
    ),
]

task_type_legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Single",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=8,
        linestyle="None",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Chained",
        markerfacecolor="black",
        markersize=8,
        linestyle="None",
    ),
]

# Position legends
legend1 = fig.legend(
    handles=model_legend_elements,
    loc="center left",
    bbox_to_anchor=(0.85, 0.7),
)
legend2 = fig.legend(
    handles=agent_legend_elements,
    loc="center left",
    bbox_to_anchor=(0.85, 0.5),
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

# Save figure
fig.savefig("completion_tokens_single_vs_chained.pdf", bbox_inches="tight", dpi=300)
logger.info("\nFigure saved to completion_tokens_single_vs_chained.pdf")
