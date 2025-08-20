import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

MAPPING = {
    "melting": "Melting",
    "quenching": "Quenching",
    "surface_energy": "Surface Energy",
}

# Read the averaged results
with Path("averaged_results_subtasks.json").open() as f:
    data = json.load(f)

# Group data by name
data_by_name = defaultdict(list)
for entry in data:
    data_by_name[entry["name"]].append(entry)

# Get unique names (should be 6 for 6 subplots)
names = list(data_by_name.keys())
logger.info(f"Found {len(names)} different task names: {names}")

# Create the plot with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Color and marker combinations for different model-agent pairs

# Mapping from data keys to legend display names and styles
model_display = {
    "claude_35": "Claude",
    "gpt_4o": "GPT-4o",
}
agent_display = {
    "react": "React",
    "tool_calling": "Tool Call",
}
model_colors = {
    "claude_35": "#ff6b35",
    "claude35": "#ff6b35",
    "claude": "#ff6b35",
    "claude_35_sonnet": "#ff6b35",
    "gpt_4o": "#28a745",
    "gpt4o": "#28a745",
}
agent_markers = {
    "react": "o",
    "tool_calling": "s",
}
agent_linestyle = {
    "react": "-",
    "tool_calling": ":",  # Changed to dotted line
}

for i, name in enumerate(names):
    if i >= 6:  # Only plot first 6 names
        break

    ax = axes_flat[i]
    entries = data_by_name[name]

    if not entries:
        ax.set_title(f"{name}\n(No data)")
        continue

    # Get subtasks from first entry (they should be consistent)
    subtasks = entries[0]["subtasks"]
    x_positions = range(len(subtasks))

    # Plot each model-agent combination
    for _j, entry in enumerate(entries):
        model = entry["model"]
        agent_type = entry["agent_type"]
        scores = entry["scores"]

        # Map to display names for legend
        model_disp = model_display.get(model, model)
        agent_disp = agent_display.get(agent_type, agent_type)
        label = f"{model_disp} ({agent_disp})"

        # Use fixed color and marker codes
        color = model_colors.get(model, "gray")
        marker = agent_markers.get(agent_type, "o")
        linestyle = agent_linestyle.get(agent_type, "-")  # Use agent-based linestyle

        # Convert None values to NaN for plotting
        scores_plot = [score if score is not None else np.nan for score in scores]

        # Plot the line
        ax.plot(
            x_positions,
            scores_plot,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=label,
            alpha=0.8,
        )

    # Customize the subplot
    # Use MAPPING dict for display name if available
    display_name = MAPPING.get(name, name)
    ax.set_title(f"{display_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Scores", fontsize=10)
    ax.set_ylim(0, 1)

    # Set y-axis ticks to prevent overlap
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8)

    # Set x-axis ticks and labels with improved rotation
    ax.set_xticks(x_positions)
    # Increased rotation angle and reduced font size for better spacing
    range_frame(ax, np.array([0, len(x_positions)]), np.array([0, 1]), pad=0.05)
    ax.set_xticklabels(subtasks, rotation=60, ha="right", fontsize=10)

    # Alternative: Use vertical rotation if labels are very long
    # ax.set_xticklabels(subtasks, rotation=90, ha='center', fontsize=7)

    # No per-subplot legend; will add a global legend later

# Hide any unused subplots
for i in range(len(names), 6):
    axes_flat[i].set_visible(False)


# Adjust layout to prevent overlap with more bottom space for rotated labels
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.15)  # Increased bottom margin for rotated labels


# Create a more compact global legend
fancy_handles = [
    Line2D(
        [0],
        [0],
        color="#ff6b35",
        marker="o",
        markersize=5,
        linestyle="-",
        label="Claude (React)",
    ),
    Line2D(
        [0],
        [0],
        color="#ff6b35",
        marker="s",
        markersize=5,
        linestyle=":",
        label="Claude (Tool Calling)",
    ),
    Line2D(
        [0],
        [0],
        color="#28a745",
        marker="o",
        markersize=5,
        linestyle="-",
        label="GPT-4o (React)",
    ),
    Line2D(
        [0],
        [0],
        color="#28a745",
        marker="s",
        markersize=5,
        linestyle=":",
        label="GPT-4o (Tool Calling)",
    ),
]
fancy_labels = [
    "Claude (React)",
    "Claude (Tool Call)",
    "GPT-4o (React)",
    "GPT-4o (Tool Call)",
]
fig.legend(
    fancy_handles,
    fancy_labels,
    loc="lower center",
    ncol=4,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.1),
)

# Save the plot as PNG and PDF
plt.savefig("subtask_performance_comparison.pdf", dpi=300, bbox_inches="tight")
