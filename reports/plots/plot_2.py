import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from matplotlib.lines import Line2D
from scipy.constants import golden

# Figure dimensions
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

lama_aesthetics.get_style("main")

# Read the task mapping
with Path("tasks.json").open() as f:
    task_mapping = json.load(f)

# Read the averaged results
with Path("averaged_results_subtasks.json").open() as f:
    data = json.load(f)


# Normalize model names to handle different naming conventions
def normalize_model_name(model_name):
    model_lower = model_name.lower()
    if any(
        claude_variant in model_lower
        for claude_variant in ["claude_35", "claude35", "claude_35_sonnet", "claude"]
    ):
        return "claude"
    elif any(gpt_variant in model_lower for gpt_variant in ["gpt_4o", "gpt4o"]):
        return "gpt4o"
    else:
        return model_name


# Group data by normalized model and agent type
data_by_model_agent = defaultdict(lambda: defaultdict(list))
for entry in data:
    model = normalize_model_name(entry["model"])
    agent_type = entry["agent_type"]
    data_by_model_agent[model][agent_type].append(entry)

# Define task categories in the desired order
task_categories = [
    "retrieval",
    "code_execution",
    "experiment_execution",
    "reasoning",
    "validation",
]

# Create a single plot
fig, ax = plt.subplots(
    1, 1, figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
)

# Color and marker combinations for different model-agent pairs
model_display = {
    "claude": "Claude",
    "gpt4o": "GPT-4o",
}
agent_display = {
    "react": "React",
    "tool_calling": "Tool Call",
}
model_colors = {
    "claude": "#768eab",
    "gpt4o": "#a285a6",
}
agent_markers = {
    "react": "o",
    "tool_calling": "D",
}
agent_linestyle = {
    "react": "-",
    "tool_calling": "--",  # Dashed line
}

# Process each model-agent combination
for model in data_by_model_agent:
    for agent_type in data_by_model_agent[model]:
        entries = data_by_model_agent[model][agent_type]

        # Collect scores for each task category
        category_scores = {category: [] for category in task_categories}

        for entry in entries:
            subtasks = entry["subtasks"]
            scores = entry["scores"]

            for subtask, score in zip(subtasks, scores, strict=False):
                if subtask in task_mapping:
                    category = task_mapping[subtask]
                    if score is not None:
                        category_scores[category].append(score)

        # Calculate average scores for each category
        avg_scores = []
        for category in task_categories:
            if category_scores[category]:
                avg_score = np.mean(category_scores[category])
            else:
                avg_score = np.nan
            avg_scores.append(avg_score)

        # Map to display names for legend
        model_disp = model_display.get(model, model)
        agent_disp = agent_display.get(agent_type, agent_type)
        label = f"{model_disp} ({agent_disp})"

        # Use fixed color and marker codes
        color = model_colors.get(model, "gray")
        marker = agent_markers.get(agent_type, "o")
        linestyle = agent_linestyle.get(agent_type, "-")

        # Plot the line
        x_positions = range(len(task_categories))
        ax.plot(
            x_positions,
            avg_scores,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6,
            label=label,
            alpha=0.8,
            fillstyle="none",
        )

# Customize the plot
ax.set_ylabel("Average pass@5", fontsize=12)
ax.set_xlabel("Task Categories", fontsize=12)
ax.set_ylim(0, 1)

# Set y-axis ticks
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="y", labelsize=10)

# Set x-axis ticks and labels
ax.set_xticks(range(len(task_categories)))
ax.set_xticklabels(
    [cat.replace("_", " ").title() for cat in task_categories], fontsize=10
)
range_frame(ax, np.array([0, len(task_categories) - 1]), np.array([0, 1]), pad=0.05)


# Adjust layout
plt.tight_layout()

# Add legend
handles = [
    Line2D([0], [0], color="#768eab", lw=4),
    Line2D([0], [0], color="#a285a6", lw=4),
    Line2D(
        [0],
        [0],
        color="gray",
        marker="o",
        markersize=8,
        linestyle="None",
        fillstyle="none",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        marker="D",
        markersize=8,
        linestyle="None",
        fillstyle="none",
    ),
]
labels = ["Claude 3.5 Sonnet", "GPT-4o", "React Agent", "Tool-Calling Agent"]
ax.legend(
    handles,
    labels,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=10,
)

# Save the plot as PNG and PDF
plt.savefig("task_category_performance_comparison.pdf", dpi=300, bbox_inches="tight")
