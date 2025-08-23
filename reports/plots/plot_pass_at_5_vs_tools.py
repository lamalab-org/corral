"""
Plot the number of tool calls against pass@5 scores for single comprehensive tasks.
It creates a scatter plot with the number of tool calls on the x-axis and pass@5 scores on the y-axis.
Each datapoint corresponds to a specific environment, agent type, and model.
The data is loaded from the processed_results.json file.
The figure is saved as pass_at_5_vs_tool_calls.pdf
"""

import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from scipy.constants import golden

# Figure dimensions
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

lama_aesthetics.get_style("main")

# Define model colors
MODEL_COLORS = {
    "Claude-3.5": "#768eab",
    "GPT-4o": "#a285a6",
}

# Load processed results
data_path = Path("processed_results.json")
with data_path.open() as f:
    results = json.load(f)

# Filter for chained=False and tool_verbosity=="workflow"
filtered = [
    r
    for r in results
    if not r.get("chained", False) and r.get("verbosity_level") == "comprehensive"
]

# Group by env, then by agent_type
plot_data = defaultdict(
    lambda: defaultdict(list)
)  # env -> agent_type -> list of (model, pass@5, pass^5, total_tool_calls)
for r in filtered:
    env = r["env"]
    agent_type = r["agent_type"]
    model = r["model"]
    score_at = r["pass@5"]
    score_pow = r["pass^5"]
    tool_calls = r["total_tool_calls"]
    plot_data[env][agent_type].append((model, score_at, score_pow, tool_calls))

# Define domain expertise order (from lower to higher expertise)
domain_order = ["ML", "OpenCatalyst", "MD", "Spectra"]
envs = [env for env in domain_order if env in plot_data]
agent_types = sorted({a for env in plot_data.values() for a in env})

# Prepare data for scatter plot
scatter_data = []

for env in envs:
    for agent_type in agent_types:
        if "react" not in agent_type and "tool" not in agent_type:
            continue  # Skip agent types that don't contain "react" or "tool"

        entries = plot_data[env].get(agent_type, [])
        for model, score_at, score_pow, tool_calls in entries:
            scatter_data.append(
                (env, agent_type, model, score_at, score_pow, tool_calls)
            )

fig, ax = plt.subplots(
    1, 1, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
)

# Separate data by model and agent type for different colors and markers
for env, agent_type, model, score_at, _score_pow, tool_calls in scatter_data:
    # Determine color based on model
    if "claude" in model.lower():
        color = MODEL_COLORS["Claude-3.5"]
    elif "gpt" in model.lower():
        color = MODEL_COLORS["GPT-4o"]
    else:
        color = "gray"

    # Different markers for different agent types
    marker = "D" if "tool" in agent_type.lower() else "o"

    # Plot the point
    ax.scatter(
        tool_calls,
        score_at,
        c=color,
        marker=marker,
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add environment label next to each point
    ax.annotate(
        env,
        (tool_calls, score_at),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
    )

# Add legend
handles = [
    mlines.Line2D(
        [0], [0], color=MODEL_COLORS["Claude-3.5"], lw=4, label="Claude 3.5 Sonnet"
    ),
    mlines.Line2D([0], [0], color=MODEL_COLORS["GPT-4o"], lw=4, label="GPT-4o"),
    mlines.Line2D(
        [0],
        [0],
        color="gray",
        marker="o",
        markersize=8,
        linestyle="None",
        label="React Agent",
    ),
    mlines.Line2D(
        [0],
        [0],
        color="gray",
        marker="D",
        markersize=8,
        linestyle="None",
        label="Tool-Calling Agent",
    ),
]

# Get data ranges for axis formatting
tool_calls_data = [data[5] for data in scatter_data]
scores_data = [data[3] for data in scatter_data]

range_frame(ax, np.array([0, 1400]), np.array(scores_data), pad=0.05)
ax.legend(handles=handles, loc="upper right", fontsize=10)

# Set axis labels and limits
ax.set_xlabel("Total Tool Calls", fontsize=12)
ax.set_ylabel("pass@5 Score", fontsize=12)

# Set tick labels fontsize for both axes
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

fig.tight_layout()
fig.savefig("pass_at_5_vs_tool_calls.pdf", bbox_inches="tight")
