"""
Plot the total completion tokens for for single comprehensive tasks.
It creates a bar plot with the total completion tokens for each environment.
The environments are in the y-axis in the order of their domain expertise.
The number of completion tokens are represented for each environment, agent and model.
The data is loaded from the processed_results.json file.
The figure is saved as completion_tokens_by_domain_expertise.pdf
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
)  # env -> agent_type -> list of (model, completion_tokens)
for r in filtered:
    env = r["env"]
    agent_type = r["agent_type"]
    model = r["model"]
    completion_tokens = r["completion_tokens"]
    plot_data[env][agent_type].append((model, completion_tokens))

# Define domain expertise order (from lower to higher expertise)
domain_order = ["ML", "OpenCatalyst", "MD", "Spectra"]
envs = [env for env in domain_order if env in plot_data]
agent_types = sorted({a for env in plot_data.values() for a in env})

# Prepare data for horizontal bars
bars = []

for env in envs:
    for agent_type in agent_types:
        if "react" not in agent_type and "tool" not in agent_type:
            continue  # Skip agent types that don't contain "react" or "tool"

        entries = plot_data[env].get(agent_type, [])
        for model, completion_tokens in entries:
            # Label with agent_type only
            label = f"{agent_type}"
            bars.append((label, completion_tokens, model))

fig, ax = plt.subplots(
    1, 1, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH)
)

y_pos = np.arange(len(bars))
tokens = [s[1] for s in bars]  # completion_tokens

# Clean up labels and determine colors and markers
labels = []
colors = []
markers = []
for label, _tokens, model in bars:
    labels.append(label)
    if "claude" in model.lower():
        colors.append(MODEL_COLORS["Claude-3.5"])
    elif "gpt" in model.lower():
        colors.append(MODEL_COLORS["GPT-4o"])

    # Different markers for different agent types
    if "tool" in label.lower():
        markers.append("D")  # Diamond for tool_calling
    else:
        markers.append("o")  # Circle for react

# Create horizontal lines with markers
for _i, (y, token_count, color, marker) in enumerate(
    zip(y_pos, tokens, colors, markers, strict=False)
):
    # Draw horizontal line from 0 to token_count
    ax.hlines(
        y,
        0,
        token_count,
        color=color,
        alpha=0.5,
        linewidth=5,
    )
    # Add marker at the end of each line
    ax.plot(
        token_count,
        y,
        marker,
        markersize=6,
        color=color,
    )

# Create environment-only y-axis labels
env_lookup = []
for env in envs:
    for agent_type in agent_types:
        if "react" not in agent_type and "tool" not in agent_type:
            continue

        entries = plot_data[env].get(agent_type, [])
        for _model, _ in entries:
            env_lookup.append(env)

# Calculate the center position for each environment's group of bars
env_positions = {}
current_pos = 0
for env in envs:
    # Count how many bars this environment has
    bar_count = 0
    for agent_type in agent_types:
        if "react" not in agent_type and "tool" not in agent_type:
            continue
        entries = plot_data[env].get(agent_type, [])
        bar_count += len(entries)

    # Calculate center position for this environment
    if bar_count > 0:
        env_positions[env] = current_pos + (bar_count - 1) / 2
        current_pos += bar_count

# Set y-axis labels at the center of each environment's group
env_ticks = list(env_positions.values())
env_labels = list(env_positions.keys())
ax.set_yticks(env_ticks)
ax.set_yticklabels(env_labels)

# Add environment group separators
current_env = env_lookup[0] if env_lookup else None
separator_positions = []
for i, env in enumerate(env_lookup[1:], 1):
    if env != current_env:
        separator_positions.append(i - 0.5)
        current_env = env

for pos in separator_positions:
    ax.axhline(pos, color="gray", linestyle=":", linewidth=1, alpha=0.5)

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
range_frame(ax, np.array([0, 250000]), y_pos, pad=0.05)
ax.legend(handles=handles, loc="upper right", fontsize=10)

# Set axis labels and limits
ax.set_xlabel("Completion Tokens", fontsize=12)
ax.set_ylabel("Required Domain Expertise", fontsize=12)

# Set tick labels fontsize for both axes
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

# Don't invert y-axis - keep natural order where ML (lower expertise) is at bottom

fig.tight_layout()
fig.savefig("completion_tokens_by_domain_expertise.pdf", bbox_inches="tight")
