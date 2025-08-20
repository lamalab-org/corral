import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

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
)  # env -> agent_type -> list of (model, pass@5, pass^5)
for r in filtered:
    env = r["env"]
    agent_type = r["agent_type"]
    model = r["model"]
    score_at = r["pass@5"]
    score_pow = r["pass^5"]
    plot_data[env][agent_type].append((model, score_at, score_pow))

# Get all envs and agent_types in sorted order for consistent plotting
envs = sorted(plot_data.keys())
agent_types = sorted({a for env in plot_data.values() for a in env})

# Prepare data for horizontal lines and dots (preserving original grouping)
bars = []
group_indices = []
env_centers = []  # Store the center position for each environment group

for env in envs:
    env_start = len(bars)

    for agent_type in agent_types:
        if "react" not in agent_type and "tool" not in agent_type:
            continue  # Skip agent types that don't contain "react" or "tool"

        entries = plot_data[env].get(agent_type, [])
        for model, score_at, score_pow in entries:
            # Only label with agent_type (exclude env and model) - preserving original logic
            label = f"{agent_type}"
            bars.append((label, score_at, score_pow, model))

    # Mark group boundary after each environment and calculate center
    if len(bars) > env_start:
        group_indices.append(len(bars) - 0.5)
        # Calculate center position between 2nd and 3rd bar of this environment group
        # Position between index env_start+1 and env_start+2 (2nd and 3rd bars)
        env_center = env_start + 1.5  # Between 2nd and 3rd bar
        env_centers.append((env_center, env))

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
# Add some space between subplots for the center labels
fig.subplots_adjust(wspace=0.3)
y_pos = np.arange(len(bars))
scores_at = [s[1] for s in bars]  # pass@5 scores
scores_pow = [s[2] for s in bars]  # pass^5 scores

# Clean up labels and determine colors and markers
labels = []
colors = []
markers = []
for label, _score_at, _score_pow, model in bars:
    labels.append(label)
    if "claude" in model.lower():
        colors.append("#ff6b35")  # Orange for Claude
    elif "gpt" in model.lower():
        colors.append("#28a745")  # Green for GPT-4o

    # Different markers for different agent types
    if "tool" in label.lower():
        markers.append("s")  # Square for tool_calling
    else:
        markers.append("o")  # Circle for react

# Create left plot (pass@5) with bars going from right to left
for _i, (y, score, color, marker) in enumerate(
    zip(y_pos, scores_at, colors, markers, strict=False)
):
    ax_left.hlines(
        y,
        score,  # Start from score
        0,  # End at 0 (right to left)
        color=color,
        alpha=0.2,
        linewidth=5,
    )
    ax_left.plot(
        score,
        y,
        marker,
        markersize=5,
        color=color,
        alpha=0.6,
    )

# Create right plot (pass^5) with bars going from left to right
for _i, (y, score, color, marker) in enumerate(
    zip(y_pos, scores_pow, colors, markers, strict=False)
):
    ax_right.hlines(
        y,
        0,
        score,
        color=color,
        alpha=0.2,
        linewidth=5,
    )
    ax_right.plot(
        score,
        y,
        marker,
        markersize=5,
        color=color,
        alpha=0.6,
    )

# Remove y-ticks since we're using centered labels between plots
# ax_left.set_yticks([pos for pos, _ in env_centers])
# ax_left.set_yticklabels([env for _, env in env_centers])

# Add group separators visually (gray lines) on both plots
for idx in group_indices[:-1]:
    ax_left.axhline(idx, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_right.axhline(idx, color="gray", linestyle=":", linewidth=1, alpha=0.5)

# Remove y-axis from both plots and add centered labels between plots
ax_left.tick_params(left=False, labelleft=False)
ax_left.spines["left"].set_visible(False)
ax_right.spines["left"].set_visible(False)

# Add environment labels in the center between the two plots
for pos, env in env_centers:
    # Calculate the y position in data coordinates, then transform to figure coordinates
    # pos is already the center of the group of bars for this environment
    y_data = pos
    # Transform from data coordinates to figure coordinates
    # The y-axis goes from -0.5 to len(bars)-0.5 in data coordinates
    y_fig = (y_data + 0.5) / len(bars)

    # Position the text in the middle between the two subplots
    fig.text(0.5, y_fig, env, ha="center", va="center", fontsize=10)

# Add legend only to the right plot
handles = [
    plt.Line2D([0], [0], color="#ff6b35", lw=4, label="Claude"),
    plt.Line2D([0], [0], color="#28a745", lw=4, label="GPT-4o"),
    plt.Line2D(
        [0],
        [0],
        color="gray",
        marker="o",
        markersize=5,
        linestyle="None",
        label="React Agent",
    ),
    plt.Line2D(
        [0],
        [0],
        color="gray",
        marker="s",
        markersize=5,
        linestyle="None",
        label="Tool Calling Agent",
    ),
]
ax_right.legend(handles=handles, loc="upper right")

# Adjust xlim for both plots
range_frame(ax_left, np.array([0, 1]), y_pos, pad=0.05)
range_frame(ax_right, np.array([0, 1]), y_pos, pad=0.05)

# Invert the x-axis on the left plot so bars go from right to left
ax_left.invert_xaxis()

# Set labels
ax_left.set_xlabel("pass@5 Score")
ax_right.set_xlabel("pass^5 Score")

# Remove the right y-axis ticks since we're sharing the y-axis
ax_right.tick_params(left=False, labelleft=False)

fig.tight_layout()
fig.savefig("pass_at_5_by_env.pdf", bbox_inches="tight")
