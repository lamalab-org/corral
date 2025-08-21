import json
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

JSON_PATH = Path(__file__).parent / "processed_results.json"

with JSON_PATH.open("r") as file:
    data = json.load(file)

SPECTRA_ENVS = ["spectra_ablations_env", "Spectra"]

# Define model colors
MODEL_COLORS = {"Claude-3.5": "#768eab", "GPT-4o": "#a285a6"}


def get_model_color(model):
    """Get color for model based on naming variations."""
    if "claude" in model.lower():
        return MODEL_COLORS["Claude-3.5"]
    elif "gpt" in model.lower():
        return MODEL_COLORS["GPT-4o"]
    else:
        return "gray"  # Default color


def get_agent_marker(agent_type):
    """Get marker for agent type based on naming variations."""
    if "tool" in agent_type.lower():
        return "D"  # Square for tool_calling
    else:
        return "o"  # Circle for react


# Define the steps and their order
STEPS = [
    ("Spectra", False),  # step 0
    ("Spectra", True),  # step 1
    ("spectra_ablations_env", None),  # step 2
]

# Prepare data structure: {model: {agent_type: {step: score}}}
results = {}
for entry in data:
    environment = entry.get("env")
    verbosity = entry.get("verbosity_level")
    if environment not in SPECTRA_ENVS or verbosity != "comprehensive":
        continue
    model = entry.get("model")
    model = "gpt_4o" if "gpt" in model.lower() else "claude_35_sonnet"
    agent_type = entry.get("agent_type")
    chained = entry.get("chained") if environment == "Spectra" else None
    score = entry.get("pass@5")
    # Find step index
    for idx, (env, ch) in enumerate(STEPS):
        if environment == env and (ch is None or chained == ch):
            step = idx
            break
    else:
        continue
    results.setdefault(model, {}).setdefault(agent_type, {})[step] = score

# Plotting
fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))

for model, agent_types in results.items():
    for agent_type, step_scores in agent_types.items():
        x = []
        y = []
        filled = []
        for step in range(3):
            x.append(step)
            y.append(step_scores.get(step, np.nan))
            # Only step 1 (chained) is empty, others are filled
            filled.append(step != 1)
        color = get_model_color(model)
        marker = get_agent_marker(agent_type)
        linestyle = "--" if "tool" in agent_type.lower() else "-"
        # Plot lines
        ax.plot(x, y, linestyle=linestyle, color=color, alpha=0.7)
        # Plot markers, filled or not
        for xi, yi, fill in zip(x, y, filled, strict=False):
            if np.isnan(yi):
                continue
            facecolor = color if fill else "none"
            edgecolor = color
            ax.scatter(
                xi,
                yi,
                marker=marker,
                s=36,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=2,
                label=f"{model} {agent_type} step {xi}"
                if xi == 0
                else None,  # avoid duplicate labels
            )

# Custom legend
handles = [
    Line2D([0], [0], color=MODEL_COLORS["Claude-3.5"], lw=4, label="Claude 3.5 Sonnet"),
    Line2D([0], [0], color=MODEL_COLORS["GPT-4o"], lw=4, label="GPT-4o"),
    Line2D(
        [0],
        [0],
        color="gray",
        marker="o",
        markersize=8,
        linestyle="None",
        label="React Agent",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        marker="D",
        markersize=8,
        linestyle="None",
        label="Tool-Calling Agent",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="k",
        label="Single",
        markerfacecolor="k",
        markeredgecolor="k",
        markersize=8,
        linestyle="None",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="k",
        label="Chained",
        markerfacecolor="none",
        markeredgecolor="k",
        markersize=8,
        linestyle="None",
    ),
]
ax.legend(
    handles=handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0,
    fontsize=10,
)
fig.subplots_adjust(right=0.75)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Single", "Chained", "Ablation\n(single)"])
ax.set_ylabel("pass@5", fontsize=12)
ax.tick_params(axis="both", which="major", labelsize=10)
range_frame(ax, np.array([0, 1, 2]), np.array([0, 1]))
plt.tight_layout()
pdf_path = Path(__file__).parent / "ablation_plot.pdf"
plt.savefig(pdf_path, format="pdf")
