import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

CLAUDE_PATH_REACT = Path("../claude/spectra_chained")
CLAUDE_PATH_TOOL_CALLING = Path("../claude/spectra_chained/tool_calling")
GPT_PATH_REACT = Path("../gpt4o/spectra_chained")
GPT_PATH_TOOL_CALLING = Path("../gpt4o/spectra_chained/tool_calling")

PATHS = [
    CLAUDE_PATH_REACT,
    CLAUDE_PATH_TOOL_CALLING,
    GPT_PATH_REACT,
    GPT_PATH_TOOL_CALLING,
]


def extract_info_from_filename(filename):
    # Handles both gpt4o-tool_calling-spectra_chained_env-brief_verbosity.json
    # and claude_35_sonnet-react-spectra_chained_env-brief_verbosity.json
    name = filename.stem
    if name.startswith("claude_35_sonnet-"):
        # claude_35_sonnet-react-spectra_chained_env-brief_verbosity.json
        parts = name.split("-")
        model = "claude_35_sonnet"
        agent = parts[1]
        desc = parts[-1]
    else:
        # gpt4o-tool_calling-spectra_chained_env-brief_verbosity.json
        parts = name.split("-")
        model = parts[0]
        agent = (
            parts[1]
            if parts[1] in ["react", "tool_calling"]
            else ("tool_calling" if "tool_calling" in parts else "react")
        )
        desc = parts[-1]
    return model, agent, desc


# Collect scores
results = defaultdict(dict)  # {(model, agent): {desc: score}}
desc_levels = set()
for path in PATHS:
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    for file in path.glob("*.json"):
        with file.open() as f:
            data = json.load(f)
        model, agent, desc = extract_info_from_filename(file)
        score = data["metrics"]["total_token_usage"]["total_tokens"]
        results[(model, agent)][desc] = score
        desc_levels.add(desc)

# Sort for consistent plotting
desc_levels = sorted(desc_levels)

# Add all possible model-agent pairs present in results, in desired order
groups = []
if any(k[0] == "claude_35_sonnet" and k[1] == "react" for k in results):
    groups.append(("claude_35_sonnet", "react"))
if any(k[0] == "claude_35_sonnet" and k[1] == "tool_calling" for k in results):
    groups.append(("claude_35_sonnet", "tool_calling"))
if any(k[0] == "gpt4o" and k[1] == "react" for k in results):
    groups.append(("gpt4o", "react"))
if any(k[0] == "gpt4o" and k[1] == "tool_calling" for k in results):
    groups.append(("gpt4o", "tool_calling"))

# Prepare data for plotting
model_scores = []
for model, agent in groups:
    for desc in desc_levels:
        label = f"{model}-{agent}-{desc}"
        score = results.get((model, agent), {}).get(desc, np.nan)
        model_scores.append((label, score))


def plot_performance(model_scores, outname):
    # Group bars by model-agent, add gray line after each group
    group_names = []
    group_indices = []
    bars = []
    for group in groups:
        group_label_prefix = f"{group[0]}-{group[1]}"
        group_bars = [
            (label, score)
            for label, score in model_scores
            if label.startswith(group_label_prefix)
        ]
        if group_bars:
            group_names.append(group_label_prefix)
            group_indices.append(len(bars) + len(group_bars) - 0.5)
            bars.extend(group_bars)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(bars))
    scores = [s[1] for s in bars]

    # Clean up labels and determine colors
    labels = []
    colors = []
    for label, _score in bars:
        # Remove "_verbosity" suffix and model name prefix
        clean_label = label.replace("_verbosity", "")
        if clean_label.startswith("claude_35_sonnet-"):
            clean_label = clean_label.replace("claude_35_sonnet-", "")
            colors.append("#ff6b35")  # Orange color for Claude
        elif clean_label.startswith("gpt4o-"):
            clean_label = clean_label.replace("gpt4o-", "")
            colors.append("#28a745")  # Green color for GPT-4o
        else:
            colors.append("#007acc")  # Default blue
        labels.append(clean_label)

    # Create colored horizontal lines and points
    for _i, (y, score, color) in enumerate(zip(y_pos, scores, colors, strict=False)):
        ax.hlines(
            y,
            0,
            score,
            color=color,
            alpha=0.2,
            linewidth=5,
        )
        ax.plot(
            score,
            y,
            "o",
            markersize=5,
            color=color,
            alpha=0.6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Add group separators visually (gray lines)
    for idx in group_indices[:-1]:
        ax.axhline(idx, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Adjust xlim for NaNs
    valid_scores = [s for s in scores if not np.isnan(s)]
    xlim = [0, max(valid_scores) * 1.05] if valid_scores else [0, 1]
    range_frame(ax, np.array(xlim), y_pos, pad=0.1)

    ax.set_xlabel("number of tokens")
    fig.tight_layout()
    fig.savefig(outname, bbox_inches="tight")


plot_performance(model_scores, "number_tokens.pdf")
