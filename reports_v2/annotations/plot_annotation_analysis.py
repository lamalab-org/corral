"""
Script to create plots from annotation analysis results.
Visualizes metrics across environments and levels.
"""

import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

# Set style using lama_aesthetics
lama_aesthetics.get_style("main")

# Define color palette for levels (dark colors for markers)
LEVEL_COLORS = {
    1: "#2ecc71",  # green
    2: "#3498db",  # blue
    3: "#e74c3c",  # red
    4: "#9b59b6",  # purple
}

# Define lighter color variants for vlines
LEVEL_COLORS_LIGHT = {
    1: "#a8e6cf",  # light green
    2: "#a8d4f0",  # light blue
    3: "#f5b7b1",  # light red
    4: "#d7bde2",  # light purple
}

# Define markers for levels
LEVEL_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
}


def load_analysis_data(filepath: str = "annotation_analysis.json") -> dict:
    """Load the annotation analysis JSON file."""
    with Path(filepath).open() as f:
        return json.load(f)


def extract_env_level_data(data: dict) -> dict:
    """Extract data organized by environment and level."""
    env_level_data = defaultdict(dict)

    for value in data.get("by_environment_level", {}).values():
        env = value["environment"]
        level = value["level"]
        env_level_data[env][level] = value

    return dict(env_level_data)


def plot_agreement_metrics_by_env(data: dict, output_dir: Path):
    """
    Plot overall agreement metrics (kappa, krippendorff alpha) for each environment,
    with different levels shown as grouped bars.
    """
    env_level_data = extract_env_level_data(data)
    environments = sorted(env_level_data.keys())
    all_levels = sorted(
        {level for env_data in env_level_data.values() for level in env_data}
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(environments))
    width = 0.2
    n_levels = len(all_levels)

    for i, level in enumerate(all_levels):
        values = []
        for env in environments:
            if level in env_level_data[env]:
                kappa = env_level_data[env][level]["agreement_metrics"].get(
                    "overall_kappa"
                )
                values.append(kappa if kappa is not None else 0)
            else:
                values.append(0)

        offset = (i - (n_levels - 1) / 2) * width
        # Use vlines + markers instead of bars
        for j, val in enumerate(values):
            ax.vlines(
                x[j] + offset,
                0,
                val,
                color=LEVEL_COLORS_LIGHT.get(level, "#d5dbdb"),
                linewidth=5,
                alpha=0.7,
            )
            ax.plot(
                x[j] + offset,
                val,
                "o",
                color=LEVEL_COLORS.get(level, "#95a5a6"),
                markersize=6,
            )

    # Create legend manually
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=LEVEL_COLORS_LIGHT[level],
            linewidth=5,
            alpha=0.7,
            marker="o",
            markerfacecolor=LEVEL_COLORS[level],
            markersize=6,
            label=f"Level {level}",
        )
        for level in all_levels
        if level in LEVEL_COLORS
    ]

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([env.upper() for env in environments], rotation=45, ha="right")
    ax.legend(handles=legend_elements, title="Difficulty Level", loc="upper right")
    ax.axhline(
        y=0.4, color="gray", linestyle="--", alpha=0.5, label="Moderate agreement"
    )
    ax.axhline(
        y=0.6, color="gray", linestyle=":", alpha=0.5, label="Substantial agreement"
    )
    ax.set_ylim(0, 1)
    ax.grid(False)

    # Apply range_frame for cleaner borders
    range_frame(ax, np.array([x.min() - width, x.max() + width]), np.array([0, 1]))

    plt.tight_layout()
    plt.savefig(output_dir / "agreement_kappa_by_env_level.pdf", bbox_inches="tight")
    plt.close()


def plot_label_distribution_by_env(data: dict, output_dir: Path):
    """
    Plot label distribution (normalized) for each environment,
    comparing across levels. Saves to environment-specific subdirectories.
    """
    env_level_data = extract_env_level_data(data)

    # Key labels to focus on (most informative)
    key_labels = [
        "validation_attempt",
        "planning_statement",
        "reasoning_statement",
        "correct_submission",
        "loop_instance",
        "hallucination",
        "syntax_error",
        "inefficient_tool_call",
        "give_up",
        "neutral",
    ]

    for env in sorted(env_level_data.keys()):
        levels_data = env_level_data[env]
        if not levels_data:
            continue

        # Create environment-specific directory
        env_dir = output_dir / env
        env_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        levels = sorted(levels_data.keys())
        x = np.arange(len(key_labels))
        width = 0.8 / len(levels)

        for i, level in enumerate(levels):
            label_counts = levels_data[level].get("label_counts", {})
            total = sum(label_counts.values())

            if total == 0:
                continue

            # Normalize by total annotations
            values = [label_counts.get(label, 0) / total * 100 for label in key_labels]

            offset = (i - (len(levels) - 1) / 2) * width
            # Use vlines + markers instead of bars
            for j, val in enumerate(values):
                ax.vlines(
                    x[j] + offset,
                    0,
                    val,
                    color=LEVEL_COLORS_LIGHT.get(level, "#d5dbdb"),
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    x[j] + offset,
                    val,
                    "o",
                    color=LEVEL_COLORS.get(level, "#95a5a6"),
                    markersize=6,
                )

        # Create legend manually
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=LEVEL_COLORS_LIGHT[level],
                linewidth=5,
                alpha=0.7,
                marker="o",
                markerfacecolor=LEVEL_COLORS[level],
                markersize=6,
                label=f"Level {level}",
            )
            for level in levels
            if level in LEVEL_COLORS
        ]

        ax.set_xlabel("Label", fontsize=12)
        ax.set_ylabel("Percentage of Annotations (%)", fontsize=12)
        ax.set_xticks(x)
        ax.grid(False)
        ax.set_xticklabels(
            [label.replace("_", "\n") for label in key_labels],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.legend(handles=legend_elements, title="Difficulty Level", loc="upper right")

        # Apply range_frame for cleaner borders
        max_val = ax.get_ylim()[1]
        range_frame(
            ax, np.array([x.min() - width, x.max() + width]), np.array([0, max_val])
        )

        plt.tight_layout()
        plt.savefig(env_dir / "label_distribution.pdf", bbox_inches="tight")
        plt.close()


def plot_error_types_comparison(data: dict, output_dir: Path):
    """
    Plot comparison of error-related labels across environments and levels.
    """
    env_level_data = extract_env_level_data(data)

    error_labels = [
        "hallucination",
        "syntax_error",
        "wrong_planning",
        "wrong_reasoning",
        "loop_instance",
        "give_up",
        "early_final_answer",
        "inefficient_tool_call",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, label in enumerate(error_labels):
        ax = axes[idx]

        env_data = defaultdict(dict)
        for env, levels_data in env_level_data.items():
            for level, ldata in levels_data.items():
                count = ldata.get("label_counts", {}).get(label, 0)
                total = sum(ldata.get("label_counts", {}).values())
                rate = (count / total * 100) if total > 0 else 0
                env_data[env][level] = rate

        environments = sorted(env_data.keys())
        all_levels = sorted({level for ed in env_data.values() for level in ed})

        x = np.arange(len(environments))
        width = 0.2

        for i, level in enumerate(all_levels):
            values = [env_data[env].get(level, 0) for env in environments]
            offset = (i - (len(all_levels) - 1) / 2) * width
            # Use vlines + markers instead of bars
            for j, val in enumerate(values):
                ax.vlines(
                    x[j] + offset,
                    0,
                    val,
                    color=LEVEL_COLORS_LIGHT.get(level, "#d5dbdb"),
                    linewidth=4,
                    alpha=0.7,
                )
                ax.plot(
                    x[j] + offset,
                    val,
                    "o",
                    color=LEVEL_COLORS.get(level, "#95a5a6"),
                    markersize=5,
                )

        ax.set_title(label.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([e[:4].upper() for e in environments], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

        ax.grid(False)
        if idx == 0:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color=LEVEL_COLORS_LIGHT[level],
                    linewidth=4,
                    alpha=0.7,
                    marker="o",
                    markerfacecolor=LEVEL_COLORS[level],
                    markersize=5,
                    label=f"L{level}",
                )
                for level in all_levels
                if level in LEVEL_COLORS
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

        # Apply range_frame for cleaner borders
        max_val = ax.get_ylim()[1]
        if max_val > 0:
            range_frame(
                ax, np.array([x.min() - width, x.max() + width]), np.array([0, max_val])
            )

    plt.tight_layout()
    plt.savefig(output_dir / "error_types_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_per_label_kappa_heatmap(data: dict, output_dir: Path):
    """
    Create a heatmap of per-label kappa values across environments.
    """
    env_level_data = extract_env_level_data(data)

    # Get all labels
    all_labels = sorted(
        {
            label
            for env_data in env_level_data.values()
            for level_data in env_data.values()
            for label in level_data.get("agreement_metrics", {}).get(
                "per_label_kappa", {}
            )
        }
    )

    # Create combined env_level keys
    env_level_keys = [
        f"{env}_L{level}"
        for env in sorted(env_level_data.keys())
        for level in sorted(env_level_data[env].keys())
    ]

    # Build matrix
    matrix = np.zeros((len(all_labels), len(env_level_keys)))
    matrix[:] = np.nan

    for j, key in enumerate(env_level_keys):
        env = key.rsplit("_L", 1)[0]
        level = int(key.rsplit("_L", 1)[1])

        kappa_dict = (
            env_level_data[env][level]
            .get("agreement_metrics", {})
            .get("per_label_kappa", {})
        )
        for i, label in enumerate(all_labels):
            val = kappa_dict.get(label)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create masked array for NaN values
    masked_matrix = np.ma.masked_invalid(matrix)

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked_matrix, cmap=cmap, aspect="auto", vmin=-0.2, vmax=1.0)

    ax.set_xticks(np.arange(len(env_level_keys)))
    ax.set_yticks(np.arange(len(all_labels)))
    ax.set_xticklabels(env_level_keys, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([label.replace("_", " ") for label in all_labels], fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's Kappa", fontsize=11)

    ax.set_xlabel("Environment - Level", fontsize=12)
    ax.set_ylabel("Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "per_label_kappa_heatmap.pdf", bbox_inches="tight")
    plt.close()


def plot_annotation_counts(data: dict, output_dir: Path):
    """
    Plot number of annotations per environment and level.
    """
    env_level_data = extract_env_level_data(data)
    environments = sorted(env_level_data.keys())
    all_levels = sorted(
        {level for env_data in env_level_data.values() for level in env_data}
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(environments))
    width = 0.2

    for i, level in enumerate(all_levels):
        values = []
        for env in environments:
            if level in env_level_data[env]:
                count = env_level_data[env][level].get("n_annotations", 0)
                values.append(count)
            else:
                values.append(0)

        offset = (i - (len(all_levels) - 1) / 2) * width
        # Use vlines + markers instead of bars
        for j, val in enumerate(values):
            ax.vlines(
                x[j] + offset,
                0,
                val,
                color=LEVEL_COLORS_LIGHT.get(level, "#d5dbdb"),
                linewidth=5,
                alpha=0.7,
            )
            ax.plot(
                x[j] + offset,
                val,
                "o",
                color=LEVEL_COLORS.get(level, "#95a5a6"),
                markersize=6,
            )

    # Create legend manually
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=LEVEL_COLORS_LIGHT[level],
            linewidth=5,
            alpha=0.7,
            marker="o",
            markerfacecolor=LEVEL_COLORS[level],
            markersize=6,
            label=f"Level {level}",
        )
        for level in all_levels
        if level in LEVEL_COLORS
    ]

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Number of Annotations", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([env.upper() for env in environments], rotation=45, ha="right")
    ax.legend(handles=legend_elements, title="Difficulty Level", loc="upper right")
    ax.grid(False)

    # Apply range_frame for cleaner borders
    max_val = ax.get_ylim()[1]
    range_frame(
        ax, np.array([x.min() - width, x.max() + width]), np.array([0, max_val])
    )

    plt.tight_layout()
    plt.savefig(output_dir / "annotation_counts.pdf", bbox_inches="tight")
    plt.close()


def plot_positive_vs_negative_labels(data: dict, output_dir: Path):
    """
    Plot ratio of positive (good behavior) vs negative (error) labels.
    """
    env_level_data = extract_env_level_data(data)

    positive_labels = [
        "validation_attempt",
        "planning_statement",
        "reasoning_statement",
        "correct_submission",
        "backtrack_trigger",
    ]
    negative_labels = [
        "hallucination",
        "syntax_error",
        "wrong_planning",
        "wrong_reasoning",
        "loop_instance",
        "give_up",
        "early_final_answer",
        "inefficient_tool_call",
        "missing_validation",
        "unnecessary_tool_use",
        "non_sense",
        "misunderstood_tool",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute rates for each env-level
    env_level_keys = []
    positive_rates = []
    negative_rates = []
    colors = []

    for env in sorted(env_level_data.keys()):
        for level in sorted(env_level_data[env].keys()):
            label_counts = env_level_data[env][level].get("label_counts", {})
            total = sum(label_counts.values())

            if total == 0:
                continue

            pos_count = sum(label_counts.get(label, 0) for label in positive_labels)
            neg_count = sum(label_counts.get(label, 0) for label in negative_labels)

            env_level_keys.append(f"{env[:4].upper()}\nL{level}")
            positive_rates.append(pos_count / total * 100)
            negative_rates.append(neg_count / total * 100)
            colors.append(LEVEL_COLORS.get(level, "#95a5a6"))

    x = np.arange(len(env_level_keys))

    # Positive labels plot
    for j, (val, color) in enumerate(zip(positive_rates, colors, strict=False)):
        ax = axes[0]
        ax.vlines(x[j], 0, val, color=color, linewidth=5, alpha=0.5)
        ax.plot(x[j], val, "o", color=color, markersize=6)
    axes[0].set_xlabel("Environment - Level", fontsize=11)
    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    axes[0].set_title("Positive Labels Rate", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(env_level_keys, fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(False)

    # Apply range_frame for positive plot
    max_val_pos = max(positive_rates) if positive_rates else 1
    range_frame(
        axes[0],
        np.array([x.min() - 0.5, x.max() + 0.5]),
        np.array([0, max_val_pos * 1.1]),
    )

    # Negative labels plot
    for j, (val, color) in enumerate(zip(negative_rates, colors, strict=False)):
        ax = axes[1]
        ax.vlines(x[j], 0, val, color=color, linewidth=5, alpha=0.5)
        ax.plot(x[j], val, "o", color=color, markersize=6)
    axes[1].set_xlabel("Environment - Level", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_title("Error Labels Rate", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(env_level_keys, fontsize=8)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(False)

    # Apply range_frame for negative plot
    max_val_neg = max(negative_rates) if negative_rates else 1
    range_frame(
        axes[1],
        np.array([x.min() - 0.5, x.max() + 0.5]),
        np.array([0, max_val_neg * 1.1]),
    )

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=LEVEL_COLORS[level],
            linewidth=5,
            alpha=0.5,
            marker="o",
            markerfacecolor=LEVEL_COLORS[level],
            markersize=6,
            label=f"Level {level}",
        )
        for level in sorted(LEVEL_COLORS.keys())
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_negative_labels.pdf", bbox_inches="tight")
    plt.close()


def plot_success_indicators(data: dict, output_dir: Path):
    """
    Plot success-related metrics (correct_submission rate, loop rate, etc.) by env and level.
    """
    env_level_data = extract_env_level_data(data)

    metrics = {
        "correct_submission": "Correct Submission Rate",
        "loop_instance": "Loop Instance Rate",
        "give_up": "Give Up Rate",
        "iteration_limit": "Iteration Limit Rate",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (label, title) in enumerate(metrics.items()):
        ax = axes[idx]

        environments = sorted(env_level_data.keys())
        all_levels = sorted({lvl for ed in env_level_data.values() for lvl in ed})

        x = np.arange(len(environments))
        width = 0.2

        for i, level in enumerate(all_levels):
            values = []
            for env in environments:
                if level in env_level_data[env]:
                    label_counts = env_level_data[env][level].get("label_counts", {})
                    total = sum(label_counts.values())
                    rate = (
                        (label_counts.get(label, 0) / total * 100) if total > 0 else 0
                    )
                    values.append(rate)
                else:
                    values.append(0)

            offset = (i - (len(all_levels) - 1) / 2) * width
            # Use vlines + markers instead of bars
            for j, val in enumerate(values):
                ax.vlines(
                    x[j] + offset,
                    0,
                    val,
                    color=LEVEL_COLORS_LIGHT.get(level, "#d5dbdb"),
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    x[j] + offset,
                    val,
                    "o",
                    color=LEVEL_COLORS.get(level, "#95a5a6"),
                    markersize=6,
                )

        ax.set_xlabel("Environment", fontsize=11)
        ax.set_ylabel("Rate (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([e.upper() for e in environments], rotation=45, ha="right")

        ax.grid(False)
        if idx == 0:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color=LEVEL_COLORS_LIGHT[level],
                    linewidth=5,
                    alpha=0.7,
                    marker="o",
                    markerfacecolor=LEVEL_COLORS[level],
                    markersize=6,
                    label=f"Level {level}",
                )
                for level in all_levels
                if level in LEVEL_COLORS
            ]
            ax.legend(
                handles=legend_elements, title="Level", loc="upper right", fontsize=9
            )

        # Apply range_frame for cleaner borders
        max_val = ax.get_ylim()[1]
        if max_val > 0:
            range_frame(
                ax, np.array([x.min() - width, x.max() + width]), np.array([0, max_val])
            )

    plt.tight_layout()
    plt.savefig(output_dir / "success_indicators.pdf", bbox_inches="tight")
    plt.close()


def plot_cooccurrence_heatmap(data: dict, output_dir: Path):
    """
    Create heatmap of label co-occurrence from overall data.
    """
    overall = data.get("overall", {})
    cooccurrence = overall.get("cooccurrence_top_pairs", [])

    if not cooccurrence:
        return

    # Get unique labels
    labels = sorted(
        {label for pair in cooccurrence for label in [pair["label_1"], pair["label_2"]]}
    )

    # Build matrix
    matrix = np.zeros((len(labels), len(labels)))
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for pair in cooccurrence:
        i = label_to_idx[pair["label_1"]]
        j = label_to_idx[pair["label_2"]]
        matrix[i, j] = pair["count"]
        matrix[j, i] = pair["count"]  # Symmetric

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(
        [label.replace("_", "\n") for label in labels],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels([label.replace("_", " ") for label in labels], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Co-occurrence Count", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "cooccurrence_heatmap.pdf", bbox_inches="tight")
    plt.close()


def plot_per_label_metrics_by_env(data: dict, output_dir: Path):
    """
    Create bar plots for each metric (kappa, percent_agreement, krippendorff_alpha)
    for each environment. Each environment gets its own directory, and each plot
    has subplots for each level showing the metric values for all labels.
    """
    env_level_data = extract_env_level_data(data)

    # Metrics to plot
    metrics = {
        "per_label_kappa": {
            "name": "Cohen's Kappa",
            "filename": "per_label_kappa",
            "ylim": (-0.2, 1.0),
            "reference_lines": [
                (0.4, "Moderate"),
                (0.6, "Substantial"),
                (0.8, "Almost perfect"),
            ],
        },
        "percent_agreement": {
            "name": "Percent Agreement",
            "filename": "percent_agreement",
            "ylim": (0, 1.05),
            "reference_lines": [(0.8, "80%"), (0.9, "90%")],
        },
        "krippendorff_alpha": {
            "name": "Krippendorff's Alpha",
            "filename": "krippendorff_alpha",
            "ylim": (-0.2, 1.0),
            "reference_lines": [(0.667, "Tentative"), (0.8, "Reliable")],
        },
    }

    # Get all possible labels from the data
    all_labels = sorted(
        {
            label
            for env_data in env_level_data.values()
            for level_data in env_data.values()
            for label in level_data.get("agreement_metrics", {}).get(
                "per_label_kappa", {}
            )
        }
    )

    for env in sorted(env_level_data.keys()):
        # Create environment-specific directory
        env_dir = output_dir / env
        env_dir.mkdir(exist_ok=True)

        levels_data = env_level_data[env]
        levels = sorted(levels_data.keys())
        n_levels = len(levels)

        if n_levels == 0:
            continue

        for metric_key, metric_info in metrics.items():
            # Determine subplot layout
            if n_levels == 1:
                fig, axes = plt.subplots(1, 1, figsize=(14, 6))
                axes = [axes]
            elif n_levels == 2:
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            elif n_levels <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                axes = axes.flatten()
            else:
                n_cols = 3
                n_rows = (n_levels + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
                axes = axes.flatten()

            for idx, level in enumerate(levels):
                ax = axes[idx]

                # Get metric values for this level
                agreement_metrics = levels_data[level].get("agreement_metrics", {})
                metric_values = agreement_metrics.get(metric_key, {})

                # Filter to labels that have values for this level
                labels_with_data = [
                    label
                    for label in all_labels
                    if label in metric_values and metric_values[label] is not None
                ]
                values = [metric_values.get(label) for label in labels_with_data]

                if not labels_with_data:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_title(f"Level {level}", fontsize=12)
                    continue

                # Create colors based on value (for markers)
                colors = []
                colors_light = []
                for v in values:
                    if v is None:
                        colors.append("#95a5a6")
                        colors_light.append("#d5dbdb")
                    elif v < 0:
                        colors.append("#c0392b")  # dark red for negative
                        colors_light.append("#f5b7b1")
                    elif v < 0.4:
                        colors.append("#d35400")  # dark orange for low
                        colors_light.append("#f9e79f")
                    elif v < 0.6:
                        colors.append("#f39c12")  # orange/yellow for moderate
                        colors_light.append("#fcf3cf")
                    elif v < 0.8:
                        colors.append("#27ae60")  # green for substantial
                        colors_light.append("#a8e6cf")
                    else:
                        colors.append("#1e8449")  # dark green for excellent
                        colors_light.append("#82e0aa")

                x = np.arange(len(labels_with_data))
                # Use vlines + markers instead of bars
                for j, (val, color, color_light) in enumerate(
                    zip(values, colors, colors_light, strict=False)
                ):
                    ax.vlines(x[j], 0, val, color=color_light, linewidth=5, alpha=0.7)
                    ax.plot(x[j], val, "o", color=color, markersize=6)

                # Add reference lines
                for ref_val, ref_name in metric_info["reference_lines"]:
                    ax.axhline(
                        y=ref_val, color="gray", linestyle="--", alpha=0.5, linewidth=1
                    )
                    ax.text(
                        len(labels_with_data) - 0.5,
                        ref_val + 0.02,
                        ref_name,
                        fontsize=8,
                        color="gray",
                        ha="right",
                    )

                ax.set_xlabel("Label", fontsize=10)
                ax.set_ylabel(metric_info["name"], fontsize=10)
                ax.set_title(
                    f'Level {level} (n={levels_data[level].get("n_multi_annotator", "?")} files)',
                    fontsize=12,
                )
                ax.set_xticks(x)
                ax.set_ylim(metric_info["ylim"])
                ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
                ax.grid(False)

                # Apply range_frame for cleaner borders
                range_frame(
                    ax,
                    np.array([x.min() - 0.5, x.max() + 0.5]),
                    np.array(metric_info["ylim"]),
                )

                # Set xticklabels after range_frame with vertical rotation
                ax.set_xticklabels(
                    [label.replace("_", " ") for label in labels_with_data],
                    rotation=90,
                    ha="center",
                    fontsize=8,
                )

            # Hide unused subplots
            for idx in range(n_levels, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                env_dir / f'{metric_info["filename"]}_by_label.pdf', bbox_inches="tight"
            )
            plt.close()

        logger.info("    - Saved plots for %s environment", env.upper())


def plot_jaccard_similarity_by_env(data: dict, output_dir: Path):
    """
    Create bar plots showing Jaccard similarity top pairs for each environment.
    Each environment gets a plot with subplots for each level showing the top
    label pairs by Jaccard similarity.
    """
    env_level_data = extract_env_level_data(data)

    for env in sorted(env_level_data.keys()):
        # Create environment-specific directory
        env_dir = output_dir / env
        env_dir.mkdir(exist_ok=True)

        levels_data = env_level_data[env]
        levels = sorted(levels_data.keys())
        n_levels = len(levels)

        if n_levels == 0:
            continue

        # Determine subplot layout
        if n_levels == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            axes = [axes]
        elif n_levels == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_levels <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            n_cols = 3
            n_rows = (n_levels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            axes = axes.flatten()

        for idx, level in enumerate(levels):
            ax = axes[idx]

            # Get jaccard similarity pairs for this level
            jaccard_pairs = levels_data[level].get("jaccard_similarity_top_pairs", [])

            if not jaccard_pairs:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(f"Level {level}", fontsize=12)
                continue

            # Take top 10 pairs (or all if less than 10)
            top_pairs = jaccard_pairs[:10]

            # Create labels for x-axis (pair names)
            pair_labels = [
                f"{p['label_1'].replace('_', ' ')}\n&\n{p['label_2'].replace('_', ' ')}"
                for p in top_pairs
            ]
            similarities = [p["similarity"] for p in top_pairs]

            # Color based on similarity value (dark for markers, light for vlines)
            colors = []
            colors_light = []
            for s in similarities:
                if s >= 0.3:
                    colors.append("#1e8449")  # dark green for high
                    colors_light.append("#82e0aa")
                elif s >= 0.15:
                    colors.append("#27ae60")  # green for moderate-high
                    colors_light.append("#a8e6cf")
                elif s >= 0.05:
                    colors.append("#f39c12")  # orange/yellow for moderate
                    colors_light.append("#fcf3cf")
                else:
                    colors.append("#d35400")  # dark orange for low
                    colors_light.append("#f9e79f")

            x = np.arange(len(pair_labels))
            # Use vlines + markers instead of bars
            for j, (val, color, color_light) in enumerate(
                zip(similarities, colors, colors_light, strict=False)
            ):
                ax.vlines(x[j], 0, val, color=color_light, linewidth=5, alpha=0.7)
                ax.plot(x[j], val, "o", color=color, markersize=6)

            # Add value labels above markers
            for j, val in enumerate(similarities):
                ax.text(
                    x[j], val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8
                )

            ax.set_xlabel("Label Pair", fontsize=10)
            ax.set_ylabel("Jaccard Similarity", fontsize=10)
            ax.set_title(
                f'Level {level} (n={levels_data[level].get("n_multi_annotator", "?")} files)',
                fontsize=12,
            )
            ax.set_xticks(x)
            ax.set_ylim(0, max(similarities) * 1.2 if similarities else 1)
            ax.grid(False)

            # Apply range_frame for cleaner borders
            max_val = max(similarities) * 1.2 if similarities else 1
            range_frame(
                ax, np.array([x.min() - 0.5, x.max() + 0.5]), np.array([0, max_val])
            )

            # Set xticklabels after range_frame with vertical rotation
            ax.set_xticklabels(pair_labels, rotation=90, ha="center", fontsize=7)

        # Hide unused subplots
        for idx in range(n_levels, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(env_dir / "jaccard_similarity.pdf", bbox_inches="tight")
        plt.close()


def main():
    """Main function to generate all plots."""
    # Setup paths
    script_dir = Path(__file__).parent
    data_file = script_dir / "analysis_results" / "annotation_analysis.json"
    output_dir = script_dir / "analysis_results" / "plots"
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading data from %s...", data_file)
    data = load_analysis_data(data_file)

    logger.info("Generating plots in %s...", output_dir)

    # Generate all plots
    logger.info("  - Agreement metrics by environment and level...")
    plot_agreement_metrics_by_env(data, output_dir)

    logger.info("  - Label distribution by environment...")
    plot_label_distribution_by_env(data, output_dir)

    logger.info("  - Error types comparison...")
    plot_error_types_comparison(data, output_dir)

    logger.info("  - Per-label kappa heatmap...")
    plot_per_label_kappa_heatmap(data, output_dir)

    logger.info("  - Annotation counts...")
    plot_annotation_counts(data, output_dir)

    logger.info("  - Positive vs negative labels...")
    plot_positive_vs_negative_labels(data, output_dir)

    logger.info("  - Success indicators...")
    plot_success_indicators(data, output_dir)

    logger.info("  - Co-occurrence heatmap...")
    plot_cooccurrence_heatmap(data, output_dir)

    logger.info("  - Per-label metrics by environment (separate directories)...")
    plot_per_label_metrics_by_env(data, output_dir)

    logger.info("  - Jaccard similarity by environment...")
    plot_jaccard_similarity_by_env(data, output_dir)

    logger.info("\nDone! All plots saved to %s", output_dir)
    logger.info("Generated plots:")
    for f in sorted(output_dir.glob("*.pdf")):
        logger.info("  - %s", f.name)


if __name__ == "__main__":
    main()
