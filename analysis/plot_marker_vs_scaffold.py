"""Quantify how marker frequencies vary by model, scaffold, and sentiment.

This analysis script loads the curated annotation dataset, computes per-trace
marker statistics, compares effect sizes for model and scaffold choices, and
summarises raw sentiment counts across model/scaffold combinations. The
resulting figures and summary tables are written to the analysis results
directory for downstream interpretation.
"""

import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "filtered_annotations.json"

CATEGORY_COLORS = {
    "reasoning": "#2196F3",  # blue
    "planning": "#9C27B0",  # purple
    "error": "#F44336",  # red
    "process": "#4CAF50",  # green
    "outcome": "#FF9800",  # orange
}

MARKER_CATEGORY = {
    "reasoning_statement": "reasoning",
    "wrong_reasoning": "reasoning",
    "backtrack_trigger": "reasoning",
    "validation_attempt": "reasoning",
    "planning_statement": "planning",
    "wrong_planning": "planning",
    "todo_list": "planning",
    "hallucination": "error",
    "non_sense": "error",
    "misunderstood_tool": "error",
    "syntax_error": "error",
    "missing_validation": "error",
    "loop_instance": "process",
    "inefficient_tool_call": "process",
    "unnecessary_tool_use": "process",
    "neutral": "process",
    "correct_submission": "outcome",
    "early_final_answer": "outcome",
    "give_up": "outcome",
    "iteration_limit": "outcome",
}

SENTIMENT_COLORS = {
    "positive": "#3C77B1",  # green
    "negative": "#C62828",  # red
    "neutral": "#7A7A7A",  # grey
}

SENTIMENT_ORDER = ["positive", "negative", "neutral"]

MODEL_DISPLAY = {
    "claude_sonnet_45": "Claude-4.5-Sonnet",
    "gpt-4o": "GPT-4o",
}

SCAFFOLD_DISPLAY = {
    "react": "ReAct",
    "tool_calling": "Tool calling",
}

MARKER_SENTIMENT = {
    "validation_attempt": "positive",
    "backtrack_trigger": "positive",
    "planning_statement": "positive",
    "reasoning_statement": "positive",
    "correct_submission": "positive",
    "todo_list": "positive",
    "neutral": "neutral",
    "iteration_limit": "negative",
    "missing_validation": "negative",
    "unnecessary_tool_use": "negative",
    "non_sense": "negative",
    "loop_instance": "negative",
    "hallucination": "negative",
    "wrong_planning": "negative",
    "wrong_reasoning": "negative",
    "syntax_error": "negative",
    "early_final_answer": "negative",
    "give_up": "negative",
    "inefficient_tool_call": "negative",
    "misunderstood_tool": "negative",
}

ANNOTATED_MARKERS = {
    "reasoning_statement",
    "planning_statement",
    "wrong_reasoning",
    "wrong_planning",
    "hallucination",
    "correct_submission",
    "todo_list",
    "early_final_answer",
    "non_sense",
    "syntax_error",
    "validation_attempt",
    "misunderstood_tool",
    "backtrack_trigger",
    "neutral",
}

DEFAULT_ANNOTATION_STYLE = {
    "xytext": (8, 0),
    "ha": "left",
    "va": "center",
}

ANNOTATION_STYLES = {
    "planning_statement": {
        "xytext": (0, 8),
        "ha": "center",
        "va": "bottom",
    },
    "loop_instance": {
        "xytext": (0, -10),
        "ha": "center",
        "va": "top",
    },
    "wrong_planning": {
        "xytext": (0, 8),
        "ha": "center",
        "va": "bottom",
    },
}

with DATA_PATH.open() as f:
    data = json.load(f)

models = sorted({d["model"] for d in data})
scaffolds = sorted({d["scaffold"] for d in data})

all_markers = set()
raw_counts = {m: {s: defaultdict(list) for s in scaffolds} for m in models}
binary = {m: {s: defaultdict(list) for s in scaffolds} for m in models}

for entry in data:
    model = entry["model"]
    scaffold = entry["scaffold"]

    trace_counts = defaultdict(int)
    trace_present = defaultdict(int)
    for ann in entry["annotations"].values():
        for marker in ann["markers"]:
            trace_counts[marker] += 1
            trace_present[marker] = 1
            all_markers.add(marker)

    for marker in all_markers:
        raw_counts[model][scaffold][marker]
        binary[model][scaffold][marker]

    for marker in all_markers:
        raw_counts[model][scaffold][marker].append(trace_counts.get(marker, 0))
        binary[model][scaffold][marker].append(trace_present.get(marker, 0))

# Second pass to back-fill zeros for markers discovered later
for marker in all_markers:
    for m in models:
        for s in scaffolds:
            n_traces = sum(1 for d in data if d["model"] == m and d["scaffold"] == s)
            current = len(raw_counts[m][s][marker])
            if current < n_traces:
                raw_counts[m][s][marker].extend([0] * (n_traces - current))
                binary[m][s][marker].extend([0] * (n_traces - current))

all_markers = sorted(all_markers)

# Aggregate sentiment counts per model x scaffold
sentiment_counts = defaultdict(lambda: defaultdict(int))
sentiment_totals = defaultdict(int)
for entry in data:
    combo = (entry["model"], entry["scaffold"])
    for ann in entry["annotations"].values():
        for marker in ann["markers"]:
            sent = MARKER_SENTIMENT.get(marker, "negative")
            sentiment_counts[combo][sent] += 1
            sentiment_totals[combo] += 1


def compute_impacts(stat_dict):
    """Compute normalized model and scaffold effect sizes for each marker.

    Args:
        stat_dict: Nested mapping of per-trace marker statistics organized by
            model and scaffold.

    Returns:
        A tuple containing marker names, model impact values, and scaffold
        impact values.
    """
    marker_names = []
    model_impacts = []
    scaffold_impacts = []

    for marker in all_markers:
        means = {}
        for m in models:
            for s in scaffolds:
                means[(m, s)] = np.mean(stat_dict[m][s][marker])

        global_mean = np.mean([means[(m, s)] for m in models for s in scaffolds])

        model_diffs = [
            abs(means[(models[0], s)] - means[(models[1], s)]) for s in scaffolds
        ]
        model_impact = np.mean(model_diffs)

        scaffold_diffs = [
            abs(means[(m, scaffolds[0])] - means[(m, scaffolds[1])]) for m in models
        ]
        scaffold_impact = np.mean(scaffold_diffs)

        if global_mean > 1e-9:
            model_impact /= global_mean
            scaffold_impact /= global_mean

        marker_names.append(marker)
        model_impacts.append(model_impact)
        scaffold_impacts.append(scaffold_impact)

    return marker_names, np.array(model_impacts), np.array(scaffold_impacts)


def aggregate_impacts_by_sentiment(marker_names, model_impacts, scaffold_impacts):
    """Aggregate effect sizes within marker sentiment groups.

    Args:
        marker_names: Marker labels in the same order as the metric arrays.
        model_impacts: Model effect sizes per marker.
        scaffold_impacts: Scaffold effect sizes per marker.

    Returns:
        Sentiment labels together with mean model and scaffold impacts per
        sentiment.
    """
    grouped = defaultdict(lambda: {"model": [], "scaffold": []})

    for marker, model_impact, scaffold_impact in zip(
        marker_names, model_impacts, scaffold_impacts, strict=False
    ):
        sentiment = MARKER_SENTIMENT.get(marker, "negative")
        grouped[sentiment]["model"].append(model_impact)
        grouped[sentiment]["scaffold"].append(scaffold_impact)

    sentiments = [sentiment for sentiment in SENTIMENT_ORDER if sentiment in grouped]
    sentiments.extend(sorted(set(grouped) - set(sentiments)))

    return (
        sentiments,
        np.array([np.mean(grouped[sentiment]["model"]) for sentiment in sentiments]),
        np.array([np.mean(grouped[sentiment]["scaffold"]) for sentiment in sentiments]),
    )


def _save_fig(fig, filename):
    """Save a figure into the analysis output directory.

    Args:
        fig: Matplotlib figure to save.
        filename: Output filename, optionally with or without a suffix.

    Returns:
        None.
    """
    out_dir = Path(__file__).parent / "results" / "figures" / "fig_5"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / Path(filename).with_suffix(".pdf")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved → {save_path}")
    plt.close(fig)


def add_panel_label(ax, label, x=-0.18, y=1.08):
    """Place a bold panel label just outside an axis."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=16,
        color="black",
        ha="center",
        va="center",
        clip_on=False,
    )


def make_combined_plot(marker_names, model_imp, scaffold_imp, filename):
    """Create a combined figure: scatter (left) + sentiment bars (right).

    Left panel: model-vs-scaffold effect-size scatter.
    Right panels: stacked sentiment proportions per model x scaffold.
    A single legend is shared between all panels.

    Args:
        marker_names: Marker labels in plotting order.
        model_imp: Model effect sizes per marker.
        scaffold_imp: Scaffold effect sizes per marker.
        filename: Output filename for the saved figure.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 0.5, 0.5], wspace=0.35)

    ax_scatter = fig.add_subplot(gs[0, 0])

    diag_end = min(max(model_imp.max(), scaffold_imp.max()) * 1.05, 2.0)
    ax_scatter.plot(
        [0, diag_end], [0, diag_end], ls="--", color="grey", alpha=0.4, lw=1
    )

    plotted_sentiments = set()
    for i, marker in enumerate(marker_names):
        sentiment = MARKER_SENTIMENT.get(marker, "negative")
        color = SENTIMENT_COLORS.get(sentiment, "#888888")
        label = sentiment.capitalize() if sentiment not in plotted_sentiments else None
        plotted_sentiments.add(sentiment)

        ax_scatter.scatter(
            scaffold_imp[i],
            model_imp[i],
            color=color,
            s=50,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=label,
        )

    ax_scatter.set_xlabel("Scaffold count")
    ax_scatter.set_ylabel("Model count")
    range_frame(ax_scatter, x=np.array([0, 2]), y=np.array([0, 2]))
    ax_scatter.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_scatter.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    add_panel_label(ax_scatter, "A")

    sentiment_order = ["positive", "neutral", "negative"]
    bar_width = 0.45
    bar_axes = []

    for col_idx, model in enumerate(models):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        bar_axes.append(ax)

        combos = [(model, s) for s in scaffolds]
        x = np.arange(len(combos))
        bottoms = np.zeros(len(combos))

        for sent in sentiment_order:
            vals = [
                sentiment_counts[c][sent] / sentiment_totals[c] * 100
                if sentiment_totals[c] > 0
                else 0
                for c in combos
            ]
            ax.bar(
                x,
                vals,
                bar_width,
                bottom=bottoms,
                color=SENTIMENT_COLORS[sent],
                edgecolor="white",
                linewidth=0.5,
            )
            for i, v in enumerate(vals):
                if v > 5:
                    ax.text(
                        x[i],
                        bottoms[i] + v / 2,
                        f"{v:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                        fontweight="bold",
                    )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([SCAFFOLD_DISPLAY.get(s, s) for s in scaffolds], fontsize=10)
        ax.set_xlabel(MODEL_DISPLAY.get(model, model), fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_idx > 0:
            ax.tick_params(labelleft=False)

    bar_axes[0].set_ylabel("Marker proportion (%)", fontsize=10)
    add_panel_label(bar_axes[0], "B", x=-0.4)

    handles, labels = ax_scatter.get_legend_handles_labels()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.legend(
        handles,
        labels,
        title="Sentiment",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(labels),
        fontsize=10,
        title_fontsize=10,
        frameon=False,
    )

    # Align x-axis spines: force all axes to share the same bottom and top
    fig.canvas.draw()
    all_axes = [ax_scatter, *bar_axes]
    y0 = min(ax.get_position().y0 for ax in all_axes)
    y1 = max(ax.get_position().y1 for ax in all_axes)
    for ax in all_axes:
        p = ax.get_position()
        ax.set_position([p.x0, y0, p.width, y1 - y0])

    _save_fig(fig, filename)


names_raw, mi_raw, si_raw = compute_impacts(raw_counts)
make_combined_plot(
    names_raw,
    mi_raw,
    si_raw,
    "marker_impact_raw_counts.pdf",
)

logger.info("\n=== Original normalised effect sizes (raw counts) ===")
logger.info(
    "\n{:<25s}  {:>12s}  {:>14s}  {:>10s}".format(
        "Marker", "Model Imp.", "Scaffold Imp.", "Dominant"
    )
)
logger.info("-" * 65)
for i, marker in enumerate(names_raw):
    dominant = "MODEL" if mi_raw[i] > si_raw[i] else "SCAFFOLD"
    logger.info(
        f"{marker:<25s}  {mi_raw[i]:>12.3f}  {si_raw[i]:>14.3f}  {dominant:>10s}"
    )
