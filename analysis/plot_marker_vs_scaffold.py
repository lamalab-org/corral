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
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
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

models = sorted({d["model"] for d in data})  # ['claude_sonnet_45', 'gpt-4o']
scaffolds = sorted({d["scaffold"] for d in data})  # ['react', 'tool_calling']

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


def make_plot(marker_names, model_imp, scaffold_imp, _title, filename):
    """Create the model-vs-scaffold effect-size scatter plot.

    Args:
        marker_names: Marker labels in plotting order.
        model_imp: Model effect sizes per marker.
        scaffold_imp: Scaffold effect sizes per marker.
        _title: Unused legacy title argument retained for call-site stability.
        filename: Output filename for the saved figure.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    max_val = max(model_imp.max(), scaffold_imp.max()) * 1.15
    ax.plot([0, max_val], [0, max_val], ls="--", color="grey", alpha=0.4, lw=1)

    plotted_sentiments = set()
    for i, marker in enumerate(marker_names):
        sentiment = MARKER_SENTIMENT.get(marker, "negative")
        color = SENTIMENT_COLORS.get(sentiment, "#888888")
        label = sentiment.capitalize() if sentiment not in plotted_sentiments else None
        plotted_sentiments.add(sentiment)

        ax.scatter(
            scaffold_imp[i],
            model_imp[i],
            color=color,
            s=80,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=label,
        )

    ax.set_xlabel("Scaffold Counts")
    ax.set_ylabel("Model Counts")

    range_frame(ax, x=np.array([0, 2]), y=np.array([0, 2]))

    ax.legend(title="Sentiment", loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    _save_fig(fig, filename)


names_raw, mi_raw, si_raw = compute_impacts(raw_counts)
make_plot(
    names_raw,
    mi_raw,
    si_raw,
    "Marker Count Decomposition — Raw Counts per Trace",
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
