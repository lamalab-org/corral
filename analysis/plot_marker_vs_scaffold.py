"""
Plot marker-type impact decomposition: model impact vs scaffold impact.

For each marker type, we compute:
  - Model impact:    mean_across_scaffolds(|mean_claude - mean_gpt4o|) / global_mean
  - Scaffold impact: mean_across_models(|mean_react - mean_tool_calling|) / global_mean

Two figures are produced:
  1. Raw counts  - mean number of marker occurrences per trace
  2. Binary presence - fraction of traces containing the marker at least once
"""

# ruff: noqa: T201

import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from sklearn.linear_model import LinearRegression

lama_aesthetics.get_style("main")

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "results" / "data" / "filtered_annotations.json"
with DATA_PATH.open() as f:
    data = json.load(f)

models = sorted({d["model"] for d in data})  # ['claude_sonnet_45', 'gpt-4o']
scaffolds = sorted({d["scaffold"] for d in data})  # ['react', 'tool_calling']

# ── Collect per-trace marker stats ───────────────────────────────────────────
# raw_counts[model][scaffold]  -> {marker: [count_per_trace, ...]}
# binary[model][scaffold]      -> {marker: [0_or_1_per_trace, ...]}
all_markers = set()
raw_counts = {m: {s: defaultdict(list) for s in scaffolds} for m in models}
binary = {m: {s: defaultdict(list) for s in scaffolds} for m in models}

for entry in data:
    model = entry["model"]
    scaffold = entry["scaffold"]

    # Count markers in this trace
    trace_counts = defaultdict(int)
    trace_present = defaultdict(int)
    for ann in entry["annotations"].values():
        for marker in ann["markers"]:
            trace_counts[marker] += 1
            trace_present[marker] = 1
            all_markers.add(marker)

    # Store counts (including 0 for markers not seen in this trace)
    for marker in all_markers:
        raw_counts[model][scaffold][marker]  # ensure key exists
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


# ── Compute impacts ─────────────────────────────────────────────────────────
def compute_impacts(stat_dict):
    """Return (marker_names, model_impacts, scaffold_impacts)."""
    marker_names = []
    model_impacts = []
    scaffold_impacts = []

    for marker in all_markers:
        # means per condition
        means = {}
        for m in models:
            for s in scaffolds:
                means[(m, s)] = np.mean(stat_dict[m][s][marker])

        # Global mean for this marker (pool all conditions)
        global_mean = np.mean([means[(m, s)] for m in models for s in scaffolds])

        # Model impact: average over scaffolds of |claude - gpt4o|
        model_diffs = [
            abs(means[(models[0], s)] - means[(models[1], s)]) for s in scaffolds
        ]
        model_impact = np.mean(model_diffs)

        # Scaffold impact: average over models of |react - tool_calling|
        scaffold_diffs = [
            abs(means[(m, scaffolds[0])] - means[(m, scaffolds[1])]) for m in models
        ]
        scaffold_impact = np.mean(scaffold_diffs)

        # Normalise by global mean (effect size) - skip if global mean ≈ 0
        if global_mean > 1e-9:
            model_impact /= global_mean
            scaffold_impact /= global_mean

        marker_names.append(marker)
        model_impacts.append(model_impact)
        scaffold_impacts.append(scaffold_impact)

    return marker_names, np.array(model_impacts), np.array(scaffold_impacts)


def aggregate_impacts_by_sentiment(marker_names, model_impacts, scaffold_impacts):
    """Average model/scaffold impacts within each sentiment group."""
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


# ── Categorise markers for colour coding ─────────────────────────────────────
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


# ── Linear regression β coefficients ────────────────────────────────────────
def compute_impacts_regression(stat_dict):
    """Effect-coded linear regression per marker.

    For each marker, fit:
        y = β₀ + β_model·x_model + β_scaffold·x_scaffold + β_int·x_model·x_scaffold + ε

    where x_model ∈ {-1, +1}  (-1 = first model,  +1 = second model)
          x_scaffold ∈ {-1, +1}  (-1 = first scaffold, +1 = second scaffold)

    Returns standardised β (divided by std(y)) so markers are comparable.
    Positive β_model means second model (gpt-4o) has higher values.
    Positive β_scaffold means second scaffold (tool_calling) has higher values.
    """
    marker_names = []
    beta_model = []
    beta_scaffold = []
    beta_interaction = []
    r_squared = []

    for marker in all_markers:
        y_vals = []
        x_model_vals = []
        x_scaffold_vals = []

        for m_idx, m in enumerate(models):
            for s_idx, s in enumerate(scaffolds):
                vals = stat_dict[m][s][marker]
                codes_m = -1 if m_idx == 0 else 1
                codes_s = -1 if s_idx == 0 else 1
                y_vals.extend(vals)
                x_model_vals.extend([codes_m] * len(vals))
                x_scaffold_vals.extend([codes_s] * len(vals))

        y = np.array(y_vals, dtype=float)
        x_m = np.array(x_model_vals, dtype=float)
        x_s = np.array(x_scaffold_vals, dtype=float)
        x_int = x_m * x_s

        X = np.column_stack([x_m, x_s, x_int])

        reg = LinearRegression().fit(X, y)
        y_std = y.std()

        # Standardise coefficients (if y has no variance, coefficients are 0)
        betas = reg.coef_ / y_std if y_std > 1e-12 else np.zeros(3)

        marker_names.append(marker)
        beta_model.append(betas[0])
        beta_scaffold.append(betas[1])
        beta_interaction.append(betas[2])
        r_squared.append(reg.score(X, y))

    return (
        marker_names,
        np.array(beta_model),
        np.array(beta_scaffold),
        np.array(beta_interaction),
        np.array(r_squared),
    )


def aggregate_regression_by_sentiment(marker_names, beta_m, beta_s, beta_int, r2):
    """Average regression variance contributions within each sentiment group."""
    grouped = defaultdict(lambda: {"b2_m": [], "b2_s": [], "b2_i": [], "r2": []})

    for marker, beta_model, beta_scaffold, beta_interaction, r2_value in zip(
        marker_names, beta_m, beta_s, beta_int, r2, strict=False
    ):
        sentiment = MARKER_SENTIMENT.get(marker, "negative")
        grouped[sentiment]["b2_m"].append(beta_model**2)
        grouped[sentiment]["b2_s"].append(beta_scaffold**2)
        grouped[sentiment]["b2_i"].append(beta_interaction**2)
        grouped[sentiment]["r2"].append(r2_value)

    sentiments = [sentiment for sentiment in SENTIMENT_ORDER if sentiment in grouped]
    sentiments.extend(sorted(set(grouped) - set(sentiments)))

    return (
        sentiments,
        np.array([np.mean(grouped[sentiment]["b2_m"]) for sentiment in sentiments]),
        np.array([np.mean(grouped[sentiment]["b2_s"]) for sentiment in sentiments]),
        np.array([np.mean(grouped[sentiment]["b2_i"]) for sentiment in sentiments]),
        np.array([np.mean(grouped[sentiment]["r2"]) for sentiment in sentiments]),
    )


# ── Plotting helpers ─────────────────────────────────────────────────────────
def _save_fig(fig, filename):
    out_dir = Path(__file__).parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / Path(filename).with_suffix(".pdf")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


def make_plot(marker_names, model_imp, scaffold_imp, _title, filename):
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    # Diagonal reference line (equal impact)
    max_val = max(model_imp.max(), scaffold_imp.max()) * 1.15
    ax.plot([0, max_val], [0, max_val], ls="--", color="grey", alpha=0.4, lw=1)

    # Plot each sentiment average
    plotted_sentiments = set()
    for i, marker in enumerate(marker_names):
        sentiment = marker
        color = SENTIMENT_COLORS.get(sentiment, "#888888")
        label = sentiment.capitalize() if sentiment not in plotted_sentiments else None
        plotted_sentiments.add(sentiment)

        ax.scatter(
            model_imp[i],
            scaffold_imp[i],
            color=color,
            s=80,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=label,
        )

        ax.annotate(
            sentiment.capitalize(),
            (model_imp[i], scaffold_imp[i]),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontsize=9,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel("Model impact")
    ax.set_ylabel("Scaffold impact")

    # Apply range_frame (handles padding and spine bounds automatically)
    range_frame(ax, x=np.array([0, 2]), y=np.array([0, 1.75]))

    ax.legend(title="Sentiment", loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    _save_fig(fig, filename)


def make_plot_regression(marker_names, beta_m, beta_s, _beta_int, r2, title, filename):
    """Scatter of β_model vs β_scaffold (standardised), with sign = direction."""
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    # Cross-hair at origin
    ax.axhline(0, color="grey", alpha=0.3, lw=0.8)
    ax.axvline(0, color="grey", alpha=0.3, lw=0.8)

    # Scale R² to bubble size
    sizes = 40 + 260 * r2

    plotted_categories = set()
    for i, marker in enumerate(marker_names):
        cat = MARKER_CATEGORY.get(marker, "process")
        color = CATEGORY_COLORS.get(cat, "#888888")
        label = cat if cat not in plotted_categories else None
        plotted_categories.add(cat)

        ax.scatter(
            beta_m[i],
            beta_s[i],
            color=color,
            s=sizes[i],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label=label,
            alpha=0.8,
        )

        ax.annotate(
            marker.replace("_", " "),
            (beta_m[i], beta_s[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color=color,
            alpha=0.85,
        )

    ax.set_xlabel(
        f"β_model (standardised)\n"
        f"← {models[0].replace('_', ' ')}    {models[1].replace('_', ' ')} →"
    )
    ax.set_ylabel(
        f"β_scaffold (standardised)\n"
        f"← {scaffolds[0]}    {scaffolds[1].replace('_', ' ')} →"
    )
    ax.set_title(title)

    range_frame(ax, x=beta_m, y=beta_s)
    ax.legend(title="Category", loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")

    # Quadrant annotations
    ax.annotate(
        f"{models[1]}\n+ {scaffolds[1]}",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6,
        color="grey",
        alpha=0.5,
    )
    ax.annotate(
        f"{models[0]}\n+ {scaffolds[0]}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=6,
        color="grey",
        alpha=0.5,
    )

    # Note about bubble size = R²
    ax.annotate(
        r"bubble size $\propto$ $R^2$",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7,
        color="grey",
        style="italic",
    )

    fig.tight_layout()
    _save_fig(fig, filename)


def make_stacked_bar_regression(marker_names, b2_m, b2_s, b2_i, r2, _title, filename):
    """Stacked bar plot: per group, show mean variance contributions.

    For each group we show the mean of β²_model, β²_scaffold, β²_interaction
    and annotate the mean R² at the end of each bar.
    """
    # Sort markers by R² descending
    order = np.argsort(-r2)
    names_sorted = [marker_names[i] for i in order]
    b2_m = b2_m[order]
    b2_s = b2_s[order]
    b2_i = b2_i[order]
    r2_sorted = r2[order]

    label_colors = [SENTIMENT_COLORS.get(n, "#888888") for n in names_sorted]
    label_fontsize = plt.rcParams["xtick.labelsize"]

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    y_pos = np.arange(len(names_sorted))

    # Stacked horizontal bars
    bar_h = 0.5
    ax.barh(
        y_pos, b2_m, height=bar_h, color="#2196F3", label=r"$\beta^2_{\mathrm{model}}$"
    )
    ax.barh(
        y_pos,
        b2_s,
        height=bar_h,
        left=b2_m,
        color="#FF9800",
        label=r"$\beta^2_{\mathrm{scaffold}}$",
    )
    ax.barh(
        y_pos,
        b2_i,
        height=bar_h,
        left=b2_m + b2_s,
        color="#9C27B0",
        label=r"$\beta^2_{\mathrm{interaction}}$",
    )

    # Annotate R² value at the end of each bar
    bar_ends = b2_m + b2_s + b2_i
    x_max = bar_ends.max()
    x_pad = max(x_max * 0.02, 0.005)
    for i, (bar_end, r2_val) in enumerate(zip(bar_ends, r2_sorted, strict=False)):
        ax.text(
            bar_end + x_pad,
            y_pos[i],
            rf"$R^2$={r2_val:.3f}",
            va="center",
            ha="left",
            fontsize=label_fontsize,
            color="#555555",
        )

    ax.set_xlim(0, x_max + 7 * x_pad)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.capitalize() for n in names_sorted], fontsize=label_fontsize)
    # Colour y-tick labels by sentiment
    for tick_label, color in zip(ax.get_yticklabels(), label_colors, strict=False):
        tick_label.set_color(color)

    ax.invert_yaxis()  # highest R² at top
    ax.set_xlabel(r"Variance explained ($\beta^2$)")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=label_fontsize,
    )
    range_frame(ax, x=np.array([0, 0.175]), y=np.array([0, 2]))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, filename)


# ── Generate plots ──────────────────────────────────────────────────────────

# 1. Original normalised effect-size plots
names_raw, mi_raw, si_raw = compute_impacts(raw_counts)
sentiment_names_raw, sentiment_mi_raw, sentiment_si_raw = (
    aggregate_impacts_by_sentiment(names_raw, mi_raw, si_raw)
)
make_plot(
    sentiment_names_raw,
    sentiment_mi_raw,
    sentiment_si_raw,
    "Marker Impact Decomposition — Raw Counts per Trace",
    "marker_impact_raw_counts.pdf",
)


# 2. Linear regression β decomposition
names_r, bm_raw, bs_raw, bi_raw, r2_raw = compute_impacts_regression(raw_counts)

# 3. Stacked bar plots — Regression β² decomposition
sentiment_names_r, sentiment_b2m, sentiment_b2s, sentiment_b2i, sentiment_r2 = (
    aggregate_regression_by_sentiment(names_r, bm_raw, bs_raw, bi_raw, r2_raw)
)
make_stacked_bar_regression(
    sentiment_names_r,
    sentiment_b2m,
    sentiment_b2s,
    sentiment_b2i,
    sentiment_r2,
    r"Variance Decomposition (Regression $\beta^2$) — Raw Counts",
    "stacked_regression_raw_counts.pdf",
)

# Original
print("\n=== Original normalised effect sizes (raw counts) ===")
print(
    "\n{:<25s}  {:>12s}  {:>14s}  {:>10s}".format(
        "Marker", "Model Imp.", "Scaffold Imp.", "Dominant"
    )
)
print("-" * 65)
for i, marker in enumerate(names_raw):
    dominant = "MODEL" if mi_raw[i] > si_raw[i] else "SCAFFOLD"
    print(f"{marker:<25s}  {mi_raw[i]:>12.3f}  {si_raw[i]:>14.3f}  {dominant:>10s}")

# Regression
print("\n=== Regression standardised β (raw counts) ===")
print(
    "\n{:<25s}  {:>10s}  {:>12s}  {:>14s}  {:>6s}".format(
        "Marker", "β_model", "β_scaffold", "β_interact.", "R²"
    )
)
print("-" * 72)
for i, marker in enumerate(names_r):
    print(
        f"{marker:<25s}  {bm_raw[i]:>+10.4f}  {bs_raw[i]:>+12.4f}"
        f"  {bi_raw[i]:>+14.4f}  {r2_raw[i]:>6.3f}"
    )
