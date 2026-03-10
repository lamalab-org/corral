"""Quantify how marker frequencies vary by model, scaffold, and task content.

This analysis script loads the curated annotation dataset, computes per-trace
marker statistics, compares effect sizes for model and scaffold choices, and
fits regression models that optionally include task-description embeddings. The
resulting figures and summary tables are written to the analysis results
directory for downstream interpretation.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "filtered_annotations.json"
REPO_ROOT = Path(__file__).resolve().parent.parent

N_EMBEDDING_COMPONENTS = 10

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


def _build_trace_index() -> dict[str, Path]:
    """Index trace files by basename across both report directories.

    Returns:
        A mapping from JSON file name to its absolute path in `reports` or
        `reports_v2`.
    """
    idx: dict[str, Path] = {}
    for directory in ("reports", "reports_v2"):
        for fp in (REPO_ROOT / directory).rglob("*.json"):
            idx[fp.name] = fp
    return idx


def _load_retrosynthesis_prompts() -> dict[str, str]:
    """Load retrosynthesis task prompts keyed by task identifier.

    Returns:
        A mapping from retrosynthesis task id to prompt text.
    """
    prompts: dict[str, str] = {}
    base = REPO_ROOT / "tasks" / "retrosynthesis" / "environments"
    for task_file in base.glob("level_*/tasks/*.json"):
        with task_file.open() as f:
            tasks = json.load(f)
        for task in tasks:
            prompts[task["id"]] = task["input"]["prompt"]
    return prompts


def _load_wetlab_prompts() -> dict[str, str]:
    """Load wetlab task prompts keyed by task identifier.

    Returns:
        A mapping from wetlab task id to prompt text.
    """
    prompts: dict[str, str] = {}
    base = REPO_ROOT / "tasks" / "wetlab" / "wetlab" / "tasks_json"
    for task_file in base.glob("level_*/*.json"):
        with task_file.open() as f:
            tasks = json.load(f)
        for task in tasks:
            prompts[task["id"]] = task["input"]["prompt"]
    return prompts


def _load_afm_prompts() -> dict[str, str]:
    """Load AFM task descriptions keyed by task identifier.

    Returns:
        A mapping from AFM task id to prompt text.
    """
    prompts: dict[str, str] = {}
    base = REPO_ROOT / "tasks" / "afm" / "src" / "enviroment"
    for task_file in base.glob("tasks_*.json"):
        with task_file.open() as f:
            d = json.load(f)
        for task_id, info in d.items():
            prompts[task_id] = info["description"]
    return prompts


def extract_task_descriptions(entries: list[dict]) -> dict[str, str]:
    """Resolve task descriptions for annotated traces.

    Args:
        entries: Filtered annotation records.

    Returns:
        A mapping from `fileId` to the best available task description,
        preferring prompts embedded in trace files and falling back to task
        definition files for environments whose traces omit the prompt.
    """
    trace_idx = _build_trace_index()
    retro_prompts = _load_retrosynthesis_prompts()
    wetlab_prompts = _load_wetlab_prompts()
    afm_prompts = _load_afm_prompts()

    descriptions: dict[str, str] = {}

    for entry in entries:
        fid = entry["fileId"]
        if fid in descriptions:
            continue

        if fid in trace_idx:
            try:
                with trace_idx[fid].open() as tf:
                    trace = json.load(tf)
                msgs = (
                    trace.get("messages", trace) if isinstance(trace, dict) else trace
                )
                for msg in msgs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        descriptions[fid] = str(msg.get("content", ""))
                        break
            except Exception:
                pass

        if fid not in descriptions and entry["env"] == "retrosynthesis":
            m = re.match(r"(make_\d+_lvl\d+)_\d{8}_\d{6}", fid.replace(".json", ""))
            if m and m.group(1) in retro_prompts:
                descriptions[fid] = retro_prompts[m.group(1)]

        if fid not in descriptions and entry["env"] == "wetlab":
            m = re.match(r"(qualysis_lvl\d+_\d+)_\d{8}_\d{6}", fid.replace(".json", ""))
            if m and m.group(1) in wetlab_prompts:
                descriptions[fid] = wetlab_prompts[m.group(1)]

        if fid not in descriptions and entry["env"] == "afm":
            m = re.match(
                r"(afm_experiment_level_\d+)_\d{8}_\d{6}", fid.replace(".json", "")
            )
            if m and m.group(1) in afm_prompts:
                descriptions[fid] = afm_prompts[m.group(1)]

    logger.info(
        f"Task descriptions extracted: {len(descriptions)}/{len({e['fileId'] for e in entries})}"
    )
    return descriptions


def compute_task_embeddings(
    entries: list[dict],
    descriptions: dict[str, str],
    n_components: int = N_EMBEDDING_COMPONENTS,
) -> dict[str, np.ndarray]:
    """Embed task descriptions and reduce them with PCA.

    Args:
        entries: Filtered annotation records.
        descriptions: Mapping from `fileId` to task description.
        n_components: Target dimensionality for the PCA projection.

    Returns:
        A mapping from `fileId` to its PCA-reduced sentence embedding.
    """
    fid_to_desc: dict[str, str] = {}
    for entry in entries:
        fid = entry["fileId"]
        if fid in descriptions:
            fid_to_desc[fid] = descriptions[fid]

    unique_texts = sorted(set(fid_to_desc.values()))
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    logger.info(f"Embedding {len(unique_texts)} unique task descriptions …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    raw_embeddings = model.encode(unique_texts, show_progress_bar=True)

    n_components = min(n_components, len(unique_texts), raw_embeddings.shape[1])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(raw_embeddings)
    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA: {n_components} components explain {explained:.1%} of embedding variance"
    )

    fid_embeddings: dict[str, np.ndarray] = {}
    for fid, desc in fid_to_desc.items():
        fid_embeddings[fid] = reduced[text_to_idx[desc]]

    return fid_embeddings


trace_order: dict[tuple[str, str], list[str]] = defaultdict(list)
for entry in data:
    trace_order[(entry["model"], entry["scaffold"])].append(entry["fileId"])

task_descriptions = extract_task_descriptions(data)
task_embeddings = compute_task_embeddings(data, task_descriptions)


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


def compute_impacts_regression(stat_dict):
    """Fit an effect-coded regression for each marker.

    Args:
        stat_dict: Nested mapping of per-trace marker statistics organized by
            model and scaffold.

    Returns:
        Marker names, standardized coefficients for model, scaffold, and their
        interaction, plus the regression $R^2$ for each marker.
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


def compute_impacts_regression_with_embeddings(stat_dict):
    """Fit the marker regression with task embeddings as extra predictors.

    Args:
        stat_dict: Nested mapping of per-trace marker statistics organized by
            model and scaffold.

    Returns:
        Marker names, standardized coefficients for model, scaffold, and their
        interaction, regression $R^2$ with embeddings, and baseline $R^2$
        without embeddings.
    """
    marker_names = []
    beta_model = []
    beta_scaffold = []
    beta_interaction = []
    r_squared = []
    r_squared_no_emb = []

    n_emb = N_EMBEDDING_COMPONENTS

    for marker in all_markers:
        y_vals = []
        x_model_vals = []
        x_scaffold_vals = []
        emb_rows = []

        for m_idx, m in enumerate(models):
            for s_idx, s in enumerate(scaffolds):
                vals = stat_dict[m][s][marker]
                codes_m = -1 if m_idx == 0 else 1
                codes_s = -1 if s_idx == 0 else 1
                fids = trace_order[(m, s)]

                for val, fid in zip(vals, fids, strict=False):
                    y_vals.append(val)
                    x_model_vals.append(codes_m)
                    x_scaffold_vals.append(codes_s)
                    if fid in task_embeddings:
                        emb_rows.append(task_embeddings[fid])
                    else:
                        emb_rows.append(np.zeros(n_emb))

        y = np.array(y_vals, dtype=float)
        x_m = np.array(x_model_vals, dtype=float)
        x_s = np.array(x_scaffold_vals, dtype=float)
        x_int = x_m * x_s
        emb_matrix = np.array(emb_rows, dtype=float)

        X_base = np.column_stack([x_m, x_s, x_int])
        reg_base = LinearRegression().fit(X_base, y)
        r2_base = reg_base.score(X_base, y)

        X_full = np.column_stack([x_m, x_s, x_int, emb_matrix])
        reg_full = LinearRegression().fit(X_full, y)
        y_std = y.std()

        betas_full = reg_full.coef_[:3] / y_std if y_std > 1e-12 else np.zeros(3)

        marker_names.append(marker)
        beta_model.append(betas_full[0])
        beta_scaffold.append(betas_full[1])
        beta_interaction.append(betas_full[2])
        r_squared.append(reg_full.score(X_full, y))
        r_squared_no_emb.append(r2_base)

    return (
        marker_names,
        np.array(beta_model),
        np.array(beta_scaffold),
        np.array(beta_interaction),
        np.array(r_squared),
        np.array(r_squared_no_emb),
    )


def aggregate_regression_by_sentiment(marker_names, beta_m, beta_s, beta_int, r2):
    """Aggregate regression contributions within marker sentiment groups.

    Args:
        marker_names: Marker labels in the same order as the metric arrays.
        beta_m: Standardized model coefficients.
        beta_s: Standardized scaffold coefficients.
        beta_int: Standardized interaction coefficients.
        r2: Regression $R^2$ values.

    Returns:
        Sentiment labels together with mean squared coefficients and mean $R^2$
        per sentiment.
    """
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

    ax.set_xlabel("Scaffold impact")
    ax.set_ylabel("Model impact")

    range_frame(ax, x=np.array([0, 2]), y=np.array([0, 2]))

    ax.legend(title="Sentiment", loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    _save_fig(fig, filename)


def make_plot_regression(marker_names, beta_m, beta_s, _beta_int, r2, title, filename):
    """Create the regression coefficient scatter plot.

    Args:
        marker_names: Marker labels in plotting order.
        beta_m: Standardized model coefficients.
        beta_s: Standardized scaffold coefficients.
        _beta_int: Unused interaction coefficients retained for API symmetry.
        r2: Regression $R^2$ values used for bubble sizes.
        title: Plot title.
        filename: Output filename for the saved figure.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    ax.axhline(0, color="grey", alpha=0.3, lw=0.8)
    ax.axvline(0, color="grey", alpha=0.3, lw=0.8)

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
    """Create a stacked bar chart of regression variance contributions.

    Args:
        marker_names: Marker or sentiment labels.
        b2_m: Mean squared model coefficients.
        b2_s: Mean squared scaffold coefficients.
        b2_i: Mean squared interaction coefficients.
        r2: Mean regression $R^2$ values.
        _title: Unused legacy title argument retained for call-site stability.
        filename: Output filename for the saved figure.

    Returns:
        None.
    """
    if set(marker_names).issubset(SENTIMENT_ORDER):
        order = np.array(
            sorted(
                range(len(marker_names)),
                key=lambda i: SENTIMENT_ORDER.index(marker_names[i]),
            )
        )
    else:
        order = np.argsort(-r2)
    names_sorted = [marker_names[i] for i in order]
    b2_m = b2_m[order]
    b2_s = b2_s[order]
    b2_i = b2_i[order]
    r2_sorted = r2[order]

    label_colors = [
        SENTIMENT_COLORS.get(MARKER_SENTIMENT.get(n) or n, "#888888")
        for n in names_sorted
    ]
    label_fontsize = plt.rcParams["xtick.labelsize"]

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    y_pos = np.arange(len(names_sorted))

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
    for tick_label, color in zip(ax.get_yticklabels(), label_colors, strict=False):
        tick_label.set_color(color)

    ax.invert_yaxis()
    ax.set_xlabel(r"Variance explained ($\beta^2$)")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=label_fontsize,
    )
    range_frame(ax, x=np.array([0, max(0.175, x_max)]), y=np.array([0, 2]))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    _save_fig(fig, filename)


names_raw, mi_raw, si_raw = compute_impacts(raw_counts)
make_plot(
    names_raw,
    mi_raw,
    si_raw,
    "Marker Impact Decomposition — Raw Counts per Trace",
    "marker_impact_raw_counts.pdf",
)


names_r, bm_raw, bs_raw, bi_raw, r2_raw = compute_impacts_regression(raw_counts)

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

names_emb, bm_emb, bs_emb, bi_emb, r2_emb, r2_emb_base = (
    compute_impacts_regression_with_embeddings(raw_counts)
)

(
    sentiment_names_emb,
    sentiment_b2m_emb,
    sentiment_b2s_emb,
    sentiment_b2i_emb,
    sentiment_r2_emb,
) = aggregate_regression_by_sentiment(names_emb, bm_emb, bs_emb, bi_emb, r2_emb)
make_stacked_bar_regression(
    sentiment_names_emb,
    sentiment_b2m_emb,
    sentiment_b2s_emb,
    sentiment_b2i_emb,
    sentiment_r2_emb,
    r"Variance Decomposition (Regression $\beta^2$ + Task Embeddings) — Raw Counts",
    "stacked_regression_raw_counts_with_embeddings.pdf",
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

logger.info("\n=== Regression standardised β (raw counts) ===")
logger.info(
    "\n{:<25s}  {:>10s}  {:>12s}  {:>14s}  {:>6s}".format(
        "Marker", "β_model", "β_scaffold", "β_interact.", "R²"
    )
)
logger.info("-" * 72)
for i, marker in enumerate(names_r):
    logger.info(
        f"{marker:<25s}  {bm_raw[i]:>+10.4f}  {bs_raw[i]:>+12.4f}"
        f"  {bi_raw[i]:>+14.4f}  {r2_raw[i]:>6.3f}"
    )

logger.info("\n=== R² comparison: baseline vs with task embeddings ===")
logger.info(
    "\n{:<25s}  {:>10s}  {:>10s}  {:>10s}".format("Marker", "R² base", "R² +emb", "ΔR²")
)
logger.info("-" * 60)
for i, marker in enumerate(names_emb):
    delta = r2_emb[i] - r2_emb_base[i]
    logger.info(
        f"{marker:<25s}  {r2_emb_base[i]:>10.4f}  {r2_emb[i]:>10.4f}  {delta:>+10.4f}"
    )

logger.info(
    f"\nMean R² baseline:        {np.mean(r2_emb_base):.4f}"
    f"\nMean R² with embeddings: {np.mean(r2_emb):.4f}"
    f"\nMean ΔR²:                {np.mean(r2_emb - r2_emb_base):+.4f}"
)
