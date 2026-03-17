"""
Plot anti-pattern and good-workflow *family*-level summaries from reasoning_reports,
using **raw_total** (absolute occurrence counts) instead of the trace-fraction used
in plot_family_patterns.py.

All plots use family-aggregated labels (evidence_handling, hypothesis_testing,
discovery, …) instead of the individual pattern names.

Usage:
  python analysis/plot_family_patterns_raw_total.py
  python analysis/plot_family_patterns_raw_total.py --summary-path /path/to/summary.json
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.patches import Patch

lama_aesthetics.get_style("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "reasoning_reports" / "analysis" / "annotation_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "figures" / "raw_total"

ANTIPATTERN_FAMILY_FIELDS = [
    "evidence_handling",
    "hypothesis_evaluation",
    "belief_revision_commitment",
]
SUBGRAPH_FAMILY_FIELDS = [
    "hypothesis_testing",
    "discovery",
    "search_optimization",
]
ALL_FAMILY_FIELDS = ANTIPATTERN_FAMILY_FIELDS + SUBGRAPH_FAMILY_FIELDS

METRIC_FIELDS = [
    "workflow_completeness",
    "loop_density",
    "update_grounding_rate",
    "orphan_evidence_rate",
    "refute_neglect_rate",
    "hypothesis_switch_without_eval_rate",
    "scientificness_score",
]

MODEL_COLORS = {"claude_sonnet_45": "#768eab", "gpt_4o": "#a285a6"}
GOOD_COLOR = "#5d8aa8"
BAD_COLOR = "#c05746"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def display_name(key: str, width: int = 18) -> str:
    return textwrap.fill(key.replace("_", " "), width=width)


def short_name(key: str) -> str:
    abbrevs = {
        "evidence_handling": "Evidence handling",
        "hypothesis_evaluation": "Hypothesis eval.",
        "belief_revision_commitment": "Belief rev. & commit.",
        "hypothesis_testing": "Hypothesis testing",
        "discovery": "Discovery",
        "search_optimization": "Search optim.",
    }
    return abbrevs.get(key, key.replace("_", " "))


def _raw_total(data: dict, section: str, field: str) -> float:
    v = data.get(section, {}).get(field, {}).get("raw_total")
    if v is None:
        return 0.0
    return float(v)


def _metric(data: dict, field: str) -> float:
    v = data.get("metric_means", {}).get(field)
    if v is None:
        return float("nan")
    return float(v)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved: {path}")


def _avg_family_raw_total_over_levels(
    summary: dict, model: str, env: str, section: str, field: str
) -> float:
    """Sum raw_total over all levels for a given model/env combo."""
    mel = summary["groupings"]["by_model_env_level"]
    prefix = f"{model}/{env}/"
    total = 0.0
    for k, data in mel.items():
        if k.startswith(prefix):
            total += _raw_total(data, section, field)
    return total


def _avg_metric_over_levels(summary: dict, model: str, env: str, field: str) -> float:
    """Average a metric over all levels for a given model/env combo."""
    mel = summary["groupings"]["by_model_env_level"]
    prefix = f"{model}/{env}/"
    vals = []
    for k, data in mel.items():
        if k.startswith(prefix):
            v = _metric(data, field)
            if not np.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _get_model_env_combos(summary: dict) -> list[tuple[str, str]]:
    """Return sorted list of (model, env) tuples present in the data."""
    combos = set()
    for k in summary["groupings"]["by_model_env_level"]:
        parts = k.split("/")
        combos.add((parts[0], parts[1]))
    return sorted(combos)


def plot_family_delta_model(summary: dict, out: Path) -> None:
    """
    Diverging bar: antipattern_family_global + subgraph_family_global,
    comparing two models by raw_total difference.
    Positive = model_a higher, negative = model_b higher.
    """
    models = list(summary["groupings"]["by_model"].keys())
    if len(models) < 2:
        logger.info("  skipped (need 2 models)")
        return
    model_a, model_b = models[0], models[1]
    data_a = summary["groupings"]["by_model"][model_a]
    data_b = summary["groupings"]["by_model"][model_b]

    labels = []
    deltas = []
    for f in ANTIPATTERN_FAMILY_FIELDS:
        va = _raw_total(data_a, "antipattern_family_global", f)
        vb = _raw_total(data_b, "antipattern_family_global", f)
        labels.append(short_name(f))
        deltas.append(va - vb)
    for f in SUBGRAPH_FAMILY_FIELDS:
        va = _raw_total(data_a, "subgraph_family_global", f)
        vb = _raw_total(data_b, "subgraph_family_global", f)
        labels.append(short_name(f))
        deltas.append(va - vb)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1.1))
    y = np.arange(len(labels))
    color_a = MODEL_COLORS.get(model_a, "gray")
    color_b = MODEL_COLORS.get(model_b, "gray")
    colors = [color_a if d >= 0 else color_b for d in deltas]
    for i, (delta, color) in enumerate(zip(deltas, colors, strict=False)):
        ax.hlines(y[i], 0, delta, color=color, alpha=0.5, linewidth=5)
        ax.plot(delta, y[i], "o", markersize=5, color=color, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"← {model_b} higher | {model_a} higher →\n(raw_total difference)")
    range_frame(ax, np.array([min(deltas), max(deltas)]), y, nice=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "family_delta_model.pdf")


def plot_family_trends_by_env(summary: dict, out: Path) -> None:
    """
    Grouped bar chart: x-axis = environments,
    different colours = family elements (anti-pattern + subgraph families).
    Values are raw_total occurrence counts.
    """
    envs = list(summary["groupings"]["by_env"].keys())
    all_fields = [
        (f, "antipattern_family_global") for f in ANTIPATTERN_FAMILY_FIELDS
    ] + [(f, "subgraph_family_global") for f in SUBGRAPH_FAMILY_FIELDS]

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1.2))
    x = np.arange(len(envs))
    n_groups = len(all_fields)
    bar_width = 0.8 / n_groups
    cmap = plt.get_cmap("tab10")

    for i, (field, section) in enumerate(all_fields):
        vals = [
            _raw_total(summary["groupings"]["by_env"][env], section, field)
            for env in envs
        ]
        offset = (i - (n_groups - 1) / 2) * bar_width
        ax.vlines(
            x + offset,
            0,
            vals,
            label=short_name(field),
            color=cmap(i),
            alpha=0.5,
            linewidth=5,
        )
        ax.plot(x + offset, vals, "o", markersize=5, color=cmap(i), alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Raw total occurrences")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    all_vals = [
        _raw_total(summary["groupings"]["by_env"][env], section, field)
        for field, section in all_fields
        for env in envs
    ]
    range_frame(ax, x, np.array([0.0, max(all_vals)]), pad=0.1, pad_x=0.0)
    fig.tight_layout()
    _save(fig, out / "family_trends_by_env.pdf")


def plot_family_patterns_by_model(summary: dict, out: Path) -> None:
    """Grouped horizontal bars: family raw_total for each model."""
    models = list(summary["groupings"]["by_model"].keys())
    all_fields = [
        (f, "antipattern_family_global") for f in ANTIPATTERN_FAMILY_FIELDS
    ] + [(f, "subgraph_family_global") for f in SUBGRAPH_FAMILY_FIELDS]
    field_labels = [short_name(f) for f, _ in all_fields]

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1.1))
    y = np.arange(len(all_fields))
    bar_height = 0.35

    for i, model in enumerate(models):
        data = summary["groupings"]["by_model"][model]
        vals = [_raw_total(data, section, f) for f, section in all_fields]
        offset = (i - (len(models) - 1) / 2) * bar_height
        ax.hlines(
            y + offset,
            0,
            vals,
            label=model,
            color=MODEL_COLORS.get(model, f"C{i}"),
            alpha=0.5,
            linewidth=5,
        )
        ax.plot(
            vals,
            y + offset,
            "o",
            markersize=5,
            color=MODEL_COLORS.get(model, f"C{i}"),
            alpha=0.6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(field_labels, fontsize=7)
    ax.set_xlabel("Raw total occurrences")
    ax.legend(fontsize=8)
    all_vals = [
        _raw_total(summary["groupings"]["by_model"][m], section, f)
        for f, section in all_fields
        for m in models
    ]
    range_frame(ax, np.array([0.0, max(all_vals)]), y)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "family_patterns_by_model.pdf")


def plot_good_bad_balance_by_env(summary: dict, out: Path) -> None:
    """
    Stacked horizontal bars: for each environment,
    total good-workflow raw_total vs total anti-pattern raw_total.
    """
    envs = list(summary["groupings"]["by_env"].keys())
    good_totals = []
    bad_totals = []
    for env in envs:
        data = summary["groupings"]["by_env"][env]
        g_vals = [
            _raw_total(data, "subgraph_family_global", f)
            for f in SUBGRAPH_FAMILY_FIELDS
        ]
        b_vals = [
            _raw_total(data, "antipattern_family_global", f)
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        good_totals.append(sum(g_vals))
        bad_totals.append(sum(b_vals))

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.2))
    y = np.arange(len(envs))
    ax.barh(
        y, good_totals, color=GOOD_COLOR, alpha=0.8, label="Good workflow (total count)"
    )
    ax.barh(
        y,
        bad_totals,
        left=good_totals,
        color=BAD_COLOR,
        alpha=0.8,
        label="Anti-pattern (total count)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(envs, fontsize=7)
    ax.set_xlabel("Raw total occurrences")
    ax.legend(fontsize=7, loc="lower right")
    totals = np.array(good_totals) + np.array(bad_totals)
    range_frame(ax, np.array([0.0, float(totals.max())]), y, nice=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "good_bad_balance_by_env.pdf")


def plot_overall_families_ranked(summary: dict, out: Path) -> None:
    """
    Single horizontal bar chart with ALL family labels ranked by raw_total.
    Good families in blue, anti-pattern families in red.
    """
    overall = summary["groupings"]["overall"]

    items = [
        (short_name(f), _raw_total(overall, "subgraph_family_global", f), "good")
        for f in SUBGRAPH_FAMILY_FIELDS
    ]
    items.extend(
        (short_name(f), _raw_total(overall, "antipattern_family_global", f), "bad")
        for f in ANTIPATTERN_FAMILY_FIELDS
    )

    items.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 0.8))
    y = np.arange(len(items))
    vals_arr = np.array([v for _, v, _ in items])
    for i, (_name, val, kind) in enumerate(items):
        color = GOOD_COLOR if kind == "good" else BAD_COLOR
        ax.hlines(i, 0, val, color=color, alpha=0.5, linewidth=5)
        ax.plot(val, i, "o", markersize=5, color=color, alpha=0.6)
        ax.text(
            val + vals_arr.max() * 0.01,
            i,
            f"{val:.0f}",
            va="center",
            fontsize=6,
            color=color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([item[0] for item in items], fontsize=7)
    ax.set_xlabel("Raw total occurrences")
    legend_elements = [
        Patch(facecolor=GOOD_COLOR, alpha=0.75, label="Good workflow family"),
        Patch(facecolor=BAD_COLOR, alpha=0.75, label="Anti-pattern family"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    range_frame(ax, np.array([0.0, float(vals_arr.max())]), y, nice=False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "overall_families_ranked.pdf")


def plot_radar_envs(summary: dict, out: Path) -> None:
    """
    Single radar chart with 6 edges (all family labels).
    Each environment is a series. Values are raw_total occurrence counts
    normalised per chart to [0, 1] for shape comparison.
    """
    envs = list(summary["groupings"]["by_env"].keys())
    labels = [short_name(f) for f in ALL_FAMILY_FIELDS]
    n = len(ALL_FAMILY_FIELDS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    all_raw: list[float] = []
    for env in envs:
        data = summary["groupings"]["by_env"][env]
        all_raw.extend(
            [
                _raw_total(data, "antipattern_family_global", f)
                for f in ANTIPATTERN_FAMILY_FIELDS
            ]
        )
        all_raw.extend(
            [
                _raw_total(data, "subgraph_family_global", f)
                for f in SUBGRAPH_FAMILY_FIELDS
            ]
        )
    global_max = max(all_raw) if all_raw else 1.0

    cmap = plt.get_cmap("tab10")
    env_colors = {env: cmap(i) for i, env in enumerate(envs)}

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH * 0.7, TWO_COL_WIDTH * 0.7),
        subplot_kw={"polar": True},
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=6)
    ax.set_ylim(0, global_max)
    tick_vals = np.linspace(0, global_max, 5)[1:]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f"{v:.0f}" for v in tick_vals], fontsize=5, alpha=0.5)

    for env in envs:
        data = summary["groupings"]["by_env"][env]
        vals = [
            _raw_total(data, "antipattern_family_global", f)
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        vals.extend(
            _raw_total(data, "subgraph_family_global", f)
            for f in SUBGRAPH_FAMILY_FIELDS
        )
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, linewidth=1.5, label=env, color=env_colors[env])
        ax.fill(angles, vals_closed, alpha=0.08, color=env_colors[env])

    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    _save(fig, out / "radar_families_by_env.pdf")


def plot_scientificness_vs_antipatterns_env(summary: dict, out: Path) -> None:
    """
    Scatter: each point is a model/env combination (summed over levels).
    x = total anti-pattern raw_total, y = scientificness score (averaged over levels).
    """
    combos = _get_model_env_combos(summary)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH * 1.5, ONE_COL_HEIGHT * 1.5))
    cmap = plt.get_cmap("tab10")
    envs_seen = {}
    env_idx = 0
    markers = {"claude_sonnet_45": "o", "gpt_4o": "s"}
    all_x: list[float] = []
    all_y: list[float] = []

    for model, env in combos:
        if env not in envs_seen:
            envs_seen[env] = cmap(env_idx)
            env_idx += 1
        color = envs_seen[env]
        marker = markers.get(model, "^")

        bad_totals = [
            _avg_family_raw_total_over_levels(
                summary, model, env, "antipattern_family_global", f
            )
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        total_bad = sum(bad_totals)
        ss = _avg_metric_over_levels(summary, model, env, "scientificness_score")

        ax.scatter(
            total_bad,
            ss,
            color=color,
            marker=marker,
            s=60,
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            f"{model}/{env}",
            (total_bad, ss),
            fontsize=4.5,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )
        all_x.append(float(total_bad))
        if not np.isnan(ss):
            all_y.append(float(ss))

    env_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=6, label=e
        )
        for e, c in envs_seen.items()
    ]
    model_handles = [
        plt.Line2D(
            [0], [0], marker=m, color="gray", markersize=6, linestyle="None", label=mod
        )
        for mod, m in markers.items()
    ]
    ax.legend(
        handles=env_handles + model_handles,
        fontsize=5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    ax.set_xlabel("Total anti-pattern occurrences (raw_total sum)")
    ax.set_ylabel("Scientificness score")
    if all_x and all_y:
        range_frame(ax, np.array(all_x), np.array(all_y))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, out / "scientificness_vs_antipatterns_env.pdf")


ALL_PLOTS = [
    ("01", "family_delta_model", plot_family_delta_model),
    ("02", "family_trends_by_env", plot_family_trends_by_env),
    ("03", "family_patterns_by_model", plot_family_patterns_by_model),
    ("04", "good_bad_balance_by_env", plot_good_bad_balance_by_env),
    ("05", "overall_families_ranked", plot_overall_families_ranked),
    ("06", "radar_families_by_env", plot_radar_envs),
    (
        "07",
        "scientificness_vs_antipatterns_env",
        plot_scientificness_vs_antipatterns_env,
    ),
]


def main(
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    only: str | None = None,
) -> None:
    """Generate family-level raw-total plots from the annotation summary.

    The command loads the aggregated reasoning summary, creates the output
    directory if needed, and renders the selected plot set using absolute
    family occurrence counts.

    Args:
        summary_path: Path to the annotation_summary.json file produced by
            reasoning report analysis.
        output_dir: Directory where generated figures will be written.
        only: Optional comma-separated list of plot numbers to generate, such
            as "01,03". When omitted, all plots are rendered.
    """
    summary = load_summary(Path(summary_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected = None
    if only:
        selected = {item.strip() for item in only.split(",") if item.strip()}

    logger.info(f"Generating raw_total plots in {out} ...")
    for num, name, func in ALL_PLOTS:
        if selected and num not in selected:
            continue
        logger.info(f"  [{num}] {name}")
        func(summary, out)

    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
