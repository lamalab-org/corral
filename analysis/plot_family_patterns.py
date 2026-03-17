"""
Plot anti-pattern and good-workflow *family*-level summaries from reasoning_reports.

All plots use family-aggregated labels (evidence_handling, hypothesis_testing,
discovery, …) instead of the individual pattern names used in
reports/plots/plot_antipatterns.py.

Usage:
  python analysis/plot_family_patterns.py
  python analysis/plot_family_patterns.py --summary-path /path/to/summary.json
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from loguru import logger
from matplotlib.patches import Patch
from scipy.constants import golden

ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

lama_aesthetics.get_style("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "reasoning_reports" / "analysis" / "annotation_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "figures"

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
NEGATIVE_PATTERN_LABEL = "Reasoning breakdowns"


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


def _frac(data: dict, section: str, field: str) -> float:
    v = data.get(section, {}).get(field, {}).get("fraction")
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


def _avg_family_frac_over_levels(
    summary: dict, model: str, env: str, section: str, field: str
) -> float:
    """Average a family fraction over all levels for a given model/env combo."""
    mel = summary["groupings"]["by_model_env_level"]
    prefix = f"{model}/{env}/"
    vals = []
    for k, data in mel.items():
        if k.startswith(prefix):
            vals.append(_frac(data, section, field))
    return float(np.mean(vals)) if vals else 0.0


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
    averaged over all envs and levels, comparing two models.
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
        va = _frac(data_a, "antipattern_family_global", f)
        vb = _frac(data_b, "antipattern_family_global", f)
        labels.append(short_name(f))
        deltas.append(va - vb)
    for f in SUBGRAPH_FAMILY_FIELDS:
        va = _frac(data_a, "subgraph_family_global", f)
        vb = _frac(data_b, "subgraph_family_global", f)
        labels.append(short_name(f))
        deltas.append(va - vb)

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.1)
    )
    y = np.arange(len(labels))
    color_a = MODEL_COLORS.get(model_a, "gray")
    color_b = MODEL_COLORS.get(model_b, "gray")
    colors = [color_a if d >= 0 else color_b for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"← {model_b} worse | {model_a} worse →")
    ax.set_title(f"Family Pattern Delta: {model_a} - {model_b}")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "family_delta_model.pdf")


def plot_family_trends_by_env(summary: dict, out: Path) -> None:
    """
    Grouped bar chart: x-axis = environments (averaged over levels),
    different colours = family elements (anti-pattern + subgraph families).
    """
    envs = list(summary["groupings"]["by_env"].keys())
    all_fields = [
        (f, "antipattern_family_global") for f in ANTIPATTERN_FAMILY_FIELDS
    ] + [(f, "subgraph_family_global") for f in SUBGRAPH_FAMILY_FIELDS]

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.2)
    )
    x = np.arange(len(envs))
    n_groups = len(all_fields)
    bar_width = 0.8 / n_groups
    cmap = plt.get_cmap("tab10")

    for i, (field, section) in enumerate(all_fields):
        vals = [
            _frac(summary["groupings"]["by_env"][env], section, field) for env in envs
        ]
        offset = (i - (n_groups - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=short_name(field),
            color=cmap(i),
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Fraction of traces")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Family Pattern Trends by Environment")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    _save(fig, out / "family_trends_by_env.pdf")


def plot_family_patterns_by_model(summary: dict, out: Path) -> None:
    """Grouped horizontal bars: family fraction for each model."""
    models = list(summary["groupings"]["by_model"].keys())
    all_fields = [
        (f, "antipattern_family_global") for f in ANTIPATTERN_FAMILY_FIELDS
    ] + [(f, "subgraph_family_global") for f in SUBGRAPH_FAMILY_FIELDS]
    field_labels = [short_name(f) for f, _ in all_fields]

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.1)
    )
    y = np.arange(len(all_fields))
    bar_height = 0.35

    for i, model in enumerate(models):
        data = summary["groupings"]["by_model"][model]
        vals = [_frac(data, section, f) for f, section in all_fields]
        offset = (i - (len(models) - 1) / 2) * bar_height
        ax.barh(
            y + offset,
            vals,
            bar_height * 0.9,
            label=model,
            color=MODEL_COLORS.get(model, f"C{i}"),
            alpha=0.8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(field_labels, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction of traces")
    ax.set_title("Family Patterns by Model")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "family_patterns_by_model.pdf")


def plot_good_bad_balance_by_env(summary: dict, out: Path) -> None:
    """
    Stacked horizontal bars: for each environment (averaged over levels),
    mean good-workflow family fraction vs mean anti-pattern family fraction.
    """
    envs = list(summary["groupings"]["by_env"].keys())
    good_means = []
    bad_means = []
    for env in envs:
        data = summary["groupings"]["by_env"][env]
        g_vals = [
            _frac(data, "subgraph_family_global", f) for f in SUBGRAPH_FAMILY_FIELDS
        ]
        b_vals = [
            _frac(data, "antipattern_family_global", f)
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        good_means.append(np.mean(g_vals))
        bad_means.append(np.mean(b_vals))

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.2)
    )
    y = np.arange(len(envs))
    ax.barh(
        y, good_means, color=GOOD_COLOR, alpha=0.8, label="Good workflow (mean frac.)"
    )
    ax.barh(
        y,
        bad_means,
        left=good_means,
        color=BAD_COLOR,
        alpha=0.8,
        label="Anti-pattern (mean frac.)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(envs, fontsize=7)
    ax.set_xlabel("Mean fraction of traces")
    ax.set_title("Good vs Bad Reasoning Balance by Environment")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "good_bad_balance_by_env.pdf")


def plot_overall_families_ranked(summary: dict, out: Path) -> None:
    """
    Single horizontal bar chart with ALL family labels ranked by prevalence.
    Good families in blue, anti-pattern families in red.
    """
    overall = summary["groupings"]["overall"]

    items = [
        (short_name(f), _frac(overall, "subgraph_family_global", f), "good")
        for f in SUBGRAPH_FAMILY_FIELDS
    ]
    items.extend(
        (short_name(f), _frac(overall, "antipattern_family_global", f), "bad")
        for f in ANTIPATTERN_FAMILY_FIELDS
    )

    items.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH * 0.8)
    )
    y = np.arange(len(items))
    for i, (_name, val, kind) in enumerate(items):
        color = GOOD_COLOR if kind == "good" else BAD_COLOR
        ax.barh(i, val, color=color, alpha=0.75)
        ax.text(val + 0.01, i, f"{val:.0%}", va="center", fontsize=6, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels([item[0] for item in items], fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Fraction of traces")
    ax.set_title("Overall: All Pattern Families Ranked by Prevalence")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    legend_elements = [
        Patch(facecolor=GOOD_COLOR, alpha=0.75, label="Good workflow family"),
        Patch(facecolor=BAD_COLOR, alpha=0.75, label="Anti-pattern family"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out / "overall_families_ranked.pdf")


def plot_radar_envs(summary: dict, out: Path) -> None:
    """
    Single radar chart with 6 edges (all family labels).
    Each environment is a series (averaged over all levels).
    """
    envs = list(summary["groupings"]["by_env"].keys())
    labels = [short_name(f) for f in ALL_FAMILY_FIELDS]
    n = len(ALL_FAMILY_FIELDS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    cmap = plt.get_cmap("tab10")
    env_colors = {env: cmap(i) for i, env in enumerate(envs)}

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH_INCH * 0.7, TWO_COL_WIDTH_INCH * 0.7),
        subplot_kw={"polar": True},
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=5, alpha=0.5)

    for env in envs:
        data = summary["groupings"]["by_env"][env]
        vals = [
            _frac(data, "antipattern_family_global", f)
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        vals.extend(
            _frac(data, "subgraph_family_global", f) for f in SUBGRAPH_FAMILY_FIELDS
        )
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, linewidth=1.5, label=env, color=env_colors[env])
        ax.fill(angles, vals_closed, alpha=0.08, color=env_colors[env])

    ax.set_title("Family Profiles by Environment", fontsize=9, pad=20)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    _save(fig, out / "radar_families_by_env.pdf")


def plot_scientificness_vs_antipatterns_env(summary: dict, out: Path) -> None:
    """
    Scatter: each point is a model/env combination (averaged over levels).
    x = mean anti-pattern family fraction, y = scientificness score.
    """
    combos = _get_model_env_combos(summary)

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH_INCH * 1.5, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH * 1.5)
    )
    cmap = plt.get_cmap("tab10")
    envs_seen = {}
    env_idx = 0
    markers = {"claude_sonnet_45": "o", "gpt_4o": "s"}

    for model, env in combos:
        if env not in envs_seen:
            envs_seen[env] = cmap(env_idx)
            env_idx += 1
        color = envs_seen[env]
        marker = markers.get(model, "^")

        bad_fracs = [
            _avg_family_frac_over_levels(
                summary, model, env, "antipattern_family_global", f
            )
            for f in ANTIPATTERN_FAMILY_FIELDS
        ]
        mean_bad = np.mean(bad_fracs)
        ss = _avg_metric_over_levels(summary, model, env, "scientificness_score")

        ax.scatter(
            mean_bad,
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
            (mean_bad, ss),
            fontsize=4.5,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Build legend entries for envs and models
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

    ax.set_xlabel("Mean anti-pattern family fraction")
    ax.set_ylabel("Scientificness score")
    ax.set_title("Scientificness vs Anti-pattern (by Env & Model)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to annotation_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated plot numbers to generate (e.g. '01,03'). Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_path)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.only:
        selected = set(args.only.split(","))

    logger.info(f"Generating plots in {out} ...")
    for num, name, func in ALL_PLOTS:
        if selected and num not in selected:
            continue
        logger.info(f"  [{num}] {name}")
        func(summary, out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
