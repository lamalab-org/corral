"""
Plot individual anti-pattern and productive-pattern prevalences, grouped by family,
with a coupled heatmap showing prevalence across reasoning-type groups.

Top panel:   heatmap with one row per environment group (workflow execution,
             strategic reasoning, hypothesis-driven enquiry).
Bottom panel: vertical bar chart averaged over all models, domains, and levels.

Horizontal variant of plot_overall_pattern_bars.py.

Usage:
  python analysis/plot_overall_pattern_bars_horizontal.py
  python analysis/plot_overall_pattern_bars_horizontal.py --summary-path /path/to/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

lama_aesthetics.get_style("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "reasoning_reports" / "analysis" / "annotation_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "figures" / "fig_epistemology"

# Pattern families must stay synchronized with analyze.py
ANTIPATTERN_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "untested_claim",
        "contradiction_without_repair",
        "one_sided_confirmation",
    ],
    "evidence_handling": [
        "evidence_non_uptake",
        "disconnected_evidence",
        "unsupported_judgment",
        "uninformative_test",
    ],
    "experimental_strategy": [
        "stalled_revision",
        "fixed_belief_trace",
        "premature_commitment",
    ],
}

SUBGRAPH_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "refutation_driven_belief_revision",
        "hypothesis_reranking",
        "evidence_led_hypothesis_generation",
    ],
    "evidence_handling": [
        "convergent_multi_test_evidence",
        "explore_then_test_transition",
    ],
    "experimental_strategy": [
        "fixed_hypothesis_test_tuning",
        "precommitted_test_plan",
        "evidence_guided_test_redesign",
    ],
}

FAMILY_ORDER = ["hypothesis_generation", "evidence_handling", "experimental_strategy"]

GOOD_COLOR = "#1a5276"
BAD_COLOR = "#c0392b"

FAMILY_DISPLAY = {
    "hypothesis_generation": "Hypothesis Generation",
    "evidence_handling": "Evidence Handling",
    "experimental_strategy": "Experimental Strategy",
}

PATTERN_SHORT: dict[str, str] = {
    # Antipatterns
    "untested_claim": "Untested Claim",
    "contradiction_without_repair": "Contrad. w/o Repair",
    "one_sided_confirmation": "One-Sided Confirm.",
    "evidence_non_uptake": "Evidence Non-Uptake",
    "disconnected_evidence": "Disconnected Evid.",
    "unsupported_judgment": "Unsupported Judgment",
    "uninformative_test": "Uninformative Test",
    "stalled_revision": "Stalled Revision",
    "fixed_belief_trace": "Fixed Belief Trace",
    "premature_commitment": "Premature Commit.",
    # Productive subgraphs
    "refutation_driven_belief_revision": "Refutation-Driven Rev.",
    "hypothesis_reranking": "Hypothesis Reranking",
    "evidence_led_hypothesis_generation": "Evidence-Led Hyp. Gen.",
    "convergent_multi_test_evidence": "Convergent Multi-Test",
    "explore_then_test_transition": "Explore→Test Trans.",
    "fixed_hypothesis_test_tuning": "Fixed-Hyp. Test Tuning",
    "precommitted_test_plan": "Precommitted Plan",
    "evidence_guided_test_redesign": "Evidence-Guided Redesign",
}

ENV_GROUPS: dict[str, list[str]] = {
    "Workflow": ["ml", "afm", "catalyst", "md"],
    "Strategic": ["retrosynthesis"],
    "Hyp.-Driven": ["spectra", "wetlab", "resistor"],
}


def _pretty(key: str) -> str:
    """Turn a snake_case pattern key into a readable label."""
    return PATTERN_SHORT.get(key, key.replace("_", " ").title())


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frac(data: dict, section: str, field: str) -> float:
    v = data.get(section, {}).get(field, {}).get("fraction")
    if v is None:
        return 0.0
    return float(v)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, format=path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    logger.info(f"  saved: {path}")


def _group_frac(by_env: dict, envs: list[str], section: str, field: str) -> float:
    """Weighted-average fraction across *envs* (weighted by n_traces)."""
    total_traces = 0
    weighted_sum = 0.0
    for env in envs:
        env_data = by_env.get(env)
        if env_data is None:
            continue
        n = env_data.get("n_traces", 0)
        frac = _frac(env_data, section, field)
        weighted_sum += frac * n
        total_traces += n
    if total_traces == 0:
        return 0.0
    return weighted_sum / total_traces


def _build_rows(overall: dict) -> list[dict]:
    """Return a list of row dicts in display order.

    Two sections: Productive Motifs, then Reasoning Breakdowns.
    Each dict has keys: label, value, color, kind ('header' | 'bar'),
    and for bars: pattern_key, section.
    """
    all_subgraphs = [
        {
            "label": _pretty(pat),
            "value": _frac(overall, "subgraph_presence_global", pat),
            "color": GOOD_COLOR,
            "kind": "bar",
            "pattern_key": pat,
            "section": "subgraph_presence_global",
        }
        for fam in FAMILY_ORDER
        for pat in SUBGRAPH_FAMILIES[fam]
    ]
    all_subgraphs.sort(key=lambda r: r["value"], reverse=True)

    all_antipatterns = [
        {
            "label": _pretty(pat),
            "value": _frac(overall, "antipattern_presence_global", pat),
            "color": BAD_COLOR,
            "kind": "bar",
            "pattern_key": pat,
            "section": "antipattern_presence_global",
        }
        for fam in FAMILY_ORDER
        for pat in ANTIPATTERN_FAMILIES[fam]
    ]
    all_antipatterns.sort(key=lambda r: r["value"], reverse=True)

    rows: list[dict] = []
    rows.append({"label": "Productive Motifs", "kind": "header"})
    rows.extend(all_subgraphs)
    rows.append({"label": "Reasoning Breakdowns", "kind": "header"})
    rows.extend(all_antipatterns)
    return rows


def plot(summary: dict, out: Path) -> None:
    overall = summary["groupings"]["overall"]
    by_env = summary["groupings"]["by_env"]
    rows = _build_rows(overall)

    bar_rows: list[dict] = []
    x_positions: list[float] = []
    section_spans: list[tuple[float, float, str]] = []  # (x_start, x_end, label)
    x = 0.0
    sec_start = 0.0
    sec_label = ""

    for row in rows:
        if row["kind"] == "header":
            # close previous section span if it had bars
            if bar_rows and sec_label:
                section_spans.append((sec_start, x_positions[-1], sec_label))
            if bar_rows:
                x += 1.5  # gap between sections
            sec_start = x
            sec_label = row["label"]
        else:
            bar_rows.append(row)
            x_positions.append(x)
            x += 1.0

    if bar_rows and sec_label:
        section_spans.append((sec_start, x_positions[-1], sec_label))

    x_arr = np.array(x_positions)
    n_bars = len(bar_rows)

    group_names = list(ENV_GROUPS.keys())
    n_groups = len(group_names)

    fig, (ax_heat, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 0.7),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    bar_width = 0.7
    rounding = 0.15
    values = [r["value"] * 100 for r in bar_rows]
    colors = [r["color"] for r in bar_rows]
    max_val = max(values) if values else 1.0

    # Approximate the axis aspect ratio so FancyBboxPatch renders rounded corners proportionally in display space
    x_range = (x_arr[-1] + 0.8) - (x_arr[0] - 0.8)
    bar_ax_height = TWO_COL_WIDTH * (2.5 / 3.5)  # approximate bar-axis height
    mutation_aspect = (bar_ax_height * x_range) / (TWO_COL_HEIGHT * max_val * 1.15)

    for xp, val, color in zip(x_arr, values, colors, strict=False):
        if val > 0:
            box = FancyBboxPatch(
                (xp - bar_width / 2, 0),
                bar_width,
                val,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                mutation_aspect=mutation_aspect,
            )
            ax_bar.add_patch(box)
        ax_bar.text(
            xp,
            val + 0.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )

    ax_bar.set_xlim(x_arr[0] - 0.8, x_arr[-1] + 0.8)
    ax_bar.set_ylim(0, max_val * 1.15)
    ax_bar.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=100, decimals=0),
    )
    ax_bar.set_xticks(x_arr)
    ax_bar.set_xticklabels(
        [r["label"] for r in bar_rows],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="y", labelsize=10)

    for sec_start, sec_end, label in section_spans:
        mid = (sec_start + sec_end) / 2
        ax_bar.text(
            mid,
            max_val * 1.05,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    heat_matrix = np.full((n_groups, n_bars), np.nan)
    for bi, brow in enumerate(bar_rows):
        for gi, gname in enumerate(group_names):
            heat_matrix[gi, bi] = _group_frac(
                by_env,
                ENV_GROUPS[gname],
                brow["section"],
                brow["pattern_key"],
            )

    cmap = LinearSegmentedColormap.from_list(
        "heat",
        ["#ffffff", "#999999", "#333333"],
        N=256,
    )
    vmin = 0.0
    vmax = float(np.nanmax(heat_matrix)) if np.nanmax(heat_matrix) > 0 else 1.0

    cell_w = 1.0
    cell_h = 1.0

    for gi in range(n_groups):
        for bi in range(n_bars):
            val = heat_matrix[gi, bi]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            fc = cmap(norm_val)
            rect = plt.Rectangle(
                (x_arr[bi] - cell_w / 2, gi - cell_h / 2),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_heat.add_patch(rect)
            text_color = "white" if norm_val > 0.55 else "#333333"
            ax_heat.text(
                x_arr[bi],
                gi,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    ax_heat.set_ylim(-0.5, n_groups - 0.5)
    ax_heat.set_xlim(x_arr[0] - 0.8, x_arr[-1] + 0.8)
    ax_heat.set_yticks(range(n_groups))
    ax_heat.set_yticklabels(group_names, fontsize=10, fontweight="bold")
    ax_heat.invert_yaxis()
    ax_heat.set_xticks([])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(left=False, bottom=False)

    fig.subplots_adjust(bottom=0.28, left=0.18, right=0.97, top=0.98)

    _save(fig, out / "overall_pattern_bars_horizontal.pdf")


def main(
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    """Generate the aggregated pattern bar chart (horizontal) with heatmap.

    Args:
        summary_path: Path to the annotation_summary.json file.
        output_dir: Directory where the figure will be written.
    """
    summary = load_summary(Path(summary_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating horizontal pattern bar chart + heatmap in {out} ...")
    plot(summary, out)
    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
