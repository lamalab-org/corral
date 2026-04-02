"""
Plot individual anti-pattern and productive-pattern prevalences, grouped by family,
with a coupled heatmap showing prevalence across reasoning-type groups.

Left panel:  horizontal bar chart averaged over all models, domains, and levels.
Right panel: heatmap with one column per environment group (workflow execution,
             strategic reasoning, hypothesis-driven enquiry).

Usage:
  python analysis/plot_overall_pattern_bars.py
  python analysis/plot_overall_pattern_bars.py --summary-path /path/to/summary.json
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
from lama_aesthetics import ONE_COL_WIDTH
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
    fig.savefig(path, format=path.suffix.lstrip("."), bbox_inches="tight")
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
    """Return a list of row dicts in display order (top→bottom).

    Two sections: Productive Motifs, then Reasoning Breakdowns.
    Each dict has keys: label, value, color, kind ('header' | 'bar'),
    and for bars: pattern_key, section.
    """
    # Collect all productive subgraph patterns across families
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

    # Collect all antipatterns across families
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

    n_rows = len(rows)
    row_height = 0.22
    header_extra = 0.18
    row_step = 0.32
    fig_height = n_rows * row_height + sum(
        header_extra for r in rows if r["kind"] == "header"
    )
    fig_width = ONE_COL_WIDTH * 1.25
    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
    )

    bar_height = 0.15
    rounding = 0.20
    y_pos = 0.0
    y_positions = []

    for row in rows:
        if row["kind"] == "header":
            y_pos -= header_extra
        y_positions.append(y_pos)
        y_pos -= row_step

    y_arr = np.array(y_positions)
    x_lim = 100.0

    y_range = (y_arr.max() + 0.5) - (y_arr.min() - 0.5)
    bar_ax_width = fig_width / 2  # approximate bar-axis width
    mutation_aspect = (bar_ax_width * y_range) / (fig_height * x_lim)

    for y, row in zip(y_arr, rows, strict=False):
        if row["kind"] == "header":
            ax_bar.text(
                -0.35,
                y,
                row["label"],
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                transform=ax_bar.get_yaxis_transform(),
            )
        else:
            val = row["value"]
            pct = val * 100
            color = row["color"]
            if pct > 0:
                box = FancyBboxPatch(
                    (0, y - bar_height / 2),
                    pct,
                    bar_height,
                    boxstyle=f"round,pad=0,rounding_size={rounding}",
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.85,
                    mutation_aspect=mutation_aspect,
                )
                ax_bar.add_patch(box)
            ax_bar.text(
                pct + 0.8,
                y,
                f"{val:.0%}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )
            ax_bar.text(
                -0.01,
                y,
                row["label"],
                ha="right",
                va="center",
                fontsize=10,
                transform=ax_bar.get_yaxis_transform(),
            )

    ax_bar.set_xlim(0, x_lim)
    ax_bar.set_ylim(y_arr.min() - 0.5, y_arr.max() + 0.15)
    ax_bar.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax_bar.set_yticks([])
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.tick_params(axis="x", labelsize=9)

    group_names = list(ENV_GROUPS.keys())
    n_groups = len(group_names)

    # Build matrix: one row per bar-row, one col per env group
    bar_indices = [i for i, r in enumerate(rows) if r["kind"] == "bar"]
    heat_matrix = np.full((len(bar_indices), n_groups), np.nan)

    for ri, idx in enumerate(bar_indices):
        row = rows[idx]
        for gi, gname in enumerate(group_names):
            heat_matrix[ri, gi] = _group_frac(
                by_env,
                ENV_GROUPS[gname],
                row["section"],
                row["pattern_key"],
            )

    # Grey-scale colormap: white at 0, dark grey at 1
    cmap = LinearSegmentedColormap.from_list(
        "heat",
        ["#ffffff", "#999999", "#333333"],
        N=256,
    )

    # Map bar y-positions to heatmap cell edges
    bar_y = y_arr[bar_indices]
    cell_h = row_step
    col_w = 1.0

    ax_heat.set_xlim(0, n_groups * col_w)
    ax_heat.set_ylim(y_arr.min() - 0.5, y_arr.max() + 0.15)

    vmin, vmax = (
        0.0,
        float(np.nanmax(heat_matrix)) if np.nanmax(heat_matrix) > 0 else 1.0,
    )

    for ri, y in enumerate(bar_y):
        for gi in range(n_groups):
            val = heat_matrix[ri, gi]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            fc = cmap(norm_val)
            rect = plt.Rectangle(
                (gi * col_w, y - cell_h / 2),
                col_w,
                cell_h,
                facecolor=fc,
                edgecolor="none",
            )
            ax_heat.add_patch(rect)
            # Text label inside cell
            text_color = "white" if norm_val > 0.55 else "#333333"
            ax_heat.text(
                gi * col_w + col_w / 2,
                y,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    # Column headers at the bottom, rotated
    for gi, gname in enumerate(group_names):
        ax_heat.text(
            gi * col_w + col_w / 2,
            y_arr.min() - 0.45,
            gname,
            ha="right",
            va="top",
            fontsize=7,
            fontweight="bold",
            rotation=45,
            rotation_mode="anchor",
        )

    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    ax_heat.spines["left"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["bottom"].set_visible(False)

    fig.tight_layout()

    _save(fig, out / "overall_pattern_bars.pdf")


def main(
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    """Generate the aggregated pattern bar chart with heatmap.

    Args:
        summary_path: Path to the annotation_summary.json file.
        output_dir: Directory where the figure will be written.
    """
    summary = load_summary(Path(summary_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating overall pattern bar chart + heatmap in {out} ...")
    plot(summary, out)
    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
