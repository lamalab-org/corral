"""
Plot individual anti-pattern and productive-pattern prevalences, grouped by family.

Produces a single horizontal bar chart averaged over **all models, domains, and
levels** (the ``overall`` grouping from the annotation summary).  Families appear
as bold headers on the left; within each family every constituent pattern is shown
as a rounded rectangular bar with the percentage label at the end.

Usage:
  python analysis/plot_overall_pattern_bars.py
  python analysis/plot_overall_pattern_bars.py --summary-path /path/to/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from lama_aesthetics import TWO_COL_WIDTH
from loguru import logger
from matplotlib.patches import FancyBboxPatch, Patch

lama_aesthetics.get_style("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "reasoning_reports" / "analysis" / "annotation_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "figures"

# ── Family → individual pattern mapping (mirrors analyze.py) ──────────────
ANTIPATTERN_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "untested_hypothesis",
        "unresolved_contradiction",
        "confirmation_only",
    ],
    "evidence_handling": [
        "evidence_ignored",
        "orphan_evidence",
        "judgment_without_evidence",
        "test_without_evidence",
    ],
    "experimental_strategy": [
        "dead_end_update",
        "no_belief_revision",
        "hypothesis_to_commitment_shortcut",
    ],
}

SUBGRAPH_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "popperian_falsification",
        "bayesian_belief_updating",
        "abductive",
    ],
    "evidence_handling": [
        "triangulation",
        "exploratory_to_confirmatory",
    ],
    "experimental_strategy": [
        "ml_make_it_work",
        "preregistered",
        "active_learning",
    ],
}

FAMILY_ORDER = ["hypothesis_generation", "evidence_handling", "experimental_strategy"]

GOOD_COLOR = "#1a5276"
BAD_COLOR = "#c0392b"

FAMILY_DISPLAY = {
    "hypothesis_generation": "Hypothesis generation",
    "evidence_handling": "Evidence handling",
    "experimental_strategy": "Experimental strategy",
}

PATTERN_SHORT: dict[str, str] = {
    "untested_hypothesis": "Untested hyp.",
    "unresolved_contradiction": "Unresolved contrad.",
    "confirmation_only": "Confirmation only",
    "evidence_ignored": "Evidence ignored",
    "orphan_evidence": "Orphan evidence",
    "judgment_without_evidence": "Judgment w/o evid.",
    "test_without_evidence": "Test w/o evid.",
    "dead_end_update": "Dead-end update",
    "no_belief_revision": "No belief revision",
    "hypothesis_to_commitment_shortcut": "Hyp.\u2192commit shortcut",
    "popperian_falsification": "Popper. falsification",
    "bayesian_belief_updating": "Bayesian updating",
    "abductive": "Abductive",
    "triangulation": "Triangulation",
    "exploratory_to_confirmatory": "Explor.\u2192confirm.",
    "ml_make_it_work": "ML make-it-work",
    "preregistered": "Preregistered",
    "active_learning": "Active learning",
}


def _pretty(key: str) -> str:
    """Turn a snake_case pattern key into a readable label."""
    return PATTERN_SHORT.get(key, key.replace("_", " ").capitalize())


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frac(data: dict, section: str, field: str) -> float:
    v = data.get(section, {}).get(field, {}).get("fraction")
    if v is None:
        return 0.0
    return float(v)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved: {path}")


# ── Build ordered list of rows ────────────────────────────────────────────
def _build_rows(overall: dict) -> list[dict]:
    """Return a list of row dicts in display order (top→bottom).

    Each dict has keys: label, value, color, kind ('header' | 'bar').
    """
    rows: list[dict] = []
    for fam in FAMILY_ORDER:
        rows.append({"label": FAMILY_DISPLAY[fam], "kind": "header"})
        # Collect all patterns in this family
        fam_bars = [
            {
                "label": _pretty(pat),
                "value": _frac(overall, "subgraph_presence_global", pat),
                "color": GOOD_COLOR,
                "kind": "bar",
            }
            for pat in SUBGRAPH_FAMILIES[fam]
        ] + [
            {
                "label": _pretty(pat),
                "value": _frac(overall, "antipattern_presence_global", pat),
                "color": BAD_COLOR,
                "kind": "bar",
            }
            for pat in ANTIPATTERN_FAMILIES[fam]
        ]
        # Sort by value descending
        fam_bars.sort(key=lambda r: r["value"], reverse=True)
        rows.extend(fam_bars)
    return rows


def plot(summary: dict, out: Path) -> None:
    overall = summary["groupings"]["overall"]
    n_traces = overall.get("n_traces", "?")
    rows = _build_rows(overall)

    n_rows = len(rows)
    row_height = 0.30
    header_extra = 0.22
    fig_height = n_rows * row_height + sum(
        header_extra for r in rows if r["kind"] == "header"
    )
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, fig_height))

    bar_height = 0.22
    rounding = 0.25  # FancyBboxPatch corner radius (data coords, x in 0-100)
    y_pos = 0.0
    y_positions = []

    for row in rows:
        if row["kind"] == "header":
            y_pos -= header_extra
        y_positions.append(y_pos)
        y_pos -= 0.42

    # Flip so first row is at top
    y_arr = np.array(y_positions)

    x_lim = 100.0  # x-axis in percent (0-100)

    # Aspect-ratio correction so rounded corners look circular on screen
    y_range = (y_arr.max() + 0.5) - (y_arr.min() - 0.5)
    mutation_aspect = (TWO_COL_WIDTH * y_range) / (fig_height * x_lim)

    for y, row in zip(y_arr, rows, strict=False):
        if row["kind"] == "header":
            ax.text(
                -0.28,
                y,
                row["label"],
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                transform=ax.get_yaxis_transform(),
            )
        else:
            val = row["value"]
            pct = val * 100  # convert to 0-100 scale
            color = row["color"]
            # Draw rounded rectangle
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
                ax.add_patch(box)
            # Percentage label
            ax.text(
                pct + 0.8,
                y,
                f"{val:.0%}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )
            # Row label
            ax.text(
                -0.01,
                y,
                row["label"],
                ha="right",
                va="center",
                fontsize=8.5,
                transform=ax.get_yaxis_transform(),
            )

    # Axis formatting
    ax.set_xlim(0, x_lim)
    ax.set_ylim(y_arr.min() - 0.5, y_arr.max() + 0.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    # ── Legend (bottom-left) + trace count (bottom-right) ─────────────
    legend_elements = [
        Patch(facecolor=GOOD_COLOR, alpha=0.85, label="Productive patterns"),
        Patch(facecolor=BAD_COLOR, alpha=0.85, label="Reasoning breakdowns"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(-0.28, -0.04),
        ncol=2,
        frameon=False,
        borderpad=0,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.1,
    )

    ax.text(
        1.0,
        -0.04,
        f"$N$ = {n_traces} traces across all domains and models for ReAct",
        fontsize=9,
        ha="right",
        va="top",
        transform=ax.transAxes,
        color="grey",
    )

    fig.tight_layout()
    _save(fig, out / "overall_pattern_bars.pdf")


def main(
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    """Generate the aggregated pattern bar chart.

    Args:
        summary_path: Path to the annotation_summary.json file.
        output_dir: Directory where the figure will be written.
    """
    summary = load_summary(Path(summary_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating overall pattern bar chart in {out} ...")
    plot(summary, out)
    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
