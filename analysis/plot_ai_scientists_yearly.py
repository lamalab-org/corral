"""Plot yearly counts of AI-scientist papers (strict and expanded definitions).

Reads yearly_counts.csv and produces:
- individual strict / expanded line plots and cumulative versions
- a dual-axis comparison plot (log scale) showing strict vs expanded counts
  together with the strict-to-expanded ratio to illustrate that AI-scientist
  papers are gaining territory relative to general AI-for-chemistry papers

Usage:
  python analysis/plot_ai_scientists_yearly.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from matplotlib.ticker import MaxNLocator, NullLocator

lama_aesthetics.get_style("main")
plt.rcParams["font.size"] = 10

DATA_PATH = (
    Path(__file__).parent
    / "openalex"
    / "openalex_ai_scientists_output"
    / "yearly_counts.csv"
)
OUTPUT_DIR = Path(__file__).parent / "results" / "figures" / "openalex"

STRICT_COLOR = "#7150e0"
EXPANDED_COLOR = "#50b0e0"
RATIO_COLOR = "#e05050"
RATIO_COLOR_CUM = "#e0a030"


def _plot_yearly(
    years: np.ndarray,
    counts: np.ndarray,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
    )

    ax.plot(
        years,
        counts,
        marker="o",
        linewidth=2,
        color=STRICT_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.set_xticklabels(years.astype(int), rotation=45, ha="right")

    range_frame(ax, x=years.astype(float), y=counts.astype(float), pad=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _frame_log_dual(
    ax_left: Axes,
    ax_right: Axes,
    years: np.ndarray,
    counts: np.ndarray,
    ratio: np.ndarray,
) -> None:
    """Apply Tufte-style range-frame to a dual-axis plot with log-scale left y."""
    # --- X axis: use range_frame for x only (nice ticks on years) ---
    x_min, x_max = float(years.min()), float(years.max())
    ax_left.set_xticks(years)
    ax_left.spines["bottom"].set_bounds(x_min, x_max)
    ax_left.spines["bottom"].set_position(("outward", 10))
    x_pad = 0.03 * (x_max - x_min)
    ax_left.set_xlim(x_min - x_pad, x_max + x_pad)

    # --- Left y axis: log-scale ticks aligned to decade powers ---
    pos_counts = counts[counts > 0]
    lo = 10 ** math.floor(math.log10(pos_counts.min()))
    hi = 10 ** math.ceil(math.log10(pos_counts.max()))
    yticks = [
        10**e
        for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1)
        if lo <= 10**e <= hi
    ]
    ax_left.set_yticks(yticks, minor=False)
    ax_left.set_yticklabels([f"$10^{{{int(math.log10(t))}}}$" for t in yticks])
    ax_left.yaxis.set_minor_locator(NullLocator())
    ax_left.set_ylim(lo * 0.5, hi * 2)
    ax_left.spines["left"].set_bounds(lo, hi)
    ax_left.spines["left"].set_position(("outward", 10))

    # Hide top/right spines on the left axis
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # --- Right y axis (ratio) ---
    r_min, r_max = float(ratio.min()), float(ratio.max())
    locator = MaxNLocator(nbins=5)
    r_ticks = np.asarray(locator.tick_values(r_min, r_max))
    r_ticks = r_ticks[(r_ticks >= r_min * 0.95) & (r_ticks <= r_max * 1.05)]
    if len(r_ticks) < 2:
        r_ticks = np.array([r_min, r_max])
    r_lo, r_hi = float(r_ticks[0]), float(r_ticks[-1])
    ax_right.set_yticks(r_ticks)
    r_pad = 0.03 * (r_hi - r_lo) if r_hi > r_lo else 0.1
    ax_right.set_ylim(r_lo - r_pad, r_hi + r_pad)
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_bounds(r_lo, r_hi)
    ax_right.spines["right"].set_position(("outward", 10))
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)


def _plot_comparison(
    years: np.ndarray,
    strict: np.ndarray,
    expanded: np.ndarray,
    output_path: Path,
) -> None:
    """Dual-axis log-scale plot: counts on the left, strict/expanded ratio on the right."""
    ratio = strict / np.where(expanded > 0, expanded, 1) * 100  # percentage

    fig, ax1 = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    # Left axis — paper counts (log scale)
    ax1.semilogy(
        years,
        expanded,
        marker="s",
        linewidth=2,
        color=EXPANDED_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="AI for chemistry & materials",
    )
    ax1.semilogy(
        years,
        strict,
        marker="o",
        linewidth=2,
        color=STRICT_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="AI scientist",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of papers (log scale)")
    ax1.set_xticks(years)
    ax1.set_xticklabels(years.astype(int), rotation=45, ha="right")

    # Right axis — ratio
    ax2 = ax1.twinx()
    ax2.plot(
        years,
        ratio,
        marker="D",
        linewidth=2,
        color=RATIO_COLOR,
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linestyle="--",
        label="AI-scientist share in AI for chemistry (%)",
    )
    ax2.set_ylabel("AI-scientist share in\nAI for chemistry (%)", color=RATIO_COLOR)
    ax2.tick_params(axis="y", labelcolor=RATIO_COLOR)

    # --- Manual spine framing for dual-axis log plot ---
    all_counts = np.concatenate([strict, expanded])
    _frame_log_dual(
        ax1,
        ax2,
        years.astype(float),
        all_counts,
        ratio,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_cumulative(
    years: np.ndarray,
    strict: np.ndarray,
    expanded: np.ndarray,
    output_path: Path,
) -> None:
    """Dual-axis log-scale plot with cumulative counts and yearly share."""
    cum_strict = np.cumsum(strict)
    cum_expanded = np.cumsum(expanded)
    ratio = strict / np.where(expanded > 0, expanded, 1) * 100  # yearly share

    fig, ax1 = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    # Left axis — cumulative paper counts (log scale)
    ax1.semilogy(
        years,
        cum_expanded,
        marker="s",
        linewidth=2,
        color=EXPANDED_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="AI for chemistry & materials (cumulative)",
    )
    ax1.semilogy(
        years,
        cum_strict,
        marker="o",
        linewidth=2,
        color=STRICT_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="AI scientist (cumulative)",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative papers (log scale)")
    ax1.set_xticks(years)
    ax1.set_xticklabels(years.astype(int), rotation=45, ha="right")

    # Right axis — yearly ratio
    ax2 = ax1.twinx()
    ax2.plot(
        years,
        ratio,
        marker="D",
        linewidth=2,
        color=RATIO_COLOR_CUM,
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linestyle="--",
        label="AI-scientist share in AI for chemistry (%)",
    )
    ax2.set_ylabel("AI-scientist share in\nAI for chemistry (%)", color=RATIO_COLOR_CUM)
    ax2.tick_params(axis="y", labelcolor=RATIO_COLOR_CUM)

    # --- Manual spine framing for dual-axis log plot ---
    all_counts = np.concatenate([cum_strict, cum_expanded])
    _frame_log_dual(
        ax1,
        ax2,
        years.astype(float),
        all_counts,
        ratio,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    counts_df = pd.read_csv(DATA_PATH)

    years = counts_df["publication_year"].to_numpy()
    strict = counts_df["strict_papers"].to_numpy()
    expanded = counts_df["expanded_papers"].to_numpy()

    _plot_yearly(
        years,
        strict,
        ylabel="Number of papers",
        output_path=OUTPUT_DIR / "yearly_strict_counts.pdf",
    )

    _plot_yearly(
        years,
        expanded,
        ylabel="Number of papers",
        output_path=OUTPUT_DIR / "yearly_expanded_counts.pdf",
    )

    _plot_yearly(
        years,
        np.cumsum(strict),
        ylabel="Number of papers",
        output_path=OUTPUT_DIR / "yearly_strict_counts_cumulative.pdf",
    )

    _plot_yearly(
        years,
        np.cumsum(expanded),
        ylabel="Number of papers",
        output_path=OUTPUT_DIR / "yearly_expanded_counts_cumulative.pdf",
    )

    # Comparison: strict vs expanded on log scale + ratio
    _plot_comparison(
        years,
        strict,
        expanded,
        output_path=OUTPUT_DIR / "yearly_strict_vs_expanded.pdf",
    )

    # Cumulative comparison
    _plot_comparison_cumulative(
        years,
        strict,
        expanded,
        output_path=OUTPUT_DIR / "yearly_strict_vs_expanded_cumulative.pdf",
    )


if __name__ == "__main__":
    main()
