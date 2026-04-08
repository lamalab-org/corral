"""Plot yearly counts of AI-scientist papers (strict and expanded definitions).

Reads `analysis/results_ai_scientists/yearly_counts.csv` and produces two
line-plot figures — one for the *strict* paper count and one for the *expanded*
paper count — using the shared Corral plotting style.

Usage:
  python analysis/plot_ai_scientists_yearly.py
"""

from __future__ import annotations

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

DATA_PATH = (
    Path(__file__).parent / "openalex" / "results_ai_scientists" / "yearly_counts.csv"
)
OUTPUT_DIR = Path(__file__).parent / "results" / "figures" / "openalex"

LINE_COLOR = "#7150e0"


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
        color=LINE_COLOR,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.set_xticklabels(years.astype(int), rotation=45, ha="right")

    range_frame(ax, x=years.astype(float), y=counts.astype(float))

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


if __name__ == "__main__":
    main()
