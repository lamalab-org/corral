"""Spearman rho on the 3 new test models vs budget, one line per method.

Reads legacy_to_new_cv_results.csv (already has rho_new_only per method and
budget) and plots it for the 3 methods under real consideration —
random, stratified_proportional, stratified_irt (irt_maxinfo dropped, since
it has no environment-starvation guarantee and stratified_irt matches or
beats its ranking performance anyway).

This is the direct evidence for "which method to pick at budget ~20-25":
rho computed among the 6 (new model x scaffold) subjects, using an item set
fit ONLY on the 3 legacy models — the strictest generalization test we have.

Usage:
    python plot_new_models_spearman_by_method.py
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

METHODS = {
    "random": "#7150e0",
    "stratified_proportional": "#16a34a",
    "stratified_irt": "#0891b2",
}


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, colour in METHODS.items():
        g = df[df.method == method].drop_duplicates("budget").sort_values("budget")
        ax.plot(g.budget, g["rho_new_only"], color=colour, linewidth=2, marker="o", markersize=3, label=method)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvspan(20, 25, color="gray", alpha=0.08, zorder=0)
    ax.text(22.5, ax.get_ylim()[0] + 0.01, "planned\nbudget", fontsize=8, ha="center", color="gray")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ among the 6 new (model, scaffold) subjects")
    ax.set_title(
        "Ranking preservation on the 3 new test models\n(item set fit ONLY on the 3 legacy models)", fontsize=11
    )
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "legacy_to_new_cv_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "new_models_spearman_by_method.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
