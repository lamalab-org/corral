"""Spearman rho over ALL 12 subjects (6 legacy + 6 new) vs budget, per method.

Companion to plot_new_models_spearman_by_method.py, which only correlated
the 6 new-model subjects against each other (rho_new_only). This uses
rho_all from legacy_to_new_cv_results.csv instead — ranking across the full
set of 6 models x 2 scaffolds, legacy and new together.

The item set is still built the same way for all 3 methods: stratified_irt's
order comes from a 2PL fit on the 3 legacy models only (never touching the
3 new models' data); random / stratified_proportional never look at
response data at all. Only the EVALUATION changed — from "new-models-only"
to "everyone."

Usage:
    python plot_all_subjects_spearman_by_method.py
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
        ax.plot(g.budget, g["rho_all"], color=colour, linewidth=2, marker="o", markersize=3, label=method)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ across all 12 subjects (6 legacy + 6 new)")
    ax.set_title(
        "Ranking preservation, ALL 6 models (legacy + new) together\n"
        "(item set fit ONLY on the 3 legacy models)",
        fontsize=11,
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

    out = Path(output) if output else OUT_DIR / "all_subjects_spearman_by_method.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
