"""Spearman rho vs budget for stratified_irt, split into 3 groups.

Reads legacy_to_new_cv_results.csv (already has rho_all / rho_legacy_only /
rho_new_only per method/budget) and plots the 3 as separate lines, filtered
to stratified_irt only — the single item set fit on the 3 legacy models.

  rho_all:          ranking across all 12 (6 legacy + 6 new) subjects
  rho_legacy_only:  ranking among the 6 legacy subjects (the ones the fit saw)
  rho_new_only:     ranking among the 6 new subjects (never seen at fit time)
                     — this is the sharpest test of real generalization.

Usage:
    python plot_irt_spearman_groups.py
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

LINES = {
    "rho_all": ("#7150e0", "all 12 subjects"),
    "rho_legacy_only": ("#16a34a", "6 legacy subjects (seen at fit time)"),
    "rho_new_only": ("#dc2626", "6 new subjects (never seen at fit time)"),
}


def plot_spearman_groups(df: pd.DataFrame, output_path: Path) -> None:
    df = df[df.method == "stratified_irt"].drop_duplicates("budget").sort_values("budget")

    fig, ax = plt.subplots(figsize=(7, 5))
    for col, (colour, label) in LINES.items():
        ax.plot(df.budget, df[col], color=colour, linewidth=2, marker="o", markersize=3, label=label)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("stratified_irt: ranking preservation by subject group\n(item set fit ONLY on the 3 legacy models)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
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

    out = Path(output) if output else OUT_DIR / "irt_spearman_groups.pdf"
    plot_spearman_groups(df, out)


if __name__ == "__main__":
    fire.Fire(main)
