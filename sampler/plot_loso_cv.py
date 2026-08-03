"""Plot leave-one-subject-out CV results: MSE and ranking vs budget.

Reads loso_cv_results.csv from run_loso_cv.py — one row per (method,
held_out subject, budget). Aggregates median + IQR ACROSS the 12 held-out
folds (not repeats): this band answers "how much does the result depend on
which subject we're generalizing to", which is the meaningful variability
for an out-of-sample check, as opposed to the within-design sampling noise
shown in plot_sampler_comparison.py.

Usage:
    python plot_loso_cv.py
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

SAMPLER_COLOURS = {
    "random": "#7150e0",
    "stratified_proportional": "#16a34a",
    "irt_maxinfo": "#dc2626",
    "stratified_irt": "#0891b2",
}


def _band(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("budget")[col]
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})


def plot_loso(results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for method, df in results.groupby("method"):
        colour = SAMPLER_COLOURS.get(method, "#999999")
        b = _band(df, "mse")
        ax.plot(b.index, b["median"], color=colour, linewidth=2, label=method)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("held-out subject MSE")
    ax.set_title("Held-out score reconstruction (lower is better)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for method, df in results.groupby("method"):
        colour = SAMPLER_COLOURS.get(method, "#999999")
        b = _band(df, "spearman")
        ax.plot(b.index, b["median"], color=colour, linewidth=2, label=method)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ (12-subject ranking, held-out included)")
    ax.set_title("Ranking preservation (higher is better)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    n_folds = results["held_out"].nunique()
    fig.suptitle(f"Leave-one-subject-out CV — median ± IQR across {n_folds} held-out folds", fontsize=11)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "loso_cv_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "loso_cv_comparison.pdf"
    plot_loso(df, out)


if __name__ == "__main__":
    fire.Fire(main)
