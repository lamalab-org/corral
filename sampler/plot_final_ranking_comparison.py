"""Headline plot: global ranking preservation vs. budget, all 5 methods head
to head, with comparable error bars.

Reads final_ranking_comparison_results.csv from run_final_ranking_comparison.py
— every method there has the same number of independent replicates per
budget (n_draws), so the median ± IQR bands mean the same thing for all 5
lines: random/stratified draws vs. stratified_bayes_irt's posterior
bootstrap resamples.

Metric: rho_all_global — Spearman rho between mini-set and full-set scores,
pooled over items, across all 12 subjects (6 legacy + 6 new). See
plot_ranking_by_env_group.py for the per-environment / new-models-only
breakdown.

Usage:
    python plot_final_ranking_comparison.py
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

# Fixed hue order (never re-cycled per subset) - validated colorblind-safe as
# a set via dataviz's scripts/validate_palette.js (green<->amber sits in the
# 6-8 CVD-separation floor band, legal because every line also carries a
# direct end-of-line label, not color alone).
METHOD_COLOURS = {
    "random": "#7150e0",
    "stratified_equal": "#b45309",
    "stratified_proportional": "#16a34a",
    "stratified_proportional_nested": "#2563eb",
    "stratified_bayes_irt": "#dc2626",
}
METHOD_LABELS = {
    "random": "random",
    "stratified_equal": "stratified (equal)",
    "stratified_proportional": "stratified (proportional)",
    "stratified_proportional_nested": "stratified (proportional, nested by level)",
    "stratified_bayes_irt": "stratified + Bayesian IRT (MCMC)",
}


def _band(df: pd.DataFrame, col: str, group: str = "budget") -> pd.DataFrame:
    g = df.groupby(group)[col]
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    n_draws = df.groupby(["method", "budget"]).size().min()
    for method in METHOD_COLOURS:
        sub = df[df.method == method]
        if sub.empty:
            continue
        colour = METHOD_COLOURS[method]
        b = _band(sub, "rho_all_global")
        ax.plot(b.index, b["median"], color=colour, linewidth=2, marker="o", markersize=3, label=METHOD_LABELS[method])
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.15, linewidth=0)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items, spectra+wetlab+retro pool; resistor kept whole)")
    ax.set_ylabel("Spearman ρ (mini-set vs. full-set ranking, all 12 subjects)")
    ax.set_title(
        f"Global ranking preservation vs. budget — median ± IQR over {n_draws} draws/boots per method",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "final_ranking_comparison_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "final_ranking_comparison.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
