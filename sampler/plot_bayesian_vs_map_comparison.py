"""Full 5-method comparison: random, stratified, MAP-IRT, Bayesian-IRT.

Merges legacy_to_new_cv_results.csv (random, stratified_proportional,
stratified_irt) with bayesian_legacy_to_new_cv_results.csv (bayes_maxinfo,
stratified_bayes_irt) and plots rho_all + MSE(new) vs budget for all 5 —
the direct test of whether posterior-averaged (uncertainty-aware) item
selection actually beats the MAP-point-based version on the metric that
mattered: ranking preservation across the FULL 12-subject set, not just
the new models against each other.

Usage:
    python plot_bayesian_vs_map_comparison.py
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

COLOURS = {
    "random": "#7150e0",
    "stratified_proportional": "#16a34a",
    "stratified_irt": "#0891b2",
    "bayes_maxinfo": "#dc2626",
    "stratified_bayes_irt": "#ea580c",
}


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for method, colour in COLOURS.items():
        g = df[df.method == method].drop_duplicates("budget").sort_values("budget")
        ax.plot(g.budget, g["rho_all"], color=colour, linewidth=2, marker="o", markersize=3, label=method)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ, all 12 subjects")
    ax.set_title("Ranking preservation, everyone together", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    new_mse = df[df.group == "new"].groupby(["method", "budget"])["sq_err"].mean().reset_index()
    for method, colour in COLOURS.items():
        g = new_mse[new_mse.method == method].sort_values("budget")
        ax.plot(g.budget, g.sq_err, color=colour, linewidth=2, marker="o", markersize=3, label=method)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("MSE on the 3 new models")
    ax.set_title("Score reconstruction, new models", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "MAP-point vs. posterior-averaged (Bayesian) item selection\n(item set fit ONLY on the 3 legacy models)",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(output: str | None = None) -> None:
    map_df = pd.read_csv(DATA_DIR / "legacy_to_new_cv_results.csv")
    bayes_df = pd.read_csv(DATA_DIR / "bayesian_legacy_to_new_cv_results.csv")
    df = pd.concat([map_df, bayes_df], ignore_index=True)

    out = Path(output) if output else OUT_DIR / "bayesian_vs_map_comparison.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
