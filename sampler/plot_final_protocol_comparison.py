"""Final protocol comparison: random, stratified_proportional, stratified_bayes_irt
— all three with proper error bands.

random / stratified_proportional bands: variability across repeated random
draws (item-sampling variance).
stratified_bayes_irt band: variability across bootstrap-resampled posterior
draws (MCMC/posterior estimation uncertainty) — see run_final_protocol_comparison.py.
Different underlying sources of randomness, but both answer the same
question for their respective method: "how much would this result vary if
we reran the design process."

Usage:
    python plot_final_protocol_comparison.py
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
    "stratified_bayes_irt": "#ea580c",
}


def _band(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("budget")[col]
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for method, colour in COLOURS.items():
        b = _band(df[df.method == method], "rho_all")
        ax.plot(b.index, b["median"], color=colour, linewidth=2, marker="o", markersize=3, label=method)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.18, linewidth=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ, all 12 subjects")
    ax.set_title("Ranking preservation, everyone together", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for method, colour in COLOURS.items():
        b = _band(df[df.method == method], "mse_new")
        ax.plot(b.index, b["median"], color=colour, linewidth=2, marker="o", markersize=3, label=method)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.18, linewidth=0)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("MSE on the 3 new models")
    ax.set_title("Score reconstruction, new models", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    n_repeats = df[df.method == "random"].groupby("budget").size().iloc[0]
    n_boot = df[df.method == "stratified_bayes_irt"].groupby("budget").size().iloc[0]
    fig.suptitle(
        f"Protocol comparison — median ± IQR "
        f"(random/stratified: {n_repeats} repeats; stratified_bayes_irt: {n_boot} posterior bootstrap resamples)\n"
        f"(item set fit ONLY on the 3 legacy models)",
        fontsize=10.5,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "final_protocol_comparison_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "final_protocol_comparison.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
