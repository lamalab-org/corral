"""Protocol comparison: random, stratified_proportional, stratified_irt (MAP),
stratified_bayes_irt.

The unstratified "maxinfo" variants (irt_maxinfo, bayes_maxinfo) are dropped
— they never had the environment-starvation guarantee stratified allocation
gives, so there's no remaining reason to keep them in the comparison.
stratified_irt (MAP) stays in alongside stratified_bayes_irt so the
MAP-vs-posterior-averaged difference is visible directly, not just asserted.

Reads the same two result files as plot_bayesian_vs_map_comparison.py and
just narrows which methods get plotted.

Usage:
    python plot_final_comparison.py
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
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
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
        "Protocol comparison\n(item set fit ONLY on the 3 legacy models)",
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
    df = df[df.method.isin(COLOURS.keys())]

    out = Path(output) if output else OUT_DIR / "final_comparison.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
