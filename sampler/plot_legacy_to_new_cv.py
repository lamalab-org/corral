"""Plot legacy-fit / new-model-evaluate generalization results.

Reads legacy_to_new_cv_results.csv from run_legacy_to_new_cv.py. One value
per (method, budget) — no repeats/folds to band here (the IRT methods are a
single deterministic fit-once-on-legacy result; the design-based baselines'
"repeats" were already averaged into the metric during collection).

Usage:
    python plot_legacy_to_new_cv.py
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


def plot_legacy_to_new(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    new_mse = df[df.group == "new"].groupby(["method", "budget"])["sq_err"].mean().reset_index()
    ax = axes[0]
    for method, g in new_mse.groupby("method"):
        g = g.sort_values("budget")
        ax.plot(g.budget, g.sq_err, color=SAMPLER_COLOURS.get(method, "#999"), linewidth=2, label=method)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("MSE on the 3 new models")
    ax.set_title("Score reconstruction, new models (lower better)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    rho = df.groupby(["method", "budget"])["rho_new_only"].first().reset_index()
    ax = axes[1]
    for method, g in rho.groupby("method"):
        g = g.sort_values("budget")
        ax.plot(g.budget, g.rho_new_only, color=SAMPLER_COLOURS.get(method, "#999"), linewidth=2, label=method)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ, among the 3 new models only")
    ax.set_title("Ranking preservation, new models (higher better)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Item set fit on 3 legacy models only, evaluated on 3 models never seen during design",
        fontsize=11,
    )
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

    out = Path(output) if output else OUT_DIR / "legacy_to_new_cv.pdf"
    plot_legacy_to_new(df, out)


if __name__ == "__main__":
    fire.Fire(main)
