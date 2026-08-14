"""Per-new-model Spearman rho vs budget, for stratified_irt.

Reads per_model_spearman_results.csv: for each of the 3 new models, rho is
computed over an 8-point set (that model's 2 scaffold variants + the 6
legacy subjects the item set was fit on) — NOT just the model's own 2
points, which would be a degenerate ±1 correlation with no information (see
run's accompanying explanation). This isolates whether one particular new
model is harder to rank correctly than the others.

Usage:
    python plot_per_model_spearman.py
"""

from pathlib import Path

import sys

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from plot_mini_item_response_matrix import MODEL_COLOURS, MODEL_NAMES  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, g in df.groupby("model"):
        g = g.sort_values("budget")
        colour = MODEL_COLOURS.get(model, "#999999")
        ax.plot(g.budget, g["rho"], color=colour, linewidth=2, marker="o", markersize=3, label=MODEL_NAMES.get(model, model))
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ (6 legacy subjects + this model's 2 scaffolds)")
    ax.set_title(
        "stratified_irt: per-new-model ranking preservation\n(each line: 8-point rho, not the degenerate 2-point version)",
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
    path = Path(results) if results else DATA_DIR / "per_model_spearman_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "per_model_spearman.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
