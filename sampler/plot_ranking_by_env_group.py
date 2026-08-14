"""Ranking preservation at the ENVIRONMENT level, new-models-only vs. all
models, side by side.

Reads final_ranking_comparison_results.csv from run_final_ranking_comparison.py.
Companion to plot_final_ranking_comparison.py, which shows the pooled
("global") rho across all items at once — this instead computes Spearman rho
SEPARATELY within each environment and averages across environments
(rho_*_per_env), so it answers "does ranking still hold up environment by
environment", not just in aggregate.

Left panel: new-model subjects only (the 6 subjects whose data never
influenced item selection — the real generalization test).
Right panel: all 12 subjects (6 legacy + 6 new).

Usage:
    python plot_ranking_by_env_group.py
"""

import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from plot_final_ranking_comparison import (
    METHOD_COLOURS,
    METHOD_LABELS,
    _band,
)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    n_draws = df.groupby(["method", "budget"]).size().min()
    panels = [
        (
            axes[0],
            "rho_new_per_env",
            "New models only (6 subjects, never seen during item selection)",
        ),
        (axes[1], "rho_all_per_env", "All 12 subjects (6 legacy + 6 new)"),
    ]
    for ax, col, title in panels:
        for method in METHOD_COLOURS:
            sub = df[df.method == method]
            if sub.empty:
                continue
            colour = METHOD_COLOURS[method]
            b = _band(sub, col)
            ax.plot(
                b.index,
                b["median"],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=3,
                label=METHOD_LABELS[method],
            )
            ax.fill_between(
                b.index, b["p25"], b["p75"], color=colour, alpha=0.15, linewidth=0
            )
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("budget (# items)")
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Spearman ρ, mean across environments")
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(
        f"Environment-level ranking preservation vs. budget — median ± IQR over {n_draws} draws/boots per method",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = (
        Path(results) if results else DATA_DIR / "final_ranking_comparison_results.csv"
    )
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "ranking_by_env_group.pdf"
    plot(df, out)


if __name__ == "__main__":
    fire.Fire(main)
