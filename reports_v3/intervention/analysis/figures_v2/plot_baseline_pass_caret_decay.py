"""Plot baseline Pass^k decay vs k — 1x6 single-row figure, all environments."""

import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import avg_matched_baseline  # noqa: E402

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]

BASELINE_COLOR = "#7A7A7A"

ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}


def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    logger.info(f"Saved to {out_path}")
    plt.close(fig)


def main():
    n_envs = len(ENVIRONMENTS)
    n_cols = 3
    n_rows = n_envs // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
        sharey=False,
    )

    for idx, env in enumerate(ENVIRONMENTS):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        result = avg_matched_baseline(env, metric_type="pass_caret")
        if result:
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=BASELINE_COLOR,
                marker="o",
                markersize=4,
                label="Baseline",
                linewidth=1.8,
            )
            range_frame(ax, np.array(ks), np.array(vals), pad=0.05)
            ax.set_xticks([1, 5, 10, 15])
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#999999",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
        if col == 0:
            ax.set_ylabel(r"Pass$^{k}$")
        if row == n_rows - 1:
            ax.set_xlabel("k")

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / "baseline_pass_caret_decay.png"
    _save_fig(fig, out_path)


if __name__ == "__main__":
    main()
