"""Plot baseline Pass@k vs k — all environments overlaid in one plot."""

import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import avg_matched_baseline  # noqa: E402

ENVIRONMENTS = [
    "spectra",
    "wetlab",
    "retrosynthesis",
    "resistor",
    "md",
    "ml",
    "catalyst",
]

ENV_LABELS = {
    "spectra": "Spectroscopic Structure Elucidation",
    "wetlab": "Inorganic Qualitative Analysis",
    "resistor": "Circuit Inference",
    "md": "Molecular Simulation",
    "ml": "ML-based Property Prediction",
    "retrosynthesis": "Retrosynthetic Planning",
    "catalyst": "Catalyst Discovery",
}

ENV_COLORS = {
    "spectra": "#16476A",
    "wetlab": "#BF092F",
    "retrosynthesis": "#7A7A7A",
    "resistor": "#E8A317",
    "md": "#2E8B57",
    "ml": "#6A5ACD",
    "catalyst": "#D2691E",
}

ENV_MARKERS = {
    "spectra": "o",
    "wetlab": "s",
    "retrosynthesis": "D",
    "resistor": "^",
    "md": "v",
    "ml": "P",
    "catalyst": "X",
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
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH * 0.5, ONE_COL_HEIGHT * 1.5))

    all_k_vals, all_y_vals = [], []

    for env in ENVIRONMENTS:
        result = avg_matched_baseline(env, metric_type="pass_at")
        if not result:
            continue
        ks, vals = result
        ax.plot(
            ks,
            vals,
            color=ENV_COLORS[env],
            marker=ENV_MARKERS[env],
            markersize=5,
            label=ENV_LABELS[env],
            linewidth=1.5,
        )
        all_k_vals.extend(ks)
        all_y_vals.extend(vals)

    if all_k_vals and all_y_vals:
        range_frame(ax, np.array(all_k_vals), np.array(all_y_vals), pad=0.05)
        ax.set_xticks([1, 5, 10, 15])

    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k")
    ax.legend(fontsize=6, framealpha=0.9, loc="lower right")

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / "baseline_pass_at_decay_combined.png"
    _save_fig(fig, out_path)


if __name__ == "__main__":
    main()
