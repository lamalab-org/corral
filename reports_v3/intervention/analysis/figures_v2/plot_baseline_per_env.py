"""Baseline Pass@k and Pass^k per environment — 1×N grid.

Each subplot shows both metrics for one environment.
Blue (#16476A) = Pass@k, Red (#BF092F) = Pass^k.

Usage:
    python plot_baseline_per_env.py            # all 6 environments
    python plot_baseline_per_env.py --no-md    # exclude md
"""

import argparse
import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import avg_matched_baseline

ALL_ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]

ENV_LABELS = {
    "spectra": "Spectroscopic Structure Elucidation",
    "wetlab": "Inorganic Qualitative Analysis",
    "retrosynthesis": "Retrosynthetic Planning",
    "resistor": "Circuit Inference",
    "md": "Molecular Simulation",
    "ml": "ML-based Property Prediction",
}

BLUE = "#16476A"
RED = "#BF092F"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-md", action="store_true", help="Exclude md environment")
    args = parser.parse_args()

    envs = [e for e in ALL_ENVIRONMENTS if not (args.no_md and e == "md")]

    fig, axes = plt.subplots(
        1,
        len(envs),
        figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT),
        sharey=True,
    )

    for ax, env in zip(axes, envs, strict=False):
        all_k, all_y = [], []

        # Pass@k
        result = avg_matched_baseline(env, metric_type="pass_at")
        if result:
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=BLUE,
                marker="o",
                markersize=3,
                label="Pass@k",
                linewidth=1.4,
            )
            all_k.extend(ks)
            all_y.extend(vals)

        # Pass^k
        result = avg_matched_baseline(env, metric_type="pass_caret")
        if result:
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=RED,
                marker="s",
                markersize=3,
                label="Pass^k",
                linewidth=1.4,
            )
            all_k.extend(ks)
            all_y.extend(vals)

        ax.set_title(ENV_LABELS[env], fontsize=6)
        ax.set_xlabel("k")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([1, 5, 10, 15])

    axes[0].set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=6, framealpha=0.9, loc="center right")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)

    out = Path(__file__).parent / "figures" / "baseline_per_env.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf")
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
