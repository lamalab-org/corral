"""Baseline Pass@k and Pass^k with blue->red gradient.

Output figures:
  1. TWO_COL_WIDTH x ONE_COL_HEIGHT -- Pass@k (left) + Pass^k (right)
  2. ONE_COL_WIDTH x ONE_COL_HEIGHT -- Pass@k only
  3. ONE_COL_WIDTH x ONE_COL_HEIGHT -- Pass^k only

Reads from results/data/intervention_reports.jsonl (downloaded from HF).

Usage:
    python plot_baseline_compact.py
    python plot_baseline_compact.py --no-md
"""

import argparse
from pathlib import Path

import lama_aesthetics
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from intervention_utils import avg_matched_baseline, load_reports
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

_SCRIPT_DIR = Path(__file__).resolve().parent

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


def make_gradient(n: int) -> list[str]:
    cmap = mcolors.LinearSegmentedColormap.from_list("br", [BLUE, RED], N=256)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    logger.info(f"Saved to {out_path}")
    plt.close(fig)


def _plot_metric(ax, metric_type, ylabel, colors, df, env_list=None, labels=None):
    if labels is None:
        labels = ENV_LABELS
    envs = env_list or ALL_ENVIRONMENTS
    all_k, all_y = [], []
    for env, color in zip(envs, colors, strict=False):
        result = avg_matched_baseline(env, metric_type=metric_type, df=df)
        if not result:
            continue
        ks, vals = result
        ax.plot(ks, vals, color=color, label=labels[env], linewidth=1.5)
        all_k.extend(ks)
        all_y.extend(vals)

    if all_k and all_y:
        range_frame(ax, np.array(all_k), np.array(all_y), pad=0.05)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-md", action="store_true", help="Exclude md environment")
    args = parser.parse_args()

    reports = load_reports()
    envs = [e for e in ALL_ENVIRONMENTS if not (args.no_md and e == "md")]
    out_dir = _SCRIPT_DIR / "results" / "figures" / "intervention"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = make_gradient(len(envs))

    # 1. Two-col: Pass@k + Pass^k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_metric(ax1, "pass_at", "Pass@k", colors, reports, env_list=envs)
    _plot_metric(ax2, "pass_caret", "Pass^k", colors, reports, env_list=envs)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center right",
            fontsize=5.5,
            framealpha=0.9,
            bbox_to_anchor=(1.22, 0.5),
        )
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_compact_twocol.png")

    # 2. One-col: Pass@k only
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_metric(ax, "pass_at", "Pass@k", colors, reports, env_list=envs)
    ax.legend(fontsize=5, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_compact_pass_at.png")

    # 3. One-col: Pass^k only
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_metric(ax, "pass_caret", "Pass^k", colors, reports, env_list=envs)
    ax.legend(fontsize=5, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_compact_pass_caret.png")


if __name__ == "__main__":
    main()
