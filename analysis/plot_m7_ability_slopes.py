"""
Forest plot of total ability slopes (λ + θ_{e,ℓ}) and (ψ + φ_{e,ℓ})
per environment × scope cell for M7.

Usage:
    python plot_m7_ability_slopes.py              # both panels
    python plot_m7_ability_slopes.py --reasoning   # reasoning only
"""

from pathlib import Path

import arviz as az
import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import ENVIRONMENT_NAMES, FONT_SIZES

RESULTS_DIR = Path(__file__).parent / "results" / "lfm-binomial"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_4"
MODEL_NAME = "model7_abilities_env_level"

KNOWLEDGE_COLOUR = "#16476A"
REASONING_COLOUR = "#BF092F"

lama_aesthetics.get_style("main")

# Compact full names (no newlines, abbreviated where needed)
ENV_DISPLAY = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "Mol. Simulation",
    "ml": "ML Prediction",
    "resistor": "Circuit Inference",
    "retro": "Retrosynthesis",
    "spectra": "Spectra",
    "wetlab": "Wetlab",
}


def _load():
    idata = az.from_netcdf(RESULTS_DIR / f"{MODEL_NAME}_trace.nc")
    df = pd.read_csv(RESULTS_DIR / "prepared_data.csv")
    posterior = idata.posterior

    mapping = (
        df[["env_level_combo", "env_level_id"]]
        .drop_duplicates()
        .sort_values("env_level_id")
    )
    combo_names = mapping["env_level_combo"].tolist()

    def _display(combo):
        env, level = combo.rsplit("_", 1)
        scope_num = level.replace("L", "S")
        full_name = ENV_DISPLAY.get(env, env)
        return f"{full_name} {scope_num}"

    display_labels = [_display(c) for c in combo_names]

    def _flat(name):
        arr = posterior[name].values
        return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])

    lam_base = _flat("knowledge_coef_base")
    theta = _flat("knowledge_envlevel_slope")
    knowledge_total = lam_base[:, None] + theta

    psi_base = _flat("reasoning_coef_base")
    phi = _flat("reasoning_envlevel_slope")
    reasoning_total = psi_base[:, None] + phi

    return display_labels, knowledge_total, reasoning_total


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved -> {path}")


def main(reasoning=False):
    logger.info("Loading M7 trace and prepared data...")
    display_labels, knowledge_total, reasoning_total = _load()
    n_cells = len(display_labels)

    k_med = np.median(knowledge_total, axis=0)
    k_lo = np.percentile(knowledge_total, 5, axis=0)
    k_hi = np.percentile(knowledge_total, 95, axis=0)

    r_med = np.median(reasoning_total, axis=0)
    r_lo = np.percentile(reasoning_total, 5, axis=0)
    r_hi = np.percentile(reasoning_total, 95, axis=0)

    order = np.argsort(r_med)
    labels = [display_labels[i] for i in order]
    k_med, k_lo, k_hi = k_med[order], k_lo[order], k_hi[order]
    r_med, r_lo, r_hi = r_med[order], r_lo[order], r_hi[order]
    y_pos = np.arange(n_cells)

    if reasoning:
        # --- Reasoning-only single-column plot ---
        fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

        for y, lo, hi in zip(y_pos, r_lo, r_hi):
            ax.plot([lo, hi], [y, y], color=REASONING_COLOUR, linewidth=1.8,
                    solid_capstyle="round", alpha=0.4)
        ax.scatter(r_med, y_pos, color=REASONING_COLOUR, s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=FONT_SIZES["legend"])
        ax.set_xlabel(r"Total reasoning slope ($\psi + \phi_{e,\ell}$)",
                      fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
        range_frame(ax, np.concatenate([r_lo, r_hi]), y_pos, pad=0.1)

        _save(fig, OUT_DIR / "m7_reasoning_slopes.png")
    else:
        # --- Both panels side by side ---
        fig, (ax_k, ax_r) = plt.subplots(
            1, 2, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.6), sharey=True
        )

        for y, lo, hi in zip(y_pos, k_lo, k_hi):
            ax_k.plot([lo, hi], [y, y], color=KNOWLEDGE_COLOUR, linewidth=1.8,
                      solid_capstyle="round", alpha=0.4)
        ax_k.scatter(k_med, y_pos, color=KNOWLEDGE_COLOUR, s=25, zorder=5,
                     edgecolors="white", linewidths=0.4)
        ax_k.axvline(0, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
        ax_k.set_yticks(y_pos)
        ax_k.set_yticklabels(labels, fontsize=FONT_SIZES["tick_label"])
        ax_k.set_xlabel(r"Total knowledge slope ($\lambda + \theta_{e,\ell}$)",
                        fontsize=FONT_SIZES["axis_label"])
        ax_k.set_title("Knowledge", fontsize=FONT_SIZES["title"], fontweight="bold")
        ax_k.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
        range_frame(ax_k, np.concatenate([k_lo, k_hi]), y_pos, pad=0.1)

        for y, lo, hi in zip(y_pos, r_lo, r_hi):
            ax_r.plot([lo, hi], [y, y], color=REASONING_COLOUR, linewidth=1.8,
                      solid_capstyle="round", alpha=0.4)
        ax_r.scatter(r_med, y_pos, color=REASONING_COLOUR, s=25, zorder=5,
                     edgecolors="white", linewidths=0.4)
        ax_r.axvline(0, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
        ax_r.set_xlabel(r"Total reasoning slope ($\psi + \phi_{e,\ell}$)",
                        fontsize=FONT_SIZES["axis_label"])
        ax_r.set_title("Reasoning", fontsize=FONT_SIZES["title"], fontweight="bold")
        ax_r.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
        range_frame(ax_r, np.concatenate([r_lo, r_hi]), y_pos, pad=0.1)

        fig.subplots_adjust(wspace=0.15)
        _save(fig, OUT_DIR / "m7_ability_slopes.png")

    # Log summary
    logger.info("Total ability slopes (median [5%, 95%]):")
    for i, idx in enumerate(order):
        logger.info(
            f"  {display_labels[idx]:35s}  "
            f"K: {k_med[i]:+.2f} [{k_lo[i]:+.2f}, {k_hi[i]:+.2f}]  "
            f"R: {r_med[i]:+.2f} [{r_lo[i]:+.2f}, {r_hi[i]:+.2f}]"
        )


if __name__ == "__main__":
    fire.Fire(main)
