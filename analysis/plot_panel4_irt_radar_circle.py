"""
Spider chart of IRT reasoning capability (theta_R) per model across environments.

Circular frame version (standard polar projection).
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_WIDTH
from loguru import logger
from plot_config import ENVIRONMENT_NAMES, FONT_SIZES, MODEL_COLOURS, MODEL_NAMES

MODEL_COLOR_MAP = dict(zip(MODEL_NAMES.keys(), MODEL_COLOURS, strict=False))

lama_aesthetics.get_style("main")

RESULTS_DIR = Path(__file__).parent / "results" / "lfm-binomial" / "irt_baseline"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_4"


def _split_label(lb):
    """Split into at most 2 lines near the middle."""
    words = lb.split()
    if len(words) <= 2:
        return "\n".join(words) if len(words) == 2 else lb
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _make_spider(ax, pivot, model_color_map):
    """Draw a spider chart on the given polar axes (circular frame)."""
    environments = pivot.index.tolist()
    n_envs = len(environments)

    angles = np.linspace(0, 2 * np.pi, n_envs, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    all_vals = pivot.to_numpy().flatten()
    r_min = np.floor(all_vals.min())
    r_max = np.ceil(all_vals.max())

    ax.set_ylim(r_min - 0.5, r_max + 0.5)

    ax.grid(False)
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)

    ring_values = [round((r_min + r_max) / 2), round(r_max)]
    for rv in ring_values:
        ring_angles = np.linspace(0, 2 * np.pi, 100)
        ax.plot(ring_angles, [rv] * 100, color="#d0d0d0", linewidth=0.4, zorder=0)

    for angle in angles[:-1]:
        ax.plot(
            [angle, angle],
            [r_min - 0.5, r_max],
            color="#d0d0d0",
            linewidth=0.4,
            zorder=0,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [_split_label(env) for env in environments],
        fontsize=FONT_SIZES["tick_label"] + 2,
    )
    ax.tick_params(axis="x", pad=15)

    for model_id, display_name in MODEL_NAMES.items():
        if display_name not in pivot.columns:
            continue
        values = pivot[display_name].tolist()
        values += values[:1]
        color = model_color_map.get(model_id, "#999999")

        ax.plot(
            angles,
            values,
            linewidth=1.2,
            color=color,
            label=display_name,
            marker="o",
            markersize=3,
            markeredgewidth=0,
            zorder=3,
        )
        ax.fill(angles, values, alpha=0.1, color=color, zorder=1)


def plot_capability_radar(output_path: Path):
    """Generate a spider chart for reasoning capability (theta_R)."""
    reasoning_df = pd.read_csv(RESULTS_DIR / "reasoning_theta.csv")

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH),
        subplot_kw={"projection": "polar"},
    )

    plot_df = reasoning_df.copy()
    plot_df["model"] = plot_df["model"].map(MODEL_NAMES)
    plot_df["environment"] = plot_df["environment"].map(ENVIRONMENT_NAMES)

    pivot = plot_df.pivot_table(
        index="environment", columns="model", values="theta_mean"
    )

    env_order = [
        "spectra",
        "wetlab",
        "resistor",
        "retro",
        "afm",
        "md",
        "catalyst",
        "ml",
    ]
    env_display_order = [
        ENVIRONMENT_NAMES[e] for e in env_order if ENVIRONMENT_NAMES[e] in pivot.index
    ]
    pivot = pivot.reindex(env_display_order)

    _make_spider(ax, pivot, MODEL_COLOR_MAP)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(MODEL_NAMES),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved -> {output_path}")


def main():
    plot_capability_radar(OUT_DIR / "capability_radar_circle.png")


if __name__ == "__main__":
    main()
