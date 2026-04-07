"""
Spider chart of IRT reasoning capability (θ_R) per model across environments.

Circular frame version (standard polar projection).
"""

import importlib.util
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_WIDTH
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Load plot_config
plot_config_spec = importlib.util.spec_from_file_location(
    "plot_config", REPO_ROOT / "analysis" / "plot_config.py"
)
if plot_config_spec is None or plot_config_spec.loader is None:
    raise ImportError("Could not load plot config")
plot_config = importlib.util.module_from_spec(plot_config_spec)
plot_config_spec.loader.exec_module(plot_config)

MODEL_NAMES = plot_config.MODEL_NAMES
MODEL_COLOURS = plot_config.MODEL_COLOURS
ENVIRONMENT_NAMES = plot_config.ENVIRONMENT_NAMES
FONT_SIZES = plot_config.FONT_SIZES

MODEL_COLOR_MAP = dict(zip(MODEL_NAMES.keys(), MODEL_COLOURS, strict=False))

lama_aesthetics.get_style("main")

# Paths
RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "lfm-binomial" / "irt_baseline"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


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

    # Angles for each spoke
    angles = np.linspace(0, 2 * np.pi, n_envs, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Compute data range
    all_vals = pivot.values.flatten()
    r_min = np.floor(all_vals.min())
    r_max = np.ceil(all_vals.max())

    ax.set_ylim(r_min - 0.5, r_max + 0.5)

    # Remove default grid and spines
    ax.grid(False)
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)

    # Draw two reference rings (middle and outer) — no labels
    ring_values = [round((r_min + r_max) / 2), round(r_max)]
    for rv in ring_values:
        ring_angles = np.linspace(0, 2 * np.pi, 100)
        ax.plot(ring_angles, [rv] * 100, color="#d0d0d0", linewidth=0.4, zorder=0)

    # Draw spoke lines (only within data range)
    for angle in angles[:-1]:
        ax.plot(
            [angle, angle],
            [r_min - 0.5, r_max],
            color="#d0d0d0",
            linewidth=0.4,
            zorder=0,
        )

    # Spoke labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [_split_label(env) for env in environments],
        fontsize=FONT_SIZES["tick_label"] + 2,
    )
    ax.tick_params(axis="x", pad=15)

    # Plot each model
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
    """Generate a spider chart for reasoning capability (θ_R)."""
    reasoning_df = pd.read_csv(RESULTS_DIR / "reasoning_theta.csv")

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH),
        subplot_kw={"projection": "polar"},
    )

    # Map to display names
    df = reasoning_df.copy()
    df["model"] = df["model"].map(MODEL_NAMES)
    df["environment"] = df["environment"].map(ENVIRONMENT_NAMES)

    pivot = df.pivot_table(index="environment", columns="model", values="theta_mean")

    # Order environments by cognitive group
    ENV_ORDER = [
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
        ENVIRONMENT_NAMES[e] for e in ENV_ORDER if ENVIRONMENT_NAMES[e] in pivot.index
    ]
    pivot = pivot.reindex(env_display_order)

    _make_spider(ax, pivot, MODEL_COLOR_MAP)

    # Legend below
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
    logger.success(f"Saved → {output_path}")


def main():
    plot_capability_radar(OUTPUT_DIR / "capability_radar_circle.png")


if __name__ == "__main__":
    main()
