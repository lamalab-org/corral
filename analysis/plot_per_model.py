"""
Create one figure per model comparing reasoning vs knowledge scores per env.

Produces:
  - analysis/claude_scores.pdf
  - analysis/gpt_scores.pdf
  - analysis/gpt_oss_scores.pdf
"""

import json
import logging
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

logger = logging.getLogger(__name__)

lama_aesthetics.get_style("main")

ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH = ROOT / "analysis" / "qa_scores.json"
OUT_DIR = ROOT / "analysis"

# ── Environment mapping ──────────────────────────────────────────────────
# Maps a display label -> (reasoning_env, knowledge_env, knowledge_new_env | None)
ENV_MAP = {
    "AFM": ("afm", "afm_qa", "afm_new_qa"),
    "Catalyst": ("catalyst", "catalyst_qa", None),
    "MD": ("corral_md", "md_qa", "md_new_qa"),
    "ML": ("ml", "ml_qa", None),
    "Resistor": ("resistor", "resistor_qa", None),
    "Retro": ("retro", "retro_qa", None),
    "Spectra": ("spectra", "spectra_qa", None),
    "Wetlab": ("wetlab", "wetlab", None),
}

SERIES_COLORS = {
    "Reasoning": "#E07B39",
    "Knowledge": "#5B8DB8",
    "Knowledge (new)": "#7CAA68",
}

MODEL_LABELS = {
    "claude": "Claude",
    "gpt": "GPT-4o",
    "gpt_oss": "GPT-OSS",
}

MODELS = list(MODEL_LABELS.keys())


def plot_model(model: str, lookup: dict, out_path: Path) -> None:
    """Plot all envs for a single model."""

    env_labels = list(ENV_MAP.keys())
    n_envs = len(env_labels)

    # Pre-compute y positions and gather data
    y_positions = []
    all_scores = []
    draw_items = []  # (y, score, color)
    group_centres = []

    y = 0
    for env_label in env_labels:
        reasoning_env, knowledge_env, knowledge_new_env = ENV_MAP[env_label]

        bars_in_group = 0

        # Knowledge (new) — on top
        if knowledge_new_env is not None:
            score = lookup.get(("knowledge", knowledge_new_env, model))
            if score is not None:
                draw_items.append((y, score, SERIES_COLORS["Knowledge (new)"]))
                y_positions.append(y)
                all_scores.append(score)
                bars_in_group += 1
                y += 1

        # Knowledge
        score = lookup.get(("knowledge", knowledge_env, model))
        if score is not None:
            draw_items.append((y, score, SERIES_COLORS["Knowledge"]))
            y_positions.append(y)
            all_scores.append(score)
            bars_in_group += 1
            y += 1

        # Reasoning — at bottom
        score = lookup.get(("reasoning", reasoning_env, model))
        if score is not None:
            draw_items.append((y, score, SERIES_COLORS["Reasoning"]))
            y_positions.append(y)
            all_scores.append(score)
            bars_in_group += 1
            y += 1

        # Group label at centre
        if bars_in_group > 0:
            centre = y - 1 - (bars_in_group - 1) / 2
            group_centres.append((centre, env_label))

        y += 0.6  # gap between groups

    if not all_scores:
        logger.info("No data for %s, skipping.", model)
        return

    fig, ax = plt.subplots(
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 0.5 + n_envs * 0.45)
    )

    for yi, score, color in draw_items:
        ax.hlines(yi, 0, score, color=color, alpha=0.2, linewidth=5)
        ax.plot(score, yi, "o", markersize=5, color=color, alpha=0.6)

    ax.set_yticks([c for c, _ in group_centres])
    ax.set_yticklabels([lbl for _, lbl in group_centres])
    ax.set_xlabel("Overall score")
    ax.set_title(MODEL_LABELS[model])

    # Legend
    for label, color in SERIES_COLORS.items():
        ax.plot([], [], "o", color=color, label=label)
    ax.legend(loc="lower right", frameon=False)

    range_frame(ax, np.array([0, 1]), np.array(y_positions))
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_path.relative_to(ROOT))
    logger.info("Saved %s", png_path.relative_to(ROOT))


def main():
    with SCORES_PATH.open() as f:
        data = json.load(f)

    # Build lookup: (category, env, model) -> score
    lookup = {
        (s["category"], s["environment"], s["model"]): s["overall_score"] for s in data
    }

    for model in MODELS:
        out_path = OUT_DIR / f"{model}_scores.pdf"
        plot_model(model, lookup, out_path)


if __name__ == "__main__":
    main()
