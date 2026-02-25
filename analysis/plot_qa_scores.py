"""
Create horizontal lollipop-style bar plots for QA scores.

Produces two figures:
  - analysis/reasoning_scores.pdf
  - analysis/knowledge_scores.pdf
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

MODEL_COLORS = {
    "claude": "#E07B39",
    "gpt": "#5B8DB8",
    "gpt_oss": "#7CAA68",
}

MODEL_LABELS = {
    "claude": "Claude",
    "gpt": "GPT-4o",
    "gpt_oss": "GPT-OSS",
}


def plot_category(scores: list[dict], category: str, out_path: Path) -> None:
    """Plot a single category (reasoning or knowledge)."""
    # Group by environment, preserving order
    envs: list[str] = []
    for s in scores:
        if s["environment"] not in envs:
            envs.append(s["environment"])

    models = list(MODEL_COLORS.keys())
    n_envs = len(envs)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT + n_envs * 0.15))

    # Build lookup: (env, model) -> score
    lookup = {(s["environment"], s["model"]): s["overall_score"] for s in scores}

    y_positions = []
    y_labels = []
    all_scores = []

    for env_idx, env in enumerate(envs):
        for m_idx, model in enumerate(models):
            score = lookup.get((env, model))
            if score is None:
                continue
            i = env_idx * (n_models + 0.5) + m_idx
            color = MODEL_COLORS[model]

            ax.hlines(
                i,
                0,
                score,
                color=color,
                alpha=0.2,
                linewidth=5,
            )
            ax.plot(
                score,
                i,
                "o",
                markersize=5,
                color=color,
                alpha=0.6,
            )
            y_positions.append(i)
            all_scores.append(score)

        # Label at the centre of the group
        centre = env_idx * (n_models + 0.5) + (n_models - 1) / 2
        y_labels.append((centre, env))

    ax.set_yticks([pos for pos, _ in y_labels])
    ax.set_yticklabels([label for _, label in y_labels])
    ax.set_xlabel("Overall score")
    ax.set_title(f"{category.capitalize()} QA scores")

    # Legend
    for model in models:
        ax.plot([], [], "o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
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

    reasoning = [s for s in data if s["category"] == "reasoning"]
    knowledge = [s for s in data if s["category"] == "knowledge"]

    if reasoning:
        plot_category(reasoning, "reasoning", OUT_DIR / "reasoning_scores.pdf")
    else:
        logger.info("No reasoning scores found.")

    if knowledge:
        plot_category(knowledge, "knowledge", OUT_DIR / "knowledge_scores.pdf")
    else:
        logger.info("No knowledge scores found.")


if __name__ == "__main__":
    main()
