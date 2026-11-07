"""
Panel 2 — Mean log-probability per environment group.

Averages the per-environment mean logprobs within each group
(Hypothesis-driven inquiry, Strategic reasoning, Workflow construction).
Gradient coloring from #BF092F (least negative) to #16476A (most negative).
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import ENVIRONMENT_GROUPS, FONT_SIZES
from plot_utils import load_logprobs_stats

GROUP_COLORS = {
    "Hypothesis-driven inquiry": "#8B5CF6",
    "Strategic reasoning": "#E07A5F",
    "Workflow construction": "#4C78A8",
}

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "panel2_logprobs_grouped.pdf"
OUT_FILE_1COL = OUT_DIR / "panel2_logprobs_grouped_1col.pdf"


# ── Data ─────────────────────────────────────────────────────────────────────


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group stats: average the per-environment token-weighted means."""
    rows = []
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        env_means = []
        for env in group_info["environments"]:
            grp = df[df["environment"] == env]
            if grp.empty:
                continue
            total_count = int(grp["logprob_count"].sum())
            if total_count == 0:
                continue
            env_means.append(float(grp["logprob_sum"].sum()) / total_count)

        if env_means:
            rows.append(
                {
                    "group": group_name,
                    "mean": float(np.mean(env_means)),
                    "n_envs": len(env_means),
                }
            )

    stats = (
        pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    )
    stats["color"] = stats["group"].map(GROUP_COLORS)
    return stats


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_grouped_logprobs(stats: pd.DataFrame, output_path: Path, width=TWO_COL_WIDTH):
    """Horizontal barplot of mean logprob per environment group."""
    fig, ax = plt.subplots(figsize=(width, ONE_COL_HEIGHT))

    labels = stats["group"].tolist()
    values = stats["mean"].tolist()
    colors = stats["color"].tolist()

    y_pos = np.arange(len(values))

    bars = ax.barh(y_pos, values, color=colors, height=0.35)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="right",
            fontsize=FONT_SIZES["tick_label"],
        )

    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
    ax.set_xlabel(
        "Mean log-probability",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )

    range_frame(ax, np.array([min(values), 0]), y_pos, pad=0.15)

    # Set after range_frame so it doesn't get overridden
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading logprobs data …")
    logprobs_df = load_logprobs_stats()
    logger.info(
        f"  {len(logprobs_df):,} rows | environments: "
        f"{sorted(logprobs_df['environment'].unique())}"
    )

    stats = compute_group_stats(logprobs_df)
    logger.info("\nGroup stats (mean logprob across environments):")
    for _, row in stats.iterrows():
        logger.info(
            f"  {row['group']:30s}  mean={row['mean']:+.4f}  n_envs={row['n_envs']}"
        )

    plot_grouped_logprobs(stats, OUT_FILE)
    plot_grouped_logprobs(stats, OUT_FILE_1COL, width=ONE_COL_WIDTH)


if __name__ == "__main__":
    main()
