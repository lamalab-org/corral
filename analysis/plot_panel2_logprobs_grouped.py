"""
Panel 2 — Mean log-probability per environment group.

Averages the per-environment mean logprobs within each group
(Hypothesis-driven inquiry, Strategic reasoning, Workflow construction).
Gradient coloring from #BF092F (least negative) to #16476A (most negative).
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import ENVIRONMENT_GROUPS, FONT_SIZES
from plot_utils import load_logprobs_data

COLOR_HIGH = "#BF092F"  # largest (least negative) mean
COLOR_LOW = "#16476A"  # lowest (most negative) mean

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "panel2_logprobs_grouped.pdf"


# ── Data ─────────────────────────────────────────────────────────────────────


def _pool_nonzero_tokens(series) -> np.ndarray:
    """Concatenate all non-zero, finite token logprobs across all messages."""
    arrays = []
    for lp in series:
        if not isinstance(lp, list | np.ndarray) or len(lp) == 0:
            continue
        arr = np.asarray(lp, dtype=np.float32)
        arr = arr[np.isfinite(arr) & (arr != 0.0)]
        if arr.size > 0:
            arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def _make_gradient(n: int):
    """Return n colors linearly interpolated from COLOR_HIGH to COLOR_LOW."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", [COLOR_HIGH, COLOR_LOW], N=n
    )
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group stats: average the per-environment mean logprobs within each group."""
    rows = []
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        env_means = []
        for env in group_info["environments"]:
            grp = df[df["environment"] == env]
            if grp.empty:
                continue
            tokens = _pool_nonzero_tokens(grp["per_token_logprob"])
            if tokens.size == 0:
                continue
            env_means.append(float(np.mean(tokens)))

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
    stats["color"] = _make_gradient(len(stats))
    return stats


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_grouped_logprobs(stats: pd.DataFrame, output_path: Path):
    """Horizontal barplot of mean logprob per environment group."""
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    labels = stats["group"].tolist()
    values = stats["mean"].tolist()
    colors = stats["color"].tolist()

    y_pos = np.arange(len(values))

    bars = ax.barh(y_pos, values, color=colors, height=0.6)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="right",
            fontsize=FONT_SIZES["tick_label"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
    ax.set_xlabel(
        "Mean log-probability",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )

    range_frame(ax, np.array([min(values), 0]), y_pos, pad=0.15)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading logprobs data …")
    logprobs_df = load_logprobs_data()
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


if __name__ == "__main__":
    main()
