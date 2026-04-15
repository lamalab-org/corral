"""
Panel 2d — Mean log-probability per environment.

Hypothesis: in common-domain environments (ml, md) the model assigns higher
(less negative) token log-probabilities than in specialised domains (spectra,
retro), reflecting greater familiarity with the vocabulary and reasoning steps.

Exactly-zero logprob tokens are excluded because they are for special tokens like <|endoftext|> or <|im_end|>.
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import FONT_SIZES
from plot_utils import load_logprobs_stats

ENVIRONMENT_NAMES = {
    "afm": "AFM Experiment Execution",
    "catalyst": "Adsorption Surface Construction",
    "md": "Molecular Simulation",
    "ml": "ML-based Property Prediction",
    "resistor": "Circuit Inference",
    "retro": "Retrosynthetic Planning",
    "spectra": "Spectroscopic Structure Elucidation",
    "wetlab": "Inorganic Qualitative Analysis",
}


lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "panel2_logprobs.pdf"


# ── Data ─────────────────────────────────────────────────────────────────────


def compute_env_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-environment token-weighted mean from pre-aggregated (sum, count) rows."""
    rows = []
    for env, grp in df.groupby("environment"):
        total_count = int(grp["logprob_count"].sum())
        if total_count == 0:
            continue
        mean = float(grp["logprob_sum"].sum()) / total_count
        rows.append(
            {
                "environment": env,
                "display_name": ENVIRONMENT_NAMES.get(env, env),
                "mean": mean,
                "n_tokens": total_count,
                "color": "#7150e0",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean", ascending=False)  # least negative on top
        .reset_index(drop=True)
    )


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_mean_logprobs(stats: pd.DataFrame, output_path: Path):
    """Horizontal barplot of mean logprob per environment (token-pooled)."""
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    labels = stats["display_name"].tolist()
    values = stats["mean"].tolist()
    colors = stats["color"].tolist()

    y_pos = np.arange(len(values))

    bars = ax.barh(
        y_pos,
        values,
        color=colors,
        height=0.6,
    )

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
    logprobs_df = load_logprobs_stats()
    logger.info(
        f"  {len(logprobs_df):,} rows | environments: {sorted(logprobs_df['environment'].unique())}"
    )

    stats = compute_env_stats(logprobs_df)
    logger.info("\nEnvironment stats (mean logprob, non-zero tokens):")
    for _, row in stats.iterrows():
        logger.info(
            f"  {row['environment']:12s}  mean={row['mean']:+.4f}"
            f"  n_tokens={row['n_tokens']:,}"
        )

    plot_mean_logprobs(stats, OUT_FILE)


if __name__ == "__main__":
    main()
