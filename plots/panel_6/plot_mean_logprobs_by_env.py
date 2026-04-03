"""
Panel 6 — Mean log-probability per environment.

Hypothesis: in common-domain environments (ml, md) the model assigns higher
(less negative) token log-probabilities than in specialised domains (spectra,
retro), reflecting greater familiarity with the vocabulary and reasoning steps.

Exactly-zero logprob tokens are excluded because they are for special tokens like <|endoftext|> or <|im_end|>.
"""

import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "analysis"))

from plot_config import (  # noqa: E402
    FONT_SIZES,
)
from plot_utils import load_logprobs_data  # noqa: E402

ENVIRONMENT_NAMES = {
    "afm": "AFM Experiment\nExecution",
    "catalyst": "Adsorption Surface\nConstruction",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "resistor": "Circuit\nInference",
    "retro": "Retrosynthetic\nPlanning",
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
}


lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "mean_logprobs_by_env.png"


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


def compute_env_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-environment stats from a flat pool of all non-zero tokens.

    Every token gets equal weight regardless of message length.
    std is the std of individual token logprobs (spread of the distribution).

    Returns DataFrame sorted descending by mean (least negative first = top).
    Columns: environment, display_name, mean, std, n_tokens, color
    """
    rows = []
    for env, grp in df.groupby("environment"):
        tokens = _pool_nonzero_tokens(grp["per_token_logprob"])
        if tokens.size == 0:
            continue
        rows.append(
            {
                "environment": env,
                "display_name": ENVIRONMENT_NAMES.get(env, env).replace("\n", " "),
                "mean": float(np.mean(tokens)),
                "n_tokens": int(tokens.size),
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
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

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
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="right",
            # color="white",
            fontsize=FONT_SIZES["tick_label"] - 1,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick_label"])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel(
        "Mean log-probability",
        fontsize=FONT_SIZES["axis_label"],
    )

    range_frame(ax, np.array([min(values), 0]), y_pos, pad=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading logprobs data …")
    logprobs_df = load_logprobs_data()
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
