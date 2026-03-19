"""
Panel 6 — Correlation between mean log-probability and task scores.

Scatter plots showing per-environment mean logprob (token-pooled, non-zero)
against three score sources:
  (a) task score from the logprobs dataset itself,
  (b) Average Score from the main benchmark reports,
  (c) QA overall_score from the QA topic reports.

Each point is one environment; a linear fit + Pearson r is overlaid.
"""

import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from adjustText import adjust_text
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "analysis"))

from plot_config import FONT_SIZES  # noqa: E402
from plot_utils import load_logprobs_data, load_qa_data, load_reports_data  # noqa: E402

ENVIRONMENT_NAMES = {
    "afm": "AFM operation",
    "catalyst": "Surface construction",
    "md": "Molecular simulation",
    "ml": "Build property predictor",
    "resistor": "Circuit Inference",
    "retro": "Retrosynthetic planning",
    "spectra": "Spectroscopic elucidation",
    "wetlab": "Qualitative analysis",
}

COLOUR = "#7150e0"

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "logprob_score_correlation.png"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pool_nonzero_tokens(series) -> np.ndarray:
    """Concatenate all non-zero, finite token logprobs across all messages."""
    arrays = []
    for lp in series:
        if not isinstance(lp, (list, np.ndarray)) or len(lp) == 0:
            continue
        arr = np.asarray(lp, dtype=np.float32)
        arr = arr[np.isfinite(arr) & (arr != 0.0)]
        if arr.size > 0:
            arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def _env_mean_logprob(df_logprobs: pd.DataFrame) -> dict[str, float]:
    """Return {env: mean_logprob} from the logprobs dataset."""
    result = {}
    for env, grp in df_logprobs.groupby("environment"):
        tokens = _pool_nonzero_tokens(grp["per_token_logprob"])
        if tokens.size > 0:
            result[env] = float(np.mean(tokens))
    return result


# ── Score computations ───────────────────────────────────────────────────────


def _logprobs_task_scores(df_logprobs: pd.DataFrame) -> dict[str, float]:
    """Mean task score per environment from the logprobs dataset."""
    task_scores = (
        df_logprobs.groupby(["environment", "task", "trial"])["score"]
        .first()
        .reset_index()
    )
    return task_scores.groupby("environment")["score"].mean().to_dict()


def _reports_avg_scores(df_reports: pd.DataFrame) -> dict[str, float]:
    """Mean Average Score per environment from the benchmark reports."""
    return df_reports.groupby("environment")["Average Score"].mean().to_dict()


def _qa_overall_scores(df_qa: pd.DataFrame, qa_type: str = "qa") -> dict[str, float]:
    """Mean QA overall_score per environment for the given qa_type."""
    qa = df_qa[df_qa["qa_type"] == qa_type]
    return qa.groupby("env")["overall_score"].mean().to_dict()


# ── Plot ─────────────────────────────────────────────────────────────────────


def _add_correlation_panel(
    ax,
    logprob_map: dict[str, float],
    score_map: dict[str, float],
    ylabel: str,
):
    """Draw one scatter panel: mean logprob (x) vs score (y)."""
    envs = sorted(set(logprob_map) & set(score_map))
    if len(envs) < 2:
        ax.set_visible(False)
        return

    x = np.array([logprob_map[e] for e in envs])
    y = np.array([score_map[e] for e in envs])

    ax.scatter(x, y, color=COLOUR, s=30, zorder=3)

    texts = []
    for e, xi, yi in zip(envs, x, y):
        label = ENVIRONMENT_NAMES.get(e, e)
        texts.append(
            ax.text(
                xi,
                yi,
                label,
                fontsize=FONT_SIZES["tick_label"] - 1,
            )
        )
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    # Linear fit + Pearson r
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_fit, slope * x_fit + intercept, color=COLOUR, alpha=0.4, lw=1)
    ax.text(
        0.05,
        0.95,
        f"r = {r_value:.2f}\np = {p_value:.3f}",
        transform=ax.transAxes,
        fontsize=FONT_SIZES["tick_label"],
        va="top",
    )

    ax.set_xlabel("Mean log-probability", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(labelsize=FONT_SIZES["tick_label"])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    range_frame(ax, x, y, pad=0.15)


def plot_correlation(
    logprob_map: dict[str, float],
    logprobs_scores: dict[str, float],
    reports_scores: dict[str, float],
    qa_scores: dict[str, float],
    reasoning_qa_scores: dict[str, float],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 4, figsize=(ONE_COL_WIDTH * 3, ONE_COL_WIDTH * 0.85))

    panels = [
        (axes[0], logprobs_scores, "GPT-OSS score"),
        (axes[1], reports_scores, "Benchmark score"),
        (axes[2], qa_scores, "QA score"),
        (axes[3], reasoning_qa_scores, "Reasoning QA score"),
    ]

    for ax, score_map, ylabel in panels:
        _add_correlation_panel(ax, logprob_map, score_map, ylabel)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved -> {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading datasets ...")
    df_logprobs = load_logprobs_data()
    df_reports = load_reports_data()
    df_qa = load_qa_data()

    logprob_map = _env_mean_logprob(df_logprobs)
    logprobs_scores = _logprobs_task_scores(df_logprobs)
    reports_scores = _reports_avg_scores(df_reports)
    qa_scores = _qa_overall_scores(df_qa, qa_type="qa")
    reasoning_qa_scores = _qa_overall_scores(df_qa, qa_type="reasoning_qa")

    logger.info("Per-env mean logprob:")
    for e in sorted(logprob_map):
        lp = logprob_map[e]
        lps = logprobs_scores.get(e, float("nan"))
        rs = reports_scores.get(e, float("nan"))
        qs = qa_scores.get(e, float("nan"))
        rqs = reasoning_qa_scores.get(e, float("nan"))
        logger.info(
            f"  {e:12s}  logprob={lp:+.4f}  gpt_oss={lps:.3f}"
            f"  benchmark={rs:.3f}  qa={qs:.3f}  reasoning_qa={rqs:.3f}"
        )

    plot_correlation(
        logprob_map, logprobs_scores, reports_scores,
        qa_scores, reasoning_qa_scores, OUT_FILE,
    )


if __name__ == "__main__":
    main()
