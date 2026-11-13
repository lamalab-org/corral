"""Baseline Pass@k and Pass^k averaged over environment groups.

Groups environments into cognitive-task categories and averages
the matched-baseline metrics across all environments within each group.

Reads from results/data/intervention_reports.jsonl (downloaded from HF).
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from intervention_utils import avg_matched_baseline, load_reports
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

_SCRIPT_DIR = Path(__file__).resolve().parent

ENVIRONMENT_GROUPS = {
    "Hypothesis-driven inquiry": {
        "description": "Reason from observations to hidden structure",
        "environments": ["spectra", "wetlab", "resistor"],
    },
    "Strategic reasoning": {
        "description": "Navigate combinatorial spaces under constraints",
        "environments": ["retrosynthesis"],
    },
    "Workflow construction": {
        "description": "Assemble and execute computational protocols",
        "environments": ["catalyst", "md", "ml"],
    },
}

GROUP_COLORS = {
    "Hypothesis-driven inquiry": "#8B5CF6",
    "Strategic reasoning": "#E07A5F",
    "Workflow construction": "#4C78A8",
}


def _filter_available_envs(reports, envs):
    """Return only environments that exist in the data."""
    available = set(reports["environment"].unique())
    return [e for e in envs if e in available]


def _avg_group_baseline(group_envs, metric_type, reports):
    """Average the matched-baseline metric across environments in a group."""
    all_vals = []
    for env in group_envs:
        result = avg_matched_baseline(env, metric_type=metric_type, df=reports)
        if result is None:
            continue
        _, vals = result
        all_vals.append(vals)
    if not all_vals:
        return None
    min_len = min(len(v) for v in all_vals)
    truncated = [v[:min_len] for v in all_vals]
    avg = np.mean(truncated, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    logger.info(f"Saved to {out_path}")
    plt.close(fig)


def _plot_grouped_metric(ax, metric_type, ylabel, reports, annotate_lines=False):
    all_k, all_y = [], []
    line_ends = []  # (x, y, gname, color) for inline labels
    for gname, ginfo in ENVIRONMENT_GROUPS.items():
        envs = _filter_available_envs(reports, ginfo["environments"])
        if not envs:
            continue
        result = _avg_group_baseline(envs, metric_type, reports)
        if result is None:
            continue
        ks, vals = result
        ax.plot(
            ks,
            vals,
            color=GROUP_COLORS[gname],
            label=gname,
            linewidth=1.5,
        )
        all_k.extend(ks)
        all_y.extend(vals)
        line_ends.append((ks[-1], vals[-1], gname, GROUP_COLORS[gname]))

    if all_k and all_y:
        range_frame(ax, np.array(all_k), np.array(all_y), pad=0.05)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)

    if annotate_lines:
        # Sort by y so labels stack predictably, then stagger
        line_ends.sort(key=lambda t: t[1])
        for i, (x, y, gname, color) in enumerate(line_ends):
            ax.annotate(
                gname,
                xy=(x, y),
                xytext=(4, (i - 1) * 10),
                textcoords="offset points",
                color=color,
                fontsize=5.5,
                va="center",
                ha="left",
            )


def main():
    reports = load_reports()
    out_dir = _SCRIPT_DIR / "results" / "figures" / "intervention"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Two-col: Pass@k + Pass^k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.5 * ONE_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_grouped_metric(ax1, "pass_at", "Pass@k", reports, annotate_lines=True)
    _plot_grouped_metric(ax2, "pass_caret", "Pass^k", reports, annotate_lines=True)
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_grouped_twocol.png")

    # 2. One-col: Pass@k only
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_grouped_metric(ax, "pass_at", "Pass@k", reports, annotate_lines=True)
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_grouped_pass_at.png")

    # 3. One-col: Pass^k only
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))
    _plot_grouped_metric(ax, "pass_caret", "Pass^k", reports, annotate_lines=True)
    fig.tight_layout()
    _save_fig(fig, out_dir / "baseline_grouped_pass_caret.png")


if __name__ == "__main__":
    main()
