"""Compact 1x2 intervention plot: Pass@k (left) and Pass^k (right).

Averaged across all environments and both agents.
Lines use a blue→red gradient ordered by intervention strength:
  success_stepn1 (blue) → … → failed_stepn1 (red).
"""

import json
import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import avg_matched_baseline  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
ENVIRONMENTS = ["spectra", "wetlab", "resistor", "ml"]
AGENTS = ["react", "toolcalling"]

# Ordered from "most success info" (blue) to "most failure info" (red)
ORDERED_STEPS = [
    "success_stepn1",
    "success_stepn2",
    "success_step2",
    "success_step1",
    "failed_step1",
    "failed_step2",
    "failed_stepn2",
    "failed_stepn1",
]

STEP_LABELS = {
    "success_stepn1": "S: Step n-1",
    "success_stepn2": "S: Step n-2",
    "success_step2": "S: Step 2",
    "success_step1": "S: Step 1",
    "failed_step1": "F: Step 1",
    "failed_step2": "F: Step 2",
    "failed_stepn2": "F: Step n-2",
    "failed_stepn1": "F: Step n-1",
}

STEP_MARKERS = {
    "success_stepn1": "^",
    "success_stepn2": "v",
    "success_step2": "D",
    "success_step1": "s",
    "failed_step1": "s",
    "failed_step2": "D",
    "failed_stepn2": "v",
    "failed_stepn1": "^",
}

BLUE = "#16476A"
RED = "#BF092F"
BASELINE_COLOR = "#7A7A7A"


def make_gradient(n: int) -> list[str]:
    """Create n colours evenly spaced from BLUE to RED."""
    cmap = mcolors.LinearSegmentedColormap.from_list("br", [BLUE, RED], N=256)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def load_metrics(env: str, agent: str, step: str) -> dict | None:
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with report_glob[0].open() as f:
        return json.load(f)["metrics"]


def extract(metrics: dict, prefix: str) -> tuple[list[int], list[float]]:
    """Extract Pass@k or Pass^k series from a metrics dict."""
    ks, vals = [], []
    for k in range(1, 16):
        key = f"{prefix}{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def avg_step(
    env_list: list[str], step: str, prefix: str
) -> tuple[list[int], list[float]] | None:
    """Average a metric across envs x agents for one intervention step."""
    all_vals = []
    for env in env_list:
        for agent in AGENTS:
            m = load_metrics(env, agent, step)
            if m is None:
                continue
            _, vals = extract(m, prefix)
            if vals:
                all_vals.append(vals)
    if not all_vals:
        return None
    min_len = min(len(v) for v in all_vals)
    truncated = [v[:min_len] for v in all_vals]
    avg = np.mean(truncated, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def avg_baseline(
    env_list: list[str], metric_type: str
) -> tuple[list[int], list[float]] | None:
    """Average matched baseline across envs."""
    all_vals = []
    for env in env_list:
        result = avg_matched_baseline(env, metric_type=metric_type)
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


def main():
    colors = make_gradient(len(ORDERED_STEPS))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT),
        sharey=False,
    )

    panels = [
        (axes[0], "Pass@", "pass_at", "Pass@k"),
        (axes[1], "Pass^", "pass_caret", "Pass^k"),
    ]

    for ax, prefix, metric_type, ylabel in panels:
        all_k, all_y = [], []

        # Baseline
        result = avg_baseline(ENVIRONMENTS, metric_type)
        if result:
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=BASELINE_COLOR,
                marker="o",
                markersize=3,
                label="Baseline",
                linewidth=1.6,
                zorder=10,
                linestyle="--",
            )
            all_k.extend(ks)
            all_y.extend(vals)

        # Intervention steps
        for step, color in zip(ORDERED_STEPS, colors, strict=False):
            result = avg_step(ENVIRONMENTS, step, prefix)
            if result is None:
                continue
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=color,
                marker=STEP_MARKERS[step],
                markersize=3,
                label=STEP_LABELS[step],
                linewidth=1.2,
            )
            all_k.extend(ks)
            all_y.extend(vals)

        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        if all_k and all_y:
            range_frame(ax, np.array(all_k), np.array(all_y), pad=0.05)

    # Single legend to the right
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center right",
            fontsize=5.5,
            framealpha=0.9,
            bbox_to_anchor=(1.18, 0.5),
        )

    fig.tight_layout()
    out = Path(__file__).parent / "figures" / "intervention_compact.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf")
    logger.info(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
