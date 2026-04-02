"""Cleveland dot plot: % change in Pass@1 vs baseline, per environment and step.

Single plot. Environments on y-axis, % change on x-axis.
Failed steps in red, success steps in blue. Marker/alpha encodes step distance.
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

RUNS_DIR = Path(__file__).parent / "runs"

ENVIRONMENTS = ["spectra", "wetlab", "resistor"]
ENV_LABELS = {"spectra": "Spectra", "wetlab": "Wetlab", "resistor": "Resistor"}
AGENTS = ["react", "toolcalling"]

STEPS = [
    "failed_stepn2",
    "failed_stepn1",
    "failed_step2",
    "failed_step1",
    "success_step1",
    "success_step2",
    "success_stepn1",
    "success_stepn2",
]

FAILED_COLOR = "#d62728"
SUCCESS_COLOR = "#1f77b4"

STEP_ALPHA = {
    "step1": 1.0,
    "step2": 0.75,
    "stepn1": 0.5,
    "stepn2": 0.35,
}

STEP_MARKERS = {
    "step1": "o",
    "step2": "s",
    "stepn1": "D",
    "stepn2": "^",
}

STEP_SIZES = {
    "step1": 50,
    "step2": 45,
    "stepn1": 40,
    "stepn2": 35,
}


def load_pass1(env: str, agent: str, step: str) -> float | None:
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        return json.load(f)["metrics"].get("Pass@1")


def avg_pass1(env: str, step: str) -> float | None:
    vals = []
    for agent in AGENTS:
        v = load_pass1(env, agent, step)
        if v is not None:
            vals.append(v)
    return np.mean(vals) if vals else None


def main():
    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH * 0.65, TWO_COL_HEIGHT * 0.55))

    y_positions = {env: i for i, env in enumerate(ENVIRONMENTS)}
    n_env = len(ENVIRONMENTS)

    # Vertical offset per step so dots don't overlap
    step_offsets = {}
    n_steps = len(STEPS)
    for i, step in enumerate(STEPS):
        step_offsets[step] = (i - (n_steps - 1) / 2) * 0.06

    # Zero reference line
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="-", zorder=1)

    # Track which labels we've added to legend
    legend_added = set()

    for env in ENVIRONMENTS:
        baseline = avg_pass1(env, "baseline")
        if baseline is None or baseline == 0:
            continue

        y_base = y_positions[env]

        for step in STEPS:
            val = avg_pass1(env, step)
            if val is None:
                continue

            pct_change = ((val - baseline) / baseline) * 100

            is_failed = step.startswith("failed_")
            suffix = step.replace("failed_", "").replace("success_", "")
            color = FAILED_COLOR if is_failed else SUCCESS_COLOR
            alpha = STEP_ALPHA[suffix]
            marker = STEP_MARKERS[suffix]
            size = STEP_SIZES[suffix]

            y = y_base + step_offsets[step]

            # Legend label
            short = suffix.replace("step", "Step ")
            prefix_label = "F" if is_failed else "S"
            label_key = f"{prefix_label}: {short}"
            label = label_key if label_key not in legend_added else None
            if label:
                legend_added.add(label_key)

            ax.scatter(
                pct_change,
                y,
                color=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=label,
                zorder=5,
                edgecolors="white",
                linewidths=0.3,
            )
            # Connecting line to zero
            ax.plot(
                [0, pct_change],
                [y, y],
                color=color,
                alpha=alpha * 0.5,
                linewidth=0.8,
                zorder=2,
            )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([ENV_LABELS[e] for e in ENVIRONMENTS])
    ax.set_xlabel("% Change in Pass@1 vs Baseline")

    # Collect all x values for range_frame
    all_x = []
    for env in ENVIRONMENTS:
        baseline = avg_pass1(env, "baseline")
        if baseline is None or baseline == 0:
            continue
        for step in STEPS:
            val = avg_pass1(env, step)
            if val is not None:
                all_x.append(((val - baseline) / baseline) * 100)

    range_frame(
        ax,
        np.array(all_x),
        np.array(list(y_positions.values()) * (len(all_x) // n_env + 1))[: len(all_x)],
        pad=0.1,
        nice=False,
    )
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([ENV_LABELS[e] for e in ENVIRONMENTS])

    ax.legend(loc="lower right", fontsize=6, framealpha=0.9, ncol=2)

    fig.tight_layout()

    out_path = Path(__file__).parent / "intervention_dotplot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
