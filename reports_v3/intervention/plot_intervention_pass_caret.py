"""Plot Pass^k vs k averaged over ReAct & ToolCalling.

2x3 grid: top row = success interventions, bottom row = failed interventions.
Columns = environments. Baseline shown in both rows as reference.
Opacity decreases for steps further from baseline.
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
AGENTS = ["react", "toolcalling"]

FAILED_STEPS = ["failed_step1", "failed_step2", "failed_stepn1", "failed_stepn2"]
SUCCESS_STEPS = ["success_step1", "success_step2", "success_stepn1", "success_stepn2"]

FAILED_COLOR = "#d62728"
SUCCESS_COLOR = "#1f77b4"
BASELINE_COLOR = "#333333"

STEP_ALPHA = {
    "step1": 0.85,
    "step2": 0.6,
    "stepn1": 0.4,
    "stepn2": 0.25,
}

STEP_MARKERS = {
    "step1": "s",
    "step2": "D",
    "stepn1": "^",
    "stepn2": "v",
}


def load_metrics(env: str, agent: str, step: str) -> dict | None:
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        return json.load(f)["metrics"]


def extract_pass_caret(metrics: dict) -> tuple[list[int], list[float]]:
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass^{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def avg_pass_caret(env: str, step: str) -> tuple[list[int], list[float]] | None:
    all_vals = []
    for agent in AGENTS:
        m = load_metrics(env, agent, step)
        if m is None:
            continue
        ks, vals = extract_pass_caret(m)
        all_vals.append(vals)
    if not all_vals:
        return None
    avg = np.mean(all_vals, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def plot_row(axes_row, environments, steps, step_color, row_label):
    for col, env in enumerate(environments):
        ax = axes_row[col]
        all_k_vals = []
        all_y_vals = []

        # Baseline reference
        result = avg_pass_caret(env, "baseline")
        if result:
            ks, vals = result
            ax.plot(
                ks,
                vals,
                color=BASELINE_COLOR,
                marker="o",
                markersize=4,
                label="Baseline",
                linewidth=1.8,
                alpha=1.0,
                zorder=10,
            )
            all_k_vals.extend(ks)
            all_y_vals.extend(vals)

        # Intervention steps
        for step_name in steps:
            prefix = step_name.split("_")[0] + "_"
            suffix = step_name.replace(prefix, "")
            result = avg_pass_caret(env, step_name)
            if result is None:
                continue
            ks, vals = result
            alpha = STEP_ALPHA[suffix]
            marker = STEP_MARKERS[suffix]
            label = suffix.replace("step", "Step ")
            ax.plot(
                ks,
                vals,
                color=step_color,
                marker=marker,
                markersize=4,
                label=label,
                linewidth=1.3,
                alpha=alpha,
            )
            all_k_vals.extend(ks)
            all_y_vals.extend(vals)

        ax.set_title(env.capitalize())
        if col == 0:
            ax.set_ylabel(f"{row_label}\nPass^k (avg ReAct + TC)")

        range_frame(ax, np.array(all_k_vals), np.array(all_y_vals), pad=0.05)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    plot_row(axes[0], ENVIRONMENTS, SUCCESS_STEPS, SUCCESS_COLOR, "Success")
    plot_row(axes[1], ENVIRONMENTS, FAILED_STEPS, FAILED_COLOR, "Failed")

    for ax in axes[1]:
        ax.set_xlabel("k")
    # Remove duplicate titles on bottom row
    for ax in axes[1]:
        ax.set_title("")

    fig.tight_layout()

    out_path = Path(__file__).parent / "intervention_pass_caret_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
