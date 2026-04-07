"""Plot Pass^k vs k averaged over ReAct & ToolCalling.

2x3 grid: top row = success interventions, bottom row = failed interventions.
Columns = environments. Baseline shown in both rows as reference.
Opacity decreases for steps further from baseline.
"""

import json
import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import avg_matched_baseline, extract_matched_pass_caret

RUNS_DIR = Path(__file__).parent.parent / "runs"

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
AGENTS = ["react", "toolcalling"]

FAILED_STEPS = ["failed_step1", "failed_step2", "failed_stepn1", "failed_stepn2"]
SUCCESS_STEPS = ["success_step1", "success_step2", "success_stepn1", "success_stepn2"]

FAILED_COLOR = "#BF092F"
SUCCESS_COLOR = "#16476A"
BASELINE_COLOR = "#7A7A7A"

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
    min_len = min(len(v) for v in all_vals)
    truncated = [v[:min_len] for v in all_vals]
    avg = np.mean(truncated, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}


def plot_row(axes_row, environments, steps, step_color, row_label):
    for col, env in enumerate(environments):
        ax = axes_row[col]
        all_k_vals = []
        all_y_vals = []

        # Baseline reference (matched to intervention tasks)
        result = avg_matched_baseline(env, metric_type="pass_caret")
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

        ax.set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{row_label}\nPass^k")

        if all_k_vals and all_y_vals:
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#999999",
            )
            ax.set_xticks([])
            ax.set_yticks([])


AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}


def plot_row_agent(axes_row, environments, agent, steps, step_color, row_label):
    """Like plot_row but for a single agent (no averaging)."""
    for col, env in enumerate(environments):
        ax = axes_row[col]
        all_k_vals = []
        all_y_vals = []

        # Baseline for this agent (matched to intervention tasks)
        result = extract_matched_pass_caret(env, agent)
        if result is not None:
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

        # Intervention steps for this agent
        for step_name in steps:
            prefix = step_name.split("_")[0] + "_"
            suffix = step_name.replace(prefix, "")
            m = load_metrics(env, agent, step_name)
            if m is None:
                continue
            ks, vals = extract_pass_caret(m)
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

        if col == 0:
            ax.set_ylabel(f"{row_label}\nPass^k")

        if all_k_vals and all_y_vals:
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper right", fontsize=5, framealpha=0.9)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#999999",
            )
            ax.set_xticks([])
            ax.set_yticks([])


def main():
    n_envs = len(ENVIRONMENTS)

    # --- Averaged version (2 rows) ---
    fig, axes = plt.subplots(
        2,
        n_envs,
        figsize=(TWO_COL_WIDTH * n_envs / 3, TWO_COL_HEIGHT),
        sharey=True,
    )

    plot_row(axes[0], ENVIRONMENTS, SUCCESS_STEPS, SUCCESS_COLOR, "Success")
    plot_row(axes[1], ENVIRONMENTS, FAILED_STEPS, FAILED_COLOR, "Failed")

    for ax in axes[1]:
        ax.set_xlabel("k")
    for ax in axes[1]:
        ax.set_title("")

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / "intervention_pass_caret_plot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    print(f"Saved to {out_path}")
    plt.close(fig)

    # --- Per-agent version (4 rows) ---
    fig, axes = plt.subplots(
        4,
        n_envs,
        figsize=(TWO_COL_WIDTH, 1.5 * TWO_COL_HEIGHT),
        sharey=True,
    )

    plot_row_agent(
        axes[0], ENVIRONMENTS, "react", SUCCESS_STEPS, SUCCESS_COLOR, "Success\n(ReAct)"
    )
    plot_row_agent(
        axes[1],
        ENVIRONMENTS,
        "toolcalling",
        SUCCESS_STEPS,
        SUCCESS_COLOR,
        "Success\n(ToolCalling)",
    )
    plot_row_agent(
        axes[2], ENVIRONMENTS, "react", FAILED_STEPS, FAILED_COLOR, "Failed\n(ReAct)"
    )
    plot_row_agent(
        axes[3],
        ENVIRONMENTS,
        "toolcalling",
        FAILED_STEPS,
        FAILED_COLOR,
        "Failed\n(ToolCalling)",
    )

    # Titles only on top row
    for col, env in enumerate(ENVIRONMENTS):
        axes[0, col].set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
    for row in range(1, 4):
        for ax in axes[row]:
            ax.set_title("")

    # x-labels only on bottom row
    for ax in axes[3]:
        ax.set_xlabel("k")

    fig.tight_layout()

    out_path = (
        Path(__file__).parent / "figures" / "intervention_pass_caret_per_agent.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
