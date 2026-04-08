"""Plot Pass@k vs k — separate figures for success and failed interventions.

Averaged: 2×3 grid per figure (6 environments, 3 per row).
Per-agent: 4×3 grid per figure (2 agents × 2 env-rows, 3 cols each).
"""

import json
import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import avg_matched_baseline, extract_matched_pass_at

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
AGENTS = ["react", "toolcalling"]
N_COLS = 3

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

ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}

AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}


def load_metrics(env: str, agent: str, step: str) -> dict | None:
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        return json.load(f)["metrics"]


def extract_pass_at(metrics: dict) -> tuple[list[int], list[float]]:
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass@{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def avg_pass_at(env: str, step: str) -> tuple[list[int], list[float]] | None:
    all_vals = []
    for agent in AGENTS:
        m = load_metrics(env, agent, step)
        if m is None:
            continue
        ks, vals = extract_pass_at(m)
        all_vals.append(vals)
    if not all_vals:
        return None
    min_len = min(len(v) for v in all_vals)
    truncated = [v[:min_len] for v in all_vals]
    avg = np.mean(truncated, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def _plot_env_avg(ax, env, steps, step_color):
    """Plot a single env subplot for the averaged version."""
    all_k_vals, all_y_vals = [], []

    result = avg_matched_baseline(env, metric_type="pass_at")
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

    for step_name in steps:
        prefix = step_name.split("_")[0] + "_"
        suffix = step_name.replace(prefix, "")
        result = avg_pass_at(env, step_name)
        if result is None:
            continue
        ks, vals = result
        ax.plot(
            ks,
            vals,
            color=step_color,
            marker=STEP_MARKERS[suffix],
            markersize=4,
            label=suffix.replace("step", "Step "),
            linewidth=1.3,
            alpha=STEP_ALPHA[suffix],
        )
        all_k_vals.extend(ks)
        all_y_vals.extend(vals)

    return bool(all_k_vals and all_y_vals), all_k_vals, all_y_vals


def _plot_env_agent(ax, env, agent, steps, step_color):
    """Plot a single env subplot for a specific agent."""
    all_k_vals, all_y_vals = [], []

    result = extract_matched_pass_at(env, agent)
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

    for step_name in steps:
        prefix = step_name.split("_")[0] + "_"
        suffix = step_name.replace(prefix, "")
        m = load_metrics(env, agent, step_name)
        if m is None:
            continue
        ks, vals = extract_pass_at(m)
        ax.plot(
            ks,
            vals,
            color=step_color,
            marker=STEP_MARKERS[suffix],
            markersize=4,
            label=suffix.replace("step", "Step "),
            linewidth=1.3,
            alpha=STEP_ALPHA[suffix],
        )
        all_k_vals.extend(ks)
        all_y_vals.extend(vals)

    return bool(all_k_vals and all_y_vals), all_k_vals, all_y_vals


def _format_subplot(ax, env, has_data, all_k_vals=None, all_y_vals=None):
    """Apply common formatting to a subplot."""
    ax.set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
    if has_data and all_k_vals and all_y_vals:
        range_frame(
            ax,
            np.array(all_k_vals),
            np.array(all_y_vals),
            pad=0.05,
        )
    elif not has_data:
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


def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    print(f"Saved to {out_path}")
    plt.close(fig)


def plot_avg_figure(steps, step_color, label, filename):
    """Create a 2×3 averaged figure for one intervention type (success or failed)."""
    n_rows = len(ENVIRONMENTS) // N_COLS
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
        sharey=True,
    )

    for idx, env in enumerate(ENVIRONMENTS):
        row, col = idx // N_COLS, idx % N_COLS
        ax = axes[row, col]
        has_data, k_vals, y_vals = _plot_env_avg(ax, env, steps, step_color)
        _format_subplot(ax, env, has_data, all_k_vals=k_vals, all_y_vals=y_vals)
        if col == 0:
            ax.set_ylabel("Pass@k")

    for ax in axes[-1]:
        ax.set_xlabel("k")

    # Shared legend in bottom-right subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[-1, -1].legend(
            handles,
            labels,
            loc="lower right",
            fontsize=6,
            framealpha=0.9,
        )

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / filename
    _save_fig(fig, out_path)


def plot_per_agent_figure(steps, step_color, label, filename):
    """Create a 4×3 per-agent figure (2 agents × 2 env-rows)."""
    n_env_rows = len(ENVIRONMENTS) // N_COLS
    n_total_rows = len(AGENTS) * n_env_rows
    fig, axes = plt.subplots(
        n_total_rows,
        N_COLS,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * n_total_rows / 2),
        sharey=True,
    )

    for agent_idx, agent in enumerate(AGENTS):
        row_offset = agent_idx * n_env_rows
        for idx, env in enumerate(ENVIRONMENTS):
            erow, col = idx // N_COLS, idx % N_COLS
            ax = axes[row_offset + erow, col]
            has_data, k_vals, y_vals = _plot_env_agent(
                ax,
                env,
                agent,
                steps,
                step_color,
            )
            _format_subplot(ax, env, has_data, all_k_vals=k_vals, all_y_vals=y_vals)
            if col == 0 and erow == 0:
                ax.set_ylabel(f"{AGENT_LABELS[agent]}\nPass@k")
            elif col == 0:
                ax.set_ylabel("Pass@k")

    for ax in axes[-1]:
        ax.set_xlabel("k")

    # Shared legend in bottom-right subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[-1, -1].legend(
            handles,
            labels,
            loc="lower right",
            fontsize=5,
            framealpha=0.9,
        )

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / filename
    _save_fig(fig, out_path)


def main():
    # --- Averaged: separate success & failed ---
    plot_avg_figure(
        SUCCESS_STEPS,
        SUCCESS_COLOR,
        "Success",
        "intervention_pass_at_success.png",
    )
    plot_avg_figure(
        FAILED_STEPS,
        FAILED_COLOR,
        "Failed",
        "intervention_pass_at_failed.png",
    )

    # --- Per-agent: separate success & failed ---
    plot_per_agent_figure(
        SUCCESS_STEPS,
        SUCCESS_COLOR,
        "Success",
        "intervention_pass_at_success_per_agent.png",
    )
    plot_per_agent_figure(
        FAILED_STEPS,
        FAILED_COLOR,
        "Failed",
        "intervention_pass_at_failed_per_agent.png",
    )


if __name__ == "__main__":
    main()
