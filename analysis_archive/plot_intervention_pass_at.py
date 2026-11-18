"""Plot Pass@k vs k averaged over ReAct & ToolCalling, plus per-agent version.

Averaged grid: top row = success interventions, bottom row = failed.
Per-agent grid: rows = Success(ReAct), Success(TC), Failed(ReAct), Failed(TC).
Columns = environments. Baseline shown in both rows as reference.

Reads from results/data/intervention_reports.jsonl (downloaded from HF).
"""

import argparse
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from intervention_utils import (
    avg_matched_baseline,
    extract_matched_pass_at,
    load_metrics,
    load_reports,
)
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

lama_aesthetics.get_style("main")

_SCRIPT_DIR = Path(__file__).resolve().parent

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
AGENTS = ["react", "toolcalling"]

FAILED_STEPS = ["failed_step1", "failed_step2", "failed_stepn1", "failed_stepn2"]
SUCCESS_STEPS = ["success_step1", "success_step2", "success_stepn1", "success_stepn2"]

FAILED_COLOR = "#E07A5F"
SUCCESS_COLOR = "#4C78A8"
BASELINE_COLOR = "#7A7A7A"

STEP_ALPHA = {"step1": 0.85, "step2": 0.6, "stepn1": 0.4, "stepn2": 0.25}
STEP_MARKERS = {"step1": "s", "step2": "D", "stepn1": "^", "stepn2": "v"}

ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}

AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}


def extract_pass_at(metrics: dict) -> tuple[list[int], list[float]]:
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass@{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def avg_pass_at(df, env: str, step: str) -> tuple[list[int], list[float]] | None:
    all_vals = []
    for agent in AGENTS:
        m = load_metrics(env, agent, step, df)
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


def plot_row(axes_row, environments, steps, step_color, row_label, df):
    for col, env in enumerate(environments):
        ax = axes_row[col]
        all_k_vals, all_y_vals = [], []

        result = avg_matched_baseline(env, metric_type="pass_at", df=df)
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
            result = avg_pass_at(df, env, step_name)
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

        ax.set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{row_label}\nPass@k")

        if all_k_vals and all_y_vals:
            ax.set_ylim(-0.05, 1.05)
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


def plot_row_agent(
    axes_row, environments, agent, steps, step_color, row_label, df, ylim=None
):
    for col, env in enumerate(environments):
        ax = axes_row[col]
        all_k_vals, all_y_vals = [], []

        result = extract_matched_pass_at(env, agent, df)
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
            m = load_metrics(env, agent, step_name, df)
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

        if col == 0:
            ax.set_ylabel(f"{row_label}\nPass@k")

        if all_k_vals and all_y_vals:
            range_frame(ax, np.array(all_k_vals), np.array(all_y_vals), pad=0.05)
            ax.set_xticks([1, 5, 10, 15])
            if ylim:
                ax.set_ylim(ylim)
                ax.set_yticks(
                    [t for t in [0, 0.25, 0.5, 0.75, 1.0] if ylim[0] <= t <= ylim[1]]
                )
            else:
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-md", action="store_true", help="Exclude md environment")
    args = parser.parse_args()

    reports = load_reports()
    envs = [e for e in ENVIRONMENTS if e != "md"] if args.no_md else ENVIRONMENTS
    n_envs = len(envs)
    out_dir = _SCRIPT_DIR / "results" / "figures" / "intervention"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Averaged version (2 rows) ---
    fig, axes = plt.subplots(
        2, n_envs, figsize=(TWO_COL_WIDTH * n_envs / 3, TWO_COL_HEIGHT), sharey=True
    )
    plot_row(axes[0], envs, SUCCESS_STEPS, SUCCESS_COLOR, "Success", reports)
    plot_row(axes[1], envs, FAILED_STEPS, FAILED_COLOR, "Failed", reports)
    for ax in axes[1]:
        ax.set_xlabel("k")
        ax.set_title("")
    fig.tight_layout()

    out_path = out_dir / "intervention_pass_at_plot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    logger.info(f"Saved to {out_path}")
    plt.close(fig)

    # --- Per-agent version (4 rows) ---
    success_ylim = (0.3, 1.05)
    failed_ylim = (-0.05, 1.05)
    fig, axes = plt.subplots(
        4,
        n_envs,
        figsize=(TWO_COL_WIDTH, 1.5 * TWO_COL_HEIGHT),
        sharey="row",
        sharex=True,
    )
    plot_row_agent(
        axes[0],
        envs,
        "react",
        SUCCESS_STEPS,
        SUCCESS_COLOR,
        "Success\n(ReAct)",
        reports,
        ylim=success_ylim,
    )
    plot_row_agent(
        axes[1],
        envs,
        "toolcalling",
        SUCCESS_STEPS,
        SUCCESS_COLOR,
        "Success\n(ToolCalling)",
        reports,
        ylim=success_ylim,
    )
    plot_row_agent(
        axes[2],
        envs,
        "react",
        FAILED_STEPS,
        FAILED_COLOR,
        "Failed\n(ReAct)",
        reports,
        ylim=failed_ylim,
    )
    plot_row_agent(
        axes[3],
        envs,
        "toolcalling",
        FAILED_STEPS,
        FAILED_COLOR,
        "Failed\n(ToolCalling)",
        reports,
        ylim=failed_ylim,
    )

    for col, env in enumerate(envs):
        axes[0, col].set_title(ENV_LABELS.get(env, env.capitalize()), fontsize=8)
    for row in range(1, 4):
        for ax in axes[row]:
            ax.set_title("")
    for ax in axes[3]:
        ax.set_xlabel("k")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(
            handles, labels, loc="lower right", fontsize=5, framealpha=0.9
        )
        axes[1, -1].legend(
            handles, labels, loc="lower right", fontsize=5, framealpha=0.9
        )
        handles_f, labels_f = axes[2, 0].get_legend_handles_labels()
        axes[3, -1].legend(
            handles_f, labels_f, loc="lower right", fontsize=5, framealpha=0.9
        )

    fig.tight_layout()
    out_path = out_dir / "intervention_pass_at_per_agent.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(
        out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", format="pdf"
    )
    logger.info(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
