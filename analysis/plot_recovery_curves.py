"""Plot recovery curves: success rate vs. number of injected steps.

Uses lama_aesthetics style with range_frame for consistent publication-quality plots.

Reads from results/data/intervention_results.csv (output of aggregate_intervention_results.py).
"""

import argparse
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervention_utils import filter_baseline_to_matched_tasks
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

_SCRIPT_DIR = Path(__file__).resolve().parent

STEP_ORDER = {1: "Step 1", 2: "Step 2", -2: "Step n-2", -1: "Step n-1"}
ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}
AGENTS = ["react", "toolcalling"]
AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}

SUCCESS_COLOR = "#16476A"
FAILED_COLOR = "#BF092F"
BASELINE_COLOR = "#7A7A7A"

AGENT_MARKERS = {"react": "o", "toolcalling": "s"}
AGENT_LINESTYLES = {"react": "-", "toolcalling": "--"}


def bootstrap_ci(series, n_boot=1000, ci=0.95):
    vals = series.to_numpy()
    rng = np.random.default_rng(42)
    means = np.sort(
        [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    )
    return means[int((1 - ci) / 2 * n_boot)], means[int((1 + ci) / 2 * n_boot)]


def _plot_recovery_panel(
    ax, env_df, env, intervention_type, show_title=True, ylim=None
):
    color = SUCCESS_COLOR if intervention_type == "success" else FAILED_COLOR
    all_x, all_y = [], []

    for agent in AGENTS:
        agent_df = env_df[env_df["agent"] == agent]
        baseline = agent_df[agent_df["intervention"] == "none"]
        if not baseline.empty:
            baseline_sr = baseline["success"].mean()
            ax.axhline(
                baseline_sr,
                color=BASELINE_COLOR,
                linestyle=AGENT_LINESTYLES[agent],
                alpha=0.4,
                linewidth=0.8,
            )
            all_y.append(baseline_sr)

        int_df = agent_df[agent_df["intervention"] == intervention_type]
        if int_df.empty:
            continue

        grouped = int_df.groupby("num_steps")
        steps, rates, ci_lo, ci_hi = [], [], [], []
        for ns in sorted(STEP_ORDER.keys(), key=list(STEP_ORDER.keys()).index):
            if ns not in grouped.groups:
                continue
            group = grouped.get_group(ns)
            sr = group["success"].mean()
            lo, hi = bootstrap_ci(group["success"])
            steps.append(ns)
            rates.append(sr)
            ci_lo.append(lo)
            ci_hi.append(hi)

        x_pos = list(range(len(steps)))
        ax.plot(
            x_pos,
            rates,
            color=color,
            marker=AGENT_MARKERS[agent],
            linestyle=AGENT_LINESTYLES[agent],
            linewidth=1.5,
            markersize=5,
            label=AGENT_LABELS[agent],
        )
        ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.12, color=color)

        if agent == "react":
            for xi, yi in zip(x_pos, rates, strict=False):
                ax.annotate(
                    f"{yi:.2f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=5,
                    color=color,
                )

        all_x.extend(x_pos)
        all_y.extend(rates)

    if show_title:
        ax.set_title(ENV_LABELS[env], fontsize=8)

    if all_x and all_y:
        range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08, nice=False)
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(-0.05, 1.05)
        all_step_keys = list(STEP_ORDER.keys())
        n_ticks = max(all_x) + 1 if all_x else 0
        used_steps = all_step_keys[:n_ticks]
        ax.set_xticks(list(range(n_ticks)))
        ax.set_xticklabels(
            [STEP_ORDER[s] for s in used_steps], rotation=30, ha="right", fontsize=7
        )
    elif not all_x:
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


def plot_recovery_curves(df, output_dir):
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1.8), sharey="row"
    )
    env_rows = [ENVIRONMENTS[:3], ENVIRONMENTS[3:6]]
    ylim_map = {"success": (0.2, 1.05), "failed": (-0.05, 0.8)}

    for row_offset, intervention_type in enumerate(["success", "failed"]):
        for sub_row, envs in enumerate(env_rows):
            grid_row = row_offset * 2 + sub_row
            for col, env in enumerate(envs):
                ax = axes[grid_row, col]
                env_df = df[df["env"] == env]
                _plot_recovery_panel(
                    ax,
                    env_df,
                    env,
                    intervention_type,
                    show_title=True,
                    ylim=ylim_map[intervention_type],
                )
                if col == 0:
                    if sub_row == 0:
                        row_label = (
                            "Success" if intervention_type == "success" else "Failed"
                        )
                        ax.set_ylabel(f"{row_label}\nSuccess Rate")
                    else:
                        ax.set_ylabel("Success Rate")

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="-",
            alpha=0.4,
            label="Baseline (ReAct)",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="--",
            alpha=0.4,
            label="Baseline (TC)",
        ),
        Line2D([0], [0], color=SUCCESS_COLOR, marker="o", label="Success (ReAct)"),
        Line2D(
            [0],
            [0],
            color=SUCCESS_COLOR,
            marker="s",
            linestyle="--",
            label="Success (TC)",
        ),
        Line2D([0], [0], color=FAILED_COLOR, marker="o", label="Failed (ReAct)"),
        Line2D(
            [0],
            [0],
            color=FAILED_COLOR,
            marker="s",
            linestyle="--",
            label="Failed (TC)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=6.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["pdf", "png"]:
        out = output_dir / f"recovery_curves.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def plot_recovery_curves_averaged(df, output_dir):
    n_envs = len(ENVIRONMENTS)
    fig, axes = plt.subplots(1, n_envs, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    for col, env in enumerate(ENVIRONMENTS):
        ax = axes[col]
        env_df = df[df["env"] == env]
        all_x, all_y = [], []

        for agent in AGENTS:
            baseline_df = env_df[
                (env_df["intervention"] == "none") & (env_df["agent"] == agent)
            ]
            if not baseline_df.empty:
                baseline_sr = baseline_df["success"].mean()
                ax.axhline(
                    baseline_sr,
                    color=BASELINE_COLOR,
                    linestyle=AGENT_LINESTYLES[agent],
                    alpha=0.4,
                    linewidth=0.8,
                )
                all_y.append(baseline_sr)

        all_steps_used = []
        for intervention_type, color in [
            ("success", SUCCESS_COLOR),
            ("failed", FAILED_COLOR),
        ]:
            for agent in AGENTS:
                int_df = env_df[
                    (env_df["intervention"] == intervention_type)
                    & (env_df["agent"] == agent)
                ]
                if int_df.empty:
                    continue
                grouped = int_df.groupby("num_steps")
                steps, rates, ci_lo, ci_hi = [], [], [], []
                for ns in sorted(STEP_ORDER.keys(), key=list(STEP_ORDER.keys()).index):
                    if ns not in grouped.groups:
                        continue
                    group = grouped.get_group(ns)
                    sr = group["success"].mean()
                    lo, hi = bootstrap_ci(group["success"])
                    steps.append(ns)
                    rates.append(sr)
                    ci_lo.append(lo)
                    ci_hi.append(hi)

                x_pos = list(range(len(steps)))
                ax.plot(
                    x_pos,
                    rates,
                    color=color,
                    marker=AGENT_MARKERS[agent],
                    linestyle=AGENT_LINESTYLES[agent],
                    linewidth=1.5,
                    markersize=5,
                )
                ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.10, color=color)

                if agent == "react":
                    for xi, yi in zip(x_pos, rates, strict=False):
                        offset_y = 7 if intervention_type == "success" else -12
                        ax.annotate(
                            f"{yi:.2f}",
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, offset_y),
                            ha="center",
                            fontsize=5,
                            color=color,
                        )

                all_x.extend(x_pos)
                all_y.extend(rates)
                if len(steps) > len(all_steps_used):
                    all_steps_used = steps

        ax.set_title(ENV_LABELS[env], fontsize=8)
        if col == 0:
            ax.set_ylabel("Success Rate")
        if all_x and all_y:
            range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08, nice=False)
            ax.set_xticks(list(range(len(all_steps_used))))
            ax.set_xticklabels(
                [STEP_ORDER[s] for s in all_steps_used],
                rotation=30,
                ha="right",
                fontsize=7,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="-",
            alpha=0.4,
            label="Baseline (ReAct)",
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="--",
            alpha=0.4,
            label="Baseline (TC)",
        ),
        Line2D([0], [0], color=SUCCESS_COLOR, marker="o", label="Success (ReAct)"),
        Line2D(
            [0],
            [0],
            color=SUCCESS_COLOR,
            marker="s",
            linestyle="--",
            label="Success (TC)",
        ),
        Line2D([0], [0], color=FAILED_COLOR, marker="o", label="Failed (ReAct)"),
        Line2D(
            [0],
            [0],
            color=FAILED_COLOR,
            marker="s",
            linestyle="--",
            label="Failed (TC)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=6.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        out = output_dir / f"recovery_curves_averaged.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def _plot_react_only_panel(ax, env_df, env, show_title=True):
    all_x, all_y = [], []
    baseline_df = env_df[env_df["intervention"] == "none"]
    if not baseline_df.empty:
        baseline_sr = baseline_df["success"].mean()
        ax.axhline(
            baseline_sr, color=BASELINE_COLOR, linestyle="--", alpha=0.4, linewidth=0.8
        )
        all_y.append(baseline_sr)

    all_steps_used = []
    for intervention_type, color in [
        ("success", SUCCESS_COLOR),
        ("failed", FAILED_COLOR),
    ]:
        int_df = env_df[env_df["intervention"] == intervention_type]
        if int_df.empty:
            continue
        grouped = int_df.groupby("num_steps")
        steps, rates, std_lo, std_hi = [], [], [], []
        for ns in sorted(STEP_ORDER.keys(), key=list(STEP_ORDER.keys()).index):
            if ns not in grouped.groups:
                continue
            group = grouped.get_group(ns)
            task_rates = group.groupby("task_id")["success"].mean()
            mean_sr = task_rates.mean()
            std_sr = task_rates.std()
            steps.append(ns)
            rates.append(mean_sr)
            std_lo.append(mean_sr - std_sr)
            std_hi.append(mean_sr + std_sr)

        x_pos = list(range(len(steps)))
        ax.plot(
            x_pos,
            rates,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(x_pos, std_lo, std_hi, alpha=0.15, color=color)

        for xi, yi in zip(x_pos, rates, strict=False):
            offset_y = 7 if intervention_type == "success" else -12
            ax.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=5,
                color=color,
            )

        all_x.extend(x_pos)
        all_y.extend(rates)
        all_y.extend(std_lo)
        all_y.extend(std_hi)
        if len(steps) > len(all_steps_used):
            all_steps_used = steps

    if show_title:
        ax.set_title(ENV_LABELS[env], fontsize=8)
    if all_x and all_y:
        range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08, nice=False)
        ax.set_xticks(list(range(len(all_steps_used))))
        ax.set_xticklabels(
            [STEP_ORDER[s] for s in all_steps_used], rotation=30, ha="right", fontsize=8
        )
        ax.tick_params(axis="y", labelsize=8)


def plot_recovery_curves_react_only_1row(df, output_dir, env_list=None):
    REACT_ENV_ORDER = env_list or ENVIRONMENTS
    react_df = df[df["agent"] == "react"]
    n_envs = len(REACT_ENV_ORDER)
    fig, axes = plt.subplots(1, n_envs, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    for col, env in enumerate(REACT_ENV_ORDER):
        ax = axes[col]
        env_df = react_df[react_df["env"] == env]
        _plot_react_only_panel(ax, env_df, env)
        if col == 0:
            ax.set_ylabel("Success Rate", fontsize=8)

    legend_elements = [
        Line2D(
            [0], [0], color=BASELINE_COLOR, linestyle="--", alpha=0.4, label="Baseline"
        ),
        Line2D([0], [0], color=SUCCESS_COLOR, marker="o", label="Success"),
        Line2D([0], [0], color=FAILED_COLOR, marker="o", label="Failed"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        out = output_dir / f"recovery_curves_react_only_1row.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-md", action="store_true", help="Exclude md environment from plots"
    )
    args = parser.parse_args()

    results_path = _SCRIPT_DIR / "results" / "data" / "intervention_results.csv"
    if not results_path.exists():
        logger.error(
            f"{results_path} not found. Run aggregate_intervention_results.py first."
        )
        raise SystemExit(1)

    results_df = pd.read_csv(results_path)
    results_df = filter_baseline_to_matched_tasks(results_df)
    output_dir = _SCRIPT_DIR / "results" / "figures" / "intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_list = None
    if args.no_md:
        env_list = [e for e in ENVIRONMENTS if e != "md"]

    plot_recovery_curves(results_df, output_dir)
    plot_recovery_curves_averaged(results_df, output_dir)
    plot_recovery_curves_react_only_1row(results_df, output_dir, env_list=env_list)

    logger.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
