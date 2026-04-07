"""Plot recovery curves: success rate vs. number of injected steps.

Uses lama_aesthetics style with range_frame for consistent publication-quality plots.

Main plot: 2x3 grid (rows = success/failed, cols = environments).
Per-task plots: one figure per environment.

Reads from analysis/results.csv (output of aggregate_results.py).
"""

import sys
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT
from utils import filter_baseline_to_matched_tasks  # noqa: E402

# Step label ordering for x-axis
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
    """Compute bootstrap confidence interval for a mean."""
    vals = series.to_numpy()
    rng = np.random.default_rng(42)
    means = np.sort(
        [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    )
    return means[int((1 - ci) / 2 * n_boot)], means[int((1 + ci) / 2 * n_boot)]


def plot_recovery_curves(df: pd.DataFrame, output_dir: Path):
    """2xN grid: rows = success/failed interventions, cols = environments."""
    n_envs = len(ENVIRONMENTS)
    fig, axes = plt.subplots(
        2, n_envs, figsize=(TWO_COL_WIDTH * n_envs / 3, TWO_COL_HEIGHT)
    )

    for col, env in enumerate(ENVIRONMENTS):
        env_df = df[df["env"] == env]

        for row, intervention_type in enumerate(["success", "failed"]):
            ax = axes[row, col]
            color = SUCCESS_COLOR if intervention_type == "success" else FAILED_COLOR
            all_x, all_y = [], []

            for agent in AGENTS:
                agent_df = env_df[env_df["agent"] == agent]

                # Baseline (separate per agent, matching line style)
                baseline = agent_df[agent_df["intervention"] == "none"]
                if not baseline.empty:
                    baseline_sr = baseline["success"].mean()
                    ls_agent = AGENT_LINESTYLES[agent]
                    ax.axhline(
                        baseline_sr,
                        color=BASELINE_COLOR,
                        linestyle=ls_agent,
                        alpha=0.4,
                        linewidth=0.8,
                    )
                    # Include baseline in range so range_frame doesn't clip it
                    all_y.append(baseline_sr)

                # Intervention
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
                marker = AGENT_MARKERS[agent]
                ls = AGENT_LINESTYLES[agent]

                ax.plot(
                    x_pos,
                    rates,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    linewidth=1.5,
                    markersize=5,
                    label=AGENT_LABELS[agent],
                )
                ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.12, color=color)

                # Value labels
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

                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [STEP_ORDER[s] for s in steps], rotation=30, ha="right", fontsize=7
                )

            # Titles and labels
            if row == 0:
                ax.set_title(ENV_LABELS[env], fontsize=8)
            if col == 0:
                row_label = "Success" if intervention_type == "success" else "Failed"
                ax.set_ylabel(f"{row_label}\nSuccess Rate")

            if all_x and all_y:
                range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08, nice=False)
                ax.set_xticks(list(range(len(steps))))
                ax.set_xticklabels(
                    [STEP_ORDER[s] for s in steps], rotation=30, ha="right", fontsize=7
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

    # Legend
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
        out = output_dir / f"recovery_curves.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def plot_per_task_recovery(df: pd.DataFrame, output_dir: Path):
    """Per-task recovery curves (one figure per environment)."""
    for env in ENVIRONMENTS:
        env_df = df[df["env"] == env]
        tasks = sorted(env_df["task_id"].unique())
        n_tasks = len(tasks)

        fig, axes = plt.subplots(
            len(AGENTS),
            n_tasks,
            figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT),
            sharey=True,
        )
        if n_tasks == 1:
            axes = axes.reshape(-1, 1)
        if len(AGENTS) == 1:
            axes = axes.reshape(1, -1)

        for row, agent in enumerate(AGENTS):
            for col, task_id in enumerate(tasks):
                ax = axes[row, col]
                task_df = env_df[
                    (env_df["task_id"] == task_id) & (env_df["agent"] == agent)
                ]

                # Baseline
                baseline = task_df[task_df["intervention"] == "none"]
                if not baseline.empty:
                    ax.axhline(
                        baseline["success"].mean(),
                        color=BASELINE_COLOR,
                        linestyle=":",
                        alpha=0.5,
                        linewidth=0.8,
                    )

                for intervention_type, color in [
                    ("success", SUCCESS_COLOR),
                    ("failed", FAILED_COLOR),
                ]:
                    int_df = task_df[task_df["intervention"] == intervention_type]
                    if int_df.empty:
                        continue

                    grouped = int_df.groupby("num_steps")
                    steps = sorted(
                        grouped.groups.keys(),
                        key=lambda s: list(STEP_ORDER.keys()).index(s)
                        if s in STEP_ORDER
                        else 99,
                    )
                    rates = [grouped.get_group(s)["success"].mean() for s in steps]
                    x = list(range(len(steps)))

                    ax.plot(
                        x,
                        rates,
                        color=color,
                        marker="o",
                        linewidth=1.3,
                        markersize=4,
                        label=f"{intervention_type}" if col == 0 and row == 0 else None,
                    )
                    for xi, yi in zip(x, rates, strict=False):
                        ax.annotate(
                            f"{yi:.2f}",
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha="center",
                            fontsize=4.5,
                            color=color,
                        )
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [STEP_ORDER.get(s, str(s)) for s in steps],
                        rotation=45,
                        ha="right",
                        fontsize=5,
                    )

                if row == 0:
                    ax.set_title(task_id, fontsize=6)
                if col == 0:
                    ax.set_ylabel(f"{AGENT_LABELS[agent]}\nSuccess Rate", fontsize=7)
                ax.set_ylim(-0.05, 1.05)

        fig.suptitle(
            f"{ENV_LABELS[env]} — Per-Task Recovery", fontsize=11, fontweight="bold"
        )
        legend_elements = [
            Line2D([0], [0], color=SUCCESS_COLOR, marker="o", label="Success trace"),
            Line2D([0], [0], color=FAILED_COLOR, marker="o", label="Failed trace"),
            Line2D([0], [0], color=BASELINE_COLOR, linestyle=":", label="Baseline"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=7,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        out = output_dir / f"recovery_per_task_{env}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
        plt.close(fig)


def plot_recovery_curves_averaged(df: pd.DataFrame, output_dir: Path):
    """1xN grid: both success & failed, agents as solid (ReAct) / dashed (TC)."""
    n_envs = len(ENVIRONMENTS)
    fig, axes = plt.subplots(1, n_envs, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    for col, env in enumerate(ENVIRONMENTS):
        ax = axes[col]
        env_df = df[df["env"] == env]
        all_x, all_y = [], []

        # Baselines per agent
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

        # Intervention lines: one per (intervention_type, agent)
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
                marker = AGENT_MARKERS[agent]
                ls = AGENT_LINESTYLES[agent]

                ax.plot(
                    x_pos,
                    rates,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    linewidth=1.5,
                    markersize=5,
                )
                ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.10, color=color)

                # Only label ReAct lines (solid) to avoid clutter
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

    # Legend
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


def plot_recovery_curves_combined(df: pd.DataFrame, output_dir: Path):
    """2xN grid: rows = agents (ReAct/TC), cols = environments.

    Both success and failed lines in the same panel.
    """
    n_envs = len(ENVIRONMENTS)
    fig, axes = plt.subplots(
        2, n_envs, figsize=(TWO_COL_WIDTH * n_envs / 3, TWO_COL_HEIGHT)
    )

    for row, agent in enumerate(AGENTS):
        for col, env in enumerate(ENVIRONMENTS):
            ax = axes[row, col]
            agent_env_df = df[(df["env"] == env) & (df["agent"] == agent)]
            all_x, all_y = [], []

            # Baseline
            baseline_df = agent_env_df[agent_env_df["intervention"] == "none"]
            if not baseline_df.empty:
                baseline_sr = baseline_df["success"].mean()
                ax.axhline(
                    baseline_sr,
                    color=BASELINE_COLOR,
                    linestyle="--",
                    alpha=0.4,
                    linewidth=0.8,
                )
                all_y.append(baseline_sr)

            # Both intervention types
            all_steps_used = []
            for intervention_type, color in [
                ("success", SUCCESS_COLOR),
                ("failed", FAILED_COLOR),
            ]:
                int_df = agent_env_df[agent_env_df["intervention"] == intervention_type]
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
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                )
                ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.15, color=color)

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

            if row == 0:
                ax.set_title(ENV_LABELS[env], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{AGENT_LABELS[agent]}\nSuccess Rate")

            if all_x and all_y:
                range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08, nice=False)
                ax.set_xticks(list(range(len(all_steps_used))))
                ax.set_xticklabels(
                    [STEP_ORDER[s] for s in all_steps_used],
                    rotation=30,
                    ha="right",
                    fontsize=7,
                )

    # x-labels on bottom row only
    for ax in axes[1]:
        ax.set_xlabel("")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="--",
            alpha=0.4,
            label="Baseline",
        ),
        Line2D([0], [0], color=SUCCESS_COLOR, marker="o", label="Success"),
        Line2D([0], [0], color=FAILED_COLOR, marker="o", label="Failed"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=7,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["pdf", "png"]:
        out = output_dir / f"recovery_curves_combined.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def _plot_react_only_panel(ax, env_df, env, show_title=True):
    """Plot a single ReAct-only recovery panel."""
    all_x, all_y = [], []

    # Baseline
    baseline_df = env_df[env_df["intervention"] == "none"]
    if not baseline_df.empty:
        baseline_sr = baseline_df["success"].mean()
        ax.axhline(
            baseline_sr,
            color=BASELINE_COLOR,
            linestyle="--",
            alpha=0.4,
            linewidth=0.8,
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
            [STEP_ORDER[s] for s in all_steps_used],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=8)


def plot_recovery_curves_react_only(df: pd.DataFrame, output_dir: Path):
    """2x3 grid: ReAct only, both success & failed in same panel.

    Row 1: spectra, wetlab, retrosynthesis
    Row 2: resistor, md, ml
    """
    REACT_ENV_ORDER = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
    react_df = df[df["agent"] == "react"]

    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    # Row 0: first 3 envs
    for col, env in enumerate(REACT_ENV_ORDER[:3]):
        ax = axes[0, col]
        env_df = react_df[react_df["env"] == env]
        _plot_react_only_panel(ax, env_df, env)
        if col == 0:
            ax.set_ylabel("Success Rate", fontsize=8)

    # Row 1: next 3 envs
    for col, env in enumerate(REACT_ENV_ORDER[3:]):
        ax = axes[1, col]
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
        out = output_dir / f"recovery_curves_react_only.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def plot_recovery_curves_react_only_1row(df: pd.DataFrame, output_dir: Path):
    """1x5 grid: ReAct only, all environments in one row."""
    REACT_ENV_ORDER = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
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
    results_path = INTERVENTION_ROOT / "analysis" / "results.csv"
    if not results_path.exists():
        logger.error(f"{results_path} not found. Run aggregate_results.py first.")
        sys.exit(1)

    results_df = pd.read_csv(results_path)
    results_df = filter_baseline_to_matched_tasks(results_df)
    output_dir = INTERVENTION_ROOT / "analysis" / "figures"
    output_dir.mkdir(exist_ok=True)

    plot_recovery_curves(results_df, output_dir)
    plot_recovery_curves_averaged(results_df, output_dir)
    plot_recovery_curves_combined(results_df, output_dir)
    plot_recovery_curves_react_only(results_df, output_dir)
    plot_recovery_curves_react_only_1row(results_df, output_dir)
    plot_per_task_recovery(results_df, output_dir)

    logger.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
