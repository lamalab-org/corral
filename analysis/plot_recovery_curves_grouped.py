"""Recovery curves averaged over environment groups.

Groups environments into cognitive-task categories and plots
the mean success rate across all environments within each group.

Reads from results/data/intervention_results.csv.
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervention_utils import filter_baseline_to_matched_tasks
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

_SCRIPT_DIR = Path(__file__).resolve().parent

STEP_ORDER = {1: "Step 1", 2: "Step 2", -2: "Step n-2", -1: "Step n-1"}
AGENTS = ["react", "toolcalling"]
AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}

SUCCESS_COLOR = "#4C78A8"
FAILED_COLOR = "#E07A5F"
BASELINE_COLOR = "#7A7A7A"

AGENT_MARKERS = {"react": "o", "toolcalling": "s"}
AGENT_LINESTYLES = {"react": "-", "toolcalling": "--"}

ENVIRONMENT_GROUPS = {
    "Hypothesis-driven\ninquiry": {
        "description": "Reason from observations to hidden structure",
        "environments": ["spectra", "wetlab", "resistor"],
    },
    "Strategic\nreasoning": {
        "description": "Navigate combinatorial spaces under constraints",
        "environments": ["retrosynthesis"],
    },
    "Workflow\nconstruction": {
        "description": "Assemble and execute computational protocols",
        "environments": ["catalyst", "md", "ml"],
    },
}


def bootstrap_ci(vals, n_boot=1000, ci=0.95):
    arr = np.asarray(vals)
    rng = np.random.default_rng(42)
    means = np.sort(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    return means[int((1 - ci) / 2 * n_boot)], means[int((1 + ci) / 2 * n_boot)]


def _filter_available_envs(results_df, envs):
    """Return only environments that exist in the data."""
    available = set(results_df["env"].unique())
    return [e for e in envs if e in available]


def _plot_grouped_panel(ax, group_df, group_name, intervention_type, ylim=None):
    """Plot recovery curve for a group, averaging across environments."""
    color = SUCCESS_COLOR if intervention_type == "success" else FAILED_COLOR
    all_x, all_y = [], []

    for agent in AGENTS:
        agent_df = group_df[group_df["agent"] == agent]
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

        all_x.extend(x_pos)
        all_y.extend(rates)

    ax.set_title(group_name, fontsize=8)

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


def plot_grouped_recovery(results_df, output_dir):
    """One row per intervention type, one column per group."""
    group_names = list(ENVIRONMENT_GROUPS.keys())
    n_groups = len(group_names)
    ylim_map = {"success": (0.2, 1.05), "failed": (-0.05, 0.8)}

    fig, axes = plt.subplots(
        2, n_groups, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 2), sharey="row"
    )

    for row, intervention_type in enumerate(["success", "failed"]):
        for col, gname in enumerate(group_names):
            ax = axes[row, col]
            envs = _filter_available_envs(
                results_df, ENVIRONMENT_GROUPS[gname]["environments"]
            )
            group_df = results_df[results_df["env"].isin(envs)]
            _plot_grouped_panel(
                ax, group_df, gname, intervention_type, ylim=ylim_map[intervention_type]
            )
            if col == 0:
                label = "Success" if intervention_type == "success" else "Failed"
                ax.set_ylabel(f"{label}\nSuccess Rate")

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
        bbox_to_anchor=(0.5, 1.03),
        fontsize=6.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["pdf", "png"]:
        out = output_dir / f"recovery_curves_grouped.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def plot_grouped_recovery_react_only(results_df, output_dir):
    """Single row, ReAct only, one column per group."""
    group_names = list(ENVIRONMENT_GROUPS.keys())
    n_groups = len(group_names)
    react_df = results_df[results_df["agent"] == "react"]

    fig, axes = plt.subplots(
        1, n_groups, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT), sharey=True
    )

    # Collect global y range for consistent range_frame across shared axes
    global_y = []
    for _col, gname in enumerate(group_names):
        envs = _filter_available_envs(
            react_df, ENVIRONMENT_GROUPS[gname]["environments"]
        )
        group_df = react_df[react_df["env"].isin(envs)]
        for intervention_type in ["success", "failed"]:
            int_df = group_df[group_df["intervention"] == intervention_type]
            if int_df.empty:
                continue
            for ns in STEP_ORDER:
                grp = int_df[int_df["num_steps"] == ns]
                if not grp.empty:
                    global_y.append(grp["success"].mean())
        baseline = group_df[group_df["intervention"] == "none"]
        if not baseline.empty:
            global_y.append(baseline["success"].mean())

    for col, gname in enumerate(group_names):
        ax = axes[col]
        envs = _filter_available_envs(
            react_df, ENVIRONMENT_GROUPS[gname]["environments"]
        )
        group_df = react_df[react_df["env"].isin(envs)]

        all_x, all_y = [], []
        baseline = group_df[group_df["intervention"] == "none"]
        if not baseline.empty:
            baseline_sr = baseline["success"].mean()
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
            int_df = group_df[group_df["intervention"] == intervention_type]
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
            ax.fill_between(x_pos, ci_lo, ci_hi, alpha=0.12, color=color)

            all_x.extend(x_pos)
            all_y.extend(rates)
            global_y.extend(rates)
            if len(steps) > len(all_steps_used):
                all_steps_used = steps

        ax.set_title(gname, fontsize=8)
        if col == 0:
            ax.set_ylabel("Success Rate", fontsize=8)
        if all_x and all_y:
            range_frame(ax, np.array(all_x), np.array(global_y), pad=0.08, nice=False)
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
            linestyle="--",
            alpha=0.4,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            color=SUCCESS_COLOR,
            marker="o",
            label="Intervention with successful trace",
        ),
        Line2D(
            [0],
            [0],
            color=FAILED_COLOR,
            marker="o",
            label="Intervention with failed trace",
        ),
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
        out = output_dir / f"recovery_curves_grouped_react_only.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"Saved: {out}")
    plt.close(fig)


def main():
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

    plot_grouped_recovery(results_df, output_dir)
    plot_grouped_recovery_react_only(results_df, output_dir)

    logger.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
