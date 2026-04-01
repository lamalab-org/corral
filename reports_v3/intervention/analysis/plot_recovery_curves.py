"""Plot recovery curves: success rate vs. number of injected steps.

For each environment, plots success rate (y) vs. intervention level (x),
with separate lines for success trace, failed trace, and baseline.
Separate panels per environment, lines per agent type.

Reads from analysis/results.csv (output of aggregate_results.py).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import INTERVENTION_ROOT

# Step label ordering for x-axis
STEP_ORDER = {0: "baseline", 1: "step 1", 2: "step 2", -2: "step n-2", -1: "step n-1"}


def bootstrap_ci(series, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval for a mean."""
    vals = series.to_numpy()
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lower = means[int((1 - ci) / 2 * n_boot)]
    upper = means[int((1 + ci) / 2 * n_boot)]
    return lower, upper


def plot_recovery_curves(df: pd.DataFrame, output_dir: Path):
    """Create recovery curve plots."""
    envs = sorted(df["env"].unique())
    agents = sorted(df["agent"].unique())

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5), sharey=True)
    if len(envs) == 1:
        axes = [axes]

    colors = {
        ("success", "react"): "#2196F3",
        ("success", "toolcalling"): "#64B5F6",
        ("failed", "react"): "#F44336",
        ("failed", "toolcalling"): "#EF9A9A",
    }
    markers = {"react": "o", "toolcalling": "s"}
    linestyles = {"success": "-", "failed": "--"}

    for ax, env in zip(axes, envs, strict=False):
        env_df = df[df["env"] == env]

        for agent in agents:
            agent_df = env_df[env_df["agent"] == agent]

            # Baseline
            baseline = agent_df[agent_df["intervention"] == "none"]
            if not baseline.empty:
                baseline_sr = baseline["success"].mean()
                ax.axhline(
                    baseline_sr,
                    color="gray",
                    linestyle=":",
                    alpha=0.7,
                    label=f"baseline ({agent})" if agent == agents[0] else None,
                )

            # Intervention lines
            for intervention_type in ["success", "failed"]:
                int_df = agent_df[agent_df["intervention"] == intervention_type]
                if int_df.empty:
                    continue

                # Group by num_steps
                grouped = int_df.groupby("num_steps")
                steps = []
                rates = []
                ci_lower = []
                ci_upper = []

                for ns, group in sorted(
                    grouped,
                    key=lambda x: list(STEP_ORDER.keys()).index(x[0])
                    if x[0] in STEP_ORDER
                    else 99,
                ):
                    sr = group["success"].mean()
                    lo, hi = bootstrap_ci(group["success"])
                    steps.append(ns)
                    rates.append(sr)
                    ci_lower.append(lo)
                    ci_upper.append(hi)

                x_positions = list(range(len(steps)))
                x_labels = [STEP_ORDER.get(s, str(s)) for s in steps]

                color = colors.get((intervention_type, agent), "gray")
                marker = markers.get(agent, "o")
                ls = linestyles.get(intervention_type, "-")

                ax.plot(
                    x_positions,
                    rates,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    linewidth=2,
                    markersize=8,
                    label=f"{intervention_type} ({agent})",
                )
                ax.fill_between(
                    x_positions,
                    ci_lower,
                    ci_upper,
                    alpha=0.15,
                    color=color,
                )

                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=30, ha="right")

        ax.set_title(env.capitalize(), fontsize=14, fontweight="bold")
        ax.set_xlabel("Intervention Level")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Success Rate")

    # Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(len(by_label), 5),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = output_dir / "recovery_curves.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    logger.info(f"Saved: {output_path}")

    output_path_png = output_dir / "recovery_curves.png"
    fig.savefig(output_path_png, bbox_inches="tight", dpi=150)
    logger.info(f"Saved: {output_path_png}")

    plt.close(fig)


def plot_per_task_recovery(df: pd.DataFrame, output_dir: Path):
    """Plot per-task recovery curves (one plot per env)."""
    envs = sorted(df["env"].unique())

    for env in envs:
        env_df = df[df["env"] == env]
        tasks = sorted(env_df["task_id"].unique())
        agents = sorted(env_df["agent"].unique())

        n_tasks = len(tasks)
        fig, axes = plt.subplots(
            len(agents), n_tasks, figsize=(4 * n_tasks, 4 * len(agents)), sharey=True
        )
        if n_tasks == 1:
            axes = axes.reshape(-1, 1)
        if len(agents) == 1:
            axes = axes.reshape(1, -1)

        for row, agent in enumerate(agents):
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
                        color="gray",
                        linestyle=":",
                        alpha=0.7,
                    )

                for intervention_type, color in [
                    ("success", "#2196F3"),
                    ("failed", "#F44336"),
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
                        x, rates, color=color, marker="o", linewidth=1.5, markersize=5
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [STEP_ORDER.get(s, str(s)) for s in steps],
                        rotation=45,
                        ha="right",
                        fontsize=7,
                    )

                if row == 0:
                    ax.set_title(task_id, fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{agent}\nSuccess Rate", fontsize=9)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{env.capitalize()} - Per-Task Recovery", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        output_path = output_dir / f"recovery_per_task_{env}.pdf"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved: {output_path}")
        plt.close(fig)


def main():
    results_path = INTERVENTION_ROOT / "analysis" / "results.csv"
    if not results_path.exists():
        logger.info(f"ERROR: {results_path} not found. Run aggregate_results.py first.")
        sys.exit(1)

    results_df = pd.read_csv(results_path)
    output_dir = INTERVENTION_ROOT / "analysis" / "figures"
    output_dir.mkdir(exist_ok=True)

    plot_recovery_curves(results_df, output_dir)
    plot_per_task_recovery(results_df, output_dir)

    logger.info(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
