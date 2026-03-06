"""Plot tool-call summaries per task trial across benchmark groupings.

This script loads the combined benchmark dataset from
`analysis/results/data/reports.jsonl`, iterates over the nested `Task Results`
trial data, extracts the number of tool calls made in each trial, and plots that
quantity for three different aggregations:

- environment
- model
- agent type
- environment by agent type, split into one figure per model

- environment: horizontal box-and-whisker plot
- model: ridgeline (joy) plot
- agent type: ridgeline (joy) plot
- environment by agent type: grouped bar chart for each model
"""

import json
import math
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "avg_tool_calls_per_task_by_environment.pdf"
OUT_FILE_MODEL = OUT_DIR / "avg_tool_calls_per_task_by_model.pdf"
OUT_FILE_AGENT = OUT_DIR / "avg_tool_calls_per_task_by_agent_type.pdf"
OUT_FILE_ENV_AGENT_TEMPLATE = "avg_tool_calls_per_environment_by_agent_{model_slug}.pdf"

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
}

MODEL_LABELS = {
    "claude-4.5": "Claude 4.5",
    "gpt-4o": "GPT-4o",
    "gpt-oss-120b": "GPT-OSS-120B",
}

AGENT_TYPE_LABELS = {
    "react": "ReAct",
    "tool_calling": "Tool calling",
}

PLOT_COLOR = "#4C72B0"
GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
AGENT_COLORS = ["#64B5CD", "#D08770", "#A3BE8C", "#B48EAD"]


def get_group_centers(
    n_groups: int, *, bar_width: float = 0.22, gap_width: float = 0.16
) -> np.ndarray:
    """Return evenly spaced group centers for grouped categorical plots."""
    return np.arange(n_groups) * (bar_width * 2 + gap_width + bar_width * 0.3)


def get_axis_max(values: pd.Series | np.ndarray, *, minimum: float = 1.0) -> float:
    """Return a rounded y-axis maximum with a small amount of headroom."""
    cleaned_values = pd.Series(values).dropna()
    if cleaned_values.empty:
        return minimum

    max_value = cleaned_values.max()
    if max_value <= 0:
        return minimum

    if max_value < 20:
        return max(minimum, math.ceil(max_value * 1.1))

    return max(minimum, math.ceil(max_value * 1.05 / 10) * 10)


def slugify_model_name(model_name: str) -> str:
    """Create a filesystem-safe slug for a model key."""
    return model_name.replace("-", "_").replace(".", "_")


def extract_tool_call_count(trial: dict) -> int | None:
    """Return the tool call count for a trial, if available."""
    total_calls = trial.get("total_calls")
    if total_calls is not None:
        return int(total_calls)

    tool_calls = trial.get("tool_calls")
    if isinstance(tool_calls, list):
        return len(tool_calls)

    successful_calls = trial.get("successful_calls")
    failed_calls = trial.get("failed_calls")
    if successful_calls is not None or failed_calls is not None:
        return int(successful_calls or 0) + int(failed_calls or 0)

    return None


def plot_group_boxplots(
    ax: Axes,
    results_df: pd.DataFrame,
    group_col: str,
    label_map: dict[str, str],
    y_label: str,
) -> None:
    group_keys = sorted(results_df[group_col].dropna().unique())
    plot_data = [
        results_df.loc[
            results_df[group_col].eq(group_key), "tool_calls_per_trial"
        ].to_numpy()
        for group_key in group_keys
    ]
    non_empty = [
        (group_key, group_values)
        for group_key, group_values in zip(group_keys, plot_data, strict=True)
        if len(group_values) > 0
    ]
    group_keys = [group_key for group_key, _ in non_empty]
    plot_data = [group_values for _, group_values in non_empty]
    labels = [label_map.get(group_key, str(group_key)) for group_key in group_keys]
    positions = np.arange(len(group_keys))

    boxplot = ax.boxplot(
        plot_data,
        positions=positions,
        vert=False,
        widths=0.6,
        notch=True,
        patch_artist=True,
        showmeans=False,
        medianprops={"color": "white", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.1, "color": "#555555"},
        capprops={"linewidth": 1.1, "color": "#555555"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
    )

    for idx, box in enumerate(boxplot["boxes"]):
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.75)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tool Calls per Task Trial")
    ax.set_ylabel(y_label)

    range_frame(ax, np.array([0, 100]), positions)


def plot_ridgeline(
    results_df: pd.DataFrame,
    group_col: str,
    label_map: dict[str, str],
    out_path: Path,
    *,
    figsize: tuple[float, float] = (ONE_COL_WIDTH, ONE_COL_HEIGHT),
) -> None:
    """Create a ridgeline (joy) plot of tool-call distributions."""
    group_keys = sorted(results_df[group_col].dropna().unique())
    plot_data = [
        results_df.loc[results_df[group_col].eq(k), "tool_calls_per_trial"].to_numpy()
        for k in group_keys
    ]
    non_empty = [
        (k, v) for k, v in zip(group_keys, plot_data, strict=True) if len(v) > 0
    ]
    group_keys = [k for k, _ in non_empty]
    plot_data = [v for _, v in non_empty]
    labels = [label_map.get(k, str(k)) for k in group_keys]
    n_groups = len(group_keys)

    overlap = 0.6
    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(figsize[0], max(figsize[1], n_groups * 0.9)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    x_min = 0.0
    x_max = float(max(v.max() for v in plot_data)) * 1.05
    x_grid = np.linspace(x_min, x_max, 500)

    for idx in range(n_groups):
        ax = axes[idx]
        vals = plot_data[idx].astype(float)
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]

        if vals.std() == 0:
            density = np.zeros_like(x_grid)
            density[np.argmin(np.abs(x_grid - vals[0]))] = 1.0
        else:
            kde = gaussian_kde(vals, bw_method=0.35)
            density = kde(x_grid)

        density = density / density.max() if density.max() > 0 else density

        ax.fill_between(x_grid, density, alpha=0.65, color=color)
        ax.plot(x_grid, density, color=color, linewidth=1.2)

        median = float(np.median(vals))
        ax.axvline(median, color="black", linewidth=0.9, linestyle="--", alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1 + overlap)
        ax.set_yticks([])
        ax.patch.set_alpha(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            -0.01,
            0.15,
            labels[idx],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )

    axes[-1].set_xlabel("Tool Calls per Task Trial")
    axes[-1].spines["bottom"].set_visible(True)

    fig.subplots_adjust(hspace=-overlap)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_environment_agent_bars_by_model(results_df: pd.DataFrame) -> list[Path]:
    """Save one grouped lollipop-style chart per model for environment-agent averages."""
    saved_paths: list[Path] = []

    for model_key in sorted(results_df["model"].dropna().unique()):
        model_results_df = results_df.loc[results_df["model"].eq(model_key)].copy()
        if model_results_df.empty:
            continue

        environments = sorted(model_results_df["environment"].dropna().unique())
        agent_types = sorted(model_results_df["agent_type"].dropna().unique())
        if not environments or not agent_types:
            continue

        summary = model_results_df.groupby(
            ["environment", "agent_type"], as_index=False
        )["tool_calls_per_trial"].mean()

        bar_width = 0.22
        gap_width = 0.16
        x_centers = get_group_centers(
            len(environments), bar_width=bar_width, gap_width=gap_width
        )

        fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

        for idx, agent_type in enumerate(agent_types):
            means = []
            for environment in environments:
                match = summary.loc[
                    summary["environment"].eq(environment)
                    & summary["agent_type"].eq(agent_type),
                    "tool_calls_per_trial",
                ]
                means.append(float(match.iloc[0]) if not match.empty else np.nan)

            offsets = x_centers + (idx - (len(agent_types) - 1) / 2) * bar_width
            valid_points = [
                (offset, mean)
                for offset, mean in zip(offsets, means, strict=True)
                if not np.isnan(mean)
            ]
            if not valid_points:
                continue

            valid_offsets = [offset for offset, _ in valid_points]
            valid_means = [mean for _, mean in valid_points]
            ax.vlines(
                valid_offsets,
                0,
                valid_means,
                color=AGENT_COLORS[idx % len(AGENT_COLORS)],
                alpha=0.5,
                linewidth=5,
                label=AGENT_TYPE_LABELS.get(agent_type, str(agent_type)),
            )
            ax.plot(
                valid_offsets,
                valid_means,
                "o",
                markersize=6,
                color=AGENT_COLORS[idx % len(AGENT_COLORS)],
                alpha=0.75,
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [
                ENV_LABELS.get(environment, str(environment))
                for environment in environments
            ]
        )
        ax.set_xlabel("Environment")
        ax.set_ylabel("Average Tool Calls\nper Task Trial")
        ax.set_title(MODEL_LABELS.get(model_key, str(model_key)))
        ax.set_ylim(0, get_axis_max(summary["tool_calls_per_trial"], minimum=1.0))
        ax.legend(
            title="Agent Type",
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.8, 1.4),
            ncol=len(agent_types),
            borderaxespad=0,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        range_frame(
            ax,
            np.array([x_centers.min(), x_centers.max()]),
            np.array([0, get_axis_max(summary["tool_calls_per_trial"], minimum=1.0)]),
        )

        plt.tight_layout()
        out_path = OUT_DIR / OUT_FILE_ENV_AGENT_TEMPLATE.format(
            model_slug=slugify_model_name(model_key)
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    records = []
    skipped_trials = 0

    with DATA_PATH.open() as fh:
        for raw_line in fh:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            row = json.loads(stripped_line)
            environment = row.get("environment")
            model = row.get("model")
            agent_type = row.get("agent_type")
            task_results = row.get("Task Results", {})
            if isinstance(task_results, str):
                task_results = json.loads(task_results)

            for task_data in task_results.values():
                for trial in task_data.get("trials", []):
                    tool_call_count = extract_tool_call_count(trial)
                    if tool_call_count is None:
                        skipped_trials += 1
                        continue

                    records.append(
                        {
                            "environment": environment,
                            "model": model,
                            "agent_type": agent_type,
                            "tool_calls_per_trial": tool_call_count,
                        }
                    )

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise ValueError(f"No trials with tool call counts found in {DATA_PATH}")

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))
    plot_group_boxplots(ax, results_df, "environment", ENV_LABELS, "Environment")

    plt.tight_layout()
    fig.savefig(OUT_FILE, bbox_inches="tight")

    plot_ridgeline(results_df, "model", MODEL_LABELS, OUT_FILE_MODEL)

    plot_ridgeline(results_df, "agent_type", AGENT_TYPE_LABELS, OUT_FILE_AGENT)

    env_agent_paths = plot_environment_agent_bars_by_model(results_df)

    logger.info(f"Skipped {skipped_trials} trials without tool call counts")
    logger.info(f"Saved figure to {OUT_FILE}")
    logger.info(f"Saved figure to {OUT_FILE_MODEL}")
    logger.info(f"Saved figure to {OUT_FILE_AGENT}")
    for path in env_agent_paths:
        logger.info(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
