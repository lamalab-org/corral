"""Compare how much tool interaction each benchmark slice requires per trial.

The script works at the trial level so repeated retries remain visible in the
distribution plots. It pairs a capped environment boxplot with ridgeline views
for models and scaffolds, then optionally adds model-specific environment
breakdowns to show whether the same environment is tool-heavy for all models or
only a subset of them.
"""

import json
import math
from pathlib import Path

import fire
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
FIGURES_DIR = Path(__file__).parent / "results" / "figures"
OUT_DIR = FIGURES_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUT_DIR = FIGURES_DIR / "analysis"
ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    "tool_calling": "Tool Calling",
}

PLOT_COLOR = "#4C72B0"
GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
AGENT_COLORS = ["#64B5CD", "#D08770", "#A3BE8C", "#B48EAD"]
DISPLAY_PERCENTILE = 99.0
ENVIRONMENT_BOXPLOT_MAX = 32.0


def get_group_centers(
    n_groups: int, *, bar_width: float = 0.22, gap_width: float = 0.16
) -> np.ndarray:
    """Compute evenly spaced anchors for grouped environment summaries.

    Args:
        n_groups: Number of categorical groups to place on the axis.
        bar_width: Horizontal footprint allocated to one subgroup.
        gap_width: Extra spacing inserted between neighboring groups.

    Returns:
        np.ndarray: Center coordinate for each categorical group.
    """
    return np.arange(n_groups) * (bar_width * 2 + gap_width + bar_width * 0.3)


def get_axis_max(values: pd.Series | np.ndarray, *, minimum: float = 1.0) -> float:
    """Choose a rounded upper bound for count-based y-axes.

    Args:
        values: Numeric values that should fit on the axis.
        minimum: Fallback maximum when the input is empty or non-positive.

    Returns:
        float: Rounded axis maximum with light headroom.
    """
    cleaned_values = pd.Series(values).dropna()
    if cleaned_values.empty:
        return minimum

    max_value = cleaned_values.max()
    if max_value <= 0:
        return minimum

    if max_value < 20:
        return max(minimum, math.ceil(max_value * 1.1))

    return max(minimum, math.ceil(max_value * 1.05 / 10) * 10)


def get_display_cap(
    values: pd.Series | np.ndarray,
    *,
    percentile: float = DISPLAY_PERCENTILE,
    minimum: float = 1.0,
) -> float:
    """Estimate a plotting cap that hides only the most extreme tail values.

    Args:
        values: Numeric values used to determine the display cutoff.
        percentile: High percentile used as the visual cap.
        minimum: Fallback cap when the input is empty or non-positive.

    Returns:
        float: Rounded cap derived from the requested percentile.
    """
    cleaned_values = pd.Series(values).dropna()
    if cleaned_values.empty:
        return minimum

    percentile_value = float(np.percentile(cleaned_values, percentile))
    if percentile_value <= 0:
        return minimum

    return get_axis_max(
        np.array([percentile_value]),
        minimum=minimum,
    )


def slugify_model_name(model_name: str) -> str:
    """Create deterministic filenames for per-model output figures.

    Args:
        model_name: Raw model identifier from the benchmark table.

    Returns:
        str: Filesystem-safe model slug.
    """
    return model_name.replace("-", "_").replace(".", "_")


def extract_tool_call_count(trial: dict) -> int | None:
    """Recover a comparable tool-call count from heterogeneous trial schemas.

    Different trace formats record the same quantity under different keys. The
    fallback order prefers explicit totals, then raw call lists, then separate
    success and failure counters.

    Args:
        trial: Trial payload from the nested report structure.

    Returns:
        int | None: Tool-call count for the trial, or `None` when no count can
        be inferred safely.
    """
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
    *,
    max_display_value: float | None = None,
    box_color: str = PLOT_COLOR,
) -> None:
    """Draw horizontal boxplots for one categorical grouping.

    The optional display cap is intended for the environment view, where a few
    outlier trials can otherwise stretch the axis enough to hide the bulk of the
    distribution.

    Args:
        ax: Axis that receives the boxplots.
        results_df: Trial-level tool-call table.
        group_col: Column defining the categorical groups.
        label_map: Human-readable labels for group values.
        y_label: Label shown on the categorical axis.
        max_display_value: Optional hard cap for values included in the plot.
        box_color: Fill color used for all boxes.

    Returns:
        None: The function mutates the provided axis.
    """
    group_keys = sorted(results_df[group_col].dropna().unique())
    filtered_results_df = results_df
    if max_display_value is not None:
        filtered_results_df = results_df.loc[
            results_df["tool_calls_per_trial"].le(max_display_value)
        ]

    plot_data = [
        filtered_results_df.loc[
            filtered_results_df[group_col].eq(group_key), "tool_calls_per_trial"
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

    for box in boxplot["boxes"]:
        box.set_facecolor(box_color)
        box.set_edgecolor(box_color)
        box.set_alpha(0.75)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tool Calls Per Task Trial")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 30)

    range_frame(ax, np.array([0, 30]), positions)


def plot_ridgeline(
    results_df: pd.DataFrame,
    group_col: str,
    label_map: dict[str, str],
    out_path: Path,
    *,
    figsize: tuple[float, float] = (ONE_COL_WIDTH, ONE_COL_HEIGHT),
) -> None:
    """Render overlapping density estimates for one grouping variable.

    Median reference lines are included because the KDE shape can look smoother
    than the underlying discrete call counts actually are.

    Args:
        results_df: Trial-level tool-call table.
        group_col: Column defining the ridgeline groups.
        label_map: Human-readable labels for group values.
        out_path: Destination path for the saved figure.
        figsize: Base figure size in inches.

    Returns:
        None: The function saves the ridgeline figure.
    """
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
    x_max = get_display_cap(results_df["tool_calls_per_trial"], minimum=1.0)
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

    axes[-1].set_xlabel("Tool Calls Per Task Trial")
    axes[-1].spines["bottom"].set_visible(True)

    fig.subplots_adjust(hspace=-overlap)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_environment_agent_bars_by_model(
    results_df: pd.DataFrame, out_dir: Path
) -> list[Path]:
    """Write one environment-by-scaffold summary for each model.

    Splitting the chart by model avoids a single overloaded legend while still
    making cross-model differences easy to compare by filename.

    Args:
        results_df: Trial-level tool-call table.
        out_dir: Directory where the per-model figures are written.

    Returns:
        list[Path]: Paths of the figures that were successfully saved.
    """
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
        ax.set_ylabel("Average Tool Calls\nPer Task Trial")
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
        out_path = out_dir / OUT_FILE_ENV_AGENT_TEMPLATE.format(
            model_slug=slugify_model_name(model_key)
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main(analysis: bool = False) -> None:
    """Generate the tool-call distribution figures from the combined reports.

    Args:
        analysis: Whether to create the extra per-model environment summaries.

    Returns:
        None: The function saves the requested PDFs and logs skipped trials.
    """
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
    plot_group_boxplots(
        ax,
        results_df,
        "environment",
        ENV_LABELS,
        "Environment",
        max_display_value=ENVIRONMENT_BOXPLOT_MAX,
        box_color=PLOT_COLOR,
    )

    plt.tight_layout()
    fig.savefig(OUT_FILE, bbox_inches="tight")

    plot_ridgeline(results_df, "model", MODEL_LABELS, OUT_FILE_MODEL)

    plot_ridgeline(results_df, "agent_type", AGENT_TYPE_LABELS, OUT_FILE_AGENT)

    env_agent_paths: list[Path] = []
    if analysis:
        ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        env_agent_paths = plot_environment_agent_bars_by_model(
            results_df, ANALYSIS_OUT_DIR
        )

    logger.info(f"Skipped {skipped_trials} trials without tool call counts")
    logger.info(f"Applied display cap at the {DISPLAY_PERCENTILE}th percentile")
    logger.info(f"Saved figure to {OUT_FILE}")
    logger.info(f"Saved figure to {OUT_FILE_MODEL}")
    logger.info(f"Saved figure to {OUT_FILE_AGENT}")
    for path in env_agent_paths:
        logger.info(f"Saved figure to {path}")


if __name__ == "__main__":
    fire.Fire(main)
