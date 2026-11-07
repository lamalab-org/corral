"""Study response length through output tokens normalized by message count.

The analysis focuses on output tokens per message instead of raw completion
tokens so long multi-turn traces are not automatically treated as more verbose
than short traces. In addition to environment-level means, the script provides
distribution plots on log-like axes because a small number of unusually verbose
trials would otherwise dominate the horizontal scale.
"""

import json
import math
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

lama_aesthetics.get_style("main")

DATA_PATH = Path(__file__).parent / "results" / "data" / "reports.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUT_DIR = Path(__file__).parent / "results" / "figures" / "analysis"
ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_ENV = OUT_DIR / "avg_output_tokens_per_message_by_environment.pdf"
OUT_FILE_MODEL_SYMLOG = OUT_DIR / "avg_output_tokens_per_message_by_model_symlog.pdf"
OUT_FILE_AGENT_SYMLOG = (
    OUT_DIR / "avg_output_tokens_per_message_by_agent_type_symlog.pdf"
)
OUT_FILE_ENV_AGENT = (
    ANALYSIS_OUT_DIR / "avg_output_tokens_per_message_by_environment_and_agent.pdf"
)
OUT_FILE_ENV_MODEL = (
    ANALYSIS_OUT_DIR / "avg_output_tokens_per_message_by_environment_and_model.pdf"
)

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
MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
AGENT_COLORS = ["#64B5CD", "#D08770", "#A3BE8C", "#B48EAD"]


def get_axis_max(values: pd.Series | np.ndarray, *, minimum: float = 1.0) -> float:
    """Choose a rounded upper bound that keeps horizontal plots readable.

    Args:
        values: Numeric values that should fit within the plotted x-range.
        minimum: Fallback axis maximum when the input is empty or non-positive.

    Returns:
        float: Rounded axis maximum with a small amount of headroom.
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


def load_trial_dataframe() -> tuple[pd.DataFrame, int]:
    """Extract per-trial output-token rates from the nested report dataset.

    Trials are skipped when either completion-token accounting or message
    histories are missing, because the normalization would otherwise be
    ill-defined.

    Returns:
        tuple[pd.DataFrame, int]: Long-form trial table and the number of trials
        excluded for missing token or message information.
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
                    token_usage = trial.get("token_usage") or {}
                    completion_tokens = token_usage.get("completion_tokens")
                    messages = trial.get("messages")

                    is_missing_messages = not messages
                    if completion_tokens is None or is_missing_messages:
                        skipped_trials += 1
                        continue

                    message_count = len(messages)
                    if message_count == 0:
                        skipped_trials += 1
                        continue

                    records.append(
                        {
                            "environment": environment,
                            "model": model,
                            "agent_type": agent_type,
                            "output_tokens_per_message": completion_tokens
                            / message_count,
                        }
                    )

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise ValueError(
            f"No trials with both completion tokens and messages found in {DATA_PATH}"
        )

    return results_df, skipped_trials


def plot_environment_summary(results_df: pd.DataFrame) -> None:
    """Write the environment-level mean chart for output tokens per message.

    Args:
        results_df: Trial-level output-token table.

    Returns:
        None: The function saves the environment summary figure.
    """
    agg = results_df.groupby("environment", as_index=False)[
        "output_tokens_per_message"
    ].mean()
    environments = sorted(agg["environment"].dropna().unique())
    y_pos = np.arange(len(environments))
    tokens = [
        agg.loc[agg["environment"].eq(environment), "output_tokens_per_message"].iloc[0]
        for environment in environments
    ]
    labels = [
        ENV_LABELS.get(environment) or str(environment).capitalize()
        for environment in environments
    ]

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    ax.hlines(y_pos, 0, tokens, color=PLOT_COLOR, alpha=0.5, linewidth=5)
    ax.plot(tokens, y_pos, "o", markersize=6, color="black", alpha=0)
    for y, token_count in zip(y_pos, tokens, strict=True):
        ax.plot(token_count, y, "o", markersize=6, color=PLOT_COLOR, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Output Tokens Per Message")
    ax.set_ylabel("Environment")

    range_frame(ax, np.array([0, get_axis_max(np.array(tokens), minimum=1.0)]), y_pos)

    plt.tight_layout()
    fig.savefig(OUT_FILE_ENV, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {OUT_FILE_ENV}")


def plot_scaled_distribution_summary(
    results_df: pd.DataFrame,
    *,
    group_col: str,
    label_map: dict[str, str],
    colors: list[str],
    out_file: Path,
    x_scale: str = "symlog",
    show_legend: bool = False,
) -> None:
    """Write a distribution plot that remains legible under heavy right tails.

    `symlog` is the safer default because a few trials have zero-valued rates,
    while pure `log` scaling is only meaningful after those trials are dropped.

    Args:
        results_df: Trial-level output-token table.
        group_col: Column defining the comparison groups.
        label_map: Human-readable labels for group values.
        colors: Color cycle used for the grouped point clouds.
        out_file: Destination path for the saved figure.
        x_scale: Either `log` or `symlog`.
        show_legend: Whether to show the distribution-encoding legend.

    Returns:
        None: The function saves the requested figure.
    """
    if x_scale not in {"log", "symlog"}:
        raise ValueError("x_scale must be either 'log' or 'symlog'")

    grouped = (
        results_df.groupby(group_col)["output_tokens_per_message"]
        .mean()
        .sort_values(ascending=True)
    )
    groups = list(grouped.index)
    raw_values_per_group = [
        results_df.loc[results_df[group_col].eq(group), "output_tokens_per_message"]
        .dropna()
        .to_numpy(dtype=float)
        for group in groups
    ]

    if x_scale == "log":
        values_per_group = [values[values > 0] for values in raw_values_per_group]
        skipped_non_positive = sum(
            len(raw_values) - len(values)
            for raw_values, values in zip(
                raw_values_per_group, values_per_group, strict=True
            )
        )
        if skipped_non_positive:
            logger.warning(
                "Skipped "
                f"{skipped_non_positive} non-positive trials in the log-scaled "
                f"{group_col} distribution plot"
            )
    else:
        values_per_group = raw_values_per_group

    non_empty = [
        (group, values)
        for group, values in zip(groups, values_per_group, strict=True)
        if len(values) > 0
    ]
    if not non_empty:
        logger.warning(
            f"Skipping scaled distribution plot for {group_col}: insufficient data"
        )
        return

    groups = [group for group, _ in non_empty]
    values_per_group = [values for _, values in non_empty]
    positions = np.arange(1, len(groups) + 1)
    labels = [label_map.get(group, str(group)) for group in groups]
    plot_colors = [colors[i % len(colors)] for i in range(len(groups))]
    flattened_values = np.concatenate(values_per_group)
    positive_values = flattened_values[flattened_values > 0]

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH * 0.75, ONE_COL_HEIGHT))

    boxplot = ax.boxplot(
        values_per_group,
        vert=False,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1f1f1f", "linewidth": 1.4},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )

    for patch in boxplot["boxes"]:
        patch.set_visible(False)

    for idx, (position, values, color) in enumerate(
        zip(positions, values_per_group, plot_colors, strict=True)
    ):
        rng = np.random.default_rng(100 + idx)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), position) + jitter,
            s=3,
            color=color,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax.scatter(
            values.mean(),
            position,
            marker="D",
            s=52,
            color="#1f1f1f",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Output Tokens Per Message ({x_scale.title()} Scale)")

    upper_limit = get_axis_max(flattened_values, minimum=1.0)
    if x_scale == "log":
        lower_limit = float(positive_values.min())
        if lower_limit >= 1:
            lower_limit = 10 ** math.floor(math.log10(lower_limit))
        else:
            lower_limit *= 0.9
        ax.set_xscale("log")
        x_range = np.array([lower_limit, upper_limit])
        x_ticks = np.array([10**exp for exp in range(5)], dtype=float)
        x_ticks = x_ticks[(x_ticks >= lower_limit) & (x_ticks <= upper_limit)]
    else:
        linthresh = 1.0
        if len(positive_values) > 0:
            linthresh = max(1.0, float(np.quantile(positive_values, 0.05)))
        ax.set_xscale("symlog", linthresh=linthresh)
        x_range = np.array([0, upper_limit])
        x_ticks = np.array([10**exp for exp in range(5)], dtype=float)
        x_ticks = x_ticks[x_ticks <= upper_limit]

    if len(x_ticks) == 0:
        x_ticks = np.array([x_range[-1]])

    # Force the left spine to terminate on an explicit tick even on symlog axes.
    if x_ticks[0] != 0:
        x_ticks = np.concatenate([[0], x_ticks])
    # Keep the right spine anchored to a fixed endpoint across related figures.
    if x_ticks[-1] != 1e4:
        x_ticks = np.concatenate([x_ticks, [1e4]])

    ax.set_xticks(x_ticks)
    tick_labels = []
    for tick in x_ticks:
        if tick == 0:
            tick_labels.append("")  # hide the auto "0" label; placed manually below
        else:
            tick_labels.append(rf"$10^{{{int(math.log10(tick))}}}$")
    ax.set_xticklabels(tick_labels)

    # Manual placement prevents the zero label from colliding with the axis spine.
    ax.annotate(
        "$0$",
        xy=(0, 0),
        xycoords=("data", "axes fraction"),
        xytext=(-6, -14),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=ax.xaxis.get_major_ticks()[0].label1.get_fontsize(),
        annotation_clip=False,
    )

    # `range_frame()` assumes linear padding; on log-like axes that produces an
    # oversized empty region near zero and compresses the actual distributions.
    y_min, y_max = positions.min(), positions.max()
    y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 0.5
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(-0.5, 1e4 * 1.5)
    ax.spines["left"].set_bounds(y_min, y_max)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(0, 1e4)
    ax.spines["bottom"].set_position(("outward", 10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color="#1f1f1f", linewidth=1.4, label="Median"),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#1f1f1f",
                markerfacecolor="#1f1f1f",
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=0,
                markersize=7,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#666666",
                markerfacecolor="#666666",
                linewidth=0,
                alpha=0.35,
                markersize=5,
                label="Trials",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure to {out_file}")


def get_group_centers(
    n_groups: int, *, stem_width: float = 0.22, gap_width: float = 0.18
) -> np.ndarray:
    """Compute group anchors for the grouped environment summaries.

    Args:
        n_groups: Number of environments to plot.
        stem_width: Horizontal footprint allocated to one subgroup.
        gap_width: Extra spacing inserted between adjacent environments.

    Returns:
        np.ndarray: Center coordinate for each environment group.
    """
    return np.arange(n_groups) * (
        len(range(1)) * 0 + stem_width + gap_width + stem_width * 0.3
    )


def get_group_frame(x_centers: np.ndarray) -> np.ndarray:
    """Return axis bounds that remain valid for empty or populated group layouts.

    Args:
        x_centers: Group center coordinates on the x-axis.

    Returns:
        np.ndarray: Two-value range suitable for `range_frame()`.
    """
    if len(x_centers) == 0:
        return np.array([0.0, 1.0])
    return np.array([x_centers.min(), x_centers.max()])


def plot_environment_grouped_lollipops(
    results_df: pd.DataFrame,
    *,
    group_col: str,
    label_map: dict[str, str],
    colors: list[str],
    legend_title: str,
    out_file: Path,
) -> None:
    """Write environment averages split by a secondary grouping variable.

    This view is only generated in analysis mode because it is mainly useful for
    checking whether environment-level means are driven by particular models or
    scaffolds.

    Args:
        results_df: Trial-level output-token table.
        group_col: Column used to split each environment into multiple stems.
        label_map: Human-readable labels for group values.
        colors: Color cycle for subgroup markers.
        legend_title: Title shown above the subgroup legend.
        out_file: Destination path for the saved figure.

    Returns:
        None: The function saves the grouped environment figure.
    """
    environments = sorted(results_df["environment"].dropna().unique())
    groups = sorted(results_df[group_col].dropna().unique())
    if not environments or not groups:
        logger.warning(
            f"Skipping grouped environment plot for {group_col}: insufficient data"
        )
        return

    summary = results_df.groupby(["environment", group_col], as_index=False)[
        "output_tokens_per_message"
    ].mean()

    stem_width = 0.18
    gap_width = 0.16
    x_centers = np.arange(len(environments)) * (
        len(groups) * stem_width + gap_width + stem_width * 0.3
    )

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))

    for idx, group in enumerate(groups):
        offsets = x_centers + (idx - (len(groups) - 1) / 2) * stem_width
        means = []
        for environment in environments:
            match = summary.loc[
                summary["environment"].eq(environment) & summary[group_col].eq(group),
                "output_tokens_per_message",
            ]
            means.append(float(match.iloc[0]) if not match.empty else np.nan)

        valid_offsets = [
            offset
            for offset, score in zip(offsets, means, strict=True)
            if not np.isnan(score)
        ]
        valid_scores = [score for score in means if not np.isnan(score)]
        color = colors[idx % len(colors)]
        ax.vlines(
            valid_offsets,
            0,
            valid_scores,
            color=color,
            alpha=0.5,
            linewidth=4,
            label=label_map.get(group, str(group)),
        )
        ax.plot(
            valid_offsets,
            valid_scores,
            "o",
            markersize=5.5,
            color=color,
            alpha=0.7,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [ENV_LABELS.get(environment, str(environment)) for environment in environments]
    )
    ax.set_xlabel("Environment")
    ax.set_ylabel("Average Output\nTokens Per Message")
    range_frame(
        ax,
        get_group_frame(x_centers),
        np.array([0, get_axis_max(summary["output_tokens_per_message"], minimum=1.0)]),
    )
    ax.legend(title=legend_title, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {out_file}")


def main(analysis: bool = False) -> None:
    """Generate the output-token figures, with optional subgroup breakdowns.

    Args:
        analysis: Whether to emit the extra environment-by-subgroup figures.

    Returns:
        None: The function saves the requested PDFs and logs skipped trials.
    """
    results_df, skipped_trials = load_trial_dataframe()

    plot_environment_summary(results_df)
    if analysis:
        plot_environment_grouped_lollipops(
            results_df,
            group_col="agent_type",
            label_map=AGENT_TYPE_LABELS,
            colors=AGENT_COLORS,
            legend_title="Agent",
            out_file=OUT_FILE_ENV_AGENT,
        )
        plot_environment_grouped_lollipops(
            results_df,
            group_col="model",
            label_map=MODEL_LABELS,
            colors=MODEL_COLORS,
            legend_title="Model",
            out_file=OUT_FILE_ENV_MODEL,
        )
    plot_scaled_distribution_summary(
        results_df,
        group_col="model",
        label_map=MODEL_LABELS,
        colors=MODEL_COLORS,
        out_file=OUT_FILE_MODEL_SYMLOG,
        x_scale="symlog",
        show_legend=True,
    )
    plot_scaled_distribution_summary(
        results_df,
        group_col="agent_type",
        label_map=AGENT_TYPE_LABELS,
        colors=AGENT_COLORS,
        out_file=OUT_FILE_AGENT_SYMLOG,
        x_scale="symlog",
    )

    logger.info(f"Skipped {skipped_trials} trials without message histories")


if __name__ == "__main__":
    fire.Fire(main)
