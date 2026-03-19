"""Combine environment-behavior summaries into one labeled multi-panel figure.

The panel gathers the main environment-facing views from the action-mix,
output-token, and tool-call analyses so they can be compared side by side in a
single appendix-style figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_action_distributions_across_environments as action_plots
import plot_avg_output_tokens as output_token_plots
import plot_avg_tool_calls_per_task as tool_call_plots
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

lama_aesthetics.get_style("main")

LABEL_SIZE = 10

OUT_DIR = Path(__file__).parent / "results" / "figures" / "fig_4_app"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "app_fig4_behavior_panel.pdf"


def add_panel_label(ax, label: str, x: float = -0.18, y: float = 1.08) -> None:
    """Place a bold panel label just outside an axis.

    Args:
        ax: Axis that receives the label.
        label: Panel identifier.
        x: Horizontal position in axes coordinates.
        y: Vertical position in axes coordinates.

    Returns:
        None: The function mutates the provided axis.
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=16,
        color="black",
        ha="center",
        va="center",
        clip_on=False,
    )


def load_tool_call_dataframe() -> tuple[pd.DataFrame, int]:
    """Extract trial-level tool-call counts from the combined report file.

    Returns:
        tuple[pd.DataFrame, int]: Long-form tool-call table and the number of
        skipped trials without a recoverable call count.
    """
    records: list[dict[str, object]] = []
    skipped_trials = 0

    with tool_call_plots.DATA_PATH.open() as fh:
        for raw_line in fh:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue

            row = json.loads(stripped_line)
            task_results = row.get("Task Results", {})
            if isinstance(task_results, str):
                task_results = json.loads(task_results)

            for task_data in task_results.values():
                for trial in task_data.get("trials", []):
                    tool_call_count = tool_call_plots.extract_tool_call_count(trial)
                    if tool_call_count is None:
                        skipped_trials += 1
                        continue

                    records.append(
                        {
                            "environment": row.get("environment"),
                            "model": row.get("model"),
                            "agent_type": row.get("agent_type"),
                            "tool_calls_per_trial": tool_call_count,
                        }
                    )

    results_df = pd.DataFrame(records)
    if results_df.empty:
        raise ValueError(
            f"No trials with tool call counts found in {tool_call_plots.DATA_PATH}"
        )

    return results_df, skipped_trials


def add_text_legend(
    ax,
    title: str,
    entries: list[tuple[str, str]],
    *,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] | None = None,
    ncol: int = 1,
) -> None:
    """Add a compact text-only legend block to an axis.

    Args:
        ax: Axis that receives the legend.
        title: Legend title.
        entries: Sequence of `(marker, label)` pairs.
        loc: Legend location passed to Matplotlib.
        bbox_to_anchor: Optional anchor point for legend placement.
        ncol: Number of legend columns.

    Returns:
        None: The function mutates the provided axis.
    """
    handles = [
        Line2D([], [], linestyle="None", label=f"{marker}: {label}")
        for marker, label in entries
    ]
    legend_kwargs: dict = {
        "handles": handles,
        "title": title,
        "loc": loc,
        "frameon": False,
        "handlelength": 0,
        "handletextpad": 0,
        "borderpad": 0.2,
        "labelspacing": 0.3,
        "fontsize": LABEL_SIZE,
        "title_fontsize": LABEL_SIZE,
        "ncol": ncol,
    }
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    ax.legend(**legend_kwargs)


def plot_action_distribution_panel(
    ax,
    distribution_df: pd.DataFrame,
    *,
    subgroup_col: str,
    subgroup_order: list[str],
    subgroup_labels: dict[str, str],
) -> None:
    """Draw one environment-level stacked action-share panel on an axis.

    Args:
        ax: Axis that receives the plot.
        distribution_df: Long-form action-share table.
        subgroup_col: Column used to split each environment into subgroups.
        subgroup_order: Ordered subgroup values to plot.
        subgroup_labels: Display labels for subgroup values.

    Returns:
        None: The function mutates the provided axis.
    """
    env_order = [
        env
        for env in action_plots.ENV_LABELS
        if env in set(distribution_df["environment"])
    ]
    pivot_df = action_plots.make_pivot(distribution_df, ["environment", subgroup_col])

    n_env = len(env_order)
    n_subgroups = len(subgroup_order)
    x_centers = np.arange(n_env, dtype=float)
    total_group_width = 0.8
    bar_width = total_group_width / max(n_subgroups, 1)
    offsets = (np.arange(n_subgroups, dtype=float) - (n_subgroups - 1) / 2) * bar_width
    subgroup_markers = {
        subgroup: chr(ord("a") + subgroup_index)
        for subgroup_index, subgroup in enumerate(subgroup_order)
    }

    for subgroup_index, subgroup in enumerate(subgroup_order):
        x_positions = x_centers + offsets[subgroup_index]
        bottom = np.zeros(n_env, dtype=float)

        for action_category in action_plots.ACTION_ORDER:
            heights: list[float] = []
            for environment in env_order:
                index_key = (environment, subgroup)
                if index_key in pivot_df.index:
                    value = pd.to_numeric(
                        pd.Series([pivot_df.loc[index_key, action_category]]),
                        errors="coerce",
                    ).iloc[0]
                    heights.append(0.0 if pd.isna(value) else float(value))
                else:
                    heights.append(0.0)

            ax.bar(
                x_positions,
                heights,
                width=bar_width * 0.92,
                bottom=bottom,
                color=action_plots.ACTION_COLORS[action_category],
                edgecolor="white",
                linewidth=0.4,
            )
            bottom += np.array(heights, dtype=float)

    ax.set_xlim(-0.6, max(n_env - 0.4, 0.6))
    ax.set_ylim(0, 1.06)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [action_plots.ENV_LABELS.get(env, str(env).capitalize()) for env in env_order]
    )
    ax.set_ylabel("Fraction of Tool Calls by Action Type")

    if n_env > 0:
        range_frame(
            ax,
            np.array([x_centers.min(), x_centers.max()]),
            np.array([0, 1.0]),
            pad=0.1,
            pad_x=0.0,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    add_text_legend(
        ax,
        subgroup_col.replace("_", " ").title(),
        [
            (subgroup_markers[subgroup], subgroup_labels.get(subgroup, str(subgroup)))
            for subgroup in subgroup_order
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(subgroup_order),
    )

    if n_env > 0 and n_subgroups > 1:
        example_positions = x_centers[0] + offsets
        for x_position, subgroup in zip(example_positions, subgroup_order, strict=True):
            ax.text(
                x_position,
                1.015,
                subgroup_markers[subgroup],
                transform=ax.transData,
                ha="center",
                va="bottom",
                fontsize=LABEL_SIZE,
                fontweight="bold",
                clip_on=False,
            )


def plot_output_tokens_environment(ax, results_df: pd.DataFrame) -> None:
    """Draw the environment-level output-token summary on an axis.

    Args:
        ax: Axis that receives the plot.
        results_df: Trial-level output-token table.

    Returns:
        None: The function mutates the provided axis.
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
        action_plots.ENV_LABELS.get(environment) or str(environment).capitalize()
        for environment in environments
    ]

    ax.hlines(
        y_pos,
        0,
        tokens,
        color=output_token_plots.PLOT_COLOR,
        alpha=0.5,
        linewidth=5,
    )
    for y, token_count in zip(y_pos, tokens, strict=True):
        ax.plot(
            token_count,
            y,
            "o",
            markersize=6,
            color=output_token_plots.PLOT_COLOR,
            alpha=0.85,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Output Tokens Per Message")
    ax.set_ylabel("")
    range_frame(
        ax,
        np.array(
            [
                0,
                output_token_plots.get_axis_max(np.array(tokens), minimum=1.0),
            ]
        ),
        y_pos,
    )


def plot_output_token_distribution(
    ax,
    results_df: pd.DataFrame,
    *,
    group_col: str,
    label_map: dict[str, str],
    colors: list[str],
    x_scale: str = "symlog",
    show_legend: bool = False,
) -> None:
    """Draw an output-token distribution panel on an existing axis.

    Args:
        ax: Axis that receives the plot.
        results_df: Trial-level output-token table.
        group_col: Column defining the comparison groups.
        label_map: Human-readable labels for group values.
        colors: Color cycle for the grouped point clouds.
        x_scale: Either `log` or `symlog`.
        show_legend: Whether to draw the distribution encoding legend.

    Returns:
        None: The function mutates the provided axis.
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
    ax.set_ylabel("")

    upper_limit = output_token_plots.get_axis_max(flattened_values, minimum=1.0)
    if x_scale == "log":
        lower_limit = float(positive_values.min())
        if lower_limit >= 1:
            lower_limit = 10 ** np.floor(np.log10(lower_limit))
        else:
            lower_limit *= 0.9
        ax.set_xscale("log")
        x_ticks = np.array([10**exp for exp in range(5)], dtype=float)
        x_ticks = x_ticks[(x_ticks >= lower_limit) & (x_ticks <= upper_limit)]
    else:
        linthresh = 1.0
        if len(positive_values) > 0:
            linthresh = max(1.0, float(np.quantile(positive_values, 0.05)))
        ax.set_xscale("symlog", linthresh=linthresh)
        x_ticks = np.array([10**exp for exp in range(5)], dtype=float)
        x_ticks = x_ticks[x_ticks <= upper_limit]

    if len(x_ticks) == 0:
        x_ticks = np.array([upper_limit])

    if x_ticks[0] != 0:
        x_ticks = np.concatenate([[0], x_ticks])
    if x_ticks[-1] != 1e4:
        x_ticks = np.concatenate([x_ticks, [1e4]])

    ax.set_xticks(x_ticks)
    tick_labels = []
    for tick in x_ticks:
        if tick == 0:
            tick_labels.append("")
        else:
            tick_labels.append(rf"$10^{{{int(np.log10(tick))}}}$")
    ax.set_xticklabels(tick_labels)
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


def plot_tool_call_ridgeline_panel(
    fig,
    panel_spec,
    results_df: pd.DataFrame,
    *,
    group_col: str,
    label_map: dict[str, str],
):
    """Draw one ridgeline panel inside a nested subplot specification.

    Args:
        fig: Parent figure.
        panel_spec: Subplot specification for the full panel.
        results_df: Trial-level tool-call table.
        group_col: Column defining the ridgeline groups.
        label_map: Human-readable labels for group values.

    Returns:
        The topmost axis of the ridgeline stack, or *None* when no data is
        available.
    """
    group_keys = sorted(results_df[group_col].dropna().unique())
    plot_data = [
        results_df.loc[results_df[group_col].eq(group_key), "tool_calls_per_trial"]
        .to_numpy()
        .astype(float)
        for group_key in group_keys
    ]
    non_empty = [
        (group_key, group_values)
        for group_key, group_values in zip(group_keys, plot_data, strict=True)
        if len(group_values) > 0
    ]
    if not non_empty:
        logger.warning(f"Skipping ridgeline plot for {group_col}: insufficient data")
        return None

    group_keys = [group_key for group_key, _ in non_empty]
    plot_data = [group_values for _, group_values in non_empty]
    labels = [label_map.get(group_key, str(group_key)) for group_key in group_keys]
    n_groups = len(group_keys)
    overlap = 0.6
    inner_grid = panel_spec.subgridspec(n_groups, 1, hspace=-overlap)

    axes = []
    shared_ax = None
    x_min = 0.0
    x_max = tool_call_plots.get_display_cap(
        results_df["tool_calls_per_trial"], minimum=1.0
    )
    x_grid = np.linspace(x_min, x_max, 500)

    for idx in range(n_groups):
        ax = fig.add_subplot(inner_grid[idx, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        axes.append(ax)

        values = plot_data[idx]
        color = tool_call_plots.GROUP_COLORS[idx % len(tool_call_plots.GROUP_COLORS)]

        if values.std() == 0:
            density = np.zeros_like(x_grid)
            density[np.argmin(np.abs(x_grid - values[0]))] = 1.0
        else:
            kde = tool_call_plots.gaussian_kde(values, bw_method=0.35)
            density = kde(x_grid)

        density = density / density.max() if density.max() > 0 else density
        ax.fill_between(x_grid, density, alpha=0.65, color=color)
        ax.plot(x_grid, density, color=color, linewidth=1.2)
        ax.axvline(
            float(np.median(values)),
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )

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
            fontsize=LABEL_SIZE,
        )

        if idx < n_groups - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False)

    axes[-1].set_xlabel("Tool Calls Per Task Trial")
    axes[-1].spines["bottom"].set_visible(True)
    return axes[0]


def main() -> None:
    """Build and save the combined environment-behavior panel.

    Returns:
        None: The function saves the panel PDF.
    """
    reports_df = action_plots.load_reports_df()
    action_distribution_df, unknown_tools = action_plots.build_action_distribution_df(
        reports_df
    )
    output_token_df, skipped_output_trials = output_token_plots.load_trial_dataframe()
    tool_call_df, skipped_tool_call_trials = load_tool_call_dataframe()

    if action_distribution_df.empty:
        raise RuntimeError(
            "No tool-level action data could be extracted from reports.jsonl"
        )

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 6.5 * ONE_COL_HEIGHT))
    outer_grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.9, 1, 1, 1, 1],
        hspace=0.65,
    )
    top_section_grid = outer_grid[0].subgridspec(
        2,
        1,
        height_ratios=[11.2, 0.2],
        hspace=0.45,
    )
    top_grid = top_section_grid[0].subgridspec(1, 2, wspace=0.26)
    token_dist_grid = outer_grid[2].subgridspec(1, 2, wspace=0.40)
    ridge_grid = outer_grid[4].subgridspec(1, 2, wspace=0.42)

    ax_action_agent = fig.add_subplot(top_grid[0, 0])
    ax_action_model = fig.add_subplot(top_grid[0, 1])
    ax_action_legend = fig.add_subplot(top_section_grid[1, 0])
    ax_output_tokens = fig.add_subplot(outer_grid[1, 0])
    ax_output_tokens_agent = fig.add_subplot(token_dist_grid[0, 0])
    ax_output_tokens_model = fig.add_subplot(token_dist_grid[0, 1])
    ax_tool_calls = fig.add_subplot(outer_grid[3, 0])

    ax_action_legend.axis("off")

    plot_action_distribution_panel(
        ax_action_agent,
        action_distribution_df,
        subgroup_col="agent_type",
        subgroup_order=[
            agent
            for agent in action_plots.AGENT_TYPE_LABELS
            if agent in set(action_distribution_df["agent_type"])
        ],
        subgroup_labels=action_plots.AGENT_TYPE_LABELS,
    )
    plot_action_distribution_panel(
        ax_action_model,
        action_distribution_df,
        subgroup_col="model",
        subgroup_order=[
            model
            for model in action_plots.MODEL_LABELS
            if model in set(action_distribution_df["model"])
        ],
        subgroup_labels=action_plots.MODEL_LABELS,
    )
    plot_output_tokens_environment(ax_output_tokens, output_token_df)
    plot_output_token_distribution(
        ax_output_tokens_agent,
        output_token_df,
        group_col="agent_type",
        label_map=action_plots.AGENT_TYPE_LABELS,
        colors=output_token_plots.AGENT_COLORS,
        x_scale="symlog",
        show_legend=False,
    )
    plot_output_token_distribution(
        ax_output_tokens_model,
        output_token_df,
        group_col="model",
        label_map=action_plots.MODEL_LABELS,
        colors=output_token_plots.MODEL_COLORS,
        x_scale="symlog",
        show_legend=True,
    )
    tool_call_plots.plot_group_boxplots(
        ax_tool_calls,
        tool_call_df,
        "environment",
        action_plots.ENV_LABELS,
        "",
        max_display_value=tool_call_plots.ENVIRONMENT_BOXPLOT_MAX,
        box_color=tool_call_plots.PLOT_COLOR,
    )
    ridge_ax_agent = plot_tool_call_ridgeline_panel(
        fig,
        ridge_grid[0, 0],
        tool_call_df,
        group_col="agent_type",
        label_map=action_plots.AGENT_TYPE_LABELS,
    )
    ridge_ax_model = plot_tool_call_ridgeline_panel(
        fig,
        ridge_grid[0, 1],
        tool_call_df,
        group_col="model",
        label_map=action_plots.MODEL_LABELS,
    )

    category_handles = [
        Patch(
            facecolor=action_plots.ACTION_COLORS[action_category],
            label=action_plots.ACTION_LABELS[action_category],
        )
        for action_category in action_plots.ACTION_ORDER
    ]
    ax_action_legend.legend(
        handles=category_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.6),
        ncol=len(action_plots.ACTION_ORDER),
        frameon=False,
        title="Action Types",
        fontsize=LABEL_SIZE,
        title_fontsize=LABEL_SIZE,
    )

    # Unify tick label sizes across all axes
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=LABEL_SIZE)

    # --- Place panel labels with pixel-perfect column alignment ------------
    # Render once so all layout positions are finalised.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _tight_fig_bbox(ax):
        """Return the tight bounding box of *ax* in figure coordinates."""
        tb = ax.get_tightbbox(renderer)
        if tb is None:
            return ax.get_position()
        return tb.transformed(fig.transFigure.inverted())

    # Define left-column and right-column label specifications.
    left_specs = [
        (ax_action_agent, "A"),
        (ax_output_tokens, "C"),
        (ax_output_tokens_agent, "D"),
        (ax_tool_calls, "F"),
    ]
    right_specs = [
        (ax_action_model, "B"),
        (ax_output_tokens_model, "E"),
    ]
    if ridge_ax_agent is not None:
        left_specs.append((ridge_ax_agent, "G"))
    if ridge_ax_model is not None:
        right_specs.append((ridge_ax_model, "H"))

    left_bboxes = [_tight_fig_bbox(ax) for ax, _ in left_specs]
    right_bboxes = [_tight_fig_bbox(ax) for ax, _ in right_specs]

    # Aligned x = leftmost tight-bbox edge in each column, minus padding.
    X_PAD = 0.018
    RIGHT_X_OFFSET = 0.1  # shift B, E, H a bit to the right
    Y_PAD_TOP = 0.008  # y padding for A and B (top row)
    Y_PAD_REST = 0.002  # y padding for C-H (lower rows, moved down)
    left_x = min(bb.x0 for bb in left_bboxes) - X_PAD
    right_x = min(bb.x0 for bb in right_bboxes) - X_PAD + RIGHT_X_OFFSET

    top_labels = {"A", "B"}
    for (_ax, label), bb in zip(left_specs, left_bboxes, strict=True):
        y_pad = Y_PAD_TOP if label in top_labels else Y_PAD_REST
        fig.text(
            left_x,
            bb.y1 + y_pad,
            label,
            fontweight="bold",
            fontsize=16,
            color="black",
            ha="right",
            va="bottom",
            clip_on=False,
        )
    for (_ax, label), bb in zip(right_specs, right_bboxes, strict=True):
        y_pad = Y_PAD_TOP if label in top_labels else Y_PAD_REST
        fig.text(
            right_x,
            bb.y1 + y_pad,
            label,
            fontweight="bold",
            fontsize=16,
            color="black",
            ha="right",
            va="bottom",
            clip_on=False,
        )

    logger.info(f"Label alignment -- left_x={left_x:.4f}, right_x={right_x:.4f}")

    fig.savefig(OUT_FILE, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved figure to {OUT_FILE}")
    logger.info(f"Skipped {skipped_output_trials} trials without message histories")
    logger.info(f"Skipped {skipped_tool_call_trials} trials without tool call counts")
    if unknown_tools:
        logger.info(
            "Encountered {} uncategorized tool names while building action shares",
            len(unknown_tools),
        )


if __name__ == "__main__":
    main()
