"""Full coverage heatmap showing all model x agent configurations across all environment-level combinations.

Creates a heatmap where rows are model x agent configurations (6 total) and columns are
all environment-level combinations tested (e.g., AFM-1, AFM-2, Catalyst-1, MD-1, MD-2, etc.).

Usage:
    python plot_panel2_coverage_heatmap.py --verbosity_strategy=average --task_type_strategy=both
"""

from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger
from plot_config import (
    AGENT_NAMES,
    ENVIRONMENT_GROUPS,
    ENVIRONMENT_MAX_LEVELS,
    ENVIRONMENT_NAMES,
    FONT_SIZES,
    MODEL_NAMES,
)
from plot_utils import (
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== DATA COLLECTION ====================


def collect_full_coverage_data(
    df: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Collect data for full coverage heatmap."""
    data = df.copy()
    data["config"] = data["model"] + "_" + data["agent_type"]
    data["env_level"] = data["environment"] + "-" + data["level"].astype(str)
    return data.pivot_table(
        values=metric_column,
        index="config",
        columns="env_level",
        aggfunc="mean",
    )


def get_ordered_env_level_columns() -> list[str]:
    """Return env-level column keys ordered by environment group then increasing level."""
    return [
        f"{env}-{level}"
        for group_info in ENVIRONMENT_GROUPS.values()
        for env in group_info["environments"]
        for level in range(1, ENVIRONMENT_MAX_LEVELS.get(env, 1) + 1)
    ]


def format_heatmap_data(pivot_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Format and reorder heatmap data."""
    display_df = pivot_df.copy()

    ordered_cols = get_ordered_env_level_columns()
    ordered_cols = [c for c in ordered_cols if c in display_df.columns]
    remaining = [c for c in display_df.columns if c not in ordered_cols]
    display_df = display_df[ordered_cols + remaining]

    model_order = list(MODEL_NAMES.keys())
    agent_order = list(AGENT_NAMES.keys())
    parsed_rows = []
    for config in display_df.index:
        parts = config.split("_")
        model = parts[0] if parts else config
        agent = "_".join(parts[1:]) if len(parts) >= 2 else ""
        m_idx = model_order.index(model) if model in model_order else 99
        a_idx = agent_order.index(agent) if agent in agent_order else 99
        parsed_rows.append((m_idx, a_idx, model, agent, config))
    parsed_rows.sort(key=lambda x: (x[0], x[1]))

    new_rows = []
    row_meta = []
    prev_model = None
    for _m_idx, _a_idx, model, agent, config in parsed_rows:
        is_first = model != prev_model
        new_rows.append(display_df.loc[config])
        row_meta.append(
            {
                "model": MODEL_NAMES.get(model, model),
                "agent": AGENT_NAMES.get(agent, agent),
                "is_first_of_model": is_first,
            }
        )
        prev_model = model

    display_df = pd.DataFrame(new_rows)

    col_labels = []
    env_spans = []
    prev_env = None
    for idx, env_level in enumerate(display_df.columns):
        parts = env_level.split("-")
        if len(parts) >= 2:
            env = parts[0]
            level = parts[1]
            env_name = ENVIRONMENT_NAMES.get(env, env.upper())
            col_labels.append(f"S{level}")
            if env != prev_env:
                if env_spans:
                    env_spans[-1] = (*env_spans[-1][:2], idx - 1)
                env_spans.append((env_name, idx, idx))
                prev_env = env
            else:
                env_spans[-1] = (*env_spans[-1][:2], idx)
        else:
            col_labels.append(env_level)
    display_df.columns = col_labels

    return display_df, row_meta, env_spans


# ==================== PLOTTING ====================


def _get_group_column_spans(n_cols_total: int) -> list[tuple[str, int, int]]:
    """Compute (group_name, col_start, col_end) spans for the ordered columns."""
    spans = []
    col_idx = 0
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        start = col_idx
        for env in group_info["environments"]:
            max_level = ENVIRONMENT_MAX_LEVELS.get(env, 1)
            col_idx += max_level
        if col_idx > start:
            spans.append((group_name, start, col_idx - 1))
    return [(g, s, min(e, n_cols_total - 1)) for g, s, e in spans if s < n_cols_total]


def plot_full_coverage_heatmap(
    heatmap_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create full coverage heatmap with marginal mean bar charts."""
    if heatmap_data.empty:
        logger.warning("No data available for heatmap!")
        return

    display_df, row_meta, env_spans = format_heatmap_data(heatmap_data)

    n_cols = len(display_df.columns)
    n_rows = len(display_df.index)

    col_means = display_df.mean(axis=0, skipna=True)
    row_means = display_df.mean(axis=1, skipna=True)

    fig_width = min(TWO_COL_WIDTH * 1.5, max(TWO_COL_WIDTH, n_cols * 0.4))
    fig_height = max(ONE_COL_HEIGHT * 0.6, n_rows * 0.35)

    bar_size = 1.0
    bar_ratio_w = bar_size / fig_width
    bar_ratio_h = bar_size / fig_height

    fig = plt.figure(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1, bar_ratio_w],
        height_ratios=[bar_ratio_h, 1],
        wspace=0.02,
        hspace=0.02,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        display_df,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        cbar=False,
        ax=ax_heatmap,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"]},
    )

    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_heatmap.axhline(y=i, color="white", linewidth=3, zorder=5)

    group_spans = _get_group_column_spans(n_cols)
    for _group_name, _col_start, col_end in group_spans[:-1]:
        ax_heatmap.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)

    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel("")

    model_groups = []
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"]:
            model_groups.append({"model": meta["model"], "start": i})
            if len(model_groups) > 1:
                model_groups[-2]["end"] = i
    if model_groups:
        model_groups[-1]["end"] = len(row_meta)

    for i, meta in enumerate(row_meta):
        ax_heatmap.text(
            -0.01,
            i + 0.5,
            meta["agent"],
            ha="right",
            va="center",
            fontsize=FONT_SIZES["tick_label"],
            transform=ax_heatmap.get_yaxis_transform(),
        )

    ax_heatmap.set_xlabel("")
    ax_heatmap.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, ha="center")

    for env_name, col_start, col_end in env_spans:
        start_frac = (col_start + 0.15) / n_cols
        end_frac = (col_end + 0.85) / n_cols
        ax_heatmap.plot(
            [start_frac, end_frac],
            [-0.01, -0.01],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax_heatmap.transAxes,
        )
        ax_heatmap.text(
            start_frac,
            -0.06,
            env_name,
            ha="right",
            va="top",
            fontsize=FONT_SIZES["tick_label"],
            rotation=45,
            rotation_mode="anchor",
            clip_on=False,
            transform=ax_heatmap.transAxes,
        )

    purple_cmap = plt.get_cmap("Purples")
    bar_positions = np.arange(n_cols) + 0.5
    col_colors = [purple_cmap(0.3 + 0.5 * v) for v in col_means.to_numpy()]
    ax_top.bar(
        bar_positions,
        col_means.values,
        width=1.0,
        color=col_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, 1.0)
    ax_top.set_xticks([])
    for _group_name, _col_start, col_end in group_spans[:-1]:
        ax_top.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    ax_top.set_yticks([0.5, 1])
    ax_top.yaxis.set_minor_locator(plt.NullLocator())
    ax_top.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    ax_top.set_ylabel(
        "Mean score\n(per environment)",
        fontsize=FONT_SIZES["tick_label"],
        rotation=270,
        labelpad=14,
    )
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    bar_positions_y = np.arange(n_rows) + 0.5
    row_colors = [purple_cmap(0.3 + 0.5 * v) for v in row_means.to_numpy()]
    ax_right.barh(
        bar_positions_y,
        row_means.values,
        height=1.0,
        color=row_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_right.set_ylim(n_rows, 0)
    ax_right.set_xlim(0, 1.0)
    ax_right.set_yticks([])
    ax_right.set_xticks([0.5, 1])
    ax_right.xaxis.set_minor_locator(plt.NullLocator())
    ax_right.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
    ax_right.set_xlabel("Mean score\n(per agent)", fontsize=FONT_SIZES["tick_label"])
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_right.axhline(y=i, color="white", linewidth=3, zorder=5)

    for group in model_groups:
        start_frac = group["start"] / n_rows
        end_frac = group["end"] / n_rows
        start_frac_inv = 1.0 - end_frac + 0.02
        end_frac_inv = 1.0 - start_frac - 0.02
        mid_frac = (start_frac_inv + end_frac_inv) / 2.0
        ax_right.text(
            1.15,
            mid_frac,
            group["model"],
            ha="left",
            va="center",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
            clip_on=False,
            transform=ax_right.transAxes,
        )
        ax_right.plot(
            [1.05, 1.05],
            [start_frac_inv, end_frac_inv],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax_right.transAxes,
        )

    for group_name, col_start, col_end in group_spans:
        mid_data = (col_start + col_end) / 2.0 + 0.5
        start_frac = (col_start + 0.15) / n_cols
        end_frac = (col_end + 0.85) / n_cols
        mid_frac = mid_data / n_cols
        ax_top.text(
            mid_frac,
            1.15,
            group_name,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
            clip_on=False,
            transform=ax_top.transAxes,
        )
        ax_top.plot(
            [start_frac, end_frac],
            [1.05, 1.05],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax_top.transAxes,
        )

    fig.subplots_adjust(top=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


# ==================== MAIN ====================


def main(
    verbosity_strategy: str = "average",
    task_type_strategy: str = "both",
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str | None = None,
) -> None:
    """Generate full coverage heatmap showing all tested configurations."""
    logger.info("=" * 60)
    logger.info("Full Coverage Heatmap Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")

    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        msg = f"Invalid metric: {metric}. Must be 'average_score', 'pass_at_k', or 'pass_hat_k'"
        raise ValueError(msg)

    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        msg = f"Invalid k_value: {k_value}. Must be between 1 and 5"
        raise ValueError(msg)

    metric_column = get_metric_column_name(metric, k_value)

    logger.info("Loading datasets...")
    reports_df = load_reports_data()

    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(
        filtered_df, None if verbosity_strategy == "average" else verbosity_strategy
    )
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")

    logger.info("Preparing full coverage data...")
    heatmap_data = collect_full_coverage_data(filtered_df, metric_column)
    logger.info(
        f"Matrix shape: {heatmap_data.shape[0]} configs x {heatmap_data.shape[1]} env-levels"
    )

    n_missing = heatmap_data.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Missing data points: {n_missing}")

    logger.info("Generating full coverage heatmap...")
    if output_filename is None:
        output_path = OUT_DIR / "panel2_coverage_heatmap.pdf"
    else:
        output_path = Path(output_filename)
    plot_full_coverage_heatmap(heatmap_data, output_path)


if __name__ == "__main__":
    fire.Fire(main)
