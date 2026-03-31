"""Full coverage heatmap showing all model x agent configurations across all environment-level combinations.

Creates a heatmap where rows are model x agent configurations (6 total) and columns are
all environment-level combinations tested (e.g., AFM-1, AFM-2, Catalyst-1, MD-1, MD-2, etc.).
This visualization shows the complete experimental scope and performance across all conditions.

Usage:
    python 9_full_coverage_heatmap.py --verbosity_strategy=average --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import lama_aesthetics
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger

lama_aesthetics.get_style("main")

# Add analysis and plot config to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import (  # noqa: E402
    AGENT_NAMES,
    ENVIRONMENT_GROUPS,
    ENVIRONMENT_MAX_LEVELS,
    ENVIRONMENT_NAMES,
    FONT_SIZES,
    MODEL_NAMES,
)
from plot_utils import (  # noqa: E402
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

# ==================== DATA COLLECTION ====================


def collect_full_coverage_data(
    df: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Collect data for full coverage heatmap.

    Args:
        df: Filtered benchmark reports dataframe
        metric_column: Column name of the metric to plot

    Returns:
        DataFrame with model x agent configs as rows, environment-level combos as columns
    """
    # Create model x agent configuration column
    data = df.copy()
    data["config"] = data["model"] + "_" + data["agent_type"]

    # Create environment-level column
    data["env_level"] = data["environment"] + "-" + data["level"].astype(str)

    # Pivot to create config x env_level matrix
    return data.pivot_table(
        values=metric_column,
        index="config",
        columns="env_level",
        aggfunc="mean",
    )


def get_ordered_env_level_columns() -> list[str]:
    """Return env-level column keys ordered by environment group then increasing level.

    Order follows ENVIRONMENT_GROUPS (Hypothesis-driven, Strategic, Workflow construction),
    and within each environment, levels go S1, S2, S3, ...
    """
    return [
        f"{env}-{level}"
        for group_info in ENVIRONMENT_GROUPS.values()
        for env in group_info["environments"]
        for level in range(1, ENVIRONMENT_MAX_LEVELS.get(env, 1) + 1)
    ]


def format_heatmap_data(pivot_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Format and reorder heatmap data. No gap rows — gaps are drawn visually.

    Returns:
        (display_df, row_meta) where row_meta has per-row info for label rendering.
        Each entry in row_meta: {"model": str, "agent": str, "is_first_of_model": bool}
    """
    display_df = pivot_df.copy()

    # Reorder columns by environment group, then increasing level
    ordered_cols = get_ordered_env_level_columns()
    ordered_cols = [c for c in ordered_cols if c in display_df.columns]
    remaining = [c for c in display_df.columns if c not in ordered_cols]
    display_df = display_df[ordered_cols + remaining]

    # Parse and sort rows by model then agent
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

    # Build ordered dataframe and row metadata
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

    # Format column labels (environment-level -> ENV S#)
    col_labels = []
    for env_level in display_df.columns:
        parts = env_level.split("-")
        if len(parts) >= 2:
            env = parts[0]
            level = parts[1]
            env_name = ENVIRONMENT_NAMES.get(env, env.upper())
            col_labels.append(f"{env_name} S{level}")
        else:
            col_labels.append(env_level)
    display_df.columns = col_labels

    return display_df, row_meta


# ==================== PLOTTING ====================


def _get_group_column_spans(display_df: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Compute (group_name, col_start, col_end) spans for the ordered columns."""
    spans = []
    col_idx = 0
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        start = col_idx
        for env in group_info["environments"]:
            max_level = ENVIRONMENT_MAX_LEVELS.get(env, 1)
            for _lvl in range(1, max_level + 1):
                env_name = ENVIRONMENT_NAMES.get(env, env.upper())
                col_label = f"{env_name} S{_lvl}"
                if col_label in display_df.columns:
                    col_idx += 1
        if col_idx > start:
            spans.append((group_name, start, col_idx - 1))
    return spans


def plot_full_coverage_heatmap(
    heatmap_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create full coverage heatmap with marginal mean bar charts.

    Layout: top bar chart (column means), main heatmap, right bar chart (row means).
    Y-axis: model name (bold) above its agent rows, agent names to the left.
    Thin white lines separate model groups.
    X-axis: environment labels at bottom (rotated 90°),
    group labels with lines ABOVE the top bar chart.
    """
    if heatmap_data.empty:
        logger.warning("No data available for heatmap!")
        return

    display_df, row_meta = format_heatmap_data(heatmap_data)

    n_cols = len(display_df.columns)
    n_rows = len(display_df.index)

    # Compute marginal means
    col_means = display_df.mean(axis=0, skipna=True)
    row_means = display_df.mean(axis=1, skipna=True)

    fig_width = min(TWO_COL_WIDTH * 1.5, max(TWO_COL_WIDTH, n_cols * 0.4))
    fig_height = max(ONE_COL_HEIGHT * 0.6, n_rows * 0.35)

    # Add space for marginal plots (use absolute size so both bars are same dimension)
    bar_size = 0.6  # inches for both marginal bar plots
    fig_width_total = fig_width + bar_size + 0.1
    fig_height_total = fig_height + bar_size + 0.1
    bar_ratio_w = bar_size / fig_width
    bar_ratio_h = bar_size / fig_height

    fig = plt.figure(figsize=(fig_width_total, fig_height_total))
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
    # Leave gs[0,1] empty (corner)

    # --- Main heatmap ---
    sns.heatmap(
        display_df,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        cbar=False,
        ax=ax_heatmap,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"] - 3},
    )

    # --- Thin white separator lines between model groups (horizontal) ---
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_heatmap.axhline(y=i, color="white", linewidth=3, zorder=5)

    # --- Thick white separator lines between environment groups (vertical) ---
    group_spans = _get_group_column_spans(display_df)
    for _group_name, _col_start, col_end in group_spans[:-1]:  # skip last group
        ax_heatmap.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)

    # --- Hierarchical y-axis ---
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel("")

    # Find the row span for each model group
    model_groups = []
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"]:
            model_groups.append({"model": meta["model"], "start": i})
            if len(model_groups) > 1:
                model_groups[-2]["end"] = i
    if model_groups:
        model_groups[-1]["end"] = len(row_meta)

    # Agent labels for each row (indented to make room for model name)
    for i, meta in enumerate(row_meta):
        ax_heatmap.text(
            -0.01,
            i + 0.5,
            meta["agent"],
            ha="right",
            va="center",
            fontsize=FONT_SIZES["tick_label"] - 1,
            transform=ax_heatmap.get_yaxis_transform(),
        )

    # Model name labels are on the right bar chart (see below)

    # --- X-axis: env labels at bottom ---
    ax_heatmap.set_xlabel("")
    ax_heatmap.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, ha="center")

    # --- Top bar chart (column means) ---
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
    ax_top.set_ylim(0, min(1.0, col_means.max() * 1.3))
    ax_top.set_xticks([])
    # Vertical separators matching heatmap groups
    for _group_name, _col_start, col_end in group_spans[:-1]:
        ax_top.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    ax_top.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_top.set_ylabel(
        "Mean", fontsize=FONT_SIZES["tick_label"] - 1, rotation=270, labelpad=10
    )
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # --- Right bar chart (row means) ---
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
    ax_right.set_ylim(n_rows, 0)  # Invert to match heatmap orientation
    ax_right.set_xlim(0, min(1.0, row_means.max() * 1.3))
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_right.set_xlabel("Mean", fontsize=FONT_SIZES["tick_label"] - 1)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # Add separator lines in right bar chart matching model groups
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_right.axhline(y=i, color="white", linewidth=3, zorder=5)

    # Model group labels to the right of the right bar chart (mirroring top group labels)
    for group in model_groups:
        start_frac = group["start"] / n_rows
        end_frac = group["end"] / n_rows
        # Invert fractions since y-axis is inverted
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

    # --- Group labels ABOVE the top bar chart ---
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
    """Generate full coverage heatmap showing all tested configurations.

    Args:
        verbosity_strategy: Which tool verbosity to use.
            Options: "average", "brief", "workflow", "comprehensive"
        task_type_strategy: Which task type to use.
            Options: "tasks" (category=task), "subtasks" (category=subtask), "both"
        metric: Metric to plot.
            Options: "average_score", "pass_at_k", "pass_hat_k"
        k_value: K value for Pass@k or Pass^k metrics (1-5)
        output_filename: Output filename (PDF)
    """
    logger.info("=" * 60)
    logger.info("Full Coverage Heatmap Generation")
    logger.info("=" * 60)
    logger.info(f"Verbosity strategy: {verbosity_strategy}")
    logger.info(f"Task type strategy: {task_type_strategy}")
    logger.info(f"Metric: {metric}")
    if metric in ["pass_at_k", "pass_hat_k"]:
        logger.info(f"K value: {k_value}")
    logger.info("")

    # Validate inputs
    if metric not in ["average_score", "pass_at_k", "pass_hat_k"]:
        msg = f"Invalid metric: {metric}. Must be 'average_score', 'pass_at_k', or 'pass_hat_k'"
        raise ValueError(msg)

    if metric in ["pass_at_k", "pass_hat_k"] and not 1 <= k_value <= 5:
        msg = f"Invalid k_value: {k_value}. Must be between 1 and 5"
        raise ValueError(msg)

    # Get metric column name
    metric_column = get_metric_column_name(metric, k_value)

    # Load data
    logger.info("Loading datasets...")
    reports_df = load_reports_data()

    # Filter reports data (no level filtering - we want all levels)
    logger.info("Filtering benchmark reports...")
    filtered_df = reports_df.copy()
    filtered_df = filter_by_verbosity(
        filtered_df, None if verbosity_strategy == "average" else verbosity_strategy
    )
    filtered_df = filter_by_task_type(filtered_df, task_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows")
    logger.info("")

    # Collect heatmap data
    logger.info("Preparing full coverage data...")
    heatmap_data = collect_full_coverage_data(filtered_df, metric_column)
    logger.info(
        f"Matrix shape: {heatmap_data.shape[0]} configs x {heatmap_data.shape[1]} env-levels"
    )

    # Print coverage summary
    logger.info("")
    logger.info("Experimental Coverage:")
    logger.info("-" * 40)
    logger.info(f"Configurations tested: {heatmap_data.shape[0]}")
    logger.info(f"Environment-level combinations: {heatmap_data.shape[1]}")
    logger.info(
        f"Total experimental conditions: {heatmap_data.shape[0] * heatmap_data.shape[1]}"
    )

    # Check for missing data
    n_missing = heatmap_data.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Missing data points: {n_missing}")

    # Generate plot
    logger.info("")
    logger.info("Generating full coverage heatmap...")
    if output_filename is None:
        output_path = Path(__file__).parent / "2a_full_coverage_heatmap.pdf"
    else:
        output_path = Path(output_filename)
    plot_full_coverage_heatmap(heatmap_data, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Full coverage heatmap generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
