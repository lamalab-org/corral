"""Combined panel 2 figure.

Layout (3 rows):
  Row 1: (a) Full-coverage heatmap with marginal mean bars  [full width]
  Row 2: (b) Task category performance line plot             [full width]
  Row 3: (c) Gap scatter (left)  |  (d) Log-prob bar (right) [half each]

Usage:
    python plot_panel2_combined.py
"""

from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from plot_config import (
    AGENT_NAMES,
    ENVIRONMENT_GROUPS,
    ENVIRONMENT_MAX_LEVELS,
    ENVIRONMENT_NAMES,
    FONT_SIZES,
    GROUP_COLOURS,
    MODEL_NAMES,
)
from plot_utils import (
    classify_subtask,
    filter_by_level,
    get_metric_column_name,
    load_category_tags,
    load_reports_data,
)

lama_aesthetics.get_style("main")

OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== (a) HEATMAP helpers ====================


def get_ordered_env_level_columns() -> list[str]:
    return [
        f"{env}-{level}"
        for group_info in ENVIRONMENT_GROUPS.values()
        for env in group_info["environments"]
        for level in range(1, ENVIRONMENT_MAX_LEVELS.get(env, 1) + 1)
    ]


def collect_full_coverage_data(df, metric_column):
    data = df.copy()
    data["config"] = data["model"] + "_" + data["agent_type"]
    data["env_level"] = data["environment"] + "-" + data["level"].astype(str)
    return data.pivot_table(
        values=metric_column, index="config", columns="env_level", aggfunc="mean"
    )


def format_heatmap_data(pivot_df):
    display_df = pivot_df.copy()
    ordered_cols = [
        c for c in get_ordered_env_level_columns() if c in display_df.columns
    ]
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


def _get_group_column_spans(n_cols_total):
    spans = []
    col_idx = 0
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        start = col_idx
        for env in group_info["environments"]:
            col_idx += ENVIRONMENT_MAX_LEVELS.get(env, 1)
        if col_idx > start:
            spans.append((group_name, start, col_idx - 1))
    return [(g, s, min(e, n_cols_total - 1)) for g, s, e in spans if s < n_cols_total]


def draw_heatmap(parent_gs):
    """Draw heatmap with marginal bars into the given SubplotSpec."""
    reports_df = load_reports_data()
    metric_column = get_metric_column_name("average_score", 5)
    heatmap_data = collect_full_coverage_data(reports_df, metric_column)

    display_df, row_meta, env_spans = format_heatmap_data(heatmap_data)
    n_cols = len(display_df.columns)
    n_rows = len(display_df.index)
    col_means = display_df.mean(axis=0, skipna=True)
    row_means = display_df.mean(axis=1, skipna=True)

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=parent_gs,
        width_ratios=[1, 0.08],
        height_ratios=[0.12, 1],
        wspace=0.02,
        hspace=0.02,
    )

    fig = plt.gcf()
    ax_top = fig.add_subplot(inner_gs[0, 0])
    ax_heatmap = fig.add_subplot(inner_gs[1, 0])
    ax_right = fig.add_subplot(inner_gs[1, 1])

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

    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_heatmap.axhline(y=i, color="white", linewidth=3, zorder=5)

    group_spans = _get_group_column_spans(n_cols)
    for _gn, _cs, col_end in group_spans[:-1]:
        ax_heatmap.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)

    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel("")
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

    ax_heatmap.set_xlabel("")
    ax_heatmap.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=0, ha="center")

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
            fontsize=FONT_SIZES["tick_label"] - 2,
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
        col_means.to_numpy(),
        width=1.0,
        color=col_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, min(1.0, col_means.max() * 1.3))
    ax_top.set_xticks([])
    for _gn, _cs, col_end in group_spans[:-1]:
        ax_top.axvline(x=col_end + 1, color="white", linewidth=3, zorder=5)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    ax_top.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_top.set_ylabel(
        "Mean score\n(per environment)",
        fontsize=FONT_SIZES["tick_label"] - 1,
        rotation=270,
        labelpad=14,
    )
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    bar_positions_y = np.arange(n_rows) + 0.5
    row_colors = [purple_cmap(0.3 + 0.5 * v) for v in row_means.to_numpy()]
    ax_right.barh(
        bar_positions_y,
        row_means.to_numpy(),
        height=1.0,
        color=row_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_right.set_ylim(n_rows, 0)
    ax_right.set_xlim(0, min(1.0, row_means.max() * 1.3))
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
    ax_right.set_xlabel(
        "Mean score\n(per agent)", fontsize=FONT_SIZES["tick_label"] - 1
    )
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_right.axhline(y=i, color="white", linewidth=3, zorder=5)

    model_groups = []
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"]:
            model_groups.append({"model": meta["model"], "start": i})
            if len(model_groups) > 1:
                model_groups[-2]["end"] = i
    if model_groups:
        model_groups[-1]["end"] = len(row_meta)

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

    return ax_top


# ==================== (b) TASK CATEGORY helpers ====================


def draw_task_category(ax, reports_df, category_tags):
    """Simple bar plot: 4 bars (one per category), averaged across all models & agents."""
    df_sub = reports_df[reports_df["category"] == "subtask"].copy()
    df_comp = df_sub[df_sub["Tool Verbosity"] == "comprehensive"].copy()

    category_order = ["retrieval", "execution", "reasoning", "validation"]
    scores_by_category = defaultdict(list)

    for _, row in df_comp.iterrows():
        environment = row["environment"]
        task_results = row["Task Results"]
        if not isinstance(task_results, dict):
            continue
        for subtask, result in task_results.items():
            if not isinstance(result, dict):
                continue
            pass_at_5 = result.get("Task Pass@5", None)
            if pass_at_5 is None:
                continue
            category = classify_subtask(subtask, environment, category_tags)
            if category is None:
                continue
            if category in ("code_execution", "experiment_execution"):
                category = "execution"
            scores_by_category[category].append(pass_at_5)

    x_values = np.arange(len(category_order))
    y_values = [np.mean(scores_by_category.get(cat, [0])) for cat in category_order]

    ax.bar(
        x_values,
        y_values,
        color="#7150e0",
        width=0.6,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Average Pass@5", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])

    category_labels = [cat.replace("_", " ").title() for cat in category_order]
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        category_labels, fontsize=FONT_SIZES["tick_label"], rotation=45, ha="right"
    )
    range_frame(ax, np.array([0, len(category_order) - 1]), np.array([0, 1]), pad=0.05)


# ==================== (c) SCATTER helpers ====================


def collect_gap_data(df, metric_column):
    gap_data = {}
    for env in df["environment"].unique():
        env_df = df[df["environment"] == env]
        model_scores = env_df.groupby("model")[metric_column].mean()
        agent_scores = env_df.groupby("agent_type")[metric_column].mean()
        model_gap = (
            model_scores.max() - model_scores.min() if len(model_scores) > 1 else np.nan
        )
        agent_gap = (
            agent_scores.max() - agent_scores.min() if len(agent_scores) > 1 else np.nan
        )
        if not np.isnan(model_gap) and not np.isnan(agent_gap):
            gap_data[env] = {"model_gap": model_gap, "agent_gap": agent_gap}
    return gap_data


def draw_scatter(ax, gap_data):
    env_to_group = {}
    for group_name, group_info in ENVIRONMENT_GROUPS.items():
        for env in group_info["environments"]:
            env_to_group[env] = group_name

    environments = list(gap_data.keys())
    agent_gaps = [gap_data[e]["agent_gap"] for e in environments]
    model_gaps = [gap_data[e]["model_gap"] for e in environments]

    max_gap = max(*agent_gaps, *model_gaps)
    min_gap = min(*agent_gaps, *model_gaps)
    diag = [min_gap * 0.95, max_gap * 1.05]

    ax.plot(diag, diag, "k--", linewidth=1.5, alpha=0.5, zorder=1)

    for group_name in ENVIRONMENT_GROUPS:
        group_envs = [e for e in environments if env_to_group.get(e) == group_name]
        if not group_envs:
            continue
        gx = [gap_data[e]["agent_gap"] for e in group_envs]
        gy = [gap_data[e]["model_gap"] for e in group_envs]
        ax.scatter(
            gx,
            gy,
            s=100,
            color=GROUP_COLOURS[group_name],
            alpha=0.7,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
            label=group_name,
        )

    ax.fill_between(
        diag, diag, [max_gap * 1.1] * 2, alpha=0.1, color="#7150e0", zorder=0
    )
    ax.set_xlabel(
        "Scaffold Spread",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.set_ylabel(
        "Model Spread",
        fontsize=FONT_SIZES["axis_label"],
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
    ax.set_aspect("equal", adjustable="box")

    x_range = np.array([min_gap * 0.95, max_gap * 1.05])
    y_range = np.array([min_gap * 0.95, max_gap * 1.05])
    range_frame(ax, x_range, y_range, pad=0.05)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.35, 0.2),
        fontsize=FONT_SIZES["legend"] - 1,
        framealpha=0.0,
        ncol=1,
    )


# ==================== (d) LOGPROBS helpers ====================

LOGPROB_ENV_NAMES = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor",
    "retro": "Retro",
    "spectra": "Spectra",
    "wetlab": "Wetlab",
}


def _pool_nonzero_tokens(series) -> np.ndarray:
    arrays = []
    for lp in series:
        if not isinstance(lp, list | np.ndarray) or len(lp) == 0:
            continue
        arr = np.asarray(lp, dtype=np.float32)
        arr = arr[np.isfinite(arr) & (arr != 0.0)]
        if arr.size > 0:
            arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def compute_env_stats(df):
    rows = []
    for env, grp in df.groupby("environment"):
        tokens = _pool_nonzero_tokens(grp["per_token_logprob"])
        if tokens.size == 0:
            continue
        rows.append(
            {
                "environment": env,
                "display_name": LOGPROB_ENV_NAMES.get(env, env),
                "mean": float(np.mean(tokens)),
                "n_tokens": int(tokens.size),
                "color": "#7150e0",
            }
        )
    return (
        pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    )


def draw_logprobs(ax, stats):
    labels = stats["display_name"].tolist()
    values = stats["mean"].tolist()
    colors = stats["color"].tolist()
    y_pos = np.arange(len(values))

    bars = ax.barh(y_pos, values, color=colors, height=0.6)
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=FONT_SIZES["tick_label"] - 1,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick_label"])
    ax.set_xlabel("Mean log-probability", fontsize=FONT_SIZES["axis_label"])
    range_frame(ax, np.array([min(values), max(values)]), y_pos, pad=0.15)


# ==================== MAIN ====================


def main():
    logger.info("Loading data...")
    reports_df = load_reports_data()
    category_tags = load_category_tags()

    metric_column = get_metric_column_name("average_score", 5)

    # Scatter data (default_map levels, average verbosity)
    scatter_df = filter_by_level(reports_df, "default_map")
    gap_data = collect_gap_data(scatter_df, metric_column)

    # --- Build combined figure ---
    heatmap_data = collect_full_coverage_data(reports_df, metric_column)
    display_df_tmp, _, _ = format_heatmap_data(heatmap_data)
    n_cols_tmp = len(display_df_tmp.columns)
    n_rows_tmp = len(display_df_tmp.index)
    hm_width = min(TWO_COL_WIDTH * 1.5, max(TWO_COL_WIDTH, n_cols_tmp * 0.4))
    hm_height = max(ONE_COL_HEIGHT * 0.6, n_rows_tmp * 0.35)

    fig_width = hm_width
    bottom_height = ONE_COL_HEIGHT * 1.2
    fig = plt.figure(figsize=(fig_width, hm_height + bottom_height + 1.2))

    outer_gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[hm_height, bottom_height],
        width_ratios=[1, 1],
        hspace=0.6,
        wspace=0.55,
    )

    # (a) Heatmap — spans both columns of row 0
    heatmap_slot = outer_gs[0, :]
    ax_top = draw_heatmap(heatmap_slot)

    # (b) Task category bar — row 1, left
    ax_category = fig.add_subplot(outer_gs[1, 0])
    draw_task_category(ax_category, reports_df, category_tags)

    # (c) Scatter — row 1, right
    ax_scatter = fig.add_subplot(outer_gs[1, 1])
    draw_scatter(ax_scatter, gap_data)

    # Panel labels
    for ax, label in zip(
        [ax_top, ax_category, ax_scatter],
        ["a", "b", "c"],
        strict=False,
    ):
        ax.text(
            -0.05,
            1.15,
            label,
            transform=ax.transAxes,
            fontsize=FONT_SIZES["title"] + 2,
            fontweight="bold",
            va="top",
        )

    # Save
    output_path = OUT_DIR / "panel2_combined.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
