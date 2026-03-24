"""Combined panel figure: heatmap (top), scatter (bottom-left 1/3), task category (bottom-right 2/3).

Usage:
    python 2_panel.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from lama_aesthetics import TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from matplotlib.lines import Line2D

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
    GAP_COLORS,
    MODEL_COLOURS,
    MODEL_NAMES,
)
from plot_utils import (  # noqa: E402
    filter_by_level,
    get_metric_column_name,
    load_reports_data,
)

SCATTER_COLOR = GAP_COLORS["model_gap"]

PANEL_DIR = Path(__file__).resolve().parent


# ==================== HEATMAP helpers ====================


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
    """Format heatmap data with gap rows between model groups."""
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
        if prev_model is not None and model != prev_model:
            gap = pd.Series(
                float("nan"), index=display_df.columns, name=f"__gap_{model}"
            )
            new_rows.append(gap)
            row_meta.append({"model": "", "agent": "", "is_gap": True})
        new_rows.append(display_df.loc[config])
        row_meta.append(
            {
                "model": MODEL_NAMES.get(model, model),
                "agent": AGENT_NAMES.get(agent, agent),
                "is_gap": False,
            }
        )
        prev_model = model

    display_df = pd.DataFrame(new_rows)

    col_labels = []
    for env_level in display_df.columns:
        parts = env_level.split("-")
        env = parts[0]
        level = parts[1] if len(parts) >= 2 else ""
        env_name = ENVIRONMENT_NAMES.get(env, env.upper())
        col_labels.append(f"{env_name} S{level}")
    display_df.columns = col_labels
    return display_df, row_meta


def _get_group_column_spans(display_df):
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


def draw_heatmap(ax, heatmap_data, metric_display_name):
    display_df, row_meta = format_heatmap_data(heatmap_data)

    mask = pd.DataFrame(False, index=display_df.index, columns=display_df.columns)
    for i, meta in enumerate(row_meta):
        if meta["is_gap"]:
            mask.iloc[i] = True

    sns.heatmap(
        display_df,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        cbar_kws={"label": metric_display_name, "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        annot_kws={"fontsize": FONT_SIZES["tick_label"] - 3},
    )

    # Hierarchical y-axis
    ax.set_yticks([])
    ax.set_ylabel("")
    for i, meta in enumerate(row_meta):
        if not meta["is_gap"]:
            ax.text(
                -0.02,
                i + 0.5,
                f"  {meta['agent']}",
                ha="right",
                va="center",
                fontsize=FONT_SIZES["tick_label"] - 1,
                transform=ax.get_yaxis_transform(),
            )

    prev_model = None
    model_start = 0
    for i, meta in enumerate(row_meta):
        if meta["is_gap"]:
            continue
        if meta["model"] != prev_model:
            if prev_model is not None:
                data_rows = [
                    j for j in range(model_start, i) if not row_meta[j]["is_gap"]
                ]
                if data_rows:
                    mid_y = (data_rows[0] + data_rows[-1]) / 2.0 + 0.5
                    ax.text(
                        -0.15,
                        mid_y,
                        prev_model,
                        ha="right",
                        va="center",
                        fontsize=FONT_SIZES["tick_label"],
                        fontweight="bold",
                        transform=ax.get_yaxis_transform(),
                    )
            model_start = i
            prev_model = meta["model"]
    if prev_model is not None:
        data_rows = [
            j for j in range(model_start, len(row_meta)) if not row_meta[j]["is_gap"]
        ]
        if data_rows:
            mid_y = (data_rows[0] + data_rows[-1]) / 2.0 + 0.5
            ax.text(
                -0.15,
                mid_y,
                prev_model,
                ha="right",
                va="center",
                fontsize=FONT_SIZES["tick_label"],
                fontweight="bold",
                transform=ax.get_yaxis_transform(),
            )

    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"] - 2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Group labels ABOVE the heatmap
    n_cols = len(display_df.columns)
    group_spans = _get_group_column_spans(display_df)
    for group_name, col_start, col_end in group_spans:
        mid_data = (col_start + col_end) / 2.0 + 0.5
        start_frac = (col_start + 0.15) / n_cols
        end_frac = (col_end + 0.85) / n_cols
        mid_frac = mid_data / n_cols
        ax.text(
            mid_frac,
            1.06,
            group_name,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["tick_label"] - 1,
            fontweight="bold",
            clip_on=False,
            transform=ax.transAxes,
        )
        ax.plot(
            [start_frac, end_frac],
            [1.02, 1.02],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax.transAxes,
        )


# ==================== SCATTER helpers ====================


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


def draw_scatter(ax, gap_data, metric_display_name):
    environments = list(gap_data.keys())
    agent_gaps = [gap_data[e]["agent_gap"] for e in environments]
    model_gaps = [gap_data[e]["model_gap"] for e in environments]
    env_labels = [ENVIRONMENT_NAMES.get(e, e.upper()) for e in environments]

    max_gap = max(*agent_gaps, *model_gaps)
    min_gap = min(*agent_gaps, *model_gaps)
    diag = [min_gap * 0.95, max_gap * 1.05]

    ax.plot(diag, diag, "k--", linewidth=1.5, alpha=0.5, zorder=1)
    ax.scatter(
        agent_gaps,
        model_gaps,
        s=100,
        color=SCATTER_COLOR,
        alpha=0.7,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    texts = []
    for x, y, label in zip(agent_gaps, model_gaps, env_labels, strict=False):
        texts.append(ax.text(x, y, label, fontsize=4.5, color="black", alpha=1))
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.5, "alpha": 1},
        expand_points=(1.5, 1.5),
        force_text=(0.5, 0.5),
    )

    ax.fill_between(
        diag, diag, [max_gap * 1.1] * 2, alpha=0.1, color=SCATTER_COLOR, zorder=0
    )
    ax.set_xlabel(
        f"Agent Gap ({metric_display_name})", fontsize=FONT_SIZES["axis_label"]
    )
    ax.set_ylabel(
        f"Model Gap ({metric_display_name})", fontsize=FONT_SIZES["axis_label"]
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])
    ax.set_aspect("equal", adjustable="box")
    x_range = np.array([min_gap * 0.95, max_gap * 1.05])
    y_range = np.array([min_gap * 0.95, max_gap * 1.05])
    range_frame(ax, x_range, y_range, pad=0.05)


# ==================== TASK CATEGORY helpers ====================


def load_category_tags():
    tags_path = REPO_ROOT / "analysis" / "subtask_category_tags.json"
    with tags_path.open() as f:
        return json.load(f)


def get_env_key_mapping():
    return {
        "spectra": "sptectra",
        "retro": "retrosynthesis",
        "afm": "afm",
        "catalyst": "catalyst",
        "md": "md",
        "ml": "ml",
        "resistor": "resistor",
    }


def classify_subtask(subtask, environment, category_tags):
    env_mapping = get_env_key_mapping()
    tag_env_key = env_mapping.get(environment, environment)
    env_tags = category_tags.get(tag_env_key, {})
    if not env_tags:
        return None

    if environment == "afm":
        if "subtask_level_" in subtask:
            base_name = subtask.split("_level_")[0]
            if base_name + "_level_1" in env_tags:
                return env_tags[base_name + "_level_1"]
        return env_tags.get(subtask)

    if environment == "catalyst":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[1] in env_tags:
            return env_tags[parts[1]]
        return None

    if environment == "md":
        for task_type in ["melting", "quenching", "surface_energy", "surface"]:
            if task_type in subtask:
                task_tags = env_tags.get(task_type, {})
                if "subtask_" in subtask:
                    subtask_name = subtask.split("subtask_")[-1]
                    if subtask_name == "diffusion_coefficient":
                        subtask_name = "diffusivity"
                    elif subtask_name == "tg_calculation":
                        subtask_name = "tg_detection"
                    elif subtask_name == "equilibration":
                        subtask_name = "structure_retrieval"
                    if subtask_name in task_tags:
                        return task_tags[subtask_name]
        return None

    if environment == "ml":
        parts = subtask.split("_", 1)
        if len(parts) > 1 and parts[0].isdigit():
            task_name = parts[1]
            if (
                task_name.startswith("batch_retrieve_")
                and "batch_retrieve_*" in env_tags
            ):
                return env_tags["batch_retrieve_*"]
            if task_name in env_tags:
                return env_tags[task_name]
        return None

    if environment == "resistor":
        parts = subtask.split("_", 2)
        if len(parts) >= 3 and parts[0] == "task" and parts[1].isdigit():
            pattern = "_".join(parts[2:])
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "retro":
        if "-" in subtask:
            pattern = subtask.split("-", 1)[1]
            if pattern in env_tags:
                return env_tags[pattern]
        return None

    if environment == "spectra":
        if "_subtask_" in subtask:
            subtask_num = "subtask_" + subtask.split("_subtask_")[-1]
            if subtask_num in env_tags:
                return env_tags[subtask_num]
        return None

    return None


def draw_task_category(ax, reports_df, category_tags):
    df_sub = reports_df[reports_df["category"] == "subtask"].copy()
    df_comp = df_sub[df_sub["Tool Verbosity"] == "comprehensive"].copy()

    category_order = ["retrieval", "execution", "reasoning", "validation"]

    scores_by_group = defaultdict(lambda: defaultdict(list))
    for _, row in df_comp.iterrows():
        model = row["model"]
        agent_type = row["agent_type"]
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
            scores_by_group[(model, agent_type)][category].append(pass_at_5)

    avg_scores = {}
    for (model, agent_type), cat_scores in scores_by_group.items():
        avg_scores[(model, agent_type)] = {
            cat: np.mean(scores) if scores else np.nan
            for cat, scores in cat_scores.items()
        }

    agent_display = dict(AGENT_NAMES)
    agent_markers = {"react": "o", "tool_calling": "D"}
    agent_linestyle = {"react": "-", "tool_calling": "--"}
    x_values = np.arange(len(category_order))

    for (model, agent_type), category_avgs in sorted(avg_scores.items()):
        color = MODEL_COLOURS.get(model, "gray")
        marker = agent_markers.get(agent_type, "o")
        linestyle = agent_linestyle.get(agent_type, "-")
        y_values = [category_avgs.get(cat, np.nan) for cat in category_order]
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6,
            alpha=0.8,
            fillstyle="none",
            linewidth=2,
        )

    ax.set_ylabel("Average Pass@5", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlabel("Task Category", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])

    category_labels = [cat.replace("_", "\n").title() for cat in category_order]
    ax.set_xticks(x_values)
    ax.set_xticklabels(category_labels, fontsize=FONT_SIZES["tick_label"])
    range_frame(ax, np.array([0, len(category_order) - 1]), np.array([0, 1]), pad=0.05)

    # Legend
    handles, labels = [], []
    for model in sorted({m for m, _ in avg_scores}):
        handles.append(Line2D([0], [0], color=MODEL_COLOURS.get(model, "gray"), lw=4))
        labels.append(MODEL_NAMES.get(model, model))
    for agent_type in sorted({a for _, a in avg_scores}):
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker=agent_markers.get(agent_type, "o"),
                markersize=8,
                linestyle="None",
                fillstyle="none",
            )
        )
        labels.append(f"{agent_display.get(agent_type, agent_type)} Agent")
    ax.legend(handles, labels, fontsize=FONT_SIZES["legend"], loc="upper right")


# ==================== MAIN ====================


def main():
    logger.info("Loading data...")
    reports_df = load_reports_data()
    category_tags = load_category_tags()

    metric_column = get_metric_column_name("average_score", 5)
    metric_display_name = "Average Score"

    # Heatmap data (all levels, average verbosity)
    heatmap_data = collect_full_coverage_data(reports_df, metric_column)

    # Scatter data (default_map levels, average verbosity)
    scatter_df = filter_by_level(reports_df, "default_map")
    gap_data = collect_gap_data(scatter_df, metric_column)

    # Build combined figure
    fig = plt.figure(figsize=(TWO_COL_WIDTH * 1.4, TWO_COL_WIDTH * 1.1))

    # GridSpec: 2 rows, 3 columns
    # Top row: heatmap spans all 3 columns
    # Bottom row: scatter takes 1 col, task category takes 2 cols
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.55, wspace=0.35)

    ax_heatmap = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_category = fig.add_subplot(gs[1, 1:])

    # Draw subplots
    draw_heatmap(ax_heatmap, heatmap_data, metric_display_name)
    draw_scatter(ax_scatter, gap_data, "Score")
    draw_task_category(ax_category, reports_df, category_tags)

    # Panel labels
    for ax, label in zip(
        [ax_heatmap, ax_scatter, ax_category], ["a", "b", "c"], strict=False
    ):
        ax.text(
            -0.05,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=FONT_SIZES["title"] + 2,
            fontweight="bold",
            va="top",
        )

    # Save
    output_path = PANEL_DIR / "2_panel.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
