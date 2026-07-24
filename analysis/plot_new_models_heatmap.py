"""Heatmap of Pass@5 scores for new benchmark models across environments and levels.

Reads summary JSON files directly from reports/<model>/<env>/level_N/<agent>/
and plots a heatmap with rows = model × agent and columns = environment × level.

Usage:
    python plot_new_models_heatmap.py
    python plot_new_models_heatmap.py --metric=pass_at_5
    python plot_new_models_heatmap.py --metric=average_score --output=my_heatmap.pdf
"""

import json
import re
from pathlib import Path

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

try:
    import lama_aesthetics
    from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH

    lama_aesthetics.get_style("main")
except ImportError:
    TWO_COL_WIDTH, TWO_COL_HEIGHT = 7.0, 4.0

from plot_config import ENVIRONMENT_NAMES, FONT_SIZES

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "new_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── identity maps ──────────────────────────────────────────────────────────────

MODEL_NAMES = {
    "deepseek-v3.2": "DeepSeek-V3.2",
    "kimi-k2.5": "Kimi-K2.5",
    "claude-opus-4.8": "Claude-Opus-4.8",
}

MODEL_COLOURS = {
    "deepseek-v3.2": "#0891b2",
    "kimi-k2.5": "#16a34a",
    "claude-opus-4.8": "#7c3aed",
}

AGENT_NAMES = {
    "react": "ReAct",
    "toolcalling": "ToolCalling",
}

ENV_DIR_TO_KEY = {
    "resistor_network": "resistor",
    "retrosynthesis": "retro",
    "spectra_elucidation": "spectra",
    "wetlab": "wetlab",
}

# Ordered for columns: env groups kept together, levels increasing
ENV_ORDER = ["resistor", "retro", "spectra", "wetlab"]

ENV_GROUP_SPANS = {
    "Circuit\nInference": ["resistor"],
    "Retrosynthetic\nPlanning": ["retro"],
    "Spectroscopic\nElucidation": ["spectra"],
    "Inorganic\nAnalysis": ["wetlab"],
}

METRIC_KEY_MAP = {
    "pass_at_5": "Pass@5",
    "pass_at_1": "Pass@1",
    "average_score": "Average Score",
}

# ── data loading ───────────────────────────────────────────────────────────────

_TASK_PATTERN = re.compile(r"(make|task)_\d+")


def _pick_summary_json(agent_dir: Path) -> Path | None:
    """Return the aggregate summary JSON from an agent directory.

    Prefers files without task-specific identifiers (make_N / task_N).
    Among remaining candidates picks the one whose name contains 'ALLTASKS'
    or else takes the lexicographically first file.
    """
    candidates = [
        f
        for f in agent_dir.glob("*.json")
        if not _TASK_PATTERN.search(f.stem)
    ]
    if not candidates:
        return None
    alltasks = [f for f in candidates if "ALLTASKS" in f.stem]
    return alltasks[0] if alltasks else sorted(candidates)[0]


def load_reports(metric_key: str = "Pass@5") -> pd.DataFrame:
    """Scan reports/ and return a tidy DataFrame with one row per run."""
    rows = []
    for model_dir in sorted(REPORTS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        model = model_dir.name
        if model not in MODEL_NAMES:
            continue
        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir() or env_dir.name.startswith("_"):
                continue
            env_key = ENV_DIR_TO_KEY.get(env_dir.name)
            if env_key is None:
                continue
            for level_dir in sorted(env_dir.iterdir()):
                if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
                    continue
                level = int(level_dir.name.split("_")[1])
                for agent_dir in sorted(level_dir.iterdir()):
                    if not agent_dir.is_dir():
                        continue
                    agent = agent_dir.name
                    if agent not in AGENT_NAMES:
                        continue
                    summary = _pick_summary_json(agent_dir)
                    if summary is None:
                        logger.warning(f"No summary JSON in {agent_dir}")
                        continue
                    with summary.open() as fh:
                        data = json.load(fh)
                    metrics = data.get("metrics", {})
                    value = metrics.get(metric_key)
                    if value is None:
                        logger.warning(f"Missing '{metric_key}' in {summary}")
                        continue
                    rows.append(
                        {
                            "model": model,
                            "agent": agent,
                            "env": env_key,
                            "level": level,
                            "score": float(value),
                        }
                    )
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} data points")
    return df


# ── pivot & ordering ───────────────────────────────────────────────────────────


def build_pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[tuple]]:
    """Build pivot table and return (pivot_df, row_meta, col_spans)."""
    df = df.copy()
    df["col"] = df["env"] + "-L" + df["level"].astype(str)
    df["row"] = df["model"] + "__" + df["agent"]

    pivot = df.pivot_table(values="score", index="row", columns="col", aggfunc="mean")

    # ── order columns ──
    ordered_cols = []
    col_spans: list[tuple[str, int, int]] = []  # (label, start, end)
    col_idx = 0
    for env in ENV_ORDER:
        env_cols = sorted(
            [c for c in pivot.columns if c.startswith(env + "-")],
            key=lambda c: int(c.split("-L")[1]),
        )
        if not env_cols:
            continue
        span_start = col_idx
        ordered_cols.extend(env_cols)
        col_idx += len(env_cols)
        env_name = ENVIRONMENT_NAMES.get(env, env)
        col_spans.append((env_name, span_start, col_idx - 1))
    remaining = [c for c in pivot.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)
    pivot = pivot[ordered_cols]

    # rename columns to "S<level>"
    pivot.columns = [c.split("-L")[1] for c in pivot.columns]

    # ── order rows ──
    model_order = list(MODEL_NAMES.keys())
    agent_order = list(AGENT_NAMES.keys())

    def sort_key(row_id):
        model, agent = row_id.split("__")
        return (
            model_order.index(model) if model in model_order else 99,
            agent_order.index(agent) if agent in agent_order else 99,
        )

    sorted_rows = sorted(pivot.index, key=sort_key)
    pivot = pivot.loc[sorted_rows]

    row_meta = []
    prev_model = None
    for row_id in sorted_rows:
        model, agent = row_id.split("__")
        row_meta.append(
            {
                "model": MODEL_NAMES.get(model, model),
                "agent": AGENT_NAMES.get(agent, agent),
                "model_key": model,
                "is_first_of_model": model != prev_model,
            }
        )
        prev_model = model

    return pivot, row_meta, col_spans


# ── plotting ───────────────────────────────────────────────────────────────────


def plot_heatmap(
    pivot: pd.DataFrame,
    row_meta: list[dict],
    col_spans: list[tuple],
    metric_label: str,
    output_path: Path,
) -> None:
    if pivot.empty:
        logger.error("Empty pivot table — nothing to plot")
        return

    # multiply by 100 for % display
    display = pivot * 100

    n_rows, n_cols = display.shape
    col_means = display.mean(axis=0, skipna=True)
    row_means = display.mean(axis=1, skipna=True)

    bar_ratio_w = 0.8 / TWO_COL_WIDTH
    bar_ratio_h = 0.8 / TWO_COL_HEIGHT

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
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # ── heatmap ──
    sns.heatmap(
        display,
        annot=True,
        fmt=".0f",
        cmap="Purples",
        vmin=0,
        vmax=100,
        cbar=False,
        ax=ax_heat,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": FONT_SIZES["tick_label"]},
    )

    # dividers between models
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_heat.axhline(y=i, color="white", linewidth=2.5, zorder=5)

    # dividers between env groups
    for _, _, col_end in col_spans[:-1]:
        ax_heat.axvline(x=col_end + 1, color="white", linewidth=2.5, zorder=5)

    # row labels: agent name on the left
    ax_heat.set_yticks([])
    ax_heat.set_ylabel("")
    for i, meta in enumerate(row_meta):
        ax_heat.text(
            -0.01,
            i + 0.5,
            meta["agent"],
            ha="right",
            va="center",
            fontsize=FONT_SIZES["tick_label"],
            transform=ax_heat.get_yaxis_transform(),
        )

    # x-axis: level numbers
    ax_heat.set_xlabel("")
    ax_heat.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=0, ha="center")

    # env-group underlines below heatmap
    for env_name, col_start, col_end in col_spans:
        start_frac = (col_start + 0.1) / n_cols
        end_frac = (col_end + 0.9) / n_cols
        mid_frac = (start_frac + end_frac) / 2
        ax_heat.plot(
            [start_frac, end_frac],
            [-0.01, -0.01],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax_heat.transAxes,
        )
        ax_heat.text(
            mid_frac,
            -0.06,
            env_name,
            ha="center",
            va="top",
            fontsize=FONT_SIZES["tick_label"] - 1,
            rotation=0,
            clip_on=False,
            transform=ax_heat.transAxes,
        )

    # ── top bar: mean per column ──
    purple = plt.get_cmap("Purples")
    bar_positions = np.arange(n_cols) + 0.5
    col_colors = [purple(0.3 + 0.5 * v / 100) for v in col_means.to_numpy()]
    ax_top.bar(
        bar_positions,
        col_means.values,
        width=1.0,
        color=col_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_top.set_xlim(0, n_cols)
    ax_top.set_ylim(0, 100)
    ax_top.set_xticks([])
    for _, _, col_end in col_spans[:-1]:
        ax_top.axvline(x=col_end + 1, color="white", linewidth=2.5, zorder=5)
    ax_top.yaxis.tick_right()
    ax_top.yaxis.set_label_position("right")
    ax_top.set_yticks([50, 100])
    ax_top.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])
    ax_top.set_ylabel(
        f"Mean {metric_label}\n(per level)",
        fontsize=FONT_SIZES["tick_label"],
        rotation=270,
        labelpad=14,
    )
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # env-group labels above top bar
    for env_name, col_start, col_end in col_spans:
        start_frac = (col_start + 0.1) / n_cols
        end_frac = (col_end + 0.9) / n_cols
        mid_frac = (start_frac + end_frac) / 2
        ax_top.text(
            mid_frac,
            1.18,
            env_name,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
            clip_on=False,
            transform=ax_top.transAxes,
        )
        ax_top.plot(
            [start_frac, end_frac],
            [1.08, 1.08],
            color="gray",
            linewidth=0.8,
            clip_on=False,
            transform=ax_top.transAxes,
        )

    # ── right bar: mean per row ──
    bar_pos_y = np.arange(n_rows) + 0.5
    row_colors = [purple(0.3 + 0.5 * v / 100) for v in row_means.to_numpy()]
    ax_right.barh(
        bar_pos_y,
        row_means.values,
        height=1.0,
        color=row_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_right.set_ylim(n_rows, 0)
    ax_right.set_xlim(0, 100)
    ax_right.set_yticks([])
    ax_right.set_xticks([50, 100])
    ax_right.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])
    ax_right.set_xlabel(
        f"Mean {metric_label}\n(per config)",
        fontsize=FONT_SIZES["tick_label"],
    )
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"] and i > 0:
            ax_right.axhline(y=i, color="white", linewidth=2.5, zorder=5)

    # model labels on right side
    model_groups: list[dict] = []
    for i, meta in enumerate(row_meta):
        if meta["is_first_of_model"]:
            model_groups.append({"model": meta["model"], "key": meta["model_key"], "start": i})
            if len(model_groups) > 1:
                model_groups[-2]["end"] = i
    if model_groups:
        model_groups[-1]["end"] = n_rows

    for grp in model_groups:
        start_frac_inv = 1.0 - grp["end"] / n_rows + 0.02
        end_frac_inv = 1.0 - grp["start"] / n_rows - 0.02
        mid = (start_frac_inv + end_frac_inv) / 2
        color = MODEL_COLOURS.get(grp["key"], "gray")
        ax_right.text(
            1.15,
            mid,
            grp["model"],
            ha="left",
            va="center",
            fontsize=FONT_SIZES["tick_label"],
            fontweight="bold",
            color=color,
            clip_on=False,
            transform=ax_right.transAxes,
        )
        ax_right.plot(
            [1.05, 1.05],
            [start_frac_inv, end_frac_inv],
            color=color,
            linewidth=1.2,
            clip_on=False,
            transform=ax_right.transAxes,
        )

    fig.subplots_adjust(top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main(
    metric: str = "pass_at_5",
    output: str | None = None,
) -> None:
    """Generate heatmap for new benchmark models.

    Args:
        metric: One of 'pass_at_5', 'pass_at_1', 'average_score'.
        output: Override output path (PDF).
    """
    if metric not in METRIC_KEY_MAP:
        raise ValueError(f"metric must be one of {list(METRIC_KEY_MAP)}")

    metric_key = METRIC_KEY_MAP[metric]
    metric_label = metric_key

    logger.info(f"Loading reports with metric={metric_key}")
    df = load_reports(metric_key)

    logger.info("Building pivot table")
    pivot, row_meta, col_spans = build_pivot(df)
    logger.info(f"Pivot shape: {pivot.shape}  ({pivot.shape[0]} configs × {pivot.shape[1]} levels)")

    out = Path(output) if output else OUT_DIR / f"new_models_heatmap_{metric}.pdf"
    logger.info(f"Plotting → {out}")
    plot_heatmap(pivot, row_meta, col_spans, metric_label, out)


if __name__ == "__main__":
    fire.Fire(main)
