"""Sorted response matrix over overall_trace.csv.

Rows = (model, scaffold) pairs ("subjects"). Columns = items, where one item is
a unique (environment, level, category, verbosity, task) combination — the
finest grain at which a subject actually receives a distinct score. The 5
repeated trials per item are NOT separate items: they are collapsed into a
single cell value (mean success rate, i.e. Pass@1) before plotting, since
they are repeated measurements of the same item rather than independent
items. Verbosity IS kept as part of item identity, since it is a genuine
experimental manipulation applied to the task, not measurement noise.

Rows are sorted by overall ability (best subject on top), columns by overall
difficulty (easiest item on the left) — a standard Guttman-style "shakeout"
layout for spotting consistency (or lack of it) across subjects.

Usage:
    python plot_sorted_response_matrix.py
    python plot_sorted_response_matrix.py --value=score
    python plot_sorted_response_matrix.py --category=task
"""

import sys
from pathlib import Path

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "analysis" / "results" / "data" / "overall_trace.csv"
OUT_DIR = Path(__file__).parent / "figures"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from plot_config import (  # noqa: E402
    AGENT_NAMES,
    ENVIRONMENT_COLOURS,
    MODEL_COLOURS,
    MODEL_NAMES,
)

try:
    import lama_aesthetics

    lama_aesthetics.get_style("main")
except ImportError:
    pass

ITEM_KEY = ["environment", "level", "category", "verbosity", "task"]

# plot_config.ENVIRONMENT_COLOURS is missing "wetlab" — fixed (not positional)
# fallback so the same environment always gets the same colour regardless of
# which other environments happen to be present in a given figure.
ENV_COLOUR_FALLBACK = {"wetlab": "#e377c2"}


def load_data(path: Path = DATA_PATH, category: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if category is not None:
        df = df[df["category"] == category]
    return df


def build_matrix(
    df: pd.DataFrame, value: str = "success", item_key: list[str] = ITEM_KEY
) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse trials into one cell per (model, scaffold) x item, sorted.

    Returns:
        matrix: rows sorted by descending mean, cols sorted by ascending mean.
        col_env: item -> environment, aligned to matrix.columns, for the side strip.
    """
    df = df.copy()
    df["row"] = df["model"] + "__" + df["scaffold"]
    df["item"] = df[item_key].astype(str).agg("|".join, axis=1)

    pivot = df.pivot_table(index="row", columns="item", values=value, aggfunc="mean")
    n_missing = pivot.isna().sum().sum()
    if n_missing:
        logger.warning(f"{n_missing} missing (row, item) cells — leaving as NaN")

    col_env = df.drop_duplicates("item").set_index("item")["environment"]
    col_env = col_env.reindex(pivot.columns)

    row_order = pivot.mean(axis=1, skipna=True).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0, skipna=True).sort_values(ascending=True).index

    matrix = pivot.loc[row_order, col_order]
    col_env = col_env.loc[col_order]
    return matrix, col_env


def row_label(
    row_id: str, model_names: dict = MODEL_NAMES, agent_names: dict = AGENT_NAMES
) -> str:
    model, scaffold = row_id.split("__")
    return f"{model_names.get(model, model)} · {agent_names.get(scaffold, scaffold)}"


def plot_matrix(
    matrix: pd.DataFrame,
    col_env: pd.Series,
    value_label: str,
    output_path: Path,
    model_names: dict = MODEL_NAMES,
    model_colours: dict = MODEL_COLOURS,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    n_rows, n_cols = matrix.shape
    fig_w = max(10.0, min(0.006 * n_cols, 22.0))
    fig = plt.figure(figsize=(fig_w, 3.6))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1, 0.02],
        height_ratios=[1, 0.06],
        hspace=0.08,
        wspace=0.02,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, 0], sharex=ax_heat)

    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("#e5e5e5")
    im = ax_heat.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(
        [row_label(r, model_names) for r in matrix.index], fontsize=8
    )
    for tick, row_id in zip(ax_heat.get_yticklabels(), matrix.index, strict=False):
        tick.set_color(model_colours.get(row_id.split("__")[0], "black"))
    ax_heat.set_xticks([])
    title = title or (
        f"Sorted response matrix ({n_rows} model×scaffold subjects × {n_cols} items)\n"
        f"rows: best subject top → worst bottom | columns: easiest item left → hardest right"
    )
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax_heat.set_title(title, fontsize=9)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label(value_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    envs_present = sorted(col_env.dropna().unique())
    env_colour_map = {
        env: ENVIRONMENT_COLOURS.get(env, ENV_COLOUR_FALLBACK.get(env, "#999999"))
        for env in envs_present
    }
    env_codes = {env: i for i, env in enumerate(envs_present)}
    strip = np.array([[env_codes[e] for e in col_env]])
    strip_cmap = plt.matplotlib.colors.ListedColormap(
        [env_colour_map[e] for e in envs_present]
    )
    ax_strip.imshow(
        strip,
        aspect="auto",
        cmap=strip_cmap,
        vmin=-0.5,
        vmax=len(envs_present) - 0.5,
        interpolation="none",
    )
    ax_strip.set_yticks([0])
    ax_strip.set_yticklabels(["env"], fontsize=7)
    ax_strip.set_xticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=c, markersize=8, label=e
        )
        for e, c in env_colour_map.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=7,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(
    value: str = "success",
    category: str | None = None,
    data_path: str | None = None,
    output: str | None = None,
) -> None:
    """Plot a sorted (model x scaffold) x item response matrix.

    Args:
        value: cell metric to average over trials, "success" or "score".
        category: restrict to "task" or "subtask" only; None keeps both.
        data_path: override path to overall_trace.csv.
        output: override output path (PDF; a .png is saved alongside).
    """
    path = Path(data_path) if data_path else DATA_PATH
    logger.info(f"Loading {path}")
    df = load_data(path, category=category)
    logger.info(f"{len(df)} trial rows after filtering (category={category})")

    matrix, col_env = build_matrix(df, value=value)
    logger.info(f"Matrix shape: {matrix.shape[0]} subjects x {matrix.shape[1]} items")
    logger.info(
        "Per-subject mean:\n"
        + matrix.mean(axis=1, skipna=True).rename(row_label).round(3).to_string()
    )

    tag = f"_{category}" if category else ""
    out = (
        Path(output) if output else OUT_DIR / f"sorted_response_matrix_{value}{tag}.pdf"
    )
    plot_matrix(matrix, col_env, value_label=f"mean {value} rate", output_path=out)


if __name__ == "__main__":
    fire.Fire(main)
