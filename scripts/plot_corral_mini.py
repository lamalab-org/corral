"""Plot corral-mini comparisons into corral_mini/figures.

Reads results.csv, results_unsaturated.csv (when present) and saturation.csv.
Run with:
uv run --no-project --with "lama-aesthetics @ git+https://github.com/lamalab-org/lama-aesthetics@f9aaa7e58b66cd4698ea59a6785c24a841221dd2" python scripts/plot_corral_mini.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import lama_aesthetics as lama
import matplotlib.pyplot as plt

STYLES = {  # method: (label, color, linestyle)
    "random": ("Random (median of 200)", "#474747", "--"),
    "profile": ("Score profile", "#0C5DA5", "-"),
    "rank": ("Rank", "#FF9500", "-"),
    "rank_guarded": ("Rank + score guard", "#FF2C00", "-"),
    "cost_aware": ("Cost-aware", "#00B945", "-"),
    "rank_cost": ("Rank + cost", "#845B97", "-"),
}


def series(rows: list[dict], method: str, column: str) -> list[float]:
    return [float(row[column]) for row in rows if row["method"] == method]


def legend(fig) -> None:
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, handlelength=2.5)


def level_spread(ax, rows: list[dict], method: str, x: list[float], color) -> list:
    """Draw the 25th-75th percentile of per-level Spearman as a bar on each point."""
    low = series(rows, method, "level_spearman_q25")
    high = series(rows, method, "level_spearman_q75")
    ax.vlines(x, low, high, color=color, linewidth=0.8, alpha=0.6)
    return low + high


def plot_tasks(rows: list[dict], stem: Path, note: str) -> None:
    """Mean within-level Spearman against tasks per level."""
    fig, ax = plt.subplots(figsize=(lama.ONE_COL_WIDTH * 1.6, 3.2))
    xs, ys = [], []
    for method, (label, color, linestyle) in STYLES.items():
        # Nudge methods apart so bars at the same task count do not overlap.
        dodge = (list(STYLES).index(method) - (len(STYLES) - 1) / 2) * 0.1
        x = [k + dodge for k in series(rows, method, "tasks_per_level")]
        y = series(rows, method, "mean_level_spearman")
        ax.plot(x, y, color=color, linestyle=linestyle, marker="o", label=label)
        xs += x
        ys += level_spread(ax, rows, method, x, color)
    lama.range_frame(ax, xs, ys)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Tasks per level")
    ax.set_title(
        "Within-level Spearman with the full benchmark" + note.replace(" (", "\n("),
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, stem)


def plot_cost(rows: list[dict], stem: Path, note: str) -> None:
    """Spearman, held-out losses and mean score shift against estimated cost."""
    panels = (
        ("mean_level_spearman", "Within-level Spearman ↑"),
        ("held_out_levels_with_loss", "Levels with rank inversion\n(held-out model) ↓"),
        (
            "mean_score_shift",
            "Mean score shift, mini - full\n(below 0 = mini is harder)",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(lama.TWO_COL_WIDTH * 1.25, 3.0))
    for ax, (column, title) in zip(axes, panels, strict=True):
        xs, ys = [], []
        for method, (label, color, linestyle) in STYLES.items():
            x = [usd / 1000 for usd in series(rows, method, "usd")]
            y = series(rows, method, column)
            ax.plot(x, y, color=color, linestyle=linestyle, marker="o", label=label)
            xs += x
            ys += y
            if column == "mean_level_spearman":
                ys += level_spread(ax, rows, method, x, color)
            if method == "random" and column == "mean_level_spearman":
                low = series(rows, method, "level_spearman_q25")
                for k, (xk, yk) in enumerate(zip(x, low, strict=True), start=1):
                    ax.annotate(
                        str(k), (xk, yk), (0, -8), textcoords="offset points",
                        ha="center", fontsize=7, color="#474747",
                    )  # fmt: skip
        if column == "mean_score_shift":
            ax.axhline(0, color="#474747", linewidth=0.6, linestyle=":")
        lama.range_frame(ax, xs, ys)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("Estimated cost (k USD)")
        ax.set_title(title, loc="left")
    full_usd = float(rows[0]["usd"]) / float(rows[0]["cost_fraction"])
    fig.text(
        0.01, 0.97,
        f"{note.strip(' ()').capitalize() + ' · ' if note else ''}Numbers under the random bars = tasks per level · bars = middle 50% of levels · full benchmark about ${full_usd / 1000:.1f}k",
        fontsize=8, color="#474747", va="top",
    )  # fmt: skip
    legend(fig)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    save(fig, stem)


def plot_saturation(path: Path, output_dir: Path) -> None:
    """Stacked bars: share of each level's tasks by how many of the 7 agents solve them."""
    with path.open(newline="") as handle:
        levels = list(csv.DictReader(handle))[::-1]  # first level on top
    segments = (
        ("solved_by_all", "All 7 (saturated)", "#0C5DA5"),
        ("solved_by_majority_not_all", "4-6", "#7FAEDB"),
        ("solved_by_minority", "1-3", "#C9C9C9"),
        ("solved_by_none", "None", "#EDEDED"),
    )
    labels = [
        f"{row['environment'].replace('_', ' ')} L{row['level']}" for row in levels
    ]
    fig, ax = plt.subplots(figsize=(lama.TWO_COL_WIDTH, 4.2))
    left = [0.0] * len(levels)
    for column, label, color in segments:
        share = [100 * int(r[column]) / int(r["tasks"]) for r in levels]
        ax.barh(labels, share, left=left, color=color, label=label, height=0.7)
        left = [a + b for a, b in zip(left, share, strict=True)]
    for y, row in enumerate(levels):
        tasks = int(row["tasks"])
        saturated = 100 * int(row["solved_by_all"]) / tasks
        majority = saturated + 100 * int(row["solved_by_majority_not_all"]) / tasks
        ax.text(
            103, y, f"{saturated:.0f}%  ·  {majority:.0f}%",
            va="center", fontsize=8, color="#474747",
        )  # fmt: skip
    ax.text(103, len(levels) - 0.3, "saturated · majority", fontsize=8, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of tasks in the level (%)")
    ax.set_title(
        "Agents (of 7) solving each task in at least 1 of 5 trials", loc="left"
    )
    ax.tick_params(axis="y", which="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4)
    fig.tight_layout()
    save(fig, output_dir / "saturation_by_level")


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("corral_mini"))
    parser.add_argument(
        "--saturation", type=Path, default=Path("corral_mini/saturation.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("corral_mini/figures"))
    args = parser.parse_args()
    lama.get_style("main")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = (("", ""), ("_unsaturated", " (unsaturated tasks first)"))
    for suffix, note in runs:
        path = args.results_dir / f"results{suffix}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        plot_tasks(rows, args.output_dir / f"rank_vs_tasks{suffix}", note)
        plot_cost(rows, args.output_dir / f"tradeoffs_vs_cost{suffix}", note)
    plot_saturation(args.saturation, args.output_dir)


if __name__ == "__main__":
    main()
