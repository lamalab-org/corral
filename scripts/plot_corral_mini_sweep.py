"""Plot mean within-level rank agreement and its spread across environments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

STYLES = {
    "random": ("Random", "#687782", "-"),
    "unsaturated_random": ("Unsaturated random", "#687782", "--"),
    "score_profile": ("Score profile (MSE)", "#2876B8", "-"),
    "unsaturated_score_profile": ("Unsaturated score profile", "#2876B8", "--"),
    "spearman_rank": ("Best Spearman", "#D77721", "-"),
    "guarded_spearman_rank": ("Spearman + score guard", "#D77721", "--"),
    "cost_aware": ("Cost-aware", "#348756", "-"),
    "unsaturated_cost_aware": ("Unsaturated cost-aware", "#348756", "--"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("corral_mini/sweep.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("corral_mini/figures"))
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("sweep CSV has no rows")
    grouped = {method: [] for method in STYLES}
    for row in rows:
        grouped[row["method"]].append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for method, (label, color, linestyle) in STYLES.items():
        values = sorted(grouped[method], key=lambda row: int(row["tasks_per_level"]))
        if not values:
            continue
        x = [int(row["tasks_per_level"]) for row in values]
        y = [float(row["mean_level_spearman"]) for row in values]
        low = [float(row["env_q25"]) for row in values]
        high = [float(row["env_q75"]) for row in values]
        ax.fill_between(x, low, high, color=color, alpha=0.085, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=3.5,
            linewidth=2.0 if linestyle == "-" else 1.7,
            label=label,
        )
    x_values = sorted({int(row["tasks_per_level"]) for row in rows})
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values) - 0.18, max(x_values) + 0.18)
    ax.set_ylim(0.25, 1.03)
    ax.set_xlabel("Tasks per level in each environment")
    ax.set_ylabel("Mean within-level Spearman correlation")
    fig.suptitle(
        "How task count affects within-level ranking agreement",
        x=0.11,
        y=0.96,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#DCE2E6", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.11,
        0.91,
        "Lines: mean over 15 informative levels   ·   Shaded bands: middle 50% of environments",
        color="#52616C",
        fontsize=9,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handlelength=3,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.85, bottom=0.30)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "rank_correlation_vs_tasks.png", dpi=220)
    fig.savefig(args.output_dir / "rank_correlation_vs_tasks.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
