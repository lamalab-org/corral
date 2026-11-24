"""Plot the corral-mini v2 Rank + cost tie audit.

Run with:
  uv run --no-project --with "lama-aesthetics @ git+https://github.com/lamalab-org/lama-aesthetics@f9aaa7e58b66cd4698ea59a6785c24a841221dd2" python scripts/plot_corral_mini_ties.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import lama_aesthetics as lama
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def plot(rows: list[dict[str, str]], output: Path) -> None:
    labels = [f"{r['environment'].replace('_', ' ')}  L{r['level']}" for r in rows]
    before = [int(r["sets_tied_at_best_rank"]) for r in rows]
    after = [int(r["sets_tied_after_cost"]) for r in rows]
    eligible = [int(r["eligible_sets"]) for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(lama.TWO_COL_WIDTH * 1.15, 6.4))
    ax.hlines(y, after, before, color="#C9C9C9", linewidth=1.4, zorder=1)
    ax.scatter(before, y, color="#845B97", s=30, label="Best Pass$^5$ rank", zorder=3)
    ax.scatter(
        after,
        y,
        color="#FF9500",
        marker="x",
        s=40,
        label="After cost tie-break",
        zorder=4,
    )
    for y_pos, count, total in zip(y, before, eligible, strict=True):
        ax.annotate(
            f"{count}/{total}",
            (count, y_pos),
            (7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#474747",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8, 190)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Number of eligible three-task sets tied (log scale)")
    ax.set_title("Corral-mini v2: ties in Rank + cost selection", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.text(
        0.02,
        0.94,
        "Labels = sets tied at best rank / all eligible sets · cost leaves one set in every level",
        fontsize=8,
        color="#474747",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ties",
        type=Path,
        default=ROOT / "corral_mini/rank_cost_pass_hat_5_ties.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "corral_mini/figures/rank_cost_pass_hat_5_ties",
    )
    args = parser.parse_args()
    lama.get_style("main")
    with args.ties.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plot(rows, args.output)


if __name__ == "__main__":
    main()
