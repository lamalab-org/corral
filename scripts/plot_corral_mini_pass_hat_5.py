"""Compare unsaturated-first Rank + cost and Rank + score using Pass^5.

Run with the pinned tasks table, for example:
  python scripts/plot_corral_mini_pass_hat_5.py --tasks-parquet PATH

Writes a results CSV and a cost-based PNG and PDF figure. AFM uses the
November reports recorded in the corral-mini v2 manifest, and the
environments under the manifest's `local_sources` use their reports in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import lama_aesthetics as lama
import matplotlib.pyplot as plt
import numpy as np
from corral_mini_common import (
    held_out_model_losses,
    load,
    load_afm_strata,
    load_report_strata,
    material_losses,
    row_spearman,
    select,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src/corral/subsets/corral_mini_v2.json"
METHODS = {
    "rank_cost": ("Rank + cost", "#845B97"),
    "rank": ("Rank + score", "#FF9500"),
}


def load_strata(tasks_parquet: Path) -> tuple[dict, list[str]]:
    manifest = json.loads(MANIFEST.read_text())
    prices_path = ROOT / "corral_mini/prices.json"
    strata, agents, _ = load(
        manifest["source"]["revision"], tasks_parquet, prices_path, "pass_hat_5"
    )
    strata.update(load_afm_strata(ROOT / "reports/afm/trials.csv", prices_path, agents))
    for environment, local in manifest.get("local_sources", {}).items():
        strata.update(
            load_report_strata(
                ROOT / local["trials_csv"],
                prices_path,
                agents,
                "pass_hat_5",
                environment,
            )
        )
    return dict(sorted(strata.items())), agents


def score(strata: dict, selection: dict) -> dict:
    rhos = []
    full_means = []
    mini_means = []
    level_change = []
    inversions = 0
    cost = 0.0
    for key, stratum in strata.items():
        full = stratum.passes.mean(axis=0)
        chosen = selection[key]
        mini = stratum.passes[chosen].mean(axis=0)
        if np.std(full) > 0:
            rhos.append(float(row_spearman(mini[None], full)[0]))
        inversions += material_losses(full, mini) > 0
        full_mean = float(full.mean())
        mini_mean = float(mini.mean())
        if full_mean > 0:
            full_means.append(full_mean)
            mini_means.append(mini_mean)
            level_change.append(100 * (mini_mean / full_mean - 1))
        cost += float(stratum.costs[chosen].sum())
    return {
        "mean_level_spearman": float(np.mean(rhos)),
        "level_spearman_q25": float(np.percentile(rhos, 25)),
        "level_spearman_q75": float(np.percentile(rhos, 75)),
        "levels_with_loss": inversions,
        "mean_score_change_pct": 100 * (sum(mini_means) / sum(full_means) - 1),
        "min_level_score_change_pct": min(level_change),
        "max_level_score_change_pct": max(level_change),
        "levels_with_defined_score_change": len(level_change),
        "usd": cost,
    }


def compare(strata: dict, agents: list[str], max_k: int) -> list[dict]:
    all_agents = np.arange(len(agents))
    rows = []
    for k in range(1, max_k + 1):
        for method in METHODS:
            chosen = {
                key: select(stratum, k, method, all_agents, exclude_saturated=True)
                for key, stratum in strata.items()
            }
            row = {
                "tasks_per_level": k,
                "total_tasks": k * len(strata),
                "method": method,
                **score(strata, chosen),
                "held_out_levels_with_loss": held_out_model_losses(
                    strata, agents, k, method, exclude_saturated=True
                ),
            }
            rows.append(row)
    return rows


def plot(rows: list[dict], stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(lama.TWO_COL_WIDTH * 1.25, 3.2))
    all_x = []
    spearman_y = []
    held_out_y = []
    for method, (label, color) in METHODS.items():
        series = [row for row in rows if row["method"] == method]
        x = [row["usd"] / 1000 for row in series]
        all_x.extend(x)
        spearman_y.extend(r["mean_level_spearman"] for r in series)
        held_out_y.extend(r["held_out_levels_with_loss"] for r in series)
        axes[0].plot(
            x,
            [r["mean_level_spearman"] for r in series],
            color=color,
            marker="o",
            label=label,
        )
        axes[1].plot(
            x,
            [r["held_out_levels_with_loss"] for r in series],
            color=color,
            marker="o",
            label=label,
        )
        axes[2].plot(
            x,
            [r["mean_score_change_pct"] for r in series],
            color=color,
            marker="o",
            label=label,
        )
        axes[2].plot(
            x,
            [r["min_level_score_change_pct"] for r in series],
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
        axes[2].plot(
            x,
            [r["max_level_score_change_pct"] for r in series],
            color=color,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
    axes[0].set_title("Within-level Spearman ↑", loc="left")
    axes[1].set_title("Held-out rank inversions ↓", loc="left")
    axes[2].set_title("Mean Pass$^5$ change (%)\nDashed = level min/max", loc="left")
    axes[2].axhline(0, color="#474747", linestyle=":", linewidth=0.6)
    axes[2].set_yscale("symlog", linthresh=50)
    axes[2].set_ylim(-120, 560)
    axes[2].set_yticks(
        [-100, -50, 0, 50, 100, 200, 500],
        labels=["-100", "-50", "0", "+50", "+100", "+200", "+500"],
    )
    lama.range_frame(axes[0], all_x, spearman_y)
    lama.range_frame(axes[1], all_x, held_out_y)
    for ax in axes:
        ax.set_xlabel("Estimated cost (k USD)")
        ax.set_xticks([5, 10, 15, 20])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, handlelength=2.5)
    fig.text(
        0.01,
        0.97,
        "Pass$^5$ · unsaturated tasks first · score axis uses symmetric log scale",
        fontsize=8,
        color="#474747",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-parquet", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corral_mini")
    args = parser.parse_args()
    lama.get_style("main")
    strata, agents = load_strata(args.tasks_parquet)
    rows = compare(strata, agents, args.max_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results_pass_hat_5_unsaturated.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot(rows, figures / "pass_hat_5_tradeoffs_vs_cost_unsaturated")


if __name__ == "__main__":
    main()
