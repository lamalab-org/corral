"""Plot full versus mini v2 Pass^5 and Pass@5 for each observed model.

Use the pinned tasks table named in the v2 manifest:
  python scripts/plot_corral_mini_v2_metrics.py --tasks-parquet PATH
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import lama_aesthetics as lama
import matplotlib.pyplot as plt
import numpy as np
from plot_corral_mini_v2_scores import MODEL_LABELS, ROOT, environment_scores, load_data

METRICS = {
    "pass_hat_5": "Pass$^5$",
    "pass_at_5": "Pass@5",
}


def comparison_rows(tasks_parquet: Path) -> tuple[list[dict], list[str]]:
    manifest, hat_strata, agents = load_data(tasks_parquet, "pass_hat_5")
    _, at_strata, at_agents = load_data(tasks_parquet, "pass_at_5")
    if agents != at_agents or hat_strata.keys() != at_strata.keys():
        raise ValueError("Pass^5 and Pass@5 reports do not align")
    aggregate = environment_scores(manifest, hat_strata, np.arange(len(agents)))
    environment_order = [row["environment"] for row in aggregate]
    rows = []
    for model in MODEL_LABELS:
        indices = np.array(
            [i for i, agent in enumerate(agents) if agent.endswith(f"__{model}")]
        )
        if not len(indices):
            raise ValueError(f"model {model} is missing from the reports")
        for metric, strata in (
            ("pass_hat_5", hat_strata),
            ("pass_at_5", at_strata),
        ):
            rows.extend(
                {
                    "model": model,
                    "agent_combinations": len(indices),
                    "metric": metric,
                    "environment": score["environment"],
                    "levels": score["levels"],
                    "full_score": score["full_pass_hat_5"],
                    "mini_score": score["mini_pass_hat_5"],
                    "change_percentage_points": score["change_percentage_points"],
                    "relative_change_pct": score["relative_change_pct"],
                }
                for score in environment_scores(manifest, strata, indices)
            )
    return rows, environment_order


def plot(rows: list[dict], environment_order: list[str], output: Path) -> None:
    fig, axes = plt.subplots(
        len(MODEL_LABELS),
        len(METRICS),
        figsize=(lama.TWO_COL_WIDTH * 2.05, 10.4),
        sharex=True,
    )
    y = list(range(len(environment_order)))
    for row_index, model in enumerate(MODEL_LABELS):
        for col_index, (metric, title) in enumerate(METRICS.items()):
            ax = axes[row_index, col_index]
            by_environment = {
                row["environment"]: row
                for row in rows
                if row["model"] == model and row["metric"] == metric
            }
            ordered = [by_environment[env] for env in environment_order]
            full = [100 * row["full_score"] for row in ordered]
            mini = [100 * row["mini_score"] for row in ordered]
            ax.hlines(y, full, mini, color="#C9C9C9", linewidth=1.3, zorder=1)
            ax.scatter(full, y, color="#474747", s=23, zorder=3)
            ax.scatter(mini, y, color="#845B97", s=23, zorder=4)
            for y_pos, row in zip(y, ordered, strict=True):
                ax.text(
                    103,
                    y_pos,
                    f"{row['change_percentage_points']:+.1f}",
                    va="center",
                    fontsize=7,
                    color="#474747",
                )
            ax.set_xlim(-2, 118)
            ax.set_ylim(len(y) - 0.45, -0.55)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            labels = (
                [env.replace("_", " ") for env in environment_order]
                if col_index == 0
                else [""] * len(y)
            )
            ax.set_yticks(y, labels)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_title(f"{MODEL_LABELS[model]} · {title}", loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Average score (%)")
    fig.text(
        0.02,
        0.99,
        "Full vs corral-mini v2 · grey = full · purple = mini · right labels = mini - full (pp) · equal level weight",
        fontsize=8,
        color="#474747",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=1.3, w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corral_mini")
    args = parser.parse_args()
    lama.get_style("main")
    rows, environment_order = comparison_rows(args.tasks_parquet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "v2_pass_metrics_by_model_environment.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(
        rows,
        environment_order,
        args.output_dir / "figures/v2_pass_metrics_by_model_environment",
    )


if __name__ == "__main__":
    main()
