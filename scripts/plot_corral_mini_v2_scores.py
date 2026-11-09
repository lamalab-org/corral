"""Plot full versus corral-mini v2 Pass^5 scores by environment.

Use the pinned tasks table named in the v2 manifest:
  python scripts/plot_corral_mini_v2_scores.py --tasks-parquet PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import lama_aesthetics as lama
import matplotlib.pyplot as plt
import numpy as np
from corral_mini_common import load, load_afm_strata

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src/corral/subsets/corral_mini_v2.json"
MODEL_LABELS = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "kimi-k3": "Kimi K3",
}


def load_data(tasks_parquet: Path) -> tuple[dict, dict, list[str]]:
    manifest = json.loads(MANIFEST.read_text())
    prices = ROOT / "corral_mini/prices.json"
    strata, agents, source = load(
        manifest["source"]["revision"], tasks_parquet, prices, "pass_hat_5"
    )
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("tasks table does not match the v2 manifest")
    afm_path = ROOT / manifest["afm_source"]["trials_csv"]
    if (
        hashlib.sha256(afm_path.read_bytes()).hexdigest()
        != manifest["afm_source"]["trials_csv_sha256"]
    ):
        raise ValueError("AFM trials do not match the v2 manifest")
    strata.update(load_afm_strata(afm_path, prices, agents))
    return manifest, dict(sorted(strata.items())), agents


def environment_scores(
    manifest: dict, strata: dict, agent_indices: np.ndarray
) -> list[dict]:
    by_environment: dict[str, list[tuple[float, float]]] = {}
    for (environment, level), stratum in sorted(strata.items()):
        ids = manifest["tasks_by_environment"][environment][f"level_{level}"]
        chosen = np.flatnonzero(np.isin(stratum.tasks, ids))
        if len(chosen) != manifest["tasks_per_level"]:
            raise ValueError(f"{environment} L{level}: mini tasks missing from reports")
        full = float(stratum.passes[:, agent_indices].mean())
        mini = float(stratum.passes[chosen][:, agent_indices].mean())
        by_environment.setdefault(environment, []).append((full, mini))

    rows = []
    for environment, levels in by_environment.items():
        full = float(np.mean([score[0] for score in levels]))
        mini = float(np.mean([score[1] for score in levels]))
        rows.append(
            {
                "environment": environment,
                "levels": len(levels),
                "full_pass_hat_5": full,
                "mini_pass_hat_5": mini,
                "change_percentage_points": 100 * (mini - full),
                "relative_change_pct": 100 * (mini / full - 1) if full else "",
            }
        )
    return sorted(rows, key=lambda row: row["change_percentage_points"])


def draw_scores(ax, rows: list[dict]) -> None:
    y = list(range(len(rows)))
    full = [100 * row["full_pass_hat_5"] for row in rows]
    mini = [100 * row["mini_pass_hat_5"] for row in rows]
    labels = [row["environment"].replace("_", " ") for row in rows]

    ax.hlines(y, full, mini, color="#C9C9C9", linewidth=1.5, zorder=1)
    ax.scatter(full, y, color="#474747", s=34, zorder=3)
    ax.scatter(mini, y, color="#845B97", s=34, zorder=4)
    for y_pos, row in zip(y, rows, strict=True):
        ax.text(
            103,
            y_pos,
            f"{row['change_percentage_points']:+.1f} pp",
            va="center",
            fontsize=8,
            color="#474747",
        )
    ax.set_xlim(-2, 118)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(lama.TWO_COL_WIDTH * 1.2, 4.2))
    draw_scores(ax, rows)
    ax.set_xlabel("Average Pass$^5$ score (%)")
    ax.set_title("Full benchmark to corral-mini v2", loc="left")
    fig.text(
        0.02,
        0.96,
        "Grey = full · purple = mini v2 · right labels = mini - full · seven agent/model combinations, equal level weight",
        fontsize=8,
        color="#474747",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    save(fig, output)


def plot_models(rows: list[dict], environment_order: list[str], output: Path) -> None:
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(
        len(models), 1, figsize=(lama.TWO_COL_WIDTH * 1.2, 10.5), sharex=True
    )
    for ax, model in zip(axes, models, strict=True):
        by_environment = {
            row["environment"]: row for row in rows if row["model"] == model
        }
        draw_scores(ax, [by_environment[env] for env in environment_order])
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=10)
    axes[-1].set_xlabel("Average Pass$^5$ score (%)")
    fig.text(
        0.02,
        0.985,
        "Full benchmark to corral-mini v2 · grey = full · purple = mini · right labels = mini - full (pp)",
        fontsize=8,
        color="#474747",
        va="top",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=1.5)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corral_mini")
    args = parser.parse_args()
    lama.get_style("main")
    manifest, strata, agents = load_data(args.tasks_parquet)
    rows = environment_scores(manifest, strata, np.arange(len(agents)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "v2_pass_hat_5_by_environment.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir / "figures/v2_pass_hat_5_by_environment")
    model_rows = []
    for model in MODEL_LABELS:
        indices = np.array(
            [i for i, agent in enumerate(agents) if agent.endswith(f"__{model}")]
        )
        if not len(indices):
            raise ValueError(f"model {model} is missing from the reports")
        model_rows.extend(
            {"model": model, "agent_combinations": len(indices), **row}
            for row in environment_scores(manifest, strata, indices)
        )
    with (args.output_dir / "v2_pass_hat_5_by_model_environment.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(model_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(model_rows)
    plot_models(
        model_rows,
        [row["environment"] for row in rows],
        args.output_dir / "figures/v2_pass_hat_5_by_model_environment",
    )


if __name__ == "__main__":
    main()
