"""Plot completed corral-mini v1 reports by model family.

Run from the repository root with the project's plotting dependencies:
    MPLCONFIGDIR=/private/tmp/corral-mini-mpl .venv/bin/python scripts/plot_mini_v1_reports.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import lama_aesthetics as lama
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ENVIRONMENTS = (
    ("corral_md", "Corral MD", (1, 2)),
    ("psychometrics", "Psychometrics", (1, 2)),
    ("wetlab", "Wetlab", (2, 3)),
    ("spectra_elucidation", "Spectra elucidation", (1, 2)),
    ("resistor_network", "Resistor network", (1, 2)),
    ("inference_opt", "Inference optimization", (1, 2)),
)
FAMILIES = {
    "kimi": (
        ("kimi-k3.1-pro", "K3.1 Pro", ("openhands", "react")),
        ("kimi-k2.5", "K2.5", ("openhands", "react")),
        ("kimi-k2-0905", "K2-0905", ("openhands", "react")),
    ),
    "claude": (
        ("claude-opus-5-5", "Opus 5.5", ("claude-code", "react")),
        ("claude-opus-4-7", "Opus 4.7", ("claude-code", "react")),
        ("claude-opus-4-5-20251101", "Opus 4.5", ("claude-code", "react")),
    ),
    "openai": (
        ("gpt-6.1-sol", "GPT 6.1 Sol", ("codex", "react")),
        ("gpt-5.6-sol", "GPT 5.6 Sol", ("codex", "react")),
        ("gpt-5-2025-08-07", "GPT 5", ("codex", "react")),
    ),
}


def load_reports(reports_dir: Path) -> tuple[dict, Counter]:
    runs = {}
    limit_hits: Counter = Counter()
    for environment, _, _ in ENVIRONMENTS:
        directory = reports_dir / environment
        with (directory / "runs.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if "mini-v1" not in row["run_id"]:
                    continue
                key = (environment, int(row["level"]), row["agent"], row["model"])
                if key in runs:
                    raise ValueError(f"Duplicate run: {key}")
                if int(row["trials"]) != 15 or int(row["trials_per_task"]) != 5:
                    raise ValueError(f"Expected 3 tasks x 5 trials: {key}")
                runs[key] = row
        with (directory / "trials.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if "mini-v1" not in row["run_id"]:
                    continue
                error = row["error"].lower()
                if "iteration" in error or "budget" in error:
                    key = (environment, int(row["level"]), row["agent"], row["model"])
                    limit_hits[key] += 1
    return runs, limit_hits


def columns(family: str) -> list[tuple[str, str, str]]:
    return [
        (model, agent, f"{short}\n{agent.replace('-', ' ').title()}")
        for model, short, agents in FAMILIES[family]
        for agent in agents
    ]


def draw(family: str, runs: dict, limit_hits: Counter, output_dir: Path) -> None:
    rows = [
        (env, name, level) for env, name, levels in ENVIRONMENTS for level in levels
    ]
    cols = columns(family)
    values = np.full((3, len(rows), len(cols)), np.nan)
    for i, (environment, _, level) in enumerate(rows):
        for j, (model, agent, _) in enumerate(cols):
            key = (environment, level, agent, model)
            if key in runs:
                values[:, i, j] = (
                    float(runs[key]["pass_at_k"]),
                    float(runs[key]["pass_hat_k"]),
                    limit_hits[key],
                )

    palette = (
        ("#0C5DA5", "pass@5"),
        ("#00B945", "pass^5"),
        ("#FF2C00", "iteration-limit trials"),
    )
    completed = np.count_nonzero(np.isfinite(values[2]))
    total_limit_hits = int(np.nansum(values[2]))
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 7.7), sharey=True)
    for panel, ax in enumerate(axes):
        color, title = palette[panel]
        cmap = LinearSegmentedColormap.from_list(title, ["#FFFFFF", color])
        cmap.set_bad("#EDEDED")
        matrix = np.ma.masked_invalid(values[panel])
        vmax = 15 if panel == 2 else 1
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        if panel == 2:
            title = f"{title} · {total_limit_hits}/{completed * 15}"
        ax.set_title(title, loc="left", fontsize=12, pad=17, color="#202020")
        ax.set_xticks(range(len(cols)), [label for _, _, label in cols], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=6, labelcolor="#303030")
        ax.set_yticks(
            range(len(rows)),
            [f"{name} · L{level}" for _, name, level in rows],
            fontsize=8,
        )
        ax.tick_params(axis="y", length=0, labelcolor="#303030")
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for edge in ("top", "bottom", "left", "right"):
            ax.spines[edge].set_visible(False)
        for i in range(len(rows)):
            for j in range(len(cols)):
                value = values[panel, i, j]
                if np.isnan(value):
                    label, ink = "—", "#999999"
                elif panel == 2:
                    label, ink = (
                        f"{int(value)}/15",
                        "white" if value >= 8 else "#222222",
                    )
                else:
                    label = f"{value * 100:.2f}"
                    ink = "white" if value >= 0.65 else "#222222"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.2, color=ink)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, shrink=0.68)
        cbar.set_ticks([0, 5, 10, 15] if panel == 2 else [0, 0.5, 1])
        cbar.set_ticklabels(
            ["0", "5", "10", "15"] if panel == 2 else ["0%", "50%", "100%"]
        )
        cbar.ax.tick_params(labelsize=7, labelcolor="#303030")

    display_family = {"kimi": "Kimi", "claude": "Claude", "openai": "OpenAI"}[family]
    fig.suptitle(
        f"Corral-mini v1 · {display_family} family",
        x=0.11,
        y=0.99,
        ha="left",
        fontsize=15,
        color="#202020",
    )
    fig.text(
        0.11,
        0.025,
        "Scores: 3 tasks x 5 trials per run · limit hits: trials reaching the configured 50-iteration cap (out of 15) · — = run not yet available",
        fontsize=8,
        color="#303030",
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.83, bottom=0.075, wspace=0.24)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(output_dir / f"mini_v1_{family}.{extension}", **options)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("corral_mini/figures"))
    args = parser.parse_args()
    lama.get_style("main")
    runs, limit_hits = load_reports(args.reports_dir)
    for family in FAMILIES:
        draw(family, runs, limit_hits, args.output_dir)


if __name__ == "__main__":
    main()
