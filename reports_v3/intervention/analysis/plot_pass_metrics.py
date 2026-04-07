"""Plot Pass@k and Pass^k vs k for intervention baseline runs."""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

RUNS_DIR = Path(__file__).parent.parent / "runs"

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "md", "ml"]
AGENTS = ["react", "toolcalling"]
AGENT_LABELS = {"react": "ReAct", "toolcalling": "ToolCalling"}
ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "md": "Molecular\nSimulation",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}

COLORS = {
    "spectra": "#1f77b4",
    "wetlab": "#2ca02c",
    "resistor": "#d62728",
    "md": "#ff7f0e",
    "ml": "#9467bd",
    "retrosynthesis": "#E8A317",
}

MARKERS = {
    "spectra": "o",
    "wetlab": "s",
    "resistor": "D",
    "md": "X",
    "ml": "^",
    "retrosynthesis": "P",
}


def load_metrics(env: str, agent: str) -> dict:
    report_path = (
        RUNS_DIR / env / agent / "baseline" / f"{env}_{agent}_none_report.json"
    )
    with open(report_path) as f:
        return json.load(f)["metrics"]


def extract_pass_at(metrics: dict) -> tuple[list[int], list[float]]:
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass@{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def extract_pass_caret(metrics: dict) -> tuple[list[int], list[float]]:
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass^{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return ks, vals


def main():
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT))

    all_ks = list(range(1, 16))

    for row, agent in enumerate(AGENTS):
        ax_at = axes[row, 0]
        ax_caret = axes[row, 1]

        all_pass_at_vals = []
        all_pass_caret_vals = []

        for env in ENVIRONMENTS:
            metrics = load_metrics(env, agent)
            color = COLORS[env]
            marker = MARKERS[env]

            ks, vals = extract_pass_at(metrics)
            ax_at.plot(
                ks,
                vals,
                color=color,
                marker=marker,
                markersize=5,
                label=ENV_LABELS.get(env, env.capitalize()),
                linewidth=1.5,
            )
            all_pass_at_vals.extend(vals)

            ks, vals = extract_pass_caret(metrics)
            ax_caret.plot(
                ks,
                vals,
                color=color,
                marker=marker,
                markersize=5,
                label=ENV_LABELS.get(env, env.capitalize()),
                linewidth=1.5,
            )
            all_pass_caret_vals.extend(vals)

        ax_at.set_ylabel("Pass@k")
        ax_caret.set_ylabel("Pass^k")
        ax_at.set_title(f"Pass@k — {AGENT_LABELS[agent]}")
        ax_caret.set_title(f"Pass^k — {AGENT_LABELS[agent]}")

        range_frame(
            ax_at,
            np.array(all_ks * len(ENVIRONMENTS)),
            np.array(all_pass_at_vals),
            pad=0.05,
        )
        range_frame(
            ax_caret,
            np.array(all_ks * len(ENVIRONMENTS)),
            np.array(all_pass_caret_vals),
            pad=0.05,
        )

        ax_at.legend(loc="best", framealpha=0.9)
        ax_caret.legend(loc="best", framealpha=0.9)

    axes[1, 0].set_xlabel("k")
    axes[1, 1].set_xlabel("k")

    fig.suptitle("Intervention Baseline: Pass@k and Pass^k", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(__file__).parent / "figures" / "pass_metrics_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
