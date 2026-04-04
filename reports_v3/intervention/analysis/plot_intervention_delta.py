"""Delta curves: (Pass^k_intervention - Pass^k_baseline) vs k.

Left subplot = success interventions, right = failed.
Environments as colors, steps as line styles.
Zero line = baseline reference.
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame

lama_aesthetics.get_style("main")

RUNS_DIR = Path(__file__).parent.parent / "runs"

ENVIRONMENTS = ["spectra", "wetlab", "retrosynthesis", "resistor", "ml"]
ENV_LABELS = {
    "spectra": "Spectroscopic Structure\nElucidation",
    "wetlab": "Inorganic Qualitative\nAnalysis",
    "resistor": "Circuit\nInference",
    "ml": "ML-based Property\nPrediction",
    "retrosynthesis": "Retrosynthetic\nPlanning",
}
AGENTS = ["react", "toolcalling"]

FAILED_STEPS = ["failed_step1", "failed_step2", "failed_stepn1", "failed_stepn2"]
SUCCESS_STEPS = ["success_step1", "success_step2", "success_stepn1", "success_stepn2"]

COLORS = {
    "spectra": "#1f77b4",
    "wetlab": "#2ca02c",
    "resistor": "#d62728",
    "ml": "#9467bd",
    "retrosynthesis": "#E8A317",
}

MARKERS = {
    "step1": "o",
    "step2": "s",
    "stepn1": "D",
    "stepn2": "^",
}

STEP_ALPHA = {
    "step1": 1.0,
    "step2": 0.7,
    "stepn1": 0.45,
    "stepn2": 0.3,
}


def load_metrics(env: str, agent: str, step: str) -> dict | None:
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        return json.load(f)["metrics"]


def extract_pass_caret(metrics: dict) -> np.ndarray:
    vals = []
    for k in range(1, 16):
        key = f"Pass^{k}"
        if key in metrics:
            vals.append(metrics[key])
    return np.array(vals)


def avg_pass_caret(env: str, step: str) -> np.ndarray | None:
    all_vals = []
    for agent in AGENTS:
        m = load_metrics(env, agent, step)
        if m is None:
            continue
        all_vals.append(extract_pass_caret(m))
    if not all_vals:
        return None
    return np.mean(all_vals, axis=0)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 0.75))
    ks = np.arange(1, 16)

    for ax, steps, title in [
        (axes[0], SUCCESS_STEPS, "Success interventions"),
        (axes[1], FAILED_STEPS, "Failed interventions"),
    ]:
        all_x, all_y = [], []

        # Zero reference
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="-", zorder=1)

        for env in ENVIRONMENTS:
            baseline = avg_pass_caret(env, "baseline")
            if baseline is None:
                continue

            for step_name in steps:
                suffix = step_name.split("_", 1)[1]
                intervention = avg_pass_caret(env, step_name)
                if intervention is None:
                    continue

                delta = intervention - baseline
                marker = MARKERS[suffix]
                alpha = STEP_ALPHA[suffix]
                label = f"{ENV_LABELS[env]} — {suffix.replace('step', 'Step ')}"

                ax.plot(
                    ks,
                    delta,
                    color=COLORS[env],
                    linestyle="-",
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.3,
                    alpha=alpha,
                    label=label,
                )
                all_x.extend(ks)
                all_y.extend(delta)

        ax.set_title(title)
        ax.set_xlabel("k")

        range_frame(ax, np.array(all_x), np.array(all_y), pad=0.08)
        ax.legend(loc="best", fontsize=5, framealpha=0.9, ncol=1)

    axes[0].set_ylabel("ΔPass^k  (intervention − baseline)")

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / "intervention_delta_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
