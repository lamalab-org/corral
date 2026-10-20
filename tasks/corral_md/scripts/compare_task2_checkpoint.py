"""Compare the Task 2 reference and resumed trajectories; never assert a Tg.

Historical task2-2600K-v2 audit helper; not the active Task 2 protocol.

Requires numpy, pandas, and matplotlib. Run after prepare_task2_checkpoint.py.
The fit comparisons are descriptive, not confidence intervals or proof of a
microscopic glass transition. Retain the output with the published checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
from matplotlib import pyplot as plt

COLUMNS = [
    "step",
    "time_fs",
    "temperature_K",
    "pressure_atm",
    "density_g_cm3",
    "potential_energy_kcal_mol",
    "kinetic_energy_kcal_mol",
    "total_energy_kcal_mol",
    "volume_A3",
]


def read_log(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != len(COLUMNS):
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        rows.append(values)
    table = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("step", keep="first")
    if table.empty or not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"Missing or nonfinite thermodynamic output: {path}")
    return table


def fits(table):
    part = table[table.temperature_K <= 2600]
    x, y = part.temperature_K.to_numpy(), part.density_g_cm3.to_numpy()
    single = np.polyfit(x, y, 1)
    single_sse = float(np.sum((y - np.polyval(single, x)) ** 2))
    candidates = []
    for temperature in range(500, 2301, 5):
        if min(np.sum(x < temperature), np.sum(x > temperature)) < 10:
            continue
        design = np.column_stack([np.ones(len(x)), x, np.maximum(x - temperature, 0)])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        candidates.append(
            {
                "breakpoint_K": temperature,
                "sse": float(np.sum((y - design @ coefficients) ** 2)),
                "coefficients": coefficients.tolist(),
            }
        )
    best = min(candidates, key=lambda record: record["sse"])
    windows = []
    for low, high in [
        ((300, 1000), (1800, 2600)),
        ((300, 1200), (1800, 2500)),
        ((300, 1300), (1900, 2500)),
        ((300, 1000), (2000, 2500)),
        ((500, 1200), (1800, 2400)),
    ]:
        branches = [part[part.temperature_K.between(*window)] for window in [low, high]]
        a, b = [
            np.polyfit(branch.temperature_K, branch.density_g_cm3, 1)
            for branch in branches
        ]
        windows.append(
            {
                "low_window_K": low,
                "high_window_K": high,
                "intersection_K": float((b[1] - a[1]) / (a[0] - b[0])),
                "low_slope": float(a[0]),
                "high_slope": float(b[0]),
            }
        )
    return {
        "selected_samples": len(part),
        "single_line_sse": single_sse,
        "continuous_hinge": best,
        "hinge_sse_reduction": 1 - best["sse"] / single_sse,
        "hinge_bic_gain": len(part) * np.log(single_sse / best["sse"])
        - 2 * np.log(len(part)),
        "window_fits": windows,
        "window_range_K": [
            min(w["intersection_K"] for w in windows),
            max(w["intersection_K"] for w in windows),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = read_log(root / "reference/reference.in.log")
    resumed = read_log(root / "resumed/resume.in.log")
    if reference.step.iloc[-1] != 370000 or resumed.step.iloc[-1] != 700000:
        raise ValueError(
            "Wait for the complete reference and cooling/hold/reheating runs"
        )
    reference = reference[reference.step >= 140000].copy()
    cooling = resumed[resumed.step <= 370000].copy()
    hold = resumed[resumed.step.between(370000, 470000)].copy()
    reheating = resumed[resumed.step >= 470000].copy()
    reference["target_temperature_K"] = 4000 - reference.step * 0.01
    cooling["target_temperature_K"] = 4000 - cooling.step * 0.01
    hold["target_temperature_K"] = 300
    reheating["target_temperature_K"] = 300 + (reheating.step - 470000) * 0.01
    frames = {
        "reference_cooling": reference,
        "resumed_cooling": cooling,
        "hold": hold,
        "reheating": reheating,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    bins = []
    for low in range(300, 2600, 100):
        means = [
            float(
                frame[
                    frame.target_temperature_K.between(low, low + 100, inclusive="left")
                ].density_g_cm3.mean()
            )
            for frame in [reference, cooling]
        ]
        bins.append(
            {
                "target_window_K": [low, low + 100],
                "reference_density": means[0],
                "resumed_density": means[1],
                "difference_g_cm3": means[1] - means[0],
            }
        )
    initial_delta = (cooling.iloc[0] - reference.iloc[0]).to_dict()
    metadata = json.loads((root / "preparation.json").read_text())
    metadata["validation_status"] = (
        "Full reference and resumed cycle completed; see comparison.json for restart differences and fit sensitivity."
    )
    metadata["restart_limitations"] = (
        "Restart restores NPT state, but PPPM reinitialization and floating-point dynamics need not reproduce the uninterrupted trajectory bit for bit."
    )
    terminal_hold = hold[hold.step >= 390000]
    comparison = {
        "protocol_version": metadata["protocol_version"],
        "scope": "One uninterrupted cooling trajectory and one resumed cooling/hold/reheating trajectory from its checkpoint. These are not independent replicas.",
        "initial_observable_differences_resumed_minus_reference": initial_delta,
        "exact_initial_temperature_and_density": all(
            initial_delta[name] == 0 for name in ["temperature_K", "density_g_cm3"]
        ),
        "cooling_100_K_bins": bins,
        "cooling_bin_density_rmse_g_cm3": float(
            np.sqrt(np.mean([b["difference_g_cm3"] ** 2 for b in bins]))
        ),
        "density_based_fits": {
            name: fits(frame) for name, frame in frames.items() if name != "hold"
        },
        "hold_after_20_ps": {
            "mean_temperature_K": float(terminal_hold.temperature_K.mean()),
            "mean_density_g_cm3": float(terminal_hold.density_g_cm3.mean()),
            "density_slope_g_cm3_ps": float(
                np.polyfit(
                    terminal_hold.time_fs / 1000, terminal_hold.density_g_cm3, 1
                )[0]
            ),
        },
        "limitations": [
            "One prepared configuration; additional independent preparations are needed for statistical validation.",
            "Fit-window spread is not a confidence interval; samples are time-correlated.",
            "A fitted density breakpoint does not establish a microscopic glass transition or an equilibrated liquid branch.",
            metadata["restart_limitations"],
        ],
    }
    retained = [
        "reference/reference.in",
        "reference/reference.in.log",
        "reference/pot.mod",
        "resumed/export.in",
        "resumed/resume.in",
        "resumed/resume.in.log",
        "lammps-version.txt",
    ]
    comparison["source_sha256"] = {}
    for relative in retained:
        source = root / relative
        destination = output / relative.replace("/", "_")
        shutil.copyfile(source, destination)
        comparison["source_sha256"][destination.name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    comparison_text = json.dumps(comparison, indent=2) + "\n"
    (output / "comparison.json").write_text(comparison_text)
    metadata["validation_report_sha256"] = hashlib.sha256(
        comparison_text.encode()
    ).hexdigest()
    (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, label, color in [
        ("reference_cooling", "Uninterrupted cooling", "#334155"),
        ("resumed_cooling", "Resumed cooling", "#0284c7"),
        ("reheating", "Reheating after hold", "#ea580c"),
    ]:
        data = frames[name]
        selected = data[data.temperature_K <= 2600]
        axes[0].scatter(
            selected.temperature_K,
            selected.density_g_cm3,
            s=8,
            alpha=0.45,
            color=color,
            label=label,
        )
        axes[1].plot(
            (data.step - 140000) / 1000,
            data.density_g_cm3,
            color=color,
            lw=1,
            label=label,
        )
    axes[1].plot(
        (hold.step - 140000) / 1000,
        hold.density_g_cm3,
        color="#7c3aed",
        lw=1,
        label="300 K hold",
    )
    axes[0].set(
        xlabel="Measured temperature (K)",
        ylabel="Density (g/cm³)",
        title="Density response; T ≤ 2600 K",
    )
    axes[1].set(
        xlabel="Time since checkpoint (ps)",
        ylabel="Density (g/cm³)",
        title="Reference and resumed trajectories",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(output / "checkpoint_comparison.png", dpi=160)
    fig.savefig(output / "checkpoint_comparison.pdf")


if __name__ == "__main__":
    main()
