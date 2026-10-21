"""Check real validation trajectories against both Task 2 evidence contracts.

This creates maintainer evidence, not an agent benchmark trial. Tg is explicitly
left unresolved; Level 1 should pass and Level 2 should await scientific review.
Run with corral_md installed, after prepare_task2_original.py completes. Use --level 1 for cooling/hold only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units
from ase.io import read
from compare_task2_checkpoint import read_log
from corral_md.score import WorkflowScorer
from corral_md.workflow_scoring.task_2 import PROTOCOL_VERSION


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    source, output = args.run_dir.resolve(), args.output.resolve()
    table = read_log(source / "run.log")
    if table.step.iloc[0] != 0 or table.step.iloc[-1] != (
        470000 if args.level == 1 else 840000
    ):
        raise ValueError("The complete validation cycle is required")
    output.mkdir(parents=True, exist_ok=True)
    files = [
        "run.in",
        "run.log",
        "pot.mod",
        "liq4000.dat",
        "cooling_end.dat",
        "hold_end.dat",
    ]
    if args.level == 2:
        files.append("reheating_end.dat")
    for name in files:
        shutil.copyfile(source / name, output / name)
    (output / "preparation.in").write_text(
        (source / "run.in").read_text().split("unfix hold", 1)[0]
    )
    stages = []
    for name, start, stop in [
        ("cooling", 0, 370000),
        ("hold", 370000, 470000),
        ("reheating", 470000, 840000),
    ]:
        if name == "reheating" and args.level == 1:
            continue
        part = table[table.step.between(start, stop)].copy()
        part["stage"] = name
        part["time_ps"] = part.step / 1000
        part["target_temperature_K"] = (
            4000 - 0.01 * part.step
            if name == "cooling"
            else 300 + 0.01 * (part.step - start)
            if name == "reheating"
            else 300
        )
        stages.append(part)
    trace = pd.concat(stages, ignore_index=True)
    trace.to_csv(output / "thermal.csv", index=False)
    trace[trace.stage.isin(["cooling", "hold"])].to_csv(
        output / "preparation.csv", index=False
    )
    states = {}
    for name, filename, time in [
        ("initial", "liq4000.dat", 0),
        ("cooling_end", "cooling_end.dat", 370),
        ("hold_start", "cooling_end.dat", 370),
        ("hold_end", "hold_end.dat", 470),
        ("reheating_start", "hold_end.dat", 470),
        ("reheating_end", "reheating_end.dat", 840),
    ]:
        if name.startswith("reheating") and args.level == 1:
            continue
        atoms = read(
            source / filename,
            format="lammps-data",
            atom_style="charge",
            units="real",
            sort_by_id=True,
        )
        states[name] = {
            "time_ps": time,
            "cell": atoms.cell.array.tolist(),
            "positions": atoms.positions.tolist(),
            "ids": atoms.arrays["id"].tolist(),
            "species": atoms.get_chemical_symbols(),
            "charges": atoms.get_initial_charges().tolist(),
            "velocities": (atoms.get_velocities() * units.fs).tolist(),
        }
    save(output, "boundaries.json", states)
    save(
        output,
        "preparation_boundaries.json",
        {
            key: states[key]
            for key in ["initial", "cooling_end", "hold_start", "hold_end"]
        },
    )
    transitions = {}
    for stage in ["cooling", "reheating"] if args.level == 2 else ["cooling"]:
        selected = trace[(trace.stage == stage) & (trace.temperature_K <= 2600)]
        coefficients = np.polyfit(selected.temperature_K, selected.density_g_cm3, 1)
        transitions[stage] = {
            "method": "unresolved",
            "temperature_coordinate": "temperature_K",
            "indices": selected.index.tolist(),
            "estimate_K": None,
            "reason": "This maintainer run checks evidence compatibility; a resolved physical glass transition has not been established.",
            "diagnostics": {
                "single_line_sse": float(
                    np.sum(
                        (
                            selected.density_g_cm3
                            - np.polyval(coefficients, selected.temperature_K)
                        )
                        ** 2
                    )
                )
            },
        }
    hold = trace[(trace.stage == "hold") & (trace.time_ps >= 390)]
    analysis = {
        "transitions": transitions,
        "hold": {
            "indices": hold.index.tolist(),
            "drift_indices": hold.index.tolist(),
            "drift_method": "linear_slope",
        },
    }
    settings = {
        "protocol_version": PROTOCOL_VERSION,
        "units": "real",
        "timestep_fs": 1,
        "potential": "Supplied BKS",
        "software_versions": {"lammps": "22 Jul 2025"},
    }
    manifest = {
        "settings": save(output, "settings.json", settings),
        "scripts": ["run.in"],
        "artifacts": {
            "lammps_inputs": "run.in",
            "raw_logs": "run.log",
            "potential": "pot.mod",
            "initial_data": "liq4000.dat",
            "thermal_trace": "thermal.csv",
            "boundary_states": "boundaries.json",
            "analysis": save(output, "analysis.json", analysis),
        },
        "results": {
            "cooling_tg_K": None,
            "reheating_tg_K": None,
            "delta_tg_K": None,
            "hold_temperature_K": float(hold.temperature_K.mean()),
            "hold_density_g_cm3": float(hold.density_g_cm3.mean()),
            "hold_density_drift_g_cm3_ps": float(
                np.polyfit(hold.time_ps, hold.density_g_cm3, 1)[0]
            ),
            "units": {
                "temperature": "K",
                "density": "g/cm3",
                "density_drift": "g/cm3/ps",
            },
        },
    }
    if args.level == 2:
        save(output, "level2_manifest.json", manifest)
    manifest["artifacts"].update(
        lammps_inputs="preparation.in",
        thermal_trace="preparation.csv",
        boundary_states="preparation_boundaries.json",
    )
    manifest["scripts"] = ["preparation.in"]
    save(output, "level1_manifest.json", manifest)
    for level in range(1, args.level + 1):
        report = WorkflowScorer(2, level=level).evaluate(
            str(output / f"level{level}_manifest.json")
        )
        save(output, f"level{level}_score.json", report)
        if any(check["status"] == "failed" for check in report["checks"]):
            raise ValueError(
                f"Real evidence failed Level {level}: inspect level{level}_score.json"
            )
        if level == 1 and report["score"] != 1:
            raise ValueError("The complete Level 1 simulation should pass")
        if level == 2 and set(report["pending_checks"]) != {
            "cooling_transition_reproduction",
            "reheating_transition_reproduction",
        }:
            raise ValueError(
                "Only scientific interpretation of the unresolved transitions should remain pending"
            )


if __name__ == "__main__":
    main()
