"""Plot the saved resistor bank and audit fitting with known wiring.

Run with the resistor environment and repository src directories on PYTHONPATH.
This is an oracle-topology diagnostic, not a topology solver or agent evaluation.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

# ruff: noqa: E402
# Bootstrap repository imports and configure the plotting backend before pyplot.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tasks/resistor_network/src")]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "resistor-mpl"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from resistor_network.sampler import (
    LEVELS,
    RESISTOR_POOL,
    _jacobian_rank,
    _measurement_jacobian,
    _validate_sampled_topology,
    measure_pairs,
)
from resistor_network.tools import estimate_resistor_values
from resistor_network.visualize import draw_topology

BANK = ROOT / "tasks/resistor_network"
OUT = BANK / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    with PdfPages(OUT / "all_circuits.pdf") as pdf:
        for level in (1, 2):
            paths = sorted(
                (BANK / f"environments/level_{level}/tasks_json").glob("*.json")
            )
            fig, axes = plt.subplots(2, 5, figsize=(25, 11))
            fig.suptitle(f"Level {level}: all ten saved circuits", fontsize=18)
            for path, ax in zip(paths, axes.flat, strict=True):
                task = json.loads(path.read_text())[0]
                topology = task["scoring_params"]["expected_topology"]
                measurements = task["initial_input"]["measurements"]
                pairs = [(m["node_a"], m["node_b"]) for m in measurements]
                assert _validate_sampled_topology(
                    topology, LEVELS[level], len(topology["resistors"])
                )
                assert measure_pairs(topology, pairs) == measurements
                jac = _measurement_jacobian(topology, pairs)
                singular = np.linalg.svd(jac, compute_uv=False)
                # First-order worst-direction value error from rounding alone.
                rounding_bound = 0.0005 * np.sqrt(len(pairs)) / singular[-1]
                record = {
                    "level": level,
                    "task": task["id"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "resistors": len(topology["resistors"]),
                    "nodes": len(
                        {n for a, b, _ in topology["connections"] for n in (a, b)}
                    ),
                    "measurements": len(measurements),
                    "jacobian_rank": _jacobian_rank(jac),
                    "log_value_jacobian_condition": float(singular[0] / singular[-1]),
                    "linearized_rounding_log_value_bound": float(rounding_bound),
                    "fits": [],
                }
                rng = np.random.default_rng(
                    1000 * level + int(task["id"].split("_")[1])
                )
                starts = [10.0, 47.0, 100.0] + [
                    rng.choice(RESISTOR_POOL, len(topology["resistors"])).tolist()
                    for _ in range(5)
                ]
                for attempt, start in enumerate(starts):
                    if attempt >= 2 and any(
                        f.get("exact_recovery_after_snapping") for f in record["fits"]
                    ):
                        break
                    candidate = {
                        "resistors": (
                            dict(zip(topology["resistors"], start, strict=True))
                            if isinstance(start, list)
                            else dict.fromkeys(topology["resistors"], start)
                        ),
                        "connections": topology["connections"],
                    }
                    started = perf_counter()
                    fitted = json.loads(
                        estimate_resistor_values.execute(
                            topology=json.dumps(candidate),
                            measurements=json.dumps(measurements),
                        )
                    )
                    fit = {
                        "initial_ohms": start,
                        "baseline_attempt": attempt < 2,
                        "seconds": round(perf_counter() - started, 3),
                        "optimization_info": fitted.get("optimization_info"),
                        "error": fitted.get("error"),
                    }
                    if "resistors" in fitted:
                        snapped = {
                            rid: min(RESISTOR_POOL, key=lambda r: abs(r - value))
                            for rid, value in fitted["resistors"].items()
                        }
                        fit["wrong_values_after_snapping"] = sum(
                            snapped[rid] != value
                            for rid, value in topology["resistors"].items()
                        )
                        fit["exact_recovery_after_snapping"] = (
                            snapped == topology["resistors"]
                        )
                        fit["max_relative_value_error"] = max(
                            abs(fitted["resistors"][rid] - value) / value
                            for rid, value in topology["resistors"].items()
                        )
                        fit["fitted_values"] = fitted["resistors"]
                        fit["snapped_values"] = snapped
                        fit["snapped_measurements_match"] = (
                            measure_pairs(
                                {
                                    "resistors": snapped,
                                    "connections": topology["connections"],
                                },
                                pairs,
                            )
                            == measurements
                        )
                    record["fits"].append(fit)
                    print(  # noqa: T201 -- progress output for the CLI diagnostic
                        json.dumps(
                            {
                                "level": level,
                                "task": task["id"],
                                **{
                                    k: v
                                    for k, v in fit.items()
                                    if k not in {"fitted_values", "snapped_values"}
                                },
                            }
                        ),
                        flush=True,
                    )
                records.append(record)
                draw_topology(topology, ax=ax, title=task["id"])
                individual, individual_ax = plt.subplots(figsize=(8, 8))
                draw_topology(
                    topology, ax=individual_ax, title=f"Level {level} / {task['id']}"
                )
                individual_dir = OUT / f"level_{level}"
                individual_dir.mkdir(exist_ok=True)
                individual.savefig(
                    individual_dir / f"{task['id']}.png", dpi=150, bbox_inches="tight"
                )
                pdf.savefig(individual, bbox_inches="tight")
                plt.close(individual)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(OUT / f"level_{level}_overview.png", dpi=150)
            plt.close(fig)
    (OUT / "difficulty_diagnostics.json").write_text(
        json.dumps(
            {
                "scope": "Known hidden wiring, published measurements only; existing estimate_resistor_values tool; nearest allowed-value snapping. Not an agent evaluation or global uniqueness proof.",
                "tasks": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
