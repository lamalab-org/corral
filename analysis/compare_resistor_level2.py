"""Compare a saved resistor bank on local main with the working tree."""

import argparse
import hashlib
import json
import subprocess

from review_resistor_network import OUT, ROOT

# isort: split
# The shared review module bootstraps repository imports and selects Agg.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from resistor_network.sampler import (
    RESISTOR_POOL,
    _jacobian_rank,
    _measurement_jacobian,
    _sampled_topology_quality,
    measure_pairs,
)
from resistor_network.tools import estimate_resistor_values
from resistor_network.visualize import draw_topology


def main(level=2):
    revision = subprocess.check_output(
        ["git", "rev-parse", "main"], cwd=ROOT, text=True
    ).strip()
    records = []
    with PdfPages(OUT / f"level_{level}_main_vs_current.pdf") as pdf:
        overview, axes = plt.subplots(2, 5, figsize=(25, 11))
        overview.suptitle(
            f"Level {level} on main ({revision[:10]}): all ten saved circuits",
            fontsize=18,
        )
        for index in range(10):
            relative = f"tasks/resistor_network/environments/level_{level}/tasks_json/task_{index}.json"
            raw = subprocess.check_output(
                ["git", "show", f"{revision}:{relative}"], cwd=ROOT
            )
            old = json.loads(raw)[0]
            new = json.loads((ROOT / relative).read_text())[0]
            comparison, compare_axes = plt.subplots(1, 2, figsize=(16, 8))
            for source, task, ax in zip(
                ("main", "current"), (old, new), compare_axes, strict=True
            ):
                topology = task["scoring_params"]["expected_topology"]
                measurements = task["initial_input"]["measurements"]
                pairs = [(m["node_a"], m["node_b"]) for m in measurements]
                assert measure_pairs(topology, pairs) == measurements
                quality = _sampled_topology_quality(topology)
                graph = nx.Graph()
                graph.add_edges_from((a, b) for a, b, _ in topology["connections"])
                jac = _measurement_jacobian(topology, pairs)
                singular = np.linalg.svd(jac, compute_uv=False)
                record = {
                    "source": source,
                    "task": task["id"],
                    "resistors": len(topology["resistors"]),
                    "nodes": quality.num_nodes,
                    "measurements": len(measurements),
                    "possible_measurements": quality.num_nodes
                    * (quality.num_nodes - 1)
                    // 2,
                    "cycle_rank": quality.cycle_rank,
                    "cyclic_component_ranks": [
                        r for r in quality.component_cycle_ranks if r > 0
                    ],
                    "is_biconnected": nx.is_biconnected(graph),
                    "parallel_groups": quality.parallel_groups,
                    "reduced_edges": [
                        [a, b, length]
                        for (a, b), length in quality.reduced_edges.items()
                    ],
                    "jacobian_rank": _jacobian_rank(jac),
                    "log_value_jacobian_condition": float(singular[0] / singular[-1]),
                    "sha256": hashlib.sha256(
                        raw if source == "main" else (ROOT / relative).read_bytes()
                    ).hexdigest(),
                }
                if source == "main":
                    record["fits"] = []
                    for start in (10.0, 47.0):
                        candidate = {
                            "resistors": dict.fromkeys(topology["resistors"], start),
                            "connections": topology["connections"],
                        }
                        fit = json.loads(
                            estimate_resistor_values.execute(
                                topology=json.dumps(candidate),
                                measurements=json.dumps(measurements),
                            )
                        )
                        result = {
                            "initial_ohms": start,
                            "optimization_info": fit.get("optimization_info"),
                            "error": fit.get("error"),
                        }
                        if "resistors" in fit:
                            snapped = {
                                rid: min(RESISTOR_POOL, key=lambda r: abs(r - value))
                                for rid, value in fit["resistors"].items()
                            }
                            result["exact_recovery_after_snapping"] = (
                                snapped == topology["resistors"]
                            )
                            result["snapped_measurements_match"] = (
                                measure_pairs(
                                    {
                                        "resistors": snapped,
                                        "connections": topology["connections"],
                                    },
                                    pairs,
                                )
                                == measurements
                            )
                        record["fits"].append(result)
                    draw_topology(
                        topology,
                        ax=axes.flat[index],
                        title=f"{task['id']}: {record['resistors']} resistors",
                    )
                draw_topology(
                    topology,
                    ax=ax,
                    title=f"{source} / {task['id']}: {record['resistors']} resistors, {record['measurements']} measurements",
                )
                records.append(record)
                print(json.dumps(record), flush=True)  # noqa: T201 -- CLI progress
            comparison.tight_layout()
            pdf.savefig(comparison)
            if index == 9:
                comparison.savefig(
                    OUT / f"level_{level}_main_vs_current_task_9.png", dpi=150
                )
            plt.close(comparison)
        overview.tight_layout(rect=(0, 0, 1, 0.96))
        overview.savefig(OUT / f"level_{level}_main_overview.png", dpi=150)
        plt.close(overview)
    (OUT / f"level_{level}_main_comparison.json").write_text(
        json.dumps({"main_commit": revision, "tasks": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=(1, 2), default=2)
    main(parser.parse_args().level)
