"""Visualize resistor_network circuit topologies as graphs.

Reads a task JSON file (or a directory of them) and draws each circuit's
`expected_topology` as a node/edge graph, with edges labeled by resistor id
and ohm value. Terminal nodes (A/B) are highlighted; parallel resistors
between the same node pair are drawn as separate curved edges.

Usage (from tasks/resistor_network, with the venv set up per README.md):
    uv run python -m resistor_network.visualize environments/level_1/tasks_json/task_0.json
    uv run python -m resistor_network.visualize environments/level_2/tasks_json --out figures/level_2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from corral.report.logging import event

TERMINAL_COLOR = "#f4a261"
NODE_COLOR = "#8ecae6"


def build_graph(topology: dict) -> nx.MultiGraph:
    """Build a MultiGraph from a `{"resistors": ..., "connections": ...}` topology dict."""
    resistors = topology["resistors"]
    connections = topology["connections"]
    g = nx.MultiGraph()
    for node_a, node_b, r_id in connections:
        g.add_edge(node_a, node_b, key=r_id, resistor=r_id, ohms=resistors[r_id])
    return g


def draw_topology(topology: dict, ax=None, title: str | None = None) -> None:
    """Draw a single circuit topology onto a matplotlib axes."""
    g = build_graph(topology)
    ax = ax or plt.gca()
    pos = nx.spring_layout(g, seed=0)

    node_colors = [TERMINAL_COLOR if n in ("A", "B") else NODE_COLOR for n in g.nodes]
    nx.draw_networkx_nodes(
        g, pos, node_color=node_colors, node_size=700, edgecolors="black", ax=ax
    )
    nx.draw_networkx_labels(g, pos, font_size=9, font_weight="bold", ax=ax)

    # group parallel edges between the same node pair so they can be curved apart
    by_pair: dict[frozenset, list] = {}
    for u, v, key, data in g.edges(keys=True, data=True):
        by_pair.setdefault(frozenset((u, v)), []).append((u, v, key, data))

    for pair_edges in by_pair.values():
        n = len(pair_edges)
        for i, (u, v, key, data) in enumerate(pair_edges):
            rad = 0.0 if n == 1 else 0.15 * (i - (n - 1) / 2)
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
            )
            mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
            # nudge label perpendicular to the edge, scaled with curvature
            dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
            label_pos = (mid[0] - dy * rad, mid[1] + dx * rad)
            ax.text(
                *label_pos,
                f"{key}\n{data['ohms']:g}Ω",
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
            )

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _load_tasks(path: Path) -> list[dict]:
    tasks = []
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for f in files:
        with f.open() as fh:
            tasks.extend(json.load(fh))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize resistor_network circuit topologies"
    )
    parser.add_argument(
        "path", type=Path, help="Task JSON file, or directory of task JSON files"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: show interactively)",
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.path)
    if not tasks:
        raise SystemExit(f"No tasks found at {args.path}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        topology = task["scoring_params"]["expected_topology"]
        title = f"{task['id']} (level {task.get('level', '?')}, {task.get('num_resistors', len(topology['resistors']))} resistors)"

        fig, ax = plt.subplots(figsize=(8, 8))
        draw_topology(topology, ax=ax, title=title)

        if args.out:
            out_path = args.out / f"{task['id']}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            event(
                "INFO",
                "visualize.figure_written",
                subsystem="runtime",
                benchmark="resistor_network",
                task_id=task["id"],
                path=str(out_path),
            )
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
