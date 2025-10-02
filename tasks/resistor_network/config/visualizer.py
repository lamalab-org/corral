"""
Resistor Network Visualizer

Loads JSON task files and creates visual diagrams of resistor networks.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from loguru import logger


def load_task_json(filepath: str) -> dict[str, Any]:
    """Load task JSON file."""
    with Path(filepath).open("r") as f:
        return json.load(f)


def create_network_graph(topology: dict[str, Any]) -> nx.Graph:
    """
    Create a NetworkX graph from topology.

    Args:
        topology: Dict with "resistors" and "connections"

    Returns:
        NetworkX graph with resistor labels on edges
    """
    G = nx.Graph()

    resistors = topology["resistors"]
    connections = topology["connections"]

    # Add edges with resistor info
    for node1, node2, resistor_id in connections:
        resistance = resistors.get(resistor_id, "?")

        # Handle parallel resistors (multiple edges between same nodes)
        if G.has_edge(node1, node2):
            # Get existing label
            existing = G[node1][node2].get("label", "")
            new_label = f"{existing}\n{resistor_id}={resistance}Ω"
            G[node1][node2]["label"] = new_label
        else:
            G.add_edge(
                node1,
                node2,
                label=f"{resistor_id}={resistance}Ω",
                resistor=resistor_id,
                resistance=resistance,
            )

    return G


def visualize_network(
    topology: dict[str, Any],
    title: str = "Resistor Network",
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 8),
):
    """
    Visualize a resistor network.

    Args:
        topology: Network topology dict
        title: Plot title
        save_path: Path to save image (optional)
        show: Whether to display plot
        figsize: Figure size
    """
    G = create_network_graph(topology)

    fig, ax = plt.subplots(figsize=figsize)

    # Use hierarchical layout if possible, otherwise spring
    try:
        # Try to use hierarchical layout for better readability
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except (nx.NetworkXError, ValueError):
        pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node in ["A", "B"]:  # Terminal nodes
            node_colors.append("#ff6b6b")
        else:
            node_colors.append("#4ecdc4")

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=1500, alpha=0.9, ax=ax
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)

    # Draw edges (resistors)
    nx.draw_networkx_edges(G, pos, width=3, alpha=0.6, edge_color="#95a5a6", ax=ax)

    # Draw edge labels (resistor values)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        font_size=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    # Add legend
    terminal_patch = mpatches.Patch(color="#ff6b6b", label="Terminal Nodes (A, B)")
    internal_patch = mpatches.Patch(color="#4ecdc4", label="Internal Nodes")
    ax.legend(handles=[terminal_patch, internal_patch], loc="upper right")

    # Add info box
    info_text = f"Resistors: {len(topology['resistors'])}\n"
    info_text += f"Nodes: {len(G.nodes())}\n"
    info_text += f"Connections: {len(topology['connections'])}"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved visualization to {}", save_path)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_task_file(
    json_filepath: str, output_dir: str | None = None, show: bool = False
) -> None:
    """
    Load a task JSON file and visualize all networks (main task and subtasks).

    Args:
        json_filepath: Path to task JSON file
        output_dir: Directory to save visualizations (if None, uses same dir as JSON)
        show: Whether to display plots
    """
    # Load JSON
    data = load_task_json(json_filepath)

    # Set output directory
    output_path = (
        Path(output_dir) if output_dir is not None else Path(json_filepath).parent
    )
    if str(output_path) == "":
        output_path = Path()
    output_path.mkdir(exist_ok=True)

    base_name = Path(json_filepath).stem

    # Visualize main task
    if "main_task" in data:
        for task_data in data["main_task"].values():
            topology = task_data["scoring_params"]["expected_topology"]
            title = f"Main Task: {task_data['name']}"
            save_path = str((output_path / f"{base_name}_main_task.jpg").resolve())

            visualize_network(topology, title=title, save_path=save_path, show=show)

    # Visualize subtasks
    if "subtasks" in data:
        for i, (_subtask_id, subtask_data) in enumerate(data["subtasks"].items()):
            topology = subtask_data["scoring_params"]["expected_topology"]
            title = f"Subtask {i+1}: {subtask_data['name']}"
            save_path = str((output_path / f"{base_name}_subtask_{i+1}.jpg").resolve())

            visualize_network(topology, title=title, save_path=save_path, show=show)

    logger.info("\nVisualization complete! Images saved in: {}", str(output_path))


def visualize_all_tasks_in_directory(
    directory: str, output_subdir: str = "visualizations", show: bool = False
) -> None:
    """
    Visualize all JSON task files in a directory.

    Args:
        directory: Directory containing JSON files
        output_subdir: Subdirectory name for visualizations
        show: Whether to display plots
    """
    base_path = Path(directory)
    output_path = base_path / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in base_path.iterdir() if p.suffix == ".json")

    logger.info("Found {} JSON files", len(json_files))

    for json_file in json_files:
        logger.info("\nProcessing {}...", json_file.name)
        filepath = str(json_file)
        try:
            visualize_task_file(filepath, output_dir=str(output_path), show=show)
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as exc:
            logger.error("Error processing {}: {}", json_file.name, exc)

    logger.info("\nAll visualizations saved to: {}", str(output_path))


def create_comparison_plot(
    topologies: list[dict[str, Any]],
    titles: list[str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Create a side-by-side comparison of multiple networks.

    Args:
        topologies: List of topology dicts
        titles: List of titles for each topology
        save_path: Path to save image
        show: Whether to display plot
    """
    n = len(topologies)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    if n == 1:
        axes = [axes]

    if len(titles) != n:
        raise ValueError("Length of titles must match number of topologies")

    for ax, topology, title in zip(axes, topologies, titles, strict=False):
        G = create_network_graph(topology)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        node_colors = [
            "#ff6b6b" if node in ["A", "B"] else "#4ecdc4" for node in G.nodes()
        ]

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=1000, alpha=0.9, ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="#95a5a6", ax=ax)

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved comparison to {}", save_path)

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse

    logger.info("Resistor Network Visualizer")
    logger.info("=" * 70)

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Visualize resistor network from JSON files"
    )
    parser.add_argument("json_file", nargs="?", help="JSON file to visualize")
    parser.add_argument("--output", "-o", help="Output directory for visualizations")
    args = parser.parse_args()

    # Example 1: Visualize a single JSON file
    if args.json_file:
        logger.info("\nVisualizing: {}", args.json_file)
        visualize_task_file(args.json_file, output_dir=args.output, show=False)

    # Example 2: Visualize all JSON files in current directory
    else:
        logger.info("\nUsage:")
        logger.info("  python visualizer.py <task_file.json> [--output OUTPUT_DIR]")
        logger.info("  or")
        logger.info(
            "  python visualizer.py [--output OUTPUT_DIR]  (visualizes all .json files in current dir)"
        )
        logger.info("\nVisualizing all JSON files in current directory...")
        visualize_all_tasks_in_directory(
            ".", output_subdir=args.output or "visualizations", show=False
        )

    logger.info("\n" + "=" * 70)
    logger.info("Done!")

# ============================================================================
# HELPER FUNCTION: Quick visualization from topology dict
# ============================================================================


def quick_viz(topology: dict[str, Any], title: str = "Network") -> None:
    """Quick visualization helper for interactive use."""
    visualize_network(topology, title=title, show=True)
