"""
Resistor Network Task and Subtask Generator

This module creates benchmarking tasks for resistor network topology inference.
It combines subnetworks into backbone structures to create main tasks and subtasks.
"""

import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger

# ============================================================================
# CIRCUIT SIMULATION FUNCTION
# ============================================================================

SUBNETWORK_LIBRARY = json.load(
    Path(__file__).parent.joinpath("NETWORK_DICT_EASY.json").open()
)["SUBNETWORK_LIBRARY"]
BACKBONE_LIBRARY = json.load(
    Path(__file__).parent.joinpath("NETWORK_DICT_EASY.json").open()
)["BACKBONE_LIBRARY"]


def simulate_circuit_resistance(topology: dict, terminal_nodes: list[str]) -> float:
    """
    Calculate resistance between two terminal nodes using nodal analysis.

    Args:
        topology: Dict with "resistors" and "connections"
        terminal_nodes: List of exactly two node names

    Returns:
        Resistance in ohms between the two terminals
    """
    resistors = topology["resistors"]
    connections = topology["connections"]

    if len(terminal_nodes) != 2:
        raise ValueError("Must specify exactly two terminal nodes")

    # Build node list
    nodes = set()
    for conn in connections:
        nodes.add(conn[0])
        nodes.add(conn[1])

    node_list = sorted(nodes)
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Create conductance matrix
    G = np.zeros((n, n))
    for node1, node2, resistor_id in connections:
        if resistor_id not in resistors:
            raise ValueError(f"Resistor {resistor_id} not found")

        resistance = resistors[resistor_id]
        if resistance <= 0:
            raise ValueError(f"Resistance must be positive: {resistance}")

        conductance = 1.0 / resistance
        i, j = node_to_idx[node1], node_to_idx[node2]

        G[i, i] += conductance
        G[j, j] += conductance
        G[i, j] -= conductance
        G[j, i] -= conductance

    # Solve for resistance between terminals
    term1_idx = node_to_idx[terminal_nodes[0]]
    term2_idx = node_to_idx[terminal_nodes[1]]

    # Apply 1A current
    Ia = np.zeros(n)
    Ia[term1_idx] = 1.0
    Ia[term2_idx] = -1.0

    # Remove reference equation
    if n == 1:
        return 0.0

    if term2_idx == n - 1:
        G_reduced = G[:-1, :-1]
        I_reduced = Ia[:-1]
    else:
        G_reduced = np.delete(np.delete(G, term2_idx, 0), term2_idx, 1)
        I_reduced = np.delete(Ia, term2_idx)

    try:
        if G_reduced.size == 0:
            raise np.linalg.LinAlgError("Reduced matrix is empty")
        V_reduced = np.linalg.solve(G_reduced, I_reduced)
    except np.linalg.LinAlgError as err:
        raise ValueError("Circuit not solvable") from err

    # Insert reference voltage
    if term2_idx == n - 1:
        V = np.append(V_reduced, 0)
    else:
        V = np.insert(V_reduced, term2_idx, 0)

    return abs(V[term1_idx] - V[term2_idx])


def relabel_subnetwork_nodes_with_gap(
    subnetwork: dict, start_node: str, end_node: str, gap_id: str
) -> dict:
    """
    Relabel subnetwork nodes using gap-based naming convention.

    Args:
        subnetwork: Subnetwork definition
        start_node: First terminal node label (e.g., 'A', 'X1')
        end_node: Last terminal node label (e.g., 'X1', 'X2', 'B')
        gap_id: Gap identifier (e.g., 'a', 'b', 'c')

    Returns:
        Relabeled subnetwork
    """
    orig_terminals = subnetwork["terminals"]
    if len(orig_terminals) != 2:
        raise ValueError("Subnetwork must have exactly 2 terminals")

    # Create mapping for terminals
    node_mapping = {orig_terminals[0]: start_node, orig_terminals[1]: end_node}

    # Find internal nodes
    internal_nodes = set()
    for conn in subnetwork["connections"]:
        internal_nodes.add(conn[0])
        internal_nodes.add(conn[1])
    internal_nodes -= set(orig_terminals)

    # Map internal nodes using gap-based naming
    for i, node in enumerate(sorted(internal_nodes), start=1):
        node_mapping[node] = f"N_{gap_id}{i}"

    # Apply mapping
    return {
        "name": subnetwork["name"],
        "resistors": subnetwork["resistors"].copy(),
        "connections": [
            [node_mapping[conn[0]], node_mapping[conn[1]], conn[2]]
            for conn in subnetwork["connections"]
        ],
        "terminals": [start_node, end_node],
    }


def assemble_network(
    backbone_key: str, subnetwork_keys: list[str], resistance_scale: float = 1.0
) -> tuple[dict, list[dict]]:
    """
    Assemble a complete network by inserting subnetworks into backbone gaps.
    """
    backbone = BACKBONE_LIBRARY[backbone_key]
    gaps = backbone["gaps"]

    if len(subnetwork_keys) != len(gaps):
        raise ValueError(f"Need {len(gaps)} subnetworks for {len(gaps)} gaps")

    # Create node labels for backbone
    node_labels = ["A"]
    node_labels.extend([f"X{i+1}" for i in range(len(gaps) - 1)])
    node_labels.append("B")

    # Initialize collections
    all_resistors = {}
    all_connections = []
    relabeled_subnetworks = []
    resistor_counter = 1

    for i, subnet_key in enumerate(subnetwork_keys):
        subnet = SUBNETWORK_LIBRARY[subnet_key].copy()
        gap_id = gaps[i]  # Use the gap identifier (e.g., 'a', 'b', 'c')

        # Relabel nodes with gap-based naming
        start = node_labels[i]
        end = node_labels[i + 1]
        relabeled = relabel_subnetwork_nodes_with_gap(subnet, start, end, gap_id)

        # Rename resistors to ensure uniqueness
        resistor_mapping = {}
        for old_r_id in relabeled["resistors"]:
            new_r_id = f"R{resistor_counter}"
            resistor_mapping[old_r_id] = new_r_id
            all_resistors[new_r_id] = (
                relabeled["resistors"][old_r_id] * resistance_scale
            )
            resistor_counter += 1

        # Update connections with new resistor IDs
        all_connections.extend(
            [
                [conn[0], conn[1], resistor_mapping[conn[2]]]
                for conn in relabeled["connections"]
            ]
        )

        # Store relabeled subnetwork
        relabeled["resistors"] = {
            resistor_mapping[k]: v * resistance_scale
            for k, v in relabeled["resistors"].items()
        }
        relabeled["connections"] = [
            [conn[0], conn[1], resistor_mapping[conn[2]]]
            for conn in relabeled["connections"]
        ]
        relabeled_subnetworks.append(relabeled)

    complete_network = {"resistors": all_resistors, "connections": all_connections}

    return complete_network, relabeled_subnetworks


# ============================================================================
# MEASUREMENT GENERATION
# ============================================================================


def generate_measurements(topology: dict, tolerance: float = 0.1) -> list[dict]:
    """
    Generate all pairwise resistance measurements for a network.

    Args:
        topology: Network topology
        tolerance: Measurement tolerance (not used in generation, just for reference)

    Returns:
        List of measurement dictionaries
    """
    # Get all unique nodes
    logger.info("tolerance not used", tolerance)
    nodes = set()
    for conn in topology["connections"]:
        nodes.add(conn[0])
        nodes.add(conn[1])

    node_list = sorted(nodes)
    measurements = []

    # Generate all pairwise measurements
    for node_a, node_b in combinations(node_list, 2):
        try:
            resistance = simulate_circuit_resistance(topology, [node_a, node_b])
            measurements.append(
                {"node_a": node_a, "node_b": node_b, "resistance": round(resistance, 3)}
            )
        except Exception as e:
            logger.info(f"Warning: Could not measure {node_a}-{node_b}: {e}")

    return measurements


# ============================================================================
# TASK GENERATION
# ============================================================================


def generate_task(
    task_id: str,
    backbone_key: str,
    subnetwork_keys: list[str],
    resistance_scale: float = 1.0,
    tolerance: float = 0.1,
) -> dict:
    """
    Generate a complete main task.

    Args:
        task_id: Unique task identifier
        backbone_key: Backbone to use
        subnetwork_keys: Subnetworks to insert
        resistance_scale: Scale resistor values
        tolerance: Measurement tolerance

    Returns:
        Complete task definition
    """
    # Assemble network
    complete_network, subnetworks = assemble_network(
        backbone_key, subnetwork_keys, resistance_scale
    )

    # Generate measurements
    measurements = generate_measurements(complete_network, tolerance)

    # Count total resistors
    _num_resistors = len(complete_network["resistors"])

    # Create task
    return {
        task_id: {
            "name": f"{task_id}",  # Resistor Network: {BACKBONE_LIBRARY[backbone_key]['name']}",
            "description": "Infer the circuit topology and resistor values from node-to-node resistance measurements.",
            "tools": [
                "calculate_series_resistance",
                "calculate_parallel_resistance",
                "delta_to_wye_transform",
                "wye_to_delta_transform",
                "simulate_circuit_resistance",
                "validate_measurements",
            ],
            "scoring_function": "resistor_topology",
            "scoring_params": {
                "expected_topology": complete_network,
                "expected_measurements": measurements,
                "tolerance": tolerance,
                "use_functional_scoring": True,
                "topology_weight": 0.0,
                "functional_weight": 1,
                "exact_values_weight": 0.0,
            },
            "submission_format": 'JSON string with circuit topology (e.g., {"resistors": {"R1": x, ...}, "connections": [["A","B","R1"], ...]})',
            "input_from_tasks": [],
            "initial_input": {
                "measurements": measurements,
                "notes": [
                    # f"There are {num_resistors} resistors total",
                    # f"Network uses backbone: {BACKBONE_LIBRARY[backbone_key]['name']}",
                    f"Assume ideal resistors; treat measurements as exact within ±{tolerance} ohm.",
                    "The resistances are labeled from left to right as R1, R2, R3, ..., etc.",
                ],
            },
        }
    }


def generate_subtasks(
    task_id: str,
    backbone_key: str,
    subnetwork_keys: list[str],
    resistance_scale: float = 1.0,
    tolerance: float = 0.1,
) -> dict:
    """
    Generate subtasks for each subnetwork and final assembly.

    Args:
        task_id: Base task identifier
        backbone_key: Backbone to use
        subnetwork_keys: Subnetworks to insert
        resistance_scale: Scale resistor values
        tolerance: Measurement tolerance

    Returns:
        Dictionary of subtask definitions
    """
    # Assemble network
    complete_network, subnetworks = assemble_network(
        backbone_key, subnetwork_keys, resistance_scale
    )

    subtasks = {}

    # Create subtask for each subnetwork
    for i, subnet in enumerate(subnetworks):
        subnet_id = f"{task_id}_subnet_{i+1}"
        subnet_measurements = generate_measurements(subnet, tolerance)

        subtasks[subnet_id] = {
            "name": f"{task_id}_subnet_{i+1}",  # f"Subnetwork {i+1}: {subnet['name']}",
            "description": f"Identify topology of subnetwork {i+1} between nodes {subnet['terminals'][0]} and {subnet['terminals'][1]}.",
            "tools": [
                "calculate_series_resistance",
                "calculate_parallel_resistance",
                "delta_to_wye_transform",
                "wye_to_delta_transform",
                "simulate_circuit_resistance",
                "validate_measurements",
            ],
            "scoring_function": "resistor_topology",
            "scoring_params": {
                "expected_topology": subnet,
                "expected_measurements": subnet_measurements,
                "tolerance": tolerance,
                "use_functional_scoring": True,
                "topology_weight": 0.0,
                "functional_weight": 1.0,
                "exact_values_weight": 0.0,
            },
            "submission_format": 'JSON string with circuit topology (e.g., {"resistors": {"R1": x, ...}, "connections": [["A","B","R1"], ...]})',
            "input_from_tasks": [],
            "initial_input": {
                "measurements": subnet_measurements,
                "notes": [
                    f"This is subnetwork {i+1} of the complete circuit",
                    f"Terminals: {subnet['terminals'][0]} to {subnet['terminals'][1]}",
                    SUBNETWORK_LIBRARY[subnetwork_keys[i]].get("description", ""),
                    "The resistances are labeled from left to right as R1, R2, R3, ..., etc.",
                ],
            },
        }

    # Create final assembly subtask
    all_measurements = generate_measurements(complete_network, tolerance)
    assembly_id = f"{task_id}_final_assembly"

    subtasks[assembly_id] = {
        "name": "Final Network Assembly",
        "description": "Combine all subnetworks to form the complete resistor network topology.",
        "tools": [
            "calculate_series_resistance",
            "calculate_parallel_resistance",
            "simulate_circuit_resistance",
            "validate_measurements",
        ],
        "scoring_function": "resistor_topology",
        "scoring_params": {
            "expected_topology": complete_network,
            "expected_measurements": all_measurements,
            "tolerance": tolerance,
            "use_functional_scoring": True,
            "topology_weight": 0.0,
            "functional_weight": 1.0,
            "exact_values_weight": 0.0,
        },
        "submission_format": 'JSON string with circuit topology (e.g., {"resistors": {"R1": x, ...}, "connections": [["A","B","R1"], ...]})',
        "input_from_tasks": [
            f"{task_id}_subnet_{i+1}" for i in range(len(subnetworks))
        ],
        "initial_input": {
            "measurements": all_measurements,
            "notes": [
                f"Combine all {len(subnetworks)} subnetworks",
                f"Total resistors: {len(complete_network['resistors'])}",
            ],
        },
    }

    return subtasks


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_random_task(
    task_id: str = "random_task", num_subnetworks: int = 3
) -> tuple[dict, dict]:
    """
    Create a random task by sampling backbone and subnetworks.

    Args:
        task_id: Task identifier
        num_subnetworks: Number of subnetworks (determines backbone)

    Returns:
        (task, subtasks) - Main task and subtask definitions
    """
    # Select appropriate backbone
    backbones = {
        k: v for k, v in BACKBONE_LIBRARY.items() if len(v["gaps"]) == num_subnetworks
    }

    if not backbones:
        raise ValueError(f"No backbone with {num_subnetworks} gaps available")

    backbone_key = random.choice(list(backbones.keys()))

    # Sample subnetworks
    subnetwork_keys = random.choices(list(SUBNETWORK_LIBRARY.keys()), k=num_subnetworks)

    # Generate task and subtasks
    task = generate_task(task_id, backbone_key, subnetwork_keys)
    subtasks = generate_subtasks(task_id, backbone_key, subnetwork_keys)

    return task, subtasks


def save_tasks_to_file(
    task: dict, subtasks: dict, filename: str = "resistor_tasks.json"
):
    """Save tasks to JSON file."""
    output = {"main_task": task, "subtasks": subtasks}

    with Path(filename).open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Tasks saved to {filename}")


def save_single_task_to_file(data, filename: str):
    """Save a single dictionary or list to a JSON file."""
    with Path(filename).open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Data saved to {filename}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# if __name__ == "__main__":
#     # Example 1: Create specific task
#     logger.info("=" * 70)
#     logger.info("Example 1: Creating specific task")
#     logger.info("=" * 70)

#     # task = generate_task(
#     #     task_id="circuit_linear_3",
#     #     backbone_key="parallel_branch",
#     #     subnetwork_keys=["series_2r", "parallel_2r"],
#     #     resistance_scale=1.0,
#     # )

#     # subtasks = generate_subtasks(
#     #     task_id="circuit_linear_3",
#     #     backbone_key="parallel_branch",
#     #     subnetwork_keys=["series_2r", "parallel_2r"],
#     # )

#     # logger.info(json.dumps(task, indent=2))
#     # logger.info("\nNumber of subtasks:", len(subtasks))

#     # # save_single_task_to_file(task, "example_main_task.json")
#     # # save_single_task_to_file(subtasks, "example_subtasks.json")
#     # # save to a json file
#     # save_tasks_to_file(task, subtasks, "example_resistor_tasks.json")

#     # Example 2: Create random task
#     # logger.info("\n" + "=" * 70)
#     # logger.info("Example 2: Creating random task")
#     # logger.info("=" * 70)

#     counter = 0
#     for sub in [2, 3, 4]:
#         for _ in range(2):
#             logger.info(
#                 f"\nCreating random task with {sub} subnetworks (trial {counter})"
#             )

#             # These lines need to be INSIDE the inner loop
#             random_task, random_subtasks = create_random_task(
#                 f"task_{counter}", num_subnetworks=sub
#             )
#             logger.info(f"Created task with {len(random_subtasks)} subtasks")
#             save_tasks_to_file(random_task, random_subtasks, f"task_{counter}.json")
#             save_single_task_to_file(random_task, f"task_{counter}_main.json")
#             save_single_task_to_file(random_subtasks, f"task_{counter}_subtasks.json")

#             counter += 1


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Generating all tasks and subtasks")
    logger.info("=" * 70)

    # Initialize collections for all tasks and subtasks
    all_main_tasks = {}
    all_subtasks = {}

    counter = 0
    for sub in [2, 3, 4]:
        for _ in range(2):
            logger.info(
                f"\nCreating random task with {sub} subnetworks (trial {counter})"
            )

            # Generate task and subtasks
            random_task, random_subtasks = create_random_task(
                f"task_{counter}", num_subnetworks=sub
            )
            logger.info(f"Created task with {len(random_subtasks)} subtasks")

            # Add to collections
            # Extract the actual task dict (it's nested under task_id)
            task_id = f"task_{counter}"
            all_main_tasks[task_id] = random_task[task_id]
            save_single_task_to_file(random_task, f"task_{counter}_main.json")

            # Add all subtasks from this task
            all_subtasks.update(random_subtasks)

            counter += 1

    # Save all tasks to single files
    logger.info("\n" + "=" * 70)
    logger.info("Saving all tasks to files")
    logger.info("=" * 70)

    save_single_task_to_file(all_main_tasks, "all_main_tasks.json")
    save_single_task_to_file(all_subtasks, "all_subtasks.json")

    logger.info(f"\nTotal main tasks: {len(all_main_tasks)}")
    logger.info(f"Total subtasks: {len(all_subtasks)}")
