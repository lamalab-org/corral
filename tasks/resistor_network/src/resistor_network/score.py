import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from corral.utils.tool_helpers import smart_resolve_path

if "CORRAL_WORK_DIR" not in os.environ:
    raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


def check_resistor_topology(
    expected_topology: dict[str, Any], tolerance: float = 0.1, require_both: bool = True
) -> Callable[[str], float]:
    """
    Returns a scoring function that checks if proposed topology matches expected one.

    Args:
        expected_topology: dict containing the correct circuit topology and resistor values
        tolerance: Relative tolerance for resistor value comparison (default 10%)

    Returns:
        Scoring function that takes a topology JSON string/path and returns score 0.0-1.0
    """
    logger.info(f"Creating topology checker with tolerance {tolerance}")

    def score_fn(topology_input: str) -> float:
        try:
            logger.info(f"check_resistor_topology: input={topology_input!r}")

            # Try to resolve as path first
            resolved_input = smart_resolve_path(topology_input.strip())
            logger.info(f"check_resistor_topology: resolved={resolved_input!r}")

            # Load topology data
            topology_data = None
            if Path(resolved_input).exists():
                with Path(resolved_input).open() as f:
                    topology_data = json.load(f)
                logger.info("Loaded topology from file")
            else:
                # Try parsing as JSON string
                try:
                    topology_data = json.loads(topology_input)
                    logger.info("Parsed topology from JSON string")
                except json.JSONDecodeError:
                    topology_data = json.loads(resolved_input)
                    logger.info("Parsed topology from resolved string")

            if not topology_data:
                logger.error("No topology data found")
                return 0.0

            # Check if required keys exist
            if "resistors" not in topology_data or "connections" not in topology_data:
                logger.error("Missing required keys: resistors or connections")
                return 0.0

            proposed_resistors = topology_data["resistors"]
            proposed_connections = topology_data["connections"]
            expected_resistors = expected_topology["resistors"]
            expected_connections = expected_topology["connections"]

            # Score topology structure (connections)
            topology_score = _score_topology_structure(
                proposed_connections, expected_connections
            )
            logger.info(f"Topology structure score: {topology_score}")

            # Score resistor values
            resistor_score = _score_resistor_values(
                proposed_resistors, expected_resistors, tolerance
            )
            logger.info(f"Resistor values score: {resistor_score}")

            if require_both:
                # Both must be perfect
                return 1.0 if (topology_score == 1.0 and resistor_score == 1.0) else 0.0
            else:
                # Either being perfect is enough
                return 1.0 if (topology_score == 1.0 or resistor_score == 1.0) else 0.0

        except Exception:
            return 0.0

    return score_fn


def _score_topology_structure(proposed: list, expected: list) -> float:
    """Score how well the proposed connections match expected ones.
    Binary scoring: 1.0 if exact match, 0.0 otherwise
    """
    if len(proposed) != len(expected):
        return 0.0

    # Normalize connections for comparison (order shouldn't matter)
    def normalize_connection(conn):
        return (*sorted([conn[0], conn[1]]), conn[2])

    proposed_normalized = {normalize_connection(conn) for conn in proposed}
    expected_normalized = {normalize_connection(conn) for conn in expected}

    # Check for exact match
    if proposed_normalized == expected_normalized:
        return 1.0

    overlap = len(proposed_normalized & expected_normalized)
    total = len(expected_normalized)
    logger.info(f"Partial topology match: {overlap}/{total}")
    return 0


def _score_resistor_values(proposed: dict, expected: dict, tolerance: float) -> float:
    """Score how well proposed resistor values match expected ones.
    Binary scoring: 1.0 if ALL resistors within tolerance, 0.0 otherwise
    """
    if set(proposed.keys()) != set(expected.keys()):
        return 0.0  # Must have same resistor names

    for resistor_id, expected_val in expected.items():
        proposed_val = proposed[resistor_id]

        if expected_val == 0:
            if proposed_val != 0:
                return 0.0  # Fail immediately if any resistor wrong
        else:
            relative_error = abs(proposed_val - expected_val) / expected_val
            if relative_error > tolerance:
                return 0.0  # Fail immediately if any resistor wrong

    return 1.0


def check_resistance_measurements(
    expected_measurements: list[dict[str, Any]], tolerance: float = 0.05
) -> Callable[[str], float]:
    """
    Returns a scoring function that validates a topology against expected measurements.

    Args:
        expected_measurements: list of measurement dicts with node_a, node_b, resistance
        tolerance: Relative tolerance for resistance comparison (default 5%)

    Returns:
        Scoring function that takes a topology and returns measurement match score 0.0-1.0
    """
    logger.info(
        f"Creating measurement checker with {len(expected_measurements)} measurements"
    )

    def score_fn(topology_input: str) -> float:
        try:
            logger.info(f"check_resistance_measurements: input={topology_input!r}")

            # Load topology
            resolved_input = smart_resolve_path(topology_input.strip())
            topology_data = None

            if Path(resolved_input).exists():
                with Path(resolved_input).open() as f:
                    topology_data = json.load(f)
            else:
                try:
                    topology_data = json.loads(topology_input)
                except json.JSONDecodeError:
                    topology_data = json.loads(resolved_input)

            if not topology_data:
                return 0.0

            scores = []

            for measurement in expected_measurements:
                try:
                    # This would call the actual circuit simulation
                    predicted_resistance = _simulate_resistance(
                        topology_data, measurement["node_a"], measurement["node_b"]
                    )

                    expected_resistance = measurement["resistance"]

                    if expected_resistance == 0:
                        score = 1.0 if abs(predicted_resistance) < 1e-6 else 0.0
                    else:
                        relative_error = (
                            abs(predicted_resistance - expected_resistance)
                            / expected_resistance
                        )
                        score = max(0.0, 1.0 - relative_error / tolerance)

                    scores.append(score)
                    logger.info(
                        f"Measurement {measurement['node_a']}-{measurement['node_b']}: "
                        f"expected={expected_resistance}, predicted={predicted_resistance}, score={score}"
                    )

                except Exception as e:
                    logger.error(f"Failed to simulate measurement: {e}")
                    scores.append(0.0)

            final_score = sum(scores) / len(scores) if scores else 0.0
            logger.info(f"Overall measurement score: {final_score}")
            return final_score

        except Exception as e:
            logger.error(f"Error in measurement checking: {e}", exc_info=True)
            return 0.0

    return score_fn


def _simulate_resistance(topology: dict, node_a: str, node_b: str) -> float:
    """
    Simulate resistance between two nodes in a topology.
    """
    try:
        resistors = topology["resistors"]
        connections = topology["connections"]

        # Build adjacency matrix for nodal analysis
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
            resistance = resistors[resistor_id]
            conductance = 1.0 / resistance
            i, j = node_to_idx[node1], node_to_idx[node2]

            G[i, i] += conductance
            G[j, j] += conductance
            G[i, j] -= conductance
            G[j, i] -= conductance

        # Solve for resistance between nodes
        term1_idx = node_to_idx[node_a]
        term2_idx = node_to_idx[node_b]

        # Apply 1A current and solve for voltage
        Ia = np.zeros(n)
        Ia[term1_idx] = 1.0
        Ia[term2_idx] = -1.0

        # Remove reference equation
        G_reduced = (
            G[:-1, :-1]
            if term2_idx == n - 1
            else np.delete(np.delete(G, term2_idx, 0), term2_idx, 1)
        )
        I_reduced = Ia[:-1] if term2_idx == n - 1 else np.delete(Ia, term2_idx)

        V_reduced = np.linalg.solve(G_reduced, I_reduced)

        # Insert reference voltage
        if term2_idx == n - 1:
            Va = np.append(V_reduced, 0)
        else:
            Va = np.insert(V_reduced, term2_idx, 0)

        return abs(Va[term1_idx] - Va[term2_idx])

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return float("inf")


def check_complete_circuit_solution(
    expected_topology: dict[str, Any],
    expected_measurements: list[dict[str, Any]],
    topology_weight: float = 0.6,
    measurement_weight: float = 0.4,
    tolerance: float = 0.1,
) -> Callable[[str], float]:
    """
    Returns a comprehensive scoring function that checks both topology and measurements.
    """
    logger.info("Creating complete circuit solution checker")

    topology_scorer = check_resistor_topology(expected_topology, tolerance)
    measurement_scorer = check_resistance_measurements(expected_measurements, tolerance)

    def score_fn(solution_input: str) -> float:
        try:
            topology_score = topology_scorer(solution_input)
            measurement_score = measurement_scorer(solution_input)

            final_score = (
                topology_weight * topology_score
                + measurement_weight * measurement_score
            )

            logger.info(
                f"Complete solution - topology: {topology_score}, "
                f"measurements: {measurement_score}, final: {final_score}"
            )

            return final_score

        except Exception as e:
            logger.error(f"Error in complete solution checking: {e}", exc_info=True)
            return 0.0

    return score_fn


def check_resistor_values_only(
    expected_values: dict[str, float], tolerance: float = 0.1
) -> Callable[[str], float]:
    """
    Returns a scoring function that only checks if resistor values are correct.
    """
    logger.info(
        f"Creating resistor values checker for {len(expected_values)} resistors"
    )

    def score_fn(values_input: str) -> float:
        try:
            logger.info(f"check_resistor_values_only: input={values_input!r}")

            # Try to resolve and load values
            resolved_input = smart_resolve_path(values_input.strip())
            values_data = None

            if Path(resolved_input).exists():
                with Path(resolved_input).open() as f:
                    data = json.load(f)
                    # Extract resistor values if it's a full topology
                    values_data = data.get("resistors", data)
            else:
                try:
                    data = json.loads(values_input)
                    values_data = data.get("resistors", data)
                except json.JSONDecodeError:
                    data = json.loads(resolved_input)
                    values_data = data.get("resistors", data)

            if not values_data:
                return 0.0

            score = _score_resistor_values(values_data, expected_values, tolerance)
            logger.info(f"Resistor values score: {score}")
            return score

        except Exception as e:
            logger.error(f"Error checking resistor values: {e}", exc_info=True)
            return 0.0

    return score_fn


def check_valid_circuit_json(json_path: str) -> float:
    """
    Check if a valid circuit topology JSON file exists at the given path.
    """
    try:
        json_path = json_path.strip()
        if not json_path:
            logger.warning("Empty path provided to check_valid_circuit_json")
            return 0.0

        if not Path(json_path).exists():
            logger.info(f"Circuit JSON file not found at: {json_path}")
            return 0.0

        if not Path(json_path).is_file():
            logger.info(f"Path exists but is not a file: {json_path}")
            return 0.0

        with Path(json_path).open("r", encoding="utf-8") as f:
            circuit_data = json.load(f)

        # Validate circuit structure
        if not isinstance(circuit_data, dict):
            logger.info("Circuit JSON is not a dictionary")
            return 0.0

        required_keys = ["resistors", "connections"]
        if not all(key in circuit_data for key in required_keys):
            logger.info(f"Circuit JSON missing required keys: {required_keys}")
            return 0.0

        # Basic validation of resistors
        resistors = circuit_data["resistors"]
        if not isinstance(resistors, dict) or not resistors:
            logger.info("Invalid or empty resistors section")
            return 0.0

        # Basic validation of connections
        connections = circuit_data["connections"]
        if not isinstance(connections, list) or not connections:
            logger.info("Invalid or empty connections section")
            return 0.0

        # Check connection format
        for conn in connections:
            if not isinstance(conn, list) or len(conn) != 3:
                logger.info(
                    "Invalid connection format - should be [node1, node2, resistor_id]"
                )
                return 0.0

        logger.info(
            f"Valid circuit JSON with {len(resistors)} resistors and {len(connections)} connections"
        )
        return 1.0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in file {json_path}: {e}")
        return 0.0
    except Exception as e:
        logger.error(
            f"Error validating circuit JSON file {json_path}: {e}", exc_info=True
        )
        return 0.0
