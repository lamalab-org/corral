import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from resistor_network.utils import get_resistance_between_nodes

from corral.utils.tool_helpers import smart_resolve_path


def check_resistor_topology(
    expected_topology: dict[str, Any],
    use_functional_scoring: bool,
    topology_weight: float,
    functional_weight: float,
    exact_values_weight: float,
    tolerance: float = 0.1,
    require_both: bool = True,
    expected_measurements: list[dict[str, Any]] | None = None,
) -> Callable[[str], float]:
    """
    Enhanced scoring function that checks:
    1. Topology structure (connections)
    2. Functional behavior (does it produce expected measurements?)
    3. Optionally: exact resistor values

    Args:
        expected_topology: Expected circuit topology (required)
        use_functional_scoring: Whether to use functional validation (required)
        topology_weight: Weight for topology structure score (required, set 0.0 to disable)
        functional_weight: Weight for functional behavior score (required, set 0.0 to disable)
        exact_values_weight: Weight for exact resistor values score (required, set 0.0 to disable)
        tolerance: Tolerance for measurements and resistor values (default: 0.1)
        require_both: Legacy parameter - ignored when use_functional_scoring=True (default: True)
        expected_measurements: List of expected resistance measurements (required if functional_weight > 0)

    Notes:
        - At least one weight must be > 0
        - If functional_weight > 0, expected_measurements must be provided
        - Weights are normalized automatically in weighted scoring mode

    Example configurations:
        # Pure functional scoring (for subtasks with arbitrary resistor names):
        use_functional_scoring=True, topology_weight=0.0, functional_weight=1.0, exact_values_weight=0.0

        # Functional + topology (for main tasks):
        use_functional_scoring=True, topology_weight=0.5, functional_weight=0.5, exact_values_weight=0.0

        # Original strict mode (backward compatible):
        use_functional_scoring=False, topology_weight=0.5, functional_weight=0.0, exact_values_weight=0.5
    """
    # Validate configuration
    if topology_weight < 0 or functional_weight < 0 or exact_values_weight < 0:
        raise ValueError("All weights must be non-negative")

    if topology_weight == 0 and functional_weight == 0 and exact_values_weight == 0:
        raise ValueError(
            "At least one weight must be > 0. "
            "Set topology_weight, functional_weight, or exact_values_weight to enable scoring."
        )

    if functional_weight > 0 and not expected_measurements:
        raise ValueError(
            "expected_measurements must be provided when functional_weight > 0"
        )

    logger.info(
        f"Creating enhanced topology checker - functional: {use_functional_scoring}, "
        f"weights(topology={topology_weight}, functional={functional_weight}, exact_values={exact_values_weight})"
    )

    def score_fn(topology_input: str) -> float:
        try:
            logger.info(f"ENHANCED SCORING INPUT: {topology_input!r}")
            logger.info(f"EXPECTED TOPOLOGY: {expected_topology}")

            # Parse topology (same logic as before)
            topology_data = None
            input_stripped = topology_input.strip()

            if input_stripped.startswith("{") and input_stripped.endswith("}"):
                try:
                    topology_data = json.loads(input_stripped)
                    logger.info("Parsed topology from direct JSON string")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse as direct JSON: {e}")
                    return 0.0

            if (
                not topology_data
                or "resistors" not in topology_data
                or "connections" not in topology_data
            ):
                logger.error("Invalid topology data")
                return 0.0

            proposed_resistors = topology_data["resistors"]
            proposed_connections = topology_data["connections"]
            if not isinstance(proposed_resistors, dict) or not isinstance(
                proposed_connections, list
            ):
                return 0.0

            # Do not let the simulator silently ignore declared or undefined
            # components. This also makes the complexity floor meaningful.
            referenced = set()
            for connection in proposed_connections:
                if not isinstance(connection, list | tuple) or len(connection) != 3:
                    return 0.0
                node_a, node_b, resistor_id = connection
                if not isinstance(node_a, str) or not isinstance(node_b, str):
                    return 0.0
                if resistor_id not in proposed_resistors:
                    return 0.0
                if resistor_id in referenced:
                    return 0.0
                referenced.add(resistor_id)
            if referenced != set(proposed_resistors):
                return 0.0
            expected_resistors = expected_topology["resistors"]
            expected_connections = expected_topology["connections"]

            logger.info(f"PROPOSED TOPOLOGY: {topology_data}")

            # 1. Score topology structure (connections)
            topology_score = _score_topology_structure(
                proposed_connections, expected_connections
            )
            logger.info(f"Topology structure score: {topology_score}")

            scores = {"topology": topology_score}
            weights = {"topology": topology_weight}

            logger.info(
                f"Function parameters - use_functional_scoring: {use_functional_scoring}, functional_weight: {functional_weight}, exact_values_weight: {exact_values_weight}"
            )

            # 2. Score functional behavior
            if (
                use_functional_scoring
                and expected_measurements
                and functional_weight > 0
            ):
                functional_score = _score_functional_behavior(
                    topology_data, expected_measurements, tolerance
                )
                scores["functional"] = functional_score
                weights["functional"] = functional_weight
                logger.info(f"Functional behavior score: {functional_score}")

            # 3. Score exact resistor values (should be enabled by default for backward compatibility)
            if exact_values_weight > 0:
                exact_values_score = _score_resistor_values(
                    proposed_resistors, expected_resistors, tolerance
                )
                scores["exact_values"] = exact_values_score
                weights["exact_values"] = exact_values_weight
                logger.info(f"Exact values score: {exact_values_score}")
            else:
                logger.info(
                    f"Exact values scoring disabled (weight={exact_values_weight})"
                )

            # Calculate final score based on require_both setting
            if use_functional_scoring:
                # New behavior: use weighted or require_both logic for functional scoring
                if False:  # Replace with actual condition
                    # All enabled components must be perfect (score = 1.0)
                    required_components = [
                        component for component, weight in weights.items() if weight > 0
                    ]
                    all_perfect = all(
                        scores[component] == 1.0 for component in required_components
                    )
                    final_score = 1.0 if all_perfect else 0.0
                    logger.info(
                        f"FUNCTIONAL REQUIRE_BOTH=True: All components perfect? {all_perfect}"
                    )
                else:
                    # Use weighted scoring for functional mode
                    total_weight = sum(weights.values())
                    if total_weight == 0:
                        logger.error("No scoring components enabled")
                        return 0.0

                    final_score = (
                        sum(
                            scores[component] * weight
                            for component, weight in weights.items()
                        )
                        / total_weight
                    )
                    logger.info("FUNCTIONAL REQUIRE_BOTH=False: Using weighted scoring")
            else:
                # Original behavior: binary logic for backward compatibility
                topology_score = scores.get("topology", 0.0)
                resistor_score = scores.get("exact_values", 0.0)

                if require_both:
                    # Both topology AND resistors must be perfect
                    final_score = (
                        1.0
                        if (topology_score == 1.0 and resistor_score == 1.0)
                        else 0.0
                    )
                    logger.info(
                        f"ORIGINAL REQUIRE_BOTH=True: topology={topology_score}, resistors={resistor_score}, result={final_score}"
                    )
                else:
                    # Either topology OR resistors being perfect is enough
                    final_score = (
                        1.0 if (topology_score == 1.0 or resistor_score == 1.0) else 0.0
                    )
                    logger.info(
                        f"ORIGINAL REQUIRE_BOTH=False: topology={topology_score}, resistors={resistor_score}, result={final_score}"
                    )

            logger.info(f"COMPONENT SCORES: {scores}")
            logger.info(f"WEIGHTS: {weights}")
            logger.info(f"FINAL SCORE: {final_score}")

            return final_score

        except Exception as e:
            logger.error(f"ENHANCED SCORING ERROR: {e}", exc_info=True)
            return 0.0

    return score_fn


def _score_functional_behavior(
    topology_data: dict, expected_measurements: list[dict], tolerance: float
) -> float:
    """
    Score how well the topology functionally matches expected resistance measurements.

    Binary version:
    - Each measurement: if predicted resistance is within tolerance → 1, else → 0
    - Final score: 1.0 only if ALL measurements pass, otherwise 0.0
    """
    if not expected_measurements:
        logger.warning("No expected measurements provided for functional scoring")
        return 1.0  # Default to success if no measurements to check

    scores = []

    for measurement in expected_measurements:
        try:
            node_a = measurement["node_a"]
            node_b = measurement["node_b"]
            expected_resistance = measurement["resistance"]

            predicted_resistance = _simulate_resistance(topology_data, node_a, node_b)

            relative_error = 0.0
            if expected_resistance == 0:
                score = 1.0 if abs(predicted_resistance) < 1e-6 else 0.0
            else:
                relative_error = (
                    abs(predicted_resistance - expected_resistance)
                    / expected_resistance
                )
                score = 1.0 if relative_error <= tolerance else 0.0

            scores.append(score)
            logger.info(
                f"Functional test {node_a}-{node_b}: expected={expected_resistance}, "
                f"predicted={predicted_resistance}, relative_error={relative_error:.4f}, score={score}"
            )

        except Exception as e:
            logger.error(f"Failed to test measurement {measurement}: {e}")
            scores.append(0.0)

    # All measurements must pass for a score of 1.0
    final_functional_score = 1.0 if (scores and all(s == 1.0 for s in scores)) else 0.0
    logger.info(
        f"Overall functional score: {final_functional_score} (passed {sum(scores)}/{len(scores)} measurements)"
    )
    return final_functional_score


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
    Simulate resistance between two nodes in a topology, delegating to the
    canonical nodal-analysis solver (`utils.get_resistance_between_nodes`) so
    this scoring path can never silently drift from the ground-truth
    simulator or the agent-facing `simulate_circuit_resistance` tool.
    """
    try:
        return get_resistance_between_nodes(json.dumps(topology), [node_a, node_b])
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

    # Use backward-compatible mode (original strict binary scoring)
    topology_scorer = check_resistor_topology(
        expected_topology=expected_topology,
        use_functional_scoring=False,
        topology_weight=0.5,
        functional_weight=0.0,
        exact_values_weight=0.5,
        tolerance=tolerance,
        require_both=True,
    )
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


# ---------------------------------------------------------------------------
# Conductance-matrix scoring
# ---------------------------------------------------------------------------


def _parse_topology(topology_input: str) -> dict[str, Any] | None:
    """Parse and structurally validate a submitted topology.

    Returns None for anything malformed: bad JSON, missing keys, connections that
    are not `[node_a, node_b, resistor_id]`, self-loops, resistor ids that are
    undefined / reused across connections / declared but never connected, and
    non-positive or non-finite resistances. Every one of these would otherwise let
    the submission claim components the simulator silently ignores.
    """
    stripped = topology_input.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    resistors = data.get("resistors")
    connections = data.get("connections")
    if not isinstance(resistors, dict) or not resistors:
        return None
    if not isinstance(connections, list) or not connections:
        return None

    referenced: set[str] = set()
    for connection in connections:
        if not isinstance(connection, list | tuple) or len(connection) != 3:
            return None
        node_a, node_b, resistor_id = connection
        if not isinstance(node_a, str) or not isinstance(node_b, str):
            return None
        if node_a == node_b:
            return None  # self-loop carries no current
        if resistor_id not in resistors or resistor_id in referenced:
            return None
        value = resistors[resistor_id]
        if not isinstance(value, int | float) or isinstance(value, bool):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        referenced.add(resistor_id)
    if referenced != set(resistors):
        return None
    return data


def conductance_map(topology: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Merged conductance (siemens) per node pair.

    This is the canonical form of a resistor network: resistors sharing a node pair
    are in parallel, and no measurement can ever separate them -- effective
    resistance depends only on the graph Laplacian, which sees their *sum*. Two
    circuits have the same conductance map exactly when they are electrically
    indistinguishable on the given nodes.
    """
    conductances: dict[tuple[str, str], float] = {}
    for node_a, node_b, resistor_id in topology["connections"]:
        pair = (node_a, node_b) if node_a < node_b else (node_b, node_a)
        conductances[pair] = (
            conductances.get(pair, 0.0) + 1.0 / (topology["resistors"][resistor_id])
        )
    return conductances


def check_conductance_topology(
    expected_topology: dict[str, Any], tolerance: float = 0.1
) -> Callable[[str], float]:
    """Score a submitted topology against the expected *conductance map*.

    Scoring the conductance map rather than the resistance measurements grades the
    thing that is actually identifiable from the data, and nothing else:

    - Resistor ids, ordering, and how many physical resistors sit on a node pair are
      all free. Submitting `47 ohm || 33 ohm` or the equivalent single `19.4 ohm` both
      score 1.0, because parallel resistors are provably indistinguishable.
    - The node set and the set of connected node pairs must match exactly. A
      submission cannot invent a connection (a near-open 80k-ohm resistor across a pair
      the real circuit leaves unconnected is still a claim that a component is
      there), nor drop the only resistor on a pair.
    - Each pair's merged conductance must agree within `tolerance` relative error.

    Comparing conductances is strictly tighter than comparing simulated
    measurements, which accumulate slack through the solve and let materially
    different circuits pass.
    """
    expected_conductances = conductance_map(expected_topology)
    expected_nodes = {node for pair in expected_conductances for node in pair}

    def score_fn(topology_input: str) -> float:
        try:
            proposed = _parse_topology(topology_input)
            if proposed is None:
                logger.info("Conductance scoring: submission failed structural checks")
                return 0.0

            proposed_conductances = conductance_map(proposed)
            proposed_nodes = {node for pair in proposed_conductances for node in pair}

            if proposed_nodes != expected_nodes:
                logger.info(
                    "Conductance scoring: node set mismatch "
                    f"(missing={sorted(expected_nodes - proposed_nodes)}, "
                    f"invented={sorted(proposed_nodes - expected_nodes)})"
                )
                return 0.0

            if set(proposed_conductances) != set(expected_conductances):
                logger.info(
                    "Conductance scoring: connection set mismatch "
                    f"(missing={sorted(set(expected_conductances) - set(proposed_conductances))}, "
                    f"extra={sorted(set(proposed_conductances) - set(expected_conductances))})"
                )
                return 0.0

            for pair, expected_g in expected_conductances.items():
                proposed_g = proposed_conductances[pair]
                relative_error = abs(proposed_g - expected_g) / expected_g
                if relative_error > tolerance:
                    logger.info(
                        f"Conductance scoring: {pair} expected {expected_g:.6g} S, "
                        f"got {proposed_g:.6g} S (rel err {relative_error:.4f})"
                    )
                    return 0.0

            logger.info("Conductance scoring: exact match on every node pair")
            return 1.0

        except Exception as e:
            logger.error(f"CONDUCTANCE SCORING ERROR: {e}", exc_info=True)
            return 0.0

    return score_fn
