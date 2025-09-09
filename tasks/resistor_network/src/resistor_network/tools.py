import itertools
import json
from dataclasses import dataclass

import numpy as np

from corral.backend.tool import tool


@dataclass
class CircuitMeasurement:
    """Represents a measurement between two nodes"""

    node_a: str
    node_b: str
    voltage: float | None = None
    current: float | None = None
    resistance: float | None = None


@dataclass
class CircuitTopology:
    """Represents a proposed circuit topology"""

    resistors: dict[str, float]  # resistor_id -> resistance value
    connections: list[tuple[str, str, str]]  # (node1, node2, resistor_id)


@tool
def calculate_series_resistance(resistances: list[float]) -> float:
    """[BRIEF] Calculate total resistance of resistors connected in series. [/BRIEF]

    [DETAILED] Computes the equivalent resistance when multiple resistors are connected
    end-to-end in a single path. In series configuration, current flows through each
    resistor sequentially, and the total resistance is the sum of individual resistances.
    This is fundamental for analyzing any resistor network. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have identified a series branch in a circuit
    - To simplify complex networks by combining series elements
    - As a building block for more complex resistance calculations
    - When testing hypotheses about circuit topology
    [/PROCEDURAL]

    Args:
        resistances: List of resistance values in ohms

    Returns:
        float: Total series resistance in ohms

    Example:
        calculate_series_resistance([10, 20, 30]) -> 60.0
    """
    if not resistances:
        return 0.0
    return sum(resistances)


@tool
def calculate_parallel_resistance(resistances: list[float]) -> float:
    """[BRIEF] Calculate total resistance of resistors connected in parallel. [/BRIEF]

    [DETAILED] Computes the equivalent resistance when multiple resistors are connected
    across the same two nodes. In parallel configuration, current divides among the
    resistors, and the reciprocal of total resistance equals the sum of reciprocals
    of individual resistances. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have identified parallel branches in a circuit
    - To simplify complex networks by combining parallel elements
    - When testing different topology hypotheses
    - As part of iterative network reduction
    [/PROCEDURAL]

    Args:
        resistances: List of resistance values in ohms

    Returns:
        float: Total parallel resistance in ohms

    Example:
        calculate_parallel_resistance([10, 20]) -> 6.67
    """
    if not resistances:
        return float("inf")
    if any(r <= 0 for r in resistances):
        raise ValueError("All resistances must be positive")

    reciprocal_sum = sum(1 / r for r in resistances)
    return 1 / reciprocal_sum


@tool
def delta_to_wye_transform(ra: float, rb: float, rc: float) -> dict[str, float]:
    """[BRIEF] Convert delta (triangle) resistor configuration to wye (star) configuration. [/BRIEF]

    [DETAILED] Transforms a three-resistor delta network into an equivalent three-resistor
    wye network. This is essential for solving complex resistor networks that cannot be
    reduced using simple series/parallel combinations. The transformation preserves the
    resistance between any two external nodes. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When encountering triangle configurations that block series/parallel reduction
    - As part of network analysis strategy for complex topologies
    - When testing circuit topology hypotheses involving triangular connections
    - Before applying nodal analysis to simplify the network
    [/PROCEDURAL]

    Args:
        ra: Resistance between nodes A and B in delta configuration
        rb: Resistance between nodes B and C in delta configuration
        rc: Resistance between nodes C and A in delta configuration

    Returns:
        Dict with keys 'r1', 'r2', 'r3' for wye resistor values

    Example:
        delta_to_wye_transform(30, 30, 30) -> {'r1': 10, 'r2': 10, 'r3': 10}
    """
    total = ra + rb + rc
    if total == 0:
        raise ValueError("Sum of delta resistances cannot be zero")

    r1 = (rb * rc) / total  # Connected to node A
    r2 = (ra * rc) / total  # Connected to node B
    r3 = (ra * rb) / total  # Connected to node C

    return {"r1": r1, "r2": r2, "r3": r3}


@tool
def wye_to_delta_transform(r1: float, r2: float, r3: float) -> dict[str, float]:
    """[BRIEF] Convert wye (star) resistor configuration to delta (triangle) configuration. [/BRIEF]

    [DETAILED] Transforms a three-resistor wye network into an equivalent three-resistor
    delta network. This is the inverse of delta-to-wye transformation and is useful when
    the delta form provides easier analysis or when testing different topology hypotheses. [/DETAILED]

    Args:
        r1: Wye resistor connected to node A
        r2: Wye resistor connected to node B
        r3: Wye resistor connected to node C

    Returns:
        Dict with keys 'ra', 'rb', 'rc' for delta resistor values
    """
    denominator = r1 * r2 + r2 * r3 + r3 * r1
    if any(r <= 0 for r in [r1, r2, r3]):
        raise ValueError("All resistances must be positive")

    ra = denominator / r3  # Between nodes A and B
    rb = denominator / r1  # Between nodes B and C
    rc = denominator / r2  # Between nodes C and A

    return {"ra": ra, "rb": rb, "rc": rc}


@tool
def simulate_circuit_resistance(topology: str, terminal_nodes: list[str]) -> float:
    """[BRIEF] Simulate total resistance between specified terminals in a given circuit topology. [/BRIEF]

    [DETAILED] Takes a circuit topology description and calculates the equivalent resistance
    between two terminal nodes using nodal analysis. This tool allows testing of topology
    hypotheses by comparing calculated resistance with measured values. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - To validate a proposed circuit topology against measurements
    - After constructing a hypothesis about resistor values and connections
    - For iterative testing of different network configurations
    - As verification step before final answer submission
    [/PROCEDURAL]

    Args:
        topology: JSON string describing circuit with format:
                 {"resistors": {"R1": 10, "R2": 20},
                  "connections": [["node1", "node2", "R1"], ["node2", "node3", "R2"]]}
        terminal_nodes: List of two node names to measure resistance between

    Returns:
        float: Equivalent resistance between terminals in ohms

    Example:
        topology = '{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"], ["B", "C", "R2"]]}'
        simulate_circuit_resistance(topology, ["A", "C"]) -> 30.0
    """
    try:
        circuit = json.loads(topology)
        resistors = circuit["resistors"]
        connections = circuit["connections"]

        if len(terminal_nodes) != 2:
            raise ValueError("Must specify exactly two terminal nodes")

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
            if resistor_id not in resistors:
                raise ValueError(f"Resistor {resistor_id} not found in resistor list")

            resistance = resistors[resistor_id]
            if resistance <= 0:
                raise ValueError(f"Resistance must be positive, got {resistance}")

            conductance = 1.0 / resistance
            i, j = node_to_idx[node1], node_to_idx[node2]

            G[i, i] += conductance
            G[j, j] += conductance
            G[i, j] -= conductance
            G[j, i] -= conductance

        # Solve for resistance between terminal nodes
        term1_idx = node_to_idx[terminal_nodes[0]]
        term2_idx = node_to_idx[terminal_nodes[1]]

        # Apply 1A current between terminals and solve for voltage
        Ia = np.zeros(n)
        Ia[term1_idx] = 1.0
        Ia[term2_idx] = -1.0

        # Remove one equation (use term2 as reference)
        G_reduced = (
            G[:-1, :-1]
            if term2_idx == n - 1
            else np.delete(np.delete(G, term2_idx, 0), term2_idx, 1)
        )
        I_reduced = Ia[:-1] if term2_idx == n - 1 else np.delete(Ia, term2_idx)

        try:
            V_reduced = np.linalg.solve(G_reduced, I_reduced)
        except np.linalg.LinAlgError as err:
            raise ValueError("Circuit is not solvable (possibly disconnected)") from err

        # Insert reference voltage
        if term2_idx == n - 1:
            V = np.append(V_reduced, 0)
        else:
            V = np.insert(V_reduced, term2_idx, 0)

        # Resistance is voltage difference with 1A current
        return abs(V[term1_idx] - V[term2_idx])

    except Exception as e:
        raise ValueError(f"Error simulating circuit: {e!s}") from e


@tool
def validate_measurements(topology: str, measurements: str) -> dict[str, float]:
    """[BRIEF] Validate if proposed topology matches all given measurements. [/BRIEF]

    [DETAILED] Compares the resistance/voltage/current predictions of a proposed circuit
    topology against actual measurements. Returns error metrics to assess how well the
    proposed solution matches the experimental data. Essential for validating hypotheses. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - After proposing a complete circuit topology and resistor values
    - To quantify how well your hypothesis matches the measurements
    - Before final submission to check solution quality
    - During iterative refinement of topology hypotheses
    [/PROCEDURAL]

    Args:
        topology: JSON string describing proposed circuit topology
        measurements: JSON string with actual measurements format:
                     [{"node_a": "A", "node_b": "B", "resistance": 10.0}, ...]

    Returns:
        Dict with validation results including total_error, max_error, and per_measurement errors

    Example:
        measurements = '[{"node_a": "A", "node_b": "B", "resistance": 15.0}]'
        topology = '{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"]]}'
        validate_measurements(topology, measurements) -> {"total_error": 5.0, "max_error": 5.0, ...}
    """
    try:
        measurements_data = json.loads(measurements)
        errors = []
        detailed_errors = []

        for measurement in measurements_data:
            node_a = measurement["node_a"]
            node_b = measurement["node_b"]

            try:
                predicted_resistance = simulate_circuit_resistance(
                    topology, [node_a, node_b]
                )

                if "resistance" in measurement:
                    actual_resistance = measurement["resistance"]
                    error = abs(predicted_resistance - actual_resistance)
                    errors.append(error)
                    detailed_errors.append(
                        {
                            "nodes": f"{node_a}-{node_b}",
                            "predicted": predicted_resistance,
                            "actual": actual_resistance,
                            "error": error,
                            "relative_error": error / actual_resistance
                            if actual_resistance != 0
                            else float("inf"),
                        }
                    )

            except Exception as e:
                # If simulation fails, assign large error
                errors.append(1000.0)
                detailed_errors.append(
                    {
                        "nodes": f"{node_a}-{node_b}",
                        "error": 1000.0,
                        "error_type": f"simulation_failed: {e!s}",
                    }
                )

        return {
            "total_error": sum(errors),
            "max_error": max(errors) if errors else 0,
            "mean_error": sum(errors) / len(errors) if errors else 0,
            "detailed_errors": detailed_errors,
            "num_measurements": len(measurements_data),
        }

    except Exception as e:
        return {"error": f"Validation failed: {e!s}"}


@tool
def propose_simple_topology(num_resistors: int, topology_type: str) -> str:
    """[BRIEF] Generate a simple circuit topology hypothesis for testing. [/BRIEF]

    [DETAILED] Creates standard circuit configurations (series, parallel, series-parallel)
    with placeholder resistor values. Useful for starting hypothesis generation when you
    have an idea about the circuit complexity but need a structured starting point. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - At the beginning of analysis to generate initial topology hypotheses
    - When you know approximately how many resistors are present
    - To create templates for manual modification
    - For systematic exploration of possible configurations
    [/PROCEDURAL]

    Args:
        num_resistors: Number of resistors in the circuit
        topology_type: Type of configuration ("series", "parallel", "series_parallel", "bridge")

    Returns:
        str: JSON string with proposed topology structure

    Example:
        propose_simple_topology(3, "series") ->
        '{"resistors": {"R1": 10, "R2": 10, "R3": 10},
          "connections": [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]]}'
    """
    topology = {"resistors": {}, "connections": []}

    # Create resistor entries with placeholder values
    for i in range(1, num_resistors + 1):
        topology["resistors"][f"R{i}"] = 10.0  # Placeholder value

    if topology_type == "series":
        # Chain resistors in series A-R1-B-R2-C-R3-D...
        for i in range(1, num_resistors + 1):
            node1 = chr(ord("A") + i - 1)  # A, B, C, ...
            node2 = chr(ord("A") + i)  # B, C, D, ...
            topology["connections"].append([node1, node2, f"R{i}"])

    elif topology_type == "parallel":
        # All resistors between same two nodes A and B
        for i in range(1, num_resistors + 1):
            topology["connections"].append(["A", "B", f"R{i}"])

    elif topology_type == "series_parallel" and num_resistors >= 3:
        # Mix of series and parallel
        # R1 in series, then R2||R3 in parallel
        topology["connections"].append(["A", "B", "R1"])
        topology["connections"].append(["B", "C", "R2"])
        topology["connections"].append(["B", "C", "R3"])

        # Add any remaining resistors in series
        for i in range(4, num_resistors + 1):
            node1 = chr(ord("A") + i - 2)
            node2 = chr(ord("A") + i - 1)
            topology["connections"].append([node1, node2, f"R{i}"])

    elif topology_type == "bridge" and num_resistors >= 5:
        # Wheatstone bridge configuration
        topology["connections"] = [
            ["A", "B", "R1"],
            ["A", "C", "R2"],
            ["B", "D", "R3"],
            ["C", "D", "R4"],
            ["B", "C", "R5"],  # Bridge resistor
        ]

    else:
        # Default to series for unsupported configurations
        for i in range(1, num_resistors + 1):
            node1 = chr(ord("A") + i - 1)
            node2 = chr(ord("A") + i)
            topology["connections"].append([node1, node2, f"R{i}"])

    return json.dumps(topology, indent=2)


@tool
def estimate_resistor_values(topology: str, measurements: str) -> str:
    """[BRIEF] Estimate resistor values for a given topology to match measurements. [/BRIEF]

    [DETAILED] Uses optimization techniques to find resistor values that best fit the
    given measurements for a fixed circuit topology. This tool helps complete a
    topology hypothesis by determining appropriate component values. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - After determining the likely circuit topology
    - When you have the connections but need to find resistor values
    - For fine-tuning an initial guess of resistor values
    - As final step before validation
    [/PROCEDURAL]

    Args:
        topology: JSON string with circuit topology (resistor values will be optimized)
        measurements: JSON string with measurement data to fit

    Returns:
        str: JSON string with optimized topology including estimated resistor values
    """
    try:
        circuit = json.loads(topology)
        _measurements_data = json.loads(measurements)

        # Simple optimization: try different resistance combinations
        resistor_names = list(circuit["resistors"].keys())
        best_topology = circuit.copy()
        best_error = float("inf")

        # Try common resistance values
        common_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

        # For small numbers of resistors, try all combinations
        if len(resistor_names) <= 3:
            for values in itertools.product(common_values, repeat=len(resistor_names)):
                test_topology = circuit.copy()
                for i, resistor in enumerate(resistor_names):
                    test_topology["resistors"][resistor] = values[i]

                validation = validate_measurements(
                    json.dumps(test_topology), measurements
                )
                if (
                    "total_error" in validation
                    and validation["total_error"] < best_error
                ):
                    best_error = validation["total_error"]
                    best_topology = test_topology.copy()

        # For larger circuits, use a simple heuristic approach
        else:
            # Start with equal values and adjust based on measurements
            for base_value in common_values:
                test_topology = circuit.copy()
                for resistor in resistor_names:
                    test_topology["resistors"][resistor] = base_value

                validation = validate_measurements(
                    json.dumps(test_topology), measurements
                )
                if (
                    "total_error" in validation
                    and validation["total_error"] < best_error
                ):
                    best_error = validation["total_error"]
                    best_topology = test_topology.copy()

        return json.dumps(best_topology, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Estimation failed: {e!s}"})


@tool
def generate_test_measurements(topology: str, terminal_pairs: list[list[str]]) -> str:
    """[BRIEF] Generate theoretical measurements for a given circuit topology. [/BRIEF]

    [DETAILED] Calculates what the resistance measurements would be between specified
    terminal pairs for a given circuit. Useful for testing your tools and understanding
    how different topologies produce different measurement patterns. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - To understand what measurements a proposed topology would produce
    - For testing and debugging your circuit analysis approach
    - To generate additional synthetic measurements for validation
    - When exploring how topology changes affect measurements
    [/PROCEDURAL]

    Args:
        topology: JSON string describing the circuit
        terminal_pairs: List of node pairs to measure between

    Returns:
        str: JSON string with theoretical measurements

    Example:
        terminal_pairs = [["A", "B"], ["A", "C"], ["B", "C"]]
        generate_test_measurements(topology, terminal_pairs) ->
        '[{"node_a": "A", "node_b": "B", "resistance": 15.0}, ...]'
    """
    try:
        measurements = []

        for pair in terminal_pairs:
            if len(pair) != 2:
                continue

            try:
                resistance = simulate_circuit_resistance(topology, pair)
                measurements.append(
                    {
                        "node_a": pair[0],
                        "node_b": pair[1],
                        "resistance": round(resistance, 3),
                    }
                )
            except Exception as e:
                measurements.append(
                    {
                        "node_a": pair[0],
                        "node_b": pair[1],
                        "error": f"Could not measure: {e!s}",
                    }
                )

        return json.dumps(measurements, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Measurement generation failed: {e!s}"})
