import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from corral.backend.tool import Tool, tool


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


def calculate_series_resistance(resistances: list[float]) -> float:
    """[BRIEF] Calculate total resistance of resistors connected in series. [/BRIEF]

    [DETAILED] Computes the equivalent resistance when multiple resistors are connected
    end-to-end in a single path. In series configuration, current flows through each
    resistor sequentially, and the total resistance is the sum of individual resistances.
    This is fundamental for analyzing any resistor network. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you have identified a series branch in a circuit.
    - Best suited for simplifying complex networks by combining series elements.
    - Recommended as a building block for more complex resistance calculations.
    - Use when testing hypotheses about circuit topology where series connections are assumed.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Takes a list of individual resistance values.
    - Sums all the provided resistance values.
    - The result represents the total equivalent resistance.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a series connection of resistors within a larger circuit diagram. [/PREREQUISITE]
        2. [CURRENT] Apply this tool to calculate their combined resistance. [/CURRENT]
        3. [FOLLOW_UP] Replace the series resistors with their equivalent resistance in the circuit for further analysis or simplification. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `calculate_series_resistance([10, 20, 30])`
    - `calculate_series_resistance([5.5, 12.3, 7.2, 1.0])`
    [/SYNTACTICAL]

    Args:
        resistances : [BRIEF] List of resistance values in ohms. [/BRIEF]
                      [DETAILED] A list containing floating-point numbers, each representing the resistance of an individual resistor. All values must be positive. [/DETAILED]
                      [SYNTACTIC] Format: `[float, float, ...]` [/SYNTACTIC]
                      [EXAMPLES] `[10, 20, 30]`, `[100.5, 200]` [/EXAMPLES]

    Returns:
        float: [BRIEF] Total series resistance in ohms. [/BRIEF]
               [DETAILED] A single floating-point number representing the sum of all input resistances. [/DETAILED]
               [EXAMPLES] `60.0` (for `[10, 20, 30]`), `300.5` (for `[100.5, 200]`) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When an empty list of resistances is provided. [/ERROR_WHEN]
            [ERROR_DETAILS] The sum of an empty list is undefined in this context, indicating no resistors are present. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure the `resistances` list contains at least one valid resistance value. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When a resistance value is not a positive number. [/ERROR_WHEN]
            [ERROR_DETAILS] Resistances in a physical circuit are typically positive. Zero or negative values would lead to non-physical results. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure all resistance values in the input list are positive floating-point numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Assumes ideal resistors.
    - Only applicable for purely series connections.
    [/LIMITATIONS]
    """
    if not resistances:
        raise ValueError("No resistances provided")
    return sum(resistances)


@tool
def calculate_parallel_resistance(resistances: list[float]) -> float:
    """[BRIEF] Calculate total resistance of resistors connected in parallel. [/BRIEF]

    [DETAILED] Computes the equivalent resistance when multiple resistors are connected
    across the same two nodes. In parallel configuration, current divides among the
    resistors, and the reciprocal of total resistance equals the sum of reciprocals
    of individual resistances. This calculation is crucial for simplifying parallel branches in a circuit. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you have identified parallel branches in a circuit.
    - Best suited for simplifying complex networks by combining parallel elements.
    - Recommended when testing different topology hypotheses involving parallel connections.
    - Use as part of iterative network reduction strategies.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Takes a list of individual resistance values.
    - Calculates the reciprocal of each resistance.
    - Sums these reciprocal values.
    - Takes the reciprocal of the sum to find the total parallel resistance.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a parallel connection of resistors within a larger circuit diagram. [/PREREQUISITE]
        2. [CURRENT] Apply this tool to calculate their combined resistance. [/CURRENT]
        3. [FOLLOW_UP] Replace the parallel resistors with their equivalent resistance in the circuit for further analysis or simplification. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `calculate_parallel_resistance([10, 20])`
    - `calculate_parallel_resistance([100, 200, 300])`
    [/SYNTACTICAL]

    Args:
        resistances : [BRIEF] List of resistance values in ohms. [/BRIEF]
                      [DETAILED] A list containing floating-point numbers, each representing the resistance of an individual resistor. All values must be positive. [/DETAILED]
                      [SYNTACTIC] Format: `[float, float, ...]` [/SYNTACTIC]
                      [EXAMPLES] `[10, 20]`, `[100.5, 200, 50]` [/EXAMPLES]

    Returns:
        float: [BRIEF] Total parallel resistance in ohms. [/BRIEF]
               [DETAILED] A single floating-point number representing the equivalent resistance of all input resistors connected in parallel. [/DETAILED]
               [EXAMPLES] `6.67` (for `[10, 20]`), `54.545` (for `[100, 200, 300]`) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When an empty list of resistances is provided. [/ERROR_WHEN]
            [ERROR_DETAILS] An empty list means no resistors are in parallel, making the calculation undefined. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure the `resistances` list contains at least one positive resistance value. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When any resistance value is zero or negative. [/ERROR_WHEN]
            [ERROR_DETAILS] Resistances in parallel must be positive for a valid physical interpretation and to avoid division by zero. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure all resistance values in the input list are positive floating-point numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Assumes ideal resistors.
    - Only applicable for purely parallel connections.
    [/LIMITATIONS]
    """
    if not resistances:
        raise ValueError("No resistances provided")
    if any(r <= 0 for r in resistances):
        raise ValueError("All resistances must be positive")

    return 1 / sum(1 / r for r in resistances)


@tool
def delta_to_wye_transform(ra: float, rb: float, rc: float) -> str:
    """[BRIEF] Convert delta (triangle) resistor configuration to wye (star) configuration. [/BRIEF]

    [DETAILED] Transforms a three-resistor delta network into an equivalent three-resistor
    wye network. This is essential for solving complex resistor networks that cannot be
    reduced using simple series/parallel combinations. The transformation preserves the
    resistance between any two external nodes, simplifying nodal analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when encountering triangle (delta) configurations that block series/parallel reduction in a circuit.
    - Best suited as part of a network analysis strategy for complex topologies, especially bridge circuits.
    - Recommended when testing circuit topology hypotheses involving triangular connections.
    - Use before applying nodal analysis to simplify the network's structure.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Takes three resistance values (Ra, Rb, Rc) representing the resistors in a delta configuration.
    - Calculates the equivalent Wye (star) resistances (R1, R2, R3) using standard transformation formulas:
        - R1 = (Rb * Rc) / (Ra + Rb + Rc)
        - R2 = (Ra * Rc) / (Ra + Rb + Rc)
        - R3 = (Ra * Rb) / (Ra + Rb + Rc)
    - Returns these three calculated resistances.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a delta (triangle) configuration in the circuit that prevents further series/parallel simplification. [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the three delta resistor values to obtain their equivalent wye resistor values. [/CURRENT]
        3. [FOLLOW_UP] Substitute the original delta network with the equivalent wye network in the circuit diagram, which should now allow for series/parallel reduction or simpler nodal analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `delta_to_wye_transform(30, 30, 30)`
    - `delta_to_wye_transform(100, 50, 75)`
    [/SYNTACTICAL]

    Args:
        ra : [BRIEF] Resistance between nodes A and B in delta configuration. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected between nodes A and B in the delta network. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `30`, `100` [/EXAMPLES]
        rb : [BRIEF] Resistance between nodes B and C in delta configuration. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected between nodes B and C in the delta network. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `30`, `50` [/EXAMPLES]
        rc : [BRIEF] Resistance between nodes C and A in delta configuration. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected between nodes C and A in the delta network. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `30`, `75` [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with keys 'r1', 'r2', 'r3' for wye resistor values. [/BRIEF]
             [DETAILED] A JSON string containing a dictionary with three keys: 'r1', 'r2', and 'r3', whose values are the calculated equivalent resistances for the wye network, each being a floating-point number. R1 is connected to original node A, R2 to B, and R3 to C. [/DETAILED]
             [EXAMPLES] `{"r1": 10.0, "r2": 10.0, "r3": 10.0}` (for `delta_to_wye_transform(30, 30, 30)`) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When the sum of delta resistances (ra + rb + rc) is zero. [/ERROR_WHEN]
            [ERROR_DETAILS] A zero sum in the denominator of the transformation formulas would lead to division by zero, indicating an invalid or non-physical delta configuration. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure all input resistances `ra`, `rb`, and `rc` are positive values. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When any input resistance (ra, rb, or rc) is zero or negative. [/ERROR_WHEN]
            [ERROR_DETAILS] Physical resistors have positive resistance values. Zero or negative values would result in non-physical wye resistances. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure `ra`, `rb`, and `rc` are all positive floating-point numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Assumes ideal resistors.
    - Only applicable for 3-resistor delta configurations.
    [/LIMITATIONS]
    """
    total = ra + rb + rc
    if total == 0:
        raise ValueError("Sum of delta resistances cannot be zero")
    if any(r <= 0 for r in [ra, rb, rc]):
        raise ValueError("All resistances must be positive")

    r1 = (rb * rc) / total  # Connected to node A
    r2 = (ra * rc) / total  # Connected to node B
    r3 = (ra * rb) / total  # Connected to node C

    result = {"r1": r1, "r2": r2, "r3": r3}
    return json.dumps(result)


@tool
def wye_to_delta_transform(r1: float, r2: float, r3: float) -> str:
    """[BRIEF] Convert wye (star) resistor configuration to delta (triangle) configuration. [/BRIEF]

    [DETAILED] Transforms a three-resistor wye network into an equivalent three-resistor
    delta network. This is the inverse of delta-to-wye transformation and is useful when
    the delta form provides easier analysis or when testing different topology hypotheses. This transformation is key for circuit simplification in cases where a wye configuration is encountered. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to convert a wye (star) configuration into an equivalent delta (triangle) configuration.
    - Best suited for situations where the delta form simplifies further series/parallel reductions or nodal analysis.
    - Recommended when testing different topology hypotheses where converting a wye to a delta might offer a clearer path to a solution.
    - Use when a wye configuration makes direct analysis difficult.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Takes three resistance values (R1, R2, R3) representing the resistors in a wye configuration.
    - Calculates the equivalent Delta (triangle) resistances (Ra, Rb, Rc) using standard inverse transformation formulas:
        - Ra = (R1*R2 + R2*R3 + R3*R1) / R3
        - Rb = (R1*R2 + R2*R3 + R3*R1) / R1
        - Rc = (R1*R2 + R2*R3 + R3*R1) / R2
    - Returns these three calculated resistances.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a wye (star) configuration in the circuit that is difficult to analyze directly. [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the three wye resistor values to obtain their equivalent delta resistor values. [/CURRENT]
        3. [FOLLOW_UP] Substitute the original wye network with the equivalent delta network in the circuit diagram, which should now allow for simpler analysis or further circuit reduction. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `wye_to_delta_transform(10, 10, 10)`
    - `wye_to_delta_transform(50, 75, 100)`
    [/SYNTACTICAL]

    Args:
        r1 : [BRIEF] Wye resistor connected to node A. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected from the center of the wye to node A. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `10`, `50` [/EXAMPLES]
        r2 : [BRIEF] Wye resistor connected to node B. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected from the center of the wye to node B. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `10`, `75` [/EXAMPLES]
        r3 : [BRIEF] Wye resistor connected to node C. [/BRIEF]
             [DETAILED] A positive floating-point number representing the resistance of the resistor connected from the center of the wye to node C. [/DETAILED]
             [SYNTACTIC] Format: `float` (positive) [/SYNTACTIC]
             [EXAMPLES] `10`, `100` [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with keys 'ra', 'rb', 'rc' for delta resistor values. [/BRIEF]
             [DETAILED] A JSON string containing a dictionary with three keys: 'ra', 'rb', and 'rc', whose values are the calculated equivalent resistances for the delta network, each being a floating-point number. Ra is between original nodes A and B, Rb between B and C, and Rc between C and A. [/DETAILED]
             [EXAMPLES] `{"ra": 30.0, "rb": 30.0, "rc": 30.0}` (for `wye_to_delta_transform(10, 10, 10)`) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When any input resistance (r1, r2, or r3) is zero or negative. [/ERROR_WHEN]
            [ERROR_DETAILS] Physical resistors have positive resistance values. Zero or negative values would lead to non-physical delta resistances or division by zero in the transformation formulas. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure `r1`, `r2`, and `r3` are all positive floating-point numbers. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Assumes ideal resistors.
    - Only applicable for 3-resistor wye configurations.
    [/LIMITATIONS]
    """
    denominator = r1 * r2 + r2 * r3 + r3 * r1
    if any(r <= 0 for r in [r1, r2, r3]):
        raise ValueError("All resistances must be positive")

    ra = denominator / r3  # Between nodes A and B
    rb = denominator / r1  # Between nodes B and C
    rc = denominator / r2  # Between nodes C and A

    result = {"ra": ra, "rb": rb, "rc": rc}
    return json.dumps(result)


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
def validate_measurements(topology: str, measurements: str) -> str:
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
                predicted_resistance = simulate_circuit_resistance.execute(
                    topology=topology, terminal_nodes=[node_a, node_b]
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

        result = {
            "total_error": sum(errors),
            "max_error": max(errors) if errors else 0,
            "mean_error": sum(errors) / len(errors) if errors else 0,
            "detailed_errors": detailed_errors,
            "num_measurements": len(measurements_data),
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Validation failed: {e!s}"}, indent=2)


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
    """[BRIEF] Estimate resistor values using numerical optimization (robust approach). [/BRIEF]

    [DETAILED] Uses scipy optimization to find resistor values that minimize the squared error
    between predicted and measured resistances. Much more robust than brute-force search. [/DETAILED]

    Args:
        topology: JSON string describing circuit topology with initial resistor value guesses
        measurements: JSON string with actual measurements

    Returns:
        str: JSON string with optimized resistor values and optimization info
    """
    try:
        circuit = json.loads(topology)
        measurements_data = json.loads(measurements)

        # Extract resistor names and initial values
        resistor_names = list(circuit["resistors"].keys())
        initial_values = [circuit["resistors"][name] for name in resistor_names]

        def objective_function(resistor_values: np.ndarray) -> float:
            """Calculate total squared error for given resistor values"""
            # Create test topology with current resistor values
            test_topology = circuit.copy()
            test_topology["resistors"] = dict(
                zip(resistor_names, resistor_values, strict=False)
            )

            total_error = 0.0
            for measurement in measurements_data:
                try:
                    # Use the simulation tool to predict resistance
                    predicted = simulate_circuit_resistance.execute(
                        topology=json.dumps(test_topology),
                        terminal_nodes=[measurement["node_a"], measurement["node_b"]],
                    )
                    predicted = float(predicted)  # Convert string to float
                    actual = measurement["resistance"]
                    error = (predicted - actual) ** 2  # Squared error
                    total_error += error

                except Exception:
                    # Penalize simulation failures heavily
                    total_error += 1e6

            return total_error

        # Define bounds (resistors should be positive, reasonable range)
        bounds = [(0.1, 10000) for _ in resistor_names]  # 0.1Ω to 10kΩ

        # Run optimization
        result = minimize(
            objective_function,
            x0=initial_values,
            method="L-BFGS-B",  # Good for bounded problems
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success:
            # Create optimized topology
            optimized_topology = circuit.copy()
            optimized_topology["resistors"] = {
                name: round(float(value), 0)
                for name, value in zip(resistor_names, result.x, strict=False)
            }

            return json.dumps(
                {
                    "resistors": optimized_topology["resistors"],
                    "connections": circuit["connections"],
                    "optimization_info": {
                        "success": True,
                        "final_error": float(result.fun),
                        "iterations": result.nit,
                        "message": result.message,
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "error": f"Optimization failed: {result.message}",
                    "optimization_info": {
                        "success": False,
                        "final_error": float(result.fun),
                        "iterations": result.nit,
                    },
                }
            )

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


def create_tools() -> dict[str, Tool]:
    """Create all available tools"""
    return {
        "calculate_series_resistance": calculate_series_resistance,
        "calculate_parallel_resistance": calculate_parallel_resistance,
        "delta_to_wye_transform": delta_to_wye_transform,
        "wye_to_delta_transform": wye_to_delta_transform,
        "simulate_circuit_resistance": simulate_circuit_resistance,
        "validate_measurements": validate_measurements,
        "propose_simple_topology": propose_simple_topology,
        "estimate_resistor_values": estimate_resistor_values,
        "generate_test_measurements": generate_test_measurements,
    }
