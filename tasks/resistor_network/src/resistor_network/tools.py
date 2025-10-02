import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from corral.backend.tool import Tool, tool
from resistor_network.utils import get_resistance_between_nodes


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
    resistor sequentially, and the total resistance is the sum of individual resistances.[/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you have identified a series branch in a circuit.
    - Best suited for simplifying complex networks by combining series elements.
    - Use when testing hypotheses about circuit topology where series connections are assumed.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Takes a list of individual resistance values.
    - Sums all the provided resistance values.
    - The result represents the total equivalent resistance.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Identify a series connection of resistors within a larger circuit. [/PREREQUISITE]
        2. [CURRENT] Apply this tool to calculate their combined resistance. [/CURRENT]
        3. [FOLLOW_UP] Consider the series resistors with their equivalent resistance in the circuit for further analysis or simplification. [/FOLLOW_UP]
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
    between two specified terminal nodes using nodal analysis. This tool allows for the
    testing and validation of topology hypotheses by comparing calculated resistance with
    measured values. It provides a foundational capability for circuit analysis and design. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use to validate a proposed circuit topology against known or measured values.
    - Best suited after constructing a hypothesis about resistor values and their connections.
    - Recommended for iterative testing of different network configurations to see their impact on equivalent resistance.
    - Use as a verification step before finalizing a circuit design or submitting a solution.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses a JSON string representing the circuit's resistors and their connections.
    - Identifies all unique nodes in the circuit and creates a mapping to numerical indices.
    - Constructs a conductance matrix (G matrix) based on the connections and resistor values.
    - Applies a 1A current source between the specified `terminal_nodes`.
    - Solves the resulting system of linear equations (G * V = I) to find the node voltages.
    - The equivalent resistance is then calculated as the absolute voltage difference between the `terminal_nodes` (V_terminal1 - V_terminal2) since the applied current is 1A.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Have a defined circuit `topology` (resistors and connections) and the `terminal_nodes` between which to measure resistance. [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the `topology` and `terminal_nodes` to get the simulated resistance. [/CURRENT]
        3. [FOLLOW_UP] Compare the `simulate_circuit_resistance` output with actual measurements or desired specifications to validate the topology using `validate_measurements`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `simulate_circuit_resistance('{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"], ["B", "C", "R2"]]}', ["A", "C"])`
    - `simulate_circuit_resistance('{"resistors": {"R1": 50, "R2": 50, "R3": 100}, "connections": [["N1", "N2", "R1"], ["N1", "N2", "R2"], ["N2", "N3", "R3"]]}', ["N1", "N3"])`
    [/SYNTACTICAL]

    Args:
        topology : [BRIEF] JSON string describing circuit. [/BRIEF]
                   [DETAILED] A JSON string that defines the circuit's components and their interconnections. It must contain a "resistors" dictionary (mapping resistor IDs to their resistance values) and a "connections" list (each entry being a list `[node1, node2, resistor_id]`). [/DETAILED]
                   [SYNTACTIC] Format: `{"resistors": {"R1": 10, "R2": 20}, "connections": [["node1", "node2", "R1"]]}` [/SYNTACTIC]
                   [EXAMPLES] `'{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"], ["B", "C", "R2"]]}'`, `'{"resistors": {"R_par1": 30, "R_par2": 60}, "connections": [["N_in", "N_out", "R_par1"], ["N_in", "N_out", "R_par2"]]}` [/EXAMPLES]
        terminal_nodes : [BRIEF] List of two node names to measure resistance between. [/BRIEF]
                         [DETAILED] A list containing exactly two strings, where each string is the name of a node in the circuit. The tool will calculate the equivalent resistance between these two specified nodes. [/DETAILED]
                         [SYNTACTIC] Format: `["node_start", "node_end"]` [/SYNTACTIC]
                         [EXAMPLES] `["A", "C"]`, `["input_node", "output_node"]` [/EXAMPLES]

    Returns:
        float: [BRIEF] Equivalent resistance between terminals in ohms. [/BRIEF]
               [DETAILED] A floating-point number representing the total equivalent resistance measured between the two specified terminal nodes in the given circuit topology. [/DETAILED]
               [EXAMPLES] `30.0` (for a 10Ω and 20Ω resistor in series) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When the `topology` JSON string is invalid or malformed. [/ERROR_WHEN]
            [ERROR_DETAILS] `json.loads` fails, or required keys ("resistors", "connections") are missing. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure the `topology` string is a valid JSON and adheres to the specified structure. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When `terminal_nodes` does not contain exactly two node names. [/ERROR_WHEN]
            [ERROR_DETAILS] Resistance is defined between two distinct points. Fewer or more nodes are ambiguous. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Provide a list with exactly two string elements for `terminal_nodes`. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When a `resistor_id` in `connections` is not found in the `resistors` dictionary. [/ERROR_WHEN]
            [ERROR_DETAILS] An undefined resistor ID indicates an inconsistency in the circuit description. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure all resistor IDs used in `connections` are defined in the `resistors` dictionary. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When a resistor has a non-positive resistance value (<= 0). [/ERROR_WHEN]
            [ERROR_DETAILS] Nodal analysis assumes positive resistances. Zero or negative resistance can cause mathematical issues. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure all resistor values in the `resistors` dictionary are positive floating-point numbers. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When the circuit is not solvable (e.g., disconnected). [/ERROR_WHEN]
            [ERROR_DETAILS] `np.linalg.solve` might fail if the conductance matrix is singular, implying the circuit is ill-posed or disconnected. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Review the circuit `topology` for open circuits, short circuits, or disconnected components that prevent a unique solution. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Assumes ideal resistors (no inductance, capacitance).
    - Only calculates equivalent resistance; does not simulate transient behavior or AC circuits.
    - Circuit must be a passive network for resistance calculation.
    - Numerical stability issues can arise for extremely large/small resistance values.
    [/LIMITATIONS]
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
        # This handles cases where term2_idx is the last element
        if n == 1:  # Handle single-node circuit, which implies shorted
            return 0.0

        if term2_idx == n - 1:
            G_reduced = G[:-1, :-1]
            I_reduced = Ia[:-1]
        else:
            G_reduced = np.delete(np.delete(G, term2_idx, 0), term2_idx, 1)
            I_reduced = np.delete(Ia, term2_idx)

        try:
            # Handle cases where G_reduced might be empty or singular (e.g., two nodes directly connected with no resistors to other nodes)
            if G_reduced.size == 0:
                # If only two nodes and directly connected without other paths, resistance is sum of direct path.
                # This specific case is handled by the loop over connections
                # If G_reduced is empty after removing rows/cols, it implies a 2-node circuit with no other connections.
                # In such cases, if a direct resistor exists between term1 and term2, its value is the resistance.
                # This logic is complex and better handled by checking for direct connections first.
                # For simplicity here, if the reduced matrix is empty or singular, it's likely an error unless it's a very simple 2-node series circuit.
                raise np.linalg.LinAlgError(
                    "Reduced conductance matrix is empty or singular"
                )

            V_reduced = np.linalg.solve(G_reduced, I_reduced)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "Circuit is not solvable (possibly disconnected or ill-conditioned)"
            ) from err

        # Insert reference voltage (0V at term2)
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

    [DETAILED] Compares the resistance predictions of a proposed circuit topology against
    actual measurements. It quantifies the discrepancy by calculating error metrics,
    which are essential for validating hypotheses about circuit structure and resistor values.
    This tool helps in refining and confirming circuit designs. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after proposing a complete circuit topology and its corresponding resistor values.
    - Best suited to quantify how well your hypothesis matches available experimental measurement data.
    - Recommended before final solution submission to assess the quality and accuracy of the proposed circuit.
    - Use during iterative refinement of topology hypotheses to guide adjustments.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the input JSON strings for the proposed circuit `topology` and the `measurements`.
    - Iterates through each measurement provided in the `measurements` list.
    - For each measurement, it calls `get_resistance_between_nodes` to predict the resistance between the specified nodes in the proposed `topology`.
    - Compares the `predicted_resistance` with the `actual_resistance` from the measurement.
    - Calculates the absolute error and relative error for each measurement.
    - Aggregates these errors to provide `total_error`, `max_error`, and `mean_error`, along with detailed error for each measurement.
    - Returns a JSON string summarizing the validation results.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Have a proposed `topology` (e.g., from `propose_simple_topology` and `estimate_resistor_values`) and a set of `measurements` (actual data). [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the `topology` and `measurements` to get a quantitative assessment of the match. [/CURRENT]
        3. [FOLLOW_UP] If errors are high, iterate back to refining the `topology` or re-estimating resistor values using `estimate_resistor_values`. If errors are acceptable, consider the topology validated. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `validate_measurements('{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"]]}', '[{"node_a": "A", "node_b": "B", "resistance": 15.0}]')`
    - `validate_measurements(proposed_circuit_topology, experimental_data)`
    [/SYNTACTICAL]

    Args:
        topology : [BRIEF] JSON string describing proposed circuit topology. [/BRIEF]
                   [DETAILED] A JSON string conforming to the `CircuitTopology` structure, including resistor IDs, their estimated values, and the connections between nodes. This represents your hypothesis about the circuit's structure. [/DETAILED]
                   [SYNTACTIC] Format: {"resistors": {"R1": 10, "R2": 20}, "connections": [["node1", "node2", "R1"]]} [/SYNTACTIC]
                [EXAMPLES] '{"resistors": {"R1": 100, "R2": 50}, "connections": [["A", "B", "R1"], ["B", "C", "R2"]]}' [/EXAMPLES]
        measurements : [BRIEF] JSON string with actual measurements. [/BRIEF]
                [DETAILED] A JSON string representing a list of CircuitMeasurement objects. Each object should contain node_a, node_b, and at least resistance (though voltage and current are also possible if the tool were to be extended for them). These are the real-world observations. [/DETAILED]
                [SYNTACTIC] Format: [{"node_a": "A", "node_b": "B", "resistance": 10.0}, ...] [/SYNTACTIC]
                [EXAMPLES] '[{"node_a": "A", "node_b": "B", "resistance": 15.0}]', '[{"node_a": "N1", "node_b": "N3", "resistance": 150.0}, {"node_a": "N2", "node_b": "N4", "resistance": 75.0}]' [/EXAMPLES]

    Returns:
    str: [BRIEF] JSON string with validation results including error metrics. [/BRIEF]
         [DETAILED] A JSON string containing a dictionary with various error metrics: `total_error`, `max_error`, `mean_error`, and `detailed_errors` (a list of per-measurement errors including predicted, actual, absolute error, and relative error). It also includes `num_measurements`. This output helps quantify the accuracy of the proposed topology. [/DETAILED]
         [EXAMPLES] `{"total_error": 5.0, "max_error": 5.0, "mean_error": 5.0, "detailed_errors": [{"nodes": "A-B", "predicted": 10.0, "actual": 15.0, "error": 5.0, "relative_error": 0.333}], "num_measurements": 1}` [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When the `topology` or `measurements` JSON strings are malformed or invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] `json.loads` fails, or required keys are missing from the input dictionaries/lists. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure both input strings are valid JSON and adhere to the specified data structures. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When `get_resistance_between_nodes` fails for a given measurement. [/ERROR_WHEN]
            [ERROR_DETAILS] Indicates an issue with the proposed `topology` itself (e.g., disconnected nodes, invalid resistor IDs) preventing a simulation. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Examine the `error_type` in the `detailed_errors` for specific simulation failures and debug the `topology` accordingly. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Currently only validates resistance measurements. Voltage and current validation are not yet implemented.
    - Assumes the measurement data is accurate and reliable.
    - Large errors can occur if the proposed topology is drastically different from the actual circuit or if resistor values are far off.
    [/LIMITATIONS]
    """

    try:
        measurements_data = json.loads(measurements)
        errors = []
        detailed_errors = []

        for measurement in measurements_data:
            node_a = measurement["node_a"]
            node_b = measurement["node_b"]

            try:
                predicted_resistance = get_resistance_between_nodes(
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

        [DETAILED] Creates standard, foundational circuit configurations (series, parallel,
    series-parallel, bridge) with placeholder resistor values (defaulting to 10.0 ohms).
    This tool is invaluable for initiating the hypothesis generation process when you
    have a basic idea about the circuit's complexity or expected structure but need a
    structured, pre-defined starting point for exploration. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use at the beginning of circuit analysis to generate initial, basic topology hypotheses.
    - Best suited when you have an approximate idea of how many resistors are present in the circuit.
    - Recommended to create templates for manual modification and refinement based on measurements.
    - Use for systematic exploration of common resistor configurations.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Initializes an empty circuit topology dictionary.
    - Adds `num_resistors` placeholder resistors (R1, R2, ...) with a default value of 10.0 ohms.
    - Based on `topology_type`:
        - "series": Connects resistors end-to-end (A-R1-B-R2-C...).
        - "parallel": Connects all resistors between two common nodes (A and B).
        - "series_parallel": Creates a basic configuration with one resistor in series, and two in parallel, adding others in series if `num_resistors` is greater than 3.
        - "bridge": Creates a Wheatstone bridge configuration (requires at least 5 resistors).
        - If `topology_type` is not recognized or `num_resistors` is too low for complex types, it defaults to a series configuration.
    - Returns the constructed topology as a JSON string.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Have a preliminary idea of the number of resistors and a general type of circuit (e.g., "series", "parallel"). [/PREREQUISITE]
        2. [CURRENT] Apply this tool with `num_resistors` and `topology_type` to get a starting `topology`. [/CURRENT]
        3. [FOLLOW_UP] Use `estimate_resistor_values` with this generated topology and actual `measurements` to refine the resistor values. Then, `validate_measurements` to check the fit. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `propose_simple_topology(3, "series")`
    - `propose_simple_topology(2, "parallel")`
    - `propose_simple_topology(5, "bridge")`
    [/SYNTACTICAL]

    Args:
        num_resistors : [BRIEF] Number of resistors in the circuit. [/BRIEF]
                        [DETAILED] An integer indicating how many individual resistors should be included in the generated topology. This influences the complexity and number of elements in the proposed circuit. [/DETAILED]
                        [SYNTACTIC] Format: `int` (positive) [/SYNTACTIC]
                        [EXAMPLES] `3`, `5`, `2` [/EXAMPLES]
        topology_type : [BRIEF] Type of configuration. [/BRIEF]
                        [DETAILED] A string specifying the desired basic arrangement of resistors. Valid options are "series", "parallel", "series_parallel", or "bridge". If an invalid type is provided or `num_resistors` is too low for the chosen type, it defaults to "series". [/DETAILED]
                        [SYNTACTIC] Format: `"series"`, `"parallel"`, `"series_parallel"`, `"bridge"` [/SYNTACTIC]
                        [EXAMPLES] `"series"`, `"parallel"`, `"bridge"` [/EXAMPLES]
                        [CHOICES] Valid options: "series", "parallel", "series_parallel", "bridge"

    Returns:
        str: [BRIEF] JSON string with proposed topology structure. [/BRIEF]
             [DETAILED] A JSON string representing a `CircuitTopology` object. It includes a "resistors" dictionary (with `R1`, `R2`, etc., initially set to 10.0 ohms) and a "connections" list defining how these resistors are wired based on the `topology_type`. [/DETAILED]
             [EXAMPLES] `'{"resistors": {"R1": 10, "R2": 10, "R3": 10}, "connections": [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]]}'` (for `propose_simple_topology(3, "series")`) [/EXAMPLES]

    [RAISES] Exceptions:
        None explicitly raised by the tool itself, but downstream tools using this output might raise errors if the generated topology is invalid for their operations.
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Generates only basic, predefined topologies. Complex or arbitrary circuit designs require manual modification.
    - Resistor values are placeholders (10.0 ohms) and need to be refined using `estimate_resistor_values`.
    - Node naming is sequential (A, B, C, ...) and may not align with complex real-world naming conventions.
    - "series_parallel" and "bridge" types have minimum `num_resistors` requirements.
    [/LIMITATIONS]
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
    """[BRIEF] Estimate resistor values using numerical optimization. [/BRIEF]

        [DETAILED] Utilizes the SciPy `minimize` function to find the optimal resistor values
    that minimize the total squared error between the resistance predicted by the
    `get_resistance_between_nodes` tool and the actual `measurements`. This approach
    is robust and efficient for refining initial resistor value guesses within a
    known or hypothesized circuit topology. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you have a proposed circuit `topology` (connections are fixed) but need to determine the precise resistor values.
    - Best suited for refining placeholder resistor values obtained from tools like `propose_simple_topology`.
    - Recommended for iterative improvement of a circuit model by fitting it to experimental data.
    - Use when brute-force search for resistor values is computationally infeasible or inefficient.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the input JSON strings for the initial `topology` (with resistor IDs and initial guesses) and the `measurements`.
    - Extracts the names of all resistors in the topology.
    - Defines an `objective_function` that takes a set of resistor values, constructs a temporary circuit `topology` with these values, and then uses `get_resistance_between_nodes` to predict resistances for all measurement pairs.
    - The `objective_function` calculates the sum of squared differences between predicted and actual resistances, returning this total error.
    - `scipy.optimize.minimize` (using the 'L-BFGS-B' method) is then employed to find the set of resistor values that minimizes this `objective_function`, subject to bounds (e.g., resistances must be positive).
    - If successful, it returns the optimized resistor values and optimization details.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Have a fixed circuit `topology` (e.g., from `propose_simple_topology` or a known design) and a set of `measurements` from the actual circuit. [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the `topology` (containing initial resistor value guesses) and `measurements` to find optimized resistor values. [/CURRENT]
        3. [FOLLOW_UP] Use the `validate_measurements` tool with the optimized topology to confirm the improved fit, or proceed with the refined resistor values in further circuit analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `estimate_resistor_values('{"resistors": {"R1": 10, "R2": 20}, "connections": [["A", "B", "R1"]]}', '[{"node_a": "A", "node_b": "B", "resistance": 15.0}]')`
    - `estimate_resistor_values(initial_topology_json, experimental_measurements_json)`
    [/SYNTACTICAL]

    Args:
        topology : [BRIEF] JSON string describing circuit topology with initial resistor value guesses. [/BRIEF]
                   [DETAILED] A JSON string conforming to the `CircuitTopology` structure. It must include a "resistors" dictionary with resistor IDs and their *initial estimated* resistance values, and a "connections" list defining the circuit structure. These initial values are the starting point for optimization. [/DETAILED]
                   [SYNTACTIC] Format: `{"resistors": {"R1": 10, "R2": 20}, "connections": [["node1", "node2", "R1"]]}` [/SYNTACTIC]
                   [EXAMPLES] `'{"resistors": {"R1": 10, "R2": 10, "R3": 10}, "connections": [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]]}'` [/EXAMPLES]
        measurements : [BRIEF] JSON string with actual measurements. [/BRIEF]
                       [DETAILED] A JSON string representing a list of `CircuitMeasurement` objects, each containing `node_a`, `node_b`, and `resistance`. These are the actual observed resistance values against which the model will be optimized. [/DETAILED]
                       [SYNTACTIC] Format: `[{"node_a": "A", "node_b": "B", "resistance": 10.0}, ...]` [/SYNTACTIC]
                       [EXAMPLES] `'[{"node_a": "A", "node_b": "B", "resistance": 15.0}, {"node_a": "C", "node_b": "D", "resistance": 35.0}]'` [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with optimized resistor values and optimization info. [/BRIEF]
             [DETAILED] A JSON string containing a dictionary. If successful, it includes the `optimized_resistors` (mapping resistor IDs to their newly estimated values, rounded to integers), the original `connections`, and an `optimization_info` sub-dictionary with details like `success` status, `final_error`, `iterations`, and `message`. If optimization fails, it provides an `error` message. [/DETAILED]
             [EXAMPLES] `{"resistors": {"R1": 15, "R2": 30}, "connections": [...], "optimization_info": {"success": true, "final_error": 0.001, "iterations": 50, "message": "CONVERGENCE: NORM_OF_GRADIENT_<=_TF_GAUSSIAN_SUM"}` (on success) [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERRORS]
            [ERROR_WHEN] When the `topology` or `measurements` JSON strings are malformed or invalid. [/ERROR_WHEN]
            [ERROR_DETAILS] `json.loads` fails, or required keys are missing from the input dictionaries/lists. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure both input strings are valid JSON and adhere to the specified data structures. [/ERROR_RECOVERY]
        ValueError: [ERRORS]
            [ERROR_WHEN] When `get_resistance_between_nodes` encounters an error during optimization (e.g., invalid intermediate topology). [/ERROR_WHEN]
            [ERROR_DETAILS] The objective function will assign a large penalty, but persistent simulation errors can lead to optimization failure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Ensure the initial `topology` is valid and the bounds for resistor values are reasonable, as extreme values might cause simulation instability. [/ERROR_RECOVERY]
        RuntimeError: [ERRORS]
            [ERROR_WHEN] When the numerical optimization algorithm fails to converge to a solution. [/ERROR_WHEN]
            [ERROR_DETAILS] Indicated by `result.success` being `False` and a `message` explaining the failure (e.g., maximum iterations reached, bounds violated). [/ERROR_DETAILS]
            [ERROR_RECOVERY] Try: Adjust initial resistor value guesses, widen the bounds, increase `maxiter`, or re-evaluate if the chosen `topology` is appropriate for the measurements. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Numerical optimization can get stuck in local minima if initial guesses are poor, or the solution space is complex.
    - Requires a robust `get_resistance_between_nodes` function; errors in simulation propagate to the optimization.
    - Assumes the provided topology (connections) is correct, only optimizing resistor values.
    - Computationally intensive for very large circuits with many unknown resistors.
    [/LIMITATIONS]
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
                    predicted = get_resistance_between_nodes(
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

        [DETAILED] Calculates what the resistance measurements would be between specified terminal pairs for a given circuit topology.
        This tool is useful for testing your circuit analysis tools, understanding the behavior of different topologies.
        It provides a ground truth for a given circuit design. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use to understand what resistance measurements a proposed or known topology would produce.
    - Best suited for testing and debugging your circuit analysis approach or custom tools.
    - Recommended to generate additional synthetic measurements for validation of `estimate_resistor_values` or `validate_measurements`.
    - Use when exploring how changes in topology or resistor values affect the overall circuit measurements.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the input `topology` JSON string.
    - Iterates through each `terminal_pair` provided in the list.
    - For each pair, it calls the `get_resistance_between_nodes` tool to calculate the equivalent resistance between those two nodes.
    - Stores the calculated resistance along with the `node_a` and `node_b` in a list of measurement dictionaries.
    - If a simulation fails for a specific `terminal_pair`, it records an error message for that measurement.
    - Returns a JSON string containing the list of theoretical measurements.
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Have a fully defined circuit `topology` (including resistor values) and a list of `terminal_pairs` where measurements are desired. [/PREREQUISITE]
        2. [CURRENT] Apply this tool with the `topology` and `terminal_pairs` to obtain a set of theoretical resistance measurements. [/CURRENT]
        3. [FOLLOW_UP] Use these generated measurements to test the `estimate_resistor_values` tool (by trying to recover the original resistor values), or to test the `validate_measurements` tool (by comparing against the same topology). [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - `generate_test_measurements(my_topology_json, [["A", "B"], ["A", "C"], ["B", "C"]])`
    - `generate_test_measurements('{"resistors": {"R1": 10, "R2": 20}, "connections": [["N1", "N2", "R1"], ["N2", "N3", "R2"]]}', [["N1", "N3"]])`
    [/SYNTACTICAL]

    Args:
        topology : [BRIEF] JSON string describing the circuit. [/BRIEF]
                   [DETAILED] A JSON string conforming to the `CircuitTopology` structure, containing both the resistor IDs with their precise resistance values and the connections between nodes. This is the circuit for which theoretical measurements are to be generated. [/DETAILED]
                   [SYNTACTIC] Format: `{"resistors": {"R1": 10, "R2": 20}, "connections": [["node1", "node2", "R1"]]}` [/SYNTACTIC]
                   [EXAMPLES] `'{"resistors": {"R1": 100, "R2": 200, "R3": 300}, "connections": [["A", "B", "R1"], ["B", "C", "R2"], ["A", "C", "R3"]]}'` [/EXAMPLES]
        terminal_pairs : [BRIEF] List of node pairs to measure between. [/BRIEF]
                         [DETAILED] A list of lists, where each inner list contains two strings representing the names of the nodes between which the equivalent resistance should be calculated. Each pair signifies one theoretical measurement point. [/DETAILED]
                         [SYNTACTIC] Format: `[["node_a", "node_b"], ["node_x", "node_y"], ...]` [/SYNTACTIC]
                         [EXAMPLES] `[["A", "B"], ["A", "C"], ["B", "C"]]`, `[["input", "output"]]` [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with theoretical measurements. [/BRIEF]
             [DETAILED] A JSON string representing a list of measurement dictionaries. Each dictionary will include `node_a`, `node_b`, and the `resistance` (rounded to 3 decimal places) between those nodes, as calculated by the `simulate_circuit_resistance` tool. If a simulation fails for a pair, an "error" key will be present instead of "resistance". [/DETAILED]
             [EXAMPLES] `'[{"node_a": "A", "node_b": "B", "resistance": 15.0}, {"node_a": "A", "node_b": "C", "resistance": 45.0}]'` [/EXAMPLES]

    [RAISES] Exceptions:
        json.JSONDecodeError: [BRIEF] If `topology` is not a valid JSON string.
                              [DETAILED] This occurs if the input `topology` string cannot be parsed into a valid JSON object, which is required for circuit definition.
        Exception: [BRIEF] General error during measurement generation.
                   [DETAILED] Catches any other unforeseen errors that might occur during the iteration through terminal pairs or calls to `get_resistance_between_nodes`, returning an error message for the overall process. Specific measurement errors are handled per-pair.

    [PERFORMANCE] Performance notes:
    - Time complexity: O(M * S), where M is the number of `terminal_pairs` and S is the time complexity of `get_resistance_between_nodes`.
    - Memory usage: Proportional to the size of the `topology` and the number of `terminal_pairs`.
    - Network calls: None (assuming `get_resistance_between_nodes` is an internal function or tool).
    - File I/O: None.

    [LIMITATIONS] Known limitations:
    - Relies entirely on the accuracy and robustness of the `get_resistance_between_nodes` tool.
    - Does not validate the `topology` for circuit correctness (e.g., disconnected components, short circuits) beyond what `get_resistance_between_nodes` handles.
    - Handles only resistance measurements; cannot generate other types of circuit measurements (e.g., voltage, current).

    [RELATED] Related tools:
    - `get_resistance_between_nodes()`: Directly called by this tool to perform individual resistance calculations.
    - `estimate_resistor_values()`: Can use the output of this tool as input for validation.
    - `validate_measurements()`: Can use the output of this tool to compare against actual measurements or another theoretical set.
    """
    try:
        measurements = []

        for pair in terminal_pairs:
            if len(pair) != 2:
                continue

            try:
                resistance = get_resistance_between_nodes(topology, pair)
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
