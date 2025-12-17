import json

import numpy as np


def get_resistance_between_nodes(topology: str, terminal_nodes: list[str]) -> float:
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
