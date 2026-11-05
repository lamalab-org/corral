import json
import math
from collections.abc import Callable
from typing import Any

from loguru import logger
from resistor_network.sampler import RESISTOR_POOL
from resistor_network.utils import get_resistance_between_nodes


def _parse_topology(topology_input: str) -> dict[str, Any] | None:
    """Parse and validate a submitted topology."""
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
            return None
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
    """Return merged conductance in siemens for each node pair."""
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
    """Score a submitted topology against the expected conductance map."""
    expected_conductances = conductance_map(expected_topology)
    expected_nodes = {node for pair in expected_conductances for node in pair}

    def score_fn(topology_input: str) -> float:
        try:
            proposed = _parse_topology(topology_input)
            if proposed is None:
                logger.info("Conductance scoring: submission failed structural checks")
                return 0.0

            if any(
                value not in RESISTOR_POOL for value in proposed["resistors"].values()
            ):
                logger.info("Conductance scoring: resistor value outside allowed pool")
                return 0.0

            proposed_conductances = conductance_map(proposed)
            proposed_nodes = {node for pair in proposed_conductances for node in pair}

            if len(proposed["resistors"]) != len(expected_topology["resistors"]):
                logger.info(
                    "Conductance scoring: resistor-count mismatch "
                    f"(expected={len(expected_topology['resistors'])}, "
                    f"got={len(proposed['resistors'])})"
                )
                return 0.0

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


def check_sparse_circuit_solution(
    expected_topology: dict[str, Any],
    expected_measurements: list[dict[str, Any]],
    measurement_tolerance: float = 0.000500001,
    allowed_resistor_values: list[float] | None = None,
) -> Callable[[str], float]:
    """Simulate every public reading without demanding the hidden wiring.

    The hidden answer supplies only the public node set and component count.
    Absolute tolerance represents three-decimal rounding, not a loose relative
    fit or a sum-of-errors threshold. All physical and simulation checks apply
    equally to the original circuit and alternative indistinguishable circuits.
    An optional published value pool constrains every submitted resistor;
    values may repeat and no private inventory or per-value counts are used.
    """
    if not expected_measurements:
        raise ValueError("Sparse scoring requires published measurements")
    if not math.isfinite(measurement_tolerance) or measurement_tolerance <= 0:
        raise ValueError("Measurement tolerance must be positive and finite")
    allowed = None
    if allowed_resistor_values is not None:
        if not allowed_resistor_values or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
            for value in allowed_resistor_values
        ):
            raise ValueError("Allowed resistor values must be positive and finite")
        allowed = frozenset(allowed_resistor_values)
        if any(
            value not in allowed for value in expected_topology["resistors"].values()
        ):
            raise ValueError(
                "Expected resistor values must belong to the published pool"
            )
    expected_nodes = {node for c in expected_topology["connections"] for node in c[:2]}
    expected_count = len(expected_topology["resistors"])
    if {
        node for m in expected_measurements for node in (m["node_a"], m["node_b"])
    } != expected_nodes:
        raise ValueError("Every expected node must appear in the public measurements")

    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def score_fn(topology_input: str) -> float:
        try:
            # Reject duplicate object keys rather than silently keeping the
            # last resistor value, then reuse common structural parsing.
            json.loads(topology_input, object_pairs_hook=unique_keys)
            proposed = _parse_topology(topology_input)
            if proposed is None or len(proposed["resistors"]) != expected_count:
                return 0.0
            if allowed is not None and any(
                value not in allowed for value in proposed["resistors"].values()
            ):
                return 0.0
            adjacency: dict[str, set[str]] = {}
            pairs = set()
            for a, b, _rid in proposed["connections"]:
                pair = tuple(sorted((a, b)))
                if pair in pairs:
                    return 0.0
                pairs.add(pair)
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
            if set(adjacency) != expected_nodes:
                return 0.0
            reachable, frontier = {next(iter(adjacency))}, [next(iter(adjacency))]
            while frontier:
                unseen = adjacency[frontier.pop()] - reachable
                reachable.update(unseen)
                frontier.extend(unseen)
            if reachable != expected_nodes:
                return 0.0
            # Independent nodal solves, not the generator's optimizer or its
            # acceptance flags. Every individual observation must pass.
            for measurement in expected_measurements:
                predicted = get_resistance_between_nodes(
                    topology_input, [measurement["node_a"], measurement["node_b"]]
                )
                if (
                    not math.isfinite(predicted)
                    or abs(predicted - measurement["resistance"])
                    > measurement_tolerance
                ):
                    return 0.0
            return 1.0
        except (ValueError, TypeError, KeyError, OverflowError):
            return 0.0

    return score_fn
