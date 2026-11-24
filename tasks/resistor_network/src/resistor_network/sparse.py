"""Sparse observations without an adjacency hint; v4 publishes a 30-value pool.

Numerical fitting here is an offline audit, never an agent-facing tool. Local
identifiability and bounded rewiring searches are not global uniqueness proofs;
the scorer consequently accepts any observationally indistinguishable circuit.
"""

import itertools
import math
import random
import uuid

import networkx as nx
import numpy as np
from resistor_network.sampler import (
    SCORING_TOLERANCE,
    SUBMISSION_FORMAT,
    TOOLS,
    Block,
    LevelConfig,
    _IdFactory,
    _jacobian_rank,
    _measurement_jacobian,
    _network_response,
    _validate_sampled_topology,
    all_node_pairs,
    compose_series,
    find_non_load_bearing_resistors,
    measure_pairs,
    measurement_budget,
    randomize_labels,
    relabel_circuit,
    rounding_stability,
)
from scipy.optimize import least_squares

BENCHMARK_VERSION = 4
DEFAULT_SEEDS = {1: 202610141, 2: 202610142}
MAX_REWIRE_AUDIT = 24
MEASUREMENT_ABS_TOLERANCE = 0.000500001


def _random_core(
    rng: random.Random, count: int, dense: bool, node_fraction: float = 0.60
) -> nx.Graph:
    """Varied cores, not subdivisions of a small fixed motif library."""
    if dense:
        nodes = max(12, round(count * 0.55))
        nodes += nodes % 2
        graph = nx.random_regular_graph(3, nodes, seed=rng.getrandbits(32))
        if not nx.is_biconnected(graph):
            return _random_core(rng, count, dense)
    else:
        nodes = max(5, math.ceil(count * node_fraction))
        graph = nx.cycle_graph(nodes)
    missing = [
        pair
        for pair in itertools.combinations(graph.nodes, 2)
        if not graph.has_edge(*pair)
    ]
    rng.shuffle(missing)
    graph.add_edges_from(missing[: count - graph.number_of_edges()])
    assert graph.number_of_edges() == count
    return graph


def _block(
    rng: random.Random,
    ids: _IdFactory,
    count: int,
    dense: bool,
    node_fraction: float = 0.60,
    value_pool: tuple[float, ...] | None = None,
) -> Block:
    graph = _random_core(rng, count, dense, node_fraction)
    nodes = {node: ids.node() for node in graph}
    a, b = rng.sample(list(graph), 2)
    resistors, connections = {}, []
    for x, y in graph.edges:
        rid = ids.resistor()
        # Explicit pools constrain discrete tasks. Continuous calibration uses
        # reproducible log-uniform values without a public range constraint.
        resistors[rid] = (
            float(rng.choice(value_pool))
            if value_pool is not None
            else round(math.exp(rng.uniform(math.log(10), math.log(200))), 6)
        )
        connections.append((nodes[x], nodes[y], rid))
    return Block(resistors, connections, nodes[a], nodes[b])


def build_sparse_topology(rng: random.Random, config: LevelConfig, count: int) -> dict:
    ids = _IdFactory()
    if config.level == 1:
        modules = rng.choice([n for n in (2, 3) if 6 * n <= count])
        sizes = [6] * modules
        for _ in range(count - sum(sizes)):
            sizes[rng.randrange(modules)] += 1
        block = compose_series(
            [
                _block(
                    rng,
                    ids,
                    size,
                    False,
                    config.module_node_fraction,
                    config.resistor_values,
                )
                for size in sizes
            ]
        )
    else:
        block = _block(rng, ids, count, True, value_pool=config.resistor_values)
    return randomize_labels(relabel_circuit(block), rng)


def select_sparse_pairs(
    rng: random.Random,
    topology: dict,
    attempts: int = 120,
    *,
    budget: int | None = None,
) -> list[tuple[str, str]] | None:
    """Draw balanced random subsets; do not seed them with physical edges.

    Acceptance checks prevent rank deficiency, full probe stars and publishing
    the whole adjacency support. Observability uses the hidden topology offline;
    measurement identities remain unrestricted candidate connections publicly.
    """
    pairs = all_node_pairs(topology)
    count = len(topology["resistors"])
    budget = measurement_budget(count) if budget is None else budget
    if budget > len(pairs) // 2:
        return None
    nodes = sorted({node for pair in pairs for node in pair})
    edges = {tuple(sorted(c[:2])) for c in topology["connections"]}
    response = _measurement_jacobian(topology, pairs)
    indexes = {pair: i for i, pair in enumerate(pairs)}
    for _ in range(attempts):
        chosen = [
            ("A", "B"),
            *rng.sample([p for p in pairs if p != ("A", "B")], budget - 1),
        ]
        coverage = {node: sum(node in pair for pair in chosen) for node in nodes}
        if min(coverage.values()) < 2 or max(coverage.values()) >= len(nodes) - 1:
            continue
        if len(edges - set(chosen)) < math.ceil(count / 3):
            continue
        jacobian = response[[indexes[pair] for pair in chosen]]
        if _jacobian_rank(jacobian) != count:
            continue
        selected = sorted(chosen)
        if rounding_stability(topology, selected, value_pool=None)["stable"]:
            return selected
    return None


def fit_known_wiring(
    topology: dict,
    measurements: list[dict],
    start: list[float] | None = None,
    *,
    respect_rounding: bool = False,
) -> dict:
    """Offline continuous fit; optionally solve rounded reading intervals.

    Generation retains its original least-squares audit for reproducibility.
    Interval refinement is an additional feasibility diagnostic.
    """
    candidate = {
        "resistors": dict(topology["resistors"]),
        "connections": [list(c) for c in topology["connections"]],
    }
    ids = list(candidate["resistors"])
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    targets = np.asarray([m["resistance"] for m in measurements])

    def assign(values):
        candidate["resistors"] = dict(zip(ids, np.exp(values), strict=True))

    def residual(values):
        assign(values)
        return _network_response(candidate, pairs)[0] - targets

    def jacobian(values):
        assign(values)
        return _measurement_jacobian(candidate, pairs)

    result = least_squares(
        residual,
        np.log(start if start is not None else [47.0] * len(ids)),
        jac=jacobian,
        bounds=(math.log(1e-3), math.log(1e8)),
        max_nfev=200,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    # Rounded observations describe intervals. A least-squares optimum can
    # violate one interval even when a feasible circuit exists. Refine against
    # interval violations before declaring the wiring inconsistent.
    if respect_rounding and np.max(np.abs(result.fun)) > MEASUREMENT_ABS_TOLERANCE:
        interior_tolerance = MEASUREMENT_ABS_TOLERANCE - 1e-8

        def interval_residual(values):
            errors = residual(values)
            return np.sign(errors) * np.maximum(np.abs(errors) - interior_tolerance, 0)

        def interval_jacobian(values):
            active = np.abs(residual(values)) > interior_tolerance
            return jacobian(values) * active[:, None]

        result = least_squares(
            interval_residual,
            result.x,
            jac=interval_jacobian,
            bounds=(math.log(1e-3), math.log(1e8)),
            max_nfev=100,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-14,
        )
    assign(result.x)
    return {
        "topology": candidate,
        "success": bool(result.success),
        "max_absolute_error": float(np.max(np.abs(residual(result.x)))),
        "evaluations": int(result.nfev),
    }


def audit_sparse_ambiguity(
    rng: random.Random,
    topology: dict,
    measurements: list[dict],
    value_pool: tuple[float, ...] | None = None,
) -> dict:
    """Bounded joint refits after rewiring onto ANY visible node pair."""
    pairs = all_node_pairs(topology)
    edges = {tuple(sorted(c[:2])) for c in topology["connections"]}
    moves = [
        (i, pair)
        for i in range(len(topology["connections"]))
        for pair in pairs
        if pair not in edges
    ]
    rng.shuffle(moves)
    report = {
        "ambiguous": False,
        "exhaustive": False,
        "candidates_available": len(moves),
        "candidates_checked": 0,
        "limit": MAX_REWIRE_AUDIT,
        "scope": "Random single-edge rewires over all visible pairs, jointly refitted continuous values from two starts"
        + (
            ", then snapped to the published value pool"
            if value_pool is not None
            else ""
        )
        + "; not a uniqueness proof.",
        "witness": None,
    }
    for index, pair in moves[:MAX_REWIRE_AUDIT]:
        candidate = {
            "resistors": dict(topology["resistors"]),
            "connections": [list(c) for c in topology["connections"]],
        }
        candidate["connections"][index][:2] = pair
        graph = nx.Graph([c[:2] for c in candidate["connections"]])
        if len(graph) != len({n for p in pairs for n in p}) or not nx.is_connected(
            graph
        ):
            continue
        report["candidates_checked"] += 1
        for start in (list(topology["resistors"].values()), None):
            fitted = fit_known_wiring(
                candidate, measurements, start, respect_rounding=value_pool is not None
            )
            error = fitted["max_absolute_error"]
            if value_pool is not None:
                fitted["topology"]["resistors"] = {
                    rid: min(
                        value_pool, key=lambda choice: abs(math.log(choice / value))
                    )
                    for rid, value in fitted["topology"]["resistors"].items()
                }
                targets = np.asarray([m["resistance"] for m in measurements])
                measured_pairs = [(m["node_a"], m["node_b"]) for m in measurements]
                error = float(
                    np.max(
                        np.abs(
                            _network_response(fitted["topology"], measured_pairs)[0]
                            - targets
                        )
                    )
                )
            if error <= MEASUREMENT_ABS_TOLERANCE:
                report.update(ambiguous=True, witness=fitted["topology"])
                return report
    return report


def sample_sparse_circuit(
    rng: random.Random, config: LevelConfig, count: int, max_attempts: int = 200
) -> tuple[dict, list[dict], dict]:
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not config.resistor_range[0] <= count <= config.resistor_range[1]
    ):
        raise ValueError(
            f"Level {config.level} circuits require {config.resistor_range[0]}-{config.resistor_range[1]} resistors"
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for _ in range(max_attempts):
        topology = build_sparse_topology(rng, config, count)
        if not _validate_sampled_topology(topology, config, count):
            continue
        budget = count + math.ceil(count / config.measurement_redundancy_divisor)
        pairs = select_sparse_pairs(rng, topology, budget=budget)
        if pairs is None:
            continue
        measurements = measure_pairs(topology, pairs)
        if find_non_load_bearing_resistors(topology, measurements):
            continue
        fitted = fit_known_wiring(
            topology, measurements, respect_rounding=config.resistor_values is not None
        )
        if not fitted["success"] or any(
            abs(fitted["topology"]["resistors"][rid] / value - 1)
            > SCORING_TOLERANCE / 4
            for rid, value in topology["resistors"].items()
        ):
            continue
        if config.resistor_values is not None:
            snapped = {
                "resistors": {
                    rid: min(
                        config.resistor_values,
                        key=lambda choice: abs(math.log(choice / value)),
                    )
                    for rid, value in fitted["topology"]["resistors"].items()
                },
                "connections": fitted["topology"]["connections"],
            }
            if measure_pairs(snapped, pairs) != measurements:
                continue
        audit = audit_sparse_ambiguity(
            rng, topology, measurements, value_pool=config.resistor_values
        )
        # Reject demonstrated ambiguities; unseen equivalent networks remain
        # valid submissions rather than being penalized by the hidden answer.
        if audit["ambiguous"]:
            continue
        audit["rounding_stability"] = rounding_stability(
            topology,
            pairs,
            value_pool=list(config.resistor_values)
            if config.resistor_values is not None
            else None,
        )
        audit["known_wiring_fit"] = {
            key: value for key, value in fitted.items() if key != "topology"
        }
        return topology, measurements, audit
    raise RuntimeError(
        f"Could not sample an observable level-{config.level} circuit with {count} resistors after {max_attempts} attempts"
    )


def build_sparse_task(
    rng: random.Random, config: LevelConfig, index: int, count: int
) -> dict:
    topology, measurements, audit = sample_sparse_circuit(rng, config, count)
    nodes = sorted({n for c in topology["connections"] for n in c[:2]})
    task = {
        "id": f"task_{index}",
        "name": f"task_{index}",
        "description": "Infer a full resistor network consistent with the effective-resistance measurements. Resistances are in ohms.",
        "tools": list(TOOLS),
        "scoring_function": "sparse_circuit_solution",
        "scoring_params": {
            "expected_topology": topology,
            "expected_measurements": measurements,
            "measurement_tolerance": MEASUREMENT_ABS_TOLERANCE,
        },
        "submission_format": SUBMISSION_FORMAT,
        "input_from_tasks": [],
        "initial_input": {
            "measurements": measurements,
            "notes": [
                f"The circuit contains {count} positive, finite, ideal resistors and exactly the nodes appearing in the measurements. It is connected, with no self-loops or parallel edges.",
                "Measurements are rounded to 3 decimal places. Any node pair may have a resistor; resistor IDs are arbitrary.",
                "Submit all resistors and nodes. Any circuit satisfying these constraints and matching EVERY supplied effective resistance within 0.000500001 ohms is accepted.",
            ],
        },
        "uuid": str(uuid.uuid4()),
        "level": config.level,
        "num_resistors": count,
        "sampling_metadata": {
            "benchmark_version": BENCHMARK_VERSION,
            "topology_family": config.topology_family,
            "ambiguity_audit": audit,
            "measurement_policy": {
                "budget": count
                + math.ceil(count / config.measurement_redundancy_divisor),
                "formula": f"m + ceil(m / {config.measurement_redundancy_divisor})",
                "min_measurements_per_node": 2,
                "candidate_connections": "all pairs of visible nodes",
                "selection": "random balanced subsets, screened for local observability; no physical-edge support or probe star",
                "unmeasured_physical_edges": sum(
                    tuple(sorted(c[:2]))
                    not in {(m["node_a"], m["node_b"]) for m in measurements}
                    for c in topology["connections"]
                ),
                "nodes": len(nodes),
            },
        },
    }
    if config.resistor_values is not None:
        task["scoring_params"]["allowed_resistor_values"] = list(config.resistor_values)
        task["initial_input"]["notes"].append(
            f"Each resistor must have a value from this list (ohms): {list(config.resistor_values)}. Values may repeat; no per-value inventory is provided."
        )
        task["sampling_metadata"]["value_pool"] = list(config.resistor_values)
    return task
