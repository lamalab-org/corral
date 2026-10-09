"""Generate resistor-network tasks."""

import itertools
import json
import math
import random
import uuid
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from resistor_network.utils import get_resistance_between_nodes
from scipy.optimize import least_squares, minimize

RESISTOR_POOL: list[float] = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100]

SCORING_TOLERANCE: float = 0.02
TASKS_PER_LEVEL: int = 10

SENSITIVITY_OPEN_MULTIPLIER: float = 1e4
SENSITIVITY_SHORT_MULTIPLIER: float = 1e-4
MEASUREMENT_ROUNDING_ERROR = 0.0005
MAX_ROUNDING_GAP_FRACTION = 0.25
MAX_REFIT_CANDIDATES = 48

TOOLS: list[str] = [
    "calculate_series_resistance",
    "calculate_parallel_resistance",
    "delta_to_wye_transform",
    "wye_to_delta_transform",
    "simulate_circuit_resistance",
    "validate_circuit_topology",
    "validate_measurements",
    "propose_simple_topology",
    "estimate_resistor_values",
]

SUBMISSION_FORMAT = (
    'JSON string with circuit topology (e.g., {"resistors": {"R1": x, ...}, '
    '"connections": [["A","B","R1"], ...]})'
)


@dataclass
class Block:
    """A two-terminal resistor-network fragment."""

    resistors: dict[str, float]
    connections: list[tuple[str, str, str]]
    terminal_a: str
    terminal_b: str

    @property
    def num_resistors(self) -> int:
        return len(self.resistors)


class _IdFactory:
    """Generate unique temporary ids."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def node(self) -> str:
        return f"_n{next(self._counter)}"

    def resistor(self) -> str:
        return f"_r{next(self._counter)}"


def leaf_resistor(rng: random.Random, ids: _IdFactory) -> Block:
    """A single resistor between two fresh terminals."""
    a, b = ids.node(), ids.node()
    rid = ids.resistor()
    return Block({rid: float(rng.choice(RESISTOR_POOL))}, [(a, b, rid)], a, b)


def leaf_bridge(rng: random.Random, ids: _IdFactory) -> Block:
    """Create a five-resistor Wheatstone bridge."""
    a, b = ids.node(), ids.node()
    n1, n2 = ids.node(), ids.node()
    rids = [ids.resistor() for _ in range(5)]
    resistors = {rid: float(rng.choice(RESISTOR_POOL)) for rid in rids}
    connections = [
        (a, n1, rids[0]),
        (a, n2, rids[1]),
        (n1, b, rids[2]),
        (n2, b, rids[3]),
        (n1, n2, rids[4]),  # the bridge element
    ]
    return Block(resistors, connections, a, b)


def compose_series(blocks: list[Block]) -> Block:
    """Compose blocks in series."""
    if not blocks:
        raise ValueError("Need at least one block to compose in series")
    merged = blocks[0]
    for nxt in blocks[1:]:
        merged = _series_pair(merged, nxt)
    return merged


def _series_pair(b1: Block, b2: Block) -> Block:
    old, new = b2.terminal_a, b1.terminal_b

    def relabel(n: str) -> str:
        return new if n == old else n

    connections = list(b1.connections) + [
        (relabel(n1), relabel(n2), rid) for n1, n2, rid in b2.connections
    ]
    resistors = {**b1.resistors, **b2.resistors}
    return Block(resistors, connections, b1.terminal_a, b2.terminal_b)


def compose_parallel(blocks: list[Block], ids: _IdFactory) -> Block:
    """Compose blocks in parallel."""
    if not blocks:
        raise ValueError("Need at least one block to compose in parallel")
    a_new, b_new = ids.node(), ids.node()
    resistors: dict[str, float] = {}
    connections: list[tuple[str, str, str]] = []
    for blk in blocks:
        resistors.update(blk.resistors)

        def relabel(n: str, blk: Block = blk) -> str:
            if n == blk.terminal_a:
                return a_new
            if n == blk.terminal_b:
                return b_new
            return n

        connections.extend(
            (relabel(n1), relabel(n2), rid) for n1, n2, rid in blk.connections
        )
    return Block(resistors, connections, a_new, b_new)


def build_separable_loops(
    rng: random.Random,
    ids: _IdFactory,
    path_lengths: tuple[int, ...] = (2, 3),
    loop_modules: int = 2,
) -> Block:
    """Build separate loops in series with one resistor at each end."""
    if len(path_lengths) != 2 or any(
        isinstance(length, bool) or not isinstance(length, int) or length < 2
        for length in path_lengths
    ):
        raise ValueError("Each loop requires two paths of at least two resistors")
    if (
        isinstance(loop_modules, bool)
        or not isinstance(loop_modules, int)
        or loop_modules < 1
    ):
        raise ValueError("loop_modules must be a positive integer")
    blocks = [leaf_resistor(rng, ids)]
    for _ in range(loop_modules):
        paths = [
            compose_series([leaf_resistor(rng, ids) for _ in range(length)])
            for length in path_lengths
        ]
        blocks.append(compose_parallel(paths, ids))
    blocks.append(leaf_resistor(rng, ids))
    return compose_series(blocks)


def subdivide_edges(
    block: Block, rng: random.Random, ids: _IdFactory, path_length: int = 2
) -> Block:
    """Replace every edge by a path, sampling each added resistor independently.

    The original resistor remains in its path. This expands the topology;
    it does not preserve the original edge's equivalent resistance.
    """
    if (
        isinstance(path_length, bool)
        or not isinstance(path_length, int)
        or path_length < 1
    ):
        raise ValueError("path_length must be a positive integer")
    resistors = dict(block.resistors)
    connections = []
    for a, b, rid in block.connections:
        nodes = [a, *(ids.node() for _ in range(path_length - 1)), b]
        resistor_ids = [rid]
        for _ in range(path_length - 1):
            extra_id = ids.resistor()
            resistor_ids.append(extra_id)
            resistors[extra_id] = float(rng.choice(RESISTOR_POOL))
        connections.extend(
            (nodes[i], nodes[i + 1], edge_id) for i, edge_id in enumerate(resistor_ids)
        )
    return Block(resistors, connections, block.terminal_a, block.terminal_b)


def build_coupled_bridge(
    rng: random.Random,
    ids: _IdFactory,
    path_lengths: tuple[int, ...] = (3, 3, 3, 3, 2, 2),
) -> Block:
    """Subdivide a Wheatstone bridge, optionally adding a sixth A-B path.

    The fifth path is the cross-branch. The sixth closes a third independent
    cycle in the same biconnected core, making the reduced graph K4. The default
    lengths keep 16 resistors by shortening the cross-branch for the new path.
    Five path lengths still build the original two-cycle primitive.
    """
    if len(path_lengths) not in {5, 6} or any(
        isinstance(length, bool) or not isinstance(length, int) or length < 2
        for length in path_lengths
    ):
        raise ValueError(
            "A coupled bridge requires five or six paths of at least two resistors"
        )
    bridge = leaf_bridge(rng, ids)
    if len(path_lengths) == 6:
        rid = ids.resistor()
        bridge.resistors[rid] = float(rng.choice(RESISTOR_POOL))
        bridge.connections.append((bridge.terminal_a, bridge.terminal_b, rid))
    resistors = {}
    connections = []
    for (a, b, rid), length in zip(bridge.connections, path_lengths, strict=True):
        edge = Block({rid: bridge.resistors[rid]}, [(a, b, rid)], a, b)
        path = subdivide_edges(edge, rng, ids, length)
        resistors.update(path.resistors)
        connections.extend(path.connections)
    return Block(resistors, connections, bridge.terminal_a, bridge.terminal_b)


def _partition(rng: random.Random, total: int, parts: int) -> list[int]:
    """Split a total into random positive parts."""
    parts = min(parts, total)
    cuts = sorted(rng.sample(range(1, total), parts - 1)) if parts > 1 else []
    boundaries = [0, *cuts, total]
    return [boundaries[i + 1] - boundaries[i] for i in range(parts)]


def _partition_for_parallel(rng: random.Random, total: int, parts: int) -> list[int]:
    """Split resistors into observable parallel branches."""
    parts = min(parts, max(1, total // 2))
    if parts <= 1:
        return [total]
    sizes = [2] * parts
    for _ in range(total - 2 * parts):
        sizes[rng.randrange(parts)] += 1
    if len(sizes) >= 2 and sizes[0] >= 3 and rng.random() < 0.5:
        sizes[0] -= 1
        sizes.append(1)
    return sizes


def build_random_block(
    rng: random.Random,
    ids: _IdFactory,
    target_resistors: int,
    max_depth: int,
    allow_bridge: bool,
    bridge_prob: float = 0.35,
) -> Block:
    """Build a random block with the requested resistor count."""
    if target_resistors <= 1:
        return leaf_resistor(rng, ids)

    if max_depth <= 0:
        return compose_series(
            [leaf_resistor(rng, ids) for _ in range(target_resistors)]
        )

    if allow_bridge and target_resistors >= 5 and rng.random() < bridge_prob:
        bridge = leaf_bridge(rng, ids)
        remainder = target_resistors - bridge.num_resistors
        if remainder <= 0:
            return bridge
        extra = build_random_block(
            rng, ids, remainder, max_depth - 1, allow_bridge, bridge_prob
        )
        pair = [bridge, extra] if rng.random() < 0.5 else [extra, bridge]
        return compose_series(pair)

    mode = rng.choice(["series", "parallel"])
    if mode == "parallel" and target_resistors >= 3:
        branches = rng.randint(2, min(3, max(2, target_resistors // 2)))
        sizes = _partition_for_parallel(rng, target_resistors, branches)
        children = [
            build_random_block(rng, ids, size, max_depth - 1, allow_bridge, bridge_prob)
            for size in sizes
        ]
        return compose_parallel(children, ids)

    branches = rng.randint(2, min(4, target_resistors))
    sizes = _partition(rng, target_resistors, branches)
    children = [
        build_random_block(rng, ids, size, max_depth - 1, allow_bridge, bridge_prob)
        for size in sizes
    ]
    return compose_series(children)


def relabel_circuit(block: Block) -> dict:
    """Relabel temporary nodes and resistors."""
    node_map = {block.terminal_a: "A", block.terminal_b: "B"}
    resistor_map: dict[str, str] = {}
    node_counter = itertools.count(1)
    resistor_counter = itertools.count(1)

    connections: list[list[str]] = []
    for n1, n2, rid in block.connections:
        for n in (n1, n2):
            if n not in node_map:
                node_map[n] = f"N{next(node_counter)}"
        if rid not in resistor_map:
            resistor_map[rid] = f"R{next(resistor_counter)}"
        connections.append([node_map[n1], node_map[n2], resistor_map[rid]])

    resistors = {resistor_map[rid]: val for rid, val in block.resistors.items()}
    return {"resistors": resistors, "connections": connections}


def randomize_labels(topology: dict, rng: random.Random) -> dict:
    """Hide construction order while keeping the two terminals fixed."""
    internal = sorted(
        {n for a, b, _ in topology["connections"] for n in (a, b)} - {"A", "B"}
    )
    shuffled = internal.copy()
    rng.shuffle(shuffled)
    names = dict(zip(internal, shuffled, strict=True))
    ids = list(topology["resistors"])
    shuffled_ids = ids.copy()
    rng.shuffle(shuffled_ids)
    resistor_names = dict(zip(ids, shuffled_ids, strict=True))
    connections = [
        [names.get(a, a), names.get(b, b), resistor_names[rid]]
        for a, b, rid in topology["connections"]
    ]
    rng.shuffle(connections)
    return {
        "resistors": {
            rid: topology["resistors"][old]
            for old, rid in sorted(resistor_names.items(), key=lambda item: item[1])
        },
        "connections": connections,
    }


def _build_varied_loops(rng: random.Random, ids: _IdFactory, count: int) -> Block:
    """A chain or branching cactus of independently separable loops."""
    modules = rng.randint(2, min(4, count // 3))
    tails = rng.randint(0, min(2, count - 3 * modules))
    sizes = [3] * modules
    for _ in range(count - tails - sum(sizes)):
        sizes[rng.randrange(modules)] += 1
    loops = []
    for size in sizes:
        first = rng.randint(1, size - 1)
        loops.append(
            compose_parallel(
                [
                    compose_series([leaf_resistor(rng, ids) for _ in range(length)])
                    for length in (first, size - first)
                ],
                ids,
            )
        )
    if rng.random() < 0.5:
        core = compose_series(loops)
    else:
        hub = ids.node()
        connections, resistors = [], {}
        for loop in loops:
            connections.extend(
                (
                    hub if a == loop.terminal_a else a,
                    hub if b == loop.terminal_a else b,
                    rid,
                )
                for a, b, rid in loop.connections
            )
            resistors.update(loop.resistors)
        core = Block(resistors, connections, loops[0].terminal_b, loops[1].terminal_b)
    before = rng.randint(0, tails)
    return compose_series(
        [
            *[leaf_resistor(rng, ids) for _ in range(before)],
            core,
            *[leaf_resistor(rng, ids) for _ in range(tails - before)],
        ]
    )


def _build_varied_bridge(rng: random.Random, ids: _IdFactory, count: int) -> Block:
    """Subdivide varied irreducible cores, with no external terminal tails."""
    cores = [nx.complete_graph(4), nx.wheel_graph(5), nx.wheel_graph(6)]
    almost_complete = nx.complete_graph(5)
    almost_complete.remove_edge(0, 1)
    cores.append(almost_complete)
    eligible = [g for g in cores if g.number_of_edges() <= count]
    graph = rng.choice(eligible)
    nodes = {n: ids.node() for n in graph.nodes}
    a, b = rng.sample(list(graph.nodes), 2)
    lengths = [1] * graph.number_of_edges()
    for _ in range(count - sum(lengths)):
        lengths[rng.randrange(len(lengths))] += 1
    edges = []
    for (x, y), length in zip(graph.edges, lengths, strict=True):
        rid = ids.resistor()
        edge = Block(
            {rid: float(rng.choice(RESISTOR_POOL))},
            [(nodes[x], nodes[y], rid)],
            nodes[x],
            nodes[y],
        )
        edges.append(subdivide_edges(edge, rng, ids, length))
    return Block(
        {rid: value for edge in edges for rid, value in edge.resistors.items()},
        [connection for edge in edges for connection in edge.connections],
        nodes[a],
        nodes[b],
    )


def _network_response(
    topology: dict, pairs: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Batched nodal solution and analytic derivative with respect to log R."""
    nodes = sorted({n for a, b, _ in topology["connections"] for n in (a, b)})
    indexes = {n: i for i, n in enumerate(nodes[:-1])}
    incidence = np.zeros((len(nodes) - 1, len(topology["connections"])))
    values = []
    for column, (a, b, rid) in enumerate(topology["connections"]):
        if a in indexes:
            incidence[indexes[a], column] = 1
        if b in indexes:
            incidence[indexes[b], column] = -1
        values.append(topology["resistors"][rid])
    values = np.asarray(values)
    covariance = np.linalg.inv((incidence / values) @ incidence.T)
    probes = np.zeros((len(pairs), len(nodes) - 1))
    for row, (a, b) in enumerate(pairs):
        if a in indexes:
            probes[row, indexes[a]] = 1
        if b in indexes:
            probes[row, indexes[b]] = -1
    voltages = probes @ covariance
    resistance = np.sum(voltages * probes, axis=1)
    derivatives = (voltages @ incidence) ** 2 / values
    # Expose columns in resistor-dictionary order, matching the public helper.
    connection_ids = [rid for _, _, rid in topology["connections"]]
    columns = [connection_ids.index(rid) for rid in topology["resistors"]]
    return resistance, derivatives[:, columns]


def measure_pairs(topology: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    """Measure selected node pairs."""
    topology_json = json.dumps(topology)
    return [
        {
            "node_a": node_a,
            "node_b": node_b,
            "resistance": round(
                get_resistance_between_nodes(topology_json, [node_a, node_b]), 3
            ),
        }
        for node_a, node_b in pairs
    ]


def all_node_pairs(topology: dict) -> list[tuple[str, str]]:
    nodes = sorted({n for conn in topology["connections"] for n in conn[:2]})
    return list(itertools.combinations(nodes, 2))


def compute_all_measurements(topology: dict) -> list[dict]:
    """Measure every node pair."""
    return measure_pairs(topology, all_node_pairs(topology))


def _measurement_jacobian(topology: dict, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Analytic sensitivity to log resistor values."""
    return _network_response(topology, pairs)[1]


def _jacobian_rank(jacobian: np.ndarray) -> int:
    """Return the normalized Jacobian rank."""
    if jacobian.size == 0:
        return 0
    column_scales = np.max(np.abs(jacobian), axis=0)
    active = column_scales > 1e-10
    if not np.any(active):
        return 0
    normalized = jacobian[:, active] / column_scales[active]
    return int(np.linalg.matrix_rank(normalized, tol=1e-8))


def _effective_unknown_count(topology: dict) -> int:
    """Count unique node-pair conductances."""
    return len(
        {
            tuple(sorted((node_a, node_b)))
            for node_a, node_b, _resistor_id in topology["connections"]
        }
    )


def _parallel_values_are_unique(
    topology: dict, tolerance: float = SCORING_TOLERANCE
) -> bool:
    """Check uniqueness of allowed values in parallel groups."""
    groups: dict[tuple[str, str], list[str]] = {}
    for node_a, node_b, resistor_id in topology["connections"]:
        pair = tuple(sorted((node_a, node_b)))
        groups.setdefault(pair, []).append(resistor_id)

    for resistor_ids in groups.values():
        if len(resistor_ids) < 2:
            continue
        target_values = tuple(
            sorted(topology["resistors"][resistor_id] for resistor_id in resistor_ids)
        )
        target_conductance = sum(1 / value for value in target_values)
        for candidate in itertools.combinations_with_replacement(
            RESISTOR_POOL, len(resistor_ids)
        ):
            if candidate == target_values:
                continue
            conductance = sum(1 / value for value in candidate)
            if abs(conductance - target_conductance) / target_conductance <= tolerance:
                return False
    return True


def measurement_budget(num_resistors: int) -> int:
    """Static one-third redundancy, independent of model or agent performance."""
    return num_resistors + (num_resistors + 2) // 3


def rounding_stability(topology: dict, pairs: list[tuple[str, str]]) -> dict:
    """Bound first-order log-value error from any three-decimal rounding errors."""
    jacobian = _measurement_jacobian(topology, pairs)
    rank = _jacobian_rank(jacobian)
    if rank != len(topology["resistors"]):
        return {"stable": False, "rank": rank, "max_gap_fraction": None}
    uncertainty = np.abs(np.linalg.pinv(jacobian)) @ np.full(
        len(pairs), MEASUREMENT_ROUNDING_ERROR
    )
    gaps = np.asarray(
        [
            min(
                abs(math.log(other / value))
                for other in RESISTOR_POOL
                if other != value
            )
            for value in topology["resistors"].values()
        ]
    )
    fraction = float(np.max(uncertainty / gaps))
    return {
        "stable": fraction <= MAX_ROUNDING_GAP_FRACTION,
        "rank": rank,
        "max_gap_fraction": fraction,
        "limit": MAX_ROUNDING_GAP_FRACTION,
        "scope": "First-order local bound, not a global uniqueness proof.",
    }


def select_published_pairs(
    rng: random.Random,
    topology: dict,
    redundancy: int | None = None,
) -> list[tuple[str, str]] | None:
    """Publish a probe star, support-containing candidate pairs, and informative decoys.

    Benchmark generation always uses m+ceil(m/3). The optional redundancy
    override is for primitive experiments only; it never changes served banks.
    Every physical edge is in the public candidate-pair set. This constraint
    is explicitly disclosed to the agent and used by the independent solver.
    """
    pairs = all_node_pairs(topology)
    count = len(topology["resistors"])
    budget = measurement_budget(count) if redundancy is None else count + redundancy
    if budget >= len(pairs):
        return None
    graph = nx.Graph()
    graph.add_edges_from((a, b) for a, b, _ in topology["connections"])
    roots = list(graph.nodes)
    rng.shuffle(roots)
    roots.sort(key=lambda node: graph.degree[node], reverse=True)
    required = {tuple(sorted((a, b))) for a, b, _ in topology["connections"]} | {
        ("A", "B")
    }
    chosen = None
    for root in roots:
        proposed = required | {
            tuple(sorted((root, node))) for node in graph if node != root
        }
        if len(proposed) <= budget:
            chosen = proposed
            break
    if chosen is None:
        return None
    jacobian = _measurement_jacobian(topology, pairs)
    gaps = np.asarray(
        [
            min(
                abs(math.log(other / value))
                for other in RESISTOR_POOL
                if other != value
            )
            for value in topology["resistors"].values()
        ]
    )
    scaled = jacobian * gaps
    order = pairs.copy()
    rng.shuffle(order)
    indexes = {pair: i for i, pair in enumerate(pairs)}
    while len(chosen) < budget:
        coverage = {node: sum(node in pair for pair in chosen) for node in graph}

        def information(pair, coverage=coverage):
            rows = [indexes[p] for p in sorted(chosen | {pair})]
            singular = np.linalg.svd(scaled[rows], compute_uv=False)
            weakest = float(singular[-1]) if len(singular) == count else 0.0
            missing = sum(coverage[node] < 2 for node in pair)
            return missing, weakest

        chosen.add(max((pair for pair in order if pair not in chosen), key=information))
    selected = sorted(chosen)
    if any(sum(node in pair for pair in selected) < 2 for node in graph):
        return None
    if not rounding_stability(topology, selected)["stable"]:
        return None
    return selected


def reconstruct_from_measurements(measurements: list[dict], num_resistors: int) -> dict:
    """Recover a circuit using ONLY its public candidate-pair measurements.

    Minimize sum(g_e * measured_R_e) - log(det(grounded_L(g))) over nonnegative
    conductances on published pairs. Its strict convexity in the grounded
    Laplacian gives a unique optimum for unrounded consistent observations
    under the disclosed support constraint. Rounded-input acceptance additionally
    requires discrete snapping and exact agreement at the published precision.
    No hidden wiring or resistor values are supplied to this solver.
    """
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    nodes = sorted({node for pair in pairs for node in pair})
    if len(nodes) < 2 or len(set(pairs)) != len(pairs):
        return {"recovered": False, "reason": "Invalid measurement pairs"}
    indexes = {node: i for i, node in enumerate(nodes[:-1])}
    incidence = np.zeros((len(nodes) - 1, len(pairs)))
    for column, (a, b) in enumerate(pairs):
        if a in indexes:
            incidence[indexes[a], column] = 1
        if b in indexes:
            incidence[indexes[b], column] = -1
    targets = np.asarray([m["resistance"] for m in measurements])

    def objective(conductances):
        laplacian = (incidence * conductances) @ incidence.T
        sign, determinant = np.linalg.slogdet(laplacian)
        if sign <= 0:
            return 1e30, np.zeros_like(conductances)
        try:
            covariance = np.linalg.inv(laplacian)
        except np.linalg.LinAlgError:
            return 1e30, np.zeros_like(conductances)
        predicted = np.sum(incidence * (covariance @ incidence), axis=0)
        return float(targets @ conductances - determinant), targets - predicted

    attempts = []
    for start in (0.02, 0.05):
        result = minimize(
            objective,
            np.full(len(pairs), start),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 0.2)] * len(pairs),
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-7, "maxls": 50},
        )
        selected = [
            (pair, value)
            for pair, value in zip(pairs, result.x, strict=True)
            if value >= 0.5 / max(RESISTOR_POOL)
        ]
        attempts.append(
            {
                "success": bool(result.success),
                "iterations": int(result.nit),
                "selected_resistors": len(selected),
            }
        )
        if len(selected) != num_resistors:
            continue
        topology = {"resistors": {}, "connections": []}
        for index, ((a, b), conductance) in enumerate(selected, 1):
            rid = f"R{index}"
            topology["resistors"][rid] = float(
                min(RESISTOR_POOL, key=lambda value: abs(value - 1 / conductance))
            )
            topology["connections"].append([a, b, rid])
        try:
            matches = measure_pairs(topology, pairs) == measurements
        except ValueError:
            matches = False
        if matches:
            return {
                "recovered": True,
                "topology": topology,
                "attempts": attempts,
                "method": "Convex candidate-pair conductance recovery, then discrete snapping",
            }
    return {
        "recovered": False,
        "attempts": attempts,
        "method": "Convex candidate-pair conductance recovery, then discrete snapping",
    }


@dataclass(frozen=True)
class LevelConfig:
    """Structural families with overlapping sizes and a static measurement policy."""

    level: int
    topology_family: Literal["separable_loops", "coupled_bridge"]
    resistor_range: tuple[int, int]
    max_tail_edges: int = 2


LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(1, "separable_loops", (8, 12)),
    2: LevelConfig(2, "coupled_bridge", (10, 14), max_tail_edges=0),
}


def _min_required_nodes(num_resistors: int) -> int:
    """Return the minimum node count for small samples."""
    return min(num_resistors + 1, 4)


def _biconnected_components(
    adjacency: dict[str, set[str]],
) -> list[set[tuple[str, str]]]:
    """Return edge sets of biconnected components in a simple graph."""
    discovered: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[tuple[str, str]] = []
    components: list[set[tuple[str, str]]] = []

    def visit(node: str, parent: str | None) -> None:
        discovered[node] = low[node] = len(discovered)
        for neighbor in sorted(adjacency[node]):
            edge = tuple(sorted((node, neighbor)))
            if neighbor not in discovered:
                stack.append(edge)
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= discovered[node]:
                    component = {stack.pop()}
                    while edge not in component:
                        component.add(stack.pop())
                    components.append(component)
            elif neighbor != parent and discovered[neighbor] < discovered[node]:
                stack.append(edge)
                low[node] = min(low[node], discovered[neighbor])

    for node in sorted(adjacency):
        if node not in discovered:
            visit(node, None)
    return components


def _series_parallel_reduction(
    edges: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Reduce a graph while preserving A/B and tracking original edge counts.

    Dictionary keys merge parallel connections. Counts include every original
    edge, so subdivision and series/parallel reductions cannot hide long paths.
    """
    reduced = dict(edges)
    while True:
        adjacency: dict[str, set[str]] = {}
        for a, b in reduced:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        node = next(
            (
                n
                for n in sorted(adjacency)
                if n not in {"A", "B"} and len(adjacency[n]) == 2
            ),
            None,
        )
        if node is None:
            return reduced
        a, b = sorted(adjacency[node])
        count = reduced.pop(tuple(sorted((a, node))))
        count += reduced.pop(tuple(sorted((b, node))))
        pair = tuple(sorted((a, b)))
        reduced[pair] = reduced.get(pair, 0) + count


@dataclass(frozen=True)
class TopologyQuality:
    """Final-graph metrics; tail_edges counts edges outside all coupled cores."""

    cycle_rank: int
    parallel_groups: int
    branch_nodes: int
    num_nodes: int
    component_cycle_ranks: tuple[int, ...]
    loop_sizes: tuple[int, ...]
    coupled_core_count: int
    tail_edges: int
    reduced_edges: dict[tuple[str, str], int]


def _sampled_topology_quality(topology: dict) -> TopologyQuality:
    """Measure the final graph, merging parallel pairs for cycle metrics."""
    edges = topology["connections"]
    nodes = {node for a, b, _ in edges for node in (a, b)}
    degrees = dict.fromkeys(nodes, 0)
    pairs: dict[tuple[str, str], int] = {}
    adjacency: dict[str, set[str]] = {}
    for a, b, _ in edges:
        degrees[a] += 1
        degrees[b] += 1
        pair = tuple(sorted((a, b)))
        pairs[pair] = pairs.get(pair, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    cycle_rank = len(pairs) - len(nodes) + 1
    parallel_groups = sum(count > 1 for count in pairs.values())
    branch_nodes = sum(
        degree >= 3 for node, degree in degrees.items() if node not in {"A", "B"}
    )
    components = _biconnected_components(adjacency)
    ranks = [
        len(component) - len({n for edge in component for n in edge}) + 1
        for component in components
    ]
    cores = [
        component for component, rank in zip(components, ranks, strict=True) if rank > 1
    ]
    core_edges = set().union(*cores)
    # Count original resistor edges, not the edges remaining after reduction.
    tail_edges = sum(count for pair, count in pairs.items() if pair not in core_edges)
    return TopologyQuality(
        cycle_rank=cycle_rank,
        parallel_groups=parallel_groups,
        branch_nodes=branch_nodes,
        num_nodes=len(nodes),
        component_cycle_ranks=tuple(sorted(ranks)),
        loop_sizes=tuple(
            sorted(
                len(c) for c, rank in zip(components, ranks, strict=True) if rank == 1
            )
        ),
        coupled_core_count=len(cores),
        tail_edges=tail_edges,
        reduced_edges=_series_parallel_reduction(pairs),
    )


def _validate_sampled_topology(
    topology: dict, config: LevelConfig, num_resistors: int
) -> bool:
    """Validate a sampled topology."""
    if not isinstance(topology, dict):
        return False
    resistors = topology.get("resistors")
    connections = topology.get("connections")
    if (
        not isinstance(resistors, dict)
        or not resistors
        or not isinstance(connections, list)
        or len(resistors) != num_resistors
        or len(connections) != num_resistors
        or not config.resistor_range[0] <= num_resistors <= config.resistor_range[1]
    ):
        return False

    nodes: set[str] = set()
    degrees: dict[str, int] = {}
    referenced: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for connection in connections:
        if not isinstance(connection, list) or len(connection) != 3:
            return False
        a, b, rid = connection
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            return False
        if not isinstance(rid, str) or rid not in resistors or rid in referenced:
            return False
        if (
            isinstance(resistors[rid], bool)
            or not isinstance(resistors[rid], int | float)
            or not math.isfinite(resistors[rid])
        ):
            return False
        if resistors[rid] <= 0 or resistors[rid] not in RESISTOR_POOL:
            return False
        referenced.add(rid)
        nodes.update((a, b))
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    if referenced != set(resistors) or {"A", "B"} - nodes:
        return False
    if not _parallel_values_are_unique(topology):
        return False
    if any(degrees[node] < 2 for node in nodes - {"A", "B"}):
        return False

    reachable = {"A"}
    frontier = ["A"]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    if reachable != nodes:
        return False

    quality = _sampled_topology_quality(topology)
    if quality.parallel_groups:
        return False
    if config.topology_family == "separable_loops":
        return (
            2 <= quality.cycle_rank <= 4
            and all(rank <= 1 for rank in quality.component_cycle_ranks)
            and sum(rank == 0 for rank in quality.component_cycle_ranks)
            <= config.max_tail_edges
        )
    if config.topology_family == "coupled_bridge":
        graph = nx.Graph()
        graph.add_edges_from(adjacency_edge for adjacency_edge in quality.reduced_edges)
        return (
            quality.cycle_rank >= 2
            and quality.coupled_core_count == 1
            and nx.is_biconnected(nx.Graph([(a, b) for a, b, _ in connections]))
            and quality.tail_edges == 0
            and len(graph) >= 4
            and set(quality.reduced_edges) != {("A", "B")}
        )
    return False


def _resistor_is_load_bearing(
    topology: dict,
    measurements: list[dict],
    resistor_id: str,
    tolerance: float = SCORING_TOLERANCE,
) -> bool:
    """Return whether a resistor affects a published measurement."""
    topology_json = json.dumps(topology)
    original_value = topology["resistors"][resistor_id]

    def _escapes_tolerance(multiplier: float) -> bool:
        perturbed = json.loads(topology_json)
        perturbed["resistors"][resistor_id] = original_value * multiplier
        perturbed_json = json.dumps(perturbed)
        for m in measurements:
            try:
                predicted = get_resistance_between_nodes(
                    perturbed_json, [m["node_a"], m["node_b"]]
                )
            except ValueError:
                return True
            expected = m["resistance"]
            if expected == 0:
                if abs(predicted) > 1e-6:
                    return True
                continue
            if abs(predicted - expected) / expected > tolerance:
                return True
        return False

    if not _escapes_tolerance(SENSITIVITY_OPEN_MULTIPLIER):
        return False
    return _escapes_tolerance(SENSITIVITY_SHORT_MULTIPLIER)


def find_non_load_bearing_resistors(
    topology: dict, measurements: list[dict], tolerance: float = SCORING_TOLERANCE
) -> list[str]:
    """Return resistors not constrained by the measurements."""
    return [
        rid
        for rid in topology["resistors"]
        if not _resistor_is_load_bearing(topology, measurements, rid, tolerance)
    ]


def _conductances(topology: dict) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for a, b, rid in topology["connections"]:
        pair = tuple(sorted((a, b)))
        result[pair] = result.get(pair, 0.0) + 1 / topology["resistors"][rid]
    return result


def _refit_rewired_candidates(topology: dict, measurements: list[dict]) -> dict:
    """Audit a bounded, reproducible mix of edge moves and degree-preserving swaps."""
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    allowed = set(pairs)
    connections = topology["connections"]
    present = {tuple(sorted((a, b))) for a, b, _ in connections}
    moves, swaps = [], []
    for index, (_a, _b, rid) in enumerate(connections):
        for a, b in sorted(allowed - present):
            candidate = [list(edge) for edge in connections]
            candidate[index] = [a, b, rid]
            moves.append(candidate)
    for first, second in itertools.combinations(range(len(connections)), 2):
        a, b, rid = connections[first]
        c, d, other_rid = connections[second]
        if len({a, b, c, d}) != 4:
            continue
        for new_edges in (((a, c), (b, d)), ((a, d), (b, c))):
            normalized = {tuple(sorted(edge)) for edge in new_edges}
            if not normalized <= allowed or normalized & present:
                continue
            candidate = [list(edge) for edge in connections]
            candidate[first] = [*new_edges[0], rid]
            candidate[second] = [*new_edges[1], other_rid]
            swaps.append(candidate)
    audit_rng = random.Random(0)
    audit_rng.shuffle(moves)
    audit_rng.shuffle(swaps)
    candidates = moves[: MAX_REFIT_CANDIDATES // 2] + swaps[: MAX_REFIT_CANDIDATES // 2]
    report = {
        "ambiguous": False,
        "witness": None,
        "candidates_available": len(moves) + len(swaps),
        "candidates_checked": 0,
        "complete": len(candidates) == len(moves) + len(swaps),
        "scope": "At most 24 edge moves and 24 two-edge swaps, joint continuous refit from two starts, then allowed-value snapping. Not exhaustive.",
    }
    targets = np.asarray([m["resistance"] for m in measurements])
    resistor_ids = list(topology["resistors"])
    original_nodes = {n for pair in pairs for n in pair}
    for edges in candidates:
        graph = nx.Graph([(a, b) for a, b, _ in edges])
        if set(graph) != original_nodes or not nx.is_connected(graph):
            continue
        report["candidates_checked"] += 1
        candidate = {"resistors": dict(topology["resistors"]), "connections": edges}

        def residual(log_values, candidate=candidate):
            candidate["resistors"] = dict(
                zip(resistor_ids, np.exp(log_values), strict=True)
            )
            return _network_response(candidate, pairs)[0] - targets

        def jacobian(log_values, candidate=candidate):
            candidate["resistors"] = dict(
                zip(resistor_ids, np.exp(log_values), strict=True)
            )
            return _measurement_jacobian(candidate, pairs)

        for start in (list(topology["resistors"].values()), [47.0] * len(resistor_ids)):
            fitted = least_squares(
                residual,
                np.log(start),
                jac=jacobian,
                bounds=(math.log(min(RESISTOR_POOL)), math.log(max(RESISTOR_POOL))),
                max_nfev=150,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            candidate["resistors"] = {
                rid: float(
                    min(
                        RESISTOR_POOL,
                        key=lambda value: abs(value - math.exp(log_value)),
                    )
                )
                for rid, log_value in zip(resistor_ids, fitted.x, strict=True)
            }
            if measure_pairs(candidate, pairs) == measurements:
                report["ambiguous"] = True
                report["witness"] = candidate
                return report
    return report


def audit_measurement_ambiguity(topology: dict, measurements: list[dict]) -> dict:
    """Search a finite neighborhood for score-distinct, measurement-equivalent circuits.

    Candidates keep the visible node set and resistor count, and use only the
    allowed value pool. They are NOT restricted to either benchmark family.
    Search every single-resistor reconnection/value replacement, every pairwise
    value swap, and every pairwise node-label swap, plus bounded joint refits.
    All candidates respect the publicly disclosed measured-pair support.

    Equality means identical resistances after rounding to three decimals, not
    the scoring tolerance. The report records coverage and a concrete witness
    when found. No witness in this neighborhood is not a uniqueness proof.
    """
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    if not measurements or measure_pairs(topology, pairs) != measurements:
        raise ValueError("Ambiguity audit requires consistent published measurements")
    if any(value not in RESISTOR_POOL for value in topology["resistors"].values()):
        raise ValueError("Ambiguity audit requires values from RESISTOR_POOL")

    nodes = sorted({n for a, b, _ in topology["connections"] for n in (a, b)})
    resistor_ids = list(topology["resistors"])
    report = {
        "ambiguous": False,
        "exhaustive": False,
        "neighborhood_complete": False,
        "precision_decimals": 3,
        "value_pool": list(RESISTOR_POOL),
        "scope": "Same visible nodes and resistor count; connections restricted to published pairs, no topology-family restriction. "
        "Single-resistor reconnection/value replacement with other values fixed, "
        "pairwise value swaps, pairwise node-label swaps, plus separately recorded bounded joint refits.",
        "candidates_checked": {
            "single_resistor_changes": 0,
            "value_swaps": 0,
            "node_swaps": 0,
        },
        "invalid_candidates": 0,
        "score_equivalent_candidates": 0,
        "witness": None,
    }

    expected = _conductances(topology)

    def candidates():
        for index, (a, b, rid) in enumerate(topology["connections"]):
            candidate = {
                "resistors": dict(topology["resistors"]),
                "connections": [list(edge) for edge in topology["connections"]],
            }
            for x, y in pairs:
                for value in RESISTOR_POOL:
                    if (x, y) == tuple(sorted((a, b))) and value == topology[
                        "resistors"
                    ][rid]:
                        continue
                    candidate["connections"][index] = [x, y, rid]
                    candidate["resistors"][rid] = float(value)
                    yield "single_resistor_changes", candidate
        for first, second in itertools.combinations(resistor_ids, 2):
            values = dict(topology["resistors"])
            if values[first] == values[second]:
                continue
            values[first], values[second] = values[second], values[first]
            yield (
                "value_swaps",
                {"resistors": values, "connections": topology["connections"]},
            )
        for first, second in itertools.combinations(nodes, 2):
            names = {first: second, second: first}
            yield (
                "node_swaps",
                {
                    "resistors": topology["resistors"],
                    "connections": [
                        [names.get(a, a), names.get(b, b), rid]
                        for a, b, rid in topology["connections"]
                    ],
                },
            )

    for kind, candidate in candidates():
        report["candidates_checked"][kind] += 1
        proposed = _conductances(candidate)
        if not set(proposed) <= set(pairs) or len(proposed) != len(
            candidate["resistors"]
        ):
            report["invalid_candidates"] += 1
            continue
        adjacency: dict[str, set[str]] = {}
        for a, b in proposed:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        reachable = {nodes[0]}
        frontier = [nodes[0]]
        while frontier:
            for neighbor in adjacency.get(frontier.pop(), set()) - reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
        if set(adjacency) != set(nodes) or reachable != set(nodes):
            report["invalid_candidates"] += 1
            continue
        if proposed.keys() == expected.keys() and all(
            abs(proposed[pair] - value) / value <= SCORING_TOLERANCE
            for pair, value in expected.items()
        ):
            report["score_equivalent_candidates"] += 1
            continue
        predictions = _network_response(candidate, pairs)[0]
        if all(
            round(float(value), 3) == m["resistance"]
            for value, m in zip(predictions, measurements, strict=True)
        ):
            report["ambiguous"] = True
            report["witness"] = json.loads(json.dumps(candidate))
            return report
    report["neighborhood_complete"] = True
    report["joint_refit"] = _refit_rewired_candidates(topology, measurements)
    if report["joint_refit"]["ambiguous"]:
        report["ambiguous"] = True
        report["witness"] = report["joint_refit"]["witness"]
    return report


def sample_circuit(
    rng: random.Random, config: LevelConfig, num_resistors: int, max_attempts: int = 200
) -> tuple[dict, list[dict]]:
    """Sample a pilot circuit after structural, measurement and ambiguity checks."""
    topology, measurements, _audit = _sample_circuit(
        rng, config, num_resistors, max_attempts
    )
    return topology, measurements


def _sample_circuit(
    rng: random.Random, config: LevelConfig, num_resistors: int, max_attempts: int = 200
) -> tuple[dict, list[dict], dict]:
    """Return the accepted circuit plus offline ambiguity-search coverage."""
    if (
        isinstance(num_resistors, bool)
        or not isinstance(num_resistors, int)
        or not config.resistor_range[0] <= num_resistors <= config.resistor_range[1]
    ):
        raise ValueError(
            f"Level {config.level} circuits require {config.resistor_range[0]}-{config.resistor_range[1]} resistors"
        )
    if config.topology_family not in {"separable_loops", "coupled_bridge"}:
        raise ValueError(f"Unknown topology family: {config.topology_family}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    best_loadbearing_bad_count: int | None = None
    ambiguous_count = 0

    for _ in range(max_attempts):
        ids = _IdFactory()
        if config.topology_family == "separable_loops":
            block = _build_varied_loops(rng, ids, num_resistors)
        else:
            block = _build_varied_bridge(rng, ids, num_resistors)
        topology = randomize_labels(relabel_circuit(block), rng)
        if not _validate_sampled_topology(topology, config, num_resistors):
            continue

        published = select_published_pairs(rng, topology)
        if published is None:
            continue
        measurements = measure_pairs(topology, published)
        bad_ids = find_non_load_bearing_resistors(topology, measurements)
        if bad_ids:
            if (
                best_loadbearing_bad_count is None
                or len(bad_ids) < best_loadbearing_bad_count
            ):
                best_loadbearing_bad_count = len(bad_ids)
            continue

        recovery = reconstruct_from_measurements(measurements, num_resistors)
        if not recovery["recovered"] or _conductances(
            recovery["topology"]
        ) != _conductances(topology):
            continue
        recovery.pop("topology")
        audit = audit_measurement_ambiguity(topology, measurements)
        audit["reference_recovery"] = recovery
        audit["rounding_stability"] = rounding_stability(topology, published)
        if audit["ambiguous"]:
            ambiguous_count += 1
            continue
        return topology, measurements, audit

    raise RuntimeError(
        f"Could not sample a valid level-{config.level} circuit with "
        f"{num_resistors} resistors after {max_attempts} attempts "
        f"(best non-load-bearing count: {best_loadbearing_bad_count}; "
        f"demonstrated ambiguous candidates: {ambiguous_count})"
    )


def build_task(
    rng: random.Random, config: LevelConfig, index: int, num_resistors: int
) -> dict:
    """Build one task definition."""
    topology, measurements, audit = _sample_circuit(rng, config, num_resistors)

    task_id = f"task_{index}"
    return {
        "id": task_id,
        "name": task_id,
        "description": (
            "Infer the circuit topology and resistor values from node-to-node resistance "
            "measurements."
        ),
        "tools": list(TOOLS),
        "scoring_function": "resistor_conductance",
        "scoring_params": {
            "expected_topology": topology,
            "tolerance": SCORING_TOLERANCE,
        },
        "submission_format": SUBMISSION_FORMAT,
        "input_from_tasks": [],
        "initial_input": {
            "measurements": measurements,
            "notes": [
                f"The circuit contains {len(topology['resistors'])} resistors.",
                f"Allowed resistor values are {RESISTOR_POOL} ohms.",
                "Assume ideal resistors. Measurements are exact to 3 decimal places.",
                (
                    "Resistor ids (R1, R2, ...) are not assigned in any particular spatial "
                    "order; infer both the topology and the values from the measurements."
                ),
                (
                    "Not every node pair was measured. Every physical resistor joins a pair "
                    "that appears in the measurements; some measured pairs have no direct resistor."
                ),
                (
                    "There are no direct parallel resistors. Submit the full circuit with every "
                    "named node and resistor; a smaller equivalent circuit is not accepted."
                ),
                (
                    "The circuit consists of separable loop modules (a cactus graph), with at most two terminal tails."
                    if config.level == 1
                    else "The circuit has one coupled bridge core and no external tails."
                ),
            ],
        },
        "uuid": str(uuid.uuid4()),
        "level": config.level,
        "num_resistors": len(topology["resistors"]),
        "sampling_metadata": {
            "topology_family": config.topology_family,
            "ambiguity_audit": audit,
        },
    }


def generate_level_tasks(level: int, count: int, seed: int) -> list[dict]:
    """Generate ten varied, non-isomorphic tasks with a deterministic size schedule."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Available levels: {sorted(LEVELS)}")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != TASKS_PER_LEVEL
    ):
        raise ValueError(
            f"Each benchmark level must contain exactly {TASKS_PER_LEVEL} tasks"
        )

    config = LEVELS[level]
    rng = random.Random(seed)
    tasks, graphs = [], []
    for index in range(count):
        size = config.resistor_range[0] + round(
            index * (config.resistor_range[1] - config.resistor_range[0]) / (count - 1)
        )
        for _attempt in range(100):
            task = build_task(rng, config, index, size)
            graph = nx.Graph()
            graph.add_edges_from(
                (a, b)
                for a, b, _ in task["scoring_params"]["expected_topology"][
                    "connections"
                ]
            )
            # Uncolored graph isomorphism rejects mere node-label/terminal changes.
            if any(nx.is_isomorphic(graph, previous) for previous in graphs):
                continue
            graphs.append(graph)
            task["sampling_metadata"]["batch_seed"] = seed
            task["sampling_metadata"]["measurement_policy"] = {
                "budget": measurement_budget(size),
                "formula": "m + ceil(m / 3)",
                "min_measurements_per_node": 2,
                "candidate_connections": "published measurement pairs only",
                "selection": "probe star + physical edges + weakest-direction decoys",
            }
            tasks.append(task)
            break
        else:
            raise RuntimeError(
                f"Could not find a structurally distinct level-{level} task of size {size}"
            )
    return tasks
