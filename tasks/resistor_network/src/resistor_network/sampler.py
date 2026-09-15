"""Generate resistor-network tasks."""

import itertools
import json
import math
import random
import uuid
from dataclasses import dataclass

import numpy as np
from resistor_network.utils import get_resistance_between_nodes

RESISTOR_POOL: list[float] = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100]

SCORING_TOLERANCE: float = 0.1

SENSITIVITY_OPEN_MULTIPLIER: float = 1e4
SENSITIVITY_SHORT_MULTIPLIER: float = 1e-4

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


def _measurement_jacobian(
    topology: dict, pairs: list[tuple[str, str]], step: float = 1e-5
) -> np.ndarray:
    """Estimate measurement sensitivity to resistor values."""
    topology_json = json.dumps(topology)
    resistor_ids = list(topology["resistors"])
    jacobian = np.zeros((len(pairs), len(resistor_ids)), dtype=float)

    for column, resistor_id in enumerate(resistor_ids):
        value = topology["resistors"][resistor_id]
        low = json.loads(topology_json)
        high = json.loads(topology_json)
        low["resistors"][resistor_id] = value * math.exp(-step)
        high["resistors"][resistor_id] = value * math.exp(step)
        low_json = json.dumps(low)
        high_json = json.dumps(high)
        for row, pair in enumerate(pairs):
            low_measurement = get_resistance_between_nodes(low_json, list(pair))
            high_measurement = get_resistance_between_nodes(high_json, list(pair))
            jacobian[row, column] = (high_measurement - low_measurement) / (2 * step)

    return jacobian


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


def select_published_pairs(
    rng: random.Random,
    topology: dict,
    redundancy: int = 2,
) -> list[tuple[str, str]]:
    """Select node pairs with full local rank and small redundancy."""
    pairs = all_node_pairs(topology)
    if not pairs:
        return []

    jacobian = _measurement_jacobian(topology, pairs)
    num_unknowns = len(topology["resistors"])
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    chosen: set[tuple[str, str]] = {("A", "B")} if ("A", "B") in pairs else set()

    # Cover every node before selecting independent measurements.
    covered: set[str] = {n for pair in chosen for n in pair}
    for pair in shuffled:
        if pair[0] not in covered or pair[1] not in covered:
            chosen.add(pair)
            covered.update(pair)

    def current_rank(selected: set[tuple[str, str]]) -> int:
        indexes = [pairs.index(pair) for pair in selected]
        return _jacobian_rank(jacobian[indexes, :])

    # Add pairs greedily by rank increase.
    while current_rank(chosen) < num_unknowns:
        candidates = [pair for pair in shuffled if pair not in chosen]
        if not candidates:
            break
        ranked = [(current_rank(chosen | {pair}), pair) for pair in candidates]
        best_rank = max(rank for rank, _pair in ranked)
        best_pairs = [pair for rank, pair in ranked if rank == best_rank]
        chosen.add(rng.choice(best_pairs))

    # Add a small rounding margin.
    remaining = [pair for pair in shuffled if pair not in chosen]
    rng.shuffle(remaining)
    chosen.update(remaining[: max(0, redundancy)])
    return sorted(chosen)


@dataclass
class LevelConfig:
    """Sampling rules for one difficulty level."""

    level: int
    resistor_range: tuple[int, int]
    max_depth: int
    allow_bridge: bool
    min_nodes: int
    min_cycle_rank: int
    measurement_redundancy: int = 2


LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(
        level=1,
        resistor_range=(8, 12),
        max_depth=3,
        allow_bridge=False,
        min_nodes=6,
        min_cycle_rank=1,
    ),
    2: LevelConfig(
        level=2,
        resistor_range=(10, 16),
        max_depth=5,
        allow_bridge=True,
        min_nodes=7,
        min_cycle_rank=2,
    ),
}


def _min_required_nodes(num_resistors: int) -> int:
    """Return the minimum node count for small samples."""
    return min(num_resistors + 1, 4)


def _sampled_topology_quality(topology: dict) -> tuple[int, int, int, int]:
    """Return canonical graph complexity metrics."""
    edges = topology["connections"]
    nodes = {node for a, b, _ in edges for node in (a, b)}
    degrees = {node: 0 for node in nodes}
    pairs: dict[tuple[str, str], int] = {}
    for a, b, _ in edges:
        degrees[a] += 1
        degrees[b] += 1
        pair = tuple(sorted((a, b)))
        pairs[pair] = pairs.get(pair, 0) + 1
    cycle_rank = len(pairs) - len(nodes) + 1
    parallel_groups = sum(count > 1 for count in pairs.values())
    branch_nodes = sum(
        degree >= 3 for node, degree in degrees.items() if node not in {"A", "B"}
    )
    return cycle_rank, parallel_groups, branch_nodes, len(nodes)


def _validate_sampled_topology(
    topology: dict, config: LevelConfig, num_resistors: int
) -> bool:
    """Validate a sampled topology."""
    resistors = topology.get("resistors")
    connections = topology.get("connections")
    if (
        not isinstance(resistors, dict)
        or not resistors
        or not isinstance(connections, list)
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
        if rid not in resistors or rid in referenced:
            return False
        if not isinstance(resistors[rid], int | float) or not math.isfinite(
            resistors[rid]
        ):
            return False
        if resistors[rid] <= 0:
            return False
        referenced.add(rid)
        nodes.update((a, b))
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    if referenced != set(resistors) or {"A", "B"} - nodes:
        return False
    in_benchmark_range = num_resistors >= config.resistor_range[0]
    min_nodes = (
        min(config.min_nodes, num_resistors + 1)
        if in_benchmark_range
        else _min_required_nodes(num_resistors)
    )
    min_cycle_rank = (
        min(config.min_cycle_rank, max(0, num_resistors - 1))
        if in_benchmark_range
        else 0
    )
    if len(nodes) < min_nodes:
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

    cycle_rank, _parallel_groups, _branch_nodes, _num_nodes = _sampled_topology_quality(
        topology
    )
    return cycle_rank >= min_cycle_rank


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


def sample_circuit(
    rng: random.Random, config: LevelConfig, num_resistors: int, max_attempts: int = 200
) -> tuple[dict, list[dict]]:
    """Sample a valid circuit and its measurements."""
    best_loadbearing_bad_count: int | None = None

    for _ in range(max_attempts):
        ids = _IdFactory()
        block = build_random_block(
            rng,
            ids,
            target_resistors=num_resistors,
            max_depth=config.max_depth,
            allow_bridge=config.allow_bridge,
        )
        topology = relabel_circuit(block)
        if not _validate_sampled_topology(topology, config, num_resistors):
            continue

        published = select_published_pairs(
            rng, topology, redundancy=config.measurement_redundancy
        )
        measurements = measure_pairs(topology, published)
        bad_ids = find_non_load_bearing_resistors(topology, measurements)
        if bad_ids:
            if (
                best_loadbearing_bad_count is None
                or len(bad_ids) < best_loadbearing_bad_count
            ):
                best_loadbearing_bad_count = len(bad_ids)
            continue

        return topology, measurements

    raise RuntimeError(
        f"Could not sample a valid level-{config.level} circuit with "
        f"{num_resistors} resistors after {max_attempts} attempts "
        f"(best non-load-bearing count: {best_loadbearing_bad_count})"
    )


def build_task(
    rng: random.Random, config: LevelConfig, index: int, num_resistors: int
) -> dict:
    """Build one task definition."""
    topology, measurements = sample_circuit(rng, config, num_resistors)

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
                "Assume ideal resistors. Measurements are exact to 3 decimal places.",
                "Resistor ids (R1, R2, ...) are not assigned in any particular spatial "
                "order; infer both the topology and the values from the measurements.",
                "Not every node pair was measured.",
            ],
        },
        "uuid": str(uuid.uuid4()),
        "level": config.level,
        "num_resistors": len(topology["resistors"]),
    }


def generate_level_tasks(level: int, count: int, seed: int) -> list[dict]:
    """Generate a task batch for one level."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Available levels: {sorted(LEVELS)}")
    if count < 1:
        raise ValueError("count must be >= 1")

    config = LEVELS[level]
    rng = random.Random(seed)
    low, high = config.resistor_range
    tasks = []
    for i in range(count):
        frac = i / (count - 1) if count > 1 else 0.0
        num_resistors = round(low + frac * (high - low))
        tasks.append(build_task(rng, config, i, num_resistors))
    return tasks
