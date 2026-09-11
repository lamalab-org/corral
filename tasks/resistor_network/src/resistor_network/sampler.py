"""Circuit sampler engine for the resistor_network task.

Instead of hand-curating each benchmark circuit, this module procedurally
assembles bigger resistor networks out of small two-terminal primitives
(a single resistor, or a Wheatstone-bridge motif) by recursively composing
them in series and/or parallel. Node names and resistor ids are kept
globally unique during composition so blocks can always be merged safely,
then relabeled into readable names (A/B terminals, N1.. internal nodes,
R1.. resistors) once the circuit is final.

Ground truth is never hand-computed: every pairwise node-to-node resistance
is obtained by actually simulating the assembled topology with nodal
analysis (`get_resistance_between_nodes`, the same simulator the tools and
scoring functions use), so the measurements handed to the agent are always
consistent with the sampled topology.

`generate_level_tasks` is the entry point used by
`environments/level_N/generate_tasks.py` to produce the static task JSON
files that the server loads.
"""

import itertools
import json
import math
import random
import uuid
from dataclasses import dataclass

from resistor_network.utils import get_resistance_between_nodes

# "Nice" resistor values (E-series flavored), so sampled circuits read like
# real components rather than arbitrary floats.
# Keep values distinguishable without allowing a 1-ohm branch to mask a
# 500-ohm branch in parallel. This E12-like range spans one decade.
RESISTOR_POOL: list[float] = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82, 100]

# Scoring is purely functional (see `check_resistor_topology` in score.py): a submitted
# topology is graded only on whether it reproduces every pairwise measurement within this
# relative tolerance, never on resistor count, ids, or exact values. `build_task` pulls its
# `scoring_params["tolerance"]` from this same constant so the sampler's own load-bearing
# check (below) can never silently drift from the tolerance actually used to grade agents.
SCORING_TOLERANCE: float = 0.1

# Multipliers used to test whether a resistor's value is constrained by the measurements.
# By Rayleigh's monotonicity law, the effective resistance between any two nodes is a
# monotonic function of any single resistor's value with all others held fixed -- so as a
# resistor sweeps from near-zero to near-infinite, every measurement sweeps monotonically
# between its values at those two extremes. Testing exactly these two extremes is therefore
# both necessary and sufficient: if neither escapes `SCORING_TOLERANCE`, no intermediate
# value (doubling, +/-30%, etc.) could either, so checking additional multipliers would add
# cost without adding coverage. These multipliers push every pool value well outside the
# sampled range while remaining finite.
# range in either direction while remaining finite (the simulator requires strictly positive
# resistances).
SENSITIVITY_OPEN_MULTIPLIER: float = 1e4  # proxy for "resistor removed / open circuit"
SENSITIVITY_SHORT_MULTIPLIER: float = 1e-4  # proxy for "resistor shorted to ~0 ohms"

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
    """A two-terminal resistor network fragment.

    `resistors` and node names inside `connections` use placeholder ids
    assigned by `_IdFactory`, guaranteed unique across an entire sampling
    run, so two blocks can always be composed without accidental collisions.
    """

    resistors: dict[str, float]
    connections: list[tuple[str, str, str]]
    terminal_a: str
    terminal_b: str

    @property
    def num_resistors(self) -> int:
        return len(self.resistors)


class _IdFactory:
    """Generates globally-unique placeholder node/resistor ids for one sampling run."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def node(self) -> str:
        return f"_n{next(self._counter)}"

    def resistor(self) -> str:
        return f"_r{next(self._counter)}"


# ---------------------------------------------------------------------------
# Primitive blocks
# ---------------------------------------------------------------------------


def leaf_resistor(rng: random.Random, ids: _IdFactory) -> Block:
    """A single resistor between two fresh terminals."""
    a, b = ids.node(), ids.node()
    rid = ids.resistor()
    return Block({rid: float(rng.choice(RESISTOR_POOL))}, [(a, b, rid)], a, b)


def leaf_bridge(rng: random.Random, ids: _IdFactory) -> Block:
    """A Wheatstone-bridge motif: 5 resistors that cannot be reduced by
    series/parallel combination alone, forcing real nodal analysis (or a
    delta-wye transform) to solve.
    """
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


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose_series(blocks: list[Block]) -> Block:
    """Chain blocks end-to-end: terminal_b of one is identified with terminal_a of the next."""
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
    """Merge blocks between two freshly-created, shared terminals."""
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
    """Split `total` into `parts` positive integers at random cut points."""
    parts = min(parts, total)
    cuts = sorted(rng.sample(range(1, total), parts - 1)) if parts > 1 else []
    boundaries = [0, *cuts, total]
    return [boundaries[i + 1] - boundaries[i] for i in range(parts)]


def build_random_block(
    rng: random.Random,
    ids: _IdFactory,
    target_resistors: int,
    max_depth: int,
    allow_bridge: bool,
    bridge_prob: float = 0.35,
) -> Block:
    """Recursively compose primitives in series/parallel to reach exactly
    `target_resistors` resistors, within roughly `max_depth` levels of nesting.
    """
    if target_resistors <= 1:
        return leaf_resistor(rng, ids)

    if max_depth <= 0:
        # Spend the remaining resistors as one flat bank when the depth budget
        # is exhausted.
        leaves = [leaf_resistor(rng, ids) for _ in range(target_resistors)]
        mode = rng.choice(["series", "parallel"])
        return (
            compose_series(leaves)
            if mode == "series"
            else compose_parallel(leaves, ids)
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

    branches = rng.randint(2, min(4, target_resistors))
    sizes = _partition(rng, target_resistors, branches)
    children = [
        build_random_block(rng, ids, size, max_depth - 1, allow_bridge, bridge_prob)
        for size in sizes
    ]
    mode = rng.choice(["series", "parallel"])
    return (
        compose_series(children)
        if mode == "series"
        else compose_parallel(children, ids)
    )


# ---------------------------------------------------------------------------
# Relabeling + ground truth simulation
# ---------------------------------------------------------------------------


def relabel_circuit(block: Block) -> dict:
    """Turn placeholder ids into readable names: terminals A/B, internal
    nodes N1.., resistors R1.. (assigned in first-seen order).
    """
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


def compute_all_measurements(topology: dict) -> list[dict]:
    """Ground truth: simulate every pairwise node-to-node resistance via
    nodal analysis (`get_resistance_between_nodes`), the same simulator the
    task's own tools and scoring functions use.
    """
    nodes = sorted({n for conn in topology["connections"] for n in conn[:2]})
    topology_json = json.dumps(topology)
    measurements = []
    for node_a, node_b in itertools.combinations(nodes, 2):
        resistance = get_resistance_between_nodes(topology_json, [node_a, node_b])
        measurements.append(
            {"node_a": node_a, "node_b": node_b, "resistance": round(resistance, 3)}
        )
    return measurements


# ---------------------------------------------------------------------------
# Levels + task assembly
# ---------------------------------------------------------------------------


@dataclass
class LevelConfig:
    level: int
    resistor_range: tuple[int, int]
    max_depth: int
    allow_bridge: bool
    min_nodes: int
    min_cycle_rank: int
    require_parallel: bool
    min_parallel_groups: int


LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(
        level=1,
        resistor_range=(8, 12),
        max_depth=3,
        allow_bridge=False,
        min_nodes=6,
        min_cycle_rank=2,
        require_parallel=True,
        min_parallel_groups=2,
    ),
    2: LevelConfig(
        level=2,
        resistor_range=(10, 16),
        max_depth=5,
        allow_bridge=True,
        min_nodes=7,
        min_cycle_rank=2,
        require_parallel=True,
        min_parallel_groups=1,
    ),
}


def _min_required_nodes(num_resistors: int) -> int:
    """A structural-richness floor so a sampled circuit can never collapse to a
    single equivalent-resistance measurement (e.g. an all-parallel bank between
    just two nodes), which would leave the inference problem trivially
    under-determined for functional scoring: any resistor combo matching that
    one number would pass, regardless of the real topology.

    This is deliberately a small constant, not a target scaling with
    `num_resistors`: it only needs to rule out the fully-collapsed case, not
    push circuits toward any particular shape. A floor that scaled up with
    circuit size (e.g. ~half the resistor count) would, for small circuits,
    approach `num_resistors + 1` -- the node count only a pure series chain
    can reach -- and silently bias every sampled circuit toward series-only
    topologies by rejecting parallel/mixed ones in the retry loop below.
    """
    return min(num_resistors + 1, 4)


def _sampled_topology_quality(topology: dict) -> tuple[int, int, int, int]:
    """Return simple graph-complexity metrics used to choose among candidates."""
    edges = topology["connections"]
    nodes = {node for a, b, _ in edges for node in (a, b)}
    degrees = {node: 0 for node in nodes}
    pairs: dict[tuple[str, str], int] = {}
    for a, b, _ in edges:
        degrees[a] += 1
        degrees[b] += 1
        pair = tuple(sorted((a, b)))
        pairs[pair] = pairs.get(pair, 0) + 1
    cycle_rank = len(edges) - len(nodes) + 1
    parallel_groups = sum(count > 1 for count in pairs.values())
    branch_nodes = sum(degree >= 3 for node, degree in degrees.items() if node not in {"A", "B"})
    return cycle_rank, parallel_groups, branch_nodes, len(nodes)


def _validate_sampled_topology(
    topology: dict, config: LevelConfig, num_resistors: int
) -> bool:
    """Check structural validity and minimum richness before measuring a sample."""
    resistors = topology.get("resistors")
    connections = topology.get("connections")
    if not isinstance(resistors, dict) or not resistors or not isinstance(connections, list):
        return False

    nodes: set[str] = set()
    degrees: dict[str, int] = {}
    referenced: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for connection in connections:
        if not isinstance(connection, list) or len(connection) != 3:
            return False
        a, b, rid = connection
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            return False
        if rid not in resistors or rid in referenced:
            return False
        if not isinstance(resistors[rid], (int, float)) or not math.isfinite(resistors[rid]):
            return False
        if resistors[rid] <= 0:
            return False
        referenced.add(rid)
        nodes.update((a, b))
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        pair = tuple(sorted((a, b)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if referenced != set(resistors) or {"A", "B"} - nodes:
        return False
    # Tests and callers may request toy circuits below the level's benchmark
    # range. Apply the full level contract only to in-range benchmark samples.
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
        return False  # no dangling internal branch

    reachable = {"A"}
    frontier = ["A"]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    if reachable != nodes:
        return False  # no disconnected/open component

    quality = _sampled_topology_quality(topology)
    if quality[1] < min_cycle_rank:
        return False
    parallel_groups = sum(count > 1 for count in pair_counts.values())
    required_parallel_groups = config.min_parallel_groups if in_benchmark_range else 0
    if config.require_parallel and parallel_groups < required_parallel_groups:
        return False
    return True


def _resistor_is_load_bearing(
    topology: dict,
    measurements: list[dict],
    resistor_id: str,
    tolerance: float = SCORING_TOLERANCE,
) -> bool:
    """True iff `resistor_id`'s value is actually constrained by `measurements`.

    Perturbs the resistor to both sensitivity extremes (`SENSITIVITY_OPEN_MULTIPLIER` and
    `SENSITIVITY_SHORT_MULTIPLIER`) and checks whether *each* independently moves at least
    one measurement outside `tolerance` relative error, using the exact same simulator
    (`get_resistance_between_nodes`) and comparison `check_resistor_topology` uses to grade
    submissions. If either extreme survives undetected, the resistor's true value is
    functionally indistinguishable from that extreme (including, for the open-circuit
    extreme, from a submission that omits the resistor entirely) -- it is not load-bearing.
    """
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
                # Solver failure (e.g. near-singular) is itself a maximal deviation
                # from the expected finite measurement.
                return True
            expected = m["resistance"]
            if expected == 0:
                if abs(predicted) > 1e-6:
                    return True
                continue
            if abs(predicted - expected) / expected > tolerance:
                return True
        return False

    # Check the empirically-dominant exploit direction (open circuit) first so the common
    # "resistor is not load-bearing" case short-circuits after a single pass.
    if not _escapes_tolerance(SENSITIVITY_OPEN_MULTIPLIER):
        return False
    return _escapes_tolerance(SENSITIVITY_SHORT_MULTIPLIER)


def find_non_load_bearing_resistors(
    topology: dict, measurements: list[dict], tolerance: float = SCORING_TOLERANCE
) -> list[str]:
    """Ids of every resistor in `topology` that isn't load-bearing given `measurements`
    (see `_resistor_is_load_bearing`). An empty list means the circuit is fully load-bearing:
    no resistor's value can be changed (including toward removing it entirely) without at
    least one measurement moving outside `tolerance`.
    """
    return [
        rid
        for rid in topology["resistors"]
        if not _resistor_is_load_bearing(topology, measurements, rid, tolerance)
    ]


def sample_circuit(
    rng: random.Random, config: LevelConfig, num_resistors: int, max_attempts: int = 200
) -> dict:
    """Sample one relabeled circuit topology for a given level and resistor
    budget, retrying (deterministically, since `rng` keeps advancing) until it
    both clears the structural-richness floor from `_min_required_nodes` and
    has every resistor load-bearing (see `find_non_load_bearing_resistors`) --
    i.e. no resistor's value can be changed toward either extreme without
    detection, which would otherwise let a submission omit or grossly
    misvalue it and still score a perfect match.
    """
    best_candidate: dict | None = None
    best_quality: tuple[int, int, int, int] | None = None
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

        measurements = compute_all_measurements(topology)
        bad_ids = find_non_load_bearing_resistors(topology, measurements)
        if bad_ids:
            if best_loadbearing_bad_count is None or len(bad_ids) < best_loadbearing_bad_count:
                best_loadbearing_bad_count = len(bad_ids)
            continue

        quality = _sampled_topology_quality(topology)
        if best_quality is None or quality > best_quality:
            best_candidate, best_quality = topology, quality

    if best_candidate is not None:
        return best_candidate

    # Do not emit a task that violates the sampler contract.
    raise RuntimeError(
        f"Could not sample a valid level-{config.level} circuit with "
        f"{num_resistors} resistors after {max_attempts} attempts "
        f"(best non-load-bearing count: {best_loadbearing_bad_count})"
    )


def build_task(
    rng: random.Random, config: LevelConfig, index: int, num_resistors: int
) -> dict:
    """Sample one circuit and assemble it into a full task definition dict."""
    topology = sample_circuit(rng, config, num_resistors)
    measurements = compute_all_measurements(topology)

    task_id = f"task_{index}"
    return {
        "id": task_id,
        "name": task_id,
        "description": (
            "Infer the circuit topology and resistor values from node-to-node resistance "
            "measurements."
        ),
        "tools": list(TOOLS),
        "scoring_function": "resistor_topology",
        "scoring_params": {
            "expected_topology": topology,
            "expected_measurements": measurements,
            "tolerance": SCORING_TOLERANCE,
            "use_functional_scoring": True,
            "topology_weight": 0.0,
            "functional_weight": 1,
            "exact_values_weight": 0.0,
        },
        "submission_format": SUBMISSION_FORMAT,
        "input_from_tasks": [],
        "initial_input": {
            "measurements": measurements,
            "notes": [
                "Assume ideal resistors; treat measurements as exact within ±0.1 ohm.",
                "Resistor ids (R1, R2, ...) are not assigned in any particular spatial "
                "order; infer both the topology and the values from the measurements.",
            ],
        },
        "uuid": str(uuid.uuid4()),
        "level": config.level,
        "num_resistors": len(topology["resistors"]),
    }


def generate_level_tasks(level: int, count: int, seed: int) -> list[dict]:
    """Generate `count` tasks for `level`, ramping resistor complexity roughly
    linearly across the batch so later tasks in a level are harder than earlier ones.
    """
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
