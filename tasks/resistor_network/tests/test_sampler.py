import json
import math
import random
from pathlib import Path

import pytest
from resistor_network.sampler import (
    SCORING_TOLERANCE,
    LEVELS,
    Block,
    _IdFactory,
    _min_required_nodes,
    build_random_block,
    compose_parallel,
    compose_series,
    compute_all_measurements,
    find_non_load_bearing_resistors,
    generate_level_tasks,
    leaf_bridge,
    leaf_resistor,
    relabel_circuit,
    sample_circuit,
)
from resistor_network.utils import get_resistance_between_nodes


class TestPrimitives:
    def test_leaf_resistor_has_one_resistor(self):
        block = leaf_resistor(random.Random(0), _IdFactory())
        assert block.num_resistors == 1
        assert block.terminal_a != block.terminal_b

    def test_leaf_bridge_has_five_resistors_and_bridge_element(self):
        block = leaf_bridge(random.Random(0), _IdFactory())
        assert block.num_resistors == 5
        nodes = {n for conn in block.connections for n in conn[:2]}
        # 2 terminals + 2 internal bridge nodes
        assert len(nodes) == 4


class TestComposition:
    def test_series_resistance_matches_simulation(self):
        ids = _IdFactory()
        r1 = leaf_resistor(random.Random(1), ids)
        r2 = leaf_resistor(random.Random(2), ids)
        r3 = leaf_resistor(random.Random(3), ids)
        merged = compose_series([r1, r2, r3])
        topology = relabel_circuit(merged)

        expected = sum(topology["resistors"].values())
        simulated = get_resistance_between_nodes(json.dumps(topology), ["A", "B"])
        assert simulated == pytest.approx(expected, rel=1e-9)

        # A series chain of 3 resistors has 4 distinct nodes (3 internal joins... 2 internal)
        nodes = {n for conn in topology["connections"] for n in conn[:2]}
        assert len(nodes) == 4  # A, N1, N2, B

    def test_parallel_resistance_matches_simulation(self):
        ids = _IdFactory()
        r1 = leaf_resistor(random.Random(4), ids)
        r2 = leaf_resistor(random.Random(5), ids)
        r3 = leaf_resistor(random.Random(6), ids)
        merged = compose_parallel([r1, r2, r3], ids)
        topology = relabel_circuit(merged)

        expected = 1 / sum(1 / v for v in topology["resistors"].values())
        simulated = get_resistance_between_nodes(json.dumps(topology), ["A", "B"])
        assert simulated == pytest.approx(expected, rel=1e-9)

        # Pure parallel bank collapses to exactly 2 nodes.
        nodes = {n for conn in topology["connections"] for n in conn[:2]}
        assert len(nodes) == 2

    def test_series_of_parallel_matches_simulation(self):
        ids = _IdFactory()
        group_a = compose_parallel(
            [
                leaf_resistor(random.Random(7), ids),
                leaf_resistor(random.Random(8), ids),
            ],
            ids,
        )
        group_b = leaf_resistor(random.Random(9), ids)
        merged = compose_series([group_a, group_b])
        topology = relabel_circuit(merged)

        simulated = get_resistance_between_nodes(json.dumps(topology), ["A", "B"])
        assert math.isfinite(simulated)
        assert simulated > 0

        # 2 nodes for the parallel group + 2 more for the series join = distinct A, N1, B
        nodes = {n for conn in topology["connections"] for n in conn[:2]}
        assert len(nodes) == 3


class TestGroundTruthSimulation:
    def test_measurements_are_self_consistent_with_topology(self):
        """expected_measurements must always be exactly what the simulator
        reports for expected_topology -- ground truth is never hand-derived.
        """
        rng = random.Random(42)
        ids = _IdFactory()
        block = build_random_block(
            rng, ids, target_resistors=8, max_depth=4, allow_bridge=True
        )
        topology = relabel_circuit(block)
        measurements = compute_all_measurements(topology)

        topology_json = json.dumps(topology)
        for m in measurements:
            resim = get_resistance_between_nodes(
                topology_json, [m["node_a"], m["node_b"]]
            )
            assert resim == pytest.approx(m["resistance"], abs=1e-2)


class TestStructuralRichnessFloor:
    @pytest.mark.parametrize("num_resistors", [2, 3, 4, 5, 7, 10, 15])
    def test_sampled_circuits_meet_minimum_node_floor(self, num_resistors):
        rng = random.Random(123)
        config = LEVELS[2] if num_resistors >= 5 else LEVELS[1]
        topology = sample_circuit(rng, config, num_resistors)
        nodes = {n for conn in topology["connections"] for n in conn[:2]}
        assert len(nodes) >= _min_required_nodes(num_resistors)

    def test_no_circuit_ever_has_a_single_measurement(self):
        """A single A-B measurement can't distinguish between different
        underlying topologies, which would make functional scoring trivial."""
        for level in LEVELS:
            tasks = generate_level_tasks(level=level, count=10, seed=7)
            for task in tasks:
                assert len(task["initial_input"]["measurements"]) >= 3


class TestGenerateLevelTasks:
    def test_generates_requested_count(self):
        tasks = generate_level_tasks(level=1, count=5, seed=1)
        assert len(tasks) == 5
        assert [t["id"] for t in tasks] == [f"task_{i}" for i in range(5)]

    def test_resistor_count_stays_within_level_range(self):
        for level, config in LEVELS.items():
            low, high = config.resistor_range
            tasks = generate_level_tasks(level=level, count=6, seed=3)
            for t in tasks:
                assert low <= t["num_resistors"] <= high

    def test_deterministic_given_same_seed(self):
        def strip_uuid(tasks):
            return [{k: v for k, v in t.items() if k != "uuid"} for t in tasks]

        tasks_a = generate_level_tasks(level=2, count=6, seed=99)
        tasks_b = generate_level_tasks(level=2, count=6, seed=99)
        assert strip_uuid(tasks_a) == strip_uuid(tasks_b)

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            generate_level_tasks(level=99, count=1, seed=1)

    def test_scoring_params_target_the_conductance_map(self):
        tasks = generate_level_tasks(level=1, count=3, seed=1)
        for t in tasks:
            assert t["scoring_function"] == "resistor_conductance"
            sp = t["scoring_params"]
            assert set(sp) == {"expected_topology", "tolerance"}
            assert sp["tolerance"] == SCORING_TOLERANCE
            assert sp["expected_topology"]["connections"]


class TestLoadBearingResistors:
    """Regression coverage for the 'submit a simpler network and get away with
    it' exploit: because scoring is purely functional, a resistor whose value
    doesn't move any measurement outside tolerance is undetectable if
    changed, or omitted entirely. `find_non_load_bearing_resistors` flags
    such resistors so `sample_circuit` can reject circuits containing them.
    """

    def test_dominant_small_resistor_masks_large_parallel_partner(self):
        """Regression test for the exact real-world pattern found in
        level_1/task_10.json before this fix: a 1ohm resistor in parallel
        with a 330ohm resistor dominates the combination so completely that
        the 330ohm value is not recoverable from any measurement."""
        ids = _IdFactory()
        b1 = Block({"Rsmall": 1.0}, [("x1", "y1", "Rsmall")], "x1", "y1")
        b2 = Block({"Rlarge": 330.0}, [("x2", "y2", "Rlarge")], "x2", "y2")
        topology = relabel_circuit(compose_parallel([b1, b2], ids))
        measurements = compute_all_measurements(topology)

        assert find_non_load_bearing_resistors(topology, measurements) == ["R2"]

    def test_series_chain_resistors_are_all_load_bearing(self):
        """No false positives: a plain series chain's resistors are each
        individually observable via their own endpoint measurement."""
        ids = _IdFactory()
        block = compose_series(
            [leaf_resistor(random.Random(s), ids) for s in (1, 2, 3)]
        )
        topology = relabel_circuit(block)
        measurements = compute_all_measurements(topology)

        assert find_non_load_bearing_resistors(topology, measurements) == []

    @pytest.mark.parametrize("num_resistors", [2, 5, 8, 12, 15, 18])
    def test_sample_circuit_never_returns_a_non_load_bearing_resistor(
        self, num_resistors
    ):
        rng = random.Random(4242)
        config = LEVELS[2] if num_resistors >= 5 else LEVELS[1]
        topology = sample_circuit(rng, config, num_resistors)
        measurements = compute_all_measurements(topology)

        assert find_non_load_bearing_resistors(topology, measurements) == []

    def test_generate_level_tasks_produce_fully_load_bearing_circuits(self):
        for level in LEVELS:
            for task in generate_level_tasks(level=level, count=8, seed=11):
                topo = task["scoring_params"]["expected_topology"]
                meas = task["initial_input"]["measurements"]
                assert find_non_load_bearing_resistors(topo, meas) == []

    def test_checked_in_task_json_files_have_no_non_load_bearing_resistors(self):
        """Regression guard for a real coverage gap: none of the other tests in
        this suite load the checked-in tasks_json/*.json files that the live
        server actually serves -- only the generator functions in isolation.
        Requires `environments/level_*/tasks_json/` to be (re)generated with
        this fix in place.
        """
        repo_root = Path(__file__).resolve().parents[1]
        json_paths = sorted(repo_root.glob("environments/level_*/tasks_json/*.json"))
        assert json_paths, "expected checked-in task JSON files to exist"

        for path in json_paths:
            task = json.loads(path.read_text())[0]
            topo = task["scoring_params"]["expected_topology"]
            meas = task["initial_input"]["measurements"]
            bad = find_non_load_bearing_resistors(topo, meas)
            assert bad == [], f"{path.name} has non-load-bearing resistors: {bad}"

    def test_bridge_motifs_remain_samplable_after_the_fix(self):
        """Guards against over-rejection: the load-bearing check must not make
        bridge motifs (the level-2 motif that forces real nodal analysis)
        effectively unsamplable.
        """
        n_trials, n_clean = 200, 0
        for i in range(n_trials):
            ids = _IdFactory()
            topology = relabel_circuit(leaf_bridge(random.Random(9000 + i), ids))
            measurements = compute_all_measurements(topology)
            if not find_non_load_bearing_resistors(topology, measurements):
                n_clean += 1
        assert n_clean / n_trials > 0.5
