import json
import math
import random
from pathlib import Path

import pytest
from resistor_network.sampler import (
    LEVELS,
    SCORING_TOLERANCE,
    Block,
    _IdFactory,
    _measurement_jacobian,
    build_random_block,
    compose_parallel,
    compose_series,
    compute_all_measurements,
    find_non_load_bearing_resistors,
    generate_level_tasks,
    leaf_bridge,
    leaf_resistor,
    measurement_budget,
    randomize_labels,
    relabel_circuit,
    rounding_stability,
    sample_circuit,
    select_published_pairs,
)
from resistor_network.score import check_sparse_circuit_solution
from resistor_network.sparse import fit_known_wiring
from resistor_network.utils import get_resistance_between_nodes


@pytest.mark.parametrize("level", [1, 2])
def test_saved_bank_known_wiring_fit_passes_every_simulated_reading(level):
    bank = (
        Path(__file__).resolve().parents[1]
        / "environments"
        / f"level_{level}"
        / "tasks_json"
    )
    for path in sorted(bank.glob("*.json")):
        task = json.loads(path.read_text())[0]
        truth = task["scoring_params"]["expected_topology"]
        readings = task["initial_input"]["measurements"]
        fitted = fit_known_wiring(truth, readings, respect_rounding=True)
        pool = task["scoring_params"].get("allowed_resistor_values")
        if pool is not None:
            fitted["topology"]["resistors"] = {
                rid: min(pool, key=lambda value: abs(math.log(value / fitted_value)))
                for rid, fitted_value in fitted["topology"]["resistors"].items()
            }
        assert (
            check_sparse_circuit_solution(
                truth, readings, allowed_resistor_values=pool
            )(json.dumps(fitted["topology"]))
            == 1
        ), path.name


@pytest.fixture(scope="module", params=[1, 2], ids=["level_1", "level_2"])
def benchmark_batch(request):
    """Share expensive, fully audited batches across independent assertions."""
    level = request.param
    return level, generate_level_tasks(level=level, count=10, seed=7)


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

        # Three resistors in series have two internal joins and two terminals.
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

        # The parallel group and final resistor share one node: A, N1, B.
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


class TestMeasurementSelection:
    def test_fixed_budget_coverage_and_stability(self, benchmark_batch):
        level, tasks = benchmark_batch
        for task in tasks:
            topology = task["scoring_params"]["expected_topology"]
            pairs = [
                (m["node_a"], m["node_b"])
                for m in task["initial_input"]["measurements"]
            ]
            nodes = {
                n for connection in topology["connections"] for n in connection[:2]
            }
            assert len(pairs) == measurement_budget(
                task["num_resistors"], LEVELS[level].measurement_redundancy_divisor
            )
            assert len(pairs) < math.comb(len(nodes), 2)
            assert ("A", "B") in pairs
            assert len(pairs) == len(set(pairs))
            assert all(sum(node in pair for pair in pairs) >= 2 for node in nodes)
            edges = {tuple(sorted(c[:2])) for c in topology["connections"]}
            assert len(edges - set(pairs)) >= math.ceil(task["num_resistors"] / 3)
            assert len(pairs) <= math.comb(len(nodes), 2) // 2
            assert (
                max(sum(node in pair for pair in pairs) for node in nodes)
                < len(nodes) - 1
            )
            assert rounding_stability(topology, pairs, value_pool=None)["stable"]

    def test_rejects_topology_that_cannot_fit_the_fixed_budget(self):
        ids = _IdFactory()
        topology = relabel_circuit(
            compose_series(
                [leaf_resistor(random.Random(seed), ids) for seed in range(4)]
            )
        )
        assert select_published_pairs(random.Random(0), topology) is None

    def test_analytic_jacobian_matches_independent_finite_differences(self):
        topology = relabel_circuit(leaf_bridge(random.Random(5), _IdFactory()))
        pairs = [("A", "B"), ("A", "N1"), ("N1", "N2")]
        analytical = _measurement_jacobian(topology, pairs)
        for column, rid in enumerate(topology["resistors"]):
            low, high = (
                json.loads(json.dumps(topology)),
                json.loads(json.dumps(topology)),
            )
            low["resistors"][rid] *= math.exp(-1e-5)
            high["resistors"][rid] *= math.exp(1e-5)
            for row, pair in enumerate(pairs):
                derivative = (
                    get_resistance_between_nodes(json.dumps(high), list(pair))
                    - get_resistance_between_nodes(json.dumps(low), list(pair))
                ) / 2e-5
                assert analytical[row, column] == pytest.approx(
                    derivative, rel=1e-6, abs=1e-7
                )


class TestBenchmarkStructure:
    @pytest.mark.parametrize(
        ("level", "num_resistors"), [(1, 12), (1, 16), (2, 18), (2, 24)]
    )
    def test_sampled_circuits_have_requested_size(self, level, num_resistors):
        topology, _measurements = sample_circuit(
            random.Random(123), LEVELS[level], num_resistors
        )
        assert (
            len(topology["resistors"]) == len(topology["connections"]) == num_resistors
        )

    @pytest.mark.parametrize("level", [1, 2])
    @pytest.mark.parametrize("num_resistors", [0, 2, 5, 100, True, 12.0])
    def test_sample_circuit_rejects_unsupported_sizes(self, level, num_resistors):
        with pytest.raises(ValueError, match="circuits require"):
            sample_circuit(random.Random(0), LEVELS[level], num_resistors)

    @pytest.mark.parametrize(("level", "num_resistors"), [(1, 30), (2, 26)])
    def test_sample_circuit_rejects_other_levels_size(self, level, num_resistors):
        with pytest.raises(ValueError, match="circuits require"):
            sample_circuit(random.Random(0), LEVELS[level], num_resistors)

    def test_known_wiring_values_are_observable_and_pool_is_published(
        self, benchmark_batch
    ):
        _level, tasks = benchmark_batch
        for task in tasks:
            truth = task["scoring_params"]["expected_topology"]
            fitted = fit_known_wiring(truth, task["initial_input"]["measurements"])
            assert fitted["success"]
            assert all(
                abs(fitted["topology"]["resistors"][rid] / value - 1)
                < SCORING_TOLERANCE / 4
                for rid, value in truth["resistors"].items()
            )
            pool = task["scoring_params"]["allowed_resistor_values"]
            assert len(pool) == 30
            assert set(truth["resistors"].values()) <= set(pool)
            assert str(pool) in " ".join(task["initial_input"]["notes"])

    def test_relabeling_preserves_physics_and_hides_construction_order(self):
        import networkx as nx

        topology = relabel_circuit(leaf_bridge(random.Random(0), _IdFactory()))
        shuffled = randomize_labels(topology, random.Random(7))
        first, second = nx.Graph(), nx.Graph()
        first.add_edges_from(c[:2] for c in topology["connections"])
        second.add_edges_from(c[:2] for c in shuffled["connections"])
        assert nx.is_isomorphic(first, second)
        assert shuffled != topology
        assert get_resistance_between_nodes(
            json.dumps(shuffled), ["A", "B"]
        ) == pytest.approx(
            get_resistance_between_nodes(json.dumps(topology), ["A", "B"])
        )


class TestGenerateLevelTasks:
    def test_generates_ten_tasks_per_level(self, benchmark_batch):
        level, tasks = benchmark_batch
        assert len(tasks) == 10
        assert [t["id"] for t in tasks] == [f"task_{i}" for i in range(10)]
        assert all(t["level"] == level for t in tasks)

    @pytest.mark.parametrize("level", [1, 2])
    @pytest.mark.parametrize("count", [-1, 0, 1, 5, 8, 9, 11, 18, True, 10.0])
    def test_rejects_unsupported_batch_counts(self, level, count):
        with pytest.raises(
            ValueError, match="Each benchmark level must contain exactly 10"
        ):
            generate_level_tasks(level=level, count=count, seed=1)

    def test_sizes_span_configured_range_and_graphs_are_distinct(self, benchmark_batch):
        import networkx as nx

        level, tasks = benchmark_batch
        sizes = [task["num_resistors"] for task in tasks]
        assert min(sizes) == LEVELS[level].resistor_range[0]
        assert max(sizes) == LEVELS[level].resistor_range[1]
        graphs = []
        for task in tasks:
            topology = task["scoring_params"]["expected_topology"]
            graph = nx.Graph([c[:2] for c in topology["connections"]])
            assert not any(nx.is_isomorphic(graph, previous) for previous in graphs)
            graphs.append(graph)

    def test_initial_input_states_resistor_count(self, benchmark_batch):
        _level, tasks = benchmark_batch
        for task in tasks:
            assert (
                f"The circuit contains {task['num_resistors']} positive, finite, ideal resistors"
                in " ".join(task["initial_input"]["notes"])
            )

    def test_deterministic_given_same_seed(self, benchmark_batch):
        def strip_uuid(tasks):
            return [{k: v for k, v in t.items() if k != "uuid"} for t in tasks]

        level, tasks_a = benchmark_batch
        tasks_b = generate_level_tasks(level=level, count=10, seed=7)
        assert strip_uuid(tasks_a) == strip_uuid(tasks_b)

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown level"):
            generate_level_tasks(level=99, count=10, seed=1)

    def test_scoring_params_target_all_published_measurements(self, benchmark_batch):
        _level, tasks = benchmark_batch
        for t in tasks:
            assert t["scoring_function"] == "sparse_circuit_solution"
            sp = t["scoring_params"]
            assert set(sp) == {
                "expected_topology",
                "expected_measurements",
                "measurement_tolerance",
                "allowed_resistor_values",
            }
            assert sp["expected_measurements"] == t["initial_input"]["measurements"]
            assert sp["measurement_tolerance"] == pytest.approx(0.000500001)
            assert sp["expected_topology"]["connections"]

    def test_records_seed_and_bounded_ambiguity_audit(self, benchmark_batch):
        level, tasks = benchmark_batch
        for task in tasks:
            metadata = task["sampling_metadata"]
            assert metadata["batch_seed"] == 7
            assert metadata["topology_family"] == LEVELS[level].topology_family
            audit = metadata["ambiguity_audit"]
            assert not audit["exhaustive"]
            assert not audit["ambiguous"]
            assert audit["witness"] is None
            assert 0 < audit["candidates_checked"] <= audit["limit"]
            assert audit["known_wiring_fit"]["success"]
            assert audit["rounding_stability"]["stable"]
            assert "all visible pairs" in audit["scope"]
            assert metadata["benchmark_version"] == 4


class TestLoadBearingResistors:
    """Check that published measurements respond to every resistor.

    This sampling check rejects branches masked by other resistors. Submission
    scoring separately requires the expected resistor count and conductance map.
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

    @pytest.mark.parametrize(("level", "num_resistors"), [(1, 16), (2, 24)])
    def test_sample_circuit_never_returns_a_non_load_bearing_resistor(
        self, level, num_resistors
    ):
        rng = random.Random(4242)
        topology, published = sample_circuit(rng, LEVELS[level], num_resistors)

        # Pinned against the subset the agent is shown -- the guarantee that
        # matters. It implies the same against the full set, since extra
        # measurements can only make a perturbation easier to detect.
        assert find_non_load_bearing_resistors(topology, published) == []
        assert (
            find_non_load_bearing_resistors(
                topology, compute_all_measurements(topology)
            )
            == []
        )

    def test_generate_level_tasks_produce_fully_load_bearing_circuits(
        self, benchmark_batch
    ):
        _level, tasks = benchmark_batch
        for task in tasks:
            topo = task["scoring_params"]["expected_topology"]
            meas = task["initial_input"]["measurements"]
            assert find_non_load_bearing_resistors(topo, meas) == []

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
