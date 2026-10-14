"""Regression checks for the independent-cycle gap between benchmark levels."""

import json
import random
from pathlib import Path

import networkx as nx
import pytest
from resistor_network.env import get_scoring_function
from resistor_network.sampler import (
    LEVELS,
    LevelConfig,
    _IdFactory,
    _jacobian_rank,
    _measurement_jacobian,
    _sampled_topology_quality,
    _validate_sampled_topology,
    build_coupled_bridge,
    compose_parallel,
    compose_series,
    find_non_load_bearing_resistors,
    generate_level_tasks,
    leaf_resistor,
    measure_pairs,
    measurement_budget,
    relabel_circuit,
    sample_circuit,
)

LEGACY_ONE = LevelConfig(1, "separable_loops", (8, 12))
LEGACY_TWO = LevelConfig(2, "coupled_bridge", (10, 14), max_tail_edges=0)


def graph(topology):
    result = nx.Graph()
    result.add_edges_from((a, b) for a, b, _ in topology["connections"])
    return result


def test_level_one_rejects_coupled_cycles_with_matching_size():
    topology = relabel_circuit(
        build_coupled_bridge(random.Random(42), _IdFactory(), (2, 2, 2, 3, 3))
    )
    network = graph(topology)
    assert network.number_of_edges() == 12
    assert network.number_of_nodes() == 11
    assert len(nx.cycle_basis(network)) == 2
    assert nx.is_biconnected(network)
    assert not _validate_sampled_topology(topology, LEGACY_ONE, 12)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_level_two_samples_have_one_irreducible_coupled_core(seed):
    topology, measurements = sample_circuit(random.Random(seed), LEVELS[2], 18)
    network = graph(topology)
    assert network.number_of_edges() == 18
    assert min(dict(network.degree()).values()) >= 3
    assert nx.is_biconnected(network)
    quality = _sampled_topology_quality(topology)
    assert quality.coupled_core_count == 1
    assert quality.tail_edges == 0
    assert set(quality.reduced_edges) != {("A", "B")}
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    assert _jacobian_rank(_measurement_jacobian(topology, pairs)) == 18


def test_wheatstone_primitive_remains_available():
    topology = relabel_circuit(
        build_coupled_bridge(random.Random(42), _IdFactory(), (2, 2, 2, 2, 2))
    )
    assert len(nx.cycle_basis(graph(topology))) == 2
    assert _validate_sampled_topology(topology, LEGACY_TWO, 10)


def test_multiple_cycles_are_insufficient_without_irreducibility():
    rng, ids = random.Random(42), _IdFactory()
    topology = relabel_circuit(
        compose_parallel(
            [
                compose_series([leaf_resistor(rng, ids) for _ in range(3)])
                for _ in range(4)
            ],
            ids,
        )
    )
    assert nx.is_biconnected(graph(topology))
    assert len(nx.cycle_basis(graph(topology))) == 3
    assert not _validate_sampled_topology(topology, LEGACY_TWO, 12)


def test_separate_loop_cannot_be_attached_to_level_two_core():
    rng, ids = random.Random(42), _IdFactory()
    bridge = build_coupled_bridge(rng, ids, (2, 2, 2, 2, 2))
    loop = compose_parallel(
        [
            leaf_resistor(rng, ids),
            compose_series([leaf_resistor(rng, ids) for _ in range(2)]),
        ],
        ids,
    )
    topology = relabel_circuit(compose_series([bridge, loop]))
    assert len(topology["resistors"]) == 13
    assert not _validate_sampled_topology(topology, LEGACY_TWO, 13)


@pytest.mark.parametrize("level", [1, 2])
def test_saved_tasks_obey_level_cycle_requirements(level):
    directory = (
        Path(__file__).resolve().parents[1]
        / "environments"
        / f"level_{level}"
        / "tasks_json"
    )
    paths = sorted(directory.glob("*.json"))
    assert {path.name for path in paths} == {f"task_{i}.json" for i in range(10)}
    config = LEVELS[level]
    for path in paths:
        task = json.loads(path.read_text())[0]
        assert task["id"] == path.stem
        assert task["level"] == level
        topology = task["scoring_params"]["expected_topology"]
        network = graph(topology)
        quality = _sampled_topology_quality(topology)
        if level == 1:
            assert 2 <= quality.coupled_core_count <= 3
            assert all(rank >= 2 for rank in quality.component_cycle_ranks)
        else:
            assert quality.coupled_core_count == 1
            assert set(quality.reduced_edges) != {("A", "B")}
        assert _validate_sampled_topology(topology, config, task["num_resistors"])
        if level == 2:
            assert nx.is_biconnected(network)
        measurements = task["initial_input"]["measurements"]
        assert len(measurements) == measurement_budget(
            task["num_resistors"], config.measurement_redundancy_divisor
        )
        pairs = [(m["node_a"], m["node_b"]) for m in measurements]
        assert measure_pairs(topology, pairs) == measurements
        assert (
            _jacobian_rank(_measurement_jacobian(topology, pairs))
            == task["num_resistors"]
        )
        assert find_non_load_bearing_resistors(topology, measurements) == []
        assert (
            get_scoring_function(task["scoring_function"], task["scoring_params"])(
                json.dumps(topology)
            )
            == 1.0
        )
        audit = task["sampling_metadata"]["ambiguity_audit"]
        assert 0 < audit["candidates_checked"] <= audit["limit"]
        assert not audit["ambiguous"]


@pytest.mark.parametrize("level", [1, 2])
def test_saved_tasks_reject_collapse_and_have_distinct_graphs(level):
    directory = (
        Path(__file__).resolve().parents[1]
        / "environments"
        / f"level_{level}"
        / "tasks_json"
    )
    graphs = []
    for path in sorted(directory.glob("*.json")):
        task = json.loads(path.read_text())[0]
        scorer = get_scoring_function(task["scoring_function"], task["scoring_params"])
        assert (
            scorer(
                json.dumps({"resistors": {"R1": 10}, "connections": [["A", "B", "R1"]]})
            )
            == 0.0
        )
        network = graph(task["scoring_params"]["expected_topology"])
        assert not any(nx.is_isomorphic(network, previous) for previous in graphs)
        graphs.append(network)


@pytest.mark.parametrize("level", [1, 2])
def test_saved_bank_is_reproduced_by_recorded_seed(level):
    directory = (
        Path(__file__).resolve().parents[1]
        / "environments"
        / f"level_{level}"
        / "tasks_json"
    )
    saved = [
        json.loads(path.read_text())[0] for path in sorted(directory.glob("*.json"))
    ]
    seed = saved[0]["sampling_metadata"]["batch_seed"]
    regenerated = generate_level_tasks(level, 10, seed)
    assert [{k: v for k, v in task.items() if k != "uuid"} for task in saved] == [
        {k: v for k, v in task.items() if k != "uuid"} for task in regenerated
    ]
