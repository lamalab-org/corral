"""Regression checks for the independent-cycle gap between benchmark levels."""

import json
import random
from pathlib import Path

import networkx as nx
import pytest
from resistor_network.sampler import (
    LEVELS,
    _IdFactory,
    _jacobian_rank,
    _measurement_jacobian,
    _sampled_topology_quality,
    _validate_sampled_topology,
    build_coupled_bridge,
    compose_parallel,
    compose_series,
    find_non_load_bearing_resistors,
    leaf_resistor,
    measure_pairs,
    relabel_circuit,
    sample_circuit,
)
from resistor_network.score import check_conductance_topology


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
    assert not _validate_sampled_topology(topology, LEVELS[1], 12)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_level_two_has_three_cycles_in_one_core(seed):
    topology = relabel_circuit(build_coupled_bridge(random.Random(seed), _IdFactory()))
    network = graph(topology)
    assert network.number_of_edges() == len(topology["resistors"]) == 16
    assert network.number_of_nodes() == 14
    assert len(nx.cycle_basis(network)) == 3 > LEVELS[1].cycle_rank
    assert nx.is_biconnected(network)
    assert _validate_sampled_topology(topology, LEVELS[2], 16)

    reduced = nx.Graph()
    reduced.add_edges_from(_sampled_topology_quality(topology).reduced_edges)
    assert nx.is_isomorphic(reduced, nx.complete_graph(4))


def test_original_two_cycle_bridge_is_still_available_but_not_level_two():
    topology = relabel_circuit(
        build_coupled_bridge(random.Random(42), _IdFactory(), (3, 3, 3, 3, 4))
    )
    assert len(topology["resistors"]) == 16
    assert len(nx.cycle_basis(graph(topology))) == 2
    assert not _validate_sampled_topology(topology, LEVELS[2], 16)


def test_three_cycles_with_matching_size_are_insufficient_without_coupling():
    rng, ids = random.Random(42), _IdFactory()
    # Four parallel paths match the resistor/node/cycle counts and form one
    # biconnected component, but still reduce entirely through series/parallel.
    block = compose_parallel(
        [compose_series([leaf_resistor(rng, ids) for _ in range(4)]) for _ in range(4)],
        ids,
    )
    topology = relabel_circuit(block)
    network = graph(topology)
    assert network.number_of_edges() == 16
    assert network.number_of_nodes() == 14
    assert len(nx.cycle_basis(network)) == 3
    assert nx.is_biconnected(network)
    assert not _validate_sampled_topology(topology, LEVELS[2], 16)


def test_third_cycle_cannot_be_in_a_separate_module():
    rng, ids = random.Random(42), _IdFactory()
    bridge = build_coupled_bridge(rng, ids, (2, 2, 2, 2, 2))
    loop = compose_parallel(
        [compose_series([leaf_resistor(rng, ids) for _ in range(3)]) for _ in range(2)],
        ids,
    )
    topology = relabel_circuit(compose_series([bridge, loop]))
    network = graph(topology)
    assert network.number_of_edges() == 16
    assert network.number_of_nodes() == 14
    assert len(nx.cycle_basis(network)) == 3
    assert not nx.is_biconnected(network)
    assert not _validate_sampled_topology(topology, LEVELS[2], 16)


def test_three_cycle_sampling_is_deterministic_and_observable():
    first = sample_circuit(random.Random(42), LEVELS[2], 16)
    second = sample_circuit(random.Random(42), LEVELS[2], 16)
    assert first == second
    topology, measurements = first
    assert nx.is_biconnected(graph(topology))
    pairs = [(m["node_a"], m["node_b"]) for m in measurements]
    assert _jacobian_rank(_measurement_jacobian(topology, pairs)) == 16
    assert find_non_load_bearing_resistors(topology, measurements) == []


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
        assert len(nx.cycle_basis(network)) == config.cycle_rank
        assert _validate_sampled_topology(topology, config, task["num_resistors"])
        if level == 2:
            assert nx.is_biconnected(network)
        measurements = task["initial_input"]["measurements"]
        assert len(measurements) == {1: 14, 2: 18}[level]
        pairs = [(m["node_a"], m["node_b"]) for m in measurements]
        assert measure_pairs(topology, pairs) == measurements
        assert (
            _jacobian_rank(_measurement_jacobian(topology, pairs))
            == task["num_resistors"]
        )
        assert find_non_load_bearing_resistors(topology, measurements) == []
        assert (
            check_conductance_topology(**task["scoring_params"])(json.dumps(topology))
            == 1.0
        )
        audit = task["sampling_metadata"]["ambiguity_audit"]
        assert audit["neighborhood_complete"]
        assert not audit["ambiguous"]
