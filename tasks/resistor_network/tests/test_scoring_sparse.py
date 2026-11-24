"""V3 simulation scoring must accept equivalents and reject invalid networks."""

import copy
import json

import pytest
from resistor_network.score import check_sparse_circuit_solution
from resistor_network.utils import get_resistance_between_nodes

TRUTH = {
    "resistors": {"r1": 10.125, "r2": 20.25, "r3": 30.375},
    "connections": [["A", "B", "r1"], ["A", "N1", "r2"], ["A", "N2", "r3"]],
}
MEASUREMENTS = [
    {"node_a": "A", "node_b": "B", "resistance": 10.125},
    {"node_a": "N1", "node_b": "N2", "resistance": 50.625},
]


def score(candidate):
    return check_sparse_circuit_solution(TRUTH, MEASUREMENTS)(json.dumps(candidate))


def test_continuous_values_and_different_indistinguishable_wiring_pass():
    alternative = {
        "resistors": {"x": 10.125, "y": 11.74321, "z": 50.625},
        "connections": [["B", "A", "x"], ["A", "N1", "y"], ["N1", "N2", "z"]],
    }
    assert alternative["connections"] != TRUTH["connections"]
    assert score(TRUTH) == score(alternative) == 1
    for m in MEASUREMENTS:
        assert get_resistance_between_nodes(
            json.dumps(alternative), [m["node_a"], m["node_b"]]
        ) == pytest.approx(m["resistance"])


def test_renaming_resistors_and_reversing_edges_preserves_score():
    renamed = {
        "resistors": {"x": 10.125, "y": 20.25, "z": 30.375},
        "connections": [["B", "A", "x"], ["N1", "A", "y"], ["N2", "A", "z"]],
    }
    assert score(renamed) == 1


@pytest.mark.parametrize(
    ("error", "expected"), [(0.00049, 1), (0.00051, 0), (0.001, 0)]
)
def test_absolute_rounding_tolerance_applies_to_each_reading(error, expected):
    candidate = copy.deepcopy(TRUTH)
    candidate["resistors"]["r1"] += error
    assert score(candidate) == expected


def test_checks_every_measurement_instead_of_only_terminal_pair():
    candidate = copy.deepcopy(TRUTH)
    candidate["resistors"]["r3"] *= 1.01
    assert get_resistance_between_nodes(
        json.dumps(candidate), ["A", "B"]
    ) == pytest.approx(10.125)
    assert score(candidate) == 0


@pytest.mark.parametrize(
    "value", [0, -1, True, None, "10.125", float("nan"), float("inf"), 1e-320]
)
def test_invalid_or_numerically_unsolvable_values_are_rejected(value):
    candidate = copy.deepcopy(TRUTH)
    candidate["resistors"]["r1"] = value
    assert score(candidate) == 0


def test_extra_unused_resistor_is_rejected():
    candidate = copy.deepcopy(TRUTH)
    candidate["resistors"]["unused"] = 99.9
    assert score(candidate) == 0


def test_repeated_resistor_reference_is_rejected():
    candidate = copy.deepcopy(TRUTH)
    candidate["connections"][1][2] = "r1"
    assert score(candidate) == 0


def test_parallel_edges_are_rejected_even_when_measurements_match():
    candidate = copy.deepcopy(TRUTH)
    candidate["connections"][1][:2] = ["A", "B"]
    assert score(candidate) == 0


def test_extra_hidden_node_is_rejected():
    candidate = copy.deepcopy(TRUTH)
    candidate["connections"][1][1] = "secret"
    assert score(candidate) == 0


def test_disconnected_components_are_rejected_before_simulation(monkeypatch):
    nodes = ["A", "B", "N1", "N2", "N3", "N4"]
    truth = {
        "resistors": {f"r{i}": 10 for i in range(6)},
        "connections": [[nodes[i], nodes[(i + 1) % 6], f"r{i}"] for i in range(6)],
    }
    readings = [{"node_a": "A", "node_b": n, "resistance": 10} for n in nodes[1:]]
    candidate = {
        "resistors": truth["resistors"],
        "connections": [
            ["A", "B", "r0"],
            ["B", "N1", "r1"],
            ["N1", "A", "r2"],
            ["N2", "N3", "r3"],
            ["N3", "N4", "r4"],
            ["N4", "N2", "r5"],
        ],
    }

    def unexpected_simulation(*_args):
        pytest.fail("Disconnected network reached numerical simulation")

    monkeypatch.setattr(
        "resistor_network.score.get_resistance_between_nodes", unexpected_simulation
    )
    assert check_sparse_circuit_solution(truth, readings)(json.dumps(candidate)) == 0


def test_duplicate_json_keys_are_rejected():
    submission = '{"resistors":{"r1":99,"r1":10.125,"r2":20.25,"r3":30.375},"connections":[["A","B","r1"],["A","N1","r2"],["A","N2","r3"]]}'
    assert check_sparse_circuit_solution(TRUTH, MEASUREMENTS)(submission) == 0


@pytest.mark.parametrize(
    "submission", ["garbage", "null", "[]", "{}", '{"resistors":[]}']
)
def test_malformed_submission_returns_zero(submission):
    assert check_sparse_circuit_solution(TRUTH, MEASUREMENTS)(submission) == 0


def test_missing_public_data_is_not_accepted_as_vacuous_success():
    with pytest.raises(ValueError, match="published measurements"):
        check_sparse_circuit_solution(TRUTH, [])


def test_published_pool_preserves_indistinguishable_wiring_acceptance():
    alternative = {
        "resistors": {"x": 10.125, "y": 11.74321, "z": 50.625},
        "connections": [["B", "A", "x"], ["A", "N1", "y"], ["N1", "N2", "z"]],
    }
    scorer = check_sparse_circuit_solution(
        TRUTH,
        MEASUREMENTS,
        allowed_resistor_values=[10.125, 20.25, 30.375, 11.74321, 50.625],
    )
    assert scorer(json.dumps(TRUTH)) == scorer(json.dumps(alternative)) == 1


def test_off_pool_value_is_rejected_even_when_readings_match():
    candidate = copy.deepcopy(TRUTH)
    candidate["resistors"]["r1"] += 1e-8
    assert score(candidate) == 1
    scorer = check_sparse_circuit_solution(
        TRUTH,
        MEASUREMENTS,
        allowed_resistor_values=[10.125, 20.25, 30.375],
    )
    assert scorer(json.dumps(candidate)) == 0


def test_published_pool_allows_repeating_values_without_inventory_counts():
    truth = {
        "resistors": {"r1": 10, "r2": 10, "r3": 10},
        "connections": TRUTH["connections"],
    }
    readings = [
        {"node_a": "A", "node_b": "B", "resistance": 10},
        {"node_a": "N1", "node_b": "N2", "resistance": 20},
    ]
    assert (
        check_sparse_circuit_solution(
            truth,
            readings,
            allowed_resistor_values=[10, 20],
        )(json.dumps(truth))
        == 1
    )


@pytest.mark.parametrize(
    "pool", [[], [True], [0], [-1], [float("nan")], [float("inf")], ["10"]]
)
def test_invalid_published_pool_is_rejected(pool):
    with pytest.raises(ValueError, match="positive and finite"):
        check_sparse_circuit_solution(TRUTH, MEASUREMENTS, allowed_resistor_values=pool)


def test_truth_outside_published_pool_is_a_configuration_error():
    with pytest.raises(ValueError, match="published pool"):
        check_sparse_circuit_solution(TRUTH, MEASUREMENTS, allowed_resistor_values=[10])
