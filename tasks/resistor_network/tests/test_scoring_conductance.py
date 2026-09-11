"""Tests for conductance-map scoring."""

import json

import pytest
from resistor_network.score import check_conductance_topology, conductance_map

# A -- N1 -- B, with two resistors in parallel on A-N1
TRUTH = {
    "resistors": {"R1": 47.0, "R2": 33.0, "R3": 20.0},
    "connections": [["A", "N1", "R1"], ["A", "N1", "R2"], ["N1", "B", "R3"]],
}


def score(topology, expected=TRUTH, tolerance=0.1):
    return check_conductance_topology(expected, tolerance)(json.dumps(topology))


def test_conductance_map_merges_parallel_resistors():
    assert conductance_map(TRUTH) == pytest.approx(
        {("A", "N1"): 1 / 47 + 1 / 33, ("B", "N1"): 1 / 20}
    )


def test_conductance_map_is_order_independent():
    flipped = {
        "resistors": TRUTH["resistors"],
        "connections": [[b, a, r] for a, b, r in reversed(TRUTH["connections"])],
    }
    assert conductance_map(flipped) == pytest.approx(conductance_map(TRUTH))


def test_exact_ground_truth_scores_one():
    assert score(TRUTH) == 1.0


def test_parallel_group_may_be_submitted_as_one_resistor():
    merged = {
        "resistors": {"Ra": 1 / (1 / 47 + 1 / 33), "Rb": 20.0},
        "connections": [["A", "N1", "Ra"], ["N1", "B", "Rb"]],
    }
    assert score(merged) == 1.0


def test_resistor_ids_and_connection_order_are_free():
    renamed = {
        "resistors": {"foo": 33.0, "bar": 20.0, "baz": 47.0},
        "connections": [["B", "N1", "bar"], ["N1", "A", "baz"], ["A", "N1", "foo"]],
    }
    assert score(renamed) == 1.0


def test_values_within_tolerance_pass_and_outside_fail():
    near = {
        "resistors": {"Ra": 1 / (1.05 * (1 / 47 + 1 / 33)), "Rb": 20.0},
        "connections": [["A", "N1", "Ra"], ["N1", "B", "Rb"]],
    }
    assert score(near) == 1.0  # 5% off, inside the 10% band
    far = {
        "resistors": {"Ra": 1 / (1.5 * (1 / 47 + 1 / 33)), "Rb": 20.0},
        "connections": [["A", "N1", "Ra"], ["N1", "B", "Rb"]],
    }
    assert score(far) == 0.0


def test_invented_connection_is_rejected():
    """The complete-graph exploit: a near-open resistor is still a claimed component."""
    extra = {
        "resistors": {"R1": 47.0, "R2": 33.0, "R3": 20.0, "R4": 1e9},
        "connections": [
            ["A", "N1", "R1"],
            ["A", "N1", "R2"],
            ["N1", "B", "R3"],
            ["A", "B", "R4"],
        ],
    }
    assert score(extra) == 0.0


def test_dropping_the_only_resistor_on_a_pair_is_rejected():
    missing = {
        "resistors": {"R1": 47.0, "R2": 33.0},
        "connections": [["A", "N1", "R1"], ["A", "N1", "R2"]],
    }
    assert score(missing) == 0.0


def test_halving_a_parallel_group_conductance_is_rejected():
    half = {
        "resistors": {"Ra": 47.0, "Rb": 20.0},
        "connections": [["A", "N1", "Ra"], ["N1", "B", "Rb"]],
    }
    assert score(half) == 0.0


def test_invented_internal_node_is_rejected():
    wye = {
        "resistors": {"R1": 47.0, "R2": 33.0, "R3": 10.0, "R4": 10.0},
        "connections": [
            ["A", "N1", "R1"],
            ["A", "N1", "R2"],
            ["N1", "N9", "R3"],
            ["N9", "B", "R4"],
        ],
    }
    assert score(wye) == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[1, 2, 3]",
        '{"resistors": {}, "connections": []}',
        '{"resistors": {"R1": 10}}',
        '{"resistors": {"R1": 10}, "connections": [["A", "B"]]}',
        '{"resistors": {"R1": 0}, "connections": [["A", "B", "R1"]]}',
        '{"resistors": {"R1": -5}, "connections": [["A", "B", "R1"]]}',
        '{"resistors": {"R1": 10}, "connections": [["A", "A", "R1"]]}',
        '{"resistors": {"R1": 10}, "connections": [["A", "B", "R9"]]}',
        '{"resistors": {"R1": 10}, "connections": [["A", "B", "R1"], ["B", "C", "R1"]]}',
        '{"resistors": {"R1": 10, "R2": 5}, "connections": [["A", "B", "R1"]]}',
    ],
)
def test_malformed_submissions_score_zero(bad):
    assert check_conductance_topology(TRUTH)(bad) == 0.0


def test_every_shipped_task_accepts_its_own_ground_truth():
    import glob
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = sorted(glob.glob(str(root / "environments/level_*/tasks_json/task_*.json")))
    assert paths, "no task files found"
    for path in paths:
        for task in json.loads(Path(path).read_text()):
            assert task["scoring_function"] == "resistor_conductance"
            params = task["scoring_params"]
            scorer = check_conductance_topology(**params)
            assert scorer(json.dumps(params["expected_topology"])) == 1.0
