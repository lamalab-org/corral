"""Check offline score semantics, missing labels and task-group validation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def analysis():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/analyze_decisions_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location("threshold_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scores_use_probability_not_confidence_and_distinguish_rescoring(analysis):
    assessments = [
        {
            "status": "active",
            "confidence": 0.99,
            "refused": False,
            "probabilities": {"active": 0.31, "resolved": 0.15},
        },
        {
            "status": "resolved",
            "confidence": 0.01,
            "refused": False,
            "probabilities": {"active": 0.1, "resolved": 0.4},
        },
        {
            "status": "emerging",
            "confidence": 0.9,
            "refused": False,
            "probabilities": {"active": 0.3, "resolved": 0.25},
        },
        {"status": "uncertain", "refused": True, "probabilities": {}},
    ]
    assert analysis.window_scores(assessments) == (0.4, 0.55)
    assert analysis.window_scores(assessments[2:]) == (-1, 0.55)
    assert analysis.window_scores(assessments[3:]) == (-1, -1)


def test_threshold_boundary_inclusive_and_zero_does_not_create_selected_choices(
    analysis,
):
    scores = np.array([[0.5, -1], [0.49, 0.1]])
    references = np.array([[[True, True], [False, False]]] * 2)
    predictions, _, _ = analysis.sweep(scores, references, np.ones((2, 2), dtype=bool))
    assert predictions[50].tolist() == [[True, False], [False, False]]
    assert predictions[0].tolist() == [[True, False], [True, True]]


def test_unknown_pairs_never_change_confusion_counts(analysis):
    prediction = np.array([[True], [False], [True], [False], [True]])
    reference = np.array([[True], [True], [False], [False], [False]])
    mask = np.array([[True], [True], [True], [True], [False]])
    counts = analysis.confusion(prediction, reference, mask)
    assert {key: values.tolist() for key, values in counts.items()} == {
        "tp": [1],
        "fp": [1],
        "fn": [1],
        "tn": [1],
    }
    assert analysis.f1_from_counts(counts).tolist() == [0.5]


def test_same_task_never_crosses_validation_folds(analysis):
    records = [
        {"environment": "a", "task_id": str(task), "agent": agent}
        for task in range(12)
        for agent in ("x", "y", "z")
    ]
    folds = analysis.task_folds(records)
    assert len(set(folds)) == 5
    for task in range(12):
        assert len(set(folds[3 * task : 3 * task + 3])) == 1
    assert np.array_equal(folds, analysis.task_folds(records))


def test_equal_weight_objective_ties_and_unsupported_pattern_fallback(analysis):
    objectives = np.zeros((101, 3))
    objectives[10:21] = [0.5, 0.5, 0]
    objectives[30, 0] = 0.9
    references = np.array([[[True, False, False]], [[False, True, False]]])
    threshold, per_pattern, supported = analysis.pick_thresholds(
        objectives, references, np.ones((1, 3), dtype=bool)
    )
    assert threshold == 0.1  # Mean .333 beats .3; lowest threshold wins the tie.
    assert per_pattern.tolist() == [0.3, 0.1, 0.1]
    assert supported.tolist() == [True, True, False]


def test_no_positive_predictions_or_references_has_zero_f1(analysis):
    empty = np.zeros((3, 2), dtype=bool)
    counts = analysis.confusion(empty, empty, np.ones_like(empty))
    assert analysis.f1_from_counts(counts).tolist() == [0, 0]
