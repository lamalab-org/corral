"""Information-flow, numerical-parity, and recorded-exploit regressions."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.fit import compute_fit, normalize_candidate, validate_fit
from stargazer.models import load_task
from stargazer.score import (
    LEGACY_CRITERIA,
    EvaluationCriteria,
    evaluate_submission,
    make_stargazer_scorer,
)


def test_reference_substitution_cannot_change_public_results(
    simple_task, exact_submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("private grader called during analysis")

    monkeypatch.setattr("stargazer.score._match_planets", forbidden)
    monkeypatch.setattr("stargazer.score.evaluate_submission", forbidden)
    alternate = replace(
        simple_task,
        truth_planets=(),
        task_id="SECRET",
        truth_difficulty=999,
        metadata={"reference": "SECRET", "hints": {"max_rms_ms": 999}},
    )
    for payload in (
        exact_submission,
        {"planets": []},
        {},
        {"planets": [None]},
        {"planets": [{"P_days": -1}]},
    ):
        assert validate_fit(simple_task.public_fit_context(), payload) == validate_fit(
            alternate.public_fit_context(), payload
        )
    assert "SECRET" not in json.dumps(
        simple_task.public_fit_context().public_observation()
    )
    assert set(asdict(simple_task.public_fit_context())) == {
        "observations",
        "star_mass_sun",
        "max_planets",
        "maximum_rms_factor",
        "los_axis",
        "integrator_preference",
    }


def test_run2_candidates_close_all_six_incomplete_passes():
    records = json.loads(
        (Path(__file__).parent / "fixtures/run2_candidates.json").read_text()
    )
    old_passes = new_passes = removed = 0
    assert len(records) == 72
    for row in records:
        task = load_task(DEFAULT_DATA_ROOT / "synthetic" / f"{row['task_id']}.json")
        legacy = evaluate_submission(task, row["candidate"], LEGACY_CRITERIA)
        strict = evaluate_submission(task, row["candidate"])
        public = validate_fit(task.public_fit_context(), row["candidate"])
        assert legacy.success == row["success"]
        assert legacy.match_score == pytest.approx(row["match_score"])
        assert legacy.rms_ms == pytest.approx(row["rms"])
        assert public["residuals"]["rms"] == pytest.approx(legacy.rms_ms)
        assert strict.match_score == legacy.match_score
        assert strict.success == (
            legacy.success
            and not row["assignment"]["unmatched_truth"]
            and not row["assignment"]["unmatched_guess"]
        )
        old_passes += legacy.success
        new_passes += strict.success
        removed += legacy.success and not strict.success
    assert (old_passes, new_passes, removed) == (18, 12, 6)


def test_public_fit_parity_all_references_and_omitted_jitter():
    for path in sorted((DEFAULT_DATA_ROOT / "synthetic").glob("*.json")):
        task = load_task(path)
        payload = reference_submission(
            task, json.loads(path.read_text())
        ).canonical_payload()
        assert evaluate_submission(task, payload).success
        for omitted in (False, True):
            if omitted:
                payload.pop("noise_jitter_ms")
            fit = compute_fit(
                task.public_fit_context(),
                normalize_candidate(payload, task.public_fit_context()),
            )
            private = evaluate_submission(task, payload)
            assert fit.rms_ms == private.rms_ms
            assert fit.bic == private.metrics["bic"]
            assert fit.delta_bic == private.delta_bic
            assert fit.gamma_per_instrument_ms == private.gamma_per_instrument_ms


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        None,
        {"planets": None},
        {"planets": [None]},
        {"planets": [{"P_days": True}]},
        {"planets": [{"P_days": float("nan")}]},
        {"planets": [{"P_days": float("inf")}]},
        {"planets": [{"P_days": "10"}]},
        {"planets": [], "noise_jitter_ms": {}},
        {"planets": [], "noise_jitter_ms": 10**500},
    ],
)
def test_bad_candidates_are_controlled_failures(simple_task, payload):
    result = validate_fit(simple_task.public_fit_context(), payload)
    assert result["valid"] is False
    assert result["errors"]
    assert make_stargazer_scorer(simple_task)(json.dumps(payload)) == 0


def test_infrastructure_failures_are_not_candidate_failures(
    simple_task, exact_submission, monkeypatch
):
    def fail(*_a, **_kw):
        raise RuntimeError("broken library")

    monkeypatch.setattr("stargazer.public_rv.simulate_keplerian_rv", fail)
    with pytest.raises(RuntimeError, match="broken library"):
        validate_fit(simple_task.public_fit_context(), exact_submission)
    with pytest.raises(RuntimeError, match="broken library"):
        make_stargazer_scorer(simple_task)(json.dumps(exact_submission))


def test_coverage_empty_missing_extra_duplicate_and_permuted(
    simple_task, exact_submission
):
    p = exact_submission["planets"][0]
    assert (
        evaluate_submission(simple_task, {"planets": []}).ok_complete_matching is False
    )
    for planets in ([p, p], [p, {**p, "P_days": 1000}]):
        assert not evaluate_submission(
            simple_task, {"planets": planets, "noise_jitter_ms": 0.1}
        ).ok_complete_matching
    empty = replace(simple_task, truth_planets=())
    assert evaluate_submission(empty, {"planets": []}).ok_complete_matching
    assert not evaluate_submission(empty, exact_submission).ok_complete_matching
    assert evaluate_submission(simple_task, exact_submission).ok_complete_matching


def test_optional_per_planet_gate_prevents_mean_masking(
    simple_task, exact_submission, monkeypatch
):
    # Demonstrate the declared difference between coverage v2 and per-planet v3.
    task = replace(simple_task, truth_planets=simple_task.truth_planets * 5)
    payload = {**exact_submission, "planets": exact_submission["planets"] * 5}
    distances = (0.0, 0.0, 0.0, 0.0, float(np.log(100)))
    monkeypatch.setattr(
        "stargazer.score._match_planets",
        lambda *_a: (0.802, tuple((i, i, d) for i, d in enumerate(distances))),
    )
    result = evaluate_submission(task, payload)
    assert result.ok_complete_matching
    assert result.ok_match
    assert result.ok_individual_matches
    strict = evaluate_submission(
        task, payload, EvaluationCriteria(minimum_planet_score=0.8)
    )
    assert not strict.ok_individual_matches
    assert not strict.success


def test_extreme_finite_jitter_is_a_controlled_candidate_error(simple_task):
    payload = {"planets": [], "noise_jitter_ms": 1e308}
    assert not validate_fit(simple_task.public_fit_context(), payload)["valid"]
    assert make_stargazer_scorer(simple_task)(json.dumps(payload)) == 0


def test_json_parser_limits_are_candidate_errors(simple_task):
    scorer = make_stargazer_scorer(simple_task)
    assert scorer('{"planets": [], "noise_jitter_ms": ' + "9" * 5000 + "}") == 0
    assert scorer("[" * 2000 + "]" * 2000) == 0
