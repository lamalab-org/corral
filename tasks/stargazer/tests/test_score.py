from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.models import (
    CandidateSubmission,
    Observations,
    PlanetParams,
    load_task,
    simulate_keplerian_rv,
)
from stargazer.score import (
    evaluate_submission,
    make_stargazer_scorer,
    normalize_submission,
    submit_candidate,
)


def test_exact_truth_passes_all_four_gates(simple_task, exact_submission):
    result = evaluate_submission(simple_task, exact_submission)

    assert result.success
    assert result.score == 1.0
    assert result.ok_delta_bic
    assert result.ok_rms
    assert result.ok_match
    assert result.ok_count
    assert result.match_score == 1.0


def test_missing_planet_fails_count_gate(simple_task):
    result = evaluate_submission(simple_task, {"planets": []})

    assert not result.success
    assert not result.ok_count
    assert result.count_difference == -1


def test_matching_is_independent_of_submission_order(simple_task):
    second = PlanetParams(
        P_days=43.0,
        m_sin_i_mjup=0.2,
        e=0.05,
        omega_rad=2.0,
        l_rad=5.0,
    )
    truth = (*simple_task.truth_planets, second)
    times = np.asarray(simple_task.observations.times_days)
    velocities = simulate_keplerian_rv(truth, times, simple_task.star_mass_sun)
    task = replace(
        simple_task,
        truth_planets=truth,
        observations=Observations(
            times_days=simple_task.observations.times_days,
            rvs_ms=tuple(velocities.tolist()),
            sigmas_ms=simple_task.observations.sigmas_ms,
            instruments=simple_task.observations.instruments,
        ),
    )
    planets = []
    for planet in reversed(truth):
        payload = asdict(planet)
        payload.pop("m_true_mjup")
        planets.append(payload)

    result = evaluate_submission(task, {"planets": planets})

    assert result.success
    assert result.match_score == 1.0
    assert {(truth_i, guess_i) for truth_i, guess_i, _ in result.matched_pairs} == {
        (0, 1),
        (1, 0),
    }


def test_matching_discards_assignment_beyond_released_distance_cutoff(simple_task):
    wrong = {
        "P_days": 1_000.0,
        "m_sin_i_mjup": 5.0,
        "e": 0.8,
        "omega_rad": 0.0,
        "l_rad": 0.0,
    }

    result = evaluate_submission(simple_task, {"planets": [wrong]})

    assert result.ok_count
    assert result.matched_pairs == ()
    assert not result.ok_match


def test_semi_amplitude_alias_converts_to_native_mass(simple_task, exact_submission):
    native = normalize_submission(exact_submission, simple_task)
    planet = exact_submission["planets"][0]
    amplitude = (
        28.4329
        * planet["m_sin_i_mjup"]
        * (simple_task.star_mass_sun ** (-2 / 3))
        * ((planet["P_days"] / 365.25) ** (-1 / 3))
        / np.sqrt(1 - planet["e"] ** 2)
    )
    alias = {
        "planets": [
            {
                "period_days": planet["P_days"],
                "semi_amplitude_ms": amplitude,
                "eccentricity": planet["e"],
                "omega_rad": planet["omega_rad"],
                "l_rad": planet["l_rad"],
            }
        ]
    }

    converted = normalize_submission(alias, simple_task)

    assert converted.planets[0].m_sin_i_mjup == pytest.approx(
        native.planets[0].m_sin_i_mjup
    )


def test_canonical_json_and_shared_model_normalize_identically(
    simple_task, exact_submission
):
    model = CandidateSubmission.model_validate(exact_submission)

    from_json = normalize_submission(json.dumps(exact_submission), simple_task)
    from_model = normalize_submission(model, simple_task)

    assert from_json == from_model
    assert from_json.noise_jitter_ms == 0.1
    assert from_json.planets == model.planets


def test_omitted_jitter_uses_upstream_action_residuals(simple_task, exact_submission):
    payload = {"planets": exact_submission["planets"]}
    result = normalize_submission(payload, simple_task)
    # Captured from the original action builder with the test_single dataset.
    assert result.noise_jitter_ms == pytest.approx(6.461971104042583)
    assert CandidateSubmission.model_validate(payload).noise_jitter_ms is None


def test_invalid_submission_scores_zero(simple_task):
    scorer = make_stargazer_scorer(simple_task)

    assert scorer("not json") == 0.0
    assert scorer('{"planets": [{"P_days": 1, "e": 1.2}]}') == 0.0


def test_eccentricity_and_mass_clamps_match_upstream(simple_task):
    normalized = normalize_submission(
        {"planets": [{"P_days": 10, "m_sin_i_mjup": 50, "e": 0.9}]}, simple_task
    )
    assert normalized.planets[0].e == 0.8
    assert normalized.planets[0].m_sin_i_mjup == 30.0


def test_seven_planet_runner_limit_truncates_extra_candidates(simple_task):
    planet = {"P_days": 10, "m_sin_i_mjup": 0.1, "e": 0}
    normalized = normalize_submission({"planets": [planet] * 8}, simple_task)
    assert len(normalized.planets) == simple_task.max_planets == 7


def test_task_hint_overrides_are_used(simple_task, exact_submission):
    task = replace(
        simple_task,
        metadata={
            **simple_task.metadata,
            "hints": {"target_match_score": 1.1, "max_rms_ms": 0.01},
        },
    )
    result = evaluate_submission(task, exact_submission)
    assert result.maximum_rms_ms == 0.01
    assert result.ok_rms
    assert not result.ok_match


@pytest.mark.parametrize("kind", ["reference", "omitted_jitter"])
def test_all_twenty_tasks_match_original_feedback(
    kind, protocol_reference, assert_protocol_equal
):
    for expected in protocol_reference["records"]:
        if expected["kind"] != kind:
            continue
        path = DEFAULT_DATA_ROOT / "synthetic" / f"{expected['task_id']}.json"
        raw = json.loads(path.read_text())
        task = load_task(path)
        payload = reference_submission(task, raw).canonical_payload()
        if kind == "omitted_jitter":
            payload.pop("noise_jitter_ms", None)
        session = {"steps": 0, "history": []}
        feedback = json.loads(submit_candidate(task, payload, session))
        assert_protocol_equal(feedback, expected["feedback"])
        assert_protocol_equal(session["history"][-1]["metrics"], expected["metrics"])


def test_zero_time_of_periastron_is_not_discarded(simple_task, exact_submission):
    planet = exact_submission["planets"][0]
    base = {key: value for key, value in planet.items() if key != "l_rad"}

    # Periastron times one period apart describe the same orbital phase, so a
    # T0_days of exactly 0.0 must not fall through to the default phase.
    zero = normalize_submission({"planets": [{**base, "T0_days": 0.0}]}, simple_task)
    shifted = normalize_submission(
        {"planets": [{**base, "T0_days": planet["P_days"]}]}, simple_task
    )

    assert zero.planets[0].l_rad == pytest.approx(shifted.planets[0].l_rad)


def test_null_mass_does_not_discard_the_amplitude(simple_task):
    planet = {"P_days": 10.0, "K_ms": 50.0, "e": 0.1, "omega_rad": 0.2, "l_rad": 1.0}
    expected = normalize_submission({"planets": [planet]}, simple_task)

    # An LLM writing `"m_sin_i_mjup": null` used to drop K_ms with it, leaving
    # the planet clamped to the 0.001 Mjup floor.
    for absent in (None, 0.0):
        result = normalize_submission(
            {"planets": [{**planet, "m_sin_i_mjup": absent}]}, simple_task
        )
        assert result.planets[0].m_sin_i_mjup == pytest.approx(
            expected.planets[0].m_sin_i_mjup
        )
        assert result.planets[0].m_sin_i_mjup > 0.001
