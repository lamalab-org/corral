from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from stargazer.evaluator import (
    SubmissionError,
    evaluate_submission,
    make_stargazer_scorer,
    normalize_submission,
)
from stargazer.models import (
    CandidateSubmission,
    Observations,
    PlanetParams,
    simulate_keplerian_rv,
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
                "mean_longitude_rad": planet["l_rad"],
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

    assert from_json == from_model == model


def test_legacy_nested_noise_is_accepted_as_compatibility_input(
    simple_task, exact_submission
):
    legacy = {
        "planets": exact_submission["planets"],
        "noise": {"sigma_jitter_ms": 0.25},
    }

    normalized = normalize_submission(legacy, simple_task)

    assert normalized.noise_jitter_ms == 0.25
    assert "noise" not in normalized.canonical_payload()
    assert normalized.canonical_payload()["noise_jitter_ms"] == 0.25


def test_invalid_submission_scores_zero(simple_task):
    scorer = make_stargazer_scorer(simple_task)

    assert scorer("not json") == 0.0
    assert scorer('{"planets": [{"P_days": 1, "e": 1.2}]}') == 0.0


def test_out_of_range_eccentricity_is_rejected(simple_task):
    with pytest.raises(SubmissionError, match="between 0 and 0.8"):
        normalize_submission(
            {
                "planets": [
                    {
                        "P_days": 10,
                        "m_sin_i_mjup": 1,
                        "e": 0.9,
                    }
                ]
            },
            simple_task,
        )
