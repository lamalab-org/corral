from __future__ import annotations

import json
from pathlib import Path

import pytest
from stargazer.evaluator import evaluate_submission
from stargazer.models import semi_amplitude_ms
from stargazer.tools import (
    create_analysis_session,
    evaluate_candidate,
    planet_from_fit,
    python_repl,
)


@pytest.fixture
def analysis_session(simple_task):
    observations = simple_task.observations
    session = create_analysis_session(
        times_days=observations.times_days,
        rvs_ms=observations.rvs_ms,
        sigmas_ms=observations.sigmas_ms,
        instruments=observations.instruments,
        star_mass_sun=simple_task.star_mass_sun,
    )
    yield session
    session.close()


def test_python_repl_persists_state(analysis_session):
    first = python_repl.execute(
        code="candidate = 3.5", analysis_session=analysis_session
    )
    second = python_repl.execute(
        code="candidate + 2", analysis_session=analysis_session
    )

    assert "successfully" in first
    assert float(second.strip()) == 5.5


def test_python_namespaces_are_isolated_between_trials(simple_task):
    observations = simple_task.observations
    sessions = [
        create_analysis_session(
            times_days=observations.times_days,
            rvs_ms=observations.rvs_ms,
            sigmas_ms=observations.sigmas_ms,
            instruments=observations.instruments,
            star_mass_sun=simple_task.star_mass_sun,
        )
        for _ in range(2)
    ]
    try:
        python_repl.execute(code="private_fit = 42", analysis_session=sessions[0])
        isolated = python_repl.execute(code="private_fit", analysis_session=sessions[1])
    finally:
        for session in sessions:
            session.close()

    assert "NameError" in isolated


def test_python_repl_allows_single_underscore_variables(analysis_session):
    result = python_repl.execute(
        code="_candidate_period = 17.25\n_candidate_period",
        analysis_session=analysis_session,
    )

    assert float(result.strip()) == 17.25


def test_python_repl_supports_scipy_and_persistent_user_functions(analysis_session):
    defined = python_repl.execute(
        code=(
            "from scipy.signal import lombscargle\n"
            "def _center(values):\n"
            "    return values - np.mean(values)"
        ),
        analysis_session=analysis_session,
    )
    result = python_repl.execute(
        code="float(np.std(_center(rvs_ms)))", analysis_session=analysis_session
    )

    assert "successfully" in defined
    assert float(result.strip()) > 0.0


def test_python_repl_blocks_filesystem_imports(analysis_session):
    result = python_repl.execute(
        code="import pathlib", analysis_session=analysis_session
    )

    assert "ImportError" in result
    assert "disabled" in result


def test_python_repl_blocks_task_bank_reads(analysis_session):
    data_root = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    bank_file = sorted(data_root.glob("*.json"))[0]
    result = python_repl.execute(
        code=f"import json\njson.codecs.open({str(bank_file)!r}).read()",
        analysis_session=analysis_session,
    )

    assert "PermissionError" in result
    assert "Filesystem access is unavailable" in result
    assert "truth_planets" not in result


@pytest.mark.parametrize(
    "payload",
    [
        "stargazer_planet_from_fit.__closure__[0].cell_contents",
        "simulate_keplerian_rv.__globals__['Path']('tasks/stargazer/data')",
        "np.mean.__globals__",
    ],
)
def test_python_repl_blocks_reviewer_introspection_payloads(analysis_session, payload):
    result = python_repl.execute(code=payload, analysis_session=analysis_session)

    assert "UnsafeAnalysisCode" in result
    assert "Dunder" in result


def test_python_repl_module_proxy_cannot_be_unwrapped(analysis_session):
    result = python_repl.execute(
        code="np._SafeModuleProxy__wrapped_module",
        analysis_session=analysis_session,
    )

    assert "AttributeError" in result
    assert "unavailable" in result


def test_python_repl_does_not_expose_task_or_submission_history(analysis_session):
    for name in ("benchmark_task", "history", "StargazerTask"):
        result = python_repl.execute(code=name, analysis_session=analysis_session)
        assert "NameError" in result


def test_python_repl_timeout_resets_persistent_worker(simple_task):
    observations = simple_task.observations
    session = create_analysis_session(
        times_days=observations.times_days,
        rvs_ms=observations.rvs_ms,
        sigmas_ms=observations.sigmas_ms,
        instruments=observations.instruments,
        star_mass_sun=simple_task.star_mass_sun,
        execution_timeout_seconds=0.2,
    )
    try:
        python_repl.execute(code="retained = 42", analysis_session=session)
        timeout = python_repl.execute(
            code="while True:\n    pass", analysis_session=session
        )
        after_reset = python_repl.execute(code="retained", analysis_session=session)
    finally:
        session.close()

    assert "AnalysisTimeoutError" in timeout
    assert "session was reset" in timeout
    assert "NameError" in after_reset


def test_planet_from_fit_converts_semi_amplitude(simple_task):
    truth = simple_task.truth_planets[0]
    amplitude = semi_amplitude_ms(
        truth.m_sin_i_mjup,
        truth.P_days,
        truth.e,
        simple_task.star_mass_sun,
    )

    converted = json.loads(
        planet_from_fit.execute(
            period_days=truth.P_days,
            semi_amplitude_ms=amplitude,
            eccentricity=truth.e,
            omega_rad=truth.omega_rad,
            mean_anomaly_rad=truth.l_rad - truth.omega_rad,
            star_mass_sun=simple_task.star_mass_sun,
        )
    )

    assert converted["m_sin_i_mjup"] == pytest.approx(truth.m_sin_i_mjup)
    assert converted["l_rad"] == pytest.approx(truth.l_rad)


def test_hidden_tool_arguments_are_not_exposed():
    repl_schema = python_repl.to_mcp()["inputSchema"]
    conversion_schema = planet_from_fit.to_mcp()["inputSchema"]
    evaluate_schema = evaluate_candidate.to_mcp()["inputSchema"]

    assert "analysis_session" not in repl_schema["properties"]
    assert "star_mass_sun" not in conversion_schema["properties"]
    assert "benchmark_task" not in evaluate_schema["properties"]
    assert "evaluation_session" not in evaluate_schema["properties"]
    assert "notes" not in evaluate_schema["properties"]


def test_evaluate_candidate_returns_redacted_criterion_feedback(
    simple_task, exact_submission
):
    session = {"evaluations": [], "max_evaluations": 1, "locked": False}
    planet = {
        key: exact_submission["planets"][0][key]
        for key in ("P_days", "m_sin_i_mjup", "e", "omega_rad", "l_rad")
    }

    raw = evaluate_candidate.execute(
        planets=[planet],
        noise_jitter_ms=0.0,
        benchmark_task=simple_task,
        evaluation_session=session,
    )
    feedback = json.loads(raw)

    assert feedback["accepted"]
    assert feedback["success"]
    assert feedback["criteria"]["planet_count"]["passed"]
    assert "truth" not in raw.lower()
    assert "matched_pairs" not in raw
    assert "count_difference" not in raw
    assert len(session["evaluations"]) == 1

    locked = json.loads(
        evaluate_candidate.execute(
            planets=[planet],
            benchmark_task=simple_task,
            evaluation_session=session,
        )
    )
    assert not locked["accepted"]
    assert "locked" in locked["error"]
    assert locked["remaining_evaluations"] == 0


def test_invalid_candidate_does_not_consume_evaluation(simple_task):
    session = {"evaluations": [], "max_evaluations": 2, "locked": False}

    feedback = json.loads(
        evaluate_candidate.execute(
            planets=[
                {
                    "P_days": -1,
                    "m_sin_i_mjup": 0.2,
                    "e": 0.0,
                    "omega_rad": 0.0,
                    "l_rad": 0.0,
                }
            ],
            benchmark_task=simple_task,
            evaluation_session=session,
        )
    )

    assert not feedback["accepted"]
    assert feedback["remaining_evaluations"] == 2
    assert session["evaluations"] == []


def test_evaluate_candidate_arguments_are_final_answer_json(
    simple_task, exact_submission
):
    session = {"evaluations": [], "max_evaluations": 2, "locked": False}
    planets = [
        {
            key: planet[key]
            for key in ("P_days", "m_sin_i_mjup", "e", "omega_rad", "l_rad")
        }
        for planet in exact_submission["planets"]
    ]
    feedback = json.loads(
        evaluate_candidate.execute(
            planets=planets,
            noise_jitter_ms=exact_submission["noise_jitter_ms"],
            benchmark_task=simple_task,
            evaluation_session=session,
        )
    )
    final_answer = {
        "planets": planets,
        "noise_jitter_ms": exact_submission["noise_jitter_ms"],
    }

    result = evaluate_submission(simple_task, final_answer)

    assert feedback["success"] == result.success
    assert feedback["criteria"] == result.agent_feedback()["criteria"]
    assert session["evaluations"][0]["candidate"] == final_answer


def test_valid_failures_consume_exact_evaluation_budget(simple_task):
    session = {"evaluations": [], "max_evaluations": 2, "locked": False}

    for expected_remaining in (1, 0):
        feedback = json.loads(
            evaluate_candidate.execute(
                planets=[],
                benchmark_task=simple_task,
                evaluation_session=session,
            )
        )
        assert feedback["accepted"]
        assert not feedback["success"]
        assert feedback["remaining_evaluations"] == expected_remaining

    exhausted = json.loads(
        evaluate_candidate.execute(
            planets=[],
            benchmark_task=simple_task,
            evaluation_session=session,
        )
    )
    assert not exhausted["accepted"]
    assert exhausted["remaining_evaluations"] == 0
