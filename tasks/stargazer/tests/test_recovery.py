"""Cheap decision tests; fifty-realization calibration lives in offline artifacts."""

from dataclasses import replace

import pytest
from stargazer.identifiability import BankRules, audit_noise_recovery
from stargazer.models import PublicFitContext


def solution(candidate, *, converged=True, bic=1):
    return {
        "candidate": candidate,
        "count": len(candidate["planets"]),
        "bic": bic,
        "optimizer": {"converged": converged},
        "start_optimizers": [{"converged": converged}],
    }


def test_recovery_uses_public_context_fresh_noise_and_exact_criteria(
    simple_task,
    exact_submission,
    monkeypatch,
):
    contexts = []

    def fit(context, **_kwargs):
        assert isinstance(context, PublicFitContext)
        assert not hasattr(context, "truth_planets")
        contexts.append(context)
        return [solution(exact_submission)]

    monkeypatch.setattr("stargazer.identifiability.fit_counts", fit)
    rules = BankRules(noise_realizations=3, minimum_full_passes=3)
    result = audit_noise_recovery(simple_task, rules, seed=42)
    assert result["accepted"]
    assert result["full_passes"] == 3
    assert result["criteria"]["require_complete_matching"]
    assert result["criteria"]["minimum_planet_score"] is None
    assert len({context.observations.rvs_ms for context in contexts}) == 3
    assert all(
        context.observations.times_days == simple_task.observations.times_days
        for context in contexts
    )
    assert result == audit_noise_recovery(simple_task, rules, seed=42)


@pytest.mark.parametrize(("converged", "wrong_count"), [(False, False), (True, True)])
def test_unresolved_search_or_wrong_count_cannot_pass(
    simple_task,
    exact_submission,
    monkeypatch,
    converged,
    wrong_count,
):
    candidate = {"planets": []} if wrong_count else exact_submission
    monkeypatch.setattr(
        "stargazer.identifiability.fit_counts",
        lambda *_a, **_kw: [
            solution(candidate, converged=converged),
        ],
    )
    result = audit_noise_recovery(
        simple_task, BankRules(noise_realizations=2, minimum_full_passes=2), seed=42
    )
    assert not result["accepted"]
    assert result["full_passes"] == 0
    assert result["unresolved_searches"] == (0 if converged else 2)


def test_count_selection_ignores_private_grades(
    simple_task, exact_submission, monkeypatch
):
    monkeypatch.setattr(
        "stargazer.identifiability.fit_counts",
        lambda *_a, **_kw: [
            solution({"planets": []}, bic=0),
            solution(exact_submission, bic=1),
        ],
    )
    result = audit_noise_recovery(
        simple_task, BankRules(noise_realizations=1, minimum_full_passes=1), seed=42
    )
    assert result["results"][0]["fits"][1]["full_pass"]
    assert not result["accepted"]


def test_recovery_refuses_unsupported_noise(simple_task):
    task = replace(simple_task, config={"noise": {"gp_amplitude": 1.0}})
    with pytest.raises(ValueError, match="white noise"):
        audit_noise_recovery(task, BankRules(), seed=42)


@pytest.mark.parametrize("passes", [44, 45])
def test_default_acceptance_boundary(
    simple_task, exact_submission, monkeypatch, passes
):
    candidates = iter([exact_submission] * passes + [{"planets": []}] * (50 - passes))
    monkeypatch.setattr(
        "stargazer.identifiability.fit_counts",
        lambda *_a, **_kw: [solution(next(candidates))],
    )
    result = audit_noise_recovery(simple_task, BankRules(), seed=42)
    assert result["full_passes"] == passes
    assert result["accepted"] == (passes >= 45)


def test_exhausted_competitor_is_recorded_without_rejecting_converged_winner(
    simple_task,
    exact_submission,
    monkeypatch,
):
    monkeypatch.setattr(
        "stargazer.identifiability.fit_counts",
        lambda *_a, **_kw: [
            solution(exact_submission, bic=0),
            solution({"planets": []}, converged=False, bic=1),
        ],
    )
    result = audit_noise_recovery(
        simple_task, BankRules(noise_realizations=1, minimum_full_passes=1), seed=42
    )
    assert result["accepted"]
    assert result["unresolved_count_fits"] == 1
    assert result["results"][0]["fits"][1]["status"] == "unresolved_search"
