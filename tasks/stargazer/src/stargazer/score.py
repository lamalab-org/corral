"""Private final-candidate scoring and offline legacy compatibility fixtures."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from stargazer.fit import (
    SemanticSubmissionError,
    SubmissionError,
    canonicalize_plan,
    compute_fit,
    normalize_candidate,
    parse_submission,
    validate_submission_semantics,
)
from stargazer.models import (
    CandidateSubmission,
    PlanetParams,
    StargazerTask,
    semi_amplitude_ms,
    simulate_keplerian_rv,
)


@dataclass(frozen=True)
class EvaluationCriteria:
    """Versioned thresholds for Stargazer's scientific pass gates."""

    minimum_delta_bic_per_point: float = 0.0
    maximum_rms_factor: float = 1.5
    minimum_match_score: float = 0.8
    require_count_match: bool = True
    require_complete_matching: bool = True
    minimum_planet_score: float | None = None
    use_task_hints: bool = False

    def __post_init__(self):
        if (
            self.minimum_planet_score is not None
            and not 0 <= self.minimum_planet_score <= 1
        ):
            raise ValueError("minimum_planet_score must be between 0 and 1")


LEGACY_CRITERIA = EvaluationCriteria(
    require_complete_matching=False, use_task_hints=True
)


@dataclass(frozen=True)
class EvaluationResult:
    """Complete internal evaluation result."""

    success: bool
    delta_bic: float
    delta_bic_per_point: float
    rms_ms: float
    mae_ms: float
    match_score: float
    count_difference: int
    ok_delta_bic: bool
    ok_rms: bool
    ok_match: bool
    ok_count: bool
    ok_complete_matching: bool
    ok_individual_matches: bool
    maximum_rms_ms: float
    criteria: EvaluationCriteria
    gamma_per_instrument_ms: dict[str, float]
    matched_pairs: tuple[tuple[int, int, float], ...]

    metrics: dict[str, Any] = field(default_factory=dict)
    success_details: dict[str, Any] = field(default_factory=dict)
    semantic_check: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0

    @property
    def score(self) -> float:
        """Corral's binary benchmark score."""
        return 1.0 if self.success else 0.0


def normalize_submission(submission, task: StargazerTask) -> CandidateSubmission:
    """Compatibility entry point for offline legacy audits."""
    return normalize_candidate(submission, task.public_fit_context())


def _planet_components(
    truth: PlanetParams,
    guess: PlanetParams,
    task: StargazerTask,
    times: np.ndarray,
) -> dict[str, float]:
    truth_amplitude = max(
        1e-6,
        semi_amplitude_ms(
            truth.m_sin_i_mjup,
            truth.P_days,
            truth.e,
            task.star_mass_sun,
        ),
    )
    guess_amplitude = max(
        1e-6,
        semi_amplitude_ms(
            guess.m_sin_i_mjup,
            guess.P_days,
            guess.e,
            task.star_mass_sun,
        ),
    )
    truth_rv = simulate_keplerian_rv((truth,), times, task.star_mass_sun)
    guess_rv = simulate_keplerian_rv((guess,), times, task.star_mass_sun)
    offset = float(np.mean(truth_rv - guess_rv))
    curve_rms = float(np.sqrt(np.mean((truth_rv - guess_rv - offset) ** 2)))
    phase_delta = (
        (guess.l_rad - guess.Omega_rad) - (truth.l_rad - truth.Omega_rad) + np.pi
    ) % (2.0 * np.pi) - np.pi
    return {
        "dlogP": abs(np.log(guess.P_days / truth.P_days)),
        "dlogK": abs(np.log(guess_amplitude / truth_amplitude)),
        "de": abs(guess.e - truth.e),
        "dphase": abs(float(phase_delta)),
        "rv_curve": curve_rms / truth_amplitude,
    }


def _planet_distance(
    truth: PlanetParams,
    guess: PlanetParams,
    task: StargazerTask,
    times: np.ndarray,
) -> float:
    components = _planet_components(truth, guess, task, times)
    weights = {
        "rv_curve": 4.0,
        "dlogP": 1.0,
        "dlogK": 0.5,
        "de": 0.5,
        "dphase": 0.0,
    }
    return float(sum(weights[name] * value for name, value in components.items()))


def _match_planets(
    task: StargazerTask, guesses: tuple[PlanetParams, ...], times: np.ndarray
) -> tuple[float, tuple[tuple[int, int, float], ...]]:
    truth = task.truth_planets
    if not truth or not guesses:
        base_score = 1.0 if not truth and not guesses else 0.0
        count_penalty = -0.25 * abs(len(truth) - len(guesses))
        return base_score + count_penalty, ()

    distances = np.empty((len(truth), len(guesses)), dtype=float)
    for truth_index, truth_planet in enumerate(truth):
        for guess_index, guess_planet in enumerate(guesses):
            distances[truth_index, guess_index] = _planet_distance(
                truth_planet, guess_planet, task, times
            )
    rows, columns = linear_sum_assignment(distances)
    pairs = tuple(
        (int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if distances[row, column] <= 5.0
    )
    pair_scores = [math.exp(-distance) for _, _, distance in pairs]
    base_score = float(np.mean(pair_scores)) if pair_scores else 0.0
    count_penalty = -0.25 * abs(len(truth) - len(guesses))
    return base_score + count_penalty, pairs


def _hint(task: StargazerTask, key: str, default: float, minimum: float) -> float:
    hints = task.metadata.get("hints", {})
    try:
        value = float(hints.get(key))
        return value if np.isfinite(value) and value >= minimum else default
    except (AttributeError, TypeError, ValueError):
        return default


def evaluate_submission(
    task: StargazerTask,
    submission: str | dict[str, Any] | CandidateSubmission,
    criteria: EvaluationCriteria | None = None,
) -> EvaluationResult:
    """Evaluate a candidate planetary system against hidden task truth."""
    thresholds = criteria or EvaluationCriteria()
    public_context = task.public_fit_context(
        maximum_rms_factor=thresholds.maximum_rms_factor
    )
    candidate = normalize_candidate(submission, public_context)
    fit = compute_fit(public_context, candidate)
    candidate_planets = tuple(planet.to_planet_params() for planet in candidate.planets)
    times = np.asarray(public_context.observations.times_days)
    observed = np.asarray(public_context.observations.rvs_ms)
    uncertainties = np.asarray(public_context.observations.sigmas_ms)
    offsets = fit.gamma_per_instrument_ms
    likelihood, bic = fit.likelihood, fit.bic
    delta_bic, delta_bic_per_point = fit.delta_bic, fit.delta_bic_per_point
    rms, mae = fit.rms_ms, fit.mae_ms
    match_score, pairs = _match_planets(task, candidate_planets, times)
    count_difference = len(candidate_planets) - len(task.truth_planets)

    maximum_rms = thresholds.maximum_rms_factor * float(np.median(uncertainties))
    if thresholds.use_task_hints:
        maximum_rms = _hint(task, "max_rms_ms", maximum_rms, np.nextafter(0.0, 1.0))
        thresholds = replace(
            thresholds,
            minimum_match_score=_hint(
                task, "target_match_score", thresholds.minimum_match_score, 0.0
            ),
        )
    ok_delta_bic = delta_bic_per_point > thresholds.minimum_delta_bic_per_point
    ok_rms = rms <= maximum_rms
    ok_match = match_score >= thresholds.minimum_match_score
    ok_count = not thresholds.require_count_match or count_difference == 0
    ok_complete_matching = (
        len(pairs) == len(task.truth_planets) == len(candidate_planets)
    )
    ok_individual_matches = thresholds.minimum_planet_score is None or (
        ok_complete_matching
        and all(
            math.exp(-distance) >= thresholds.minimum_planet_score
            for _, _, distance in pairs
        )
    )
    success = (
        ok_delta_bic
        and ok_rms
        and ok_match
        and ok_count
        and (not thresholds.require_complete_matching or ok_complete_matching)
        and ok_individual_matches
    )
    assignment = {
        "pairs": [list(pair) for pair in pairs],
        "unmatched_truth": [
            i
            for i in range(len(task.truth_planets))
            if i not in {pair[0] for pair in pairs}
        ],
        "unmatched_guess": [
            i
            for i in range(len(candidate_planets))
            if i not in {pair[1] for pair in pairs}
        ],
    }
    components = {
        "likelihood": likelihood / observed.size,
        "delta_bic": delta_bic_per_point,
        "neg_rms": -rms,
        "match": match_score,
        "count": -abs(count_difference),
    }
    reward = sum(
        weight * components[key]
        for key, weight in {
            "likelihood": 1.0,
            "delta_bic": 0.3,
            "neg_rms": 0.1,
            "match": 1.0,
            "count": 0.2,
        }.items()
    )
    details = {
        "median_sigma_ms": float(np.median(uncertainties)),
        "delta_bic_per_point": delta_bic_per_point,
        "rms_ms": rms,
        "ok_delta_bic": bool(ok_delta_bic),
        "ok_rms": bool(ok_rms),
        "min_delta_bic_per_point": thresholds.minimum_delta_bic_per_point,
        "max_rms_ms": maximum_rms,
        "match_score": match_score,
        "count_term": float(-abs(count_difference)),
        "ok_match": bool(ok_match),
        "ok_count": bool(ok_count),
        "min_match_score": thresholds.minimum_match_score,
        "require_count_match": thresholds.require_count_match,
    }
    comparison = {
        "submitted_planets": [
            {
                "P_days": p.P_days,
                "m_sin_i_mjup": p.m_sin_i_mjup,
                "e": p.e,
                "l_rad": p.l_rad,
                "K_ms": semi_amplitude_ms(
                    p.m_sin_i_mjup, p.P_days, p.e, task.star_mass_sun
                ),
            }
            for p in candidate_planets
        ],
        "matched_pairs": [
            {
                "guess_idx": gi,
                "distance": distance,
                "components": _planet_components(
                    task.truth_planets[ti], candidate_planets[gi], task, times
                ),
            }
            for ti, gi, distance in pairs
        ],
        "unmatched_guess": assignment["unmatched_guess"],
    }
    return EvaluationResult(
        success=success,
        delta_bic=float(delta_bic),
        delta_bic_per_point=delta_bic_per_point,
        rms_ms=rms,
        mae_ms=mae,
        match_score=match_score,
        count_difference=count_difference,
        ok_delta_bic=ok_delta_bic,
        ok_rms=ok_rms,
        ok_match=ok_match,
        ok_count=ok_count,
        ok_complete_matching=ok_complete_matching,
        ok_individual_matches=ok_individual_matches,
        maximum_rms_ms=maximum_rms,
        criteria=thresholds,
        gamma_per_instrument_ms=offsets,
        matched_pairs=pairs,
        reward=float(reward),
        success_details=details,
        comparison=comparison,
        semantic_check=validate_submission_semantics(
            canonicalize_plan(parse_submission(submission)), task.public_observation()
        ),
        metrics={
            "rv_model_source": "rv_only_keplerian_from_planets",
            "gamma_per_instrument_ms": offsets,
            "ll": likelihood,
            "bic": bic,
            "delta_bic": float(delta_bic),
            "residuals": {"rms": rms, "mae": mae},
            "bic_null": float(delta_bic + bic),
            "matching": {"score": match_score, "assignment": assignment},
            "components": components,
            "mode": "params_and_model",
        },
    )


def submit_candidate(
    task: StargazerTask, payload: dict[str, Any], session: dict[str, Any]
) -> str:
    """Reproduce historical feedback offline; never expose as an analysis tool."""
    reward, done, success, details, metrics = 0.0, False, False, {}, {}
    try:
        result = evaluate_submission(task, payload, LEGACY_CRITERIA)
    except SemanticSubmissionError as exc:
        output = "Plan rejected by semantic validator:\n" + json.dumps(
            {
                "errors": exc.check["errors"],
                "t_ref_days": exc.check["t_ref_days"],
                "hint": "Use Stargazer-native keys and ensure phase semantics are consistent.",
            },
            indent=2,
        )
    except SubmissionError as exc:
        output = f"Plan rejected: {exc}"
    else:
        session["steps"] += 1
        reward, success = result.reward, bool(result.success)
        done = success
        details, metrics = result.success_details, result.metrics
        output = json.dumps(
            {
                "reward": reward,
                "done": done,
                "success": success,
                "success_details": {
                    k: v for k, v in details.items() if k != "count_term"
                },
                "semantic_check": result.semantic_check,
                "components": {
                    k: v for k, v in metrics["components"].items() if k != "count"
                },
                "residuals": metrics["residuals"],
                "matching": metrics["matching"],
                "comparison": result.comparison,
            },
            indent=2,
        )
    session["done"] = done
    session["force_submit"] = False
    session["history"].append(
        {
            "step": len(session["history"]) + 1,
            "reward": reward,
            "done": done,
            "success": success,
            "success_details": details,
            "metrics": metrics,
        }
    )
    return output


@dataclass(frozen=True)
class StargazerScorer:
    """Private final-answer scorer. It is never called by an analysis tool."""

    task: StargazerTask
    criteria: EvaluationCriteria
    bank_hash: str = "development"
    protocol_version: str = "stargazer-blind-v1"

    def __call__(self, answer: str) -> float:
        return self.evaluate_submission(answer).score

    def evaluate_submission(self, answer: str):
        from corral.evaluation import SubmissionScore

        metadata = {
            "interaction_protocol": self.protocol_version,
            "bank_hash": self.bank_hash,
            "criteria": asdict(self.criteria),
            "scorer": (
                "stargazer-score-v3"
                if self.criteria.minimum_planet_score is not None
                else "stargazer-score-v2"
                if self.criteria.require_complete_matching
                else "stargazer-score-v1"
            ),
        }
        try:
            result = evaluate_submission(self.task, answer, self.criteria)
        except SubmissionError as exc:
            return SubmissionScore(
                score=0.0,
                feedback=str(exc),
                metadata={
                    **metadata,
                    "valid": False,
                    "failure_reasons": ["invalid_candidate"],
                },
            )
        gates = {
            "bic_gate": result.ok_delta_bic,
            "rms_gate": result.ok_rms,
            "physical_match_gate": result.ok_match,
            "count_gate": result.ok_count,
            "complete_matching_gate": not self.criteria.require_complete_matching
            or result.ok_complete_matching,
            "individual_match_gate": result.ok_individual_matches,
        }
        withheld = self.task.metadata.get("withheld_observations")
        if withheld:
            candidate = normalize_submission(answer, self.task)
            times = np.concatenate(
                (self.task.observations.times_days, withheld["times_days"])
            )
            prediction = simulate_keplerian_rv(
                tuple(p.to_planet_params() for p in candidate.planets),
                times,
                self.task.star_mass_sun,
            )[len(self.task.observations.times_days) :]
            prediction += np.asarray(
                [
                    result.gamma_per_instrument_ms[label]
                    for label in withheld["instruments"]
                ]
            )
            residual = np.asarray(withheld["rvs_ms"]) - prediction
            metadata["withheld_prediction_rms_ms"] = float(
                np.sqrt(np.mean(residual**2))
            )
            metadata["withheld_normalized_mse"] = float(
                np.mean((residual / np.asarray(withheld["sigmas_ms"])) ** 2)
            )
        return SubmissionScore(
            score=result.score,
            metadata={
                **metadata,
                "valid": True,
                "failure_reasons": [name for name, ok in gates.items() if not ok],
                "evaluation": json.loads(json.dumps(asdict(result), allow_nan=False)),
            },
        )


def make_stargazer_scorer(
    task: StargazerTask,
    criteria: EvaluationCriteria | None = None,
    *,
    bank_hash: str = "development",
    protocol_version: str = "stargazer-blind-v1",
):
    return StargazerScorer(
        task, criteria or EvaluationCriteria(), bank_hash, protocol_version
    )
