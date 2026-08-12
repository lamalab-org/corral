"""Four-criterion Stargazer evaluator.

Candidate systems are forward-modelled and must simultaneously clear the
published statistical-fit, residual-quality, physical-match, and planet-count
criteria.  The public Corral score is binary; the same evaluator also powers an
iterative tool that returns criterion-level diagnostics without revealing the
ground truth.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError
from scipy.optimize import linear_sum_assignment

from stargazer.models import (
    CandidateSubmission,
    PlanetParams,
    StargazerTask,
    mass_from_semi_amplitude,
    semi_amplitude_ms,
    simulate_keplerian_rv,
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


class SubmissionError(ValueError):
    """Raised when a candidate planetary system is malformed."""


@dataclass(frozen=True)
class EvaluationCriteria:
    """Thresholds for the four Stargazer pass gates."""

    minimum_delta_bic_per_point: float = 0.0
    maximum_rms_factor: float = 1.5
    minimum_match_score: float = 0.8
    require_count_match: bool = True


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
    maximum_rms_ms: float
    criteria: EvaluationCriteria
    gamma_per_instrument_ms: dict[str, float]
    matched_pairs: tuple[tuple[int, int, float], ...]

    @property
    def score(self) -> float:
        """Corral's binary benchmark score."""
        return 1.0 if self.success else 0.0

    def agent_feedback(self) -> dict[str, Any]:
        """Return detailed feedback with evaluator-only truth removed."""
        return {
            "success": self.success,
            "criteria": {
                "statistical_fit": {
                    "passed": self.ok_delta_bic,
                    "delta_bic": self.delta_bic,
                    "delta_bic_per_point": self.delta_bic_per_point,
                    "minimum_delta_bic_per_point": (
                        self.criteria.minimum_delta_bic_per_point
                    ),
                },
                "residual_quality": {
                    "passed": self.ok_rms,
                    "rms_ms": self.rms_ms,
                    "mae_ms": self.mae_ms,
                    "maximum_rms_ms": self.maximum_rms_ms,
                },
                "parameter_recovery": {
                    "passed": self.ok_match,
                    "match_score": self.match_score,
                    "minimum_match_score": self.criteria.minimum_match_score,
                },
                "planet_count": {"passed": self.ok_count},
            },
            "gamma_per_instrument_ms": self.gamma_per_instrument_ms,
        }


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise SubmissionError(f"{name} must be finite")
    return result


def parse_submission(
    submission: str | dict[str, Any] | CandidateSubmission,
) -> dict[str, Any]:
    """Parse a JSON or dictionary candidate submission."""
    if isinstance(submission, CandidateSubmission):
        return submission.canonical_payload()
    if isinstance(submission, dict):
        return submission
    if not isinstance(submission, str):
        raise SubmissionError("Submission must be a JSON object")
    text = submission.strip()
    fenced = _JSON_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"Submission is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SubmissionError("Submission JSON must contain an object")
    return parsed


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _canonical_planet_payload(raw_planet: dict[str, Any], index: int) -> dict[str, Any]:
    """Translate documented legacy aliases to the canonical planet fields."""
    prefix = f"planets[{index}]"
    allowed_fields = {
        "P_days",
        "period_days",
        "m_sin_i_mjup",
        "semi_amplitude_ms",
        "K_ms",
        "e",
        "eccentricity",
        "omega_rad",
        "l_rad",
        "mean_longitude_rad",
        "phase_rad",
        "phase_deg",
        "inc_rad",
        "Omega_rad",
    }
    unexpected = sorted(set(raw_planet) - allowed_fields)
    if unexpected:
        raise SubmissionError(
            f"{prefix} contains unsupported fields: {', '.join(unexpected)}"
        )

    period = _finite_float(
        _first_present(raw_planet, ("P_days", "period_days")),
        name=f"{prefix}.P_days",
    )
    if period <= 0.5:
        raise SubmissionError(f"{prefix}.P_days must be greater than 0.5")

    raw_eccentricity = _first_present(raw_planet, ("e", "eccentricity"))
    eccentricity = _finite_float(
        0.0 if raw_eccentricity is None else raw_eccentricity,
        name=f"{prefix}.e",
    )
    if not 0.0 <= eccentricity <= 0.8:
        raise SubmissionError(f"{prefix}.e must be between 0 and 0.8")

    omega = _finite_float(
        raw_planet.get("omega_rad", 0.0), name=f"{prefix}.omega_rad"
    ) % (2.0 * np.pi)
    ascending_node = _finite_float(
        raw_planet.get("Omega_rad", 0.0), name=f"{prefix}.Omega_rad"
    ) % (2.0 * np.pi)
    inclination = _finite_float(
        raw_planet.get("inc_rad", math.pi / 2.0),
        name=f"{prefix}.inc_rad",
    )

    longitude_value = _first_present(raw_planet, ("l_rad", "mean_longitude_rad"))
    if longitude_value is not None:
        mean_longitude = _finite_float(longitude_value, name=f"{prefix}.l_rad")
    elif "phase_rad" in raw_planet:
        mean_anomaly = _finite_float(
            raw_planet["phase_rad"], name=f"{prefix}.phase_rad"
        )
        mean_longitude = ascending_node + omega + mean_anomaly
    elif "phase_deg" in raw_planet:
        mean_anomaly = math.radians(
            _finite_float(raw_planet["phase_deg"], name=f"{prefix}.phase_deg")
        )
        mean_longitude = ascending_node + omega + mean_anomaly
    else:
        mean_longitude = ascending_node + omega

    raw_mass = raw_planet.get("m_sin_i_mjup")
    raw_amplitude = _first_present(raw_planet, ("semi_amplitude_ms", "K_ms"))
    if raw_mass is None and raw_amplitude is None:
        raise SubmissionError(f"{prefix} needs `m_sin_i_mjup` or `semi_amplitude_ms`")

    return {
        "P_days": period,
        "m_sin_i_mjup": raw_mass,
        "semi_amplitude_ms": raw_amplitude,
        "e": eccentricity,
        "omega_rad": omega,
        "l_rad": mean_longitude % (2.0 * np.pi),
        "inc_rad": inclination,
        "Omega_rad": ascending_node,
    }


def normalize_submission(
    submission: str | dict[str, Any] | CandidateSubmission, task: StargazerTask
) -> CandidateSubmission:
    """Parse compatibility inputs into the shared canonical submission model."""
    payload = parse_submission(submission)
    unexpected = sorted(set(payload) - {"planets", "noise_jitter_ms", "noise"})
    if unexpected:
        raise SubmissionError(
            f"Submission contains unsupported fields: {', '.join(unexpected)}"
        )
    raw_planets = payload.get("planets")
    if not isinstance(raw_planets, list):
        raise SubmissionError("`planets` must be a list")
    if len(raw_planets) > task.max_planets:
        raise SubmissionError(
            f"At most {task.max_planets} planets may be submitted for this task"
        )

    planets: list[dict[str, Any]] = []
    for index, raw_planet in enumerate(raw_planets):
        if not isinstance(raw_planet, dict):
            raise SubmissionError(f"planets[{index}] must be an object")
        canonical = _canonical_planet_payload(raw_planet, index)
        raw_mass = canonical.pop("m_sin_i_mjup")
        raw_amplitude = canonical.pop("semi_amplitude_ms")
        if raw_mass is not None:
            mass = _finite_float(raw_mass, name=f"planets[{index}].m_sin_i_mjup")
            if mass < 0.0:
                raise SubmissionError(
                    f"planets[{index}].m_sin_i_mjup must be non-negative"
                )
        else:
            amplitude = _finite_float(
                raw_amplitude, name=f"planets[{index}].semi_amplitude_ms"
            )
            if amplitude < 0.0:
                raise SubmissionError(
                    f"planets[{index}].semi_amplitude_ms must be non-negative"
                )
            mass = mass_from_semi_amplitude(
                amplitude,
                canonical["P_days"],
                canonical["e"],
                task.star_mass_sun,
            )
        canonical["m_sin_i_mjup"] = mass
        planets.append(canonical)

    noise = payload.get("noise", {})
    if noise is None:
        noise = {}
    if not isinstance(noise, dict):
        raise SubmissionError("`noise` must be an object")
    raw_jitter = payload.get("noise_jitter_ms", noise.get("sigma_jitter_ms", 0.0))
    jitter = _finite_float(raw_jitter, name="noise_jitter_ms")
    if jitter < 0.0:
        raise SubmissionError("noise_jitter_ms must be non-negative")
    try:
        return CandidateSubmission.model_validate(
            {"planets": planets, "noise_jitter_ms": jitter}
        )
    except ValidationError as exc:
        raise SubmissionError(
            f"Submission does not match the canonical schema: {exc}"
        ) from exc


def _log_likelihood(
    observed: np.ndarray,
    model: np.ndarray,
    uncertainties: np.ndarray,
    jitter: float,
) -> float:
    variance = uncertainties**2 + jitter**2 + 1e-12
    return float(
        -0.5
        * np.sum((observed - model) ** 2 / variance + np.log(2.0 * np.pi * variance))
    )


def _fit_instrument_offsets(
    observed: np.ndarray,
    planet_model: np.ndarray,
    uncertainties: np.ndarray,
    labels: np.ndarray,
    jitter: float,
) -> tuple[np.ndarray, dict[str, float]]:
    model = planet_model.copy()
    offsets: dict[str, float] = {}
    variance = uncertainties**2 + jitter**2 + 1e-12
    for label in np.unique(labels):
        mask = labels == label
        weights = 1.0 / variance[mask]
        offset = float(
            np.sum(weights * (observed[mask] - planet_model[mask])) / np.sum(weights)
        )
        offsets[str(label)] = offset
        model[mask] += offset
    return model, offsets


def _null_bic(
    observed: np.ndarray, uncertainties: np.ndarray, labels: np.ndarray
) -> float:
    null_model = np.empty_like(observed)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        weights = 1.0 / (uncertainties[mask] ** 2 + 1e-12)
        null_model[mask] = np.sum(weights * observed[mask]) / np.sum(weights)
    likelihood = _log_likelihood(observed, null_model, uncertainties, 0.0)
    return float(-2.0 * likelihood + len(unique_labels) * np.log(observed.size))


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
        "rv_curve": curve_rms / truth_amplitude,
        "dlogP": abs(math.log(guess.P_days / truth.P_days)),
        "dlogK": abs(math.log(guess_amplitude / truth_amplitude)),
        "de": abs(guess.e - truth.e),
        "dphase": abs(float(phase_delta)),
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


def evaluate_submission(
    task: StargazerTask,
    submission: str | dict[str, Any] | CandidateSubmission,
    criteria: EvaluationCriteria | None = None,
) -> EvaluationResult:
    """Evaluate a candidate planetary system against hidden task truth."""
    thresholds = criteria or EvaluationCriteria()
    candidate = normalize_submission(submission, task)
    candidate_planets = tuple(planet.to_planet_params() for planet in candidate.planets)
    observations = task.observations
    times = np.asarray(observations.times_days, dtype=float)
    observed = np.asarray(observations.rvs_ms, dtype=float)
    uncertainties = np.asarray(observations.sigmas_ms, dtype=float)
    labels = np.asarray(observations.instruments, dtype=str)

    planet_model = simulate_keplerian_rv(candidate_planets, times, task.star_mass_sun)
    model, offsets = _fit_instrument_offsets(
        observed,
        planet_model,
        uncertainties,
        labels,
        candidate.noise_jitter_ms,
    )
    likelihood = _log_likelihood(
        observed, model, uncertainties, candidate.noise_jitter_ms
    )
    parameter_count = len(candidate_planets) * 5 + len(np.unique(labels))
    bic = float(-2.0 * likelihood + parameter_count * np.log(observed.size))
    delta_bic = _null_bic(observed, uncertainties, labels) - bic
    delta_bic_per_point = float(delta_bic / observed.size)
    residuals = observed - model
    rms = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    match_score, pairs = _match_planets(task, candidate_planets, times)
    count_difference = len(candidate_planets) - len(task.truth_planets)

    maximum_rms = thresholds.maximum_rms_factor * float(np.median(uncertainties))
    ok_delta_bic = delta_bic_per_point > thresholds.minimum_delta_bic_per_point
    ok_rms = rms <= maximum_rms
    ok_match = match_score >= thresholds.minimum_match_score
    ok_count = not thresholds.require_count_match or count_difference == 0
    success = ok_delta_bic and ok_rms and ok_match and ok_count
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
        maximum_rms_ms=maximum_rms,
        criteria=thresholds,
        gamma_per_instrument_ms=offsets,
        matched_pairs=pairs,
    )


def make_stargazer_scorer(
    task: StargazerTask, criteria: EvaluationCriteria | None = None
):
    """Bind hidden task truth into Corral's one-argument scoring contract."""

    def score_stargazer_submission(answer: str) -> float:
        """Return one only when all four Stargazer criteria pass."""
        try:
            return evaluate_submission(task, answer, criteria).score
        except (SubmissionError, ValueError, TypeError, OverflowError):
            return 0.0

    return score_stargazer_submission
