"""Stargazer candidate evaluation and committed-trajectory scoring.

The reference audit uses the same action builder and evaluator as live
submit_action calls. Corral's final answer only closes the execution.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from stargazer.models import (
    CandidateSubmission,
    PlanetParams,
    StargazerTask,
    mass_from_semi_amplitude,
    semi_amplitude_ms,
    simulate_keplerian_rv,
    simulate_submission_rv,
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

    metrics: dict[str, Any] = field(default_factory=dict)
    success_details: dict[str, Any] = field(default_factory=dict)
    semantic_check: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0

    @property
    def score(self) -> float:
        """Corral's binary benchmark score."""
        return 1.0 if self.success else 0.0


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


class SemanticSubmissionError(SubmissionError):
    """A phase/parameter protocol rejection, before consuming an attempt."""

    def __init__(self, check: dict[str, Any]):
        super().__init__(json.dumps(check["errors"]))
        self.check = check


def _coerce_float(value: Any, *, name: str, default: float | None = None) -> float:
    """Best-effort conversion of LLM-provided values into finite floats.

    LLMs sometimes emit JSON nulls (parsed as Python None). Treat those as "missing"
    when a default is provided.
    """
    if value is None:
        if default is None:
            raise SubmissionError(f"`{name}` must be a real number, got null.")
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionError(
            f"`{name}` must be a real number, got {value!r}."
        ) from exc
    if not np.isfinite(out):
        raise SubmissionError(f"`{name}` must be finite, got {out}.")
    return out


def _is_positive_quantity(value) -> bool:
    """True when `value` is a usable positive number, not None, zero or junk."""
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def canonicalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize a plan in-place to avoid conflicting aliases."""
    if not isinstance(plan, dict):
        return plan
    planets_desc = plan.get("planets", [])
    if not isinstance(planets_desc, list):
        return plan
    cleaned = []
    for desc in planets_desc:
        if not isinstance(desc, dict):
            cleaned.append(desc)
            continue
        out = dict(desc)
        if "P_days" not in out and "period_days" in out:
            out["P_days"] = out["period_days"]
        if "e" not in out and "eccentricity" in out:
            out["e"] = out["eccentricity"]
        if "P_days" in out and "period_days" in out:
            out.pop("period_days", None)
        if "e" in out and "eccentricity" in out:
            out.pop("eccentricity", None)
        # Prefer Stargazer-native mass parameterization when it carries a
        # usable value; a null or zero mass must not discard the amplitude.
        if _is_positive_quantity(out.get("m_sin_i_mjup")):
            out.pop("semi_amplitude_ms", None)
            out.pop("K_ms", None)
        if "l_rad" in out:
            for k in (
                "phase_rad",
                "phase_deg",
                "phase",
                "phase_frac",
                "T0_days",
                "T_peri",
            ):
                out.pop(k, None)
        cleaned.append(out)
    plan = dict(plan)
    plan["planets"] = cleaned
    return plan


def _parse_phase_to_l_rad(
    desc: dict[str, Any],
    omega_rad: float,
    Omega_rad: float,
    period: float,
    t0: float,
) -> float:
    """Parse various phase representations and convert to l_rad (mean longitude).

    Canonical rule:
    1. If explicit `l_rad` is present, always use it.
    2. Otherwise infer from legacy phase fields using this order:
       phase_frac, phase_rad/phase_deg/phase, T0_days/T_peri.

    This keeps parser behavior consistent with semantic validation and scoring:
    when `l_rad` is provided, legacy aliases are treated as non-canonical hints.

    Legacy priority (when `l_rad` is absent):
    1. phase_frac (common agent output, interpreted as time-of-periastron fraction)
    2. phase_rad / phase_deg / phase (interpreted as mean anomaly M0)
    3. T0_days / T_peri (time of periastron)
    4. Default to 0.0
    """

    def _get_float(key: str) -> float | None:
        if key in desc:
            return _coerce_float(desc.get(key), name=key, default=None)
        return None

    # Collect all phase-related values
    l_rad_val = _get_float("l_rad")
    phase_frac_val = _get_float("phase_frac")
    phase_rad_val = _get_float("phase_rad")
    phase_deg_val = _get_float("phase_deg")
    phase_val = _get_float("phase")
    # `or` would discard a legitimate T0_days of exactly 0.0.
    t0_days_val = _get_float("T0_days")
    if t0_days_val is None:
        t0_days_val = _get_float("T_peri")

    # Helper: convert phase_frac to l_rad
    # Agent convention: tp = t0 + phase_frac * P (time of periastron)
    # At t=t0, M = -2π * phase_frac, so l = Omega + omega + M
    def l_from_phase_frac(pf: float) -> float:
        M0 = (-2.0 * np.pi * pf) % (2.0 * np.pi)
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)

    # Helper: convert M0 (mean anomaly) to l_rad
    def l_from_M0(M0: float) -> float:
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)

    # Helper: convert time of periastron to l_rad
    def l_from_T0(T0: float) -> float:
        n = 2.0 * np.pi / period
        M0 = (-n * (T0 - t0)) % (2.0 * np.pi)
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)

    # Canonical rule: explicit l_rad wins over any legacy field.
    if l_rad_val is not None:
        return l_rad_val % (2.0 * np.pi)

    candidates = []

    # No explicit l_rad: fall back to legacy fields.
    if phase_frac_val is not None and abs(phase_frac_val) > 1e-9:
        candidates.append(("phase_frac", l_from_phase_frac(phase_frac_val)))

    # Check phase_rad (as M0)
    if phase_rad_val is not None and abs(phase_rad_val) > 1e-9:
        candidates.append(("phase_rad", l_from_M0(phase_rad_val)))

    # Check phase_deg (as M0)
    if phase_deg_val is not None and abs(phase_deg_val) > 1e-9:
        M0_deg = float(np.deg2rad(phase_deg_val))
        candidates.append(("phase_deg", l_from_M0(M0_deg)))

    # Check generic phase (heuristic for radians vs degrees)
    if phase_val is not None and abs(phase_val) > 1e-9:
        if abs(phase_val) <= 2.0 * np.pi + 1e-6:
            M0_phase = phase_val
        elif abs(phase_val) <= 360.0 + 1e-6:
            M0_phase = float(np.deg2rad(phase_val))
        else:
            M0_phase = phase_val
        candidates.append(("phase", l_from_M0(M0_phase)))

    # Check time of periastron
    if t0_days_val is not None:
        candidates.append(("T0_days", l_from_T0(t0_days_val)))

    # If we found non-zero candidates, use the first one (highest priority)
    if candidates:
        return candidates[0][1]

    # Fallback: check for zero-valued fields (in case agent explicitly set them to 0)
    if l_rad_val is not None:
        return l_rad_val % (2.0 * np.pi)
    if phase_frac_val is not None:
        return l_from_phase_frac(phase_frac_val)
    if phase_rad_val is not None:
        return l_from_M0(phase_rad_val)
    if phase_deg_val is not None:
        return l_from_M0(float(np.deg2rad(phase_deg_val)))
    if phase_val is not None:
        if abs(phase_val) <= 2.0 * np.pi + 1e-6:
            return l_from_M0(phase_val)
        elif abs(phase_val) <= 360.0 + 1e-6:
            return l_from_M0(float(np.deg2rad(phase_val)))
        else:
            return l_from_M0(phase_val)

    # Default
    return 0.0


def validate_submission_semantics(
    plan: dict[str, Any],
    observation: dict[str, Any],
    *,
    phase_conflict_tol_rad: float = 0.35,
) -> dict[str, Any]:
    """Validate protocol-level parameter semantics before submission.

    Focuses on:
    - phase semantic consistency (`l_rad` vs `phase_*`/`T0_days`)
    - reference epoch usage (`t_ref = times_days[0]`)
    - legacy alias usage visibility
    """

    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
        return float(abs(d))

    def _try_float(desc: dict[str, Any], key: str) -> float | None:
        if key not in desc:
            return None
        try:
            return _coerce_float(desc.get(key), name=key, default=None)
        except SubmissionError:
            return None

    times = np.asarray(observation.get("times_days", []), dtype=float)
    t_ref = float(times[0]) if times.size else 0.0
    planets_desc = plan.get("planets", [])

    out: dict[str, Any] = {
        "ok": True,
        "t_ref_days": t_ref,
        "errors": [],
        "warnings": [],
        "per_planet": [],
    }

    if not isinstance(planets_desc, list):
        out["ok"] = False
        out["errors"].append("`planets` must be a list.")
        return out

    legacy_keys = {
        "period_days",
        "semi_amplitude_ms",
        "eccentricity",
        "phase_rad",
        "phase_deg",
        "phase",
        "phase_frac",
        "K_ms",
        "T0_days",
        "T_peri",
    }

    for i, desc in enumerate(planets_desc):
        if not isinstance(desc, dict):
            out["ok"] = False
            out["errors"].append(f"planet[{i}] must be an object.")
            continue

        period = _try_float(desc, "period_days")
        if period is None:
            period = _try_float(desc, "P_days")
        omega = _try_float(desc, "omega_rad")
        omega = float(omega) if omega is not None else 0.0
        Omega = _try_float(desc, "Omega_rad")
        Omega = float(Omega) if Omega is not None else 0.0

        if period is None or period <= 0.5:
            out["ok"] = False
            out["errors"].append(
                f"planet[{i}] invalid period; expected `P_days` > 0.5."
            )
            continue

        candidates: dict[str, float] = {}
        l_direct = _try_float(desc, "l_rad")
        if l_direct is not None:
            candidates["l_rad"] = float(l_direct % (2.0 * np.pi))

        phase_rad = _try_float(desc, "phase_rad")
        if phase_rad is not None:
            candidates["phase_rad(M0)"] = float(
                (Omega + omega + phase_rad) % (2.0 * np.pi)
            )

        phase_deg = _try_float(desc, "phase_deg")
        if phase_deg is not None:
            candidates["phase_deg(M0)"] = float(
                (Omega + omega + np.deg2rad(phase_deg)) % (2.0 * np.pi)
            )

        phase = _try_float(desc, "phase")
        if phase is not None:
            if abs(phase) <= 2.0 * np.pi + 1e-6:
                phase_as_rad = phase
            elif abs(phase) <= 360.0 + 1e-6:
                phase_as_rad = float(np.deg2rad(phase))
            else:
                phase_as_rad = phase
            candidates["phase(alias M0)"] = float(
                (Omega + omega + phase_as_rad) % (2.0 * np.pi)
            )

        phase_frac = _try_float(desc, "phase_frac")
        if phase_frac is not None:
            M0 = (-2.0 * np.pi * phase_frac) % (2.0 * np.pi)
            candidates["phase_frac(tp)"] = float((Omega + omega + M0) % (2.0 * np.pi))

        T0 = _try_float(desc, "T0_days")
        if T0 is None:
            T0 = _try_float(desc, "T_peri")
        if T0 is not None:
            n = 2.0 * np.pi / float(period)
            M0 = (-n * (T0 - t_ref)) % (2.0 * np.pi)
            candidates["T0_days(tp)"] = float((Omega + omega + M0) % (2.0 * np.pi))

        planet_info: dict[str, Any] = {
            "planet_idx": i,
            "t_ref_days": t_ref,
            "candidate_l_rad": candidates,
            "resolved_l_rad": float(
                _parse_phase_to_l_rad(desc, omega, Omega, float(period), t_ref)
            ),
        }

        keys = list(candidates.keys())
        if len(keys) >= 2:
            max_diff = 0.0
            worst_pair = None
            for a_idx in range(len(keys)):
                for b_idx in range(a_idx + 1, len(keys)):
                    ka, kb = keys[a_idx], keys[b_idx]
                    d = _ang_diff(candidates[ka], candidates[kb])
                    if d > max_diff:
                        max_diff = d
                        worst_pair = (ka, kb, d)
            planet_info["max_phase_disagreement_rad"] = max_diff
            if max_diff > phase_conflict_tol_rad and worst_pair is not None:
                # If explicit l_rad is provided, treat it as canonical and do not hard-fail.
                # Many agents keep legacy phase_* fields as placeholders (often 0.0).
                if "l_rad" in candidates:
                    out["warnings"].append(
                        f"planet[{i}] phase fields disagree ({worst_pair[0]} vs {worst_pair[1]}, Δ={worst_pair[2]:.3f} rad), "
                        "but `l_rad` is present and treated as canonical."
                    )
                else:
                    out["ok"] = False
                    out["errors"].append(
                        f"planet[{i}] phase semantics conflict: {worst_pair[0]} vs {worst_pair[1]} differ by {worst_pair[2]:.3f} rad. "
                        "Use a single convention or make them consistent."
                    )

        used_legacy = sorted(k for k in desc if k in legacy_keys)
        if used_legacy:
            out["warnings"].append(
                f"planet[{i}] used legacy aliases {used_legacy}; prefer Stargazer native keys "
                f"(P_days, m_sin_i_mjup, e, omega_rad, l_rad)."
            )
        if not any(
            k in desc
            for k in (
                "l_rad",
                "phase_rad",
                "phase_deg",
                "phase",
                "phase_frac",
                "T0_days",
                "T_peri",
            )
        ):
            out["warnings"].append(
                f"planet[{i}] has no phase field; defaulting to l_rad=0 at t_ref={t_ref}."
            )

        out["per_planet"].append(planet_info)

    return out


def normalize_submission(submission, task: StargazerTask) -> CandidateSubmission:
    """Canonicalize the original aliases, clamp parameters, and infer omitted jitter."""
    plan = canonicalize_plan(parse_submission(submission))
    check = validate_submission_semantics(plan, task.public_observation())
    if not check["ok"]:
        raise SemanticSubmissionError(check)
    planets = []
    for raw in plan["planets"][: task.max_planets]:
        period = _coerce_float(raw.get("P_days"), name="period_days/P_days")
        eccentricity = float(
            np.clip(
                _coerce_float(raw.get("e"), name="eccentricity/e", default=0), 0, 0.8
            )
        )
        omega = _coerce_float(raw.get("omega_rad"), name="omega_rad", default=0)
        node = float(
            _coerce_float(raw.get("Omega_rad"), name="Omega_rad", default=0)
            % (2 * np.pi)
        )
        inclination = float(
            np.clip(
                _coerce_float(raw.get("inc_rad"), name="inc_rad", default=np.pi / 2),
                0,
                np.pi,
            )
        )
        mass = _coerce_float(raw.get("m_sin_i_mjup"), name="m_sin_i_mjup", default=0)
        if mass <= 0:
            amplitude = float(
                np.clip(
                    _coerce_float(
                        raw.get("semi_amplitude_ms", raw.get("K_ms")),
                        name="semi_amplitude_ms/K_ms",
                        default=0,
                    ),
                    0,
                    200,
                )
            )
            mass = mass_from_semi_amplitude(
                amplitude, period, eccentricity, task.star_mass_sun
            )
        planets.append(
            {
                "P_days": period,
                "m_sin_i_mjup": float(np.clip(mass, 0.001, 30)),
                "e": eccentricity,
                "inc_rad": inclination,
                "Omega_rad": node,
                "omega_rad": omega % (2 * np.pi),
                "l_rad": _parse_phase_to_l_rad(
                    raw, omega, node, period, task.observations.times_days[0]
                ),
            }
        )
    observations = task.observations
    observed = np.asarray(observations.rvs_ms)
    weights = 1 / np.maximum(np.asarray(observations.sigmas_ms) ** 2, 1e-6)
    offset = _coerce_float(
        plan.get("rv_offset_ms"),
        name="rv_offset_ms",
        default=float(np.average(observed, weights=weights)),
    )
    # The original builder uses REBOUND for omitted-jitter residuals, even
    # though evaluation fits analytic RV-only curves and instrument offsets.
    candidate = CandidateSubmission(planets=planets)
    if plan.get("noise_jitter_ms") is None:
        model = np.full_like(observed, offset)
        if planets:
            model += simulate_submission_rv(
                task, tuple(planet.to_planet_params() for planet in candidate.planets)
            )
        jitter = float(np.sqrt(np.average((observed - model) ** 2, weights=weights)))
    else:
        jitter = _coerce_float(plan["noise_jitter_ms"], name="noise_jitter_ms")
    return candidate.model_copy(update={"noise_jitter_ms": max(0.1, jitter)})


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
    success = ok_delta_bic and ok_rms and ok_match and ok_count
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
    """Evaluate a candidate, ending the interaction only on success."""
    reward, done, success, details, metrics = 0.0, False, False, {}, {}
    try:
        result = evaluate_submission(task, payload)
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


def make_stargazer_scorer(
    task: StargazerTask, criteria: EvaluationCriteria | None = None
):
    """Standalone candidate scorer used by the reference audit."""

    def score_stargazer_submission(answer: str) -> float:
        try:
            return evaluate_submission(task, answer, criteria).score
        except (SubmissionError, ValueError, TypeError, OverflowError):
            return 0.0

    return score_stargazer_submission


def score_execution(state) -> float:
    """Score committed Stargazer submissions; Corral's final text only closes the run."""
    session = state.environment.values.get("hidden_arguments", {}).get(
        "submission_session", {}
    )
    return float(
        any(entry.get("success", False) for entry in session.get("history", []))
    )
