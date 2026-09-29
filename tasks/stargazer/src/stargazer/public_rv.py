"""Public RV numerics shared by analysis helpers and final evaluation.

Only observations, public conventions and submitted parameters enter this module.
It also works as a standalone module: no task loaders or scoring imports.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import rebound
from pydantic import BaseModel, ConfigDict

DAY_SECONDS = 86_400.0
M_SUN_KG = 1.98847e30
M_JUPITER_KG = 1.89813e27


class CandidatePlanet(BaseModel):
    """Canonical planet fields for normalized submissions and reference audits."""

    model_config = ConfigDict(extra="forbid")

    P_days: float
    m_sin_i_mjup: float
    e: float
    omega_rad: float
    l_rad: float
    inc_rad: float | None = None
    Omega_rad: float | None = None

    def to_planet_params(self) -> PlanetParams:
        """Convert the public schema to the evaluator's native dataclass."""
        return PlanetParams(
            P_days=self.P_days,
            m_sin_i_mjup=self.m_sin_i_mjup,
            e=self.e,
            omega_rad=self.omega_rad % (2.0 * math.pi),
            l_rad=self.l_rad % (2.0 * math.pi),
            inc_rad=math.pi / 2.0 if self.inc_rad is None else self.inc_rad,
            Omega_rad=0.0
            if self.Omega_rad is None
            else self.Omega_rad % (2.0 * math.pi),
        )


class CandidateSubmission(BaseModel):
    """Canonical candidate; omitted jitter is estimated by the action builder."""

    model_config = ConfigDict(extra="forbid")

    planets: list[CandidatePlanet]
    noise_jitter_ms: float | None = None

    def canonical_payload(self) -> dict[str, Any]:
        """Return JSON-compatible canonical fields, omitting compatibility defaults."""
        return self.model_dump(exclude_none=True)


@dataclass(frozen=True)
class PlanetParams:
    """Planet parameters in Stargazer's RV convention."""

    P_days: float
    m_sin_i_mjup: float
    e: float
    omega_rad: float
    l_rad: float
    inc_rad: float = math.pi / 2.0
    Omega_rad: float = 0.0
    m_true_mjup: float | None = None


@dataclass(frozen=True)
class Observations:
    """One radial-velocity time series."""

    times_days: tuple[float, ...]
    rvs_ms: tuple[float, ...]
    sigmas_ms: tuple[float, ...]
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class PublicFitContext:
    """Immutable allowlist of all data needed for candidate fitting."""

    observations: Observations
    star_mass_sun: float
    max_planets: int = 7
    maximum_rms_factor: float = 1.5
    los_axis: str = "x"
    integrator_preference: str = "whfast"

    @property
    def t_ref_days(self) -> float:
        return self.observations.times_days[0]

    def public_observation(self) -> dict[str, Any]:
        return {
            "times_days": self.observations.times_days,
            "rvs_ms": self.observations.rvs_ms,
            "sigmas_ms": self.observations.sigmas_ms,
            "instruments": self.observations.instruments,
            "meta": {"star_mass_sun": self.star_mass_sun},
        }


def semi_amplitude_ms(
    m_sin_i_mjup: float,
    period_days: float,
    eccentricity: float,
    star_mass_sun: float,
) -> float:
    """Approximate stellar RV semi-amplitude in metres per second."""
    if m_sin_i_mjup < 0.0:
        raise ValueError("m_sin_i_mjup must be non-negative")
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    if star_mass_sun <= 0.0:
        raise ValueError("star_mass_sun must be positive")
    period_years = period_days / 365.25
    return float(
        28.4329
        * m_sin_i_mjup
        * star_mass_sun ** (-2.0 / 3.0)
        * period_years ** (-1.0 / 3.0)
        / math.sqrt(1.0 - eccentricity**2)
    )


def mass_from_semi_amplitude(
    semi_amplitude: float,
    period_days: float,
    eccentricity: float,
    star_mass_sun: float,
) -> float:
    """Invert :func:`semi_amplitude_ms` for minimum planet mass."""
    if semi_amplitude < 0.0:
        raise ValueError("semi_amplitude must be non-negative")
    scale = semi_amplitude_ms(1.0, period_days, eccentricity, star_mass_sun)
    return float(semi_amplitude / scale)


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.mod(values, 2.0 * np.pi)


def _solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    eccentricity = float(np.clip(eccentricity, 0.0, 0.999999))
    if eccentricity == 0.0:
        return _wrap_angle(mean_anomaly)

    eccentric_anomaly = _wrap_angle(mean_anomaly + eccentricity * np.sin(mean_anomaly))
    for _ in range(80):
        residual = (
            eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        )
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        step = residual / derivative
        eccentric_anomaly -= step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return _wrap_angle(eccentric_anomaly)


def simulate_keplerian_rv(
    planets: tuple[PlanetParams, ...] | list[PlanetParams],
    times_days: np.ndarray,
    star_mass_sun: float,
    gamma_ms: float = 0.0,
    *,
    t_ref_days: float | None = None,
) -> np.ndarray:
    """Forward-model a non-interacting multi-planet RV signal.

    `l_rad` is mean longitude at `t_ref_days` (default: `times_days[0]`).
    In the RV-only
    convention `Omega_rad` is zero, but retaining it here also supports raw
    released task records before compatibility conversion.
    """
    times = np.asarray(times_days, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=float)

    reference_time = float(times[0]) if t_ref_days is None else float(t_ref_days)
    rv = np.zeros(times.shape, dtype=float)
    for planet in planets:
        if planet.P_days <= 0.0:
            continue
        eccentricity = float(np.clip(planet.e, 0.0, 0.95))
        omega = float(planet.omega_rad) % (2.0 * np.pi)
        ascending_node = float(planet.Omega_rad) % (2.0 * np.pi)
        mean_anomaly_0 = (float(planet.l_rad) - ascending_node - omega) % (2.0 * np.pi)
        mean_anomaly = _wrap_angle(
            mean_anomaly_0
            + (2.0 * np.pi / float(planet.P_days)) * (times - reference_time)
        )
        eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
        true_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
            np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
        )
        amplitude = semi_amplitude_ms(
            planet.m_sin_i_mjup,
            planet.P_days,
            eccentricity,
            star_mass_sun,
        )
        rv += amplitude * (np.cos(true_anomaly + omega) + eccentricity * np.cos(omega))
    return rv + float(gamma_ms)


def _simulate_legacy_rebound_rv(
    raw_config: dict[str, Any],
    planets: tuple[PlanetParams, ...],
    times: np.ndarray,
    t_ref_days: float = 0.0,
) -> np.ndarray:
    """Reproduce the clean signal in a released synthetic task."""
    simulation = rebound.Simulation()
    simulation.units = ["msun", "m", "s"]
    star = raw_config["star"]
    simulation.add(m=float(star["M_star_sun"]))

    for planet in planets:
        if planet.P_days <= 0 or not 0 <= planet.e < 1:
            raise ValueError("Invalid period or eccentricity")
        true_mass_mjup = planet.m_true_mjup
        if true_mass_mjup is None:
            sin_inclination = math.sin(planet.inc_rad)
            if abs(sin_inclination) < 1e-3:
                raise ValueError("Cannot recover a true mass for a face-on orbit")
            true_mass_mjup = planet.m_sin_i_mjup / sin_inclination
            if true_mass_mjup > 20.0:
                raise ValueError("Derived true mass exceeds 20 Jupiter masses")
        mass_solar = true_mass_mjup * M_JUPITER_KG / M_SUN_KG
        simulation.add(
            m=mass_solar,
            P=planet.P_days * DAY_SECONDS,
            h=planet.e * math.sin(planet.omega_rad),
            k=planet.e * math.cos(planet.omega_rad),
            ix=math.sin(planet.inc_rad / 2.0) * math.sin(planet.Omega_rad),
            iy=math.sin(planet.inc_rad / 2.0) * math.cos(planet.Omega_rad),
            l=planet.l_rad,
        )

    simulation.move_to_com()
    preference = str(raw_config.get("integrator_preference", "whfast")).lower()
    simulation.integrator = "ias15" if preference == "ias15" else "whfast"
    if simulation.integrator == "whfast" and planets:
        simulation.dt = max(
            min(planet.P_days for planet in planets) * DAY_SECONDS / 50.0,
            1e-3,
        )

    axis = str(raw_config.get("los_axis", "x")).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported line-of-sight axis: {axis}")

    rv = np.zeros(times.shape, dtype=float)
    for index, time_days in enumerate(times):
        simulation.integrate((float(time_days) - t_ref_days) * DAY_SECONDS)
        host = simulation.particles[0]
        rv[index] = {"x": host.vx, "y": host.vy, "z": host.vz}[axis]
    return rv


def simulate_submission_rv(
    task: PublicFitContext, planets: tuple[PlanetParams, ...]
) -> np.ndarray:
    """Reproduce the original action builder's REBOUND model and analytic fallback.

    This model estimates omitted jitter. The evaluator independently uses the
    analytic RV-only model when computing its likelihood and matching score.
    """
    times = np.asarray(task.observations.times_days, dtype=float)
    axis = task.los_axis
    axis = axis if axis in {"x", "y", "z"} else "x"
    shift = {"x": np.pi / 2, "y": np.pi, "z": 0.0}[axis]
    oriented = tuple(
        replace(
            planet,
            Omega_rad=(planet.Omega_rad + shift) % (2 * np.pi),
            l_rad=(planet.l_rad + shift) % (2 * np.pi),
        )
        for planet in planets
    )
    try:
        return _simulate_legacy_rebound_rv(
            {
                "star": {"M_star_sun": task.star_mass_sun},
                "los_axis": axis,
                "integrator_preference": task.integrator_preference,
            },
            oriented,
            times,
            t_ref_days=float(times[0]),
        )
    except Exception:
        return simulate_keplerian_rv(planets, times, task.star_mass_sun)


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


class SubmissionError(ValueError):
    """A controlled candidate-format error."""


def parse_submission(
    submission: str | dict[str, Any] | CandidateSubmission,
) -> dict[str, Any]:
    """Parse a JSON or dictionary candidate submission."""
    if isinstance(submission, CandidateSubmission):
        return submission.canonical_payload()
    if isinstance(submission, dict):
        return _check_payload(submission)
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
    except (ValueError, RecursionError) as exc:
        raise SubmissionError(
            "Submission JSON exceeds supported numeric or nesting limits"
        ) from exc
    if not isinstance(parsed, dict):
        raise SubmissionError("Submission JSON must contain an object")
    return _check_payload(parsed)


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
    except (TypeError, ValueError, OverflowError) as exc:
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


def normalize_candidate(submission, task: PublicFitContext) -> CandidateSubmission:
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


def _check_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "planets" not in payload or not isinstance(payload["planets"], list):
        raise SubmissionError("`planets` is required and must be a list")
    if any(not isinstance(planet, dict) for planet in payload["planets"]):
        raise SubmissionError("Every planet must be an object")
    # JSON-compatible data only: reject non-finite values, bools in numeric
    # fields, and structured values before they reach numerical libraries.
    numeric = {
        "P_days",
        "period_days",
        "m_sin_i_mjup",
        "e",
        "eccentricity",
        "omega_rad",
        "Omega_rad",
        "inc_rad",
        "l_rad",
        "K_ms",
        "semi_amplitude_ms",
        "phase",
        "phase_frac",
        "phase_rad",
        "phase_deg",
        "T0_days",
        "T_peri",
    }
    for planet in payload["planets"]:
        for key in numeric & planet.keys():
            value = planet[key]
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise SubmissionError(f"`{key}` must be a number or null")
                _coerce_float(value, name=key)
    for key in ("noise_jitter_ms", "rv_offset_ms"):
        value = payload.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SubmissionError(f"`{key}` must be a number or null")
            _coerce_float(value, name=key)
    return payload


@dataclass(frozen=True)
class FitDiagnostics:
    candidate: CandidateSubmission
    semantic_check: dict[str, Any]
    gamma_per_instrument_ms: dict[str, float]
    likelihood: float
    bic: float
    null_bic: float
    delta_bic: float
    delta_bic_per_point: float
    rms_ms: float
    mae_ms: float
    maximum_rms_ms: float
    planet_model_ms: np.ndarray
    model_ms: np.ndarray
    residuals_ms: np.ndarray

    def public_payload(self) -> dict[str, Any]:
        """Explicit public allowlist; never serialize a private evaluation."""
        return {
            "valid": True,
            "errors": [],
            "warnings": list(self.semantic_check["warnings"]),
            "candidate": self.candidate.canonical_payload(),
            "gamma_per_instrument_ms": dict(self.gamma_per_instrument_ms),
            "log_likelihood": self.likelihood,
            "bic": self.bic,
            "null_bic": self.null_bic,
            "delta_bic": self.delta_bic,
            "delta_bic_per_point": self.delta_bic_per_point,
            "residuals": {"rms": self.rms_ms, "mae": self.mae_ms},
            "maximum_rms_ms": self.maximum_rms_ms,
        }


def _compute_fit(
    public_context: PublicFitContext, candidate: CandidateSubmission
) -> FitDiagnostics:
    """Compute only quantities determined by observations and a candidate."""
    candidate_planets = tuple(planet.to_planet_params() for planet in candidate.planets)
    observations = public_context.observations
    times = np.asarray(observations.times_days, dtype=float)
    observed = np.asarray(observations.rvs_ms, dtype=float)
    uncertainties = np.asarray(observations.sigmas_ms, dtype=float)
    labels = np.asarray(observations.instruments, dtype=str)

    planet_model = simulate_keplerian_rv(
        candidate_planets, times, public_context.star_mass_sun
    )
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

    numbers = (
        likelihood,
        bic,
        delta_bic,
        delta_bic_per_point,
        rms,
        mae,
        *offsets.values(),
    )
    if not all(np.isfinite(value) for value in numbers):
        raise SubmissionError("Candidate produced non-finite fit diagnostics")
    return FitDiagnostics(
        candidate=candidate,
        semantic_check={"warnings": [], "errors": [], "ok": True},
        gamma_per_instrument_ms=offsets,
        likelihood=likelihood,
        bic=bic,
        null_bic=float(delta_bic + bic),
        delta_bic=float(delta_bic),
        delta_bic_per_point=delta_bic_per_point,
        rms_ms=rms,
        mae_ms=mae,
        maximum_rms_ms=public_context.maximum_rms_factor
        * float(np.median(uncertainties)),
        planet_model_ms=planet_model,
        model_ms=model,
        residuals_ms=residuals,
    )


def compute_fit(
    public_context: PublicFitContext, candidate: CandidateSubmission
) -> FitDiagnostics:
    """Classify candidate-induced numerical overflow as a validation failure."""
    try:
        return _compute_fit(public_context, candidate)
    except (OverflowError, FloatingPointError) as exc:
        raise SubmissionError("Candidate exceeds supported numerical range") from exc


def validate_fit(
    public_context: PublicFitContext, payload: Any, *, include_arrays: bool = False
) -> dict[str, Any]:
    """Validate a candidate without any reference-dependent code path."""
    try:
        parsed = parse_submission(payload)
        check = validate_submission_semantics(
            canonicalize_plan(parsed), public_context.public_observation()
        )
        candidate = normalize_candidate(parsed, public_context)
        fit = compute_fit(public_context, candidate)
        result = fit.public_payload()
    except SemanticSubmissionError as exc:
        return {
            "valid": False,
            "errors": exc.check["errors"],
            "warnings": exc.check["warnings"],
        }
    except SubmissionError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}
    warnings = list(check["warnings"])
    if len(parsed["planets"]) > public_context.max_planets:
        warnings.append(
            f"Only the first {public_context.max_planets} planets are evaluated."
        )
    warnings.append(
        "The returned candidate contains effective clipped parameters and jitter (minimum 0.1 m/s)."
    )
    result["warnings"] = warnings
    if include_arrays:
        result.update(
            planet_model_ms=fit.planet_model_ms,
            model_ms=fit.model_ms,
            residuals_ms=fit.residuals_ms,
        )
    return result


class PublicRV:
    """Local analysis helpers retaining only an immutable public data copy."""

    def __init__(self, context: PublicFitContext):
        self.context = context

    def predict(self, planets, *, times_days=None, per_planet=False):
        """Return candidate RVs (m/s), without offsets, at the original epoch.

        With per_planet=True return (total, contributions), where contributions
        has shape (number of normalized candidate planets, number of times).
        Custom times always retain context.t_ref_days as the phase epoch.
        """
        candidate = normalize_candidate(
            {"planets": planets, "noise_jitter_ms": 0.1}, self.context
        )
        times = np.asarray(
            self.context.observations.times_days if times_days is None else times_days,
            dtype=float,
        )
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("times_days must be a finite one-dimensional array")
        params = tuple(p.to_planet_params() for p in candidate.planets)
        total = simulate_keplerian_rv(
            params,
            times,
            self.context.star_mass_sun,
            t_ref_days=self.context.t_ref_days,
        )
        if not per_planet:
            return total
        contributions = np.asarray(
            [
                simulate_keplerian_rv(
                    (p,),
                    times,
                    self.context.star_mass_sun,
                    t_ref_days=self.context.t_ref_days,
                )
                for p in params
            ]
        ).reshape((len(params), times.size))
        return total, contributions

    def diagnostics(self, planets, noise_jitter_ms=0.1, *, rv_offset_ms=None):
        """Return validator scalars plus planet_model_ms, model_ms, residuals_ms.

        None jitter preserves the evaluator's legacy REBOUND estimate. Invalid
        candidates return the same compact errors as validate_fit, without arrays.
        """
        return validate_fit(
            self.context,
            {
                "planets": planets,
                "noise_jitter_ms": noise_jitter_ms,
                "rv_offset_ms": rv_offset_ms,
            },
            include_arrays=True,
        )
