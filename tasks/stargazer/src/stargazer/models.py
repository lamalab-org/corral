"""Data model and radial-velocity forward models for Stargazer.

The released synthetic bank was generated with REBOUND.  Stargazer's current
evaluator uses non-interacting, radial-velocity-only Keplerians, so legacy
synthetic tasks are converted on load while preserving their original noise
realisation.  Real archival tasks already use the RV-only convention.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import rebound
from pydantic import BaseModel, ConfigDict

DAY_SECONDS = 86_400.0
M_SUN_KG = 1.98847e30
M_JUPITER_KG = 1.89813e27


class CandidatePlanet(BaseModel):
    """Canonical agent-facing parameters for one candidate planet."""

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
    """Canonical JSON contract shared by diagnostics and final scoring."""

    model_config = ConfigDict(extra="forbid")

    planets: list[CandidatePlanet]
    noise_jitter_ms: float = 0.0

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
class InstrumentParams:
    """Instrument metadata needed to reconstruct legacy observations."""

    label: str
    gamma_ms: float = 0.0


@dataclass(frozen=True)
class Observations:
    """One radial-velocity time series."""

    times_days: tuple[float, ...]
    rvs_ms: tuple[float, ...]
    sigmas_ms: tuple[float, ...]
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class StargazerTask:
    """Normalized benchmark task with evaluator-only planetary truth."""

    task_id: str
    source: str
    truth_difficulty: int
    star_mass_sun: float
    truth_planets: tuple[PlanetParams, ...]
    observations: Observations
    instruments: tuple[InstrumentParams, ...]
    metadata: dict[str, Any]

    @property
    def max_planets(self) -> int:
        """Published submission limit for this task family."""
        return 7 if self.source == "real" else 4

    def public_summary(self) -> dict[str, Any]:
        """Return metadata safe to show to an agent."""
        times = np.asarray(self.observations.times_days, dtype=float)
        sigmas = np.asarray(self.observations.sigmas_ms, dtype=float)
        return {
            "task_id": self.task_id,
            "source": self.source,
            "level_difficulty": self.truth_difficulty,
            "star_mass_solar": self.star_mass_sun,
            "num_observations": int(times.size),
            "observation_span_days": (float(np.ptp(times)) if times.size > 1 else 0.0),
            "median_uncertainty_ms": float(np.median(sigmas)),
            "instrument_labels": sorted(set(self.observations.instruments)),
            "reference_epoch_days": float(times[0]) if times.size else 0.0,
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
) -> np.ndarray:
    """Forward-model a non-interacting multi-planet RV signal.

    ``l_rad`` is mean longitude at ``times_days[0]``.  In the RV-only
    convention ``Omega_rad`` is zero, but retaining it here also supports raw
    released task records before compatibility conversion.
    """
    times = np.asarray(times_days, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=float)

    reference_time = float(times[0])
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


def _planet_from_dict(payload: dict[str, Any]) -> PlanetParams:
    return PlanetParams(
        P_days=float(payload["P_days"]),
        m_sin_i_mjup=float(payload["m_sin_i_mjup"]),
        e=float(payload["e"]),
        omega_rad=float(payload["omega_rad"]),
        l_rad=float(payload["l_rad"]),
        inc_rad=float(payload.get("inc_rad", math.pi / 2.0)),
        Omega_rad=float(payload.get("Omega_rad", 0.0)),
        m_true_mjup=(
            None
            if payload.get("m_true_mjup") is None
            else float(payload["m_true_mjup"])
        ),
    )


def _simulate_legacy_rebound_rv(
    raw_config: dict[str, Any], planets: tuple[PlanetParams, ...], times: np.ndarray
) -> np.ndarray:
    """Reproduce the clean signal in a released synthetic task."""
    simulation = rebound.Simulation()
    simulation.units = ["msun", "m", "s"]
    star = raw_config["star"]
    simulation.add(m=float(star["M_star_sun"]))

    for planet in planets:
        sin_inclination = math.sin(planet.inc_rad)
        if abs(sin_inclination) < 1e-3:
            raise ValueError("Cannot recover a true mass for a face-on orbit")
        true_mass_mjup = (
            planet.m_true_mjup
            if planet.m_true_mjup is not None
            else planet.m_sin_i_mjup / sin_inclination
        )
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
        simulation.integrate(float(time_days) * DAY_SECONDS)
        host = simulation.particles[0]
        rv[index] = {"x": host.vx, "y": host.vy, "z": host.vz}[axis]
    return rv


def _normalize_rv_semantics(
    raw: dict[str, Any],
    planets: tuple[PlanetParams, ...],
    observations: Observations,
    instruments: tuple[InstrumentParams, ...],
) -> tuple[tuple[PlanetParams, ...], Observations, dict[str, Any]]:
    """Convert legacy REBOUND task records to Stargazer's RV-only semantics."""
    metadata = dict(raw.get("meta") or {})
    if str(metadata.get("rv_semantics", "")).startswith("rv_only"):
        return planets, observations, metadata

    raw_config = raw["config"]
    times = np.asarray(observations.times_days, dtype=float)
    old_clean = _simulate_legacy_rebound_rv(raw_config, planets, times)
    converted_planets = tuple(
        replace(
            planet,
            l_rad=(planet.l_rad - planet.Omega_rad) % (2.0 * np.pi),
            Omega_rad=0.0,
        )
        for planet in planets
    )

    instrument_offsets = {
        instrument.label: instrument.gamma_ms for instrument in instruments
    }
    star_offset = float(raw_config["star"].get("gamma_ms", 0.0))
    gamma_series = np.asarray(
        [
            star_offset + instrument_offsets.get(label, 0.0)
            for label in observations.instruments
        ],
        dtype=float,
    )
    old_model = old_clean + gamma_series
    new_model = (
        simulate_keplerian_rv(
            converted_planets,
            times,
            float(raw_config["star"]["M_star_sun"]),
        )
        + gamma_series
    )
    preserved_noise = np.asarray(observations.rvs_ms, dtype=float) - old_model
    converted_observations = replace(
        observations, rvs_ms=tuple((new_model + preserved_noise).tolist())
    )
    metadata["rv_semantics"] = "rv_only_compat"
    metadata["rv_only_compat_applied"] = True
    return converted_planets, converted_observations, metadata


def _validate_observations(observations: Observations) -> None:
    lengths = {
        len(observations.times_days),
        len(observations.rvs_ms),
        len(observations.sigmas_ms),
        len(observations.instruments),
    }
    if len(lengths) != 1 or not observations.times_days:
        raise ValueError("Observation columns must have equal, non-zero lengths")
    numeric = np.concatenate(
        [
            np.asarray(observations.times_days, dtype=float),
            np.asarray(observations.rvs_ms, dtype=float),
            np.asarray(observations.sigmas_ms, dtype=float),
        ]
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Observations contain non-finite values")
    if np.any(np.asarray(observations.sigmas_ms) <= 0.0):
        raise ValueError("Observation uncertainties must be positive")


@cache
def load_task(path: str | Path, source: str | None = None) -> StargazerTask:
    """Load and normalize one released Stargazer JSON task."""
    task_path = Path(path)
    with task_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    raw_config = raw["config"]
    planet_truth = tuple(
        _planet_from_dict(payload) for payload in raw_config.get("planets", [])
    )
    raw_observations = raw["observations"]
    times = tuple(float(value) for value in raw_observations["times_days"])
    labels = raw_observations.get("instruments") or ["default"] * len(times)
    observations = Observations(
        times_days=times,
        rvs_ms=tuple(float(value) for value in raw_observations["rvs_ms"]),
        sigmas_ms=tuple(float(value) for value in raw_observations["sigmas_ms"]),
        instruments=tuple(str(value) for value in labels),
    )
    _validate_observations(observations)

    raw_instruments = raw_config.get("instruments") or [
        {"label": "default", "gamma_ms": 0.0}
    ]
    instruments = tuple(
        InstrumentParams(
            label=str(payload.get("label", "default")),
            gamma_ms=float(payload.get("gamma_ms", 0.0)),
        )
        for payload in raw_instruments
    )
    inferred_source = source or (
        "real" if str(raw.get("task_id", "")).startswith("real_") else "synthetic"
    )
    planet_truth, observations, metadata = _normalize_rv_semantics(
        raw, planet_truth, observations, instruments
    )

    return StargazerTask(
        task_id=str(raw["task_id"]),
        source=inferred_source,
        truth_difficulty=int(raw.get("truth_difficulty", 10)),
        star_mass_sun=float(raw_config["star"]["M_star_sun"]),
        truth_planets=planet_truth,
        observations=observations,
        instruments=instruments,
        metadata=metadata,
    )
