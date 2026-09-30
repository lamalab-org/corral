"""Data model and radial-velocity forward models for Stargazer.

Tasks come from the Stargazer paper's benchmark. The released synthetic bank
was generated with REBOUND. Stargazer's current evaluator uses non-interacting,
radial-velocity-only Keplerians, so legacy synthetic tasks are converted on load
while preserving their original noise realisation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

# Compatibility imports; public numerics never import this private loader.
from stargazer.public_rv import (  # noqa: F401
    DAY_SECONDS,
    M_JUPITER_KG,
    M_SUN_KG,
    CandidatePlanet,
    CandidateSubmission,
    Observations,
    PlanetParams,
    PublicFitContext,
    _simulate_legacy_rebound_rv,
    _solve_kepler,
    _wrap_angle,
    mass_from_semi_amplitude,
    semi_amplitude_ms,
    simulate_keplerian_rv,
    simulate_submission_rv,
)


@dataclass(frozen=True)
class InstrumentParams:
    """Instrument metadata needed to reconstruct legacy observations."""

    label: str
    gamma_ms: float = 0.0


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
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def max_planets(self) -> int:
        """Published submission limit for this task family."""
        return 7

    def public_fit_context(
        self, *, maximum_rms_factor: float = 1.5
    ) -> PublicFitContext:
        """Copy only approved public conventions; never retain the task."""
        axis = self.config.get("los_axis", "x")
        integrator = self.config.get("integrator_preference", "whfast")
        return PublicFitContext(
            observations=self.observations,
            star_mass_sun=self.star_mass_sun,
            maximum_rms_factor=maximum_rms_factor,
            los_axis=axis if axis in ("x", "y", "z") else "x",
            integrator_preference="ias15" if integrator == "ias15" else "whfast",
        )

    def public_observation(self) -> dict[str, Any]:
        """The original observation envelope, with evaluator-only truth excluded."""
        meta = {
            "time_unit": "day",
            "rv_unit": "m/s",
            "instrument_labels": [instrument.label for instrument in self.instruments],
            "star_mass_sun": self.star_mass_sun,
            "los_axis": self.config.get("los_axis", "x"),
            "integrator_preference": self.config.get("integrator_preference", "whfast"),
            "engine": self.config.get("engine", "rebound"),
        }
        return {
            "times_days": self.observations.times_days,
            "rvs_ms": self.observations.rvs_ms,
            "sigmas_ms": self.observations.sigmas_ms,
            "instruments": self.observations.instruments,
            "meta": meta,
        }

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


def _normalize_rv_semantics(
    raw: dict[str, Any],
    planets: tuple[PlanetParams, ...],
    observations: Observations,
    instruments: tuple[InstrumentParams, ...],
) -> tuple[tuple[PlanetParams, ...], Observations, dict[str, Any]]:
    """Convert legacy REBOUND task records to Stargazer's RV-only semantics."""
    metadata = dict(raw.get("meta") or {})
    if str(metadata.get("rv_semantics", "")).startswith("rv_only") or metadata.get(
        "rv_only_compat_applied"
    ):
        return planets, observations, metadata

    raw_config = raw["config"]
    times = np.asarray(observations.times_days, dtype=float)
    try:
        old_clean = _simulate_legacy_rebound_rv(raw_config, planets, times)
    except Exception:
        return planets, observations, metadata
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
        config=raw_config,
    )
