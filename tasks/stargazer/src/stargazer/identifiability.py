"""Private observation-based bank diagnostics; never imported by analysis tools.

A finite multistart/bootstrap audit can flag ambiguity, not prove uniqueness.
Its settings and acceptance rules are frozen using a separate calibration bank.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle

from stargazer.fit import compute_fit
from stargazer.models import (
    CandidateSubmission,
    PublicFitContext,
    mass_from_semi_amplitude,
    semi_amplitude_ms,
    simulate_keplerian_rv,
)


@dataclass(frozen=True)
class BankRules:
    minimum_cycles: float = 4.0
    minimum_amplitude_sigma: float = 2.0
    maximum_cadence_alias: float = 0.8
    minimum_count_bic_gap: float = 6.0
    minimum_bootstrap_count_fraction: float = 0.75
    maximum_bootstrap_log_period_sd: float = 0.05
    minimum_recovery_match: float = 0.8
    starts: int = 3
    bootstraps: int = 4
    max_planets: int = 4

    def __post_init__(self):
        if self.starts < 2 or self.bootstraps < 2 or not 1 <= self.max_planets <= 7:
            raise ValueError(
                "Audit requires multiple starts and bootstraps and 1-7 planets"
            )


def _candidate(theta: np.ndarray, mass: float) -> CandidateSubmission:
    planets = []
    for log_period, log_amplitude, eccentricity, omega, longitude in theta.reshape(
        -1, 5
    ):
        period, amplitude = np.exp(log_period), np.exp(log_amplitude)
        planets.append(
            {
                "P_days": float(period),
                "m_sin_i_mjup": mass_from_semi_amplitude(
                    float(amplitude), float(period), float(eccentricity), mass
                ),
                "e": float(eccentricity),
                "omega_rad": float(omega),
                "l_rad": float(longitude),
            }
        )
    return CandidateSubmission(planets=planets, noise_jitter_ms=0.1)


def fit_counts(
    context: PublicFitContext,
    *,
    starts: int,
    max_planets: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Search candidate counts with observation-only periodogram initialization."""
    obs = context.observations
    times = np.asarray(obs.times_days)
    t = times - times[0]
    observed = np.asarray(obs.rvs_ms)
    sigma = np.asarray(obs.sigmas_ms)
    labels = np.asarray(obs.instruments)
    weights = 1 / (sigma**2 + 0.01)

    def centered(values):
        out = values.copy()
        for label in np.unique(labels):
            mask = labels == label
            out[mask] -= np.average(out[mask], weights=weights[mask])
        return out

    def residual(theta):
        candidate = _candidate(theta, context.star_mass_sun)
        model = simulate_keplerian_rv(
            tuple(p.to_planet_params() for p in candidate.planets),
            times,
            context.star_mass_sun,
        )
        return centered(observed - model) / np.sqrt(sigma**2 + 0.01)

    span = float(np.ptp(times))
    frequencies = np.linspace(1 / span, 0.5, max(2000, int(span * 8)))
    solutions = []
    theta = np.empty(0)
    for count in range(max_planets + 1):
        if count:
            res = residual(theta) * np.sqrt(sigma**2 + 0.01)
            power = lombscargle(t, res, 2 * np.pi * frequencies, normalize=True)
            frequency = float(frequencies[np.argmax(power)])
            design = np.column_stack(
                (np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t))
            )
            a, b = np.linalg.lstsq(design, res, rcond=None)[0]
            theta = np.append(
                theta,
                [
                    np.log(1 / frequency),
                    np.log(max(0.1, np.hypot(a, b))),
                    0.05,
                    0.0,
                    np.arctan2(-b, a) % (2 * np.pi),
                ],
            )
        lower = np.tile([np.log(2), np.log(0.05), 0.0, -4 * np.pi, -4 * np.pi], count)
        upper = np.tile([np.log(span), np.log(200), 0.8, 4 * np.pi, 4 * np.pi], count)
        fits = []
        for start in range(starts if count else 1):
            trial = theta.copy()
            if start and count:
                trial[0::5] += rng.normal(0, 0.005, count)
                trial[2::5] = rng.uniform(0.01, 0.3, count)
                # Preserve the fundamental phase while exploring eccentric shapes.
                trial[3::5] = rng.uniform(-np.pi, np.pi, count)
            if count:
                opt = least_squares(
                    residual,
                    np.clip(trial, lower + 1e-8, upper - 1e-8),
                    bounds=(lower, upper),
                    max_nfev=180,
                )
                trial = opt.x
            candidate = _candidate(trial, context.star_mass_sun)
            fit = compute_fit(context, candidate)
            fits.append((fit.bic, trial, candidate))
        fits.sort(key=lambda entry: entry[0])
        bic, theta, candidate = fits[0]
        solutions.append(
            {
                "count": count,
                "bic": bic,
                "candidate": candidate.canonical_payload(),
                "start_bics": [entry[0] for entry in fits],
            }
        )
    return solutions


def audit_identifiability(task, rules: BankRules | None = None) -> dict:
    """Check observation geometry, competing fits and residual-bootstrap stability."""
    from stargazer.score import EvaluationCriteria, evaluate_submission

    rules = rules or BankRules()
    context = task.public_fit_context()
    obs = context.observations
    times = np.asarray(obs.times_days)
    sigma = np.median(obs.sigmas_ms)
    span = float(np.ptp(times))
    seed = int.from_bytes(
        hashlib.sha256(np.asarray(obs.rvs_ms).tobytes()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    solutions = fit_counts(
        context, starts=rules.starts, max_planets=rules.max_planets, rng=rng
    )
    ordered = sorted(solutions, key=lambda row: row["bic"])
    best = ordered[0]
    gap = ordered[1]["bic"] - best["bic"]
    reference_count = len(task.truth_planets)
    per_planet = [
        {
            "cycles": span / p.P_days,
            "amplitude_sigma": semi_amplitude_ms(
                p.m_sin_i_mjup, p.P_days, p.e, task.star_mass_sun
            )
            / sigma,
        }
        for p in task.truth_planets
    ]
    # Spectral window above two cycles over the baseline; exclude its central lobe.
    freq = np.linspace(2 / span, 0.5, 2000)
    window = np.abs(
        np.exp(2j * np.pi * freq[:, None] * (times - times[0])).mean(axis=1)
    )
    candidate = CandidateSubmission.model_validate(best["candidate"])
    model = simulate_keplerian_rv(
        tuple(p.to_planet_params() for p in candidate.planets),
        times,
        task.star_mass_sun,
    )
    fitted = compute_fit(context, candidate)
    model += np.asarray(
        [fitted.gamma_per_instrument_ms[label] for label in obs.instruments]
    )
    residual = np.asarray(obs.rvs_ms) - model
    bootstrap = []
    periods = []
    for _ in range(rules.bootstraps):
        # Standardized residuals are resampled within each instrument.
        noise = np.empty_like(residual)
        labels = np.asarray(obs.instruments)
        uncertainties = np.asarray(obs.sigmas_ms)
        for label in np.unique(labels):
            mask = labels == label
            noise[mask] = (
                rng.choice(
                    residual[mask] / uncertainties[mask], mask.sum(), replace=True
                )
                * uncertainties[mask]
            )
        boot = replace(
            context, observations=replace(obs, rvs_ms=tuple((model + noise).tolist()))
        )
        fits = fit_counts(
            boot, starts=rules.starts, max_planets=rules.max_planets, rng=rng
        )
        selected = min(fits, key=lambda row: row["bic"])
        bootstrap.append(selected["count"])
        if selected["count"] == best["count"]:
            periods.append(
                sorted(np.log(p["P_days"]) for p in selected["candidate"]["planets"])
            )
    period_sd = (
        float(np.max(np.std(periods, axis=0))) if periods and best["count"] else None
    )
    count_fraction = float(np.mean(np.asarray(bootstrap) == best["count"]))
    recovery = evaluate_submission(
        task,
        best["candidate"],
        EvaluationCriteria(minimum_planet_score=rules.minimum_recovery_match),
    )
    checks = {
        "cycles": all(p["cycles"] >= rules.minimum_cycles for p in per_planet),
        "amplitude": all(
            p["amplitude_sigma"] >= rules.minimum_amplitude_sigma for p in per_planet
        ),
        "cadence": float(window.max()) <= rules.maximum_cadence_alias,
        "count_separation": gap >= rules.minimum_count_bic_gap,
        "count_recovery": best["count"] == reference_count,
        "parameter_recovery": recovery.success,
        "bootstrap_count": count_fraction >= rules.minimum_bootstrap_count_fraction,
        "bootstrap_period": period_sd is not None
        and period_sd <= rules.maximum_bootstrap_log_period_sd,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "rules": asdict(rules),
        "per_planet": per_planet,
        "cadence_alias": float(window.max()),
        "competing_fits": solutions,
        "count_bic_gap": gap,
        "bootstrap_counts": bootstrap,
        "bootstrap_count_fraction": count_fraction,
        "bootstrap_log_period_sd": period_sd,
        "recovered_match_score": recovery.match_score,
    }
