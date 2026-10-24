"""Level 1 Task 6: assess the recorded temperature and its settling."""

from __future__ import annotations

import numpy as np
from ase import units
from scipy.integrate import cumulative_trapezoid

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric, aliased_value, finite_array
from .level2_quantities import kind, log_rows

VERSION = "level1-task6-temperature-v1"


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _number(value):
    _require(
        type(value) in (int, float) and np.isfinite(value), "Expected a finite number"
    )
    return float(value)


def temperature_statistics(time, temperature):
    """Apply the existing sampling and terminal temperature checks only."""
    _require(
        len(time) >= 32 and time[-1] - time[0] >= 2000 and np.max(np.diff(time)) <= 100,
        "Retain at least 2 ps of equilibration, sampled at most every 100 fs",
    )
    tail = temperature[len(temperature) // 2 :]
    blocks = np.array([part.mean() for part in np.array_split(tail, 4)])
    mean = float(tail.mean())
    se = float(blocks.std(ddof=1) / 2)
    _require(
        abs(mean - 300) <= 30
        and se <= 10
        and np.ptp(blocks) <= 45
        and abs(tail[-1] - mean) <= 90,
        "Equilibration temperature has not settled near 300 K",
    )
    return {
        "equilibration_temperature_K": mean,
        "equilibration_temperature_se_K": se,
    }


def spectral_summaries(frequency, power):
    """Summarize a VDOS using trapezoidal moments and its cumulative area."""
    frequency = finite_array(frequency, ndim=1)
    power = finite_array(power, shape=frequency.shape)
    _require(
        len(frequency) >= 3
        and frequency[0] == 0
        and np.all(np.diff(frequency) > 0)
        and np.all(power >= 0),
        "Expected a nonnegative spectrum on increasing frequencies from zero",
    )
    area = float(np.trapezoid(power, frequency))
    _require(np.isfinite(area) and area > 0, "No resolved velocity spectrum")
    density = power / area
    mean = float(np.trapezoid(frequency * density, frequency))
    variance = float(np.trapezoid((frequency - mean) ** 2 * density, frequency))
    cumulative = cumulative_trapezoid(density, frequency, initial=0)
    # Take the first crossing, including when a zero-density plateau starts at
    # exactly 95%, and interpolate linearly between cumulative-area samples.
    index = int(np.searchsorted(cumulative, 0.95, side="left"))
    fraction = (0.95 - cumulative[index - 1]) / (
        cumulative[index] - cumulative[index - 1]
    )
    percentile = frequency[index - 1] + fraction * (
        frequency[index] - frequency[index - 1]
    )
    return {
        "mean_frequency_THz": mean,
        "std_frequency_THz": float(np.sqrt(variance)),
        "frequency_95pct_THz": float(percentile),
    }


def sample_time(row):
    fs = aliased_value(row, "time_fs", "elapsed_time_fs", required=False)
    ps = aliased_value(row, "time_ps", "elapsed_time_ps", required=False)
    if fs is None and ps is None:
        raise EvidenceError("Temperature samples need elapsed times in fs or ps")
    if ps is not None:
        converted = 1000 * _number(ps)
        if fs is not None:
            _require(
                np.isclose(_number(fs), converted, rtol=1e-7, atol=1e-6),
                "Conflicting sample times in fs and ps",
            )
        return converted
    return _number(fs)


def sample_temperature(row):
    temperature = aliased_value(
        row, "temperature_K", "T_K", "Temperature", "T", required=False
    )
    if temperature is not None:
        return _number(temperature)
    kinetic = aliased_value(row, "kinetic_energy_eV", "ekin_eV", required=False)
    if kinetic is None:
        momenta = aliased_value(row, "momenta", "momenta_ase_units", required=False)
        if momenta is None:
            raise EvidenceError("Samples need temperature, kinetic energy or momenta")
        momenta = finite_array(momenta, shape=(64, 3))
        masses = aliased_value(row, "masses_amu", "masses", required=False)
        masses = finite_array(
            [26.9815385] * 64 if masses is None else masses, shape=(64,)
        )
        _require(np.all(masses > 0), "Atomic masses must be positive")
        kinetic = float(np.sum(momenta**2 / masses[:, None]) / 2)
    dof = row.get("dof", 192)
    _require(
        type(dof) is int and dof in (189, 192), "Invalid kinetic degrees of freedom"
    )
    return 2 * _number(kinetic) / (dof * units.kB)


def temperature_samples(path):
    _, rows = log_rows(path)
    samples = []
    for row in rows:
        labels = {
            str(row.get(k, "")).lower() for k in ("phase", "part", "name", "stage")
        }
        if kind(row) in {"summary", "vdos", "complete"} or labels & {
            "production",
            "prod",
            "nve",
        }:
            continue
        samples.append((sample_time(row), sample_temperature(row)))
    _require(bool(samples), "No equilibration temperature samples")
    values = finite_array(samples, ndim=2)
    time, temperature = values.T
    _require(
        time[0] >= 0 and np.all(np.diff(time) > 0),
        "Temperature sample times must be nonnegative and strictly increasing",
    )
    _require(np.all(temperature > 0), "Recorded temperatures must be positive")
    return time, temperature


class Task6Scorer:
    """Grade Level 1 temperature samples without a simulation verifier."""

    task_number = 6
    level = 1
    verifier = None

    def __call__(self, submission):
        return self.score_submission(submission)

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        failures = [
            f"{c['name']}: {c['detail']}"
            for c in report["checks"]
            if c["status"] == "failed"
        ]
        return SubmissionScore(
            score=report["score"],
            feedback="; ".join(failures) or None,
            metadata={"workflow_evaluation": report},
        )

    def evaluate(self, submission):
        rubric, computed = Rubric(6, binary=True), {}
        evidence = None

        def check():
            nonlocal evidence
            evidence = Evidence(submission, task_number=6, level=1)
            time, temperature = temperature_samples(evidence.artifact("run_log"))
            computed.update(temperature_statistics(time, temperature))
            name = "equilibration_temperature_K"
            if name in evidence.results:
                reported = _number(evidence.results[name])
                _require(
                    abs(reported - computed[name]) <= 1.0 + 0.01 * abs(computed[name]),
                    f"{name}: reported {reported:g}, recomputed {computed[name]:g}",
                )
            return True

        rubric.check("equilibration_temperature", 100, check)
        return {
            **rubric.as_dict(),
            "level": 1,
            "mode": "logged_temperature",
            "scoring_policy": VERSION,
            "computed": computed,
            "evidence_sha256": evidence.fingerprint() if evidence else None,
        }
