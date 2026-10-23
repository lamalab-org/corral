"""Score aluminum preparation and VDOS from one final calculation log.

No submitted code or models are executed. Numerical screens are explicit
evaluator policy awaiting calibration across validated MACE trajectories.
Recorded model identity and continuity are consistency checks, not attestation.
"""

from __future__ import annotations

import json
import secrets

import numpy as np
from ase import Atoms, units
from scipy import signal
from scipy.integrate import cumulative_trapezoid

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric, UnsupportedEvidence, finite_array
from .level1_trusted import PINNED_TEACHER_SHA256
from .sampling import mean_standard_error
from .verification import Plan, geometry


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _number(value):
    _require(
        type(value) in (int, float) and np.isfinite(value), "Expected a finite number"
    )
    return float(value)


def read_log(evidence, level):
    """Read completed stages in execution order, retaining restartable samples."""
    stages = []
    current = None
    with evidence.artifact("run_log").open() as stream:
        header = json.loads(next(stream))
        _require(header["event"] == "start", "Missing run header")
        _require(
            header["model_sha256"] == PINNED_TEACHER_SHA256,
            "Use the supplied MACE-MP-0 model",
        )
        _require(type(header["random_seed"]) is int, "Record the initialization seed")
        _require(
            header["units"]
            == {"time": "fs", "position": "Angstrom", "energy": "eV", "momenta": "ASE"},
            "Incorrect log units",
        )
        _require(header["numbers"] == [13] * 64, "Expected 64 aluminum atoms")
        masses = finite_array(header["masses_amu"], shape=(64,))
        _require(
            np.allclose(masses, 26.9815385, atol=1e-3, rtol=0),
            "Incorrect aluminum masses",
        )
        for line in stream:
            record = json.loads(line)
            event = record["event"]
            if event == "stage":
                expected = "equilibration" if not stages else "production"
                _require(
                    current is None
                    and len(stages) < level
                    and record["name"] == expected,
                    "Incorrect stage order",
                )
                _require(
                    all(
                        type(record[k]) is int and record[k] > 0
                        for k in ("steps", "interval")
                    ),
                    "Invalid stage sampling schedule",
                )
                _require(
                    record["steps"] % record["interval"] == 0,
                    "Retain both stage endpoints",
                )
                dt = _number(record["timestep_fs"])
                _require(0 < dt <= 2, "Use an integration timestep of at most 2 fs")
                if expected == "production":
                    _require(
                        record["integrator"] == "VelocityVerlet"
                        and record["target_temperature_K"] is None,
                        "Production must use unthermostatted NVE dynamics",
                    )
                else:
                    _require(
                        record["integrator"]
                        in {
                            "Langevin",
                            "NVTBerendsen",
                            "NoseHooverChainNVT",
                            "Bussi",
                            "Andersen",
                        }
                        and record["target_temperature_K"] == 300,
                        "Equilibrate with a 300 K thermostat",
                    )
                current = {**record, "samples": []}
            elif event == "sample":
                _require(current is not None, "Sample outside a stage")
                current["samples"].append(record)
            elif event == "end":
                _require(
                    current is not None
                    and record["name"] == current["name"]
                    and record["steps"] == current["steps"],
                    "Incomplete stage",
                )
                stages.append(current)
                current = None
            else:
                raise EvidenceError(f"Unexpected log event: {event}")
    _require(
        current is None and len(stages) == level,
        "Log does not contain every completed stage",
    )
    offset = 0.0
    initial_cell = None
    for stage in stages:
        rows = stage.pop("samples")
        _require(
            len(rows) == stage["steps"] // stage["interval"] + 1 and len(rows) >= 32,
            "Insufficient or missing samples",
        )
        steps = np.arange(len(rows)) * stage["interval"]
        _require(
            all(type(row["step"]) is int for row in rows),
            "Sample steps must be integers",
        )
        _require(
            np.array_equal([row["step"] for row in rows], steps),
            "Irregular or truncated sample steps",
        )
        time = finite_array([row["time_fs"] for row in rows], shape=(len(rows),))
        _require(
            np.allclose(time, offset + steps * stage["timestep_fs"], rtol=0, atol=1e-6),
            "Incorrect or nonuniform saved times",
        )
        cell = finite_array([row["cell_A"] for row in rows], shape=(len(rows), 3, 3))
        if initial_cell is None:
            initial_cell = cell[0]
        _require(
            np.allclose(cell, initial_cell, rtol=0, atol=1e-7),
            "Cell changed during the calculation",
        )
        position = finite_array(
            [row["positions_A"] for row in rows], shape=(len(rows), 64, 3)
        )
        momenta = []
        for row in rows:
            p = row.get("momenta")
            if "velocities_A_fs" in row:
                from_velocity = (
                    finite_array(row["velocities_A_fs"], shape=(64, 3))
                    * masses[:, None]
                    / units.fs
                )
                _require(
                    p is None
                    or np.allclose(
                        finite_array(p, shape=(64, 3)),
                        from_velocity,
                        rtol=1e-6,
                        atol=1e-8,
                    ),
                    "Contradictory velocities and momenta",
                )
                p = from_velocity
            momenta.append(finite_array(p, shape=(64, 3)))
        momenta = np.array(momenta)
        kinetic = (momenta**2 / masses[None, :, None]).sum(axis=(1, 2)) / 2
        potential = finite_array(
            [row["potential_energy_eV"] for row in rows], shape=(len(rows),)
        )
        stage.update(
            time=time,
            positions=position,
            momenta=momenta,
            kinetic=kinetic,
            masses=masses,
            potential=potential,
            directional=(momenta**2 / masses[None, :, None]).sum(axis=1)
            / (64 * units.kB),
            temperature=2 * kinetic / (192 * units.kB),
            energy=kinetic + potential,
            velocity=momenta / masses[None, :, None],
            cell=initial_cell,
        )
        offset = time[-1]
    first = stages[0]
    atoms = Atoms(
        numbers=header["numbers"],
        positions=first["positions"][0],
        cell=initial_cell,
        pbc=True,
    )
    expected = np.full((3, 3), (2 * 4.05) ** 2)
    np.fill_diagonal(expected, 2 * (2 * 4.05) ** 2)
    _require(
        np.allclose(np.abs(initial_cell @ initial_cell.T), expected, atol=1e-4, rtol=0)
        and np.allclose(
            np.sort(atoms.get_all_distances(mic=True), axis=1)[:, 1:13],
            4.05 / np.sqrt(2),
            atol=1e-4,
            rtol=0,
        ),
        "Incorrect initial primitive FCC geometry",
    )
    _require(
        np.allclose(first["momenta"][0].sum(axis=0), 0, atol=1e-7)
        and 150 < first["temperature"][0] < 450,
        "Invalid 300 K initialization or center-of-mass momentum",
    )
    if level == 2:
        production = stages[1]
        delta = (production["positions"][0] - first["positions"][-1]) @ np.linalg.inv(
            initial_cell
        )
        _require(
            np.allclose(delta - np.rint(delta), 0, atol=1e-7)
            and np.allclose(
                production["momenta"][0], first["momenta"][-1], atol=1e-7, rtol=0
            )
            and np.isclose(
                production["energy"][0], first["energy"][-1], atol=1e-6, rtol=0
            ),
            "Discontinuous equilibration-to-NVE state",
        )
    return stages


def equilibration_statistics(stage):
    time, temperature = stage["time"], stage["temperature"]
    _require(
        time[-1] - time[0] >= 2000 and np.max(np.diff(time)) <= 100,
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
    _require(
        np.max(np.ptp(stage["positions"], axis=0)) > 1e-4, "No recorded atomic motion"
    )
    directional = stage["directional"][len(temperature) // 2 :]
    directional_mean = directional.mean(axis=0)
    directional_se = np.array([mean_standard_error(v) for v in directional.T])
    _require(
        np.max(np.abs(directional_mean - 300)) <= 60
        and np.max(directional_se) <= 15
        and np.max(
            np.ptp(
                [part.mean(axis=0) for part in np.array_split(directional, 4)], axis=0
            )
        )
        <= 90,
        "Equilibration directional temperatures have not settled near 300 K",
    )
    diagnostics = {
        "equilibration_temperature_K": mean,
        "equilibration_temperature_se_K": se,
        "equilibration_directional_temperature_K": directional_mean.tolist(),
        "equilibration_directional_temperature_se_K": directional_se.tolist(),
    }
    tail_time = time[len(temperature) // 2 :] / 1000
    for name, values in (("potential", stage["potential"]), ("total", stage["energy"])):
        energy = values[len(temperature) // 2 :] / 64
        error = mean_standard_error(energy)
        drift = float(np.polyfit(tail_time - tail_time.mean(), energy, 1)[0])
        blocks = np.array([part.mean() for part in np.array_split(energy, 4)])
        _require(
            error <= 0.15 * units.kB * 300
            and np.ptp(blocks) <= 0.6 * units.kB * 300
            and abs(drift) * np.ptp(tail_time) <= 0.6 * units.kB * 300,
            f"Equilibration {name} energy is drifting or insufficiently sampled",
        )
        _require(
            abs(energy[len(energy) // 2 :].mean() - energy.mean())
            <= 0.3 * units.kB * 300,
            f"Equilibration {name} energy changes across analysis windows",
        )
        diagnostics[f"equilibration_{name}_energy_eV_per_atom"] = float(energy.mean())
        diagnostics[f"equilibration_{name}_energy_se_eV_per_atom"] = error
        diagnostics[f"equilibration_{name}_energy_drift_eV_atom_ps"] = drift
    final = Atoms(
        "Al64", positions=stage["positions"][-1], cell=stage["cell"], pbc=True
    )
    distances = final.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    _require(distances.min() > 1.0, "Final aluminum state has overlapping atoms")
    return diagnostics


def _psd(velocity, dt, segment):
    # Independent Hann-Welch estimate, summing all atom/component spectra.
    frequency, power = signal.welch(
        velocity,
        fs=1000 / dt,
        window=signal.windows.hann(segment, sym=False),
        nperseg=segment,
        noverlap=segment // 2,
        nfft=segment,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=0,
    )
    power = power.sum(axis=(1, 2))
    area = np.trapezoid(power, frequency)
    _require(area > 0 and np.isfinite(area), "No resolved velocity spectrum")
    return frequency, power / area


def _bands(frequency, power, edges):
    integral = cumulative_trapezoid(power, frequency, initial=0)
    return np.diff(np.interp(edges, frequency, integral))


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


def production_statistics(stage):
    time = stage["time"]
    # read_log already checks every sample time against this recorded schedule.
    # Avoid cancellation in large absolute times changing ceil(4000 / dt).
    dt = float(stage["timestep_fs"] * stage["interval"])
    _require(
        dt <= 10, "Sample production at most every 10 fs (Nyquist at least 50 THz)"
    )
    temperature = stage["temperature"]
    blocks = np.array([part.mean() for part in np.array_split(temperature, 4)])
    _require(
        abs(temperature.mean() - 300) <= 45 and np.ptp(blocks) <= 30,
        "Unstable production temperature or not near 300 K",
    )
    energy = stage["energy"] / 64
    drift = float(np.polyfit((time - time[0]) / 1000, energy, 1)[0])
    span = float(np.ptp(energy))
    budget = 0.02 * float(stage["kinetic"].mean()) / 64
    _require(
        span <= budget and abs(drift) * np.ptp(time) / 1000 <= budget,
        "NVE energy is not conserved within 2% of mean kinetic energy",
    )
    # At <=10 fs spacing, displacement and trapezoidal velocity must agree.
    delta = np.diff(stage["positions"], axis=0) @ np.linalg.inv(stage["cell"])
    delta = (delta - np.rint(delta)) @ stage["cell"]
    predicted = (stage["velocity"][1:] + stage["velocity"][:-1]) * (dt * units.fs / 2)
    _require(
        np.max(np.abs(delta - predicted)) <= 0.03,
        "Motion is inconsistent with saved velocities or includes a reset",
    )
    com = stage["momenta"].sum(axis=1)
    _require(
        np.max(np.abs(com - com[0])) <= 1e-4, "NVE center-of-mass momentum changed"
    )
    count = len(time)
    # Fix the estimator independently of agent claims: >=4 ps periodic-Hann
    # segments give effective resolution <=0.5 THz at every allowed sample rate.
    reference_segment = int(np.ceil(4000 / dt))
    resolution = 2000 / (dt * reference_segment)
    _require(
        count // 4 >= reference_segment,
        "Trajectory is too short for four blocks of at least "
        "ceil(4000 / sampling_interval_fs) samples",
    )
    velocity = stage["velocity"] - stage["velocity"].mean(axis=1, keepdims=True)
    frequency, power = _psd(velocity, dt, reference_segment)
    _require(
        np.trapezoid(
            power[frequency >= 0.8 * frequency[-1]],
            frequency[frequency >= 0.8 * frequency[-1]],
        )
        <= 0.01,
        "Significant spectral weight approaches Nyquist; sample more frequently",
    )
    edges = np.r_[np.arange(0, frequency[-1], 2 * resolution), frequency[-1]]
    reference = _bands(frequency, power, edges)
    distances = []
    for block in np.array_split(velocity, 4):
        f, p = _psd(block, dt, reference_segment)
        distances.append(float(np.abs(_bands(f, p, edges) - reference).sum()))
    _require(
        max(distances) <= 0.25,
        "Spectral features are not stable across trajectory blocks "
        "at the fixed resolution",
    )
    return {
        "production_temperature_K": float(temperature.mean()),
        "sampling_interval_fs": dt,
        "frequency_resolution_THz": resolution,
        "normalization_area": float(np.trapezoid(power, frequency)),
        "energy_drift_eV_atom_ps": drift,
        "energy_span_eV_atom": span,
        **spectral_summaries(frequency, power),
        "block_band_l1_max": max(distances),
    }


class Task6Scorer:
    """Binary scoring of the final saved calculation and the agent's answers."""

    task_number = 6

    def __init__(self, *, level=2, verifier=None):
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.level = level
        self.verifier = verifier

    def __call__(self, submission):
        return self.score_submission(submission)

    def score_submission(self, submission):
        return self.evaluate_submission(submission).score

    def evaluate_submission(self, submission):
        report = self.evaluate(submission)
        if report["score"] is None:
            from .evaluator import PendingReviewError

            raise PendingReviewError(report)
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
        rubric = Rubric(6, binary=True, fail_fast=True)
        computed = {}
        stages, verification = [], {}
        try:
            evidence = Evidence(submission)
            fingerprint = evidence.fingerprint()
        except Exception as exc:
            rubric.check("recorded_aluminum_dynamics", 100, False, str(exc))
            return {**rubric.as_dict(), "level": self.level, "computed": computed}

        def check():
            stages.extend(read_log(evidence, self.level))
            computed.update(equilibration_statistics(stages[0]))
            tolerances = {"equilibration_temperature_K": 1.0}
            required = ()
            if self.level == 2:
                computed.update(production_statistics(stages[1]))
                required = (
                    "mean_frequency_THz",
                    "std_frequency_THz",
                )
                tolerances.update(dict.fromkeys(required, 0.01))
                tolerances.update(
                    frequency_95pct_THz=0.01,
                    production_temperature_K=1.0,
                    sampling_interval_fs=1e-6,
                    frequency_resolution_THz=1e-6,
                    energy_drift_eV_atom_ps=1e-7,
                    energy_span_eV_atom=1e-6,
                )
            for key, atol in tolerances.items():
                # Require the scientific answers; reconstructed thermal and
                # sampling diagnostics remain optional at both levels.
                if key not in required and key not in evidence.results:
                    continue
                actual = _number(evidence.results[key])
                _require(
                    abs(actual - computed[key]) <= atol + 0.01 * abs(computed[key]),
                    f"{key}: reported {actual:g}, recomputed {computed[key]:g}",
                )
            for key, value in computed.items():
                if (
                    not key.startswith("equilibration_")
                    or key in tolerances
                    or key not in evidence.results
                ):
                    continue
                actual = finite_array(evidence.results[key])
                expected = finite_array(value)
                _require(
                    actual.shape == expected.shape
                    and np.allclose(actual, expected, rtol=0.01, atol=1e-6),
                    f"{key}: reported value disagrees with recorded equilibration",
                )
            return True

        rubric.check("recorded_aluminum_dynamics", 80, check)

        def independent():
            if self.verifier is None:
                raise UnsupportedEvidence(
                    "Independent energy checks require a local or Modal backend"
                )
            plan = Plan(evidence, 6, secrets.token_hex(32))
            for stage in stages:
                n = len(stage["time"])
                indices = sorted({0, n - 1, *plan.sample(n, 3, stage["name"])})
                plan.add(
                    stage["name"],
                    "mace",
                    {
                        "frames": [
                            geometry(
                                Atoms(
                                    "Al64",
                                    positions=stage["positions"][i],
                                    cell=stage["cell"],
                                    pbc=True,
                                    masses=stage["masses"],
                                )
                            )
                            for i in indices
                        ],
                        "properties": ["energy"],
                        "sample_indices": indices,
                    },
                    {
                        "frames": [
                            {"energy": float(stage["potential"][i])} for i in indices
                        ]
                    },
                    parameters={
                        "model": "teacher.model",
                        "default_dtype": "float64",
                        "dispersion": False,
                    },
                )
            try:
                checks, _, _ = self.verifier._evaluate_plan(plan, fingerprint)
            except Exception as exc:
                raise UnsupportedEvidence(
                    f"Independent energy verification unavailable: {exc}"
                ) from exc
            verification["checks"] = checks
            _require(
                all(c["status"] != "failed" for c in checks),
                "Independent model energies disagree with the log",
            )
            if len(checks) != len(plan.jobs) or any(
                c["status"] != "passed" for c in checks
            ):
                raise UnsupportedEvidence(
                    "Independent energy verification is incomplete"
                )
            return True

        rubric.check("independent_model_calculation", 20, independent)
        if evidence.fingerprint() != fingerprint:
            raise RuntimeError(
                "Submission changed during grading; retry with frozen logs"
            )
        return {
            **rubric.as_dict(),
            "level": self.level,
            "mode": "saved_logs",
            "computed": computed,
            "verification": verification,
            "evidence_sha256": fingerprint,
        }
