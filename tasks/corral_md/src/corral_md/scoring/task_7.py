"""Aluminum expansion from one saved thermal log and a small answer object.

No submitted code is executed or dynamics rerun. Model hashes and boundary
states check consistency, not execution attestation. Sampling/locality screens
are explicit evaluator policy and need calibration against validated MACE runs.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units

from corral.evaluation import SubmissionScore

from .common import Evidence, EvidenceError, Rubric, finite_array
from .formats import artifact_reader, check_model_digest, check_units, md_records
from .level1_trusted import PINNED_TEACHER_SHA256


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _number(value):
    _require(
        type(value) in (int, float) and np.isfinite(value), "Expected a finite number"
    )
    return float(value)


def _state(record):
    return (
        finite_array(record["cell_A"], shape=(3, 3)),
        finite_array(record["positions_A"], shape=(108, 3)),
        finite_array(record["momenta"], shape=(108, 3)),
    )


def _same_state(first, second):
    cell, position, momentum = first
    delta = (second[1] - position) @ np.linalg.inv(cell)
    return (
        np.allclose(cell, second[0], atol=1e-7, rtol=0)
        and np.allclose(delta - np.rint(delta), 0, atol=1e-7, rtol=0)
        and np.allclose(momentum, second[2], atol=1e-7, rtol=0)
    )


@artifact_reader
def read_log(evidence, level):
    """Require complete chronological stages, raw pressure, and NPT continuity."""
    stages, current = [], None
    records = md_records(evidence.artifact("run_log"), 7)
    header = records[0]
    check_model_digest(header, PINNED_TEACHER_SHA256)
    _require(type(header["random_seed"]) is int, "Record the initialization seed")
    _require(header["numbers"] == [13] * 108, "Expected Al108")
    masses = finite_array(header["masses_amu"], shape=(108,))
    _require(
        np.allclose(masses, 26.9815385, atol=1e-3, rtol=0),
        "Incorrect aluminum masses",
    )
    check_units(
        header,
        {
            "time": "fs",
            "cell": "Angstrom",
            "energy": "eV",
            "pressure": "GPa",
            "momenta": "ASE",
        },
    )
    for row in records[1:]:
        if row["event"] == "stage":
            _require(current is None, "Unfinished stage")
            current = {**row, "samples": []}
        elif row["event"] == "sample":
            _require(current is not None, "Sample outside a stage")
            current["samples"].append(row)
        elif row["event"] == "end":
            _require(current is not None, "End outside a stage")
            current["end"] = row
            stages.append(current)
            current = None
        else:
            raise EvidenceError("Unexpected log event")
    _require(
        current is None and stages and (level == 2 or len(stages) == 1),
        "Missing completed stages or extra Level 1 stages",
    )
    offset, previous = 0.0, None
    for stage in stages:
        ensemble = stage["ensemble"]
        allowed = (
            {"IsotropicMTKNPT", "NPTBerendsen"}
            if ensemble == "NPT"
            else {"Langevin", "NVTBerendsen", "NoseHooverChainNVT", "Bussi", "Andersen"}
        )
        _require(
            ensemble in {"NPT", "NVT"} and stage["integrator"] in allowed,
            "Incorrect ensemble or integrator",
        )
        # ASE thermostats may serialize 400 K as 399.99999999999994 K.
        target = stage["target_temperature_K"] = round(
            _number(stage["target_temperature_K"]), 6
        )
        _require(target > 0, "Invalid temperature target")
        _require(
            stage["branch"]
            in ({"heating", "cooling"} if ensemble == "NPT" else {"pressure"}),
            "Incorrect stage branch",
        )
        if ensemble == "NPT":
            _require(
                abs(_number(stage["target_pressure_GPa"]) - 1.01325e-4) <= 1e-9,
                "NPT target must be 1.01325 bar",
            )
        else:
            _require(stage["target_pressure_GPa"] is None, "NVT must have no barostat")
        eq, prod, interval = [
            stage[k] for k in ("equilibration_steps", "production_steps", "interval")
        ]
        _require(
            all(type(v) is int and v > 0 for v in (eq, prod, interval))
            and eq % interval == 0
            and prod % interval == 0,
            "Invalid sampling schedule",
        )
        dt = _number(stage["timestep_fs"])
        _require(
            0 < dt <= 2
            and eq * dt >= 2000
            and prod * dt >= 3000
            and interval * dt <= 100
            and prod // interval >= 32,
            "Use <=2 fs steps, >=2 ps equilibration, >=3 ps production and >=32 samples at <=100 fs",
        )
        rows = stage.pop("samples")
        steps = np.arange((eq + prod) // interval + 1) * interval
        _require(
            len(rows) == len(steps)
            and all(type(r["step"]) is int for r in rows)
            and np.array_equal([r["step"] for r in rows], steps),
            "Missing, duplicate, or unordered samples",
        )
        times = finite_array([r["time_fs"] for r in rows])
        _require(
            np.allclose(times, offset + steps * dt, atol=1e-6, rtol=0)
            and stage["end"]["step"] == eq + prod
            and abs(_number(stage["end"]["time_fs"]) - times[-1]) < 1e-6,
            "Truncated or inconsistent stage times",
        )
        cell = finite_array([r["cell_A"] for r in rows], shape=(len(rows), 3, 3))
        volume = np.linalg.det(cell)
        _require(np.all(volume > 0), "Nonpositive volume")
        scale = (
            (volume / volume[0]) ** (1 / 3) if ensemble == "NPT" else np.ones(len(rows))
        )
        _require(
            np.allclose(cell, cell[0] * scale[:, None, None], atol=1e-6, rtol=0),
            "NPT must be isotropic; NVT must keep its cell fixed",
        )
        kinetic = finite_array([r["kinetic_energy_eV"] for r in rows])
        temperature = finite_array([r["temperature_K"] for r in rows])
        pressure = finite_array([r["pressure_GPa"] for r in rows])
        potential = finite_array([r["potential_pressure_GPa"] for r in rows])
        _require(
            np.all(kinetic > 0)
            and np.allclose(
                temperature, 2 * kinetic / (324 * units.kB), atol=1e-6, rtol=1e-6
            ),
            "Temperature disagrees with kinetic energy",
        )
        _require(
            np.allclose(
                pressure,
                potential + 2 * kinetic / (3 * volume * units.GPa),
                atol=1e-7,
                rtol=1e-6,
            ),
            "Total pressure must include its kinetic contribution in GPa",
        )
        start, end = _state(stage["state"]), _state(stage["end"]["state"])
        for state, index in ((start, 0), (end, -1)):
            _require(
                np.allclose(state[0], cell[index], atol=1e-7, rtol=0)
                and np.isclose(
                    (state[2] ** 2 / masses[:, None]).sum() / 2,
                    kinetic[index],
                    atol=1e-6,
                    rtol=1e-6,
                ),
                "Boundary state disagrees with thermal samples",
            )
        _require(not _same_state(start, end), "No recorded atomic evolution")
        if ensemble == "NPT" and previous is not None:
            _require(
                _same_state(previous, start),
                "NPT positions, cell, or momenta were reset",
            )
        previous = end
        offset = times[-1]
        stage["data"] = np.column_stack([temperature, volume, pressure])[steps > eq]
    first = _state(stages[0]["state"])
    atoms = Atoms("Al108", positions=first[1], cell=first[0], pbc=True)
    distances = np.sort(atoms.get_all_distances(mic=True), axis=1)
    _require(
        np.allclose(
            first[0] @ first[0].T, np.eye(3) * (3 * 4.05) ** 2, atol=1e-5, rtol=0
        )
        and np.allclose(distances[:, 1:13], 4.05 / np.sqrt(2), atol=1e-5, rtol=0)
        and np.allclose(first[2].sum(axis=0), 0, atol=1e-7, rtol=0),
        "Incorrect initial FCC geometry or center-of-mass momentum",
    )
    branches = [s["branch"] for s in stages]
    targets = [s["target_temperature_K"] for s in stages]
    _require(
        branches[0] == "heating" and targets[0] == 300,
        "Start with the 300 K NPT reference",
    )
    if level == 2:
        cooling, pressure = branches.index("cooling"), branches.index("pressure")
        heating_t, cooling_t = targets[:cooling], targets[cooling:pressure]
        _require(
            branches
            == ["heating"] * cooling
            + ["cooling"] * (pressure - cooling)
            + ["pressure"] * (len(stages) - pressure)
            and {300, 400, 500} <= set(heating_t)
            and {300, 400} <= set(cooling_t)
            and heating_t[-1] == 500
            and cooling_t[-1] == 300
            and np.all(np.diff(heating_t) > 0)
            and np.all(np.diff([500, *cooling_t]) < 0),
            "Run the continuous heating 300-400-500 K then cooling 400-300 K sequence before NVT",
        )
    return stages


def _sampling_errors(data, sampled):
    """Conservative mean errors across block lengths and correlated samples.

    Duplicate thermal records do not establish additional independent
    observations. Controlled NVT volume is excluded from sampling requirements.
    These are evidence screens, not a proof of equilibration.
    """
    n = len(data)
    deviation = data.std(axis=0, ddof=1)
    block_errors = {
        str(count): np.array(
            [part.mean(axis=0) for part in np.array_split(data, count)]
        ).std(axis=0, ddof=1)
        / np.sqrt(count)
        for count in (4, 8, 16)
        if n >= 2 * count
    }
    se = np.maximum.reduce([deviation / np.sqrt(n), *block_errors.values()])
    effective = np.full(3, float(n))
    # Cell determinants can differ at floating-point roundoff even for repeated
    # volumes. Compare at 1e-9 in native units, far below the precision screens.
    distinct = len(np.unique(np.round(data[:, sampled], 9), axis=0))
    for index in np.flatnonzero(sampled):
        column = data[:, index]
        centered = column - column.mean()
        if np.max(np.abs(centered)) < 1e-10:
            effective[index] = 1.0
            continue
        transform = np.fft.rfft(centered, n=2 * n)
        acf = np.fft.irfft(transform * transform.conj())[: len(column)]
        acf /= acf[0]
        stop = next((i for i in range(1, len(acf)) if acf[i] <= 0), len(acf))
        effective[index] = min(distinct, n / max(1, 1 + 2 * acf[1:stop].sum()))
    se = np.maximum(se, deviation / np.sqrt(effective))
    # A larger block error must also reduce the effective sample count; taking
    # the first ACF zero alone can miss slower modes beneath rapid oscillations.
    from_error = np.divide(deviation**2, se**2, out=np.ones(3), where=se > 0)
    effective[sampled] = np.maximum(
        1, np.minimum(effective[sampled], from_error[sampled])
    )
    se[~sampled] = 0
    for error in block_errors.values():
        error[~sampled] = 0
    return se, effective, block_errors, distinct


def stage_statistics(stage):
    """Reconstruct reference means and require precise, settled production."""
    data = stage["data"]
    mean = data.mean(axis=0)
    sampled = np.array([True, stage["ensemble"] == "NPT", True])
    se, effective, block_errors, distinct = _sampling_errors(data, sampled)
    target = stage["target_temperature_K"]
    limits = np.array([0.02 * target, 0.001 * mean[1], 0.1])
    _require(
        np.min(effective[sampled]) >= 16 - 1e-9 and np.all(se <= limits),
        "Insufficient independent sampling or excessive uncertainty; "
        "constant or repeated production records cannot establish precision",
    )
    halves = np.array([part.mean(axis=0) for part in np.array_split(data, 2)])
    _require(
        np.all(np.abs(halves[1] - halves[0]) <= 4 * se + limits),
        "Production temperature, volume, or pressure has not settled",
    )
    _require(
        abs(mean[0] - target) <= 0.05 * target, "Measured temperature misses its target"
    )
    if stage["ensemble"] == "NPT":
        _require(
            abs(mean[2] - 1.01325e-4) <= 3 * se[2] + 0.05,
            "Measured NPT pressure misses its target",
        )

    def summary(values, errors, counts):
        return {
            "temperature_K": float(values[0]),
            "volume_A3": float(values[1]),
            "pressure_GPa": float(values[2]),
            "standard_errors": errors.tolist(),
            "effective_samples": [
                float(count) if active else None
                for count, active in zip(counts, sampled, strict=True)
            ],
        }

    windows = []
    for fraction in (0.25, 0.5):
        start = int(len(data) * fraction)
        tail = data[start:]
        tail_mean = tail.mean(axis=0)
        tail_se, tail_effective, _, _ = _sampling_errors(tail, sampled)
        # A shorter window has fewer samples, so allow the corresponding
        # sqrt(N/n) precision loss without allowing an unresolved tail to pass.
        retained = len(tail) / len(data)
        _require(
            np.min(tail_effective[sampled]) >= 16 * retained - 1e-9
            and np.all(tail_se <= limits / np.sqrt(retained)),
            "Later production window has insufficient independent sampling or precision",
        )
        _require(
            np.all(np.abs(tail_mean - mean) <= limits + 2 * np.hypot(se, tail_se)),
            "Reference means change when early production samples are excluded",
        )
        windows.append(
            {
                "discarded_samples": start,
                "sample_count": len(tail),
                **summary(tail_mean, tail_se, tail_effective),
            }
        )
    blocks = np.array([part.mean(axis=0) for part in np.array_split(data, 8)])
    stage.update(mean=mean, se=se, blocks=blocks)
    return {
        **summary(mean, se, effective),
        "sample_count": len(data),
        "distinct_thermal_samples": distinct,
        "block_standard_errors": {
            key: error.tolist() for key, error in block_errors.items()
        },
        "production_windows": windows,
    }


def _fit(x, y):
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    _require(rank == x.shape[1], "Unresolved or degenerate derivative design")
    return coefficients, y - x @ coefficients


def expansion_statistics(stages):
    heating = [i for i, s in enumerate(stages) if s["branch"] == "heating"]
    local = [i for i in heating if abs(stages[i]["target_temperature_K"] - 400) <= 50]
    nvt = [i for i, s in enumerate(stages) if s["ensemble"] == "NVT"]
    center = next(i for i in heating if stages[i]["target_temperature_K"] == 400)
    means = np.array([s["mean"] for s in stages])
    errors = np.array([s["se"] for s in stages])
    t, v = means[local, 0], means[local, 1]
    vstar = means[center, 1]
    _require(
        len(local) >= 3
        and np.all(np.abs(t - means[center, 0]) <= 55)
        and t.min() <= means[center, 0] - 5
        and t.max() >= means[center, 0] + 5,
        "Bracket 400 K with local NPT stages 5-50 K away",
    )
    states = means[nvt]
    _require(
        len(nvt) >= 5
        and all(abs(stages[i]["target_temperature_K"] - 400) <= 50 for i in nvt)
        and np.all(np.abs(states[:, 0] - means[center, 0]) <= 55)
        and np.all(np.abs(states[:, 1] / vstar - 1) <= 0.03),
        "Keep at least five NVT states local to (400 K, V*), within 50 K and 3% volume",
    )
    temperature_axis = [i for i in nvt if abs(means[i, 1] / vstar - 1) <= 1e-5]
    volume_axis = [i for i in nvt if stages[i]["target_temperature_K"] == 400]
    _require(
        len(temperature_axis) >= 3
        and len(volume_axis) >= 3
        and any(i in volume_axis for i in temperature_axis)
        and means[temperature_axis, 0].min() <= means[center, 0] - 5
        and means[temperature_axis, 0].max() >= means[center, 0] + 5
        and means[volume_axis, 1].min() <= 0.998 * vstar
        and means[volume_axis, 1].max() >= 1.002 * vstar,
        "Resolve both pressure derivatives with bracketing temperature and volume perturbations",
    )

    def calculate(data):
        h, loc, fixed = data[heating], data[local], data[nvt]
        tc, vc = data[center, :2]
        interval, residuals = _fit(
            np.column_stack([np.ones(len(h)), h[:, 0] - tc]), h[:, 1]
        )
        local_fit, _ = _fit(
            np.column_stack([np.ones(len(loc)), loc[:, 0] - tc]), loc[:, 1]
        )
        # Scaling makes the pressure design well conditioned. Convert slopes
        # back to GPa/K and GPa/A^3 before applying thermodynamic identities.
        design = np.column_stack(
            [
                np.ones(len(fixed)),
                (fixed[:, 0] - tc) / 50,
                (fixed[:, 1] / vc - 1) / 0.01,
            ]
        )
        pressure_fit, pressure_residual = _fit(design, fixed[:, 2])
        dpdt, dpdv = pressure_fit[1] / 50, pressure_fit[2] / (0.01 * vc)
        modulus = -vc * dpdv
        _require(abs(modulus) > 1e-10, "Unresolved bulk modulus")
        return (
            np.array(
                [
                    interval[1] / h[0, 1],
                    local_fit[1] / vc,
                    modulus,
                    dpdt / modulus,
                    dpdt,
                    dpdv,
                ]
            ),
            residuals,
            pressure_residual,
        )

    values, residuals, pressure_residual = calculate(means)
    # Resample stage blocks independently; preserve T/V/P covariance within
    # each block. Inflate block fluctuations when autocorrelation gives a
    # larger mean error. Agent-provided error bars never affect acceptance.
    rng = np.random.default_rng(7)
    replicas = []
    for _ in range(256):
        data = []
        for stage in stages:
            b = stage["blocks"]
            base_se = b.std(axis=0) / np.sqrt(8)
            scale = np.divide(
                stage["se"], base_se, out=np.ones(3), where=base_se > 1e-12
            )
            delta = (b[rng.integers(8, size=8)].mean(axis=0) - b.mean(axis=0)) * scale
            # Periodic signals can cancel exactly in every block. Preserve
            # the sampling-error floor even when block resampling is constant.
            delta = np.where(base_se > 1e-12, delta, rng.normal(size=3) * stage["se"])
            data.append(stage["mean"] + delta)
        replicas.append(calculate(np.array(data))[0])
    replicas = np.array(replicas)
    se = replicas.std(axis=0, ddof=1)
    _require(
        np.all(se <= 0.25 * np.abs(values)),
        "Expansion or pressure derivatives are not resolved",
    )
    _require(
        np.all(
            np.abs(pressure_residual) <= 3 * errors[nvt, 2] + 0.1 * np.ptp(states[:, 2])
        ),
        "Pressure response is not locally linear",
    )
    # Separate left/right slopes screen curvature, not interval/local equality.
    tc = means[center, 0]
    slopes = [
        (v[t < tc].mean() - vstar) / (t[t < tc].mean() - tc),
        (v[t > tc].mean() - vstar) / (t[t > tc].mean() - tc),
    ]
    _require(
        abs(slopes[1] - slopes[0]) / vstar <= 0.25 * abs(values[1]) + 3 * se[1],
        "NPT perturbations do not support a local derivative",
    )
    for subset, column, derivative in ((temperature_axis, 0, 4), (volume_axis, 1, 5)):
        x = means[subset, column]
        slope, _ = _fit(
            np.column_stack([np.ones(len(x)), x - x.mean()]), means[subset, 2]
        )
        _require(
            abs(slope[1] - values[derivative])
            <= 0.25 * abs(values[derivative]) + 3 * se[derivative],
            "Pressure derivatives change with the fitting window",
        )
    halves = [
        calculate(
            np.array([np.array_split(s["data"], 2)[half].mean(axis=0) for s in stages])
        )[0]
        for half in (0, 1)
    ]
    _require(
        np.all(np.abs(halves[0] - halves[1]) <= 0.25 * np.abs(values) + 4 * se),
        "Derivatives are unstable across production windows",
    )
    names = (
        "beta_volume",
        "beta_npt_local",
        "bulk_modulus_GPa",
        "beta_pressure",
        "dP_dT_GPa_per_K",
        "dP_dV_GPa_per_A3",
    )
    result = dict(zip(names, map(float, values), strict=True))
    difference = float(values[3] - values[1])
    difference_se = float((replicas[:, 3] - replicas[:, 1]).std(ddof=1))
    result.update(
        standard_errors=dict(zip(names, map(float, se), strict=True)),
        local_difference_K_inv=difference,
        local_difference_se_K_inv=difference_se,
        local_routes_consistent=bool(abs(difference) <= 3 * difference_se + 1e-7),
        stable=bool(values[2] > 0),
        reference_volume_A3=float(vstar),
        volume_fit_residuals_A3=residuals.tolist(),
    )
    result["volume_differences_A3"] = {}
    for temperature in (300, 400):
        indexes = [
            next(
                i
                for i, s in enumerate(stages)
                if s["branch"] == branch and s["target_temperature_K"] == temperature
            )
            for branch in ("heating", "cooling")
        ]
        result["volume_differences_A3"][str(temperature)] = float(
            means[indexes[1], 1] - means[indexes[0], 1]
        )
    return result


class Task7Scorer:
    """Binary scoring of log completeness, sampling, and independently derived answers."""

    task_number = 7
    verifier = None

    def __init__(self, *, level=2):
        if type(level) is not int or level not in (1, 2):
            raise ValueError("level must be 1 or 2")
        self.level = level

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
        rubric, computed = Rubric(7, binary=True), {}

        def check():
            evidence = Evidence(submission, task_number=7, level=self.level)
            stages = read_log(evidence, self.level)
            computed["stages"] = [stage_statistics(s) for s in stages]

            def compare(key, actual, expected, atol):
                actual = finite_array(actual)
                expected = finite_array(expected)
                _require(
                    actual.shape == expected.shape
                    and np.all(
                        np.abs(actual - expected) <= atol + 0.01 * np.abs(expected)
                    ),
                    f"{key}: reported value differs from the log calculation",
                )

            if self.level == 1:
                for key, atol in (
                    ("temperature_K", 0.1),
                    ("volume_A3", 0.01),
                    ("pressure_GPa", 1e-6),
                ):
                    if key not in evidence.results:
                        continue
                    compare(
                        key, evidence.results[key], computed["stages"][0][key], atol
                    )
            else:
                computed.update(expansion_statistics(stages))
                for key, atol in (
                    ("beta_volume", 1e-8),
                    ("beta_npt_local", 1e-8),
                    ("bulk_modulus_GPa", 0.01),
                    ("beta_pressure", 1e-8),
                    ("local_difference_K_inv", 1e-8),
                ):
                    if key != "beta_volume" and key not in evidence.results:
                        continue
                    compare(key, evidence.results[key], computed[key], atol)
                claims = evidence.results.get("volume_differences_A3", {})
                _require(
                    isinstance(claims, dict), "Volume differences must be an object"
                )
                for t in ("300", "400"):
                    if t not in claims:
                        continue
                    compare(
                        f"volume_differences_A3.{t}",
                        claims[t],
                        computed["volume_differences_A3"][t],
                        0.01,
                    )
            return True

        rubric.check("recorded_aluminum_expansion", 100, check)
        return {
            **rubric.as_dict(),
            "level": self.level,
            "mode": "saved_logs",
            "computed": computed,
        }
