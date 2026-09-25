"""Saved-data fixtures verify task 7 without MD or potential evaluation."""

import copy
import json
import pickle
import subprocess

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from corral_md.workflow_scoring.common import Evidence, Rubric, reproducibility
from corral_md.workflow_scoring.level1 import evaluate as evaluate_level1
from corral_md.workflow_scoring.task_7 import evaluate
from scipy.stats import theilslopes

ORDER = [
    "beta_volume",
    "beta_npt_local",
    "reference_volume_A3",
    "bulk_modulus_GPa",
    "beta_pressure",
    "beta_difference",
    "volume_difference_300_A3",
    "volume_difference_400_A3",
]

GPA = 160.21766208


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path, value):
    path.write_text(json.dumps(_plain(value)))
    return str(path)


def _read(path):
    return json.loads(path.read_text())


def test_level1_accepts_untimed_production_only_trajectory(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    initial = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": [True] * 3,
    }
    rng = np.random.default_rng(7)
    momenta = rng.normal(size=(108, 3))
    momenta -= momenta.mean(axis=0)
    momenta *= np.sqrt(
        324 * units.kB * 300
        / np.sum(momenta**2 / atoms.get_masses()[:, None])
    )
    frames, rows = [], []
    for index, step in enumerate((200, 300, 400)):
        scale = 1 + index * 0.001
        frame = {
            **initial,
            "positions": (atoms.positions * scale).tolist(),
            "cell": (atoms.cell.array * scale).tolist(),
            "momenta": momenta.tolist(),
        }
        frames.append(frame)
        rows.append(
            {
                "stage": "reference",
                "phase": "production",
                "step": step,
                "time_fs": float(step),
                "temperature_K": 300.0,
                "volume_A3": float(atoms.get_volume() * scale**3),
                "pressure_GPa": 0.01 + index * 0.001,
            }
        )
    settings = {
        "model": "/workspace/models/teacher.model",
        "input_structure": "initial.json",
        "temperature_dof": 321,
        "random_seed": 7,
        "velocity_initializations": 1,
        "remove_com": True,
        "timestep_fs": 1,
        "thermostat": "NPT thermostat",
        "barostat": "isotropic barostat",
        "target_pressure_bar": 1.01325,
        "isotropic": True,
        "equilibration_steps": 100,
        "production_steps": 300,
        "trajectory_interval_steps": 100,
    }
    manifest = {
        "artifacts": {
            "initial_structure": _write(tmp_path / "initial.json", [initial]),
            "stages": _write(
                tmp_path / "stages.json",
                [{"id": "reference", "ensemble": "NPT", "target_temperature_K": 300,
                  "production": [200, 300, 400]}],
            ),
            "trajectories": {"reference": _write(tmp_path / "production.json", frames)},
            "thermal_trace": _write(tmp_path / "trace.json", rows),
            "final_state": _write(tmp_path / "final.json", [frames[-1]]),
        },
        "settings": _write(tmp_path / "settings.json", settings),
        "results": {
            "stages": {
                "reference": {
                    "temperature_K": 300.0,
                    "volume_A3": float(np.mean([row["volume_A3"] for row in rows])),
                    "pressure_GPa": 0.011,
                }
            }
        },
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    rubric = Rubric(7, fail_fast=False)
    evaluate_level1(Evidence(path), rubric, 7)
    for name in (
        "initial_fcc_geometry",
        "continuous_isotropic_npt_state",
        "measured_thermal_trace_and_means",
    ):
        assert next(c for c in rubric.checks if c["name"] == name)["status"] == "passed"

    rows[-1]["temperature_K"] += 100
    _write(tmp_path / "trace.json", rows)
    rubric = Rubric(7, fail_fast=False)
    evaluate_level1(Evidence(path), rubric, 7)
    assert next(
        c for c in rubric.checks if c["name"] == "measured_thermal_trace_and_means"
    )["status"] == "failed"


def _reference(averages, local_method="linear", pressure_method="linear"):
    """Independent implementation of the documented estimands."""
    data = np.array([averages[f"h{t}"] for t in (300, 400, 500)])
    slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
    reference = averages["h400"][1]
    degree = 1 if local_method == "linear" else 2
    local_coef = np.polyfit(data[:, 0] - averages["h400"][0], data[:, 1], degree)
    local = np.polyder(local_coef)[-1] / reference
    nvt = np.array([averages[f"n{i}"] for i in range(9)])
    dt = nvt[:, 0] - 400
    dv = nvt[:, 1] - reference
    x = np.column_stack([np.ones(9), dt, dv])
    if pressure_method == "quadratic":
        x = np.column_stack([x, dt**2, dt * dv, dv**2])
    coef = np.linalg.lstsq(x, nvt[:, 2], rcond=None)[0]
    modulus = -reference * coef[2]
    bp = coef[1] / modulus
    stats = np.array(
        [
            slope / averages["h300"][1],
            local,
            reference,
            modulus,
            bp,
            bp - local,
            averages["c300"][1] - averages["h300"][1],
            averages["c400"][1] - averages["h400"][1],
        ]
    )
    return (
        stats,
        {
            "slope_A3_per_K": slope,
            "residuals_A3": data[:, 1] - (slope * data[:, 0] + intercept),
        },
        {
            "residuals_A3": data[:, 1]
            - np.polyval(local_coef, data[:, 0] - averages["h400"][0])
        },
        {"coefficients": coef, "residuals_GPa": nvt[:, 2] - x @ coef},
    )


def _resample(config, values):
    summaries = {sid: [] for sid in values}
    statistics = []
    for draw in config["resamples"]:
        averages = {}
        for sid in values:
            indices = np.concatenate(
                [np.arange(*config["blocks"][sid][b]) for b in draw[sid]]
            )
            averages[sid] = values[sid][indices].mean(axis=0)
            summaries[sid].append(averages[sid])
        statistics.append(_reference(averages)[0])
    statistics = np.array(statistics)
    factor = (
        (len(statistics) - 1) ** 2 / len(statistics)
        if config["method"] == "delete_block_jackknife"
        else 1
    )
    covariance = np.cov(statistics, rowvar=False) * factor
    standard = np.sqrt(np.diag(covariance))
    denom = np.outer(standard, standard)
    resolution = np.maximum(1e-15, 1e-12 * np.abs(statistics.mean(axis=0)))
    resolution[5] = max(resolution[5], resolution[1], resolution[4])
    resolution[6:] = np.maximum(resolution[6:], resolution[2])
    resolved = standard > resolution
    correlation = np.divide(
        covariance,
        denom,
        out=np.zeros_like(covariance),
        where=np.outer(resolved, resolved),
    )
    errors = {
        sid: np.std(records, axis=0, ddof=1) * np.sqrt(factor)
        for sid, records in summaries.items()
    }
    return statistics, covariance, standard, correlation, errors


def _fixture(
    tmp_path, modulus=70.0, jackknife=False, failed_heating=False, heating_offsets=None
):
    rng = np.random.default_rng(153)
    structure = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    v0 = structure.get_volume()
    masses = structure.get_masses()
    direction = rng.normal(size=(108, 3))
    direction -= direction.mean(axis=0)
    direction /= np.sqrt(np.sum(direction**2 / masses[:, None]) / (321 * units.kB))
    fractional = structure.get_scaled_positions()
    heating_offsets = heating_offsets or {}
    vstar = v0 * (1 + 0.00009 * 100) + heating_offsets.get("h400", (0, 0))[1]
    stage_map = []
    raw_records = []
    trajectories = {}
    values = {}
    time = 0.0
    previous = None

    def frame(sid, time, t, vol, pressure):
        cell = np.eye(3) * vol ** (1 / 3)
        momenta = direction * np.sqrt(t)
        energy = -300 + 0.01 * t
        kinetic = 321 * units.kB * t / 2
        virial = -(pressure - 2 * kinetic / (3 * vol) * GPA) / GPA
        state = {
            "symbols": ["Al"] * 108,
            "positions": fractional @ cell,
            "cell": cell,
            "pbc": True,
            "momenta": momenta,
            "energy": energy,
            "time_fs": time,
            "stage": sid,
        }
        record = {
            "stage": sid,
            "time_fs": time,
            "temperature_K": t,
            "volume_A3": vol,
            "kinetic_energy_eV": kinetic,
            "potential_energy_eV": energy,
            "pressure_GPa": pressure,
            "sxx": virial,
            "syy": virial,
            "szz": virial,
            "syz": 0.0,
            "sxz": 0.0,
            "sxy": 0.0,
        }
        return _plain(state), record, [t, vol, pressure]

    noise = np.array([-0.6, -0.5, -0.4, -0.3, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.2, 0.1])
    noise -= noise.mean()
    for branch, temperature in [
        ("h", 300),
        ("h", 400),
        ("h", 500),
        ("c", 400),
        ("c", 300),
    ]:
        sid = f"{branch}{temperature}"
        stage_map.append(
            {
                "id": sid,
                "ensemble": "NPT",
                "branch": "heating" if branch == "h" else "cooling",
                "target_temperature_K": temperature,
                "production": [1, 13],
            }
        )
        temperature_offset, volume_offset = heating_offsets.get(sid, (0, 0))
        actual_temperature = (
            300
            if failed_heating and sid == "h500"
            else temperature + temperature_offset
        )
        frames = []
        measurements = []
        if previous is None:
            first, record, data = frame(sid, time, 300, v0, 0.000101325)
        else:
            first = copy.deepcopy(previous)
            first["stage"] = sid
            last = raw_records[-1]
            record = {**last, "stage": sid}
            data = [last["temperature_K"], last["volume_A3"], last["pressure_GPa"]]
        frames.append(first)
        measurements.append(data)
        raw_records.append(record)
        for i in range(12):
            time += 10
            t = actual_temperature + noise[i] * 3
            vol = (
                v0 * (1 + 0.00009 * (temperature - 300))
                + noise[i] * 0.5
                + (0.4 if branch == "c" else 0)
                + volume_offset
            )
            p = 0.000101325 + noise[i] * 0.05
            state, record, data = frame(sid, time, t, vol, p)
            frames.append(state)
            raw_records.append(record)
            measurements.append(data)
        previous = frames[-1]
        trajectories[sid] = _write(tmp_path / f"{sid}.json", frames)
        values[sid] = np.array(measurements)
    for i, (target_t, scale) in enumerate(
        (t, v) for t in (380, 400, 420) for v in (0.99, 1.0, 1.01)
    ):
        sid = f"n{i}"
        vol = vstar * scale
        stage_map.append(
            {
                "id": sid,
                "ensemble": "NVT",
                "target_temperature_K": target_t,
                "production": [1, 13],
            }
        )
        frames = []
        measurements = []
        for j in range(13):
            t = target_t + (noise[j - 1] * 3 if j else 0)
            p = (
                0.0025 * (t - 400)
                - modulus / vstar * (vol - vstar)
                + (noise[j - 1] * 0.01 if j else 0)
            )
            state, record, data = frame(sid, j * 10.0, t, vol, p)
            frames.append(state)
            raw_records.append(record)
            measurements.append(data)
        trajectories[sid] = _write(tmp_path / f"{sid}.json", frames)
        values[sid] = np.array(measurements)
    settings = {
        "model": "MACE-MP-0",
        "input_structure": "supplied-108-Al",
        "temperature_dof": 321,
        "random_seed": 42,
        "velocity_initializations": 1,
        "remove_com": True,
        "timestep_fs": 1,
        "thermostat": "Langevin",
        "barostat": "isotropic",
        "target_pressure_bar": 1.01325,
        "isotropic": True,
        "stress_unit": "eV/Angstrom^3",
        "stress_sign": "tensile_positive",
        "stress_kind": "potential_only",
        "local_npt": {"method": "linear", "stage_ids": ["h300", "h400", "h500"]},
        "pressure_fit": {"method": "linear", "stage_ids": [f"n{i}" for i in range(9)]},
        "resampling_seed": 987,
    }
    averages = {sid: val[1:].mean(axis=0) for sid, val in values.items()}
    stats, volume_fit, local_fit, pressure_fit = _reference(averages)
    results = {
        "beta_volume": stats[0],
        "beta_npt_local": stats[1],
        "reference_volume_A3": stats[2],
        "bulk_modulus_GPa": stats[3],
        "beta_pressure": stats[4],
        "stable": bool(stats[3] > 0),
        "volume_fit": volume_fit,
        "local_npt_fit": local_fit,
        "pressure_fit": pressure_fit,
        "units": {
            "temperature": "K",
            "volume": "Angstrom^3",
            "pressure": "GPa",
            "bulk_modulus": "GPa",
            "beta": "K^-1",
        },
        "stages": {},
    }
    for sid, val in values.items():
        times = np.arange(12) * 0.01
        selected = val[1:]
        mean = averages[sid]
        results["stages"][sid] = {
            "temperature_K": mean[0],
            "volume_A3": mean[1],
            "pressure_GPa": mean[2],
            "temperature_drift_K_per_ps": np.polyfit(times, selected[:, 0], 1)[0],
            "volume_drift_A3_per_ps": np.polyfit(times, selected[:, 1], 1)[0],
            "first_half": {
                "temperature_K": selected[:6, 0].mean(),
                "volume_A3": selected[:6, 1].mean(),
            },
            "second_half": {
                "temperature_K": selected[6:, 0].mean(),
                "volume_A3": selected[6:, 1].mean(),
            },
        }

    def bootstrap(blocks):
        return {
            "method": "block_bootstrap",
            "blocks": dict.fromkeys(values, blocks),
            "resamples": [
                {
                    sid: rng.integers(len(blocks), size=len(blocks)).tolist()
                    for sid in values
                }
                for _ in range(30)
            ],
        }

    uncertainty = bootstrap([[1, 5], [5, 9], [9, 13]])
    if jackknife:
        uncertainty["method"] = "delete_block_jackknife"
        uncertainty["resamples"] = [
            {sid: [i for i in range(3) if i != omitted] for sid in values}
            for omitted in range(3)
        ]
    alternative = bootstrap([[1, 7], [7, 13]])
    replicates, cov, se, corr, errors = _resample(uncertainty, values)
    alternate_se = _resample(alternative, values)[2]
    uncertainty["alternative"] = alternative
    for sid, error in errors.items():
        results["stages"][sid].update(
            temperature_sem_K=error[0],
            volume_sem_A3=error[1],
            pressure_sem_GPa=error[2],
        )
    results["uncertainty"] = {
        "order": ORDER,
        "replicates": replicates,
        "covariance": cov,
        "standard_errors": se,
        "correlation": corr,
        "alternative_standard_errors": alternate_se,
    }
    results["volume_differences"] = {
        str(t): {"value_A3": stats[i], "standard_error_A3": se[i]}
        for t, i in [(300, 6), (400, 7)]
    }
    results["comparison"] = {
        "difference_K_inv": stats[5],
        "standard_error_K_inv": se[5],
        "covariance_K_inv_squared": cov[1, 4],
        "z_score": stats[5] / se[5] if se[5] else None,
    }
    alternative_fit = _reference(averages, "quadratic", "quadratic")[0]
    results["sensitivity"] = {
        "beta_npt_local": alternative_fit[1],
        "beta_pressure": alternative_fit[4],
    }
    sensitivity = {
        "local_npt": {**settings["local_npt"], "method": "quadratic"},
        "pressure_fit": {**settings["pressure_fit"], "method": "quadratic"},
    }
    initial = _read(tmp_path / "h300.json")[0]
    script = tmp_path / "simulation.py"
    script.write_text("raise RuntimeError('Submitted scripts must never execute')\n")
    manifest = {
        "results": results,
        "settings": _write(tmp_path / "settings.json", settings),
        "scripts": [str(script)],
        "report": _write(
            tmp_path / "report.json", {"analysis": "Synthetic reproducibility fixture"}
        ),
        "artifacts": {
            "trajectories": trajectories,
            "stages": _write(tmp_path / "stages.json", stage_map),
            "thermal_trace": _write(tmp_path / "thermal_trace.json", raw_records),
            "initial_structure": _write(tmp_path / "initial.json", [initial]),
            "uncertainty": _write(tmp_path / "uncertainty.json", uncertainty),
            "sensitivity": _write(tmp_path / "sensitivity.json", sensitivity),
        },
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    return path


def _score(path):
    evidence = Evidence(str(path))
    rubric = Rubric(7)
    reproducibility(evidence, rubric)
    evaluate(evidence, rubric)
    return rubric


def _check(rubric, name):
    return next(c for c in rubric.checks if c["name"] == name)


@pytest.mark.parametrize("jackknife", [False, True])
def test_complete_artifact_evidence(tmp_path, jackknife):
    result = _score(_fixture(tmp_path, jackknife=jackknife))
    assert result.score == 1, [
        c for c in result.checks if c["points"] and c["status"] != "passed"
    ]
    assert sum(c["points"] for c in result.checks) == 100
    assert _check(result, "execution_and_input_provenance")["status"] == "unverified"


def test_negative_bulk_modulus_is_faithfully_diagnosed(tmp_path):
    result = _score(_fixture(tmp_path, modulus=-70))
    assert result.score == 1, [
        c for c in result.checks if c["points"] and c["status"] != "passed"
    ]
    assert _check(result, "bulk_modulus_stability_diagnostic")["status"] == "passed"


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("geometry", "initial_fcc_geometry"),
        ("continuity", "continuous_isotropic_npt_cycle"),
        ("stress_units", "raw_stress_and_kinetic_pressure"),
        ("kinetic_pressure", "raw_stress_and_kinetic_pressure"),
        ("expansion", "interval_expansion_fit"),
        ("missing_trace", "raw_stress_and_kinetic_pressure"),
        ("fixed_cell", "fixed_cell_nvt_evidence"),
    ],
)
def test_corrupt_evidence_loses_dependent_credit(tmp_path, case, failed_check):
    path = _fixture(tmp_path)
    manifest = _read(path)
    if case == "geometry":
        for filename in ("h300.json", "initial.json"):
            data = _read(tmp_path / filename)
            data[0]["positions"][0][0] += 0.1
            _write(tmp_path / filename, data)
    elif case == "continuity":
        data = _read(tmp_path / "h400.json")
        data[0]["momenta"][0][0] += 0.1
        _write(tmp_path / "h400.json", data)
    elif case == "stress_units":
        data = _read(tmp_path / "settings.json")
        data["stress_unit"] = "GPa"
        _write(tmp_path / "settings.json", data)
    elif case == "kinetic_pressure":
        data = _read(tmp_path / "thermal_trace.json")
        for row in data:
            row["pressure_GPa"] -= (
                2 * row["kinetic_energy_eV"] / (3 * row["volume_A3"]) * GPA
            )
        _write(tmp_path / "thermal_trace.json", data)
    elif case == "expansion":
        manifest["results"]["beta_volume"] *= 2
    elif case == "covariance":
        manifest["results"]["uncertainty"]["covariance"] = np.zeros((8, 8))
    elif case == "resampling":
        data = _read(tmp_path / "uncertainty.json")
        data["resamples"] = [data["resamples"][0]] * len(data["resamples"])
        _write(tmp_path / "uncertainty.json", data)
    elif case == "missing_trace":
        del manifest["artifacts"]["thermal_trace"]
    elif case == "fixed_cell":
        data = _read(tmp_path / "n0.json")
        data[5]["cell"][0][0] *= 1.01
        _write(tmp_path / "n0.json", data)
    _write(path, manifest)
    result = _score(path)
    assert result.score < 1
    assert _check(result, failed_check)["status"] == "failed"
    if case == "geometry":
        assert _check(result, "interval_expansion_fit")["earned"] == 0
        assert _check(result, "local_pressure_derivatives")["earned"] == 0
    if case == "covariance":
        assert _check(result, "local_route_comparison")["earned"] == 0


def test_constant_300k_does_not_satisfy_heating(tmp_path):
    result = _score(_fixture(tmp_path, failed_heating=True))
    assert _check(result, "continuous_isotropic_npt_cycle")["status"] == "passed"
    assert (
        _check(result, "measured_temperature_and_pressure_targets")["status"]
        == "unverified"
    )
    assert result.score is None


def test_missing_nvt_does_not_remove_volume_route_fit(tmp_path):
    path = _fixture(tmp_path)
    data = _read(path)
    del data["artifacts"]["trajectories"]["n0"]
    _write(path, data)
    result = _score(path)
    assert _check(result, "interval_expansion_fit")["status"] == "passed"
    assert _check(result, "fixed_cell_nvt_evidence")["earned"] == 0


def test_reference_geometry_allows_rotation_translation_and_order(tmp_path):
    path = _fixture(tmp_path)
    data = _read(tmp_path / "initial.json")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    data[0]["cell"] = np.asarray(data[0]["cell"]) @ rotation
    data[0]["positions"] = np.asarray(data[0]["positions"])[::-1] @ rotation + np.array(
        [0.13, 0.21, 0.08]
    )
    _write(tmp_path / "initial.json", data)
    assert _score(path).score == 1


@pytest.mark.parametrize(
    ("stress_unit", "conversion"), [("GPa", GPA), ("bar", GPA * 1e4)]
)
def test_pressure_unit_and_sign_equivalence(tmp_path, stress_unit, conversion):
    path = _fixture(tmp_path)
    settings = _read(tmp_path / "settings.json")
    settings["stress_unit"] = stress_unit
    settings["stress_sign"] = "compression_positive"
    data = _read(tmp_path / "thermal_trace.json")
    for row in data:
        for key in ["sxx", "syy", "szz", "syz", "sxz", "sxy"]:
            row[key] *= -conversion
    _write(tmp_path / "thermal_trace.json", data)
    _write(tmp_path / "settings.json", settings)
    assert _score(path).score == 1


def test_rank_deficient_pressure_sampling_loses_derivative_credit(tmp_path):
    path = _fixture(tmp_path)
    settings = _read(tmp_path / "settings.json")
    settings["pressure_fit"]["stage_ids"] = ["n0", "n4", "n8"]
    _write(tmp_path / "settings.json", settings)
    result = _score(path)
    assert _check(result, "local_pressure_derivatives")["earned"] == 0
    assert _check(result, "interval_expansion_fit")["status"] == "passed"


def test_scoring_never_executes_code_or_potential(tmp_path, monkeypatch):
    path = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Execution forbidden during artifact scoring")

    monkeypatch.setattr(Atoms, "get_potential_energy", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    result = _score(path)
    assert result.score == 1


def test_empty_submission_fails_cleanly():
    result = Rubric(7)
    evidence = Evidence({"results": {}, "artifacts": {}})
    evaluate(evidence, result)
    assert result.score == 0
    assert sum(c["points"] for c in result.checks) == 90


@pytest.mark.parametrize("corrupt", [False, True])
def test_declared_finite_difference_derivatives(tmp_path, corrupt):
    path = _fixture(tmp_path)
    manifest = _read(path)
    settings = _read(tmp_path / "settings.json")
    results = manifest["results"]
    volume = results["reference_volume_A3"]
    local_weights = np.array([-1 / 200, 0, 1 / 200])
    settings["local_npt"].update(
        method="finite_difference", derivative_weights_K_inv=local_weights.tolist()
    )
    selected = np.array(
        [
            [
                results["stages"][sid][key]
                for key in ("temperature_K", "volume_A3", "pressure_GPa")
            ]
            for sid in settings["pressure_fit"]["stage_ids"]
        ]
    )
    weights = np.zeros((2, 9))
    weights[0, [1, 7]] = [-1 / 40, 1 / 40]
    weights[1, [3, 5]] = [-1 / (0.02 * volume), 1 / (0.02 * volume)]
    gradients = weights @ selected[:, 2]
    offsets = selected[:, :2] - np.array([400, volume])
    intercept = np.mean(selected[:, 2] - offsets @ gradients)
    results["pressure_fit"] = {
        "dP_dT_GPa_per_K": gradients[0],
        "dP_dV_GPa_per_A3": gradients[1],
        "residuals_GPa": selected[:, 2] - intercept - offsets @ gradients,
    }
    results["local_npt_fit"] = {
        "derivative_A3_per_K": results["beta_npt_local"] * volume
    }
    if corrupt:
        weights[0, 1] *= 2
    settings["pressure_fit"].update(
        method="finite_difference", derivative_weights=weights
    )
    _write(tmp_path / "settings.json", settings)
    _write(path, manifest)
    result = _score(path)
    if corrupt:
        assert _check(result, "local_pressure_derivatives")["status"] == "failed"
        assert _check(result, "local_route_comparison")["status"] == "failed"
    else:
        assert result.score == 1, result.checks


def test_open_local_methods_leave_only_method_verification_pending(tmp_path):
    path = _fixture(tmp_path)
    settings = _read(tmp_path / "settings.json")
    settings["local_npt"]["method"] = "smoothing_spline"
    settings["pressure_fit"]["method"] = "local_kernel_regression"
    _write(tmp_path / "settings.json", settings)
    result = _score(path)
    assert result.score is None
    for name in ("local_npt_expansion_fit", "local_pressure_derivatives"):
        assert _check(result, name)["status"] == "unverified"
    for name in (
        "interval_expansion_fit",
        "local_route_comparison",
        "bulk_modulus_stability_diagnostic",
    ):
        assert _check(result, name)["status"] == "passed"


@pytest.mark.parametrize("corrupt", [False, True])
def test_one_sided_pressure_derivative_gets_locality_review(tmp_path, corrupt):
    path = _fixture(tmp_path)
    settings = _read(tmp_path / "settings.json")
    manifest = _read(path)
    ids = ["n5", "n7", "n8"]
    means = np.array(
        [
            [
                manifest["results"]["stages"][sid][key]
                for key in ("temperature_K", "volume_A3", "pressure_GPa")
            ]
            for sid in ids
        ]
    )
    volume = manifest["results"]["reference_volume_A3"]
    design = np.column_stack([np.ones(3), means[:, 0] - 400, means[:, 1] - volume])
    weights = np.linalg.inv(design)[1:]
    derivatives = weights @ means[:, 2]
    manifest["results"]["pressure_fit"] = {
        "dP_dT_GPa_per_K": derivatives[0],
        "dP_dV_GPa_per_A3": derivatives[1],
        "residuals_GPa": [0.0] * 3,
    }
    if corrupt:
        weights[0, 0] += 0.1
    settings["pressure_fit"] = {
        "method": "finite_difference",
        "stage_ids": ids,
        "derivative_weights": weights,
        "sampling_design": "one_sided",
    }
    _write(tmp_path / "settings.json", settings)
    _write(path, manifest)
    result = _score(path)
    if corrupt:
        assert _check(result, "local_pressure_derivatives")["status"] == "failed"
    else:
        assert result.score is None
        assert _check(result, "local_pressure_derivatives")["status"] == "unverified"
        assert _check(result, "interval_expansion_fit")["status"] == "passed"


def test_raw_thermal_traces_support_other_equilibration_diagnostics(tmp_path):
    path = _fixture(tmp_path)
    manifest = _read(path)
    for stage in manifest["results"]["stages"].values():
        for key in (
            "first_half",
            "second_half",
            "temperature_drift_K_per_ps",
            "volume_drift_A3_per_ps",
        ):
            del stage[key]
    _write(path, manifest)
    result = _score(path)
    assert result.score == 1, result.checks


def _volume_estimator_fixture(tmp_path, method, intercept_method="joint"):
    """Noisy stage means distinguish estimator choices from a perfect line."""
    path = _fixture(tmp_path, heating_offsets={"h400": (12, 12), "h500": (0, 3)})
    manifest = _read(path)
    settings = _read(tmp_path / "settings.json")
    results = manifest["results"]
    data = np.array(
        [
            [results["stages"][f"h{t}"][key] for key in ("temperature_K", "volume_A3")]
            for t in (300, 400, 500)
        ]
    )
    unweighted_slope = results["volume_fit"]["slope_A3_per_K"]
    settings["volume_fit"] = {"method": method}
    if method in ("wls", "weighted_least_squares"):
        weights = np.array([1.0, 4.0, 9.0])
        settings["volume_fit"]["weights"] = weights
        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1, w=np.sqrt(weights))
    elif method == "theil_sen":
        settings["volume_fit"]["intercept_method"] = intercept_method
        slope, intercept, *_ = theilslopes(
            data[:, 1], data[:, 0], method=intercept_method
        )
    else:
        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
    results["beta_volume"] = slope / data[0, 1]
    results["volume_fit"] = {
        "slope_A3_per_K": slope,
        "residuals_A3": data[:, 1] - (slope * data[:, 0] + intercept),
    }
    _write(tmp_path / "settings.json", settings)
    _write(path, manifest)
    return path, unweighted_slope


@pytest.mark.parametrize(
    "method",
    ["ols", "linear", "ordinary_least_squares", "unweighted_least_squares"],
)
def test_declared_unweighted_volume_fit(tmp_path, method):
    path, _ = _volume_estimator_fixture(tmp_path, method)
    result = _score(path)
    assert result.score == 1, result.checks


@pytest.mark.parametrize("method", ["wls", "weighted_least_squares"])
def test_declared_weighted_volume_fit_gets_full_credit(tmp_path, method):
    path, unweighted_slope = _volume_estimator_fixture(tmp_path, method)
    weighted_slope = _read(path)["results"]["volume_fit"]["slope_A3_per_K"]
    assert abs(weighted_slope / unweighted_slope - 1) > 0.01
    result = _score(path)
    assert result.score == 1, result.checks


@pytest.mark.parametrize("intercept_method", ["joint", "separate"])
def test_declared_robust_volume_fit_gets_full_credit(tmp_path, intercept_method):
    path, unweighted_slope = _volume_estimator_fixture(
        tmp_path, "theil_sen", intercept_method
    )
    robust_slope = _read(path)["results"]["volume_fit"]["slope_A3_per_K"]
    assert abs(robust_slope / unweighted_slope - 1) > 0.01
    result = _score(path)
    assert result.score == 1, result.checks


@pytest.mark.parametrize(
    "weights", [[1, -4, 9], [1, 0, 9], [1, 4], [1, "nan", 9], None]
)
def test_invalid_volume_fit_weights_fail(tmp_path, weights):
    path, _ = _volume_estimator_fixture(tmp_path, "wls")
    settings = _read(tmp_path / "settings.json")
    settings["volume_fit"]["weights"] = weights
    _write(tmp_path / "settings.json", settings)
    result = _score(path)
    assert _check(result, "interval_expansion_fit")["status"] == "failed"
    assert _check(result, "local_npt_expansion_fit")["status"] == "passed"


@pytest.mark.parametrize("method", ["wls", "custom_linear_regression"])
@pytest.mark.parametrize("field", ["slope", "beta", "residuals", "reference"])
def test_volume_fit_choice_does_not_hide_inconsistent_results(tmp_path, method, field):
    path, _ = _volume_estimator_fixture(tmp_path, "wls")
    manifest = _read(path)
    settings = _read(tmp_path / "settings.json")
    settings["volume_fit"]["method"] = method
    results = manifest["results"]
    if field == "slope":
        results["volume_fit"]["slope_A3_per_K"] *= 2
    elif field == "beta":
        results["beta_volume"] *= 2
    elif field == "residuals":
        results["volume_fit"]["residuals_A3"][1] += 1
    else:
        results["reference_volume_A3"] *= 2
    _write(tmp_path / "settings.json", settings)
    _write(path, manifest)
    result = _score(path)
    assert _check(result, "interval_expansion_fit")["status"] == "failed"


def test_other_declared_volume_line_estimator_gets_review(tmp_path):
    path, _ = _volume_estimator_fixture(tmp_path, "wls")
    settings = _read(tmp_path / "settings.json")
    settings["volume_fit"] = {
        "method": "custom_linear_regression",
        "objective": "minimize the sum of squared residuals weighted by [1, 4, 9]",
    }
    _write(tmp_path / "settings.json", settings)
    result = _score(path)
    assert result.score is None
    assert _check(result, "interval_expansion_fit")["status"] == "unverified"
    assert all(
        check["status"] == "passed"
        for check in result.checks
        if check["points"] and check["name"] != "interval_expansion_fit"
    )
