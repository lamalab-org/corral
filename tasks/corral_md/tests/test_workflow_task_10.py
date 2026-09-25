"""Read-only heat-capacity grading with synthetic data and adversarial changes."""

import copy
import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1 import evaluate as evaluate_level1
from corral_md.workflow_scoring.task_10 import evaluate
from scipy.stats import theilslopes


def _write(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def _read(path):
    return json.loads(path.read_text())


def _score(path):
    rubric = Rubric(10)
    evaluate(Evidence(path), rubric)
    return rubric


def _check(rubric, name):
    return next(item for item in rubric.checks if item["name"] == name)


def test_teacher_path_and_descriptive_frame_times_preserve_checks(submission):
    settings_path = submission.parent / "settings.json"
    settings = _read(settings_path)
    settings["model"] = "/workspace/models/teacher.model"
    _write(settings_path, settings)
    for role in ("initial_state", "boundary_states", "production_trajectory"):
        path = Evidence(submission).artifact(role)
        frames = _read(path)
        for frame in frames:
            frame["elapsed_time_fs"] = frame.pop("time_fs")
        _write(path, frames)
    r = _score(submission)
    assert _check(r, "recorded_nvt_model_and_initialization")["status"] == "passed"
    assert _check(r, "eight_stage_cycle_and_cumulative_time")["status"] == "passed"
    assert _check(r, "boundary_position_momentum_continuity")["status"] == "passed"


def test_level1_accepts_energyless_initial_and_one_final_boundary(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(10)
    momenta = rng.normal(size=(108, 3))
    dof = 324
    momenta *= np.sqrt(
        300 * dof * units.kB / np.sum(momenta**2 / atoms.get_masses()[:, None])
    )
    frame = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": [True] * 3,
        "masses": atoms.get_masses().tolist(),
        "momenta": momenta.tolist(),
        "time_fs": 0.0,
    }
    production = [
        {
            **frame,
            "positions": (atoms.positions + fraction * momenta * 0.001).tolist(),
            "time_fs": time,
            "energy": -300.0,
        }
        for fraction, time in ((1, 1000.0), (2, 2000.0))
    ]
    boundary = {**production[-1]}
    boundary.pop("energy")
    settings = {
        "model": "/workspace/models/teacher.model",
        "model_settings": {},
        "md": {
            "temperature_dof": dof,
            "thermostat": "Langevin",
            "timestep_fs": 1.0,
            "random_seed": 10,
            "initial_temperature_K": 300,
            "velocity_initializations": 1,
            "manual_velocity_resets": 0,
            "ensemble": "NVT (fixed cell)",
        },
        "stages": [
            {
                "id": "300K",
                "target_temperature_K": 300,
                "start_time_fs": 0,
                "end_time_fs": 2000,
                "production_start_time_fs": 1000,
                "production_end_time_fs": 2000,
            }
        ],
    }
    manifest = {
        "artifacts": {
            "initial_state": _write(tmp_path / "initial.json", [frame]),
            "boundary_states": _write(tmp_path / "boundary.json", [boundary]),
            "production_trajectory": _write(tmp_path / "production.json", production),
        },
        "settings": _write(tmp_path / "settings.json", settings),
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)

    rubric = Rubric(10, fail_fast=False)
    evaluate_level1(Evidence(path), rubric, 10)
    for name in (
        "initial_fcc_and_temperature",
        "recorded_nvt_model_and_initialization",
        "fixed_cell_and_ordered_atoms",
        "initial_stage_timing_and_continuity",
    ):
        assert _check(rubric, name)["status"] == "passed", rubric.checks

    # The single stage and recorded stride identify the untimed trajectory.
    settings["md"]["trajectory_sample_steps"] = 500
    _write(tmp_path / "settings.json", settings)
    frame.pop("time_fs")
    boundary.pop("time_fs")
    for saved in production:
        saved.pop("time_fs")
    _write(tmp_path / "initial.json", [frame])
    _write(tmp_path / "boundary.json", [boundary])
    _write(tmp_path / "production.json", production)
    masses = atoms.get_masses()[:, None]
    directional = np.sum(momenta**2 / masses, axis=0) / ((dof / 3) * units.kB)
    kinetic = np.sum(momenta**2 / masses) / 2
    trace_row = {
        "stage": "300K",
        "temperature_K": float(directional.mean()),
        **{
            f"temperature_{axis}_K": float(value)
            for axis, value in zip("xyz", directional)
        },
        "total_energy_eV": float(-300 + kinetic),
    }
    trace = [{**trace_row, "time_fs": time} for time in (0, 1500, 2000)]
    manifest["artifacts"]["thermal_trace"] = _write(tmp_path / "trace.json", trace)
    manifest["results"] = {
        "stages": {
            "300K": {
                "temperature_mean_K": float(directional.mean()),
                "directional_temperature_mean_K": directional.tolist(),
                "energy_mean_eV_per_atom": float((-300 + kinetic) / 108),
            }
        }
    }
    _write(path, manifest)
    rubric = Rubric(10, fail_fast=True)
    evaluate_level1(Evidence(path), rubric, 10)
    assert _check(rubric, "thermal_trace_and_reported_means")["status"] == "passed"
    assert _check(rubric, "production_temperature_sanity")["status"] == "passed"


def _thermal(frame):
    p = np.asarray(frame["momenta"])
    mass = np.asarray(frame["masses"])
    ke_axes = (p**2 / mass[:, None]).sum(axis=0) / 2
    temperatures = 2 * ke_axes / (107 * units.kB)
    return np.r_[
        temperatures.mean(), temperatures, (frame["energy"] + ke_axes.sum()) / 108
    ]


def _statistics(data, block_size=4):
    nblocks = len(data) // block_size
    block_means = np.array(
        [part.mean(axis=0) for part in np.split(data[: nblocks * block_size], nblocks)]
    )
    covariance = np.cov(block_means.T, ddof=1) / nblocks
    tau = []
    for column in data.T:
        centered = column - column.mean()
        autocorrelation = np.correlate(centered, centered, mode="full")[
            len(column) - 1 :
        ]
        if autocorrelation[0] < 1e-24:
            tau.append(0.5)
            continue
        normalized = autocorrelation[1 : len(column) // 2 + 1] / autocorrelation[0]
        negative = np.flatnonzero(normalized <= 0)
        stop = negative[0] if len(negative) else len(normalized)
        tau.append(0.5 + normalized[:stop].sum())
    return {
        "mean": data.mean(axis=0),
        "sem": np.sqrt(np.diag(covariance)),
        "covariance": covariance,
        "tau": tau,
        "count": nblocks,
    }


def _fit_reference(means, covariance, weights=None):
    x, y = means[:, 0], means[:, 4]
    weights = np.ones(5) if weights is None else weights
    design = np.column_stack([x, np.ones(5)])
    solution = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None
    )[0]
    # Finite-difference uncertainty oracle independent of production gradients.
    gradient = np.empty((5, 2))
    for i in range(5):
        for j, step in ((0, 1e-3), (1, 1e-6)):
            changed_x, changed_y = x.copy(), y.copy()
            if j == 0:
                changed_x[i] += step
            else:
                changed_y[i] += step
            changed_design = np.column_stack([changed_x, np.ones(5)])
            changed_slope = np.linalg.lstsq(
                changed_design * np.sqrt(weights[:, None]),
                changed_y * np.sqrt(weights),
                rcond=None,
            )[0][0]
            gradient[i, j] = (changed_slope - solution[0]) / step
    variance = sum(
        gradient[i] @ covariance[i][np.ix_([0, 4], [0, 4])] @ gradient[i]
        for i in range(5)
    )
    return solution[0], solution[1], np.sqrt(max(0, variance))


def _refresh_results(path, method="ols"):
    manifest = _read(path)
    settings = _read(path.parent / "settings.json")
    settings["fit"]["method"] = method
    frames = _read(path.parent / "production.json")
    trace = _read(path.parent / "trace.json")
    arrays, summaries, stage_results = [], [], {}
    intervals = {}
    alternative_summaries = []
    for config in settings["stages"]:
        data = np.array(
            [_thermal(frame) for frame in frames if frame["stage"] == config["id"]]
        )
        arrays.append(data)
        summary = _statistics(data, config["block_size"])
        summaries.append(summary)
        eq = [
            row
            for row in trace
            if row["stage"] == config["id"]
            and row["time_fs"] < config["production_start_time_fs"]
        ]
        t = np.array([row["temperature_K"] for row in eq])
        time = np.array([row["time_fs"] for row in eq]) / 1000
        half = len(t) // 2
        stage_results[config["id"]] = {
            "temperature_mean_K": summary["mean"][0],
            "temperature_std_K": data[:, 0].std(),
            "directional_temperature_mean_K": summary["mean"][1:4].tolist(),
            "energy_mean_eV_per_atom": summary["mean"][4],
            "temperature_sem_K": summary["sem"][0],
            "directional_temperature_sem_K": summary["sem"][1:4].tolist(),
            "energy_sem_eV_per_atom": summary["sem"][4],
            "correlation_time_samples": summary["tau"],
            "n_blocks": summary["count"],
            "equilibration": {
                "temperature_first_half_mean_K": t[:half].mean(),
                "temperature_second_half_mean_K": t[half:].mean(),
                "temperature_drift_K_per_ps": np.polyfit(time - time[0], t, 1)[0],
            },
        }
        alternative = _statistics(data[4:])
        alternative_summaries.append(alternative)
        intervals[config["id"]] = {
            "start": 4,
            "stop": len(data),
            "block_size": 4,
            "temperature_mean_K": alternative["mean"][0],
            "energy_mean_eV_per_atom": alternative["mean"][4],
            "temperature_sem_K": alternative["sem"][0],
            "energy_sem_eV_per_atom": alternative["sem"][4],
        }
    means = np.array([item["mean"] for item in summaries[:5]])
    covariance = [item["covariance"] for item in summaries[:5]]
    weights = (
        None
        if method == "ols"
        else 1 / np.array([item["sem"][4] for item in summaries[:5]]) ** 2
    )
    if weights is not None:
        settings["fit"]["weights"] = weights.tolist()
    slope, intercept, error = _fit_reference(means, covariance, weights)
    alt_weights = (
        None
        if method == "ols"
        else 1 / np.array([item["sem"][4] for item in alternative_summaries[:5]]) ** 2
    )
    alt_slope = _fit_reference(
        np.array([item["mean"] for item in alternative_summaries[:5]]),
        [item["covariance"] for item in alternative_summaries[:5]],
        alt_weights,
    )[0]
    _write(
        path.parent / "sensitivity.json",
        {"comparisons": [{"intervals": intervals, "slope_eV_per_atom_K": alt_slope}]},
    )
    energy_difference = summaries[7]["mean"][4] - summaries[0]["mean"][4]
    energy_error = np.hypot(summaries[7]["sem"][4], summaries[0]["sem"][4])
    energy_agrees = bool(abs(energy_difference) <= 2 * energy_error)
    manifest["results"] = {
        "stages": stage_results,
        "heat_capacity": {
            "fit_stage_ids": [item["id"] for item in settings["stages"][:5]],
            "slope_eV_per_atom_K": slope,
            "intercept_eV_per_atom": intercept,
            "standard_error_eV_per_atom_K": error,
            "value_kB_per_atom": slope / units.kB,
            "standard_error_kB_per_atom": error / units.kB,
            "units": "eV/(atom K)",
            "scaled_units": "k_B/atom",
        },
        "high_temperature_residuals": {
            settings["stages"][i]["id"]: summaries[i]["mean"][4]
            - (intercept + slope * summaries[i]["mean"][0])
            for i in (5, 6)
        },
        "return_state": {
            "energy_difference_eV_per_atom": energy_difference,
            "energy_difference_sem_eV_per_atom": energy_error,
            "initial_structural_mean": 12,
            "return_structural_mean": 12,
            "structural_difference": 0,
            "structural_difference_sem": 0,
            "confidence_multiplier": 2,
            "energy_agrees": energy_agrees,
            "structure_agrees": True,
            "joint_agreement": energy_agrees,
            "interpretation": "Both comparisons are consistent with return at two standard errors; finite traces limit the conclusion.",
        },
    }
    _write(path.parent / "settings.json", settings)
    _write(path, manifest)


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(120)
    p_shape = rng.normal(size=(108, 3))
    p_shape -= p_shape.mean(axis=0)
    mass = atoms.get_masses()
    p_shape /= np.sqrt(np.sum(p_shape**2 / mass[:, None], axis=0))
    position_shape = rng.normal(size=(108, 3))
    position_shape -= position_shape.mean(axis=0)
    ids = ["300_initial", "400", "500", "600", "700", "800", "900", "300_return"]
    targets = [300, 400, 500, 600, 700, 800, 900, 300]
    offsets = np.array([0.3, -0.2, 0.4, -0.5, 0.1, -0.4, 0.2, 0.1])
    energy_offsets = np.array([0.7, -0.1, -0.3, 0.9, -0.4, 0.2, -0.5, 0.1])

    def frame(stage, time, directional, energy, amplitude):
        momentum = p_shape * np.sqrt(107 * units.kB * np.asarray(directional))
        ke = np.sum(momentum**2 / mass[:, None]) / 2
        return {
            "symbols": ["Al"] * 108,
            "masses": mass.tolist(),
            "positions": (atoms.positions + amplitude * position_shape).tolist(),
            "momenta": momentum.tolist(),
            "cell": atoms.cell.array.tolist(),
            "pbc": [True] * 3,
            "energy": energy * 108 - ke,
            "stage": stage,
            "time_fs": time,
        }

    def trace_row(saved):
        thermal = _thermal(saved)
        ke = np.sum(np.asarray(saved["momenta"]) ** 2 / mass[:, None]) / 2
        return {
            "stage": saved["stage"],
            "time_fs": saved["time_fs"],
            "temperature_K": thermal[0],
            "temperature_x_K": thermal[1],
            "temperature_y_K": thermal[2],
            "temperature_z_K": thermal[3],
            "potential_energy_eV": saved["energy"],
            "kinetic_energy_eV": ke,
            "total_energy_eV": saved["energy"] + ke,
        }

    initial = frame(ids[0], 0, [300] * 3, -3.6 + 3 * units.kB * 300, 0)
    previous = initial
    stages, production, boundary, trace = [], [], [], []
    for i, (stage, target) in enumerate(zip(ids, targets, strict=True)):
        start_time = 51 * i
        start = copy.deepcopy(previous)
        start["stage"] = stage
        midpoint = frame(
            stage, start_time + 10, [target] * 3, -3.6 + 3 * units.kB * target, 0.01
        )
        trace.extend([trace_row(start), trace_row(midpoint)])
        current = []
        for j in range(32):
            temperature = target + 4 * (-1) ** j + offsets[j // 4]
            directional = temperature + np.array([1.2, -0.9, -0.3]) * (-1) ** j
            energy = (
                -3.6
                + 3 * units.kB * temperature
                + 0.0002 * (-1) ** j
                + 0.00003 * energy_offsets[j // 4]
            )
            if i in (5, 6):
                energy += 0.001 * (i - 4)
            state = frame(
                stage,
                start_time + 20 + j,
                directional,
                energy,
                0.01 + 0.001 * (-1) ** j,
            )
            current.append(state)
            trace.append(trace_row(state))
        boundary.extend([start, current[-1]])
        production.extend(current)
        previous = current[-1]
        stages.append(
            {
                "id": stage,
                "target_temperature_K": target,
                "start_time_fs": start_time,
                "end_time_fs": start_time + 51,
                "production_start_time_fs": start_time + 20,
                "production_end_time_fs": start_time + 51,
                "block_size": 4,
            }
        )
    settings = {
        "md": {
            "thermostat": "Langevin",
            "timestep_fs": 1,
            "random_seed": 120,
            "initial_temperature_K": 300,
            "velocity_initializations": 1,
            "manual_velocity_resets": 0,
            "ensemble": "NVT",
            "temperature_dof": 321,
        },
        "model": "MACE-MP-0",
        "model_settings": {"dtype": "float64"},
        "stages": stages,
        "uncertainty": {"method": "block_sem"},
        "fit": {"method": "ols", "uncertainty": "block_delta"},
        "structure": {"method": "coordination", "cutoff_A": 3.5},
    }
    script = tmp_path / "analysis.py"
    script.write_text("raise RuntimeError('Submitted code must not run')\n")
    manifest = {
        "results": {},
        "settings": _write(tmp_path / "settings.json", settings),
        "artifacts": {
            "initial_state": _write(tmp_path / "initial.json", [initial]),
            "boundary_states": _write(tmp_path / "boundaries.json", boundary),
            "production_trajectory": _write(tmp_path / "production.json", production),
            "thermal_trace": _write(tmp_path / "trace.json", trace),
            "sensitivity": str(tmp_path / "sensitivity.json"),
        },
        "scripts": [str(script)],
        "report": _write(
            tmp_path / "report.json",
            {"analysis": "Synthetic evidence for grader verification."},
        ),
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    _refresh_results(path)
    return path


def test_complete_saved_evidence_earns_all_task_points(submission):
    r = _score(submission)
    assert sum(item["points"] for item in r.checks) == 90
    assert sum(item["earned"] for item in r.checks) == 90, r.checks
    assert _check(r, "execution_provenance")["status"] == "unverified"


def test_return_comparison_needs_no_confidence_flags_or_interpretation(submission):
    manifest = _read(submission)
    manifest.pop("report")
    manifest["results"]["return_state"].pop("interpretation")
    _write(submission, manifest)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    manifest["results"]["return_state"].pop("joint_agreement")
    _write(submission, manifest)
    assert _score(submission).score == pytest.approx(0.9)


def test_empty_evidence_retains_all_possible_points():
    r = Rubric(10)
    evaluate(Evidence({}), r)
    assert sum(item["points"] for item in r.checks) == 90
    assert sum(item["earned"] for item in r.checks) == 0


@pytest.mark.parametrize(
    ("name", "field", "value", "check"),
    [
        (
            "heat_capacity",
            "slope_eV_per_atom_K",
            10,
            "measured_temperature_heat_capacity_fit",
        ),
        (
            "heat_capacity",
            "value_kB_per_atom",
            1000,
            "heat_capacity_units_and_kB_conversion",
        ),
        (
            "return_state",
            "energy_difference_eV_per_atom",
            20,
            "return_minus_initial_energy",
        ),
        (
            "return_state",
            "structural_difference",
            20,
            "consistent_structural_return_comparison",
        ),
    ],
)
def test_tampered_claims_lose_specific_credit(submission, name, field, value, check):
    manifest = _read(submission)
    manifest["results"][name][field] = value
    _write(submission, manifest)
    r = _score(submission)
    assert _check(r, check)["status"] == "failed"
    assert _check(r, "initial_fcc_and_temperature")["status"] == "passed"


def test_stage_boundary_momentum_reset_is_detected(submission):
    path = submission.parent / "boundaries.json"
    data = _read(path)
    data[2]["momenta"][0][0] += 1
    _write(path, data)
    r = _score(submission)
    assert _check(r, "boundary_position_momentum_continuity")["status"] == "failed"
    assert _check(r, "measured_temperature_heat_capacity_fit")["status"] == "passed"
    for name in [
        "return_minus_initial_energy",
        "consistent_structural_return_comparison",
    ]:
        assert _check(r, name)["status"] == "failed"
        assert "prerequisite" in _check(r, name)["detail"]


def test_boundary_potential_energy_must_match(submission):
    path = submission.parent / "boundaries.json"
    data = _read(path)
    data[4]["energy"] += 2
    _write(path, data)
    assert (
        _check(_score(submission), "boundary_position_momentum_continuity")["status"]
        == "failed"
    )


def test_variable_volume_is_detected(submission):
    path = submission.parent / "production.json"
    data = _read(path)
    data[12]["cell"][0][0] += 0.5
    _write(path, data)
    r = _score(submission)
    assert _check(r, "fixed_cell_and_ordered_atoms")["status"] == "failed"
    for name in [
        "measured_temperature_heat_capacity_fit",
        "heat_capacity_units_and_kB_conversion",
        "high_temperature_energy_residuals",
        "consistent_structural_return_comparison",
    ]:
        assert _check(r, name)["status"] == "failed"
    assert _check(r, "total_energy_per_atom_means")["status"] == "passed"


def test_return_stage_cannot_reuse_initial_identifier(submission):
    path = submission.parent / "settings.json"
    data = _read(path)
    data["stages"][-1]["id"] = data["stages"][0]["id"]
    _write(path, data)
    assert (
        _check(_score(submission), "eight_stage_cycle_and_cumulative_time")["status"]
        == "failed"
    )


def test_reset_elapsed_clock_is_detected(submission):
    path = submission.parent / "settings.json"
    data = _read(path)
    data["stages"][1]["start_time_fs"] = 0
    _write(path, data)
    assert (
        _check(_score(submission), "eight_stage_cycle_and_cumulative_time")["status"]
        == "failed"
    )


def test_directional_temperature_is_recomputed(submission):
    path = submission.parent / "trace.json"
    data = _read(path)
    data[10]["temperature_x_K"] += 1
    _write(path, data)
    assert (
        _check(_score(submission), "thermal_trace_from_saved_momenta_and_energy")[
            "status"
        ]
        == "failed"
    )


def test_off_target_temperature_requires_review_with_truthful_claims(submission):
    path = submission.parent / "production.json"
    data = _read(path)
    for frame in data:
        frame["momenta"] = (np.asarray(frame["momenta"]) * 0.1).tolist()
    _write(path, data)
    _refresh_results(submission)
    r = _score(submission)
    assert _check(r, "total_and_directional_temperature_means")["status"] == "passed"
    assert _check(r, "production_target_temperature_sanity")["status"] == "unverified"


def test_total_energy_requires_kinetic_contribution(submission):
    manifest = _read(submission)
    frames = _read(submission.parent / "production.json")
    manifest["results"]["stages"]["300_initial"]["energy_mean_eV_per_atom"] = np.mean(
        [frame["energy"] / 108 for frame in frames[:32]]
    )
    _write(submission, manifest)
    assert (
        _check(_score(submission), "total_energy_per_atom_means")["status"] == "failed"
    )


def test_residual_sign_matters(submission):
    manifest = _read(submission)
    manifest["results"]["high_temperature_residuals"]["800"] *= -1
    _write(submission, manifest)
    assert (
        _check(_score(submission), "high_temperature_energy_residuals")["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "check"),
    [
        ("fit", "method", "custom", "measured_temperature_heat_capacity_fit"),
        ("structure", "method", "custom", "consistent_structural_return_comparison"),
    ],
)
def test_unsupported_method_is_unverified(submission, section, key, value, check):
    path = submission.parent / "settings.json"
    data = _read(path)
    data[section][key] = value
    _write(path, data)
    r = _score(submission)
    assert _check(r, check)["status"] == "unverified"
    assert r.score is None
    assert _check(r, "total_and_directional_temperature_means")["status"] == "passed"
    assert _check(r, "heat_capacity_units_and_kB_conversion")["status"] == "passed"
    assert _check(r, "high_temperature_energy_residuals")["status"] == "passed"


def test_weighted_fit_with_declared_weights(submission):
    _refresh_results(submission, method="wls")
    r = _score(submission)
    assert sum(item["earned"] for item in r.checks) == 90, r.checks


def test_does_not_execute_calculator_or_modify_artifacts(submission, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Calculator execution is forbidden")

    monkeypatch.setattr(SinglePointCalculator, "get_potential_energy", forbidden)
    before = {path.name: path.read_bytes() for path in submission.parent.iterdir()}
    r = _score(submission)
    assert sum(item["earned"] for item in r.checks) == 90
    after = {path.name: path.read_bytes() for path in submission.parent.iterdir()}
    assert before == after


def test_missing_production_artifact_preserves_setup_credit(submission):
    (submission.parent / "production.json").unlink()
    r = _score(submission)
    assert _check(r, "initial_fcc_and_temperature")["status"] == "passed"
    assert _check(r, "recorded_nvt_model_and_initialization")["status"] == "passed"
    assert sum(item["points"] for item in r.checks) == 90


@pytest.mark.parametrize("cutoff", [-1, 0.01])
def test_structural_cutoff_must_be_physically_defined(submission, cutoff):
    path = submission.parent / "settings.json"
    data = _read(path)
    data["structure"]["cutoff_A"] = cutoff
    _write(path, data)
    assert (
        _check(_score(submission), "consistent_structural_return_comparison")["status"]
        == "failed"
    )


def test_geometry_allows_rigid_global_translation(submission):
    for filename in ["initial.json", "boundaries.json", "production.json"]:
        path = submission.parent / filename
        frames = _read(path)
        for frame in frames:
            frame["positions"] = (
                np.asarray(frame["positions"]) + np.array([0.31, -0.8, 0.2])
            ).tolist()
        _write(path, frames)
    r = _score(submission)
    assert sum(item["earned"] for item in r.checks) == 90, r.checks


def test_rms_structural_metric_is_reproduced(submission):
    path = submission.parent / "settings.json"
    settings = _read(path)
    settings["structure"] = {"method": "rms_displacement", "remove_translation": True}
    _write(path, settings)
    manifest = _read(submission)
    initial = np.asarray(_read(submission.parent / "initial.json")[0]["positions"])
    frames = _read(submission.parent / "production.json")
    mean_values, errors = [], []
    for stage in ["300_initial", "300_return"]:
        values = []
        for frame in frames:
            if frame["stage"] != stage:
                continue
            delta = np.asarray(frame["positions"]) - initial
            delta -= delta.mean(axis=0)
            values.append(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
        means = np.array(values).reshape(-1, 4).mean(axis=1)
        mean_values.append(np.mean(values))
        errors.append(means.std(ddof=1) / np.sqrt(len(means)))
    manifest["results"]["return_state"].update(
        {
            "initial_structural_mean": mean_values[0],
            "return_structural_mean": mean_values[1],
            "structural_difference": mean_values[1] - mean_values[0],
            "structural_difference_sem": float(np.hypot(*errors)),
        }
    )
    _write(submission, manifest)
    assert (
        _check(_score(submission), "consistent_structural_return_comparison")["status"]
        == "passed"
    )


def test_directional_anisotropy_requires_physical_review(submission):
    path = submission.parent / "production.json"
    data = _read(path)
    for frame in data:
        momentum = np.asarray(frame["momenta"])
        total_squared = np.sum(momentum**2)
        momentum[:, 1:] = 0
        momentum[:, 0] *= np.sqrt(total_squared / np.sum(momentum[:, 0] ** 2))
        frame["momenta"] = momentum.tolist()
    _write(path, data)
    _refresh_results(submission)
    r = _score(submission)
    assert _check(r, "total_and_directional_temperature_means")["status"] == "passed"
    assert _check(r, "production_target_temperature_sanity")["status"] == "unverified"


def test_truthful_nonreturn_retains_scientific_credit(submission):
    shift = 0.05 * 108
    path = submission.parent / "production.json"
    data = _read(path)
    for frame in data:
        if frame["stage"] == "300_return":
            frame["energy"] += shift
    _write(path, data)
    path = submission.parent / "boundaries.json"
    data = _read(path)
    data[-1]["energy"] += shift
    _write(path, data)
    path = submission.parent / "trace.json"
    data = _read(path)
    for row in data:
        if row["stage"] == "300_return" and row["time_fs"] >= 377:
            row["potential_energy_eV"] += shift
            row["total_energy_eV"] += shift
    _write(path, data)
    _refresh_results(submission)
    result = _read(submission)["results"]["return_state"]
    assert result["energy_difference_eV_per_atom"] == pytest.approx(0.05)
    assert result["joint_agreement"] is False
    r = _score(submission)
    assert sum(item["earned"] for item in r.checks) == 90, r.checks


def test_alternative_fit_and_structural_metric_receive_full_credit(submission):
    settings = _read(submission.parent / "settings.json")
    manifest = _read(submission)
    frames = _read(submission.parent / "production.json")
    settings["fit"]["method"] = "theil_sen"
    settings["structure"] = {"method": "nearest_neighbor_distance"}
    means = np.array(
        [
            np.mean([_thermal(f) for f in frames if f["stage"] == stage["id"]], axis=0)
            for stage in settings["stages"]
        ]
    )
    fit = theilslopes(means[:5, 4], means[:5, 0], method="joint")
    manifest["results"]["heat_capacity"].update(
        slope_eV_per_atom_K=fit.slope,
        intercept_eV_per_atom=fit.intercept,
        value_kB_per_atom=fit.slope / units.kB,
    )
    for i in (5, 6):
        manifest["results"]["high_temperature_residuals"][
            settings["stages"][i]["id"]
        ] = means[i, 4] - fit.intercept - fit.slope * means[i, 0]
    structural = []
    for stage in (settings["stages"][0], settings["stages"][7]):
        samples = []
        for frame in frames:
            if frame["stage"] == stage["id"]:
                atoms = Atoms(
                    "Al108", positions=frame["positions"], cell=frame["cell"], pbc=True
                )
                distances = atoms.get_all_distances(mic=True)
                np.fill_diagonal(distances, np.inf)
                samples.append(distances.min(axis=1).mean())
        structural.append(np.mean(samples))
    manifest["results"]["return_state"].update(
        initial_structural_mean=structural[0],
        return_structural_mean=structural[1],
        structural_difference=structural[1] - structural[0],
    )
    for values in manifest["results"]["stages"].values():
        values.pop("equilibration")
        values.pop("temperature_std_K")
    _write(submission.parent / "settings.json", settings)
    _write(submission, manifest)
    assert _score(submission).score == pytest.approx(0.9)
    manifest["results"]["return_state"]["initial_structural_mean"] += 1
    _write(submission, manifest)
    assert (
        _check(_score(submission), "consistent_structural_return_comparison")["status"]
        == "failed"
    )


def test_unknown_structure_cannot_hide_inconsistent_difference(submission):
    settings = _read(submission.parent / "settings.json")
    settings["structure"]["method"] = "bond_order"
    _write(submission.parent / "settings.json", settings)
    manifest = _read(submission)
    manifest["results"]["return_state"]["structural_difference"] = 100
    _write(submission, manifest)
    assert (
        _check(_score(submission), "consistent_structural_return_comparison")["status"]
        == "failed"
    )


def test_fit_stage_identity_is_independent_of_report_order(submission):
    manifest = _read(submission)
    manifest["results"]["heat_capacity"]["fit_stage_ids"].reverse()
    _write(submission, manifest)
    assert _score(submission).score == pytest.approx(0.9)


def test_inconsistent_supported_fit_gates_dependent_arithmetic(submission):
    manifest = _read(submission)
    claim = manifest["results"]["heat_capacity"]
    claim["slope_eV_per_atom_K"] *= 2
    claim["value_kB_per_atom"] *= 2
    for sid in ("800", "900"):
        data = manifest["results"]["stages"][sid]
        manifest["results"]["high_temperature_residuals"][sid] = (
            data["energy_mean_eV_per_atom"]
            - claim["intercept_eV_per_atom"]
            - claim["slope_eV_per_atom_K"] * data["temperature_mean_K"]
        )
    _write(submission, manifest)
    rubric = _score(submission)
    for name in (
        "measured_temperature_heat_capacity_fit",
        "heat_capacity_units_and_kB_conversion",
        "high_temperature_energy_residuals",
    ):
        assert _check(rubric, name)["status"] == "failed"


def test_three_frame_production_is_not_a_workflow_violation(submission):
    manifest = _read(submission)
    settings = _read(submission.parent / "settings.json")
    data = _read(submission.parent / "production.json")
    retained, means = [], []
    for stage in settings["stages"]:
        frames = [f for f in data if f["stage"] == stage["id"]]
        frames = [frames[0], frames[len(frames) // 2], frames[-1]]
        retained.extend(frames)
        values = np.array([_thermal(f) for f in frames])
        mean = values.mean(axis=0)
        means.append(mean)
        manifest["results"]["stages"][stage["id"]].update(
            temperature_mean_K=mean[0],
            directional_temperature_mean_K=mean[1:4].tolist(),
            temperature_std_K=values[:, 0].std(),
            energy_mean_eV_per_atom=mean[4],
        )
    means = np.asarray(means)
    slope, intercept = np.polyfit(means[:5, 0], means[:5, 4], 1)
    manifest["results"]["heat_capacity"].update(
        slope_eV_per_atom_K=slope,
        intercept_eV_per_atom=intercept,
        value_kB_per_atom=slope / units.kB,
    )
    for index in (5, 6):
        manifest["results"]["high_temperature_residuals"][
            settings["stages"][index]["id"]
        ] = means[index, 4] - intercept - slope * means[index, 0]
    manifest["results"]["return_state"]["energy_difference_eV_per_atom"] = (
        means[7, 4] - means[0, 4]
    )
    _write(submission.parent / "production.json", retained)
    _write(submission, manifest)
    assert _score(submission).score == pytest.approx(0.9)
