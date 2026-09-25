"""Synthetic saved-evidence fixtures: these tests never simulate a system."""

import json

import numpy as np
import pandas as pd
import pytest
from ase import units
from ase.build import bulk
from ase.io import read, write
from corral_md.workflow_scoring import task_1 as scoring
from corral_md.workflow_scoring.common import Evidence, Rubric, UnsupportedEvidence


def _dump(path, value):
    path.write_text(json.dumps(value))


def test_lammps_data_and_cif_are_readable_but_binary_restart_needs_review(tmp_path):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    write(tmp_path / "reference.cif", atoms)
    write(tmp_path / "prepared.data", atoms, format="lammps-data")
    (tmp_path / "final.restart").write_bytes(b"binary restart")
    manifest = tmp_path / "manifest.json"
    _dump(
        manifest,
        {
            "artifacts": {
                "reference_cell": "reference.cif",
                "prepared_state": "prepared.data",
                "heating_end_state": "final.restart",
            }
        },
    )
    evidence = Evidence(manifest)
    assert len(evidence.trajectory("reference_cell")) == 1
    assert len(evidence.trajectory("prepared_state")) == 1
    with pytest.raises(UnsupportedEvidence, match="binary restart"):
        evidence.trajectory("heating_end_state")


def _correlation_time(t, p):
    series = np.mean(np.sum(np.diff(p, axis=0) ** 2, axis=-1), axis=1)
    series -= series.mean()
    denominator = np.dot(series, series)
    positive = 0.0
    if denominator > 0:
        for lag in range(1, len(series) // 2 + 1):
            rho = np.dot(series[:-lag], series[lag:]) / denominator
            if rho <= 0:
                break
            positive += rho
    return np.mean(np.diff(t)) * (0.5 + positive)


@pytest.fixture
def evidence(tmp_path):
    rng = np.random.default_rng(782)
    reference = bulk("Si", "diamond", a=5.43, cubic=True)
    initial = reference.repeat((3, 3, 3))
    initial.set_velocities(rng.normal(size=(216, 3)))
    initial.set_velocities(
        initial.get_velocities() * np.sqrt(300 / initial.get_temperature())
    )
    time = np.linspace(1000, 1500, 251)
    cell = np.tile(initial.cell.array, (len(time), 1, 1))
    p = np.cumsum(rng.normal(scale=np.sqrt(8), size=(len(time), 216, 3)), axis=0)
    p += rng.random((1, 216, 3)) @ initial.cell.array
    v = rng.normal(size=p.shape)
    temperature = np.sum(
        initial.get_masses()[None, :, None] * (v / (units.fs * 1000)) ** 2, axis=(1, 2)
    ) / (3 * 216 * units.kB)
    v *= np.sqrt(2500 / temperature)[:, None, None]
    artifacts = {}
    for name, atoms in (("reference_cell", reference), ("prepared_state", initial)):
        write(tmp_path / f"{name}.extxyz", atoms)
        artifacts[name] = f"{name}.extxyz"
    for name, i in (("heating_end_state", 0), ("final_state", -1)):
        atoms = initial.copy()
        atoms.positions = p[i]
        atoms.set_velocities(v[i] / (units.fs * 1000))
        write(tmp_path / f"{name}.extxyz", atoms)
        artifacts[name] = f"{name}.extxyz"
    data = {
        "times_ps": time.tolist(),
        "unwrapped_positions_A": p.tolist(),
        "cells_A": cell.tolist(),
        "velocities_A_ps": v.tolist(),
        "atom_ids": list(range(1, 217)),
        "species": ["Si"] * 216,
        "pbc": [True] * 3,
    }
    _dump(tmp_path / "data.json", data)
    artifacts["diffusion_data"] = "data.json"
    density = initial.get_masses().sum() * 1.66053906660 / initial.get_volume()
    tt = np.r_[np.arange(0, 1000, 100), time]
    target = np.minimum(2500, 300 + 2.2 * tt)
    thermo = pd.DataFrame(
        {
            "step": (tt / 0.001).astype(int),
            "time_ps": tt,
            "target_temperature_K": target,
            "temperature_K": target,
            "pressure_bar": np.zeros(len(tt)),
            "density_g_cm3": np.full(len(tt), density),
        }
    )
    thermo.to_csv(tmp_path / "thermo.csv", index=False)
    artifacts["thermo"] = "thermo.csv"
    (tmp_path / "input.lmp").write_text("""units metal
atom_style full
read_data silicon.data
pair_style sw
pair_coeff * * silicon.sw Si
timestep 0.001
velocity all create 300 4567 mom yes
fix heat all npt temp 300 2500 0.1 iso 0 0 1
run 1000000
unfix heat
fix production all npt temp 2500 2500 0.1 iso 0 0 1
run 500000
""")
    artifacts["lammps_input"] = "input.lmp"
    log = "Step Temp Press Density\n" + "\n".join(
        f"{r.step} {r.temperature_K} {r.pressure_bar} {r.density_g_cm3}"
        for r in thermo.itertuples()
    )
    (tmp_path / "log.lammps").write_text(log)
    artifacts["lammps_log"] = "log.lammps"
    spec = {
        "estimator": "einstein_msd",
        "lag_steps": list(range(1, 21)),
        "origin_indices": "all",
        "remove_com_drift": True,
        "cell_motion": "lab",
        "fit_interval_ps": [10, 40],
        "fit_intercept": True,
    }
    estimate = scoring._msd(time, p, cell, spec)
    spec.update(lag_times_ps=estimate["x"].tolist(), msd_A2=estimate["y"].tolist())
    x, y = estimate["x"][estimate["selected"]], estimate["y"][estimate["selected"]]
    liquid = {"frame_indices": [0, 250], "neighbor_cutoff_A": 3.0}
    adjacency = []
    for i in (0, 250):
        d = np.linalg.norm(
            scoring._minimum_image(p[i, :, None] - p[i, None, :], cell[i]), axis=-1
        )
        adjacency.append((d < 3) & ~np.eye(216, dtype=bool))
    liquid["neighbor_survival_fraction"] = float(
        np.count_nonzero(adjacency[0] & adjacency[1]) / np.count_nonzero(adjacency[0])
    )
    sensitivity = []
    for interval in ([8, 30], [20, 40]):
        sensitivity.append(  # noqa: PERF401 - keep fixture records readable
            {
                "fit_interval_ps": interval,
                "diffusion_m2_s": scoring._msd(
                    time, p, cell, dict(spec, fit_interval_ps=interval)
                )["d"],
            }
        )
    windows = [[i, i + 100] for i in range(1000, 1500, 100)]
    block_spec = dict(spec, lag_steps=list(range(1, 11)), fit_interval_ps=[4, 20])
    values = []
    for lo, hi in windows:
        mask = (time >= lo) & (time <= hi)
        values.append(scoring._msd(time[mask], p[mask], cell[mask], block_spec)["d"])
    error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    analysis = {
        "msd": spec,
        "validation": {
            "equilibrium_window_ps": [1000, 1500],
            "temperature_mean_K": 2500,
            "density_mean_g_cm3": density,
            "temperature_half_difference_K": 0,
            "density_half_difference_g_cm3": 0,
            "pressure_mean_bar": 0,
            "pressure_half_difference_bar": 0,
        },
        "liquid": liquid,
        "diffusive": {
            "log_slope": float(np.polyfit(np.log(x), np.log(y), 1)[0]),
            "residual_fraction": float(
                np.sqrt(np.mean((y - estimate["predicted"]) ** 2)) / y.mean()
            ),
        },
        "sensitivity": sensitivity,
        "uncertainty": {
            "method": "block_sem",
            "block_windows_ps": windows,
            "lag_steps": block_spec["lag_steps"],
            "fit_interval_ps": block_spec["fit_interval_ps"],
            "uncertainty_m2_s": error,
            "block_diffusion_m2_s": values,
            "correlation_time_ps": _correlation_time(time, p),
        },
    }
    _dump(tmp_path / "analysis.json", analysis)
    artifacts["analysis"] = "analysis.json"
    _dump(tmp_path / "settings.json", {"timestep_ps": 0.001, "velocity_seed": 4567})
    manifest = {
        "results": {
            "diffusion_m2_s": estimate["d"],
            "diffusion_units": "m2/s",
            "uncertainty_m2_s": error,
        },
        "artifacts": artifacts,
        "settings": "settings.json",
    }
    _dump(tmp_path / "manifest.json", manifest)
    return tmp_path


def _score(path):
    rubric = Rubric(1)
    scoring.evaluate(Evidence(path / "manifest.json"), rubric)
    return rubric


def _by_name(rubric, name):
    return next(check for check in rubric.checks if check["name"] == name)


def test_complete_saved_data_passes(evidence):
    rubric = _score(evidence)
    assert sum(c["points"] for c in rubric.checks) == 90
    assert sum(c["earned"] for c in rubric.checks) == 90, rubric.checks
    assert _by_name(rubric, "execution_provenance")["status"] == "unverified"


def test_linked_npz_diffusion_arrays_are_read_and_cross_checked(evidence):
    manifest = json.loads((evidence / "manifest.json").read_text())
    data_path = evidence / manifest["artifacts"]["diffusion_data"]
    data = json.loads(data_path.read_text())
    arrays = {
        key: data.pop(key)
        for key in ("unwrapped_positions_A", "velocities_A_ps", "cells_A")
    }
    arrays["positions_unwrapped_A"] = arrays.pop("unwrapped_positions_A")
    archive = evidence / "trajectory_arrays.npz"
    np.savez(archive, **arrays)
    data.update(arrays_file=archive.name, positions_key="positions_unwrapped_A")
    manifest["artifacts"]["trajectory_arrays"] = archive.name
    _dump(data_path, data)
    _dump(evidence / "manifest.json", manifest)
    assert _by_name(_score(evidence), "production_data_integrity")["status"] == "passed"
    data["velocities_A_ps"] = (np.asarray(arrays["velocities_A_ps"]) + 1).tolist()
    _dump(data_path, data)
    failed = _by_name(_score(evidence), "production_data_integrity")
    assert failed["status"] == "failed"
    assert "disagree" in failed["detail"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_geometry",
        "discontinuity",
        "wrong_units",
        "wrong_diffusion",
        "forged_log",
    ],
)
def test_tampered_evidence_loses_relevant_credit(evidence, mutation):
    if mutation in {"wrong_geometry", "discontinuity"}:
        path = evidence / (
            "prepared_state.extxyz"
            if mutation == "wrong_geometry"
            else "heating_end_state.extxyz"
        )
        atoms = read(path)
        atoms.positions[0, 0] += 1
        write(path, atoms)
    elif mutation in {"wrong_units", "wrong_diffusion"}:
        path = evidence / "manifest.json"
        manifest = json.loads(path.read_text())
        if mutation == "wrong_units":
            manifest["results"]["diffusion_units"] = "A2/ps"
        else:
            manifest["results"]["diffusion_m2_s"] *= 100
        _dump(path, manifest)
    elif mutation == "forged_log":
        (evidence / "log.lammps").write_text("Step Temp Press Density\n0 300 0 999\n")
    elif mutation == "bad_uncertainty":
        path = evidence / "analysis.json"
        analysis = json.loads(path.read_text())
        analysis["uncertainty"]["block_diffusion_m2_s"][0] *= 3
        _dump(path, analysis)
    rubric = _score(evidence)
    expected = {
        "wrong_geometry": "diamond_preparation",
        "discontinuity": "state_and_thermo_continuity",
        "wrong_units": "diffusion_estimator_and_units",
        "wrong_diffusion": "reported_diffusion",
        "forged_log": "thermal_protocol_and_log",
        "bad_uncertainty": "correlation_aware_uncertainty",
    }[mutation]
    assert _by_name(rubric, expected)["earned"] == 0
    if mutation in {"wrong_geometry", "discontinuity"}:
        assert _by_name(rubric, "reported_diffusion")["earned"] == 0


def test_missing_and_malformed_data_never_throw(evidence):
    (evidence / "data.json").write_text('{"times_ps": [1, 2]}')
    rubric = _score(evidence)
    assert 0 < sum(c["earned"] for c in rubric.checks) < 90
    blank = Rubric(1)
    scoring.evaluate(Evidence({}), blank)
    assert sum(c["earned"] for c in blank.checks) == 0
    assert sum(c["points"] for c in blank.checks) == 90


def test_unsupported_analysis_is_unverified(evidence):
    path = evidence / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis["msd"]["estimator"] = "custom_diffusion"
    analysis["msd"]["calculations"] = {"diffusion_m2_s": 1e-8}
    analysis["diffusive"]["method"] = "custom_scaling"
    analysis["diffusive"]["calculations"] = {"exponent": 1.0}
    _dump(path, analysis)
    rubric = _score(evidence)
    assert _by_name(rubric, "msd_reconstruction")["status"] == "unverified"
    assert _by_name(rubric, "equilibrated_production")["status"] == "passed"
    assert _by_name(rubric, "liquid_and_diffusive_evidence")["status"] == "passed"
    assert rubric.score is None


def test_submission_scripts_are_never_executed(evidence):
    marker = evidence / "executed"
    (evidence / "danger.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    manifest = json.loads((evidence / "manifest.json").read_text())
    manifest["scripts"] = ["danger.py"]
    _dump(evidence / "manifest.json", manifest)
    _score(evidence)
    assert not marker.exists()


def test_prepared_translation_and_velocity_fluctuation_are_allowed(evidence):
    path = evidence / "prepared_state.extxyz"
    atoms = read(path)
    atoms.positions += [0.123, -0.37, 0.54]
    atoms.set_velocities(
        atoms.get_velocities() * np.sqrt(360 / atoms.get_temperature())
    )
    write(path, atoms)
    assert _by_name(_score(evidence), "diamond_preparation")["status"] == "passed"


def test_green_kubo_full_submission_uses_vacf_plateau(evidence):
    data = json.loads((evidence / "data.json").read_text())
    spec = {
        "estimator": "green_kubo",
        "lag_steps": list(range(21)),
        "origin_indices": "all",
        "remove_com_drift": True,
        "integration": "trapezoid",
        "integration_cutoff_ps": 40,
    }
    result = scoring._green_kubo(
        np.asarray(data["times_ps"]), np.asarray(data["velocities_A_ps"]), spec
    )
    spec.update(lag_times_ps=result["x"].tolist(), vacf_A2_ps2=result["y"].tolist())
    path = evidence / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis.pop("msd")
    analysis["diffusion"] = spec
    plateau = result["integral"][result["x"] >= 10]
    analysis["diffusive"] = {
        "method": "vacf_plateau",
        "interval_ps": [10, 40],
        "integral_mean_m2_s": float(plateau.mean()),
        "integral_relative_spread": float(np.ptp(plateau) / plateau.mean()),
    }
    _dump(path, analysis)
    manifest = json.loads((evidence / "manifest.json").read_text())
    manifest["results"]["diffusion_m2_s"] = result["d"]
    _dump(evidence / "manifest.json", manifest)
    rubric = _score(evidence)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    assert sum(c["points"] for c in rubric.checks) == 90
    spec["vacf_A2_ps2"][0] *= 2
    _dump(path, analysis)
    assert _by_name(_score(evidence), "msd_reconstruction")["status"] == "failed"


def test_split_ramp_linked_include_and_saved_velocity_initialization(evidence):
    path = evidence / "input.lmp"
    text = path.read_text().replace("velocity all create 300 4567 mom yes\n", "")
    text = text.replace(
        "run 1000000",
        "run 500000 start 0 stop 1000000\nrun 500000 start 0 stop 1000000",
    )
    text = text.replace(
        "pair_style sw\npair_coeff * * silicon.sw Si", "include silicon.mod"
    )
    path.write_text(text)
    (evidence / "silicon.mod").write_text(
        "pair_style sw\npair_coeff * * silicon.sw Si\n"
    )
    manifest = json.loads((evidence / "manifest.json").read_text())
    manifest["artifacts"]["potential"] = "silicon.mod"
    _dump(evidence / "manifest.json", manifest)
    rubric = _score(evidence)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    path.write_text(text.replace("stop 1000000", "stop 1200000"))
    assert _by_name(_score(evidence), "thermal_protocol_and_log")["status"] == "failed"


def test_timestep_changes_and_counter_resets_preserve_elapsed_time(evidence):
    path = evidence / "input.lmp"
    text = path.read_text().replace(
        "unfix heat", "unfix heat\nreset_timestep 0\ntimestep 0.002"
    )
    path.write_text(text.replace("run 500000", "run 250000"))
    thermo = pd.read_csv(evidence / "thermo.csv")
    production = thermo.time_ps >= 1000
    thermo.loc[production, "step"] = (
        (thermo.loc[production, "time_ps"] - 1000) / 0.002
    ).astype(int)
    thermo.to_csv(evidence / "thermo.csv", index=False)
    (evidence / "log.lammps").write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{r.step} {r.temperature_K} {r.pressure_bar} {r.density_g_cm3}"
            for r in thermo.itertuples()
        )
    )
    rubric = _score(evidence)
    assert rubric.score == pytest.approx(0.9), rubric.checks


def test_recorded_default_timestep_does_not_need_an_explicit_command(evidence):
    path = evidence / "input.lmp"
    path.write_text(path.read_text().replace("timestep 0.001\n", ""))
    rubric = _score(evidence)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    settings = json.loads((evidence / "settings.json").read_text())
    settings["timestep_ps"] = 0.002
    _dump(evidence / "settings.json", settings)
    assert _by_name(_score(evidence), "thermal_protocol_and_log")["status"] == "failed"


def test_unknown_input_control_flow_needs_review_but_missing_logs_fail(evidence):
    path = evidence / "input.lmp"
    path.write_text(path.read_text() + '\nif "1 == 1" then "print done"\n')
    rubric = _score(evidence)
    assert _by_name(rubric, "thermal_protocol_and_log")["status"] == "unverified"
    assert _by_name(rubric, "reported_diffusion")["status"] == "passed"
    assert rubric.score is None
    (evidence / "log.lammps").unlink()
    assert _by_name(_score(evidence), "thermal_protocol_and_log")["status"] == "failed"


@pytest.mark.parametrize("section", ["validation", "liquid", "diffusive"])
def test_alternative_diagnostics_require_retained_calculations(evidence, section):
    path = evidence / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis[section]["method"] = "alternative_diagnostic"
    analysis[section]["calculations"] = {"statistic": 1.0, "samples": [1, 2, 3]}
    _dump(path, analysis)
    name = {
        "validation": "equilibrated_production",
        "liquid": "liquid_and_diffusive_evidence",
        "diffusive": "diffusive_regime",
    }[section]
    rubric = _score(evidence)
    assert _by_name(rubric, name)["status"] == "unverified"
    assert _by_name(rubric, "reported_diffusion")["status"] == "passed"
    assert rubric.score is None
    analysis[section].pop("calculations")
    _dump(path, analysis)
    assert _by_name(_score(evidence), name)["status"] == "failed"
