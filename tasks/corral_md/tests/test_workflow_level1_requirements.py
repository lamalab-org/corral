"""Prompt requirements must survive internally consistent alternative evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import units
from ase.build import bulk
from corral_md.score import WorkflowScorer
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1 import evaluate
from corral_md.workflow_scoring.task_2 import _input_cycle, _logged_trace


def save(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def read(path):
    return json.loads(path.read_text())


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


@pytest.fixture
def cooling(tmp_path):
    inputs = tmp_path / "cool.in"
    inputs.write_text("fix ramp all npt temp 1300 300 100 iso 0 0 1000\nrun 20000\n")
    table = pd.DataFrame(
        {
            "time_ps": [0.0, 10.0, 20.0],
            "step": [0, 10000, 20000],
            "target_temperature_K": [1300, 800, 300],
            "temperature_K": [1305, 793, 298],
            "pressure_atm": [10, -3, 5],
            "density_g_cm3": [2.6, 2.7, 2.8],
        }
    )
    table.to_csv(tmp_path / "trace.csv", index=False)
    (tmp_path / "raw.log").write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{r.step} {r.temperature_K} {r.pressure_atm} {r.density_g_cm3}"
            for r in table.itertuples()
        )
    )
    return Evidence(
        {
            "artifacts": {
                "lammps_inputs": str(inputs),
                "thermal_trace": str(tmp_path / "trace.csv"),
                "raw_logs": str(tmp_path / "raw.log"),
            }
        }
    )


def strict_cycle(e):
    return _input_cycle(e, 20, required_timestep_fs=1.0, required_steps=20_000)


@pytest.mark.parametrize(
    "commands",
    [
        "run 20000\n",  # LAMMPS real-unit default: 1 fs.
        "timestep 1.0\nrun 20000\n",
        "run 0\nrun 10000 start 0 stop 20000\nrun 20000 upto start 0 stop 20000\n",
        "timestep 2\nrun 0\ntimestep 1\nrun 20000\n",  # Only advancing steps matter.
    ],
)
def test_task2_accepts_exact_schedule_with_equivalent_run_commands(cooling, commands):
    cooling.artifact("lammps_inputs").write_text(
        "fix ramp all npt temp 1300 300 100 iso 0 0 1000\n" + commands
    )
    assert strict_cycle(cooling)[0]
    assert _logged_trace(cooling, {"cooling"}, required_timestep_fs=1.0) is True


@pytest.mark.parametrize(
    "commands",
    [
        "timestep 2\nrun 10000\n",
        "timestep 0.5\nrun 40000\n",
        "timestep 1\nrun 19999\n",
        "timestep 1\nrun 20001\n",
        "fix adaptive all dt/reset 1 0.5 2.0 0.1\nrun 20000\n",
    ],
)
def test_task2_rejects_wrong_step_schedule(cooling, commands):
    cooling.artifact("lammps_inputs").write_text(
        "fix ramp all npt temp 1300 300 100 iso 0 0 1000\n" + commands
    )
    assert not strict_cycle(cooling)[0]
    rubric = Rubric(2)
    evaluate(cooling, rubric, 2)
    assert check(rubric.as_dict(), "cooling_input_cycle")["status"] == "failed"


def test_task2_trace_cannot_hide_wrong_steps_with_matching_raw_log(cooling):
    table = cooling.table("thermal_trace")
    table["step"] //= 2
    table.to_csv(cooling.artifact("thermal_trace"), index=False)
    cooling.artifact("raw_logs").write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{r.step} {r.temperature_K} {r.pressure_atm} {r.density_g_cm3}"
            for r in table.itertuples()
        )
    )
    assert _logged_trace(cooling, {"cooling"}) is True
    assert not _logged_trace(cooling, {"cooling"}, required_timestep_fs=1.0)[0]
    rubric = Rubric(2)
    evaluate(cooling, rubric, 2)
    assert (
        check(rubric.as_dict(), "cooling_trace_matches_raw_log")["status"] == "failed"
    )


def test_task2_trace_can_derive_steps_from_raw_log(cooling):
    table = cooling.table("thermal_trace").drop(columns="step")
    table.to_csv(cooling.artifact("thermal_trace"), index=False)
    assert _logged_trace(cooling, {"cooling"}, required_timestep_fs=1.0) is True


@pytest.fixture
def aluminum(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05).repeat((4, 4, 4))
    rng = np.random.default_rng(602311)
    p = rng.normal(size=(64, 3))
    p -= p.mean(axis=0)
    mass = atoms.get_masses()
    density = mass.sum() * 1.66053906660 / atoms.get_volume()
    frames, trace = [], []
    for i, temperature in enumerate((300, 305, 295, 300, 303, 298, 301)):
        momenta = p * np.sqrt(
            temperature * 189 * units.kB / np.sum(p**2 / mass[:, None])
        )
        frames.append(
            {
                "symbols": ["Al"] * 64,
                "positions": (atoms.positions + 0.001 * i * p).tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": [True] * 3,
                "momenta": momenta.tolist(),
                "time_fs": 100 * i,
                "energy": -220.0,
            }
        )
        trace.append(
            {
                "stage": "nvt",
                "step": 100 * i,
                "time_fs": 100 * i,
                "temperature_K": temperature,
                "pressure_GPa": -1.5 + i * 0.1,  # Nonzero pressure is valid for NVT.
                "density_g_cm3": density,
                "potential_energy_eV": -220.0,
            }
        )
    settings = {
        "model": "MACE-MP-0",
        "model_settings": {},
        "md": {
            "timestep_fs": 1,
            "temperature_dof": 189,
            "initial_temperature_K": 300,
            "target_temperature_K": 300,
            "random_seed": 602311,
            "remove_com_once": True,
            "thermostat": "Langevin",
            "equilibration_integrator": "Langevin",
        },
        "readiness": {
            "note": "Terminal temperatures remain near 300 K; retain the final momenta for NVE."
        },
    }
    (tmp_path / "run.py").write_text(
        "# Submitted scripts must never be executed by grading.\n"
    )
    manifest = {
        "settings": save(tmp_path / "settings.json", settings),
        "scripts": [str(tmp_path / "run.py")],
        "artifacts": {
            "equilibration_trajectory": save(tmp_path / "eq.json", frames),
            "thermal_trace": save(tmp_path / "trace.json", trace),
            "restartable_final_state": save(tmp_path / "final.json", [frames[-1]]),
        },
    }
    return save(tmp_path / "manifest.json", manifest)


def grade_aluminum(path):
    return WorkflowScorer(6, level=1).evaluate(path)


def test_task6_complete_prompt_evidence_needs_only_independent_model_verification(
    aluminum,
):
    report = grade_aluminum(aluminum)
    assert report["pending_checks"] == ["independent_model_calculation"], report
    assert report["earned_points"] == 99
    assert sum(c["points"] for c in report["checks"]) == 100


@pytest.mark.parametrize("field", ["step", "pressure_GPa", "density_g_cm3"])
def test_task6_requires_every_logging_field(aluminum, field):
    path = Evidence(aluminum).artifact("thermal_trace")
    rows = read(path)
    for row in rows:
        del row[field]
    save(path, rows)
    report = grade_aluminum(aluminum)
    assert report["score"] == 0
    assert check(report, "required_simulation_log_fields")["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step", 1.5),
        ("step", 99),
        ("density_g_cm3", 12),
        ("pressure_GPa", float("inf")),
    ],
)
def test_task6_rejects_invalid_or_inconsistent_logging(aluminum, field, value):
    path = Evidence(aluminum).artifact("thermal_trace")
    rows = read(path)
    rows[1][field] = value
    save(path, rows)
    assert grade_aluminum(aluminum)["score"] == 0


@pytest.mark.parametrize("unit", ["pressure_bar", "pressure_atm"])
def test_task6_accepts_other_pressure_units_and_sparse_duplicate_initial_samples(
    aluminum, unit
):
    path = Evidence(aluminum).artifact("thermal_trace")
    rows = read(path)
    for row in rows:
        row[unit] = row.pop("pressure_GPa") * (
            10000 if unit == "pressure_bar" else 10000 / 1.01325
        )
    rows = [rows[0], rows[0], rows[3], rows[-1]]
    save(path, rows)
    assert grade_aluminum(aluminum)["pending_checks"] == [
        "independent_model_calculation"
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("velocity_initializations", 2),
        ("velocity_reinitializations_during_run", 7),
        ("velocity_rescalings_during_run", 1),
        ("manual_velocity_resets", 1),
        ("manual_velocity_rescalings", 1),
        ("rescale_velocities_during_run", True),
        ("reinitialize_velocities_during_run", True),
    ],
)
def test_task6_rejects_recorded_resets_or_rescaling(aluminum, key, value):
    e = Evidence(aluminum)
    path = e._path(e.manifest["settings"])
    settings = read(path)
    settings["md"][key] = value
    save(path, settings)
    report = grade_aluminum(aluminum)
    assert report["score"] == 0
    assert check(report, "no_recorded_velocity_resets")["status"] == "failed"


def test_task6_requires_the_readiness_assessment(aluminum):
    e = Evidence(aluminum)
    settings = e.settings
    del settings["readiness"]
    save(e._path(e.manifest["settings"]), settings)
    report = grade_aluminum(aluminum)
    assert report["score"] == 0
    assert check(report, "equilibrated_state_readiness")["status"] == "failed"


def test_task6_accepts_assessment_in_results_with_an_agent_selected_terminal_window(
    aluminum,
):
    path = Path(aluminum)
    manifest = read(path)
    e = Evidence(aluminum)
    settings = e.settings
    manifest["results"] = {
        "readiness": {**settings.pop("readiness"), "window_fs": [400, 600]}
    }
    save(e._path(e.manifest["settings"]), settings)
    save(path, manifest)
    assert grade_aluminum(aluminum)["pending_checks"] == [
        "independent_model_calculation"
    ]


def test_task6_1000K_endpoint_is_not_hidden_by_earlier_cool_frames(aluminum):
    e = Evidence(aluminum)
    frames = read(e.artifact("equilibration_trajectory"))
    rows = read(e.artifact("thermal_trace"))
    frames[-1]["momenta"] = (
        np.asarray(frames[-1]["momenta"]) * np.sqrt(1000 / rows[-1]["temperature_K"])
    ).tolist()
    rows[-1]["temperature_K"] = 1000
    save(e.artifact("equilibration_trajectory"), frames)
    save(e.artifact("restartable_final_state"), [frames[-1]])
    save(e.artifact("thermal_trace"), rows)
    report = grade_aluminum(aluminum)
    assert report["score"] is None
    assert "equilibrated_state_readiness" in report["pending_checks"]
    assert check(report, "thermal_trace_from_saved_momenta")["status"] == "passed"


def test_task6_readiness_summary_must_match_data(aluminum):
    e = Evidence(aluminum)
    e.settings["readiness"]["final_temperature_K"] = 200
    save(e._path(e.manifest["settings"]), e.settings)
    report = grade_aluminum(aluminum)
    assert report["score"] == 0
    assert check(report, "equilibrated_state_readiness")["status"] == "failed"
