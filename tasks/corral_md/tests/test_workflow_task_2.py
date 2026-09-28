"""Task 2 fixtures check evidence consistency, not execution provenance."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers
from ase.io import write as write_structure
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1 import _task_2 as evaluate_level1_task_2
from corral_md.workflow_scoring.task_2 import _boundaries, _estimate, evaluate


def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def submission(tmp_path):
    rows = []
    for stage, start, end, low, high, tg in [
        ("cooling", 0, 20, 1300, 300, 750),
        ("hold", 20, 30, 300, 300, 750),
        ("reheating", 30, 50, 300, 1300, 850),
    ]:
        for index, time in enumerate(np.linspace(start, end, 41)):
            target = low + (high - low) * (time - start) / (end - start)
            measured = target + 7 * np.cos(index * 1.1)
            density = 2.8 - 1e-4 * measured - 2e-4 * max(measured - tg, 0)
            density += 0.00001 * np.sin(index)
            if stage == "hold":
                density += (time - start) * 1e-5
            rows.append(
                {
                    "stage": stage,
                    "time_ps": time,
                    "target_temperature_K": target,
                    "temperature_K": measured,
                    "pressure_atm": np.sin(index) * 100,
                    "density_g_cm3": density,
                }
            )
    trace = pd.DataFrame(rows)
    trace.to_csv(tmp_path / "thermal.csv", index=False)

    report = {
        "transitions": {},
        "hold": {},
        "equilibration_assessment": "Temperature follows the ramps; hold drift is finite.",
        "limitations": "These synthetic samples demonstrate verification, not physical equilibration.",
    }
    results = {
        "units": {"temperature": "K", "density": "g/cm3", "density_drift": "g/cm3/ps"}
    }
    for stage in ["cooling", "reheating"]:
        low_idx = list(
            trace.index[(trace.stage == stage) & (trace.temperature_K < 500)]
        )
        high_idx = list(
            trace.index[(trace.stage == stage) & (trace.temperature_K > 1050)]
        )

        def estimator(low_indices, high_indices):
            rec = {
                "method": "line_intersection",
                "low_indices": low_indices,
                "high_indices": high_indices,
                "temperature_coordinate": "temperature_K",
            }
            fit_low = np.polyfit(
                trace.iloc[low_indices].temperature_K,
                trace.iloc[low_indices].density_g_cm3,
                1,
            )
            fit_high = np.polyfit(
                trace.iloc[high_indices].temperature_K,
                trace.iloc[high_indices].density_g_cm3,
                1,
            )
            rec["estimate_K"] = float(
                (fit_high[1] - fit_low[1]) / (fit_low[0] - fit_high[0])
            )
            return rec

        primary = estimator(low_idx, high_idx)
        alternate = estimator(low_idx[1:], high_idx[:-1])
        alternate["shift_K"] = alternate["estimate_K"] - primary["estimate_K"]
        replicates = [
            estimator(low_idx[1:], high_idx),
            estimator(low_idx, high_idx[1:]),
            estimator(low_idx[:-1], high_idx[:-1]),
        ]
        primary["uncertainty"] = {
            "method": "window ensemble",
            "sampling_rationale": "Move fit-window endpoints.",
            "correlation_handling": "Window ensembles expose analysis uncertainty; time correlation is a limitation.",
            "aggregation": "sample_sd",
            "replicates": replicates,
        }
        primary["sensitivity"] = [alternate]
        report["transitions"][stage] = primary
        results[f"{stage}_tg_K"] = primary["estimate_K"]
        results[f"{stage}_uncertainty_K"] = float(
            np.std([rec["estimate_K"] for rec in replicates], ddof=1)
        )
    hold = trace.index[(trace.stage == "hold") & (trace.time_ps >= 22)].tolist()
    report["hold"] = {
        "indices": hold,
        "drift_indices": hold,
        "window_rationale": "Discard the first 2 ps.",
        "drift_method": "linear_slope",
    }
    selected = trace.iloc[hold]
    results.update(
        delta_tg_K=results["reheating_tg_K"] - results["cooling_tg_K"],
        hold_temperature_K=float(selected.temperature_K.mean()),
        hold_density_g_cm3=float(selected.density_g_cm3.mean()),
        hold_density_drift_g_cm3_ps=float(
            np.polyfit(selected.time_ps, selected.density_g_cm3, 1)[0]
        ),
    )
    states = {}
    species = ["Na", "Na", "Si", "O", "O", "O"]
    mass = sum(atomic_masses[atomic_numbers[symbol]] for symbol in species)
    for name, time, row in [
        ("initial", 0, 0),
        ("cooling_end", 20, 40),
        ("hold_start", 20, 40),
        ("hold_end", 30, 81),
        ("reheating_start", 30, 81),
        ("reheating_end", 50, 122),
    ]:
        length = (mass * 1.66053906660 / trace.iloc[row].density_g_cm3) ** (1 / 3)
        states[name] = {
            "time_ps": time,
            "cell": (np.eye(3) * length).tolist(),
            "ids": list(range(1, 7)),
            "positions": [[index * 0.1 * length] * 3 for index in range(6)],
            "velocities": [[index * 0.01] * 3 for index in range(6)],
            "charges": [0.6, 0.6, 2.4, -1.2, -1.2, -1.2],
            "species": species,
        }
    write_json(tmp_path / "boundaries.json", states)
    (tmp_path / "in.lammps").write_text("""units real
atom_style charge
boundary p p p
read_data liq1300.dat
pair_style buck/coul/long 8 12
pair_coeff * * 0 1 0
pair_coeff 1 3 101093.3472 0.243838 707.96963
pair_coeff 2 3 316001.3219145 0.193817 1260.9930729
pair_coeff 3 3 42541.49842 0.343645 4441.068122
kspace_style pppm 1e-5
fix ramp all npt temp 1300 300 100 iso 0 0 1000
run 20000
unfix ramp
fix hold all npt temp 300 300 100 iso 0 0 1000
run 10000
unfix hold
fix heat all npt temp 300 1300 100 iso 0 0 1000
run 20000
""")
    (tmp_path / "log.lammps").write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{row.time_ps * 1000:g} {row.temperature_K:.8g} {row.pressure_atm:.8g} {row.density_g_cm3:.8g}"
            for row in trace.itertuples()
        )
    )
    report_path = write_json(tmp_path / "report.json", report)
    manifest = {
        "results": results,
        "report": report_path,
        "artifacts": {
            "thermal_trace": str(tmp_path / "thermal.csv"),
            "boundary_states": str(tmp_path / "boundaries.json"),
            "lammps_inputs": str(tmp_path / "in.lammps"),
            "raw_logs": str(tmp_path / "log.lammps"),
        },
    }
    return manifest, report, trace, tmp_path


def grade(manifest):
    rubric = Rubric(2)
    evaluate(Evidence(manifest), rubric)
    return rubric


def check(rubric, name):
    return next(c for c in rubric.checks if c["name"] == name)


def test_valid_evidence_and_signed_hysteresis(submission):
    manifest, _, _, _ = submission
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    assert sum(c["points"] for c in rubric.checks) == 90
    assert check(rubric, "execution_provenance")["status"] == "unverified"


def test_level1_accepts_the_20_ps_cooling_stage(submission):
    manifest, _, trace, root = submission
    trace[trace.stage == "cooling"].to_csv(root / "thermal.csv", index=False)
    states = json.loads((root / "boundaries.json").read_text())
    write_json(
        root / "boundaries.json",
        {name: states[name] for name in ("initial", "cooling_end")},
    )
    inputs = (root / "in.lammps").read_text().split("unfix ramp", 1)[0]
    (root / "in.lammps").write_text(inputs)
    log_rows = (root / "log.lammps").read_text().splitlines()
    (root / "log.lammps").write_text("\n".join(log_rows[:42]) + "\n")
    rubric = Rubric(2)
    evaluate_level1_task_2(Evidence(manifest), rubric)
    assert check(rubric, "cooling_schedule_and_observables")["status"] == "passed"
    assert check(rubric, "supplied_silicate_source_state")["status"] == "failed"
    assert check(rubric, "cooled_endpoint_temperature")["status"] == "failed"
    assert rubric.score == pytest.approx(0.80), rubric.checks


@pytest.mark.parametrize(
    ("tamper", "failed"),
    [
        ("rate", "thermal_cycle"),
        ("hold_duration", "thermal_cycle"),
        ("velocity", "boundary_state_continuity"),
        ("charge", "boundary_state_continuity"),
        ("target_temperature", "measured_temperature_coordinate"),
        ("tg", "cooling_transition_reproduction"),
        ("difference", "signed_transition_difference"),
        ("hold_density", "hold_means"),
        ("hold_drift", "hold_density_drift"),
        ("physics", "saved_physics_and_logs"),
        ("log", "thermal_observables"),
        ("volume", "boundary_density_consistency"),
    ],
)
def test_tampered_evidence_loses_relevant_credit(submission, tamper, failed):
    manifest, report, trace, root = submission
    if tamper == "rate":
        trace.loc[trace.stage == "cooling", "target_temperature_K"] += 200
        trace.to_csv(root / "thermal.csv", index=False)
    elif tamper == "hold_duration":
        trace.loc[trace.stage == "hold", "time_ps"] += 15
        trace.to_csv(root / "thermal.csv", index=False)
    elif tamper in {"velocity", "charge"}:
        states = json.loads((root / "boundaries.json").read_text())
        if tamper == "velocity":
            states["hold_start"]["velocities"][0][0] = 3
        else:
            states["hold_start"]["charges"][0] = 3
        write_json(root / "boundaries.json", states)
    elif tamper == "target_temperature":
        record = report["transitions"]["cooling"]
        indices = record["low_indices"] + record["high_indices"]
        record["temperature_values_K"] = trace.iloc[
            indices
        ].target_temperature_K.tolist()
        write_json(root / "report.json", report)
    elif tamper == "tg":
        manifest["results"]["cooling_tg_K"] += 100
    elif tamper == "difference":
        manifest["results"]["delta_tg_K"] *= -1
    elif tamper == "hold_density":
        manifest["results"]["hold_density_g_cm3"] *= 2
    elif tamper == "hold_drift":
        manifest["results"]["hold_density_drift_g_cm3_ps"] += 1
    elif tamper == "uncertainty":
        manifest["results"]["cooling_uncertainty_K"] += 100
    elif tamper == "sensitivity":
        report["transitions"]["cooling"]["sensitivity"][0]["shift_K"] += 100
        write_json(root / "report.json", report)
    elif tamper == "physics":
        text = (root / "in.lammps").read_text().replace("iso 0 0", "iso 100 100")
        (root / "in.lammps").write_text(text)
    elif tamper == "log":
        (root / "log.lammps").write_text("Step Temp Press Density\n0 999 999 999\n")
    elif tamper == "volume":
        states = json.loads((root / "boundaries.json").read_text())
        states["reheating_end"]["cell"][0][0] *= 2
        write_json(root / "boundaries.json", states)
    rubric = grade(manifest)
    assert check(rubric, failed)["status"] == "failed"
    assert rubric.score < 0.9


def test_missing_or_malformed_artifacts_do_not_escape_checks(submission):
    manifest, _, _, root = submission
    (root / "thermal.csv").write_text("broken\nnot-a-number\n")
    rubric = grade(manifest)
    assert check(rubric, "thermal_observables")["status"] == "failed"
    empty = grade({})
    assert empty.score == 0


@pytest.mark.parametrize("legacy_report", [False, True])
def test_numerical_analysis_without_scientific_explanations(submission, legacy_report):
    manifest, analysis, _, root = submission
    analysis.pop("equilibration_assessment")
    analysis.pop("limitations")
    analysis["hold"].pop("window_rationale")
    for transition in analysis["transitions"].values():
        transition["uncertainty"].pop("sampling_rationale")
        transition["uncertainty"].pop("correlation_handling")
    if legacy_report:
        write_json(root / "report.json", analysis)
    else:
        manifest.pop("report")
        manifest["artifacts"]["analysis"] = write_json(root / "analysis.json", analysis)
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    assert sum(c["points"] for c in rubric.checks) == 90
    # Legacy optional uncertainty values do not affect the task score.
    manifest["results"]["cooling_uncertainty_K"] += 100
    assert grade(manifest).score == pytest.approx(0.9)


def test_unknown_method_is_explicitly_unverified(submission):
    manifest, report, _, root = submission
    rec = report["transitions"]["cooling"]
    rec["method"] = "custom_robust_estimator"
    rec["indices"] = rec["low_indices"] + rec["high_indices"]
    rec["calculations"] = {"estimate_K": rec["estimate_K"]}
    write_json(root / "report.json", report)
    rubric = grade(manifest)
    assert check(rubric, "cooling_transition_reproduction")["status"] == "unverified"
    assert check(rubric, "transition_selections")["status"] == "passed"
    assert rubric.score is None


@pytest.mark.parametrize("method", ["endpoint_slope", "theil_sen"])
def test_complete_alternative_estimator_and_drift_submission(submission, method):
    manifest, report, trace, root = submission
    for stage, expected in [("cooling", 750), ("reheating", 850)]:
        report["transitions"][stage] = {
            "method": "change_point",
            "indices": trace.index[trace.stage == stage].tolist(),
            "temperature_coordinate": "temperature_K",
            "candidates_K": [expected - 100, expected, expected + 100],
            "estimate_K": expected,
        }
        manifest["results"][f"{stage}_tg_K"] = expected
    manifest["results"]["delta_tg_K"] = 100
    report["hold"]["drift_method"] = method
    part = trace.iloc[report["hold"]["drift_indices"]].sort_values("time_ps")
    x, y = part.time_ps.to_numpy(), part.density_g_cm3.to_numpy()
    if method == "endpoint_slope":
        value = (y[-1] - y[0]) / (x[-1] - x[0])
    else:
        value = np.median(
            [
                (y[j] - y[i]) / (x[j] - x[i])
                for i in range(len(x))
                for j in range(i + 1, len(x))
            ]
        )
    manifest["results"]["hold_density_drift_g_cm3_ps"] = float(value)
    write_json(root / "report.json", report)
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    manifest["results"]["hold_density_drift_g_cm3_ps"] += 0.1
    assert check(grade(manifest), "hold_density_drift")["status"] == "failed"


def test_custom_drift_requires_saved_calculations_and_review(submission):
    manifest, report, _, root = submission
    report["hold"]["drift_method"] = "custom_drift"
    report["hold"]["calculations"] = {"estimates": [1e-5, 2e-5]}
    write_json(root / "report.json", report)
    rubric = grade(manifest)
    assert check(rubric, "hold_density_drift")["status"] == "unverified"
    assert check(rubric, "hold_means")["status"] == "passed"
    assert rubric.score is None
    report["hold"].pop("calculations")
    write_json(root / "report.json", report)
    assert check(grade(manifest), "hold_density_drift")["status"] == "failed"


def test_unknown_input_syntax_does_not_discard_reproduced_analysis(submission):
    manifest, _, trace, root = submission
    path = root / "in.lammps"
    path.write_text(path.read_text() + '\nif "1 == 1" then "print done"\n')
    trace["step"] = trace.time_ps * 1000
    trace.to_csv(root / "thermal.csv", index=False)
    rubric = grade(manifest)
    assert check(rubric, "thermal_cycle")["status"] == "unverified"
    assert check(rubric, "thermal_observables")["status"] == "passed"
    assert check(rubric, "cooling_transition_reproduction")["status"] == "passed"
    assert check(rubric, "hold_density_drift")["status"] == "passed"
    assert rubric.score is None


def test_periodic_default_does_not_need_an_explicit_boundary_command(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    text = path.read_text().replace("boundary p p p\n", "")
    path.write_text(text)
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    path.write_text(
        text.replace("read_data liq1300.dat", "boundary f p p\nread_data liq1300.dat")
    )
    assert check(grade(manifest), "saved_physics_and_logs")["status"] == "failed"


def test_alternative_change_point_estimator(submission):
    _, _, trace, _ = submission
    indices = trace.index[trace.stage == "reheating"].tolist()
    record = {
        "method": "change_point",
        "indices": indices,
        "temperature_coordinate": "temperature_K",
        "candidates_K": [750, 800, 850, 900, 950],
        "estimate_K": 850,
    }
    assert _estimate(trace, "reheating", record) == 850


def test_periodic_wrapping_and_atom_reordering_are_accepted(submission):
    manifest, _, _, root = submission
    states = json.loads((root / "boundaries.json").read_text())
    state = states["hold_start"]
    state["positions"][0][0] += state["cell"][0][0]
    for key in ["ids", "positions", "velocities", "charges", "species"]:
        state[key] = state[key][::-1]
    write_json(root / "boundaries.json", states)
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "passed"


@pytest.mark.parametrize(
    "change", ["missing", "zero", "duration", "timestep", "temperature", "inactive"]
)
def test_input_runs_must_support_the_thermal_cycle(submission, change):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    text = path.read_text()
    if change == "missing":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("run ")
        )
        text += "\n# run 20000\n# run 10000\n# run 20000\n"
    elif change == "zero":
        text = text.replace("run 20000", "run 0").replace("run 10000", "run 0")
    elif change == "duration":
        text = text.replace("run 20000", "run 20", 1)
    elif change == "timestep":
        text = "timestep 2\n" + text
    elif change == "temperature":
        text = text.replace("temp 1300 300", "temp 300 300")
    else:
        text = text.replace("run 20000", "unfix ramp\nrun 20000", 1)
    path.write_text(text)
    assert check(grade(manifest), "thermal_cycle")["status"] == "failed"


@pytest.mark.parametrize("change", ["missing", "coefficient", "cutoff", "override"])
def test_runs_must_use_supplied_potential(submission, change):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    text = path.read_text()
    if change == "missing":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("pair_coeff")
        )
    elif change == "coefficient":
        text = text.replace("101093.3472", "12345")
    elif change == "cutoff":
        text = text.replace("8 12", "10 10")
    else:
        text = text.replace("run 10000", "pair_coeff * * 0 1 0\nrun 10000")
    path.write_text(text)
    assert check(grade(manifest), "saved_physics_and_logs")["status"] == "failed"


def test_consistently_wrong_species_charges_are_rejected(submission):
    manifest, _, _, root = submission
    path = root / "boundaries.json"
    states = json.loads(path.read_text())
    for state in states.values():
        # Remains neutral, nonzero, and unchanged at every handoff.
        state["charges"] = (np.asarray(state["charges"]) * 2).tolist()
    write_json(path, states)
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "failed"


def test_supplied_include_literal_variables_and_split_runs(submission):
    manifest, _, _, root = submission
    with ZipFile(Path(__file__).parents[1] / "potentials.zip") as archive:
        potential = archive.read("potentials/BKS/pot.mod").decode()
    (root / "pot.mod").write_text(potential)
    manifest["artifacts"]["potential"] = str(root / "pot.mod")
    path = root / "in.lammps"
    lines = path.read_text().splitlines()
    lines = [
        line
        for line in lines
        if not line.startswith(("pair_style", "pair_coeff", "kspace_style"))
    ]
    text = "\n".join(lines).replace(
        "read_data liq1300.dat",
        "read_data liq1300.dat\ninclude /workspace/potentials/BKS/pot.mod",
    )
    text = "variable dt equal 0.5\ntimestep ${dt}\n" + text
    text = text.replace(
        "run 20000", "run 20000 start 0 stop 40000\nrun 20000 start 0 stop 40000", 1
    )
    text = text.replace(
        "fix hold all npt temp 300 300 100 iso 0 0 1000\nrun 10000",
        "fix hold all npt temp 300 300 100 iso 0 0 1000\nrun 20000",
    ).replace(
        "fix heat all npt temp 300 1300 100 iso 0 0 1000\nrun 20000",
        "fix heat all npt temp 300 1300 100 iso 0 0 1000\nrun 40000",
    )
    path.write_text(text)
    log = root / "log.lammps"
    rows = [line.split() for line in log.read_text().splitlines()[1:]]
    log.write_text(
        "Step Temp Press Density\n"
        + "\n".join(" ".join([str(float(row[0]) * 2), *row[1:]]) for row in rows)
    )
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks


def test_supplied_bks_include_and_output_expressions_need_no_potential_copy(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(("pair_style", "pair_coeff", "kspace_style"))
    ]
    text = "\n".join(lines).replace(
        "read_data liq1300.dat",
        "read_data liq1300.dat\ninclude /workspace/potentials/BKS/pot.mod",
    )
    text = text.replace(
        "fix ramp all npt",
        'variable t equal 1300-0.05*step\nfix trace all print 1000 "${t}" file trace.dat screen no\nfix ramp all npt',
    )
    path.write_text(text)
    assert grade(manifest).score == pytest.approx(0.9)


def test_nonliteral_variable_in_npt_schedule_remains_unverified(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    text = path.read_text().replace(
        "fix ramp all npt temp 1300 300",
        "variable hot equal 1300+step\nfix ramp all npt temp ${hot} 300",
    )
    path.write_text(text)
    rubric = grade(manifest)
    assert check(rubric, "thermal_cycle")["status"] == "unverified"
    assert rubric.score is None


def test_descriptive_boundary_fields_preserve_checks_and_reject_conflicts(submission):
    manifest, _, _, _ = submission
    path = Evidence(manifest).artifact("boundary_states")
    states = json.loads(path.read_text())
    names = {
        "cell": "cell_A",
        "positions": "positions_A",
        "velocities": "velocities_A_fs",
        "charges": "charges_e",
        "ids": "atom_ids",
    }
    states = {
        name: {names.get(key, key): value for key, value in state.items()}
        for name, state in states.items()
    }
    write_json(path, states)
    rubric = grade(manifest)
    assert check(rubric, "boundary_state_continuity")["status"] == "passed"
    assert check(rubric, "boundary_density_consistency")["status"] == "passed"
    states["hold_start"]["velocities"] = (
        np.asarray(states["hold_start"]["velocities_A_fs"]) + 1
    ).tolist()
    write_json(path, states)
    failure = check(grade(manifest), "boundary_state_continuity")
    assert failure["status"] == "failed"
    assert "Conflicting" in failure["detail"]


def test_linked_lammps_states_preserve_continuity_checks(submission):
    manifest, _, _, root = submission
    path = Evidence(manifest).artifact("boundary_states")
    states = json.loads(path.read_text())
    linked = {}
    manifest["artifacts"]["boundary_data"] = []
    for name, state in states.items():
        atoms = Atoms(
            state["species"], positions=state["positions"], cell=state["cell"], pbc=True
        )
        atoms.set_initial_charges(state["charges"])
        atoms.set_velocities(np.asarray(state["velocities"]) / units.fs)
        output = root / f"{name}.data"
        write_structure(
            output,
            atoms,
            format="lammps-data",
            atom_style="charge",
            units="real",
            velocities=True,
            specorder=["Na", "Si", "O"],
        )
        manifest["artifacts"]["boundary_data"].append(str(output))
        linked[name] = {
            "time_ps": state["time_ps"],
            "lammps_data": output.name,
            "atom_style": "charge",
            "atom_type_species": {"1": "Na", "2": "Si", "3": "O"},
        }
    write_json(path, linked)
    assert _boundaries(Evidence(manifest)) is True
    linked["hold_start"]["lammps_data"] = "initial.data"
    write_json(path, linked)
    assert _boundaries(Evidence(manifest)) == (False, "Cell changed at a stage handoff")
