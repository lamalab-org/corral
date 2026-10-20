"""Task 2 fixtures check evidence consistency, not execution provenance."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers
from ase.io import read as read_structure
from ase.io import write as write_structure
from corral_md.workflow_scoring.common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
)
from corral_md.workflow_scoring.level1 import _task_2 as evaluate_level1_task_2
from corral_md.workflow_scoring.task_2 import _boundaries, _estimate, evaluate


def test_boundary_density_uses_same_time_across_stage_labels(submission):
    from corral_md.workflow_scoring.task_2 import _boundary_density

    manifest, _, trace, root = submission
    # A shared boundary can be retained once under the stage that just ended.
    shared_starts = ((trace.stage == "hold") & (trace.time_ps == 370)) | (
        (trace.stage == "reheating") & (trace.time_ps == 470)
    )
    trace = trace[~shared_starts].copy()
    # A later sample can differ from the boundary density without contradiction.
    trace.loc[(trace.stage == "hold") & (trace.time_ps == 372.5), "density_g_cm3"] += (
        0.1
    )
    trace.to_csv(root / "thermal.csv", index=False)
    assert _boundary_density(Evidence(manifest)) is True

    trace.loc[trace.time_ps == 370, "density_g_cm3"] += 0.1
    trace.to_csv(root / "thermal.csv", index=False)
    outcome = _boundary_density(Evidence(manifest))
    assert outcome[0] is False
    assert "density" in outcome[1]


def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


@pytest.mark.parametrize("stage,previous,time_ps", [
    ("hold", "cooling", 370), ("reheating", "hold", 470),
])
@pytest.mark.parametrize("corruption", [None, "time", "step", "stage", "duplicate", "extra"])
def test_selected_accepts_only_unique_exact_shared_boundary(
    submission, stage, previous, time_ps, corruption
):
    from corral_md.workflow_scoring.task_2 import _selected

    _, _, trace, _ = submission
    trace = trace[~((trace.stage == stage) & (trace.time_ps == time_ps))].reset_index(drop=True)
    boundary = trace.index[(trace.stage == previous) & (trace.time_ps == time_ps)][0]
    indices = [int(boundary), *map(int, trace.index[trace.stage == stage])]
    if corruption == "time":
        trace.loc[boundary, "time_ps"] -= 0.1
    elif corruption == "step":
        trace.loc[boundary, "step"] -= 1
    elif corruption == "stage":
        trace.loc[boundary, "stage"] = "initial"
    elif corruption == "duplicate":
        duplicate = trace.loc[[boundary]].copy()
        duplicate["stage"] = stage
        trace = pd.concat([trace, duplicate], ignore_index=True)
    elif corruption == "extra":
        indices.insert(0, int(boundary - 1))
    if corruption is None:
        selected = _selected(trace, stage, indices)
        pd.testing.assert_frame_equal(selected, trace.iloc[indices])
        assert selected.iloc[0].stage == previous
    else:
        with pytest.raises(ValueError, match="different thermal stage"):
            _selected(trace, stage, indices)


@pytest.fixture
def submission(tmp_path):
    with ZipFile(Path(__file__).parents[1] / "structures.zip") as archive:
        (tmp_path / "liq4000.dat").write_bytes(
            archive.read("structures/melt/liq4000.dat")
        )
    source = read_structure(
        tmp_path / "liq4000.dat",
        format="lammps-data",
        atom_style="charge",
        units="real",
        sort_by_id=True,
    )
    rows = []
    for stage, start, end, low, high, tg in [
        ("cooling", 0, 370, 4000, 300, 1600),
        ("hold", 370, 470, 300, 300, 1600),
        ("reheating", 470, 840, 300, 4000, 1700),
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
    trace["step"] = trace.time_ps * 1000
    # Shift the synthetic density curve onto the real supplied starting density.
    # Its transition slopes and the analysis below remain independently generated.
    source_mass = sum(
        atomic_masses[atomic_numbers[s]] for s in source.get_chemical_symbols()
    )
    source_density = source_mass * 1.66053906660 / source.get_volume()
    trace["density_g_cm3"] += source_density - trace.density_g_cm3.iloc[0]
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
            trace.index[(trace.stage == stage) & (trace.temperature_K < 1000)]
        )
        high_idx = list(
            trace.index[
                (trace.stage == stage)
                & (trace.temperature_K > 2100)
                & (trace.temperature_K <= 2600)
            ]
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
    hold = trace.index[(trace.stage == "hold") & (trace.time_ps >= 390)].tolist()
    report["hold"] = {
        "indices": hold,
        "drift_indices": hold,
        "window_rationale": "Discard the first 20 ps.",
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
    species = source.get_chemical_symbols()
    mass = sum(atomic_masses[atomic_numbers[symbol]] for symbol in species)
    for name, time, row in [
        ("initial", 0, 0),
        ("cooling_end", 370, 40),
        ("hold_start", 370, 40),
        ("hold_end", 470, 81),
        ("reheating_start", 470, 81),
        ("reheating_end", 840, 122),
    ]:
        volume = mass * 1.66053906660 / trace.iloc[row].density_g_cm3
        scale = (volume / source.get_volume()) ** (1 / 3)
        velocities = source.get_velocities() * units.fs
        if name != "initial":
            velocities *= np.sqrt(
                trace.iloc[row].temperature_K / source.get_temperature()
            )
        states[name] = {
            "time_ps": time,
            "cell": (source.cell.array * scale).tolist(),
            "ids": source.arrays["id"].tolist(),
            "positions": (source.positions * scale).tolist(),
            "velocities": velocities.tolist(),
            "charges": source.get_initial_charges().tolist(),
            "species": species,
        }
    write_json(tmp_path / "boundaries.json", states)
    (tmp_path / "in.lammps").write_text("""units real
atom_style charge
boundary p p p
read_data liq4000.dat
pair_style buck/coul/long 8 12
pair_coeff * * 0 1 0
pair_coeff 1 3 101093.3472 0.243838 707.96963
pair_coeff 2 3 316001.3219145 0.193817 1260.9930729
pair_coeff 3 3 42541.49842 0.343645 4441.068122
kspace_style pppm 1e-5
fix cooling all npt temp 4000 300 100 iso 0 0 10000
run 370000
unfix cooling
fix hold all npt temp 300 300 100 iso 0 0 5000
run 100000
unfix hold
fix heat all npt temp 300 4000 100 iso 0 0 10000
run 370000
""")
    (tmp_path / "log.lammps").write_text(
        "Step Temp Press Density\n"
        + "\n".join(
            f"{row.step:g} {row.temperature_K:.8g} {row.pressure_atm:.8g} {row.density_g_cm3:.8g}"
            for row in trace.itertuples()
        )
    )
    report_path = write_json(tmp_path / "report.json", report)
    manifest = {
        "results": results,
        "report": report_path,
        "artifacts": {
            "initial_data": str(tmp_path / "liq4000.dat"),
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


@pytest.mark.parametrize("incorrect_schedule", [False, True])
def test_short_stage_labels_preserve_schedule_checks(submission, incorrect_schedule):
    manifest, _, trace, root = submission
    trace["stage"] = trace.stage.replace({"cooling": "cool", "reheating": "reheat"})
    if incorrect_schedule:
        trace.loc[trace.stage == "cool", "target_temperature_K"] += 200
    trace.to_csv(root / "thermal.csv", index=False)
    from corral_md.workflow_scoring.task_2 import _schedule

    result = _schedule(Evidence(manifest))
    assert (result is True) is not incorrect_schedule
    if incorrect_schedule:
        assert result[0] is False and "target schedule" in result[1]


def test_initial_trace_snapshot_preserves_analysis_indices(submission):
    manifest, report, trace, root = submission
    initial = trace.iloc[[0]].copy()
    initial["stage"] = "initial"
    trace = pd.concat([initial, trace], ignore_index=True)

    def shift_indices(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("indices"):
                    value[key] = [index + 1 for index in child]
                else:
                    shift_indices(child)
        elif isinstance(value, list):
            for child in value:
                shift_indices(child)

    shift_indices(report)
    write_json(root / "report.json", report)
    trace.to_csv(root / "thermal.csv", index=False)
    assert grade(manifest).score == pytest.approx(0.9)
    trace.loc[0, "temperature_K"] += 1
    trace.to_csv(root / "thermal.csv", index=False)
    assert check(grade(manifest), "thermal_cycle")["status"] == "failed"


def test_orthorhombic_boundary_lengths_preserve_continuity_checks(submission):
    manifest, _, _, root = submission
    path = root / "boundaries.json"
    states = json.loads(path.read_text())
    for state in states.values():
        state["cell"] = np.diag(state["cell"]).tolist()
    write_json(path, states)
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "passed"
    states["hold_start"]["cell"][0] *= 1.1
    write_json(path, states)
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "failed"
    states["hold_start"]["cell"] = [-1, -1, 1]
    write_json(path, states)
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "failed"


@pytest.mark.parametrize("name", ["source_data", "input_data", "initial_source"])
def test_explicit_source_data_alias_preserves_supplied_input_check(submission, name):
    manifest, _, _, root = submission
    manifest["artifacts"][name] = manifest["artifacts"].pop("initial_data")
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "passed"
    path = root / "liq4000.dat"
    path.write_text(path.read_text() + "\n")
    assert check(grade(manifest), "boundary_state_continuity")["status"] == "failed"


@pytest.mark.parametrize("short_stage_label", [False, True])
def test_level1_accepts_full_cooling_and_hold(submission, short_stage_label):
    manifest, _, trace, root = submission
    trace = trace[trace.stage.isin(["cooling", "hold"])].copy()
    if short_stage_label:
        trace["stage"] = trace.stage.replace({"cooling": "cool"})
    trace.to_csv(root / "thermal.csv", index=False)
    states = json.loads((root / "boundaries.json").read_text())
    write_json(
        root / "boundaries.json",
        {
            name: states[name]
            for name in ("initial", "cooling_end", "hold_start", "hold_end")
        },
    )
    inputs = (root / "in.lammps").read_text().split("unfix hold", 1)[0]
    (root / "in.lammps").write_text(inputs)
    log_rows = (root / "log.lammps").read_text().splitlines()
    (root / "log.lammps").write_text("\n".join(log_rows[:83]) + "\n")
    rubric = Rubric(2)
    evaluate_level1_task_2(Evidence(manifest), rubric)
    assert check(rubric, "cooling_schedule_and_observables")["status"] == "passed"
    assert check(rubric, "supplied_silicate_source_state")["status"] == "passed"
    assert check(rubric, "cooled_endpoint_temperature")["status"] == "passed"
    assert rubric.score == pytest.approx(0.90), rubric.checks


@pytest.mark.parametrize(
    "change", ["missing_hold", "short_hold", "handoff", "hold_density"]
)
def test_level1_requires_the_full_hold_and_continuous_states(submission, change):
    manifest, _, trace, root = submission
    path = root / "in.lammps"
    text = path.read_text().split("unfix hold", 1)[0]
    trace = trace[trace.stage.isin(["cooling", "hold"])].copy()
    states = json.loads((root / "boundaries.json").read_text())
    if change == "missing_hold":
        text = text.split("unfix cooling", 1)[0]
        trace = trace[trace.stage == "cooling"]
        states.pop("hold_end")
    elif change == "short_hold":
        text = text.replace("run 100000", "run 10000")
        trace.loc[trace.stage == "hold", "time_ps"] = (
            370 + (trace.loc[trace.stage == "hold", "time_ps"] - 370) / 10
        )
    elif change == "handoff":
        states["hold_start"]["velocities"][0][0] += 0.1
    else:
        states["hold_end"]["cell"][0][0] *= 1.1
    path.write_text(text)
    trace.to_csv(root / "thermal.csv", index=False)
    write_json(root / "boundaries.json", states)
    rubric = Rubric(2)
    evaluate_level1_task_2(Evidence(manifest), rubric)
    failed = (
        "cooling_input_cycle"
        if change in {"missing_hold", "short_hold"}
        else "cooled_state_identity_and_density"
    )
    assert check(rubric, failed)["status"] == "failed"


def test_checkpoint_continuation_cannot_pass_the_restored_full_cycle(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(
        path.read_text()
        .replace(
            "read_data liq4000.dat",
            "read_restart /workspace/structures/melt/liq2600.restart",
        )
        .replace("run 370000", "run 230000 start 0 stop 370000", 1)
    )
    rubric = grade(manifest)
    assert check(rubric, "thermal_cycle")["status"] == "failed"
    assert check(rubric, "boundary_state_continuity")["status"] == "failed"


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
    for stage, expected in [("cooling", 1600), ("reheating", 1700)]:
        report["transitions"][stage] = {
            "method": "change_point",
            "indices": trace.index[
                (trace.stage == stage) & (trace.temperature_K <= 2600)
            ].tolist(),
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


@pytest.mark.parametrize(
    "method",
    [
        "ordinary_least_squares_linear_slope",
        "ordinary_least_squares_density_vs_time",
        "unweighted_linear_slope",
        "unweighted linear slope of density versus elapsed time",
        "unweighted_linear_slope_vs_time",
        "unweighted_linear_slope_density_vs_time",
        "unweighted OLS density versus elapsed time",
        "unweighted linear slope of density versus elapsed time over the final 80 ps, "
        "excluding the first 20 ps as initial hold relaxation",
    ],
)
def test_explicit_ordinary_least_squares_drift_alias(submission, method):
    manifest, report, _, root = submission
    report["hold"]["drift_method"] = method
    write_json(root / "report.json", report)
    assert check(grade(manifest), "hold_density_drift")["status"] == "passed"
    manifest["results"]["hold_density_drift_g_cm3_ps"] += 0.1
    assert check(grade(manifest), "hold_density_drift")["status"] == "failed"


def test_screen_status_expressions_preserve_protocol_checks(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(
        path.read_text()
        + '\nprint "CYCLE COMPLETE step=$(step) time_ps=$(time/1000.0)"\n'
    )
    rubric = grade(manifest)
    for name in ("thermal_cycle", "saved_physics_and_logs", "boundary_state_continuity"):
        assert check(rubric, name)["status"] == "passed"
    path.write_text(path.read_text().replace("run 370000", "run 369000", 1))
    assert check(grade(manifest), "thermal_cycle")["status"] == "failed"


@pytest.mark.parametrize(
    "command",
    [
        'print "value=$(v_custom)"',
        'print "step=$(step)" file input.lammps',
    ],
)
def test_other_print_expressions_remain_unverified(submission, command):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(path.read_text() + "\n" + command + "\n")
    assert check(grade(manifest), "thermal_cycle")["status"] == "unverified"


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
        text.replace(
            "read_data liq4000.dat",
            "boundary f p p\nread_data liq4000.dat",
        )
    )
    assert check(grade(manifest), "saved_physics_and_logs")["status"] == "failed"


def test_alternative_change_point_estimator(submission):
    _, _, trace, _ = submission
    indices = trace.index[
        (trace.stage == "reheating") & (trace.temperature_K <= 2600)
    ].tolist()
    record = {
        "method": "change_point",
        "indices": indices,
        "temperature_coordinate": "temperature_K",
        "candidates_K": [1600, 1650, 1700, 1750, 1800],
        "estimate_K": 1700,
    }
    assert _estimate(trace, "reheating", record) == 1700


def test_change_point_cannot_invent_a_transition_in_a_straight_line():
    x = np.linspace(300, 2600, 100)
    trace = pd.DataFrame(
        {"stage": "cooling", "temperature_K": x, "density_g_cm3": 2.8 - 1e-4 * x}
    )
    record = {
        "method": "change_point",
        "temperature_coordinate": "temperature_K",
        "indices": list(range(len(trace))),
        "candidates_K": [800, 1600, 2000],
        "estimate_K": 800,
    }
    with pytest.raises(ValueError, match="straight density line"):
        _estimate(trace, "cooling", record)


def test_transition_fit_excludes_data_above_2600_K(submission):
    _, report, trace, _ = submission
    record = report["transitions"]["cooling"]
    record["high_indices"].append(0)  # Measured temperature is 4007 K.
    with pytest.raises(ValueError, match="above 2600 K"):
        _estimate(trace, "cooling", record)


def test_noisy_single_regime_requires_review_instead_of_an_arbitrary_tg():
    x = np.linspace(300, 2600, 200)
    y = 2.8 - 1e-4 * x + np.random.default_rng(42).normal(0, 0.01, len(x))
    candidates = [800, 1200, 1600, 2000]
    losses = []
    for candidate in candidates:
        design = np.column_stack([np.ones(len(x)), x, np.maximum(x - candidate, 0)])
        fit = np.linalg.lstsq(design, y, rcond=None)[0]
        losses.append(np.sum((y - design @ fit) ** 2))
    trace = pd.DataFrame({"stage": "cooling", "temperature_K": x, "density_g_cm3": y})
    record = {
        "method": "change_point",
        "temperature_coordinate": "temperature_K",
        "indices": list(range(len(x))),
        "candidates_K": candidates,
        "estimate_K": candidates[np.argmin(losses)],
    }
    with pytest.raises(UnsupportedEvidence, match="weak support"):
        _estimate(trace, "cooling", record)
    # Repeating the same observations must not turn weak support into a pass.
    record["indices"] *= 20
    with pytest.raises(ValueError, match="unique"):
        _estimate(trace, "cooling", record)


def test_repeated_line_fit_indices_are_rejected_by_scoring(submission):
    manifest, report, _, root = submission
    rec = report["transitions"]["cooling"]
    rec["low_indices"] *= 3
    rec["high_indices"] *= 3
    write_json(root / "report.json", report)
    rubric = grade(manifest)
    assert check(rubric, "transition_selections")["status"] == "failed"
    assert check(rubric, "cooling_transition_reproduction")["status"] == "failed"


def test_unresolved_transition_preserves_diagnostics_and_requires_review(submission):
    manifest, report, trace, root = submission
    indices = trace.index[
        (trace.stage == "cooling") & (trace.temperature_K <= 2600)
    ].tolist()
    selected = trace.iloc[indices]
    fit = np.polyfit(selected.temperature_K, selected.density_g_cm3, 1)
    record = {
        "method": "unresolved",
        "temperature_coordinate": "temperature_K",
        "indices": indices,
        "estimate_K": None,
        "reason": "The selected data need a scientific assessment of slope and window sensitivity.",
        "diagnostics": {
            "single_line_sse": float(
                np.sum(
                    (selected.density_g_cm3 - np.polyval(fit, selected.temperature_K))
                    ** 2
                )
            )
        },
    }
    report["transitions"]["cooling"] = record
    manifest["results"]["cooling_tg_K"] = None
    manifest["results"]["delta_tg_K"] = None
    write_json(root / "report.json", report)
    assert _estimate(trace, "cooling", record) is None
    rubric = grade(manifest)
    assert rubric.score is None
    assert check(rubric, "cooling_transition_reproduction")["status"] == "unverified"
    assert check(rubric, "signed_transition_difference")["status"] == "passed"
    assert check(rubric, "reported_estimates_and_units")["status"] == "passed"
    manifest["results"]["delta_tg_K"] = 0
    assert check(grade(manifest), "signed_transition_difference")["status"] == "failed"
    record["diagnostics"]["single_line_sse"] += 1
    write_json(root / "report.json", report)
    assert (
        check(grade(manifest), "cooling_transition_reproduction")["status"] == "failed"
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("temp 4000 300 100", "temp 4000 300 50"),
        ("iso 0 0 10000", "iso 0 0 1000"),
        ("iso 0 0 5000", "iso 0 0 1000"),
        ("run 370000", "run 230000"),
        ("temp 4000 300 100", "temp 1300 300 100"),
    ],
)
def test_original_ramp_and_damping_are_required(submission, old, new):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(path.read_text().replace(old, new))
    assert check(grade(manifest), "thermal_cycle")["status"] == "failed"


@pytest.mark.parametrize("initialization", ["run 0", "write_data early.dat"])
def test_data_input_allows_initial_export_or_run_zero(submission, initialization):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(
        path.read_text().replace("fix cooling", initialization + "\nfix cooling", 1)
    )
    assert check(grade(manifest), "thermal_cycle")["status"] == "passed"


def test_new_melt_has_no_checkpoint_fix_id_requirement(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    path.write_text(path.read_text().replace("fix cooling", "fix quench"))
    assert check(grade(manifest), "thermal_cycle")["status"] == "passed"


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
        text += "\n# run 370000\n# run 100000\n# run 370000\n"
    elif change == "zero":
        text = text.replace("run 370000", "run 0").replace("run 100000", "run 0")
    elif change == "duration":
        text = text.replace("run 370000", "run 20", 1)
    elif change == "timestep":
        text = "timestep 2\n" + text
    elif change == "temperature":
        text = text.replace("temp 4000 300", "temp 300 300")
    else:
        text = text.replace("run 370000", "unfix cooling\nrun 370000", 1)
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
        text = text.replace("run 100000", "pair_coeff * * 0 1 0\nrun 100000")
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


@pytest.mark.parametrize("artifact_role", ["potential", "input_files"])
def test_supplied_include_literal_variables_and_split_runs(submission, artifact_role):
    manifest, _, _, root = submission
    with ZipFile(Path(__file__).parents[1] / "potentials.zip") as archive:
        potential = archive.read("potentials/BKS/pot.mod").decode()
    (root / "pot.mod").write_text(potential)
    manifest["artifacts"][artifact_role] = [str(root / "pot.mod")]
    path = root / "in.lammps"
    lines = path.read_text().splitlines()
    lines = [
        line
        for line in lines
        if not line.startswith(("pair_style", "pair_coeff", "kspace_style"))
    ]
    text = "\n".join(lines).replace(
        "read_data liq4000.dat",
        "read_data liq4000.dat\ninclude input/pot.mod",
    )
    text = "variable dt equal 1.0\ntimestep ${dt}\n" + text
    text = text.replace(
        "run 370000",
        "run 185000 start 0 stop 370000\nrun 185000 start 0 stop 370000",
        1,
    )
    path.write_text(text)
    rubric = grade(manifest)
    assert rubric.score == pytest.approx(0.9), rubric.checks
    (root / "pot.mod").write_text(potential.replace("101093.3472", "12345"))
    assert check(grade(manifest), "saved_physics_and_logs")["status"] == "failed"


def test_supplied_bks_include_and_output_expressions_need_no_potential_copy(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(("pair_style", "pair_coeff", "kspace_style"))
    ]
    text = "\n".join(lines).replace(
        "read_data liq4000.dat",
        "read_data liq4000.dat\ninclude /workspace/potentials/BKS/pot.mod",
    )
    text = text.replace(
        "fix cooling all npt",
        'variable t equal 4000-0.01*step\nfix trace all print 1000 "${t}" file trace.dat screen no\nfix cooling all npt',
    )
    path.write_text(text)
    assert grade(manifest).score == pytest.approx(0.9)


def test_nonliteral_variable_in_npt_schedule_remains_unverified(submission):
    manifest, _, _, root = submission
    path = root / "in.lammps"
    text = path.read_text().replace(
        "fix cooling all npt temp 4000 300",
        "variable hot equal 4000+step\nfix cooling all npt temp ${hot} 300",
    )
    path.write_text(text)
    rubric = grade(manifest)
    assert check(rubric, "thermal_cycle")["status"] == "unverified"
    assert rubric.score is None


@pytest.mark.parametrize("corruption", [None, "cell_origin", "positions", "positions_lammps"])
def test_explicit_coordinate_origins_preserve_source_and_handoff_checks(submission, corruption):
    from corral_md.workflow_scoring.lammps_checks import supplied_silicate_initial_state

    manifest, _, _, root = submission
    path = root / "boundaries.json"
    states = json.loads(path.read_text())
    for index, state in enumerate(states.values()):
        origin = np.array([-2.5, -4.3, -1.9]) + index * 0.01
        state["cell_origin"] = origin.tolist()
        state["positions_lammps"] = state["positions"]
        state["positions"] = (np.asarray(state["positions"]) - origin).tolist()
    if corruption:
        values = np.asarray(states["initial"][corruption])
        values.flat[0] += 0.001
        states["initial"][corruption] = values.tolist()
    write_json(path, states)
    if corruption:
        with pytest.raises(EvidenceError, match="Conflicting boundary coordinate origins"):
            supplied_silicate_initial_state(Evidence(manifest))
    else:
        assert supplied_silicate_initial_state(Evidence(manifest))
        assert _boundaries(Evidence(manifest)) is True


@pytest.mark.parametrize(
    "relative,corruption",
    [(False, None), (True, None), (True, "positions"),
     (True, "positions_wrapped"), (True, "images"), (True, "cell_origin")],
)
def test_wrapped_image_metadata_preserves_coordinate_frame(submission, relative, corruption):
    from corral_md.workflow_scoring.lammps_checks import supplied_silicate_initial_state

    manifest, _, _, root = submission
    path = root / "boundaries.json"
    states = json.loads(path.read_text())
    for index, state in enumerate(states.values()):
        origin = np.array([-2.5, -4.3, -1.9]) + index * 0.01
        cell = np.asarray(state["cell"])
        absolute = np.asarray(state["positions"])
        images = np.floor((absolute - origin) @ np.linalg.inv(cell)).astype(int)
        state["cell_origin"] = origin.tolist()
        state["positions_wrapped"] = (absolute - images @ cell).tolist()
        state["images"] = images.tolist()
        state["positions"] = (absolute - origin if relative else absolute).tolist()
    if corruption:
        values = np.asarray(states["initial"][corruption], dtype=float)
        values.flat[0] += 0.001
        states["initial"][corruption] = values.tolist()
    write_json(path, states)
    if corruption:
        with pytest.raises(EvidenceError, match="wrapped boundary coordinate metadata"):
            supplied_silicate_initial_state(Evidence(manifest))
    else:
        assert supplied_silicate_initial_state(Evidence(manifest))
        assert _boundaries(Evidence(manifest)) is True


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


@pytest.mark.parametrize("shared_metadata", [False, True])
@pytest.mark.parametrize("path_field", ["lammps_data", "data_path"])
def test_linked_lammps_states_preserve_continuity_checks(submission, shared_metadata, path_field):
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
            path_field: output.name,
            "data_format": "lammps-data",
            "atom_style": "charge",
            "atom_type_species": {"1": "Na", "2": "Si", "3": "O"},
        }
    if shared_metadata:
        for record in linked.values():
            record.pop("atom_style")
            record.pop("atom_type_species")
        linked.update(
            description="Restartable states with common LAMMPS metadata.",
            atom_style="charge",
            atom_type_species={"1": "Na", "2": "Si", "3": "O"},
        )
    write_json(path, linked)
    assert _boundaries(Evidence(manifest)) is True
    linked["hold_start"][path_field] = "initial.data"
    write_json(path, linked)
    assert _boundaries(Evidence(manifest)) == (False, "Cell changed at a stage handoff")
    other_field = "data_path" if path_field == "lammps_data" else "lammps_data"
    linked["hold_start"][other_field] = "hold_start.data"
    write_json(path, linked)
    with pytest.raises(EvidenceError, match="Conflicting"):
        _boundaries(Evidence(manifest))


@pytest.mark.parametrize("corrupted_field", [None, "positions", "velocities", "cell"])
def test_inline_wrapped_states_agree_with_lammps_image_flags(submission, corrupted_field):
    manifest, _, _, root = submission
    path = Evidence(manifest).artifact("boundary_states")
    states = json.loads(path.read_text())
    manifest["artifacts"]["boundary_data"] = []
    for name, state in states.items():
        atoms = Atoms(
            state["species"], positions=state["positions"], cell=state["cell"], pbc=True
        )
        # Deliberately put different atoms in different periodic images.
        shifts = np.arange(len(atoms) * 3).reshape(-1, 3) % 5 - 2
        atoms.positions += shifts @ atoms.cell.array
        atoms.set_initial_charges(state["charges"])
        atoms.set_velocities(np.asarray(state["velocities"]) / units.fs)
        output = root / f"{name}.data"
        write_structure(
            output, atoms, format="lammps-data", atom_style="charge",
            units="real", velocities=True, write_image_flags=True,
            specorder=["Na", "Si", "O"],
        )
        manifest["artifacts"]["boundary_data"].append(str(output))
        state.update(
            positions=atoms.get_positions(wrap=True).tolist(),
            lammps_data=output.name,
            atom_style="charge",
            atom_type_species={"1": "Na", "2": "Si", "3": "O"},
        )
    if corrupted_field is not None:
        states["initial"][corrupted_field][0][0] += 1e-3
    write_json(path, states)
    if corrupted_field is None:
        assert _boundaries(Evidence(manifest)) is True
    else:
        with pytest.raises(EvidenceError, match=f"disagree on {corrupted_field}"):
            _boundaries(Evidence(manifest))


@pytest.mark.parametrize("level", [1, 2])
def test_task2_prompt_and_template_identify_the_original_protocol(level):
    from corral_md.submission_examples import example_files, load_example
    from corral_md.workflow_scoring import task_2

    root = Path(__file__).parents[1]
    task = json.loads(
        (root / f"environments/level_{level}/tasks_json/task_2.json").read_text()
    )[0]
    example = load_example(2, level=level)
    assert task["initial_input"]["protocol_version"] == task_2.PROTOCOL_VERSION
    assert task["initial_input"]["path to molten state"].endswith("/liq4000.dat")
    assert "run 370000" in task["description"]
    assert (
        example["files"]["settings.json"]["protocol_version"] == task_2.PROTOCOL_VERSION
    )
    assert "liq4000.dat" in example_files(2, level=level)["README.md"]
    if level == 2:
        assert "unresolved.example.json" in example["files"]
        assert "840,000" in task["description"]
        assert "2600 K" in task["description"]
    else:
        assert "470,000" in task["description"]
    assert {"initial", "cooling_end", "hold_start", "hold_end"} <= set(
        example["files"]["boundary_states.json"]
    )
