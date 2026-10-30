"""Final numerical results are independent of workflow artifact serialization."""

import json

import numpy as np
import pytest
from corral_md.scoring.evaluator import (
    EvaluationError,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring.final_quantities import FinalQuantityScorer, correlation_values
from corral_md.scoring.verification import ModalVerifier

from .test_score_task_5 import transport


def write_log(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def aluminum_log(tmp_path, task, nested=False):
    rows = [{"type": "sample", "phase": "equilibration", "step": 0}]
    for i in range(8):
        t = 300 + (-1) ** i
        row = {"type": "sample", "phase": "production", "step": i + 1}
        if task == 7:
            row.update(
                temperature_K=t, volume_A3=1800 + (-1) ** i, pressure_GPa=1.01325e-4
            )
        else:
            row.update(
                energy_eV_per_atom=-3.6 + 0.001 * (-1) ** i, Tx_K=t, Ty_K=t, Tz_K=t
            )
        rows.append(row)
    path = tmp_path / "run.jsonl"
    if nested:
        path.write_text(json.dumps({"metadata": {}, "stages": [{"samples": rows}]}))
    else:
        write_log(
            path, [{"type": "metadata", "seed": "not needed", "timestep_fs": 10}, *rows]
        )
    return {"artifacts": {"run_log": str(path)}}, path, rows


@pytest.mark.parametrize("task", [7, 10])
@pytest.mark.parametrize("nested", [False, True])
def test_means_need_only_scalar_samples(tmp_path, task, nested):
    submission, _, _ = aluminum_log(tmp_path, task, nested)
    report = check_level1_workflow(task, verification_backend="offline").evaluate(
        submission
    )
    assert report["score"] == 1, report
    assert report["scoring_policy"] == "level1-final-quantities-v2"
    stage = report["computed"]["stages"][0]
    assert stage["sample_count"] == 8  # no hidden 32-sample quality gate
    assert stage["temperature_K"] == 300
    assert stage["volume_A3" if task == 7 else "energy_eV_per_atom"] == pytest.approx(
        1800 if task == 7 else -3.6
    )


@pytest.mark.parametrize("task", [7, 10])
def test_unrequested_results_do_not_change_the_sample_score(tmp_path, task):
    submission, _, _ = aluminum_log(tmp_path, task)
    submission["results"] = (
        {"volume_A3": 2500} if task == 7 else {"energy_eV_per_atom": -2.0}
    )
    result = check_level1_workflow(task).evaluate_submission(submission)
    assert result.score == 1
    assert result.feedback is None


@pytest.mark.parametrize("task", [7, 10])
def test_wrong_final_temperature_fails(tmp_path, task):
    submission, path, rows = aluminum_log(tmp_path, task)
    for row in rows[1:]:
        if task == 7:
            row["temperature_K"] = 800
        else:
            row.update(Tx_K=800, Ty_K=800, Tz_K=800)
        row.update(T_K=300, Temperature=300, Tx=300, Ty=300, Tz=300)
    write_log(path, rows)
    assert check_level1_workflow(task)(submission) == 0


def test_pressure_alias_does_not_override_documented_gpa(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 7)
    first = check_level1_workflow(7).evaluate(submission)
    for row in rows[1:]:
        row["pressure_bar"] = 100000
    write_log(path, rows)
    second = check_level1_workflow(7).evaluate(submission)
    assert first["computed"] == second["computed"]


def test_extra_energy_fields_do_not_override_documented_per_atom_energy(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 10)
    for row in rows[1:]:
        row.update(Epot_eV="unused", Ekin_eV=None, Etot_per_atom_eV=-100)
    write_log(path, rows)
    result = check_level1_workflow(10).evaluate(submission)
    assert result["score"] == 1
    assert result["computed"]["stages"][0]["energy_eV_per_atom"] == pytest.approx(-3.6)


def test_task10_derives_overall_temperature_from_directional_samples(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 10)
    for row in rows[1:]:
        row["Tx_K"] += 10
        row["Ty_K"] -= 4
        row["Tz_K"] -= 6
    write_log(path, rows)
    report = check_level1_workflow(10).evaluate(submission)
    assert report["score"] == 1, report
    stage = report["computed"]["stages"][0]
    assert stage["temperature_K"] == 300
    assert stage["directional_temperature_K"] == [310, 296, 294]


@pytest.mark.parametrize("field", ["temperature_K", "Temperature", "T_K"])
def test_task10_ignores_unrequested_overall_temperature(tmp_path, field):
    submission, path, rows = aluminum_log(tmp_path, 10)
    for row in rows[1:]:
        row[field] = 800
    write_log(path, rows)
    result = check_level1_workflow(10).evaluate_submission(submission)
    assert result.score == 1
    assert result.feedback is None


def test_task10_checks_target_without_optional_overall_temperature(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 10)
    for row in rows[1:]:
        row.update(Tx_K=340, Ty_K=340, Tz_K=340)
    write_log(path, rows)
    assert check_level1_workflow(10)(submission) == 0


def test_missing_documented_field_reports_submission_failure(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 7)
    for row in rows[1:]:
        row["unrecognized_volume"] = row.pop("volume_A3")
    write_log(path, rows)
    result = check_level1_workflow(7).evaluate_submission(submission)
    assert result.score == 0
    assert "Missing required production sample field: volume_A3" in result.feedback


def test_missing_production_is_not_automatically_credited(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 7)
    write_log(path, rows[:1])
    assert check_level1_workflow(7)(submission) == 0


def test_nonfinite_quantity_is_not_credited(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 10)
    rows[1]["energy_eV_per_atom"] = float("nan")
    write_log(path, rows)
    assert check_level1_workflow(10)(submission) == 0


def test_restart_markers_do_not_become_thermodynamic_samples(tmp_path):
    submission, path, rows = aluminum_log(tmp_path, 10)
    rows.insert(4, {"type": "restart", "step": 3})
    rows.insert(7, {"type": "checkpoint", "step": 5})
    write_log(path, [{"type": "metadata", "equilibration_steps": 0}, *rows])
    result = check_level1_workflow(10).evaluate(submission)
    assert result["score"] == 1, result
    assert result["computed"]["stages"][0]["sample_count"] == 8


@pytest.mark.parametrize("task", [7, 10])
@pytest.mark.parametrize("nested", [False, True])
def test_metadata_targets_and_unused_fields_do_not_change_measurements(
    tmp_path, task, nested
):
    submission, path, rows = aluminum_log(tmp_path, task)
    for i, row in enumerate(rows[1:]):
        temperature = 300 + 10 * (-1) ** i
        if task == 7:
            row["temperature_K"] = temperature
        else:
            row.update(Tx_K=temperature, Ty_K=temperature, Tz_K=temperature)
    write_log(path, rows)
    scorer = check_level1_workflow(task)
    baseline = scorer.evaluate(submission)
    assert baseline["score"] == 1
    metadata = {
        "type": "meta",
        "seed": 12345,
        "T_K": 300,
        "temperature_K": 800,
        "Tx_K": 800,
        "volume_A3": -1,
        "pressure_GPa": 100,
        "energy_eV_per_atom": 100,
    }
    for row in rows[1:]:
        row.update(
            T_K=300,
            Temperature="unused",
            Tx=-100,
            volume=-1,
            momenta="unused",
            momenta_ase_units=None,
            event="warning",
            name="equilibration",
        )
    rows.append({"type": "summary", "temperature_K": 800, "Tx_mean": 800})
    if nested:
        path.write_text(
            json.dumps({"metadata": metadata, "stages": [{"samples": rows}]})
        )
    else:
        write_log(path, [metadata, *rows])
    result = scorer.evaluate(submission)
    assert result["score"] == 1, result
    assert result["computed"] == baseline["computed"]


@pytest.mark.parametrize(
    ("task", "field", "alias"),
    [
        (7, "temperature_K", "T_K"),
        (7, "volume_A3", "volume"),
        (7, "pressure_GPa", "pressure_bar"),
        (10, "Tx_K", "Tx"),
        (10, "Ty_K", "Ty"),
        (10, "Tz_K", "Tz"),
        (10, "energy_eV_per_atom", "Etot_per_atom_eV"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_metadata_or_alias_cannot_supply_a_missing_sample_field(
    tmp_path, task, field, alias, nested
):
    submission, path, rows = aluminum_log(tmp_path, task)
    missing = rows[1].pop(field)
    rows[1][alias] = missing
    metadata = {"type": "meta", field: missing}
    if nested:
        path.write_text(
            json.dumps({"metadata": metadata, "stages": [{"samples": rows}]})
        )
    else:
        write_log(path, [metadata, *rows])
    result = check_level1_workflow(task).evaluate_submission(submission)
    assert result.score == 0
    assert f"Missing required production sample field: {field}" in result.feedback


@pytest.mark.parametrize("task", [7, 10])
def test_nested_metadata_does_not_need_a_scoring_schema(tmp_path, task):
    submission, path, rows = aluminum_log(tmp_path, task)
    path.write_text(
        json.dumps(
            {"metadata": "unused", "phase": "unused", "step": "unused", "samples": rows}
        )
    )
    result = check_level1_workflow(task).evaluate_submission(submission)
    assert result.score == 1


@pytest.mark.parametrize("task", [7, 10])
def test_sample_phase_must_use_documented_field(tmp_path, task):
    submission, path, rows = aluminum_log(tmp_path, task)
    rows[1]["name"] = rows[1].pop("phase")
    write_log(path, rows)
    result = check_level1_workflow(task).evaluate_submission(submission)
    assert result.score == 0
    assert "Sample phase must be" in result.feedback


def phonon_submission(tmp_path):
    rows = [
        {
            "strain_fraction": float(s),
            "gamma_phonon_energies_eV": [0, 0, 0, 0.01, 0.02, 0.06 + 0.4 * s],
        }
        for s in np.arange(-2, 3) / 100
    ]
    path = tmp_path / "phonons.json"
    path.write_text(json.dumps({"results": rows}))
    return {"artifacts": {"calculation_log": str(path)}}, path, rows


def test_phonon_reference_uses_task_geometry_and_only_maximum(tmp_path):
    submission, path, rows = phonon_submission(tmp_path)
    scorer = FinalQuantityScorer(5, verifier=ModalVerifier(transport=transport))
    assert scorer(submission) == 1
    rows[0]["gamma_phonon_energies_eV"][3] = -123
    path.write_text(json.dumps({"results": rows}))
    assert scorer(submission) == 1  # nonmaximum modes are not scored
    rows[0]["gamma_phonon_energies_eV"][-1] *= 1.2
    path.write_text(json.dumps({"results": rows}))
    assert scorer(submission) == 0


def test_phonons_do_not_pass_without_a_reference(tmp_path):
    submission, _, _ = phonon_submission(tmp_path)
    with pytest.raises(EvaluationError):
        check_level1_workflow(5, verification_backend="offline")(submission)


def test_derivative_log_is_scored_on_maximum_not_acoustic_modes(tmp_path):
    from corral_md.scoring.task_5 import ENERGY_FACTOR

    submission, path, rows = phonon_submission(tmp_path)
    for row in rows:
        maximum = row.pop("gamma_phonon_energies_eV")[-1]
        # Correct requested maximum with altered, nonzero acoustic modes.
        energies = np.array([0.001, 0.002, 0.003, 0.01, 0.02, maximum])
        row["hessian_eV_A2"] = np.diag(
            28.0855 * (energies / ENERGY_FACTOR) ** 2
        ).tolist()
    path.write_text(json.dumps({"results": rows}))
    assert (
        FinalQuantityScorer(5, verifier=ModalVerifier(transport=transport))(submission)
        == 1
    )


def copper_log(tmp_path, dense=False):
    rng = np.random.default_rng(1)
    energies = np.zeros(2000)
    for i in range(1, len(energies)):
        energies[i] = 0.95 * energies[i - 1] + rng.normal()
    stride = 20 if dense else 1
    rows = [
        {
            "phase": "production",
            "time_fs": i * 2,
            "potential_energy_eV": float(energies[i]),
        }
        for i in range(0, len(energies), stride)
    ]
    path = tmp_path / "main.jsonl"
    write_log(path, rows)
    artifacts = {"run_logs": {"main": str(path)}}
    if dense:
        np.save(tmp_path / "prod_pe_every_step.npy", energies)
        (tmp_path / "settings.json").write_text(json.dumps({"dt_fs": 2}))
        artifacts.update(
            energy_series=str(tmp_path / "prod_pe_every_step.npy"),
            settings=str(tmp_path / "settings.json"),
        )
    return {"artifacts": artifacts}, energies, path


@pytest.mark.parametrize("dense", [False, True])
def test_correlation_needs_energies_and_times_not_atomic_states(tmp_path, dense):
    submission, energy, _ = copper_log(tmp_path, dense)
    expected = correlation_values(energy, 2)
    report = check_level1_workflow(9)(submission)
    assert report == 1
    computed = check_level1_workflow(9).evaluate(submission)["computed"]["correlation"]
    assert computed == expected


def test_corrupted_correlation_claim_fails(tmp_path):
    submission, _, _ = copper_log(tmp_path)
    submission["results"] = {
        "correlation": {"method": "first_crossing", "value_fs": 10000}
    }
    assert check_level1_workflow(9)(submission) == 0


def test_dense_sidecar_changes_evidence_identity(tmp_path):
    submission, energy, _ = copper_log(tmp_path, dense=True)
    scorer = check_level1_workflow(9)
    first = scorer.evaluate(submission)
    np.save(tmp_path / "prod_pe_every_step.npy", energy + 1)
    second = scorer.evaluate(submission)
    assert first["score"] == second["score"] == 1
    assert first["evidence_sha256"] != second["evidence_sha256"]


@pytest.mark.parametrize(
    "filename",
    ["settings.json", "analysis.json", "prod_pe_every_step.npy", "energy_series.npy"],
)
def test_unsubmitted_neighboring_files_do_not_affect_correlation(tmp_path, filename):
    submission, _, _ = copper_log(tmp_path)
    scorer = check_level1_workflow(9)
    original = scorer.evaluate(submission)
    (tmp_path / filename).write_bytes(b"unrelated file, not scoring evidence")
    assert scorer.evaluate(submission) == original


def test_explicit_correlation_analysis_is_still_checked(tmp_path):
    submission, _, _ = copper_log(tmp_path, dense=True)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"full_step_1e_fs": 10000}))
    submission["artifacts"]["analysis"] = str(analysis)
    assert check_level1_workflow(9)(submission) == 0


def test_constant_energy_correlation_is_undefined(tmp_path):
    submission, _, path = copper_log(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["potential_energy_eV"] = 5
    write_log(path, rows)
    assert check_level1_workflow(9)(submission) == 0


@pytest.mark.parametrize("task", [5, 7, 9, 10])
def test_policy_changes_are_scoped_to_level1(task):
    assert isinstance(
        check_level1_workflow(task, verification_backend="offline"), FinalQuantityScorer
    )
    assert not isinstance(
        check_level2_workflow(task, verification_backend="offline"), FinalQuantityScorer
    )
