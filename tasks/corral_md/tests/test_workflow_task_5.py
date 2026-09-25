"""Numerical evidence tests for the artifact-only strained-silicon scorer."""

import copy
import json
import pickle
import subprocess

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.task_5 import ENERGY_FACTOR, evaluate

CELL = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))
MASS = 28.0855


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def load(path):
    return json.loads(path.read_text())


def frame(cell):
    return {
        "symbols": ["Si", "Si"],
        "cell": cell.tolist(),
        "positions": (np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell).tolist(),
        "pbc": True,
        "masses": [MASS, MASS],
    }


def calculation(index, iso, z, step=0.005, delta=0):
    energies = (
        np.array(
            [
                0.04 + 0.2 * iso + 0.1 * z,
                0.05 + 0.3 * iso + 0.15 * z,
                0.06 + 0.4 * iso + 0.2 * z + 0.5 * iso * z,
            ]
        )
        + delta
    )
    spring = np.diag(MASS / 2 * (energies / ENERGY_FACTOR) ** 2)
    hessian = np.block([[spring, -spring], [-spring, spring]])
    displacement = np.eye(6) * step
    forces = -displacement @ hessian.T
    return {
        "id": f"strain_{index}",
        "structure_index": index,
        "isotropic_strain": float(iso),
        "uniaxial_strain": float(z),
        "method": "central_difference",
        "dof_order": [[a, d] for a in range(2) for d in range(3)],
        "symmetrization": "average_transpose",
        "acoustic_sum_rule": "projection",
        "plus_displacements_A": displacement.reshape(6, 2, 3).tolist(),
        "minus_displacements_A": (-displacement).reshape(6, 2, 3).tolist(),
        "plus_forces_eV_A": forces.reshape(6, 2, 3).tolist(),
        "minus_forces_eV_A": (-forces).reshape(6, 2, 3).tolist(),
        "force_constants_eV_A2": hessian.tolist(),
        "mode_real_eV": [0, 0, 0, *energies.tolist()],
        "mode_imaginary_eV": [0] * 6,
        "highest_positive_energy_eV": float(energies[-1]),
    }


@pytest.fixture
def submission(tmp_path):
    records, structures = [], []
    for iso in np.arange(-2, 3) / 100:
        for z in np.arange(-2, 3) / 100:
            index = len(records)
            records.append(calculation(index, iso, z))
            structures.append(frame(CELL * (1 + iso) @ np.diag([1, 1, 1 + z])))
    alternative = [
        calculation(
            i,
            records[i]["isotropic_strain"],
            records[i]["uniaxial_strain"],
            step=0.01,
            delta=1e-6,
        )
        for i in (0, 12, 24)
    ]
    artifacts = {
        "reference_structure": write(tmp_path / "reference.json", [frame(CELL)]),
        "strained_structures": write(tmp_path / "structures.json", structures),
        "strain_calculations": write(tmp_path / "calculations.json", records),
        "sensitivity_calculations": write(tmp_path / "sensitivity.json", alternative),
    }
    report = tmp_path / "report.md"
    report.write_text(
        "Doubling displacements alters the highest mode by 1 micro-eV, small relative to the strain response."
    )
    script = tmp_path / "simulate.py"
    script.write_text("raise RuntimeError('Never execute submitted code')\n")
    values = [record["highest_positive_energy_eV"] for record in records]
    results = {
        "fit": {
            "isotropic_coefficient_eV_per_strain": 0.4,
            "uniaxial_coefficient_eV_per_strain": 0.2,
            "intercept_eV": 0.06,
            "calculation_ids": [record["id"] for record in records],
            "residuals_eV": [
                0.5 * record["isotropic_strain"] * record["uniaxial_strain"]
                for record in records
            ],
        },
        "sensitivity": {
            "calculation_ids": [record["id"] for record in alternative],
            "delta_energy_eV": [1e-6] * 3,
            "max_abs_delta_eV": 1e-6,
            "relative_to_strain_range": 1e-6 / np.ptp(values),
        },
    }
    settings = {
        "masses_amu": [MASS, MASS],
        "model": "MACE-MP-0",
        "energy_unit": "eV",
        "force_unit": "eV/Angstrom",
        "force_constant_unit": "eV/Angstrom^2",
    }
    manifest = {
        "results": results,
        "artifacts": artifacts,
        "settings": write(tmp_path / "settings.json", settings),
        "report": str(report),
        "scripts": [str(script)],
    }
    write(tmp_path / "manifest.json", manifest)
    return tmp_path / "manifest.json"


def test_sensitivity_credit_does_not_require_a_report(submission):
    manifest = load(submission)
    manifest.pop("report")
    write(submission, manifest)
    result = score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_named_dof_order_and_transpose_average_are_equivalent(submission):
    for name in ("calculations.json", "sensitivity.json"):
        path = submission.parent / name
        records = load(path)
        for record in records:
            record["dof_order"] = [
                f"Si{atom}_{'xyz'[axis]}" for atom, axis in record["dof_order"]
            ]
            record["symmetrization"] = "Hessian transpose average"
            record["acoustic_sum_rule"] = "orthogonal translational projection"
        write(path, records)

    assert score(submission).score == pytest.approx(0.9)


def score(path):
    rubric = Rubric(5)
    evaluate(Evidence(path), rubric)
    return rubric


def check(rubric, name):
    return next(item for item in rubric.checks if item["name"] == name)


def test_model_path_is_a_valid_teacher_identity(submission):
    path = submission.parent / "settings.json"
    settings = load(path)
    settings["model"] = "/workspace/models/teacher.model"
    write(path, settings)
    assert (
        check(score(submission), "recorded_model_units_and_conventions")["status"]
        == "passed"
    )
    settings["model"] = "/workspace/models/student.model"
    write(path, settings)
    assert (
        check(score(submission), "recorded_model_units_and_conventions")["status"]
        == "failed"
    )


def test_consistent_physical_spring_system_gets_all_task_points_without_execution(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Scoring executed code or evaluated a calculator")

    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(item["points"] for item in result.checks) == 90
    assert check(result, "execution_provenance")["status"] == "unverified"


def test_conversion_from_ev_angstrom_squared_amu_to_phonon_ev():
    # Independently tabulated 1 sqrt(eV/A^2/amu) is approximately 15.6333 THz.
    assert pytest.approx(0.06465415, rel=1e-6) == ENERGY_FACTOR


@pytest.mark.parametrize(
    ("mutation", "failed"),
    [
        ("wrong_z_axis", "cartesian_strains_without_relaxation"),
        ("relaxed_atom", "cartesian_strains_without_relaxation"),
        ("reference_lattice", "reference_primitive_structure"),
        ("duplicate_pair", "complete_strain_grid_and_mapping"),
        ("wrong_mass", "natural_silicon_masses"),
        ("wrong_forces", "force_constants_reconstructed_from_raw_evidence"),
        ("bad_displacement_pairs", "force_constants_reconstructed_from_raw_evidence"),
        ("incomplete_displacements", "force_constants_reconstructed_from_raw_evidence"),
        ("saved_hessian", "force_constants_reconstructed_from_raw_evidence"),
        ("wrong_mode", "highest_positive_real_mode"),
        ("wrong_all_modes", "all_mode_energies_and_imaginary_modes"),
        ("wrong_coefficients", "unweighted_ols_strain_coefficients"),
        ("wrong_intercept", "unweighted_ols_intercept"),
        ("wrong_residuals", "all_25_fit_residuals"),
        ("exclude_fit_point", "unweighted_ols_strain_coefficients"),
    ],
)
def test_rejects_inconsistent_scientific_claims(submission, mutation, failed):
    directory = submission.parent
    records = load(directory / "calculations.json")
    if mutation == "wrong_z_axis":
        structures = load(directory / "structures.json")
        record = records[1]
        wrong = (
            CELL
            * (1 + record["isotropic_strain"])
            @ np.diag([1, 1 + record["uniaxial_strain"], 1])
        )
        structures[1] = frame(wrong)
        write(directory / "structures.json", structures)
    elif mutation == "relaxed_atom":
        structures = load(directory / "structures.json")
        structures[0]["positions"][1][0] += 0.01
        write(directory / "structures.json", structures)
    elif mutation == "reference_lattice":
        write(directory / "reference.json", [frame(CELL * 1.01)])
    elif mutation == "wrong_mass":
        settings = load(directory / "settings.json")
        settings["masses_amu"] = [29, 29]
        write(directory / "settings.json", settings)
    elif mutation.startswith("wrong_") and mutation in {
        "wrong_coefficients",
        "wrong_intercept",
        "wrong_residuals",
    }:
        manifest = load(submission)
        key = {
            "wrong_coefficients": "isotropic_coefficient_eV_per_strain",
            "wrong_intercept": "intercept_eV",
            "wrong_residuals": "residuals_eV",
        }[mutation]
        manifest["results"]["fit"][key] = [0.1] * 25 if key == "residuals_eV" else 20
        write(submission, manifest)
    elif mutation == "exclude_fit_point":
        manifest = load(submission)
        manifest["results"]["fit"]["calculation_ids"].pop()
        write(submission, manifest)
    elif mutation == "false_sensitivity":
        manifest = load(submission)
        manifest["results"]["sensitivity"]["delta_energy_eV"] = [1] * 3
        write(submission, manifest)
    elif mutation == "copied_sensitivity":
        copied = copy.deepcopy(records[:1])
        copied[0]["declared_displacement_A"] = 0.01
        write(directory / "sensitivity.json", copied)
    elif mutation == "sensitivity_wrong_raw_data":
        alternate = load(directory / "sensitivity.json")
        alternate[0]["plus_forces_eV_A"][0][0][0] *= 2
        write(directory / "sensitivity.json", alternate)
    else:
        if mutation == "duplicate_pair":
            records[0]["uniaxial_strain"] = records[1]["uniaxial_strain"]
        elif mutation == "wrong_forces":
            records[0]["plus_forces_eV_A"][0][0][0] *= 2
        elif mutation == "bad_displacement_pairs":
            records[0]["minus_displacements_A"][0][0][0] *= 2
        elif mutation == "incomplete_displacements":
            for key in (
                "plus_displacements_A",
                "minus_displacements_A",
                "plus_forces_eV_A",
                "minus_forces_eV_A",
            ):
                records[0][key].pop()
        elif mutation == "saved_hessian":
            records[0]["force_constants_eV_A2"][0][0] *= 2
        elif mutation == "wrong_mode":
            records[0]["highest_positive_energy_eV"] = records[0]["mode_real_eV"][3]
        elif mutation == "wrong_all_modes":
            records[0]["mode_real_eV"][3] += 0.01
        write(directory / "calculations.json", records)
    result = score(submission)
    assert check(result, failed)["status"] == "failed", result.checks
    assert result.score < 0.9
    if mutation in {"wrong_z_axis", "relaxed_atom", "reference_lattice", "wrong_mass"}:
        assert check(result, "unweighted_ols_strain_coefficients")["earned"] == 0
        assert check(result, "highest_positive_real_mode")["earned"] == 0


def test_accepts_arbitrary_record_dof_mode_and_displacement_pair_order(submission):
    directory = submission.parent
    order = np.array([5, 1, 4, 2, 0, 3])
    for name in ("calculations.json", "sensitivity.json"):
        records = load(directory / name)
        for record in records:
            matrix = np.asarray(record["force_constants_eV_A2"])
            record["force_constants_eV_A2"] = matrix[np.ix_(order, order)].tolist()
            record["dof_order"] = [[int(i // 3), "xyz"[i % 3]] for i in order]
            for key in (
                "plus_displacements_A",
                "minus_displacements_A",
                "plus_forces_eV_A",
                "minus_forces_eV_A",
            ):
                record[key] = record[key][::-1]
            record["mode_real_eV"] = record["mode_real_eV"][::-1]
            record["mode_imaginary_eV"] = record["mode_imaginary_eV"][::-1]
        write(directory / name, records[::-1])
    assert score(submission).score == pytest.approx(0.9)


def test_accepts_global_translation_atom_permutation_and_periodic_wrapping(submission):
    directory = submission.parent
    for name in ("reference.json", "structures.json"):
        frames = load(directory / name)
        for data in frames:
            data["positions"] = (
                np.asarray(data["positions"])[::-1]
                + np.asarray(data["cell"])[0]
                + [1, 2, 3]
            ).tolist()
        write(directory / name, frames)
    # Equal-mass two-atom springs are invariant under exchanging the two atoms.
    assert score(submission).score == pytest.approx(0.9)


def test_analytic_derivatives_npy_and_changed_processing_convention(submission):
    directory = submission.parent
    manifest = load(submission)
    records = load(directory / "calculations.json")
    for record in records:
        record["method"] = "analytic_hessian"
        record["derivative_kind"] = "force_jacobian"
        matrix = -np.asarray(record["force_constants_eV_A2"])
        path = directory / f"raw_{record['id']}.npy"
        np.save(path, matrix)
        artifact = f"derivative_{record['id']}"
        manifest["artifacts"][artifact] = str(path)
        record["raw_derivatives_eV_A2"] = {"artifact": artifact}
    alternative = copy.deepcopy(records[:1])
    alternative[0]["acoustic_sum_rule"] = "none"
    write(directory / "calculations.json", records)
    write(directory / "sensitivity.json", alternative)
    manifest["results"]["sensitivity"] = {
        "calculation_ids": [records[0]["id"]],
        "delta_energy_eV": [0],
        "max_abs_delta_eV": 0,
        "relative_to_strain_range": 0,
    }
    write(submission, manifest)
    assert score(submission).score == pytest.approx(0.9)


def test_imaginary_mode_is_not_made_positive_by_absolute_value(submission):
    directory = submission.parent
    records = load(directory / "calculations.json")
    row = records[0]
    row["method"] = "analytic_hessian"
    row["derivative_kind"] = "energy_hessian"
    # First optical mode is imaginary with a larger magnitude than any real
    # mode: taking abs(eigenvalue) would select the wrong mode.
    spring = np.diag(
        MASS / 2 * (np.array([0.12, 0.04, 0.05]) / ENERGY_FACTOR) ** 2 * [-1, 1, 1]
    )
    matrix = np.block([[spring, -spring], [-spring, spring]])
    row["raw_derivatives_eV_A2"] = matrix.tolist()
    row["force_constants_eV_A2"] = matrix.tolist()
    row["mode_real_eV"] = [0, 0, 0, 0, 0.04, 0.05]
    row["mode_imaginary_eV"] = [0.12, 0, 0, 0, 0, 0]
    row["highest_positive_energy_eV"] = 0.05
    write(directory / "calculations.json", records)
    result = score(submission)
    assert check(result, "all_mode_energies_and_imaginary_modes")["status"] == "passed"
    assert check(result, "highest_positive_real_mode")["status"] == "passed"
    row["highest_positive_energy_eV"] = 0.12
    row["mode_real_eV"][0] = 0.12
    row["mode_imaginary_eV"][0] = 0
    write(directory / "calculations.json", records)
    result = score(submission)
    assert check(result, "highest_positive_real_mode")["status"] == "failed"
    assert check(result, "all_mode_energies_and_imaginary_modes")["status"] == "failed"


def test_missing_malformed_and_unsupported_evidence_keep_explicit_denominator(
    submission,
):
    directory = submission.parent
    records = load(directory / "calculations.json")
    records[0]["method"] = "unimplemented_alternative"
    write(directory / "calculations.json", records)
    result = score(submission)
    assert (
        check(result, "force_constants_reconstructed_from_raw_evidence")["status"]
        == "unverified"
    )
    assert check(result, "unweighted_ols_intercept")["status"] == "passed"
    assert check(result, "all_mode_energies_and_imaginary_modes")["status"] == "passed"
    assert result.score is None
    assert sum(item["points"] for item in result.checks) == 90
    (directory / "calculations.json").write_text("not valid JSON")
    (directory / "structures.json").unlink()
    result = score(submission)
    assert result.score < 0.2
    assert sum(item["points"] for item in result.checks) == 90


@pytest.mark.parametrize("corrupt", [None, "weights", "forces"])
def test_forward_force_stencil_and_corruption(submission, corrupt):
    records = load(submission.parent / "calculations.json")
    for row in records:
        displacement = np.array(row["plus_displacements_A"])
        force = np.array(row["plus_forces_eV_A"])
        step = displacement[0, 0, 0]
        # A forward stencil includes an undisplaced force evaluation and need
        # not include any negative displacement or paired evaluation.
        row.update(
            method="finite_difference",
            displacements_A=np.concatenate(
                [np.zeros((1, 2, 3)), displacement]
            ).tolist(),
            forces_eV_A=np.concatenate([np.zeros((1, 2, 3)), force]).tolist(),
            derivative_weights_A_inv=np.column_stack([-np.ones(6), np.eye(6)]).tolist(),
        )
        row["derivative_weights_A_inv"] = (
            np.array(row["derivative_weights_A_inv"]) / step
        ).tolist()
        for name in (
            "plus_displacements_A",
            "minus_displacements_A",
            "plus_forces_eV_A",
            "minus_forces_eV_A",
        ):
            del row[name]
    if corrupt == "weights":
        records[0]["derivative_weights_A_inv"][0][0] += 1
    elif corrupt == "forces":
        records[0]["forces_eV_A"][1][0][0] += 1
    write(submission.parent / "calculations.json", records)
    result = score(submission)
    if corrupt is None:
        assert result.score == pytest.approx(0.9), result.checks
    else:
        assert (
            check(result, "force_constants_reconstructed_from_raw_evidence")["status"]
            == "failed"
        )
        assert check(result, "unweighted_ols_intercept")["status"] == "failed"


def test_empty_submission_has_no_task_credit(tmp_path):
    path = tmp_path / "manifest.json"
    write(path, {"artifacts": {}, "results": {}})
    result = score(path)
    assert result.score == 0
    assert sum(item["points"] for item in result.checks) == 90
