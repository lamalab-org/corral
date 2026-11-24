"""Independent harmonic fixtures exercise the compact Task 5 contract."""

import copy
import json

import numpy as np
import pytest
from corral_md.scoring import EvaluationError, WorkflowScorer
from corral_md.scoring.final_quantities import FinalQuantityScorer
from corral_md.scoring.task_5 import ENERGY_FACTOR, mode_energies
from corral_md.scoring.verification import ModalVerifier

CELL = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))
MASS = 28.0855


def hessian(iso, z):
    energies = np.array([0.025, 0.035, 0.06 + 0.4 * iso + 0.2 * z + 0.5 * iso * z])
    spring = np.diag(MASS / 2 * (energies / ENERGY_FACTOR) ** 2)
    return np.block([[spring, -spring], [-spring, spring]])


def force_pairs(matrix, step):
    forces = (-step * matrix.T).reshape(6, 2, 3)
    return {
        "displacement_A": step,
        "forces_plus_eV_A": forces.tolist(),
        "forces_minus_eV_A": (-forces).tolist(),
    }


def write(path, data):
    path.write_text(json.dumps(data))


def make_submission(root, level=2, method="analytic_hessian"):
    rows, values, pairs = [], [], []
    for iso in np.arange(-2, 3) / 100:
        for z in [0.0] if level == 1 else np.arange(-2, 3) / 100:
            cell = CELL * (1 + iso) @ np.diag([1, 1, 1 + z])
            row = {
                "isotropic_strain": float(iso),
                "uniaxial_strain": float(z),
                "structure": {
                    "numbers": [14, 14],
                    "pbc": [True] * 3,
                    "cell": cell.tolist(),
                    "positions": (
                        np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
                    ).tolist(),
                    "masses": [MASS, MASS],
                },
                "method": method,
            }
            matrix = hessian(iso, z)
            if method == "analytic_hessian":
                row["hessian_eV_A2"] = matrix.tolist()
            else:
                row.update(force_pairs(matrix, 0.01))
                row["sensitivity"] = force_pairs(matrix, 0.005)
            rows.append(row)
            pairs.append([iso, z])
            values.append(0.06 + 0.4 * iso + 0.2 * z + 0.5 * iso * z)
    log = {
        "model": "teacher.model",
        "default_dtype": "float64",
        "dispersion": False,
        "calculations": rows,
    }
    results = {"highest_positive_energies_eV": values}
    if level == 2:
        results["fit"] = {
            "intercept_eV": 0.06,
            "isotropic_coefficient_eV_per_strain": 0.4,
            "uniaxial_coefficient_eV_per_strain": 0.2,
            "residuals_eV": [0.5 * iso * z for iso, z in pairs],
        }
    write(root / "calculations.json", log)
    path = root / "manifest.json"
    write(
        path,
        {"artifacts": {"calculation_log": "calculations.json"}, "results": results},
    )
    return path


def transport(plan, fingerprint):
    # Derive reference values from geometry, never from the submitted matrices.
    frames = []
    for frame in plan.jobs[0]["frames"]:
        cell = np.array(frame["cell"])
        iso = cell[0, 1] / CELL[0, 1] - 1
        z = cell[0, 2] / cell[0, 1] - 1
        frames.append({"hessian": hessian(iso, z).tolist()})
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "release_id": "test-reference",
        "jobs": [
            {
                "id": "silicon_hessians",
                "status": "complete",
                "value": {"frames": frames},
            }
        ],
    }


def scorer(level=2, backend=transport):
    verifier = ModalVerifier(transport=backend)
    if level == 1:
        return FinalQuantityScorer(5, verifier=verifier)
    return WorkflowScorer(5, level=2, verifier=verifier)


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("method", ["analytic_hessian", "central_difference"])
def test_minimal_log_reconstructs_expected_answers(tmp_path, level, method):
    path = make_submission(tmp_path, level, method)
    if level == 1:
        manifest = json.loads(path.read_text())
        manifest.pop("results")
        write(path, manifest)
    report = scorer(level).evaluate(path)
    assert report["score"] == 1, report
    assert report["possible_points"] == 100
    calculated = report["computed" if level == 1 else "calculated_results"]
    assert len(calculated["highest_positive_energies_eV"]) == (5 if level == 1 else 25)
    assert ("fit" in calculated) is (level == 2)
    assert "artifacts" in json.loads(path.read_text())


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("payload", ["modes", "hessian", "forces"])
@pytest.mark.parametrize("label", ["missing", "custom", "misleading"])
def test_phonon_payload_does_not_require_method_configuration(
    tmp_path, level, payload, label
):
    path = make_submission(
        tmp_path,
        level,
        method="central_difference" if payload == "forces" else "analytic_hessian",
    )
    log_path = tmp_path / "calculations.json"
    log = json.loads(log_path.read_text())
    for row in log["calculations"]:
        if payload == "modes":
            row["mode_energies_eV"] = mode_energies(
                np.asarray(row.pop("hessian_eV_A2")), row["structure"]["masses"]
            ).tolist()
        if label == "missing":
            row.pop("method")
        elif label == "custom":
            row["method"] = "finite_displacement_central_0.005A"
        else:
            row["method"] = (
                "analytic_hessian" if payload == "modes" else "retained_modes"
            )
    write(log_path, log)
    report = scorer(level).evaluate(path)
    assert report["score"] == 1, report


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("wrong_modes", [False, True])
def test_retained_modes_take_precedence_and_cannot_fall_back_to_derivatives(
    tmp_path, level, wrong_modes
):
    path = make_submission(tmp_path, level)
    log_path = tmp_path / "calculations.json"
    log = json.loads(log_path.read_text())
    for row in log["calculations"]:
        matrix = np.asarray(row["hessian_eV_A2"])
        row["mode_energies_eV"] = (
            mode_energies(matrix, row["structure"]["masses"])
            * (1.5 if wrong_modes else 1)
        ).tolist()
        if not wrong_modes:
            row["hessian_eV_A2"] = (matrix * 4).tolist()
    write(log_path, log)
    if wrong_modes:
        # Keep the reported fit consistent, so only the independent reference
        # can reject the inaccurate spectrum despite its correct extra Hessian.
        manifest = json.loads(path.read_text())
        manifest["results"].pop("highest_positive_energies_eV")
        if level == 2:
            for key, value in manifest["results"]["fit"].items():
                manifest["results"]["fit"][key] = (np.asarray(value) * 1.5).tolist()
        write(path, manifest)
    report = scorer(level).evaluate(path)
    assert report["score"] == (0 if wrong_modes else 1), report
    if wrong_modes:
        assert [c["name"] for c in report["checks"] if c["status"] == "failed"] == [
            "final_quantities" if level == 1 else "independent_hessian_accuracy"
        ]


@pytest.mark.parametrize("method", [None, "retained_modes", "custom_solver"])
def test_force_pair_sensitivity_is_validated_without_method_dispatch(tmp_path, method):
    path = make_submission(tmp_path, method="central_difference")
    log_path = tmp_path / "calculations.json"
    log = json.loads(log_path.read_text())
    for row in log["calculations"]:
        if method is None:
            row.pop("method")
        else:
            row["method"] = method
    log["calculations"][0]["sensitivity"].update(
        force_pairs(hessian(-0.02, -0.02) * 1.1, 0.005)
    )
    write(log_path, log)
    report = scorer().evaluate(path)
    assert report["score"] == 0, report
    assert [c["name"] for c in report["checks"] if c["status"] == "failed"] == [
        "phonon_reconstruction_and_sensitivity"
    ]


def test_level2_needs_fit_parameters_without_repeated_tables(tmp_path):
    path = make_submission(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["results"].pop("highest_positive_energies_eV")
    manifest["results"]["fit"].pop("residuals_eV")
    manifest["results"]["fit"].pop("intercept_eV")
    write(path, manifest)
    report = scorer().evaluate(path)
    assert report["score"] == 1, report
    assert len(report["calculated_results"]["highest_positive_energies_eV"]) == 25
    assert len(report["calculated_results"]["fit"]["residuals_eV"]) == 25
    manifest["results"]["fit"].pop("isotropic_coefficient_eV_per_strain")
    write(path, manifest)
    assert scorer().evaluate(path)["score"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_pair",
        "duplicate_pair",
        "wrong_z_axis",
        "relaxed",
        "mass",
        "model",
        "dispersion",
        "asymmetric",
        "acoustic",
        "nan",
        "force_sign",
        "sensitivity_step",
        "sensitivity_forces",
        "energy",
        "coefficient",
        "intercept",
        "residual",
    ],
)
def test_invalid_evidence_and_answers_fail(tmp_path, mutation):
    path = make_submission(tmp_path, method="central_difference")
    log = json.loads((tmp_path / "calculations.json").read_text())
    manifest = json.loads(path.read_text())
    row = log["calculations"][0]
    if mutation == "missing_pair":
        log["calculations"].pop()
    elif mutation == "duplicate_pair":
        log["calculations"][1] = copy.deepcopy(row)
    elif mutation == "wrong_z_axis":
        row["structure"]["cell"][0][2] *= 1.02
    elif mutation == "relaxed":
        row["structure"]["positions"][1][0] += 0.001
    elif mutation == "mass":
        row["structure"]["masses"] = [29, 29]
    elif mutation == "model":
        log["model"] = "student.model"
    elif mutation == "dispersion":
        log["dispersion"] = True
    elif mutation in {"asymmetric", "acoustic", "nan"}:
        row["forces_plus_eV_A"][0][0][1 if mutation == "asymmetric" else 0] = (
            float("nan") if mutation == "nan" else 10
        )
    elif mutation == "force_sign":
        row["forces_plus_eV_A"], row["forces_minus_eV_A"] = (
            row["forces_minus_eV_A"],
            row["forces_plus_eV_A"],
        )

    elif mutation == "sensitivity_step":
        row["sensitivity"]["displacement_A"] = 0.01
    elif mutation == "sensitivity_forces":
        row["sensitivity"].update(force_pairs(hessian(-0.02, -0.02) * 1.1, 0.005))
    elif mutation == "energy":
        manifest["results"]["highest_positive_energies_eV"][0] *= 1000
    else:
        key = {
            "coefficient": "isotropic_coefficient_eV_per_strain",
            "intercept": "intercept_eV",
            "residual": "residuals_eV",
        }[mutation]
        manifest["results"]["fit"][key] = [0.01] * 25 if mutation == "residual" else 9
    write(tmp_path / "calculations.json", log)
    write(path, manifest)
    report = scorer().evaluate(path)
    assert report["score"] == 0, report
    assert report["possible_points"] == 100


def test_self_consistent_wrong_derivatives_fail_independent_check(tmp_path):
    path = make_submission(tmp_path)
    log = json.loads((tmp_path / "calculations.json").read_text())
    manifest = json.loads(path.read_text())
    for row in log["calculations"]:
        row["hessian_eV_A2"] = (np.array(row["hessian_eV_A2"]) * 4).tolist()
    manifest["results"]["highest_positive_energies_eV"] = (
        np.array(manifest["results"]["highest_positive_energies_eV"]) * 2
    ).tolist()
    for key, value in manifest["results"]["fit"].items():
        manifest["results"]["fit"][key] = (np.array(value) * 2).tolist()
    write(tmp_path / "calculations.json", log)
    write(path, manifest)
    report = scorer().evaluate(path)
    assert report["score"] == 0
    assert [c["name"] for c in report["checks"] if c["status"] == "failed"] == [
        "independent_hessian_accuracy"
    ]


def test_imaginary_modes_do_not_compete_with_positive_modes():
    spring = np.diag(
        MASS / 2 * (np.array([0.12, 0.04, 0.05]) / ENERGY_FACTOR) ** 2 * [-1, 1, 1]
    )
    modes = mode_energies(
        np.block([[spring, -spring], [-spring, spring]]), [MASS, MASS]
    )
    assert modes.min() == pytest.approx(-0.12)
    assert modes.max() == pytest.approx(0.05)


def test_reordered_log_and_wrapped_translated_atoms(tmp_path):
    path = make_submission(tmp_path)
    log = json.loads((tmp_path / "calculations.json").read_text())
    manifest = json.loads(path.read_text())
    log["calculations"].reverse()
    for row in log["calculations"]:
        frame = row["structure"]
        frame["positions"] = (
            np.array(frame["positions"])[::-1] + [1, 2, 3] + np.array(frame["cell"])[0]
        ).tolist()
    manifest["results"]["highest_positive_energies_eV"].reverse()
    manifest["results"]["fit"]["residuals_eV"].reverse()
    write(tmp_path / "calculations.json", log)
    write(path, manifest)
    assert scorer().evaluate(path)["score"] == 1


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reference_plan_or_live_fallback(tmp_path, level, dtype):
    path = make_submission(tmp_path, level)
    log = json.loads((tmp_path / "calculations.json").read_text())
    log["default_dtype"] = dtype
    write(tmp_path / "calculations.json", log)

    def inspect(plan, fingerprint):
        job = plan.jobs[0]
        assert job["operation"] == (
            "reference" if level == 1 or dtype == "float64" else "mace"
        )
        assert len(job["frames"]) == (5 if level == 1 else 25)
        assert all("hessian" not in frame for frame in job["frames"])
        return transport(plan, fingerprint)

    assert scorer(level, inspect).evaluate(path)["score"] == 1


def test_unavailable_backend_is_an_error_but_wrong_answers_fail(tmp_path):
    path = make_submission(tmp_path)
    grader = WorkflowScorer(5)
    with pytest.raises(EvaluationError) as caught:
        grader.evaluate(path)
    report = caught.value.report
    assert report["unsupported_checks"] == ["independent_hessian_accuracy"]
    with pytest.raises(EvaluationError):
        grader.score_submission(path)
    manifest = json.loads(path.read_text())
    manifest["results"]["fit"]["intercept_eV"] = 4
    write(path, manifest)
    assert grader.score_submission(path) == 0


@pytest.mark.parametrize(
    "failure", ["wrong_identity", "missing_job", "wrong_id", "failed", "unsupported"]
)
def test_backend_protocol(tmp_path, failure):
    path = make_submission(tmp_path)

    def bad_response(plan, fingerprint):
        response = transport(plan, fingerprint)
        if failure == "wrong_identity":
            response["challenge"] = "other"
        elif failure == "missing_job":
            response["jobs"] = []
        elif failure == "wrong_id":
            response["jobs"][0]["id"] = "other"
        else:
            response["jobs"][0]["status"] = failure
        return response

    if failure == "unsupported":
        with pytest.raises(EvaluationError):
            scorer(backend=bad_response).evaluate(path)
    else:
        assert scorer(backend=bad_response).evaluate(path)["score"] == 0


def test_log_must_stay_inside_workspace(tmp_path):
    path = make_submission(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["calculation_log"] = "../outside.json"
    write(path, manifest)
    assert scorer().evaluate(path)["score"] == 0


def test_changed_evidence_during_verification_is_rejected(tmp_path):
    path = make_submission(tmp_path)

    def mutate(plan, fingerprint):
        response = transport(plan, fingerprint)
        (tmp_path / "calculations.json").write_text("{}")
        return response

    with pytest.raises(RuntimeError, match="changed during grading"):
        scorer(backend=mutate).evaluate(path)


def test_sensitivity_selection_tolerates_strain_roundoff(tmp_path):
    path = make_submission(tmp_path, method="central_difference")
    log = json.loads((tmp_path / "calculations.json").read_text())
    row = log["calculations"][0]
    row["isotropic_strain"] += 1e-12
    del row["sensitivity"]
    write(tmp_path / "calculations.json", log)
    assert scorer().evaluate(path)["score"] == 0


def test_rounding_is_allowed(tmp_path):
    path = make_submission(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["results"]["fit"]["intercept_eV"] *= 1.005
    write(path, manifest)
    assert scorer().evaluate(path)["score"] == 1


def test_half_step_force_records_are_optional_when_independent_hessians_agree(tmp_path):
    path = make_submission(tmp_path, method="central_difference")
    log = json.loads((tmp_path / "calculations.json").read_text())
    for row in log["calculations"]:
        row.pop("sensitivity", None)
    write(tmp_path / "calculations.json", log)
    assert scorer().evaluate(path)["score"] == 1


@pytest.mark.parametrize("level", [1, 2])
def test_retained_modes_and_flat_geometry_are_verified_independently(tmp_path, level):
    path = make_submission(tmp_path, level=level)
    log = json.loads((tmp_path / "calculations.json").read_text())
    converted = []
    for row in log["calculations"]:
        energies = mode_energies(
            np.asarray(row["hessian_eV_A2"]), row["structure"]["masses"]
        )
        converted.append(
            {
                "isotropic_strain": row["isotropic_strain"],
                "uniaxial_strain": row["uniaxial_strain"],
                **row["structure"],
                "gamma_mode_energies_eV": energies.tolist(),
            }
        )
    write(tmp_path / "calculations.json", {"calculations": converted})
    report = scorer(level=level).evaluate(path)
    assert report["score"] == 1, report
    converted[0]["gamma_mode_energies_eV"][-1] *= 2
    write(tmp_path / "calculations.json", {"calculations": converted})
    assert scorer(level=level).evaluate(path)["score"] == 0
