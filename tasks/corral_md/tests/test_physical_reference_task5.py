"""Trusted phonon energies also constrain strain derivatives independently."""

import json

import numpy as np
import pytest
from corral_md.scoring import WorkflowScorer
from corral_md.scoring.task_5 import ENERGY_FACTOR, mode_energies
from corral_md.scoring.verification import ModalVerifier

from .test_score_task_5 import CELL, MASS, make_submission, write

REFERENCE_COEFFICIENTS = np.array([-0.01, -0.005])


def harmonic_hessian(iso, z, coefficients):
    energies = np.array([0.025, 0.035, 0.06 + coefficients @ [iso, z]])
    spring = np.diag(MASS / 2 * (energies / ENERGY_FACTOR) ** 2)
    return np.block([[spring, -spring], [-spring, spring]])


def trusted_response(plan, fingerprint):
    frames = []
    for frame in plan.jobs[0]["frames"]:
        cell = np.asarray(frame["cell"])
        iso = cell[0, 1] / CELL[0, 1] - 1
        z = cell[0, 2] / cell[0, 1] - 1
        frames.append(
            {"hessian": harmonic_hessian(iso, z, REFERENCE_COEFFICIENTS).tolist()}
        )
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "release_id": "independent-test-model",
        "jobs": [
            {
                "id": "silicon_hessians",
                "status": "complete",
                "value": {"frames": frames},
            }
        ],
    }


def score(path):
    return WorkflowScorer(
        5, level=2, verifier=ModalVerifier(transport=trusted_response)
    ).evaluate(path)


def submission(tmp_path, factors=(1, 1), *, retained_modes=True):
    path = make_submission(tmp_path)
    log_path = tmp_path / "calculations.json"
    log = json.loads(log_path.read_text())
    coefficients = REFERENCE_COEFFICIENTS * factors
    for row in log["calculations"]:
        matrix = harmonic_hessian(
            row["isotropic_strain"], row["uniaxial_strain"], coefficients
        )
        if retained_modes:
            row.pop("hessian_eV_A2")
            row["mode_energies_eV"] = mode_energies(matrix, [MASS, MASS]).tolist()
        else:
            row["hessian_eV_A2"] = matrix.tolist()
    write(log_path, log)
    manifest = json.loads(path.read_text())
    manifest["results"] = {
        "fit": dict(zip(("isotropic", "uniaxial"), coefficients, strict=True))
    }
    write(path, manifest)
    return path


@pytest.mark.parametrize("retained_modes", [False, True])
def test_reference_fit_comes_from_current_trusted_model(tmp_path, retained_modes):
    report = score(submission(tmp_path, retained_modes=retained_modes))
    assert report["score"] == 1, report
    reference = report["calculated_results"]["physical_reference"]
    assert reference["basis"] == "independent_teacher_model"
    assert reference["release_id"] == "independent-test-model"
    for value, target in zip(
        reference["fit"].values(), REFERENCE_COEFFICIENTS, strict=True
    ):
        assert value["reference"] == pytest.approx(target)
        assert value["allowed_lower"] == pytest.approx(1.1 * target)
        assert value["allowed_upper"] == pytest.approx(0.9 * target)
        assert "source_commit" not in value


@pytest.mark.parametrize("retained_modes", [False, True])
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("factor", [0.89, 1.11])
def test_accurate_individual_modes_do_not_allow_wrong_coefficient(
    tmp_path, retained_modes, component, factor
):
    factors = np.ones(2)
    factors[component] = factor
    report = score(submission(tmp_path, factors, retained_modes=retained_modes))
    assert report["score"] == 0, report
    assert all(check["status"] == "passed" for check in report["checks"][:-1])
    assert "Recomputed" in report["checks"][-1]["detail"]
    assert "physical reference" in report["checks"][-1]["detail"]


@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize(
    ("factor", "expected"), [(0.899999, 0), (0.9, 1), (1.1, 1), (1.100001, 0)]
)
def test_reported_coefficient_has_inclusive_ten_percent_bounds(
    tmp_path, component, factor, expected
):
    # Keep the reconstructed slope safely inside the range and within the
    # existing 1% answer-agreement allowance of the reported boundary value.
    factors = np.ones(2)
    factors[component] = 0.9005 if factor < 1 else 1.0995
    path = submission(tmp_path, factors)
    baseline = score(path)
    assert baseline["score"] == 1, baseline
    name = ("isotropic", "uniaxial")[component]
    target = baseline["calculated_results"]["physical_reference"]["fit"][name][
        "reference"
    ]
    manifest = json.loads(path.read_text())
    manifest["results"]["fit"][name] = factor * target
    write(path, manifest)
    report = score(path)
    assert report["score"] == expected, report
    if not expected:
        assert "Reported" in report["checks"][-1]["detail"]


@pytest.mark.parametrize("component", [0, 1])
def test_in_range_report_cannot_hide_out_of_range_recomputed_slope(tmp_path, component):
    factors = np.ones(2)
    factors[component] = 0.899
    path = submission(tmp_path, factors)
    manifest = json.loads(path.read_text())
    name = ("isotropic", "uniaxial")[component]
    manifest["results"]["fit"][name] = 0.9001 * REFERENCE_COEFFICIENTS[component]
    write(path, manifest)
    report = score(path)
    assert report["score"] == 0, report
    assert "Recomputed" in report["checks"][-1]["detail"]
