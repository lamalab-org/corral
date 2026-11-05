"""The Level 2 palladium ZPE must have the independently fixed magnitude."""

import json

import pytest
from corral_md.scoring.physical_reference import PALLADIUM_ZPE_EV_PER_ATOM
from corral_md.scoring.task_4 import prepare, refined_spectrum

from .test_workflow_task_4 import make_record, save_record, score


def physical_submission(tmp_path, reconstructed, reported, *, unstable=False):
    record = make_record()
    phi, sites, diagnostics = prepare(record)
    _, statistics = refined_spectrum(phi, sites, diagnostics["imaginary_tolerance_eV"])
    stiffness = (reconstructed / statistics["real_mode_energy_eV_per_atom"]) ** 2
    if unstable:
        stiffness *= -1
    record["force_constants_eV_A2"] *= stiffness
    record["forces_eV_A"] *= stiffness
    reference_energy = record["energies_eV"][0]
    record["energies_eV"] = reference_energy + stiffness * (
        record["energies_eV"] - reference_energy
    )
    save_record(tmp_path / "phonons.npz", record)
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps(
            {"artifacts": {"force_log": "phonons.npz"}, "results": {"zpe": reported}}
        )
    )
    return path


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(0.899999, 0), (0.9, 1), (1, 1), (1.1, 1), (1.100001, 0), (0.01, 0), (100, 0)],
)
def test_matching_zpe_requires_physical_magnitude(tmp_path, factor, expected):
    value = factor * PALLADIUM_ZPE_EV_PER_ATOM
    path = physical_submission(tmp_path, value, value)
    report = score(path)
    assert report["score"] == expected, report
    assert report["calculated_results"]["zpe_eV_per_atom"] == pytest.approx(value)
    if not expected:
        assert "physical reference" in report["checks"][-2]["detail"]


@pytest.mark.parametrize(
    ("reconstructed_factor", "reported_factor"),
    [(0.9, 0.899999), (0.899999, 0.9), (1.1, 1.100001), (1.100001, 1.1)],
)
def test_both_reconstructed_and_reported_zpe_must_pass(
    tmp_path, reconstructed_factor, reported_factor
):
    path = physical_submission(
        tmp_path,
        reconstructed_factor * PALLADIUM_ZPE_EV_PER_ATOM,
        reported_factor * PALLADIUM_ZPE_EV_PER_ATOM,
    )
    report = score(path)
    assert report["score"] == 0, report
    assert "physical reference" in report["checks"][-2]["detail"]


@pytest.mark.parametrize("unstable", [False, True])
def test_null_zpe_does_not_pass(tmp_path, unstable):
    path = physical_submission(
        tmp_path, PALLADIUM_ZPE_EV_PER_ATOM, None, unstable=unstable
    )
    report = score(path)
    assert report["score"] == 0, report


@pytest.mark.parametrize("unstable", [False, True])
def test_level1_does_not_apply_the_new_zpe_gate(tmp_path, unstable):
    path = physical_submission(
        tmp_path, 2 * PALLADIUM_ZPE_EV_PER_ATOM, None, unstable=unstable
    )
    report = score(path, level=1)
    assert report["score"] == 1, report
