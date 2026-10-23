"""Task 4 recording, independent phonon reconstruction, and failure cases."""

import copy
import itertools
import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import WorkflowScorer
from corral_md.scoring.common import Evidence
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.task_4 import NODES, build_plan, prepare, read_record
from corral_md.scoring.verification import ModalVerifier


def neighbors():
    sites = np.indices((5,) * 3).reshape(3, -1).T
    sites = (sites + 2) % 5 - 2
    cell = 3.89 / 2 * (np.ones((3, 3)) - np.eye(3))
    nn = np.isclose(np.linalg.norm(sites @ cell, axis=1), 3.89 / np.sqrt(2))
    return sites, nn


def independent_energies(q, sign=1):
    sites, nn = neighbors()
    eigenvalue = (
        sign
        * np.sum(1 - np.cos(2 * np.pi * np.asarray(q) @ sites[nn].T), axis=1)
        / 106.42
    )
    factor = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    return np.repeat(
        (np.sign(eigenvalue) * np.sqrt(abs(eigenvalue)) * factor)[:, None], 3, axis=1
    )


def make_record(sign=1, analytical=False):
    atoms = bulk("Pd", "fcc", a=3.89).repeat((5, 5, 5))
    sites, nn = neighbors()
    phi = np.zeros((125, 3, 3))
    phi[nn] = -sign * np.eye(3)
    phi[0] = sign * 12 * np.eye(3)
    displacements = np.vstack(
        [
            np.zeros((1, 3)),
            0.01 * np.eye(3),
            -0.01 * np.eye(3),
            0.005 * np.eye(3),
            -0.005 * np.eye(3),
        ]
    )
    positions = np.repeat(atoms.positions[None], 13, axis=0)
    positions[:, 0] += displacements
    record = {
        "positions_A": positions,
        "cell_A": atoms.cell.array,
        "numbers": atoms.numbers,
        "pbc": atoms.pbc,
        "masses_amu": atoms.get_masses(),
        "displacements_A": displacements,
        "forces_eV_A": -np.einsum("rba,fa->frb", phi, displacements),
        "energies_eV": -125
        + 0.5 * np.einsum("fa,ab,fb->f", displacements, phi[0], displacements),
        "force_constants_eV_A2": phi,
        "metadata": {
            "checkpoint_sha256": PINNED_TEACHER_SHA256,
            "default_dtype": "float64",
            "method": "analytical" if analytical else "central_difference",
            "dispersion": False,
        },
    }
    if analytical:
        lookup = {tuple(site): i for i, site in enumerate(sites)}
        hessian = np.empty((125, 3, 125, 3))
        for i in range(125):
            for j in range(125):
                hessian[i, :, j, :] = phi[
                    lookup[tuple((sites[i] - sites[j] + 2) % 5 - 2)]
                ]
        record["hessian_eV_A2"] = hessian.reshape(375, 375)
    return record


def save_record(path, record):
    np.savez_compressed(path, **{**record, "metadata": json.dumps(record["metadata"])})


def _submission(tmp_path, sign=1, analytical=False, mesh=20):
    save_record(tmp_path / "phonons.npz", make_record(sign, analytical))
    q = (np.indices((mesh,) * 3).reshape(3, -1).T + 0.5) / mesh
    samples = independent_energies(q, sign)
    band_q = np.concatenate(
        [np.linspace(a, b, 11)[:-1] for a, b in itertools.pairwise(NODES)]
        + [NODES[-1:]]
    )
    results = {
        "bands": {
            "qpoints": band_q.tolist(),
            "node_indices": [0, 10, 20, 30, 40, 50],
            "energies_eV": independent_energies(band_q, sign).tolist(),
        },
        "dos": {
            "representation": "sticks",
            "mesh": [mesh] * 3,
            "shift": [0.5] * 3,
            "energies_eV": samples.ravel().tolist(),
            "weights": [1 / len(q)] * samples.size,
        },
        "zpe_eV_per_atom": float(samples.sum(axis=1).mean() / 2) if sign > 0 else None,
        "imaginary_weight": 0 if sign > 0 else 3,
        "resolved_imaginary_weight": 0 if sign > 0 else 3,
    }
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"artifacts": {"force_log": "phonons.npz"}, "results": results})
    )
    return path


@pytest.fixture
def submission(tmp_path):
    return _submission(tmp_path)


def response(plan, fingerprint):
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "jobs": [
            {
                "id": job["id"],
                "status": "complete",
                "value": copy.deepcopy(plan.expected[job["id"]]["values"]),
            }
            for job in plan.jobs
        ],
    }


def score(path, level=2):
    return WorkflowScorer(
        4, level=level, verifier=ModalVerifier(transport=response)
    ).evaluate(path)


def change_answer(path, update):
    data = json.loads(path.read_text())
    update(data["results"])
    path.write_text(json.dumps(data))


def test_stable_spectrum_and_level1_scope(submission):
    report = score(submission)
    assert report["score"] == 1, report
    assert report["calculated_results"]["mesh"][0] >= 20
    assert report["calculated_results"]["zpe_sampling_error_eV"] < 1e-5
    change_answer(submission, lambda results: results.clear())
    assert score(submission, level=1)["score"] == 1
    assert score(submission)["score"] == 0


def test_only_zpe_is_required_and_spectra_are_derived(submission):
    data = json.loads(submission.read_text())
    data["results"] = {"zpe_eV_per_atom": data["results"]["zpe_eV_per_atom"]}
    submission.write_text(json.dumps(data))
    report = score(submission)
    assert report["score"] == 1, report
    assert report["calculated_results"]["resolved_imaginary_weight"] == 0
    assert len(report["calculated_results"]["bands"]["energies_eV"]) == 251
    assert "dos_summary" in report["calculated_results"]
    data["results"].pop("zpe_eV_per_atom")
    submission.write_text(json.dumps(data))
    assert score(submission)["score"] == 0


def test_force_constants_and_bands_against_analytic_dispersion(submission):
    from corral_md.scoring.task_4 import energies

    phi, sites, diagnostics = prepare(read_record(Evidence(submission)))
    q = np.random.default_rng(31).random((29, 3))
    assert np.allclose(energies(phi, sites, q), independent_energies(q), atol=1e-12)
    assert diagnostics["imaginary_tolerance_eV"] == 1e-7


def test_signed_instability_has_no_harmonic_zpe(tmp_path):
    path = _submission(tmp_path, sign=-1)
    assert score(path)["score"] == 1
    change_answer(path, lambda r: r.update(zpe_eV_per_atom=0))
    assert score(path)["score"] == 0
    change_answer(path, lambda r: r.update(zpe_eV_per_atom=None))
    change_answer(
        path,
        lambda r: r["bands"].update(
            energies_eV=abs(np.array(r["bands"]["energies_eV"])).tolist()
        ),
    )
    assert score(path)["score"] == 0


def test_analytical_derivative_uses_same_answer_and_independent_check(tmp_path):
    path = _submission(tmp_path, analytical=True)
    plan = build_plan(Evidence(path), "challenge")
    assert [job["id"] for job in plan.jobs] == ["palladium_forces", "palladium_hessian"]
    assert len(plan.jobs[0]["frames"]) == 4
    assert "hessian" not in plan.jobs[1]["frames"][0]
    assert score(path)["score"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "cell",
        "mass",
        "species",
        "mapping",
        "displacement",
        "forces",
        "constants",
        "nan",
        "truncated",
        "model",
    ],
)
def test_bad_records_do_not_pass(submission, defect):
    record = read_record(Evidence(submission))
    if defect == "cell":
        record["cell_A"] *= 1.1
    elif defect == "mass":
        record["masses_amu"] *= 2
    elif defect == "species":
        record["numbers"][1] = 47
    elif defect == "mapping":
        record["positions_A"][:, 2] = record["positions_A"][:, 3]
    elif defect == "displacement":
        record["positions_A"][2, 1, 0] += 0.01
    elif defect == "forces":
        record["forces_eV_A"][7, 0, 0] += 1
    elif defect == "constants":
        record["force_constants_eV_A2"] *= 2
    elif defect == "nan":
        record["forces_eV_A"][1, 0, 0] = np.nan
    elif defect == "truncated":
        record["positions_A"] = record["positions_A"][:-1]
    else:
        record["metadata"]["checkpoint_sha256"] = "0" * 64
    save_record(submission.parent / "phonons.npz", record)
    assert score(submission)["score"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "zpe",
        "imaginary",
        "resolved",
        "band_units",
        "band_path",
        "dos_weight",
        "dos_shape",
    ],
)
def test_altered_answers_fail(submission, defect):
    def update(r):
        if defect == "zpe":
            r["zpe_eV_per_atom"] *= 2
        elif defect in {"imaginary", "resolved"}:
            r[
                {
                    "imaginary": "imaginary_weight",
                    "resolved": "resolved_imaginary_weight",
                }[defect]
            ] = 1
        elif defect == "band_units":
            r["bands"]["energies_eV"] = (
                1000 * np.array(r["bands"]["energies_eV"])
            ).tolist()
        elif defect == "band_path":
            r["bands"]["qpoints"][10] = [0, 0, 0]
        elif defect == "dos_weight":
            r["dos"]["weights"] = (np.array(r["dos"]["weights"]) / 3).tolist()
        else:
            r["dos"]["energies_eV"] = [0] * len(r["dos"]["energies_eV"])

    change_answer(submission, update)
    assert score(submission)["score"] == 0


def test_coarse_sampling_does_not_hide_behind_matching_answers(tmp_path):
    assert score(_submission(tmp_path, mesh=2))["score"] == 0


def test_displacement_sensitivity_requires_resolution(submission):
    record = read_record(Evidence(submission))
    record["forces_eV_A"][1:7] *= 1.3
    save_record(submission.parent / "phonons.npz", record)
    report = score(submission)
    assert report["score"] is None
    assert "accuracy" in report["checks"][0]["detail"]


def test_offline_cannot_establish_model_identity(submission):
    assert WorkflowScorer(4).evaluate(submission)["score"] is None


def test_histogram_and_file_arrays(submission):
    data = json.loads(submission.read_text())
    samples = np.asarray(data["results"]["dos"]["energies_eV"])
    edges = np.linspace(samples.min() - 1e-8, samples.max() + 1e-8, 31)
    density = np.histogram(samples, edges)[0] * 3 / samples.size / np.diff(edges)
    data["results"]["dos"] = {
        "representation": "histogram",
        "mesh": [20] * 3,
        "shift": [0.5] * 3,
        "bin_edges_eV": edges.tolist(),
        "density_per_eV": "density",
    }
    np.save(submission.parent / "density.npy", density)
    data["artifacts"]["density"] = "density.npy"
    (submission.parent / "bands.json").write_text(
        json.dumps(data["results"].pop("bands"))
    )
    data["artifacts"]["bands"] = "bands.json"
    submission.write_text(json.dumps(data))
    assert score(submission)["score"] == 1


def test_tiny_imaginary_modes_remain_signed_but_are_numerical_residuals(tmp_path):
    path = _submission(tmp_path, sign=-1e-12)
    change_answer(
        path, lambda r: r.update(zpe_eV_per_atom=0, resolved_imaginary_weight=0)
    )
    report = score(path)
    assert report["score"] == 1, report
    assert report["calculated_results"]["spectrum_status"] == "numerical_residuals"
    assert report["calculated_results"]["imaginary_weight"] == 3


def test_rotated_cell_and_permuted_atoms_keep_the_same_spectrum(submission):
    record = read_record(Evidence(submission))
    permutation = np.r_[0, np.random.default_rng(9).permutation(np.arange(1, 125))]
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    record["cell_A"] = record["cell_A"] @ rotation
    record["positions_A"] = record["positions_A"][:, permutation] @ rotation
    # Displace Cartesian axes in the rotated laboratory frame.
    record["positions_A"][:] = record["positions_A"][0]
    record["positions_A"][:, 0] += record["displacements_A"]
    record["force_constants_eV_A2"] = (
        rotation.T @ record["force_constants_eV_A2"][permutation] @ rotation
    )
    record["forces_eV_A"] = -np.einsum(
        "rba,fa->frb", record["force_constants_eV_A2"], record["displacements_A"]
    )
    save_record(submission.parent / "phonons.npz", record)
    assert score(submission)["score"] == 1


def test_pickle_arrays_are_rejected_without_loading_objects(submission):
    record = read_record(Evidence(submission))
    record["forces_eV_A"] = np.full((13, 125, 3), object(), dtype=object)
    save_record(submission.parent / "phonons.npz", record)
    assert score(submission)["score"] == 0


def test_gaussian_dos_matches_signed_mode_sum(submission):
    data = json.loads(submission.read_text())
    values = np.asarray(data["results"]["dos"]["energies_eV"])
    sigma = 0.002
    axis = np.linspace(values.min() - 5 * sigma, values.max() + 5 * sigma, 180)
    density = np.mean(np.exp(-0.5 * ((axis[:, None] - values) / sigma) ** 2), axis=1)
    density *= 3 / (sigma * np.sqrt(2 * np.pi))
    data["results"]["dos"] = {
        "representation": "gaussian",
        "mesh": [20] * 3,
        "shift": [0.5] * 3,
        "energy_grid_eV": axis.tolist(),
        "sigma_eV": sigma,
        "density_per_eV": density.tolist(),
    }
    submission.write_text(json.dumps(data))
    assert score(submission)["score"] == 1
    data["results"]["dos"]["density_per_eV"] = (density / 3).tolist()
    submission.write_text(json.dumps(data))
    assert score(submission)["score"] == 0


def test_level1_factory_uses_the_new_calculation_plan(submission):
    from corral_md.scoring import check_level1_workflow

    change_answer(submission, lambda results: results.clear())
    scorer = check_level1_workflow(4)
    scorer.verifier.transport = response
    assert scorer.evaluate(submission)["score"] == 1
