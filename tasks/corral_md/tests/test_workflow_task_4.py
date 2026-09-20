"""Saved-force reconstruction, independent phonon sums, and evidence defects."""

import json
import pickle

import numpy as np
import pytest
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.task_4 import evaluate


def _write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def _energy(q, sign=1):
    # Independent analytic dispersion for three identical nearest-cell springs.
    eigenvalue = (
        sign * 2 * np.sum(1 - np.cos(2 * np.pi * np.asarray(q)), axis=1) / 106.42
    )
    factor = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    return np.repeat(
        (np.sign(eigenvalue) * np.sqrt(abs(eigenvalue)) * factor)[:, None], 3, axis=1
    )


def _samples(mesh, shift=0, sign=1):
    q = (np.indices((mesh,) * 3).reshape(3, -1).T + shift) / mesh
    energy = _energy(q, sign)
    return {
        "qpoints": q.tolist(),
        "weights": [1 / len(q)] * len(q),
        "energies_eV": energy.tolist(),
        "integration": {
            "kind": "uniform_grid",
            "mesh": [mesh] * 3,
            "shift": [shift] * 3,
        },
    }


def _summary(samples):
    values = np.array(samples["energies_eV"])
    weights = np.array(samples["weights"])[:, None]
    return float(np.sum(weights * np.maximum(values, 0)) / 2), float(
        np.sum(weights * (values < 0))
    )


def _submission(tmp_path, sign=1):
    primitive = 3.89 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cells = np.indices((5,) * 3).reshape(3, -1).T - 2
    indices = np.arange(125)[::-1]
    origin_row = np.where(np.all(cells == 0, axis=1))[0][0]
    source = int(indices[origin_row])
    positions = np.empty((125, 3))
    positions[indices] = (cells % 5) @ primitive
    phi = np.zeros((125, 3, 3))
    phi[origin_row] = sign * 6 * np.eye(3)
    phi[np.sum(np.abs(cells), axis=1) == 1] = -sign * np.eye(3)
    displacements = np.vstack((np.zeros((1, 3)), np.eye(3) * 0.01, -np.eye(3) * 0.01))
    weights = np.zeros((3, 7))
    weights[:, 1:4] = np.eye(3) * 50
    weights[:, 4:7] = -np.eye(3) * 50
    frames = []
    for u in displacements:
        displaced = positions.copy()
        displaced[source] += u
        force = np.zeros((125, 3))
        force[indices] = -np.einsum("rba,a->rb", phi, u)
        frames.append(
            {
                "symbols": ["Pd"] * 125,
                "positions": displaced.tolist(),
                "pbc": True,
                "cell": (5 * primitive).tolist(),
                "masses": [106.42] * 125,
                "energy": float(-125 + 0.5 * u @ phi[origin_row] @ u),
                "forces": force.tolist(),
            }
        )
    fc = {
        "primitive_cell_A": primitive.tolist(),
        "cell_translations": cells.tolist(),
        "atom_indices": indices.tolist(),
        "source_atom_index": source,
        "reference_frame": 0,
        "mass_amu": 106.42,
        "layout": "response_source",
        "values_eV_A2": phi.tolist(),
        "derivative": {
            "method": "finite_difference",
            "frame_indices": list(range(7)),
            "displacements_A": displacements.tolist(),
            "derivative_weights_A_inv": weights.tolist(),
        },
        "corrections": ["pair_symmetry", "acoustic_sum_rule"],
    }
    nodes = np.array(
        [
            [0.0, 0, 0],
            [0.5, 0, 0.5],
            [0.625, 0.25, 0.625],
            [0.5, 0.5, 0.5],
            [0, 0, 0],
            [0.375, 0.375, 0.75],
        ]
    )
    bands = {
        "qpoints": nodes.tolist(),
        "node_indices": list(range(6)),
        "energies_eV": _energy(nodes, sign).tolist(),
    }
    samples = _samples(2, sign=sign)
    alternate = _samples(3, 0.5, sign)
    dos = {
        "representation": "sticks",
        "energies_eV": np.array(samples["energies_eV"]).ravel().tolist(),
        "weights": np.repeat(samples["weights"], 3).tolist(),
    }
    zpe, imaginary = _summary(samples)
    altzpe, altimaginary = _summary(alternate)
    results = {
        "zpe_eV_per_atom": zpe,
        "zpe_kind": "harmonic_zpe" if sign == 1 else "real_modes_only",
        "imaginary_weight": imaginary,
        "resolved_imaginary_weight": imaginary,
        "validation": {
            "pair_symmetry_residual_eV_A2": 0,
            "acoustic_residual_eV_A2": 0,
            "gamma_energies_eV": [0, 0, 0],
            "sensitivity_zpe_change_eV_per_atom": altzpe - zpe,
            "sensitivity_imaginary_weight_change": altimaginary - imaginary,
        },
    }
    artifacts = {
        name: _write(tmp_path / f"{name}.json", value)
        for name, value in [
            ("structures", frames),
            ("force_constants", fc),
            ("bands", bands),
            ("bz_samples", samples),
            ("dos", dos),
            ("sensitivity", {"kind": "sampling", "samples": alternate}),
        ]
    }
    script = tmp_path / "submitted.py"
    script.write_text(
        "raise RuntimeError('Scoring must never execute submitted code')\n"
    )
    settings = {
        "checkpoint": "MACE-MP-0",
        "checkpoint_sha256": "f" * 64,
        "software_versions": {"ase": "saved-version"},
        "force_constant_method": "central displacement",
    }
    manifest = {
        "artifacts": artifacts,
        "results": results,
        "settings": _write(tmp_path / "settings.json", settings),
        "scripts": [str(script)],
        "report": _write(
            tmp_path / "report.json",
            {
                "assessment": "Finite mesh sensitivity and signed unstable modes assessed."
            },
        ),
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    return path


@pytest.fixture
def submission(tmp_path):
    return _submission(tmp_path)


def _score(path):
    rubric = Rubric(4)
    evaluate(Evidence(path), rubric)
    return rubric


def _check(rubric, name):
    return next(item for item in rubric.checks if item["name"] == name)


def _edit(path, function):
    doc = json.loads(path.read_text())
    function(doc)
    _write(path, doc)


def test_task4_complete_synthetic_evidence_and_no_execution(submission, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Executable data or force evaluation attempted")

    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(check["points"] for check in result.checks) == 90
    assert _check(result, "execution_provenance")["status"] == "unverified"


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ("cell", "fcc_supercell_and_mapping"),
        ("force", "force_constant_transformations"),
        ("positions", "derivative_reconstruction"),
        ("weights", "derivative_reconstruction"),
        ("dos", "dos_from_signed_bz_samples"),
        ("path", "band_path_and_signed_energies"),
        ("zpe", "zpe_and_imaginary_mode_accounting"),
        ("sample_energy", "zpe_and_imaginary_mode_accounting"),
        ("missing_force", "raw_energy_force_records"),
        ("diagnostic", "symmetry_and_acoustic_diagnostics"),
        ("grid", "dos_from_signed_bz_samples"),
    ],
)
def test_task4_detects_false_or_incomplete_evidence(submission, change, check):
    directory = submission.parent
    if change in {"cell", "force", "positions", "missing_force"}:

        def defect(frames):
            if change == "cell":
                frames[0]["cell"][0][0] += 0.1
            elif change == "force":
                frames[1]["forces"][0][0] += 1
            elif change == "positions":
                frames[1]["positions"][0][0] += 0.1
            else:
                frames[0].pop("forces")

        _edit(directory / "structures.json", defect)
    elif change == "weights":
        _edit(
            directory / "force_constants.json",
            lambda d: d["derivative"]["derivative_weights_A_inv"][0].__setitem__(
                1, 100
            ),
        )
    elif change == "dos":
        _edit(directory / "dos.json", lambda d: d["weights"].__setitem__(0, 0.3))
    elif change == "path":
        _edit(
            directory / "bands.json",
            lambda d: d["qpoints"].__setitem__(1, [0.25, 0.25, 0.25]),
        )
    elif change == "zpe":
        _edit(submission, lambda d: d["results"].__setitem__("zpe_eV_per_atom", 42))
    elif change == "sample_energy":
        _edit(
            directory / "bz_samples.json",
            lambda d: d["energies_eV"][1].__setitem__(0, 1),
        )
    elif change == "not_sensitivity":
        doc = json.loads((directory / "bz_samples.json").read_text())
        for key in ("qpoints", "weights", "energies_eV"):
            doc[key].reverse()
        _write(directory / "sensitivity.json", {"kind": "sampling", "samples": doc})
    elif change == "diagnostic":
        _edit(
            submission,
            lambda d: d["results"]["validation"].__setitem__(
                "acoustic_residual_eV_A2", 2
            ),
        )
    elif change == "grid":
        _edit(
            directory / "bz_samples.json",
            lambda d: d["qpoints"].__setitem__(0, d["qpoints"][1]),
        )
    result = _score(submission)
    assert _check(result, check)["status"] == "failed", result.checks
    assert result.score < 0.9
    if change in {"force", "positions", "weights", "missing_force"}:
        for dependent in (
            "band_path_and_signed_energies",
            "dos_from_signed_bz_samples",
            "zpe_and_imaginary_mode_accounting",
            "symmetry_and_acoustic_diagnostics",
        ):
            assert _check(result, dependent)["earned"] == 0, result.checks


def test_task4_signed_instability_can_receive_numerical_credit(tmp_path):
    submission = _submission(tmp_path, sign=-1)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    _edit(submission, lambda d: d["results"].__setitem__("zpe_kind", "harmonic_zpe"))
    assert (
        _check(_score(submission), "zpe_and_imaginary_mode_accounting")["status"]
        == "failed"
    )


def test_task4_residual_label_needs_numerical_tolerance_but_not_prose(tmp_path):
    submission = _submission(tmp_path, sign=-1)
    data = json.loads(submission.read_text())
    data.pop("report")
    data["results"].update(
        zpe_kind="residual_corrected_zpe",
        imaginary_tolerance_eV=1.0,
        resolved_imaginary_weight=0,
    )
    _write(submission, data)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    data["results"]["imaginary_tolerance_eV"] = 0
    _write(submission, data)
    assert (
        _check(_score(submission), "zpe_and_imaginary_mode_accounting")["status"]
        == "failed"
    )


def test_task4_cannot_discard_or_absolute_value_imaginary_modes(tmp_path):
    submission = _submission(tmp_path, sign=-1)
    _edit(
        tmp_path / "bands.json",
        lambda d: d.__setitem__("energies_eV", np.abs(d["energies_eV"]).tolist()),
    )
    _edit(
        tmp_path / "dos.json",
        lambda d: d.__setitem__("energies_eV", np.abs(d["energies_eV"]).tolist()),
    )
    result = _score(submission)
    assert _check(result, "band_path_and_signed_energies")["status"] == "failed"
    assert _check(result, "dos_from_signed_bz_samples")["status"] == "failed"


@pytest.mark.parametrize("representation", ["histogram", "gaussian"])
def test_task4_supported_continuous_dos(submission, representation):
    samples = json.loads((submission.parent / "bz_samples.json").read_text())
    energy = np.array(samples["energies_eV"]).ravel()
    weights = np.repeat(samples["weights"], 3)
    grid = np.linspace(-0.01, energy.max() + 0.01, 101)
    if representation == "histogram":
        density = np.histogram(energy, grid, weights=weights)[0] / np.diff(grid)
        dos = {
            "representation": representation,
            "bin_edges_eV": grid.tolist(),
            "density_per_eV": density.tolist(),
        }
    else:
        sigma = 0.003
        density = np.sum(
            weights[:, None]
            * np.exp(-0.5 * ((grid[None, :] - energy[:, None]) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi)),
            axis=0,
        )
        density *= 3 / np.trapezoid(density, grid)
        dos = {
            "representation": representation,
            "energy_grid_eV": grid.tolist(),
            "density_per_eV": density.tolist(),
            "sigma_eV": sigma,
            "normalization": "global",
        }
    _write(submission.parent / "dos.json", dos)
    assert _score(submission).score == pytest.approx(0.9)


def test_task4_ase_array_and_irreducible_quadrature(submission):
    directory = submission.parent
    fc = json.loads((directory / "force_constants.json").read_text())
    values = np.array(fc.pop("values_eV_A2")).swapaxes(1, 2)
    np.save(directory / "ase_c_n.npy", values)
    fc.update(layout="ASE_C_N", values_artifact="ase_c_n")
    _write(directory / "force_constants.json", fc)
    _edit(
        submission,
        lambda d: d["artifacts"].__setitem__("ase_c_n", str(directory / "ase_c_n.npy")),
    )
    samples = json.loads((directory / "bz_samples.json").read_text())
    energy = np.array(samples["energies_eV"])
    _, representative, inverse = np.unique(
        energy, axis=0, return_index=True, return_inverse=True
    )
    reduced = {
        "qpoints": np.array(samples["qpoints"])[representative].tolist(),
        "weights": (np.bincount(inverse) / len(energy)).tolist(),
        "energies_eV": energy[representative].tolist(),
        "full_grid_qpoints": samples["qpoints"],
        "representative_index": inverse.tolist(),
        "integration": dict(samples["integration"], kind="symmetry_reduced"),
    }
    _write(directory / "bz_samples.json", reduced)
    _write(
        directory / "dos.json",
        {
            "representation": "sticks",
            "energies_eV": np.array(reduced["energies_eV"]).ravel().tolist(),
            "weights": np.repeat(reduced["weights"], 3).tolist(),
        },
    )
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_task4_unsupported_methods_are_unverified(submission):
    _edit(
        submission.parent / "force_constants.json",
        lambda d: d["derivative"].__setitem__("method", "external_analytic_format"),
    )
    result = _score(submission)
    assert _check(result, "derivative_reconstruction")["status"] == "unverified"
    assert _check(result, "force_constant_transformations")["status"] == "unverified"
    assert _check(result, "band_path_and_signed_energies")["status"] == "passed"
    assert result.score is None


def test_task4_forward_stencil_is_an_equivalent_workflow(submission):
    path = submission.parent / "force_constants.json"
    data = json.loads(path.read_text())
    derivative = data["derivative"]
    derivative["frame_indices"] = [0, 1, 2, 3]
    derivative["displacements_A"] = derivative["displacements_A"][:4]
    derivative["derivative_weights_A_inv"] = (
        np.column_stack([-np.ones(3), np.eye(3)]) / 0.01
    ).tolist()
    _write(path, data)
    assert _score(submission).score == pytest.approx(0.9)


@pytest.mark.parametrize("norm", ["rms", "frobenius", "unimplemented_norm"])
def test_task4_diagnostic_norm_is_a_choice(submission, norm):
    manifest = json.loads(submission.read_text())
    manifest["results"]["validation"]["residual_norm"] = norm
    del manifest["results"]["validation"]["gamma_energies_eV"]
    _write(submission, manifest)
    result = _score(submission)
    if norm == "unimplemented_norm":
        assert result.score is None
        assert (
            _check(result, "symmetry_and_acoustic_diagnostics")["status"]
            == "unverified"
        )
    else:
        assert result.score == pytest.approx(0.9), result.checks


def test_task4_missing_and_malformed_evidence_do_not_crash():
    for manifest in (
        {},
        {"artifacts": {}, "results": {"zpe_eV_per_atom": "malformed"}},
    ):
        result = _score(manifest)
        assert result.score == 0
        assert sum(check["points"] for check in result.checks) == 90


def test_task4_analytical_hessian_is_read_as_data(submission):
    directory = submission.parent
    fc = json.loads((directory / "force_constants.json").read_text())
    phi = np.array(fc["values_eV_A2"])
    indices = fc["atom_indices"]
    source = fc["source_atom_index"]
    hessian = np.zeros((125, 3, 125, 3))
    hessian[indices, :, source, :] = phi
    hessian[source, :, indices, :] = phi.swapaxes(1, 2)
    np.save(directory / "hessian.npy", hessian.reshape(375, 375))
    fc["derivative"] = {
        "method": "analytical",
        "description": "Saved Cartesian energy Hessian",
    }
    _write(directory / "force_constants.json", fc)
    _edit(
        submission,
        lambda d: d["artifacts"].__setitem__(
            "raw_hessian", str(directory / "hessian.npy")
        ),
    )
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_task4_accepts_equivalent_primitive_basis(submission):
    directory = submission.parent
    fc = json.loads((directory / "force_constants.json").read_text())
    transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    fc["primitive_to_standard"] = transform.tolist()
    fc["primitive_cell_A"] = (transform @ np.array(fc["primitive_cell_A"])).tolist()
    fc["cell_translations"] = (np.array(fc["cell_translations"]) @ transform.T).tolist()
    _write(directory / "force_constants.json", fc)
    _edit(
        directory / "structures.json",
        lambda frames: [
            f.__setitem__("cell", (transform @ np.array(f["cell"])).tolist())
            for f in frames
        ],
    )
    _edit(
        directory / "bands.json",
        lambda d: d.__setitem__(
            "qpoints", (np.array(d["qpoints"]) @ transform.T).tolist()
        ),
    )
    _edit(
        directory / "bz_samples.json",
        lambda d: d.__setitem__(
            "qpoints", (np.array(d["qpoints"]) @ transform.T).tolist()
        ),
    )
    _edit(
        directory / "sensitivity.json",
        lambda d: d["samples"].__setitem__(
            "qpoints", (np.array(d["samples"]["qpoints"]) @ transform.T).tolist()
        ),
    )
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_task4_absolute_values_do_not_hide_force_constant_instability(tmp_path):
    submission = _submission(tmp_path, sign=-1)
    _edit(
        tmp_path / "bz_samples.json",
        lambda d: d.__setitem__("energies_eV", np.abs(d["energies_eV"]).tolist()),
    )
    _edit(
        submission,
        lambda d: d["results"].update(
            imaginary_weight=0, resolved_imaginary_weight=0, zpe_kind="harmonic_zpe"
        ),
    )
    result = _score(submission)
    assert _check(result, "zpe_and_imaginary_mode_accounting")["status"] == "failed"
    assert _check(result, "dos_from_signed_bz_samples")["status"] == "failed"
