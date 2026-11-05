"""Accepted phonon layouts retain their data and still face independent checks."""

import json

import numpy as np
import pytest
from corral_md.scoring.common import EvidenceError, UnsupportedEvidence
from corral_md.scoring.formats import normalize_force_record, phonon_log
from corral_md.scoring.task_4 import prepare
from corral_md.scoring.task_5 import mode_energies

from .test_score_task_5 import make_submission, scorer, write
from .test_workflow_task_4 import make_record


def recorded_modes(root, level):
    """Mirror the observed results/data layouts, using independent harmonic data."""
    path = make_submission(root, level)
    original = json.loads((root / "calculations.json").read_text())
    rows = []
    for record in original["calculations"]:
        frame = record["structure"]
        modes = mode_energies(np.asarray(record["hessian_eV_A2"]), frame["masses"])
        row = {
            "strain_fraction" if level == 1 else "isotropic_strain": record[
                "isotropic_strain"
            ],
            "cell": frame["cell"],
            "gamma_frequencies_eV" if level == 1 else "frequencies_eV": modes.tolist(),
        }
        if level == 1:
            row.update(
                scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                masses_amu=frame["masses"],
            )
        else:
            row.update(
                uniaxial_strain=record["uniaxial_strain"],
                positions=frame["positions"],
            )
        rows.append(row)
    log = {"results" if level == 1 else "data": rows}
    if level == 2:
        log["mass_amu"] = 28.0855
    write(root / "calculations.json", log)
    return path, log, rows


@pytest.mark.parametrize("level", [1, 2])
def test_saved_mode_layouts_score_and_still_reject_inaccurate_modes(tmp_path, level):
    path, log, rows = recorded_modes(tmp_path, level)
    report = scorer(level).evaluate(path)
    assert report["score"] == 1, report
    key = "gamma_frequencies_eV" if level == 1 else "frequencies_eV"
    # Level 1 checks the maximum only; Level 2 checks the complete spectrum.
    rows[0][key][-1 if level == 1 else 3] *= 1.5
    write(tmp_path / "calculations.json", log)
    report = scorer(level).evaluate(path)
    assert report["score"] == 0
    assert report["checks"][-1]["name"] == (
        "final_quantities" if level == 1 else "independent_hessian_accuracy"
    )


@pytest.mark.parametrize("level", [1, 2])
def test_recognized_logs_with_missing_calculations_score_zero(tmp_path, level):
    path, log, rows = recorded_modes(tmp_path, level)
    rows.clear()
    write(tmp_path / "calculations.json", log)
    assert scorer(level).evaluate(path)["score"] == 0


def test_row_container_and_mode_aliases_cannot_hide_conflicts():
    row = {"frequencies_eV": [0, 0, 0, 0.1, 0.2, 0.3]}
    with pytest.raises(EvidenceError, match="Conflicting evidence fields"):
        phonon_log({"data": [row], "calculations": []}, 2)
    row["gamma_mode_energies_eV"] = [0, 0, 0, 0.1, 0.2, 0.9]
    with pytest.raises(EvidenceError, match="Conflicting evidence fields"):
        phonon_log({"data": [row]}, 2)


def test_summary_results_are_not_mistaken_for_phonon_rows(tmp_path):
    _, log, _ = recorded_modes(tmp_path, 2)
    log["results"] = {"fit": {"intercept_eV": 0.06}}
    assert len(phonon_log(log, 2)["calculations"]) == 25


def compact_forces(*, lattice=False):
    original = make_record()
    displacements = original["displacements_A"]
    record = {
        "supercell_positions": original["positions_A"][0],
        "supercell_cell": original["cell_A"],
        "supercell_numbers": original["numbers"],
        "supercell_masses": original["masses_amu"],
        "forces": original["forces_eV_A"],
        "force_constants_raw_row0": original["force_constants_eV_A2"],
    }
    if lattice:
        record.update(
            displaced_atom=0,
            displacement_axes=np.argmax(abs(displacements), axis=1),
            displacement_values=displacements.sum(axis=1),
            force_constants_raw=record.pop("force_constants_raw_row0").reshape(
                5, 5, 5, 3, 3
            ),
            lattice_index=np.indices((5, 5, 5)).reshape(3, -1).T,
        )
        record.pop("supercell_masses")
        record["mass"] = 106.42
    else:
        expanded = np.zeros_like(original["positions_A"])
        expanded[:, 0] = displacements
        record["displacements"] = expanded
    return original, record


@pytest.mark.parametrize("lattice", [False, True])
def test_npz_supercell_species_and_compact_displacements_are_lossless(lattice):
    original, retained = compact_forces(lattice=lattice)
    converted = normalize_force_record(retained)
    for key in (
        "positions_A",
        "numbers",
        "masses_amu",
        "displacements_A",
        "forces_eV_A",
        "force_constants_eV_A2",
    ):
        np.testing.assert_allclose(converted[key], original[key])
    reconstructed, _, _ = prepare(converted)
    np.testing.assert_allclose(reconstructed, original["force_constants_eV_A2"])


def test_lattice_force_constants_follow_recorded_atom_order():
    _, record = compact_forces(lattice=True)
    expected = prepare(normalize_force_record(record))[0]
    permutation = np.r_[0, np.random.default_rng(54).permutation(np.arange(1, 125))]
    for key in ("supercell_positions", "supercell_numbers", "lattice_index"):
        record[key] = record[key][permutation]
    record["forces"] = record["forces"][:, permutation]
    actual, _, _ = prepare(normalize_force_record(record))
    np.testing.assert_allclose(actual, expected[permutation])


@pytest.mark.parametrize(
    "defect", ["axis", "missing_axis", "displacement", "indices", "species"]
)
def test_compact_force_data_still_rejects_inconsistent_evidence(defect):
    _, record = compact_forces(lattice=True)
    if defect == "axis":
        record["displacement_axes"][1] = 3
    elif defect == "missing_axis":
        del record["displacement_axes"]
    elif defect == "displacement":
        record["displacements"] = np.zeros((13, 3))
    elif defect == "indices":
        record["lattice_index"][[1, 2]] = record["lattice_index"][[2, 1]]
    else:
        record["numbers"] = np.full(125, 47)
    with pytest.raises(EvidenceError):
        normalize_force_record(record)


def indexed_forces(layout):
    """Use the two saved-run encodings with an independently known harmonic model."""
    original, record = compact_forces()
    vectors = original["displacements_A"]
    atoms = np.zeros(len(vectors))
    if layout == "atom_xyz":
        record["displacements"] = np.column_stack((atoms, vectors))
    else:
        record["displacements"] = np.column_stack(
            (atoms, np.argmax(abs(vectors), axis=1), vectors.sum(axis=1))
        )
    record.pop("supercell_numbers")
    record.pop("supercell_masses")
    record["symbols"] = np.full(125, "Pd")
    record["mass_amu"] = 106.42
    return original, record


@pytest.mark.parametrize("layout", ["atom_xyz", "atom_axis_delta"])
def test_indexed_npz_forces_reconstruct_known_constants_and_spectrum(tmp_path, layout):
    from corral_md.scoring.common import Evidence
    from corral_md.scoring.task_4 import energies, read_record

    from .test_workflow_task_4 import independent_energies

    original, record = indexed_forces(layout)
    # Archive round-trip also exercises NumPy string arrays for chemical symbols.
    np.savez(tmp_path / "forces.npz", **record)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"force_log": "forces.npz"}}))
    converted = read_record(Evidence(manifest))
    for key in (
        "positions_A",
        "displacements_A",
        "forces_eV_A",
        "numbers",
        "masses_amu",
    ):
        np.testing.assert_allclose(converted[key], original[key])
    phi, sites, _ = prepare(converted)
    np.testing.assert_allclose(phi, original["force_constants_eV_A2"])
    q = np.random.default_rng(27).random((19, 3))
    np.testing.assert_allclose(energies(phi, sites, q), independent_energies(q))
    # Each force row must retain its association with the displaced geometry.
    order = np.random.default_rng(33).permutation(len(record["forces"]))
    for key in ("displacements", "forces"):
        record[key] = record[key][order]
    np.testing.assert_allclose(prepare(normalize_force_record(record))[0], phi)


@pytest.mark.parametrize("layout", ["atom_xyz", "atom_axis_delta"])
def test_indexed_displacements_agree_with_explicit_cartesian_vectors(layout):
    original, record = indexed_forces(layout)
    record["displacements_A"] = original["displacements_A"].copy()
    normalize_force_record(record)
    record["displacements_A"][1, 0] += 0.001
    with pytest.raises(EvidenceError, match="Conflicting displacement records"):
        normalize_force_record(record)


@pytest.mark.parametrize("layout", ["atom_xyz", "atom_axis_delta"])
def test_indexed_displacements_do_not_accept_wrong_forces_or_missing_pairs(layout):
    _, record = indexed_forces(layout)
    record["forces"] = record["forces"].copy()
    record["forces"][[1, 7], 0, 0] += 1
    with pytest.raises(EvidenceError, match="force constants disagree"):
        prepare(normalize_force_record(record))
    _, record = indexed_forces(layout)
    for key in ("displacements", "forces"):
        record[key] = np.delete(record[key], 1, axis=0)
    with pytest.raises(EvidenceError, match="both central force differences"):
        prepare(normalize_force_record(record))


@pytest.mark.parametrize("layout", ["atom_xyz", "atom_axis_delta"])
def test_indexed_displacements_do_not_silently_move_another_atom(layout):
    _, record = indexed_forces(layout)
    record["displacements"][:, 0] = 1
    with pytest.raises(UnsupportedEvidence, match="atom-order adapter"):
        normalize_force_record(record)
    record["displaced_atom"] = 0
    with pytest.raises(EvidenceError, match="Conflicting displaced atom indices"):
        normalize_force_record(record)


@pytest.mark.parametrize("atom", [-1, 0.5, 125])
def test_atom_xyz_displacements_reject_invalid_indices(atom):
    _, record = indexed_forces("atom_xyz")
    record["displacements"][0, 0] = atom
    with pytest.raises(EvidenceError, match="Invalid displaced atom indices"):
        normalize_force_record(record)


@pytest.mark.parametrize("defect", ["symbol", "shape", "numbers"])
def test_force_symbols_reject_invalid_or_conflicting_species(defect):
    _, record = indexed_forces("atom_xyz")
    if defect == "symbol":
        record["symbols"][0] = "XX"
    elif defect == "shape":
        record["symbols"] = record["symbols"][:, None]
    else:
        record["numbers"] = np.full(125, 47)
    with pytest.raises(EvidenceError):
        normalize_force_record(record)


def test_explicit_cartesian_displacements_are_never_interpreted_as_indices():
    _, record = indexed_forces("atom_axis_delta")
    record["displacements_A"] = record.pop("displacements")
    converted = normalize_force_record(record)
    np.testing.assert_array_equal(
        converted["displacements_A"], record["displacements_A"]
    )
    with pytest.raises(EvidenceError, match="both central force differences"):
        prepare(converted)
