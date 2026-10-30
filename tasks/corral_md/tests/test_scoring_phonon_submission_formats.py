"""Accepted phonon layouts retain their data and still face independent checks."""

import json

import numpy as np
import pytest
from corral_md.scoring.common import EvidenceError
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
    # Change a nonmaximum optical mode, leaving the answer/fit unchanged: the
    # independent full-spectrum check must still reject the calculation.
    rows[0][key][3] *= 1.5
    write(tmp_path / "calculations.json", log)
    report = scorer(level).evaluate(path)
    assert report["score"] == 0
    assert report["checks"][-1]["name"] == "independent_hessian_accuracy"


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
