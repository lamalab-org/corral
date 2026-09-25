"""Focused checks for the prompt requirements added to level-1 LAMMPS scoring."""

import json
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.io import read, write
from corral_md.workflow_scoring.common import Evidence, EvidenceError, UnsupportedEvidence
from corral_md.workflow_scoring.lammps_checks import (
    cooled_endpoint_temperature,
    mp149_conventional_cell,
    no_silicate_state_resets,
    prepared_before_heating,
    readiness_assessment,
    silicon_input_matches_reference,
    supplied_silicate_initial_state,
    supplied_silicon_potential,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _silicon_evidence(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    write(
        tmp_path / "input" / "mp-149.data",
        atoms,
        format="lammps-data",
        atom_style="full",
        masses=True,
    )
    write(tmp_path / "input" / "mp-149.cif", atoms)
    (tmp_path / "output" / "prepared.data").write_text("saved initial state")
    _save(
        tmp_path / "manifest.json",
        {
            "artifacts": {
                "reference_cell": "input/mp-149.cif",
                "prepared_state": "output/prepared.data",
                "lammps_input": "input/heat.in",
            },
            "results": {
                "readiness_assessment": "The final temperature and density support a liquid starting state, pending equilibration."
            },
        },
    )
    return tmp_path / "input" / "heat.in", Evidence(tmp_path / "manifest.json")


def test_silicon_preparation_must_precede_heating_and_use_supplied_sw(tmp_path):
    input_path, evidence = _silicon_evidence(tmp_path)
    prelude = (
        "units metal\natom_style full\n"
        "read_data /workspace/input/mp-149.data\n"
        "replicate 3 3 3\n"
        "pair_style sw\n"
        "pair_coeff * * /workspace/potentials/SW/Si.sw Si\n"
    )
    input_path.write_text(
        prelude + "write_data /workspace/output/prepared.data\nrun 500000\n"
    )
    assert readiness_assessment(evidence)
    assert prepared_before_heating(evidence)
    assert supplied_silicon_potential(evidence)
    assert silicon_input_matches_reference(evidence)

    input_path.write_text(
        prelude + "run 500000\nwrite_data /workspace/output/prepared.data\n"
    )
    assert not prepared_before_heating(evidence)
    input_path.write_text(
        prelude.replace("/workspace/potentials/SW/Si.sw", "other.sw")
        + "write_data /workspace/output/prepared.data\nrun 500000\n"
    )
    assert not supplied_silicon_potential(evidence)


def test_silicon_reference_must_match_loaded_input(tmp_path):
    input_path, evidence = _silicon_evidence(tmp_path)
    input_path.write_text(
        "read_data /workspace/input/mp-149.data\nreplicate 3 3 3\nrun 500000\n"
    )
    assert silicon_input_matches_reference(evidence)
    shifted = bulk("Si", "diamond", a=5.7, cubic=True)
    write(tmp_path / "input" / "mp-149.cif", shifted)
    assert not silicon_input_matches_reference(Evidence(tmp_path / "manifest.json"))


def test_silicon_prebuilt_repeat_can_be_loaded_directly(tmp_path):
    input_path, _ = _silicon_evidence(tmp_path)
    prepared = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    write(
        tmp_path / "input" / "mp-149.data",
        prepared,
        format="lammps-data",
        atom_style="full",
        masses=True,
    )
    write(tmp_path / "output" / "prepared.extxyz", prepared)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["prepared_state"] = "output/prepared.extxyz"
    _save(manifest_path, manifest)
    input_path.write_text("read_data /workspace/input/mp-149.data\nrun 500000\n")
    evidence = Evidence(manifest_path)
    assert silicon_input_matches_reference(evidence)
    assert prepared_before_heating(evidence)


def test_silicon_prebuilt_repeat_must_derive_from_reference(tmp_path):
    input_path, _ = _silicon_evidence(tmp_path)
    unrelated = bulk("Si", "diamond", a=5.7, cubic=True).repeat((3, 3, 3))
    write(
        tmp_path / "input" / "mp-149.data",
        unrelated,
        format="lammps-data",
        atom_style="full",
        masses=True,
    )
    write(tmp_path / "output" / "prepared.extxyz", unrelated)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["prepared_state"] = "output/prepared.extxyz"
    _save(manifest_path, manifest)
    input_path.write_text("read_data /workspace/input/mp-149.data\nrun 500000\n")
    assert not silicon_input_matches_reference(Evidence(manifest_path))


def test_absolute_input_path_must_resolve_exactly(tmp_path):
    input_path, evidence = _silicon_evidence(tmp_path)
    input_path.write_text(
        "read_data /workspace/other/mp-149.data\nreplicate 3 3 3\nrun 500000\n"
    )
    with pytest.raises(EvidenceError, match="Missing artifact"):
        silicon_input_matches_reference(evidence)


def test_silicon_reference_matches_current_mp_cell_and_historical_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "corral_md.workflow_scoring.lammps_checks._mp149_live_reference",
        lambda: bulk("Si", "diamond", a=5.468728, cubic=True),
    )
    _, evidence = _silicon_evidence(tmp_path)
    reference_path = tmp_path / "input" / "mp-149.cif"
    write(reference_path, bulk("Si", "diamond", a=5.468728, cubic=True))
    assert mp149_conventional_cell(Evidence(tmp_path / "manifest.json"))
    write(reference_path, bulk("Si", "diamond", a=5.443702372939453, cubic=True))
    assert mp149_conventional_cell(Evidence(tmp_path / "manifest.json"))
    write(reference_path, bulk("Si", "diamond", a=5.6, cubic=True))
    assert not mp149_conventional_cell(Evidence(tmp_path / "manifest.json"))


def test_mp_reference_needs_api_access_for_automatic_credit(monkeypatch):
    from corral_md.workflow_scoring.lammps_checks import _mp149_live_reference

    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.find_dotenv", lambda **_: "")
    _mp149_live_reference.cache_clear()
    with pytest.raises(UnsupportedEvidence, match="MP_API_KEY"):
        _mp149_live_reference()


def test_silicon_assessment_must_have_content(tmp_path):
    _, evidence = _silicon_evidence(tmp_path)
    assert readiness_assessment(evidence)
    evidence.results["readiness_assessment"] = "yes"
    assert not readiness_assessment(evidence)


def _silicate_evidence(tmp_path):
    asset = Path(__file__).resolve().parents[1] / "structures.zip"
    with ZipFile(asset) as archive:
        data = archive.read("structures/melt/liq1300.dat")
    source = tmp_path / "input" / "liq1300.dat"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    atoms = read(source, format="lammps-data", atom_style="charge", units="real", sort_by_id=True)
    state = {
        "time_ps": 0,
        "ids": atoms.arrays["id"].tolist(),
        "cell": atoms.cell.array.tolist(),
        "positions": atoms.get_positions(wrap=True).tolist(),
        "velocities": (atoms.get_velocities() * units.fs).tolist(),
        "charges": atoms.get_initial_charges().tolist(),
        "species": atoms.get_chemical_symbols(),
    }
    final = json.loads(json.dumps(state))
    final["time_ps"] = 20
    velocities = np.asarray(final["velocities"], dtype=float)
    masses = atoms.get_masses()
    kinetic_temperature = np.sum(masses[:, None] * (velocities / units.fs) ** 2) / (
        (3 * len(atoms) - 3) * units.kB
    )
    final["velocities"] = (velocities * np.sqrt(310.89 / kinetic_temperature)).tolist()
    _save(tmp_path / "boundaries.json", {"initial": state, "cooling_end": final})
    _save(
        tmp_path / "trace.json",
        [
            {"temperature_K": 1300.0},
            {"temperature_K": 310.89},
        ],
    )
    input_path = tmp_path / "cool.in"
    input_path.write_text(
        "read_data /workspace/input/liq1300.dat\n"
        "fix cooling all npt temp 1300 300 100 iso 0 0 1000\n"
        "run 20000\n"
    )
    _save(
        tmp_path / "manifest.json",
        {
            "artifacts": {
                "lammps_inputs": "cool.in",
                "boundary_states": "boundaries.json",
                "thermal_trace": "trace.json",
            }
        },
    )
    return input_path, source, Evidence(tmp_path / "manifest.json")


def test_silicate_source_and_initial_boundary_are_bound_to_supplied_file(tmp_path):
    _, source, evidence = _silicate_evidence(tmp_path)
    assert supplied_silicate_initial_state(evidence)
    assert no_silicate_state_resets(evidence)
    assert cooled_endpoint_temperature(evidence)

    boundary = json.loads((tmp_path / "boundaries.json").read_text())
    original_boundary = json.loads((tmp_path / "boundaries.json").read_text())
    boundary["cooling_end"]["velocities"][0][0] += 1
    _save(tmp_path / "boundaries.json", boundary)
    assert not cooled_endpoint_temperature(Evidence(tmp_path / "manifest.json"))
    boundary = json.loads(json.dumps(original_boundary))
    boundary["initial"]["velocities"][0][0] += 0.1
    _save(tmp_path / "boundaries.json", boundary)
    assert not supplied_silicate_initial_state(Evidence(tmp_path / "manifest.json"))
    _save(tmp_path / "boundaries.json", original_boundary)
    atoms = read(source, format="lammps-data", atom_style="charge", units="real")
    write(
        source,
        atoms,
        format="lammps-data",
        atom_style="charge",
        units="real",
        masses=True,
        velocities=True,
    )
    # Serialization changes are fine when the physical source state is unchanged.
    assert supplied_silicate_initial_state(Evidence(tmp_path / "manifest.json"))
    atoms.positions[0, 0] += 0.2
    write(
        source,
        atoms,
        format="lammps-data",
        atom_style="charge",
        units="real",
        masses=True,
        velocities=True,
    )
    assert not supplied_silicate_initial_state(Evidence(tmp_path / "manifest.json"))


def test_silicate_can_read_the_mounted_asset_without_a_local_copy(tmp_path):
    input_path, source, evidence = _silicate_evidence(tmp_path)
    input_path.write_text(
        input_path.read_text().replace(
            "/workspace/input/liq1300.dat",
            "/workspace/structures/melt/liq1300.dat",
        )
    )
    source.unlink()
    assert supplied_silicate_initial_state(evidence)

    input_path.write_text(
        input_path.read_text().replace(
            "/workspace/structures/melt/liq1300.dat",
            "/workspace/other/liq1300.dat",
        )
    )
    with pytest.raises(EvidenceError, match="Missing artifact"):
        supplied_silicate_initial_state(evidence)


def test_evaluator_uses_the_bundled_silicate_input():
    archive = Path(__file__).resolve().parents[1] / "structures.zip"
    with ZipFile(archive) as zipped:
        original = zipped.read("structures/melt/liq1300.dat")
    assert hashlib.sha256(original).hexdigest() == (
        "6bd3b36b6d6b3e47c5c5a339460ad95ac88438e2ee6adafb91b96ace0ef9ecf3"
    )


def test_silicate_velocity_resets_and_hot_endpoint_are_rejected(tmp_path):
    input_path, _, evidence = _silicate_evidence(tmp_path)
    input_path.write_text(
        input_path.read_text().replace(
            "fix cooling", "velocity all scale 1300\nfix cooling"
        )
    )
    assert not no_silicate_state_resets(evidence)
    _save(
        tmp_path / "trace.json",
        [{"temperature_K": 1300}, {"temperature_K": 600}],
    )
    assert not cooled_endpoint_temperature(Evidence(tmp_path / "manifest.json"))
