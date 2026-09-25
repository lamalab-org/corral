"""Checks that bind the level-1 LAMMPS inputs to their saved states.

These checks inspect data and input files only. They cannot prove that a saved
input was executed; execution evidence is handled by the workflow verifier.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ase import units
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
from scipy.spatial import cKDTree

from .common import EvidenceError, UnsupportedEvidence
from .task_2 import _commands, _state

# SHA-256 of the individual files in the benchmark's versioned asset archives.
_SILICON_SW_SHA256 = "c6d3a7d26db28ee8e3feaf2f8059607de9a4accb45d9719fcabe59488ecf1cad"
_SILICATE_SOURCE_SHA256 = (
    "6bd3b36b6d6b3e47c5c5a339460ad95ac88438e2ee6adafb91b96ace0ef9ecf3"
)
_CU32_SOURCE_SHA256 = "8e33dafe346e7292fe72bbe2626317d616dc8a8034db4a4bd05dc8dbcd697d89"
_SILICON_SW_PATH = "/workspace/potentials/SW/Si.sw"
_SILICATE_SOURCE_PATH = "/workspace/structures/melt/liq1300.dat"
CU32_SOURCE_PATH = "/workspace/structures/cu/cu32.extxyz"


def _file_in_input(e, reference: str, *artifact_names: str) -> Path:
    """Resolve a saved LAMMPS path to one confined evidence file."""
    try:
        return e._path(reference)
    except EvidenceError:
        if Path(reference).is_absolute():
            raise
    candidates = []
    for name in artifact_names:
        if name in e._artifacts:
            candidates.extend(
                path for path in e.artifacts(name) if path.name == Path(reference).name
            )
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise EvidenceError(f"Link the input file {Path(reference).name} unambiguously")
    return unique[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structure_archive() -> Path:
    """Use the packaged archive, or the same archive in an editable checkout."""
    packaged = Path(__file__).with_name("structures.zip")
    if packaged.is_file():
        return packaged
    repository = Path(__file__).resolve().parents[3] / "structures.zip"
    if repository.is_file():
        return repository
    raise EvidenceError("Evaluator structure archive is missing")


def _bundled_structure_bytes(name: str, digest: str) -> bytes:
    with ZipFile(_structure_archive()) as archive:
        content = archive.read(name)
    if hashlib.sha256(content).hexdigest() != digest:
        raise EvidenceError("Evaluator's pinned structure reference is invalid")
    return content


@lru_cache(maxsize=1)
def supplied_cu32_cell():
    """Read the immutable Cu32 asset, which is outside run snapshots."""
    content = _bundled_structure_bytes("structures/cu/cu32.extxyz", _CU32_SOURCE_SHA256)
    return read(StringIO(content.decode()), format="extxyz")


def _positive_run(command: list[str]) -> bool:
    return command[0] == "run" and int(command[1]) > 0


def readiness_assessment(e) -> bool:
    """Require the requested brief, substantive final-state assessment."""
    value = e.results.get("readiness_assessment")
    if not isinstance(value, str):
        return False
    prose = value.lower()
    return bool(
        len(value.split()) >= 6
        and len(value.strip()) >= 30
        and re.search(r"liquid|melt|diffus|suitab|ready|starting", prose)
        and re.search(r"temperatur|\b2500\b", prose)
        and re.search(r"densit|pressur|structur|equilibrat|crystalli", prose)
    )


def prepared_before_heating(e) -> bool:
    """Reject a prepared state written after dynamics began."""
    prepared = e.artifact("prepared_state", "initial_state")
    commands = list(_commands(e))
    first_run = next(
        (i for i, command in enumerate(commands) if _positive_run(command)), None
    )
    if first_run is None:
        return False
    mentions = []
    for i, command in enumerate(commands):
        if (
            command[0]
            not in {
                "write_data",
                "write_restart",
                "read_data",
                "read_restart",
            }
            or len(command) < 2
        ):
            continue
        try:
            same_file = e._path(command[1]) == prepared
        except EvidenceError:
            # Relative output names depend on the LAMMPS working directory.
            same_file = (
                not Path(command[1]).is_absolute()
                and Path(command[1]).name == prepared.name
            )
        if same_file:
            mentions.append((i, command[0]))
    if not mentions:
        # An external preparation script can save the initial state in a
        # different format; the separate geometry check validates its content.
        return len(e.trajectory("prepared_state", "initial_state")) == 1
    # A later write to the same file invalidates an earlier initial-state save.
    last_index, kind = mentions[-1]
    return last_index < first_run and kind in {
        "write_data",
        "write_restart",
        "read_data",
        "read_restart",
    }


def supplied_silicon_potential(e) -> bool:
    """Check the active SW coefficient file at every heating run."""
    style = None
    supplied = False
    runs = 0
    for command in _commands(e):
        if command[0] == "pair_style":
            style = command[1:]
            supplied = False
        elif command[0] == "pair_coeff":
            supplied = False
            if len(command) == 5 and command[1:3] == ["*", "*"] and command[4] == "Si":
                path = command[3]
                if path == _SILICON_SW_PATH:
                    supplied = True
                else:
                    try:
                        source = _file_in_input(
                            e, path, "lammps_input", "lammps_inputs", "potential"
                        )
                    except EvidenceError:
                        pass
                    else:
                        supplied = _sha256(source) == _SILICON_SW_SHA256
        elif _positive_run(command):
            runs += 1
            if style != ["sw"] or not supplied:
                return False
    return runs > 0


def silicon_input_matches_reference(e) -> bool:
    """Check the loaded conventional cell against the submitted reference cell.

    This establishes input consistency, not provenance from Materials Project.
    """
    commands = list(_commands(e))
    first_run = next(
        (i for i, command in enumerate(commands) if _positive_run(command)), None
    )
    if first_run is None:
        return False
    read_commands = [
        command for command in commands[:first_run] if command[0] == "read_data"
    ]
    if (
        len(read_commands) != 1
        or len(read_commands[0]) < 2
        or any(
            command[0] in {"read_restart", "create_atoms", "delete_atoms"}
            for command in commands[:first_run]
        )
    ):
        return False
    path = _file_in_input(
        e, read_commands[0][1], "lammps_input", "lammps_inputs", "source_cell"
    )
    loaded = read(path, format="lammps-data", atom_style="full", units="metal")
    reference = e.trajectory("reference_cell", "starting_cell")[0]
    if len(reference) != 8:
        return False
    species_is_si = set(loaded.get_chemical_symbols()) == {"Si"}
    if not species_is_si and "type" in loaded.arrays:
        types = np.unique(loaded.arrays["type"])
        # A valid LAMMPS data file may assign masses with an input command.
        if len(types) == 1:
            species_is_si = any(
                command[0] == "mass"
                and len(command) >= 3
                and command[1] in {"*", str(types[0])}
                and abs(float(command[2]) - 28.0855) < 0.01
                for command in commands[:first_run]
            )
    if not species_is_si:
        return False
    repeats = np.ones(3, dtype=int)
    for command in commands[:first_run]:
        if command[0] == "replicate":
            if len(command) < 4:
                return False
            factors = np.asarray(command[1:4], dtype=int)
            if np.any(factors < 1):
                return False
            repeats *= factors
    if len(loaded) == 216:
        if not np.array_equal(repeats, [1, 1, 1]):
            return False
        prepared = e.trajectory("prepared_state", "initial_state")[0]
        return (
            set(prepared.get_chemical_symbols()) == {"Si"}
            and _same_periodic_geometry(
                loaded, reference.repeat((3, 3, 3)), atol=2e-3, rtol=2e-4
            )
            and _same_periodic_geometry(loaded, prepared, atol=2e-3, rtol=2e-4)
        )
    if len(loaded) != 8 or not np.array_equal(repeats, [3, 3, 3]):
        return False
    if set(reference.get_chemical_symbols()) != {"Si"}:
        return False
    return _same_periodic_geometry(loaded, reference, atol=2e-4, rtol=2e-5)


def _same_periodic_geometry(actual, expected, *, atol: float, rtol: float) -> bool:
    """Compare periodic atoms up to a common translation and atom ordering."""
    if len(actual) != len(expected) or not np.allclose(
        actual.cell.array, expected.cell.array, atol=atol, rtol=rtol
    ):
        return False
    cell = expected.cell.array
    a = actual.get_scaled_positions(wrap=True)
    b = expected.get_scaled_positions(wrap=True)
    tree = cKDTree(b, boxsize=1.0)
    fractional_bound = 2e-3 / np.linalg.svd(cell, compute_uv=False)[-1]
    for anchor in b:
        shifted = np.mod(a - a[0] + anchor, 1.0)
        distances, indices = tree.query(shifted, distance_upper_bound=fractional_bound)
        if np.any(~np.isfinite(distances)) or len(np.unique(indices)) != len(b):
            continue
        delta = shifted - b[indices]
        delta -= np.rint(delta)
        if np.max(np.linalg.norm(delta @ cell, axis=1)) < 2e-3:
            return True
    return False


@lru_cache(maxsize=1)
def _mp149_live_reference():
    """Read the same conventional mp-149 geometry used by the task tool."""
    key = os.getenv("MP_API_KEY")
    if not key:
        # Programmatic scorers are often called without the benchmark runner,
        # which otherwise loads the repository's .env before constructing tools.
        from dotenv import dotenv_values, find_dotenv

        env_file = find_dotenv(usecwd=True)
        if env_file:
            key = dotenv_values(env_file).get("MP_API_KEY")
    if not key:
        raise UnsupportedEvidence("MP_API_KEY is unavailable to verify the mp-149 cell")
    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with MPRester(key) as rester:
            documents = rester.materials.summary.search(
                material_ids=["mp-149"], fields=["structure"]
            )
        if len(documents) != 1:
            raise ValueError("mp-149 did not resolve to one structure")
        conventional = SpacegroupAnalyzer(
            documents[0].structure
        ).get_conventional_standard_structure()
        return AseAtomsAdaptor.get_atoms(conventional)
    except Exception as exc:
        raise UnsupportedEvidence(
            "The Materials Project mp-149 reference could not be verified"
        ) from exc


def mp149_conventional_cell(e) -> bool:
    """Compare the saved Si8 cell to the live Materials Project reference."""
    candidate = e.trajectory("reference_cell", "starting_cell")[0]
    reference = _mp149_live_reference()
    if not (
        len(candidate) == len(reference) == 8
        and set(candidate.get_chemical_symbols())
        == set(reference.get_chemical_symbols())
        == {"Si"}
        and np.all(candidate.pbc)
    ):
        return False
    cell = candidate.cell.array
    lattice = np.linalg.norm(cell, axis=1)
    expected = np.linalg.norm(reference.cell.array, axis=1)
    if not (
        np.allclose(lattice, expected, rtol=0.01, atol=1e-3)
        and np.allclose(
            cell @ cell.T, np.eye(3) * np.mean(lattice) ** 2, rtol=1e-3, atol=1e-3
        )
    ):
        return False
    # Both are conventional diamond cells; compare their origin-free
    # fractional distance spectra to tolerate CIF axis and atom ordering.
    for atoms in (candidate, reference):
        positions = atoms.get_scaled_positions(wrap=True)
        delta = positions[:, None, :] - positions[None, :, :]
        delta -= np.rint(delta)
        spectrum = np.sort(np.linalg.norm(delta, axis=2)[np.triu_indices(8, 1)])
        if atoms is candidate:
            saved_spectrum = spectrum
        elif not np.allclose(saved_spectrum, spectrum, rtol=0, atol=3e-3):
            return False
    return True


def supplied_silicate_initial_state(e) -> bool:
    """Require the canonical liq1300.dat as input and match its initial atoms."""
    commands = list(_commands(e))
    first_run = next(
        (i for i, command in enumerate(commands) if _positive_run(command)), None
    )
    if first_run is None:
        return False
    loads = [command for command in commands[:first_run] if command[0] == "read_data"]
    if len(loads) != 1 or len(loads[0]) < 2:
        return False
    original_bytes = _bundled_structure_bytes(
        "structures/melt/liq1300.dat", _SILICATE_SOURCE_SHA256
    )
    if loads[0][1] == _SILICATE_SOURCE_PATH:
        # Shared mounted assets are deliberately absent from run snapshots.
        source_bytes = original_bytes
        source_hash = _SILICATE_SOURCE_SHA256
    else:
        source = _file_in_input(
            e, loads[0][1], "lammps_inputs", "lammps_input", "initial_state"
        )
        source_bytes = source.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
    atoms = read(
        StringIO(source_bytes.decode()),
        format="lammps-data",
        atom_style="charge",
        units="real",
        sort_by_id=True,
    )
    if source_hash != _SILICATE_SOURCE_SHA256:
        # A physical copy may have different comments, whitespace or numeric
        # precision. Compare the actual source state with the bundled original.
        original = read(
            StringIO(original_bytes.decode()),
            format="lammps-data",
            atom_style="charge",
            units="real",
            sort_by_id=True,
        )
        if len(atoms) != len(original):
            return False
        if not np.array_equal(atoms.arrays["id"], original.arrays["id"]):
            return False
        if not np.array_equal(
            atoms.get_chemical_symbols(), original.get_chemical_symbols()
        ):
            return False
        if not np.allclose(atoms.cell.array, original.cell.array, atol=2e-5, rtol=1e-6):
            return False
        if not np.allclose(
            atoms.get_initial_charges(),
            original.get_initial_charges(),
            atol=1e-7,
            rtol=0,
        ):
            return False
        if atoms.get_velocities() is None or original.get_velocities() is None:
            return False
        if not np.allclose(
            atoms.get_velocities(), original.get_velocities(), atol=1e-6, rtol=1e-5
        ):
            return False
        difference = (atoms.positions - original.positions) @ np.linalg.inv(
            original.cell.array
        )
        difference -= np.rint(difference)
        if np.max(np.abs(difference @ original.cell.array)) >= 2e-4:
            return False
    initial = _state(e.json("boundary_states", "stage_boundaries")["initial"])
    if len(atoms) != len(initial["ids"]):
        return False
    if not np.array_equal(atoms.arrays["id"], initial["ids"]):
        return False
    if not np.array_equal(atoms.get_chemical_symbols(), initial["species"]):
        return False
    if not np.allclose(atoms.cell.array, initial["cell"], atol=2e-5, rtol=1e-6):
        return False
    if not np.allclose(
        atoms.get_initial_charges(), initial["charges"], atol=1e-7, rtol=0
    ):
        return False
    velocities = atoms.get_velocities()
    if velocities is None or not np.allclose(
        velocities * units.fs, initial["velocities"], atol=1e-7, rtol=1e-5
    ):
        return False
    difference = (atoms.positions - initial["positions"]) @ np.linalg.inv(
        atoms.cell.array
    )
    difference -= np.rint(difference)
    return bool(np.max(np.abs(difference @ atoms.cell.array)) < 2e-4)


def no_silicate_state_resets(e) -> bool:
    """Reject explicit atom or velocity resets after the supplied state is read."""
    commands = list(_commands(e))
    run_indices = [i for i, command in enumerate(commands) if _positive_run(command)]
    if not run_indices:
        return False
    last_run = run_indices[-1]
    # A later, unrelated command cannot change the already saved cooled state.
    saved = next(
        (
            i
            for i, command in enumerate(commands[last_run + 1 :], last_run + 1)
            if command[0] in {"write_data", "write_restart"}
        ),
        last_run,
    )
    loaded = False
    active = {}
    for command in commands[: saved + 1]:
        name = command[0]
        if name == "read_data" and not loaded:
            loaded = True
            continue
        if not loaded:
            continue
        if name in {
            "velocity",
            "read_data",
            "read_restart",
            "create_atoms",
            "delete_atoms",
            "replicate",
        }:
            return False
        if name == "set" and any(token in {"vx", "vy", "vz"} for token in command[1:]):
            return False
        if name == "fix" and len(command) > 3:
            active[command[1]] = command[3]
        elif name == "unfix" and len(command) > 1:
            active.pop(command[1], None)
        elif _positive_run(command) and any(
            style in {"temp/rescale", "langevin", "momentum"}
            for style in active.values()
        ):
            return False
    return loaded


def cooled_endpoint_temperature(e) -> bool:
    """Check the measured endpoint against saved velocities and 300 K target."""
    trace = e.table("thermal_trace", "thermal_traces", "trace")
    measured = (
        trace["measured_temperature_K"]
        if "measured_temperature_K" in trace
        else trace["temperature_K"]
    )
    final = float(measured.iloc[-1])
    if not np.isfinite(final) or abs(final - 300) > 100:
        return False
    state = _state(e.json("boundary_states", "stage_boundaries")["cooling_end"])
    velocities = np.asarray(state["velocities"], dtype=float) / units.fs
    masses = np.asarray(
        [atomic_masses[atomic_numbers[str(symbol)]] for symbol in state["species"]],
        dtype=float,
    )
    thermal_numerator = np.sum(masses[:, None] * velocities**2) / units.kB
    return any(
        np.isclose(final, thermal_numerator / dof, rtol=0.02, atol=2.0)
        for dof in (3 * len(masses), 3 * len(masses) - 3)
    )
