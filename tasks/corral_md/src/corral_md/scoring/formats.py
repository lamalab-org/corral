"""Interpret retained scientific data without imposing an unpublished log protocol.

These adapters only rename, reshape, or derive values from submitted data. They
never execute a submitted program, manufacture measurements, or establish model
identity from the model requested by a task. Legacy event logs remain readable.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from ase import units
from ase.data import atomic_masses, atomic_numbers

from .common import EvidenceError, UnsupportedEvidence, aliased_value, finite_array


def artifact_reader(function):
    """Expose an unsupported field layout as a reader limitation, not a zero."""

    @functools.wraps(function)
    def read(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyError as exc:
            raise UnsupportedEvidence(
                f"{function.__name__}: cannot locate {exc.args[0]!r} in this "
                "artifact layout; the task does not prescribe that field name"
            ) from exc

    return read


class ScientificRecord(dict):
    """An input mapping whose unknown layout is distinguishable from a code bug."""

    def __missing__(self, key):
        raise UnsupportedEvidence(
            f"Cannot locate {key!r} in the retained calculation data; "
            "this field name is not part of the task's submission example"
        )


def scientific_record(value):
    if isinstance(value, dict):
        return ScientificRecord(
            {key: scientific_record(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return [scientific_record(child) for child in value]
    return value


def rename(value, aliases):
    if not isinstance(value, Mapping):
        raise EvidenceError("Expected an object containing scientific records")
    result = dict(value)
    for target, names in aliases.items():
        found = aliased_value(value, target, *names, required=False)
        if found is not None:
            result[target] = found
    return result


def records(path):
    """Read JSON objects/arrays or JSONL, including whitespace between records."""
    text = Path(path).read_text()
    if not text.strip():
        raise EvidenceError("The calculation record is empty")

    def invalid(value):
        raise EvidenceError(f"Nonfinite JSON value: {value}")

    try:
        try:
            value = json.loads(text, parse_constant=invalid)
        except json.JSONDecodeError:
            value = [
                json.loads(line, parse_constant=invalid)
                for line in text.splitlines()
                if line.strip()
            ]
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Malformed calculation JSON at line {exc.lineno}") from exc
    if isinstance(value, dict):
        return value
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(r, dict) for r in value)
    ):
        raise EvidenceError(
            "Calculation records must be a nonempty object or list of objects"
        )
    return value


FIELD_ALIASES = {
    "random_seed": ("seed", "seed_velocities", "velocity_seed", "velocity_rng_seed"),
    "model_sha256": ("checkpoint_sha256", "teacher_sha256"),
    "default_dtype": ("dtype", "precision"),
    "numbers": ("atomic_numbers",),
    "masses_amu": ("masses",),
    "positions_A": ("positions", "positions_angstrom"),
    "cell_A": ("cell", "cell_angstrom"),
    "momenta": ("momenta_ase_units",),
    "potential_energy_eV": ("epot_eV", "potential_energy", "energy_eV"),
    "kinetic_energy_eV": ("ekin_eV",),
    "forces_eV_A": ("forces", "forces_eV_per_A", "forces_eV_per_angstrom"),
    "time_fs": ("elapsed_time_fs",),
    "timestep_fs": ("dt_fs",),
    "target_temperature_K": (
        "T_target_K",
        "temperature_target_K",
        "target_T_K",
        "T_target",
    ),
    "equilibration_steps": ("equil_steps", "n_equil_steps", "n_equil"),
    "production_steps": ("prod_steps", "n_prod_steps", "n_production"),
    "steps": ("nsteps", "n_steps"),
    "interval": ("sample_interval_steps", "sample_every", "sampling_interval_steps"),
    "integrator": ("thermostat",),
    "temperature_K": ("T_K",),
    "stage": ("stage_index",),
}


def physical_record(value):
    if isinstance(value, Mapping):
        value = dict(value)
        for key in ("integrator", "thermostat"):
            if isinstance(value.get(key), str):
                names = re.findall(
                    r"\b(?:Langevin|NVTBerendsen|NoseHooverChainNVT|Bussi|Andersen|IsotropicMTKNPT|NPTBerendsen|VelocityVerlet)\b",
                    value[key],
                )
                if len(set(names)) == 1:
                    value[key] = names[0]
    result = rename(value, FIELD_ALIASES)
    if "symbols" in result and "numbers" not in result:
        symbols = result["symbols"]
        if isinstance(symbols, list):
            try:
                result["numbers"] = [atomic_numbers[s] for s in symbols]
            except KeyError as exc:
                raise EvidenceError("Invalid atomic symbol") from exc
    if "numbers" in result and "masses_amu" not in result:
        numbers = np.asarray(result["numbers"])
        if numbers.dtype.kind not in "iu" or np.any(
            (numbers < 1) | (numbers >= len(atomic_masses))
        ):
            raise EvidenceError("Invalid atomic numbers")
        # ASE's natural masses are the default for explicitly identified species.
        result["masses_amu"] = atomic_masses[numbers].tolist()
    for source, target, factor in (
        ("time_ps", "time_fs", 1000),
        ("pressure_bar", "pressure_GPa", 0.0001),
        ("P_target_bar", "target_pressure_GPa", 0.0001),
    ):
        if source in result:
            converted = finite_array(result[source]) * factor
            if target in result and not np.allclose(
                finite_array(result[target]), converted, rtol=1e-7, atol=1e-9
            ):
                raise EvidenceError(f"Conflicting units in {source} and {target}")
            result[target] = converted.tolist()
    if (
        "momenta" not in result
        and "velocities_A_fs" in result
        and "masses_amu" in result
    ):
        result["momenta"] = (
            finite_array(result["velocities_A_fs"])
            * finite_array(result["masses_amu"])[:, None]
            / units.fs
        ).tolist()
    if "interval" not in result and result.get("timestep_fs"):
        spacing = aliased_value(
            result, "sample_interval_fs", "sample_every_fs", required=False
        )
        if spacing is not None:
            interval = float(spacing) / float(result["timestep_fs"])
            if interval.is_integer():
                result["interval"] = int(interval)
    return result


def event_record(value):
    row = physical_record(value)
    label = row.get("event", row.get("type", row.get("kind")))
    aliases = {
        "meta": "start",
        "metadata": "start",
        "header": "start",
        "frame": "sample",
        "step": "sample",
        "configuration": "sample",
        "finish": "end",
        "complete": "end",
        "stage_start": "stage",
        "stage_end": "end",
    }
    if isinstance(label, str):
        row["event"] = aliases.get(label.lower(), label)
    return row


def check_model_digest(header, expected):
    """Check an explicitly retained digest; absence is not a different model."""
    if "model_sha256" in header and header["model_sha256"] != expected:
        raise EvidenceError(
            "Recorded checkpoint digest differs from the supplied model"
        )


def check_units(header, expected):
    """Unit metadata is optional when quantity names/the task specify the units."""
    if "units" not in header:
        return
    supplied = header["units"]
    if not isinstance(supplied, dict):
        raise UnsupportedEvidence("Cannot interpret the units declaration")
    spellings = {
        "A": "Angstrom",
        "Å": "Angstrom",
        "angstrom": "Angstrom",
        "eV/A": "eV/Angstrom",
    }
    for name, unit in supplied.items():
        if name in expected and spellings.get(unit, unit) != expected[name]:
            raise EvidenceError(f"Incorrect {name} units: {unit}")


def calculation_log(value):
    """Normalize calculation metadata without assuming that it proves execution."""
    log = rename(
        value,
        {
            "energy_unit": ("energy_units",),
            "force_unit": ("force_units",),
            "default_dtype": ("dtype", "precision"),
            "calculations": ("records", "rows"),
        },
    )
    for key in ("teacher_model", "student_model", "model"):
        if isinstance(log.get(key), str) and Path(log[key]).name in {
            "teacher.model",
            "student.model",
        }:
            log[key] = Path(log[key]).name
    if log.get("force_unit") in {"eV/A", "eV/Å", "eV/angstrom"}:
        log["force_unit"] = "eV/Angstrom"
    # Explicit settings embedded in human-readable calculator labels are data,
    # not executable expressions; only these bounded literals are interpreted.
    description = log.get("calculator", "")
    if isinstance(description, str):
        for field, expression in (
            ("default_dtype", r"\b(float32|float64)\b"),
            ("dispersion", r"\bdispersion\s*=\s*(True|False)\b"),
        ):
            found = re.findall(expression, description)
            if len(set(found)) == 1:
                val = found[0] if field == "default_dtype" else found[0] == "True"
                if field in log and log[field] != val:
                    raise EvidenceError(f"Conflicting calculator {field}")
                log[field] = val
    return scientific_record(log)


def dataset_records(path, level):
    """Read row-oriented datasets as well as ordered calculation event logs."""
    value = records(path)
    if isinstance(value, dict):
        if "records" in value:
            value = value["records"]
        elif "datasets" in value:
            raise UnsupportedEvidence(
                "Dataset container does not establish pipeline fitting/test chronology"
            )
        else:
            value = [value]
    rows = [event_record(row) for row in value]
    if rows[0].get("event") == "start":
        return rows
    if level != 1:
        raise UnsupportedEvidence(
            "Cannot reconstruct training, pipeline freeze and test chronology from this log layout"
        )
    normalized = []
    for index, original in enumerate(rows):
        normalized_row = physical_record(original)
        row = rename(
            normalized_row,
            {
                "energy_eV": ("potential_energy_eV",),
                "positions": ("positions_A",),
                "cell": ("cell_A",),
            },
        )
        if "row_id" not in row:
            row["row_id"] = row.get("config_index", row.get("index", index))
        if not {"positions", "cell", "energy_eV"}.issubset(row):
            raise UnsupportedEvidence(
                "Cannot identify dataset geometry and total energy in this record layout"
            )
        normalized.append(row)
    first = normalized[0]
    header = {"event": "start"}
    for key in ("model_sha256", "default_dtype", "units"):
        values = [r[key] for r in normalized if key in r]
        if values:
            if any(value != values[0] for value in values[1:]):
                raise EvidenceError(f"Dataset rows disagree on {key}")
            header[key] = values[0]
    seeds = [r.get("random_seed") for r in normalized]
    if not seeds or seeds[0] is None or any(seed != seeds[0] for seed in seeds):
        raise EvidenceError("Retain one reproducible dataset generation seed")
    rng = first.get("rng")
    return [
        header,
        {
            "event": "dataset",
            "name": "train",
            "seed": seeds[0],
            "rng": rng,
            "rows": normalized,
        },
        {"event": "end"},
    ]


def phonon_log(value, level):
    # The task names the artifact, but does not prescribe its row-container key.
    if not isinstance(value, Mapping):
        # Keep malformed input in the existing evidence-validation path.
        return calculation_log(value)
    containers = (
        "records",
        "rows",
        *(key for key in ("results", "data") if isinstance(value.get(key), list)),
    )
    log = calculation_log(rename(value, {"calculations": containers}))
    rows = log.get("calculations")
    if isinstance(rows, list) and not rows:
        raise EvidenceError("The phonon calculation log contains no calculations")
    if not isinstance(rows, list):
        raise UnsupportedEvidence("Cannot identify the individual phonon calculations")
    converted = []
    for original in rows:
        row = rename(
            original,
            {
                "isotropic_strain": ("strain_fraction",),
                "hessian_eV_A2": ("hessian_eV_per_A2",),
                "mode_energies_eV": (
                    "gamma_frequencies_eV",
                    "gamma_mode_energies_eV",
                    "frequencies_eV",
                ),
            },
        )
        if level == 1:
            row.setdefault(
                "uniaxial_strain", 0.0
            )  # the task varies isotropic strain only
        if "structure" not in row:
            source_data = {**log.get("reference", {}), **row}
            source = physical_record(source_data)
            mass = aliased_value(source, "mass_amu", "mass", required=False)
            if mass is None:
                mass = aliased_value(log, "mass_amu", "mass", required=False)
            if mass is not None:
                masses = np.full(2, float(finite_array(mass, shape=())))
                if (
                    "masses_amu" in source_data or "masses" in source_data
                ) and not np.allclose(source["masses_amu"], masses, rtol=0, atol=1e-10):
                    raise EvidenceError("Conflicting phonon calculation masses")
                source["masses_amu"] = masses.tolist()
            positions = source.get("positions_A")
            if (
                positions is None
                and "scaled_positions" in source
                and "cell_A" in source
            ):
                positions = (
                    finite_array(source["scaled_positions"], shape=(2, 3))
                    @ finite_array(source["cell_A"], shape=(3, 3))
                ).tolist()
            if positions is None or "cell_A" not in source:
                raise UnsupportedEvidence(
                    "Cannot locate the retained geometry for the phonon calculation"
                )
            row["structure"] = {
                "numbers": source.get("numbers", [14, 14]),
                "pbc": source.get("pbc", [True, True, True]),
                "cell": source["cell_A"],
                "positions": positions,
                "masses": source.get("masses_amu", [28.0855, 28.0855]),
            }
        if "hessian_eV_A2" not in row and "force_constants_eV_per_A2" in row:
            constants = finite_array(row["force_constants_eV_per_A2"])
            if constants.shape == (2, 2, 3, 3):
                constants = constants.transpose(0, 2, 1, 3).reshape(6, 6)
            elif constants.shape == (2, 3, 2, 3):
                constants = constants.reshape(6, 6)
            if constants.shape != (6, 6):
                raise UnsupportedEvidence("Unsupported force-constant tensor layout")
            row["hessian_eV_A2"] = constants.tolist()
        if not (
            "mode_energies_eV" in row
            or "hessian_eV_A2" in row
            or {"forces_plus_eV_A", "forces_minus_eV_A", "displacement_A"} <= row.keys()
        ):
            raise UnsupportedEvidence(
                "Cannot identify forces, a Hessian, or mode energies in the phonon record"
            )
        converted.append(row)
    log["calculations"] = scientific_record(converted)
    return log


def _force_displacements(value, *, indexed=False, displaced_atom=None):
    """Read Cartesian vectors or the indexed tables saved by force calculations.

    Only the generic ``displacements`` field admits indexed tables. The explicit
    ``displacements_A`` field always denotes Cartesian vectors. A three-column
    indexed table is recognizable by its integer atom/axis columns and coverage
    of all three axes; ordinary central Cartesian differences have signed
    coordinate columns instead.
    """
    displacement = finite_array(value)
    atom_xyz = indexed and displacement.ndim == 2 and displacement.shape[1] == 4
    atom_axis = (
        indexed
        and displacement.ndim == 2
        and displacement.shape[1] == 3
        and np.all(displacement[:, 0] == np.rint(displacement[:, 0]))
        and np.all((displacement[:, 0] >= 0) & (displacement[:, 0] < 125))
        and set(displacement[:, 1]) == {0, 1, 2}
    )
    if atom_xyz or atom_axis:
        atoms = displacement[:, 0]
        if not np.all((atoms == np.rint(atoms)) & (atoms >= 0) & (atoms < 125)):
            raise EvidenceError("Invalid displaced atom indices")
        if displaced_atom is not None and not np.all(
            atoms == finite_array(displaced_atom, shape=())
        ):
            raise EvidenceError("Conflicting displaced atom indices")
        if np.any(atoms != 0):
            raise UnsupportedEvidence(
                "Force displacements on atoms other than atom zero need an atom-order adapter"
            )
        if atom_xyz:
            displacement = displacement[:, 1:]
        else:
            vectors = np.zeros_like(displacement)
            vectors[np.arange(len(vectors)), displacement[:, 1].astype(int)] = (
                displacement[:, 2]
            )
            displacement = vectors
    if displacement.ndim == 3:
        if displacement.shape[1:] != (125, 3):
            raise EvidenceError("Incorrect displacement-array dimensions")
        if not np.allclose(displacement[:, 1:], 0, rtol=0, atol=1e-12):
            raise UnsupportedEvidence(
                "Force displacements on multiple atoms need a different reconstruction adapter"
            )
        displacement = displacement[:, 0]
    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise UnsupportedEvidence("Cannot interpret the retained displacement layout")
    return displacement


def normalize_force_record(value):
    """Normalize NPZ force calculations; displacement order is not prescribed."""
    record = rename(
        value,
        {
            "cell_A": ("supercell_cell", "cell"),
            "positions_A": ("positions",),
            "numbers": ("supercell_numbers", "atomic_numbers"),
            "masses_amu": ("supercell_masses", "masses"),
            "forces_eV_A": ("forces", "forces_eV_per_A"),
            "energies_eV": ("energies",),
            "force_constants_eV_A2": (
                "force_constants_raw",
                "force_constants_raw_row0",
            ),
        },
    )
    if "symbols" in record:
        symbols = np.asarray(record["symbols"])
        if symbols.ndim != 1 or not len(symbols):
            raise EvidenceError("Retain one chemical symbol per atom")
        try:
            numbers = np.array([atomic_numbers[s] for s in symbols])
        except (KeyError, TypeError) as exc:
            raise EvidenceError("Invalid atomic symbol") from exc
        if "numbers" in record and not np.array_equal(record["numbers"], numbers):
            raise EvidenceError("Conflicting chemical symbols and atomic numbers")
        record["numbers"] = numbers
    displacement = None
    for key in ("displacements_A", "displacements"):
        if key not in record:
            continue
        candidate = _force_displacements(
            record[key],
            indexed=key == "displacements",
            displaced_atom=record.get("displaced_atom"),
        )
        if displacement is not None and (
            displacement.shape != candidate.shape
            or not np.allclose(displacement, candidate, rtol=0, atol=1e-12)
        ):
            raise EvidenceError("Conflicting displacement records")
        displacement = candidate
    if displacement is not None:
        record["displacements_A"] = displacement
    metadata = record.get("metadata", {})
    if isinstance(metadata, np.ndarray):
        try:
            metadata = json.loads(str(metadata.item()))
        except (ValueError, json.JSONDecodeError) as exc:
            raise EvidenceError("Malformed force-calculation metadata") from exc
    if not isinstance(metadata, dict):
        raise EvidenceError("Force metadata must describe an object")
    metadata = calculation_log(metadata)
    if "method" not in metadata:
        metadata["method"] = (
            "analytical" if "hessian_eV_A2" in record else "central_difference"
        )
    record["metadata"] = metadata
    compact_displacements = {"displacement_axes", "displacement_values"}
    if compact_displacements & record.keys():
        if not compact_displacements <= record.keys():
            raise EvidenceError("Retain both displacement axes and displacement values")
        axes = finite_array(record["displacement_axes"], ndim=1)
        values = finite_array(record["displacement_values"], shape=axes.shape)
        if not np.all((axes == np.rint(axes)) & (axes >= 0) & (axes < 3)):
            raise EvidenceError(
                "Displacement axes must be Cartesian indices 0, 1, or 2"
            )
        atom = finite_array(record.get("displaced_atom", 0), shape=())
        if atom != 0:
            raise UnsupportedEvidence(
                "A displaced atom other than atom zero needs an atom-order adapter"
            )
        displacement = np.zeros((len(axes), 3))
        displacement[np.arange(len(axes)), axes.astype(int)] = values
        if "displacements_A" in record:
            existing = finite_array(record["displacements_A"])
            if existing.shape == (len(axes), 125, 3):
                expected = np.zeros_like(existing)
                expected[:, 0] = displacement
            else:
                expected = displacement
            if existing.shape != expected.shape or not np.allclose(
                existing, expected, rtol=0, atol=1e-12
            ):
                raise EvidenceError("Conflicting displacement records")
        else:
            record["displacements_A"] = displacement
    if "forces_eV_A" not in record or "displacements_A" not in record:
        raise UnsupportedEvidence(
            "Cannot locate displaced force calculations in this NPZ layout"
        )
    displacement = finite_array(record["displacements_A"])
    if "positions_A" not in record and "supercell_positions" in record:
        reference = finite_array(record["supercell_positions"], shape=(125, 3))
        positions = np.repeat(reference[None], len(displacement), axis=0)
        positions[:, 0] += displacement
        record["positions_A"] = positions
    if "numbers" in record and "masses_amu" not in record:
        mass = aliased_value(record, "mass", "mass_Pd", "mass_amu", required=False)
        if mass is not None:
            record["masses_amu"] = np.full(len(record["numbers"]), float(mass))
        else:
            record["masses_amu"] = physical_record({"numbers": record["numbers"]})[
                "masses_amu"
            ]
    if "force_constants_eV_A2" in record:
        constants = finite_array(record["force_constants_eV_A2"])
        if constants.shape == (5, 5, 5, 3, 3):
            # A lattice-indexed tensor is not necessarily in the recorded atom
            # order. Use the saved geometry to establish that correspondence.
            frames = finite_array(record["positions_A"], ndim=3)
            if not len(frames) or frames.shape[1:] != (125, 3):
                raise EvidenceError("Expected force calculations for Pd125")
            positions = frames[0].copy()
            positions[0] -= displacement[0]
            cell = finite_array(record["cell_A"], shape=(3, 3)) / 5
            fractional = (positions - positions[0]) @ np.linalg.inv(cell)
            indices = np.rint(fractional).astype(int) % 5
            if not np.allclose(fractional, np.rint(fractional), rtol=0, atol=1e-6):
                raise EvidenceError(
                    "Force-constant lattice indices disagree with geometry"
                )
            if "lattice_index" in record:
                saved = finite_array(record["lattice_index"], shape=(125, 3))
                if not np.all(saved == np.rint(saved)) or not np.array_equal(
                    saved.astype(int) % 5, indices
                ):
                    raise EvidenceError(
                        "Force-constant lattice indices disagree with geometry"
                    )
            record["force_constants_eV_A2"] = constants[tuple(indices.T)]
    # Periodicity is part of the specified bulk calculation. An explicit
    # contradictory declaration still fails the geometry check.
    record.setdefault("pbc", [True, True, True])
    return scientific_record(record)
