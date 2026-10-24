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


def _stage_samples(stage):
    samples = aliased_value(stage, "samples", "frames", "records", required=False)
    if samples is not None:
        if not isinstance(samples, list):
            raise EvidenceError("Stage samples must be a list")
        return [physical_record(row) for row in samples]
    # Frame-major arrays are another lossless representation of the same samples.
    times = stage.get("time_fs")
    if isinstance(times, list):
        count = len(times)
        dimensions = {
            "time_fs": 1,
            "step": 1,
            "positions_A": 3,
            "momenta": 3,
            "cell_A": 3,
            "potential_energy_eV": 1,
            "forces_eV_A": 3,
            "temperature_K": 1,
            "kinetic_energy_eV": 1,
            "pressure_GPa": 1,
        }
        rows = [{} for _ in range(count)]
        for key, ndim in dimensions.items():
            if key not in stage:
                continue
            values = stage[key]
            if np.asarray(values).ndim == ndim:
                if len(values) != count:
                    raise EvidenceError(f"Misaligned frame-major {key}")
                for row, value in zip(rows, values, strict=True):
                    row[key] = value
        return rows
    raise UnsupportedEvidence(
        "Cannot identify the samples in this stage representation"
    )


def md_records(path, task):
    """Normalize explicit stage objects or ordered rows to the internal protocol.

    Completion markers are optional serialization details. Completion is still
    established from declared run lengths and actual samples by the scorer.
    """
    value = records(path)
    if isinstance(value, list):
        rows = [event_record(row) for row in value]
        # Retain the existing protocol, including malformed/truncated end markers.
        if rows[0].get("event") == "start" and any(
            r.get("event") == "stage" for r in rows
        ):
            return rows
        first = rows[0]
        if first.get("event") == "start" or first.get("phase") in {"meta", "metadata"}:
            header, samples = first, rows[1:]
        else:
            raise UnsupportedEvidence(
                "Cannot identify MD settings separately from samples"
            )
        header = physical_record(header)
        samples = [r for r in samples if r.get("event") != "end"]
        if not samples:
            raise EvidenceError("No molecular-dynamics samples were retained")
        expected = header.get("steps")
        if (
            expected is None
            and "equilibration_steps" in header
            and "production_steps" in header
        ):
            expected = header["equilibration_steps"] + header["production_steps"]
        if (
            expected is not None
            and all(isinstance(r.get("step"), int) for r in samples)
            and max(r["step"] for r in samples) < expected
        ):
            raise EvidenceError(
                "Recorded simulation ends before its declared final step"
            )
        groups = []
        for row in samples:
            label = row.get("stage", row.get("phase"))
            if task in (7, 10) and label in {
                None,
                "equilibration",
                "equil",
                "production",
                "prod",
            }:
                label = 0
            if not groups or groups[-1][0] != label:
                groups.append((label, []))
            groups[-1][1].append(row)
        stages = [
            {**header, "name": label, "samples": samples} for label, samples in groups
        ]
    else:
        header = physical_record(
            value.get("header", value.get("metadata", value.get("settings", {})))
        )
        stages = value.get("stages")
        if stages is None:
            stages = [value]
        if not isinstance(stages, list) or not stages:
            raise EvidenceError("No molecular-dynamics stages were retained")
        stages = [physical_record(stage) for stage in stages]
    output = [{**header, "event": "start"}]
    for index, original in enumerate(stages):
        stage = physical_record({**header, **original})
        samples = _stage_samples(stage)
        if not samples:
            raise EvidenceError("Empty molecular-dynamics stage")
        for row in samples:
            if (
                "cell_A" not in row
                and "cell_A" in stage
                and np.asarray(stage["cell_A"]).ndim == 2
            ):
                row["cell_A"] = stage["cell_A"]
        stage.pop("samples", None)
        stage.pop("frames", None)
        stage.pop("records", None)
        stage["event"] = "stage"
        if task in (6, 9):
            names = {
                "equil": "equilibration",
                "eq": "equilibration",
                "prod": "production",
            }
            stage["name"] = names.get(stage.get("name"), stage.get("name"))
            name = stage.get("name")
            if name in {"equilibration", "production"} and f"{name}_steps" in stage:
                stage["steps"] = stage[f"{name}_steps"]
        elif task == 10 and "stage" not in stage:
            stage["stage"] = index  # explicit list order, not temperature sorting
        end = original.get("end", {})
        end = physical_record(end) if isinstance(end, dict) else {}
        end.update(event="end")
        for key in ("name", "stage"):
            if key in stage:
                end.setdefault(key, stage[key])
        if "step" in samples[-1]:
            end.setdefault("step", samples[-1]["step"])
            end.setdefault("steps", samples[-1]["step"])
        if "time_fs" in samples[-1]:
            end.setdefault("time_fs", samples[-1]["time_fs"])
        if task == 7:
            keys = ("cell_A", "positions_A", "momenta")
            if all(k in samples[0] for k in keys):
                stage.setdefault("state", {k: samples[0][k] for k in keys})
            if all(k in samples[-1] for k in keys):
                end.setdefault("state", {k: samples[-1][k] for k in keys})
        output.append(stage)
        output.extend({**row, "event": "sample"} for row in samples)
        output.append(end)
    return output


def thermodynamic_stage(evidence, task):
    """Read a Level 1 scalar MD log and its explicitly linked restart state.

    The task asks for sampled thermodynamics and a restart, not atom-wise arrays
    at every sampling time. Keep those two representations distinct: this adapter
    never fills absent trajectories with copies of a final structure.
    """
    from ase.io import read

    path = evidence.artifact("run_log")
    value = records(path)
    if not isinstance(value, list):
        return None
    rows = [event_record(row) for row in value]
    if rows[0].get("event") != "start":
        return None
    header = rows[0]
    samples = [row for row in rows[1:] if row.get("event") == "sample"]
    if not samples or any("positions_A" in row or "momenta" in row for row in samples):
        return None
    if not any("temperature_K" in row or "kinetic_energy_eV" in row for row in samples):
        return None
    if any(row.get("event") not in {"sample", "summary", "end"} for row in rows[1:]):
        raise UnsupportedEvidence(
            "Unrecognized records mixed into thermodynamic MD log"
        )

    def require(condition, message):
        if not condition:
            raise EvidenceError(message)

    def field(mapping, key):
        require(key in mapping, f"Thermodynamic log is missing {key}")
        return mapping[key]

    links = []
    for role in ("restart_state", "restart_state_traj", "final_state"):
        if role in evidence._artifacts:
            links.extend(evidence.artifacts(role))
    for row in rows:
        values = row.get("restart_files", [])
        if not isinstance(values, list):
            raise EvidenceError("restart_files must list saved restart artifacts")
        links.extend(
            evidence.linked_path(value, path, register=True) for value in values
        )
    links = list(dict.fromkeys(links))
    require(bool(links), "Thermodynamic log does not link a restartable final state")
    final = None
    sidecars = []
    for link in links:
        if link.suffix.lower() in {".extxyz", ".xyz", ".traj"}:
            try:
                frames = read(
                    link,
                    index=":",
                    format="traj" if link.suffix == ".traj" else "extxyz",
                )
            except (OSError, ValueError, TypeError) as exc:
                raise EvidenceError(
                    f"Cannot read restart state {link.name}: {exc}"
                ) from exc
            require(len(frames) == 1, "A final restart artifact must contain one state")
            atoms = frames[0]
            require("momenta" in atoms.arrays, "Final restart state has no momenta")
            sidecars.append(
                {
                    "numbers": atoms.numbers,
                    "positions": atoms.positions,
                    "cell": atoms.cell.array,
                    "momenta": atoms.get_momenta(),
                }
            )
            if final is None:
                final = atoms
        elif link.suffix.lower() == ".npz":
            try:
                with np.load(link, allow_pickle=False) as archive:
                    keys = {"numbers", "positions", "cell", "momenta"}
                    require(
                        keys <= set(archive.files),
                        "Restart NPZ lacks species, geometry or momenta",
                    )
                    sidecars.append({key: finite_array(archive[key]) for key in keys})
            except (OSError, ValueError, TypeError) as exc:
                raise EvidenceError(
                    f"Cannot read restart NPZ {link.name}: {exc}"
                ) from exc
    if final is None:
        raise UnsupportedEvidence(
            "Cannot identify an ASE restart state in the linked artifacts"
        )
    count = 64 if task == 6 else 108
    require(
        len(final) == count and np.all(final.numbers == 13),
        f"Expected a restart state containing Al{count}",
    )
    require(np.all(final.pbc), "The restart state must preserve periodic boundaries")
    masses = finite_array(final.get_masses(), shape=(count,))
    require(
        np.allclose(masses, 26.9815385, rtol=0, atol=1e-3),
        "Incorrect aluminum restart masses",
    )
    cell = finite_array(final.cell.array, shape=(3, 3))
    require(abs(np.linalg.det(cell)) > 1e-8, "Restart cell is singular")
    expected_gram = np.eye(3) * (3 * 4.05) ** 2
    if task == 6:
        expected_gram = np.full((3, 3), (2 * 4.05) ** 2)
        np.fill_diagonal(expected_gram, 2 * (2 * 4.05) ** 2)
    require(
        np.allclose(np.abs(cell @ cell.T), expected_gram, rtol=0, atol=1e-5),
        "Incorrect fixed FCC cell in restart state",
    )
    require(
        np.allclose(
            finite_array(field(header, "cell_A"), shape=(3, 3)), cell, rtol=0, atol=1e-7
        ),
        "Restart cell differs from the logged fixed cell",
    )
    for key in ("numbers", "n_atoms", "natoms"):
        if key in header:
            expected = final.numbers if key == "numbers" else count
            require(
                np.array_equal(header[key], expected),
                "Logged atom identities/count disagree with restart",
            )
    require(
        type(field(header, "random_seed")) is int,
        "Record an integer initialization seed",
    )
    require(header.get("ensemble", "NVT") == "NVT", "Use the NVT ensemble")
    require(
        field(header, "integrator")
        in {"Langevin", "NVTBerendsen", "NoseHooverChainNVT", "Bussi", "Andersen"},
        "Use a supported fixed-cell NVT thermostat",
    )
    require(float(field(header, "target_temperature_K")) == 300, "Equilibrate at 300 K")
    removed = aliased_value(
        header, "com_removed_once", "com_momentum_removed_once", required=False
    )
    require(removed is True, "Record removal of center-of-mass momentum once")
    require(
        header.get("velocity_rescaling", False) is False,
        "Do not reset or rescale velocities during the run",
    )
    timestep = float(finite_array(field(header, "timestep_fs"), shape=()))
    require(0 < timestep <= 2, "Use an integration timestep of at most 2 fs")
    interval = field(header, "interval")
    require(
        type(interval) is int and interval > 0,
        "Invalid thermodynamic sampling interval",
    )
    if task == 6:
        total = field(header, "steps")
        equil = total
        production = 0
    else:
        equil = field(header, "equilibration_steps")
        production = field(header, "production_steps")
        require(
            type(equil) is int
            and equil > 0
            and type(production) is int
            and production > 0,
            "Invalid equilibration/production lengths",
        )
        total = equil + production
    require(
        type(total) is int and total > 0 and total % interval == 0,
        "Invalid declared run length",
    )
    require(
        equil * timestep >= 2000 and interval * timestep <= 100,
        "Retain at least 2 ps equilibration and sample at most every 100 fs",
    )
    if task == 10:
        require(
            equil % interval == 0
            and production % interval == 0
            and production * timestep >= 3000
            and production // interval >= 32,
            "Retain at least 3 ps production and 32 uniformly sampled production points",
        )
    steps = np.arange(total // interval + 1) * interval
    require(
        len(samples) == len(steps)
        and all(type(row.get("step")) is int for row in samples)
        and np.array_equal([row["step"] for row in samples], steps),
        "Missing, duplicate, unordered or truncated thermodynamic samples",
    )
    time = finite_array(
        [field(row, "time_fs") for row in samples], shape=(len(samples),)
    )
    require(
        np.allclose(time, steps * timestep, rtol=0, atol=1e-6),
        "Sample times disagree with the recorded timestep",
    )
    potential = finite_array(
        [field(row, "potential_energy_eV") for row in samples], shape=(len(samples),)
    )
    kinetic = finite_array(
        [field(row, "kinetic_energy_eV") for row in samples], shape=(len(samples),)
    )
    require(np.all(kinetic > 0), "Kinetic energies must be positive")
    temperature = finite_array(
        [field(row, "temperature_K") for row in samples], shape=(len(samples),)
    )
    dof = header.get("dof", 3 * count)
    require(
        type(dof) is int and dof in {3 * count, 3 * count - 3},
        "Invalid temperature degrees of freedom",
    )
    require(
        np.allclose(temperature, 2 * kinetic / (dof * units.kB), rtol=1e-6, atol=1e-6),
        "Temperatures disagree with recorded kinetic energies",
    )
    require(150 < temperature[0] < 450, "Invalid 300 K velocity initialization")
    directional = None
    if task == 10:
        directional = finite_array(
            [[field(row, key) for key in ("Tx_K", "Ty_K", "Tz_K")] for row in samples],
            shape=(len(samples), 3),
        )
        require(
            np.all(directional > 0)
            and np.allclose(
                directional.mean(axis=1), temperature, rtol=1e-6, atol=1e-6
            ),
            "Directional and overall temperatures disagree",
        )
    for row, epot, ekin in zip(samples, potential, kinetic, strict=True):
        for key, expected in (
            ("etot_eV", epot + ekin),
            ("etot_per_atom_eV", (epot + ekin) / count),
            ("epot_per_atom_eV", epot / count),
            ("volume_A3", abs(np.linalg.det(cell))),
            (
                "density_g_cm3",
                masses.sum() / units.mol / (abs(np.linalg.det(cell)) * 1e-24),
            ),
        ):
            if key in row:
                require(
                    np.isclose(
                        float(finite_array(row[key], shape=())),
                        expected,
                        rtol=1e-6,
                        atol=1e-6,
                    ),
                    f"Inconsistent recorded {key}",
                )
        if "cell_A" in row:
            require(
                np.allclose(
                    finite_array(row["cell_A"], shape=(3, 3)), cell, rtol=0, atol=1e-7
                ),
                "Cell changed during the calculation",
            )
    final_momenta = finite_array(final.get_momenta(), shape=(count, 3))
    if "com_momentum" in samples[0]:
        require(
            np.allclose(
                finite_array(samples[0]["com_momentum"], shape=(3,)),
                0,
                rtol=0,
                atol=1e-6,
            ),
            "Initial center-of-mass momentum was not removed",
        )
    if "com_momentum" in samples[-1]:
        require(
            np.allclose(
                final_momenta.sum(axis=0),
                finite_array(samples[-1]["com_momentum"], shape=(3,)),
                rtol=0,
                atol=1e-6,
            ),
            "Final center-of-mass momentum disagrees with the recorded sample",
        )
    require(
        np.isclose(final.get_kinetic_energy(), kinetic[-1], rtol=1e-6, atol=1e-6),
        "Final restart momenta disagree with the last logged kinetic energy",
    )
    if directional is not None:
        restart_directional = (final_momenta**2 / masses[:, None]).sum(axis=0) / (
            (dof / 3) * units.kB
        )
        require(
            np.allclose(restart_directional, directional[-1], rtol=1e-6, atol=1e-6),
            "Final restart momenta disagree with directional temperatures",
        )
    stored_energy = getattr(final.calc, "results", {}).get("energy")
    if stored_energy is not None:
        require(
            np.isclose(stored_energy, potential[-1], rtol=1e-7, atol=1e-6),
            "Restart potential energy differs from the final logged energy",
        )
    for state in sidecars:
        require(
            np.array_equal(state["numbers"], final.numbers),
            "Conflicting restart species",
        )
        require(
            np.allclose(
                finite_array(state["cell"], shape=(3, 3)), cell, rtol=0, atol=1e-7
            ),
            "Conflicting restart cells",
        )
        delta = (
            finite_array(state["positions"], shape=(count, 3)) - final.positions
        ) @ np.linalg.inv(cell)
        require(
            np.allclose(delta - np.rint(delta), 0, rtol=0, atol=1e-7)
            and np.allclose(
                finite_array(state["momenta"], shape=(count, 3)),
                final_momenta,
                rtol=0,
                atol=1e-7,
            ),
            "Restart artifacts contain different final states",
        )
    distances = final.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    require(distances.min() > 1.0, "Final aluminum state has overlapping atoms")
    counts = {"n_samples": len(samples)}
    claims = {}
    if task == 6:
        tail = temperature[len(temperature) // 2 :]
        claims.update(
            final_half_mean_T_K=float(tail.mean()), final_half_std_T_K=float(tail.std())
        )
    else:
        selected = steps > equil
        energies = (potential[selected] + kinetic[selected]) / count
        counts["n_prod_samples"] = int(selected.sum())
        claims.update(
            mean_etot_per_atom_eV=float(energies.mean()),
            mean_T=float(temperature[selected].mean()),
            first_half_E=float(energies[: len(energies) // 2].mean()),
            second_half_E=float(energies[len(energies) // 2 :].mean()),
        )
        claims.update(
            {
                key: float(directional[selected, index].mean())
                for index, key in enumerate(("mean_Tx", "mean_Ty", "mean_Tz"))
            }
        )
    for summary in (row for row in rows if row.get("event") == "summary"):
        for key, expected in counts.items():
            if key in summary:
                require(
                    type(summary[key]) is int and summary[key] == expected,
                    f"Summary {key} disagrees with retained samples",
                )
        for key, expected in claims.items():
            if key in summary:
                actual = float(finite_array(summary[key], shape=()))
                require(
                    np.isclose(actual, expected, rtol=0.01, atol=1e-6),
                    f"Summary {key} disagrees with retained samples",
                )
    check_units(
        header, {"time": "fs", "position": "Angstrom", "energy": "eV", "momenta": "ASE"}
    )
    return {
        **header,
        "name": "equilibration",
        "stage": 0,
        "time": time,
        "temperature": temperature,
        "directional": directional,
        "potential": potential,
        "kinetic": kinetic,
        "energy": (potential + kinetic) / (count if task == 10 else 1),
        "cell": cell,
        "masses": masses,
        "production": steps > equil,
        "equilibration_steps": equil,
        "production_steps": production,
        "scalar_thermodynamics": True,
        "final_atoms": final,
    }


def check_model_digest(header, expected):
    """Check an explicitly retained digest; absence is not a different model."""
    if "model_sha256" in header and header["model_sha256"] != expected:
        raise EvidenceError(
            "Recorded checkpoint digest differs from the supplied model"
        )


def check_initial_momenta(atoms, header, target, momenta):
    """Check initialization physically; reproduce a generator only if declared."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    observed = finite_array(momenta, shape=(len(atoms), 3))
    temperature = np.sum(observed**2 / atoms.get_masses()[:, None]) / (
        3 * len(atoms) * units.kB
    )
    if (
        not np.allclose(observed.sum(axis=0), 0, rtol=0, atol=1e-7)
        or not 0.5 * target < temperature < 1.5 * target
    ):
        raise EvidenceError(
            "Invalid initialization temperature or center-of-mass momentum"
        )
    method = header.get("velocity_rng", header.get("rng"))
    if method is None:
        return
    if method in {"numpy.default_rng", "numpy.default_rng(PCG64)", "PCG64"}:
        rng = np.random.default_rng(header["random_seed"])
    elif method in {"numpy.random.RandomState", "numpy.RandomState", "RandomState"}:
        rng = np.random.RandomState(header["random_seed"])
    else:
        raise UnsupportedEvidence(f"Unsupported declared velocity generator: {method}")
    expected = atoms.copy()
    MaxwellBoltzmannDistribution(expected, temperature_K=target, rng=rng)
    Stationary(expected, preserve_temperature=header.get("preserve_temperature", True))
    if not np.allclose(expected.get_momenta(), observed, rtol=0, atol=1e-7):
        raise EvidenceError(
            "Initial momenta do not reproduce the declared seed and velocity generator"
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
        methods = {
            "analytical": "analytic_hessian",
            "analytical_hessian": "analytic_hessian",
            "finite_difference": "central_difference",
            "finite-difference": "central_difference",
        }
        if isinstance(row.get("method"), str):
            row["method"] = methods.get(row["method"], row["method"])
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
        if "hessian_eV_A2" in row:
            row.setdefault("method", "retained_hessian")
        elif "mode_energies_eV" in row:
            row.setdefault("method", "retained_modes")
        elif "method" not in row:
            raise UnsupportedEvidence(
                "Cannot identify forces, a Hessian, or mode energies in the phonon record"
            )
        converted.append(row)
    log["calculations"] = scientific_record(converted)
    return log


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
            "displacements_A": ("displacements",),
            "energies_eV": ("energies",),
            "force_constants_eV_A2": (
                "force_constants_raw",
                "force_constants_raw_row0",
            ),
        },
    )
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
    if displacement.ndim == 3:
        if displacement.shape[1:] != (125, 3):
            raise EvidenceError("Incorrect displacement-array dimensions")
        if not np.allclose(displacement[:, 1:], 0, rtol=0, atol=1e-12):
            raise UnsupportedEvidence(
                "Force displacements on multiple atoms need a different reconstruction adapter"
            )
        displacement = displacement[:, 0]
        record["displacements_A"] = displacement
    if displacement.ndim != 2 or displacement.shape[1] != 3 or not len(displacement):
        raise EvidenceError("Retain a nonempty array of Cartesian displacement vectors")
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
