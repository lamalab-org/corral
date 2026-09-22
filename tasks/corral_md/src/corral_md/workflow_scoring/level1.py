"""Level-1-only rubrics for the preparatory Corral MD tasks.

Each Level 1 task is the first, reusable part of its paired Level 2 workflow.
These evaluators deliberately inspect only that shared part.  They never award
or withhold credit for the later production, fitting, or analysis stages.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers

from . import task_1, task_2, task_3, task_4, task_5, task_7, task_9, task_10
from .common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
    scientific_screen,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _supported(function: Callable[[], object]) -> Callable[[], object]:
    def check():
        try:
            return function()
        except UnsupportedEvidence as exc:
            return None, str(exc)

    return check


def _stored_energy(atoms) -> float:
    for name in ("potential_energy_eV", "potential_energy", "energy"):
        if name in atoms.info:
            return float(finite_array(atoms.info[name], shape=()))
    stored = getattr(getattr(atoms, "calc", None), "results", {})
    if "energy" not in stored:
        raise EvidenceError("A stored potential energy is required")
    return float(finite_array(stored["energy"], shape=()))


def _time_fs(atoms) -> float:
    if "time_fs" in atoms.info:
        return float(finite_array(atoms.info["time_fs"], shape=()))
    return 1000 * float(finite_array(atoms.info["time_ps"], shape=()))


def _kinetic_temperature(atoms, dof: int) -> float:
    momenta = finite_array(atoms.arrays["momenta"], shape=(len(atoms), 3))
    masses = finite_array(atoms.get_masses(), shape=(len(atoms),))
    return float(np.sum(momenta**2 / masses[:, None]) / (dof * units.kB))


def _task_1(e: Evidence, r: Rubric) -> None:
    def endpoint_state():
        initial = e.trajectory("prepared_state", "initial_state")
        final = e.trajectory("heating_end_state")
        if len(initial) != 1 or len(final) != 1:
            raise EvidenceError("Save one prepared state and one end-of-heating state")
        initial, final = initial[0], final[0]
        if (
            len(final) != 216
            or set(final.get_chemical_symbols()) != {"Si"}
            or not np.all(final.pbc)
            or np.linalg.det(final.cell.array) <= 0
        ):
            return False
        left, right = task_1._ids(initial), task_1._ids(final)
        if len(np.unique(right)) != 216 or set(left) != set(right):
            return False
        velocity = final.get_velocities()
        if velocity is None or not np.isfinite(velocity).all():
            return False
        table = task_1._thermo(e, 1000)
        mass = final.get_masses().sum()
        density = mass * 1.66053906660 / final.get_volume()
        if not task_1._close(density, table.density_g_cm3.iloc[-1], rtol=0.002):
            return False
        if abs(final.get_temperature() - table.temperature_K.iloc[-1]) > 0.02 * max(
            final.get_temperature(), table.temperature_K.iloc[-1]
        ):
            return False
        return "time_ps" not in final.info or task_1._close(final.info["time_ps"], 1000)

    r.check("diamond_preparation", 25, lambda: task_1._geometry(e))
    r.check(
        "heating_protocol_and_log",
        40,
        _supported(lambda: task_1._protocol(e, 1000)),
        "The saved input, raw log, and thermo table must describe only the requested 300-2500 K preparation.",
    )
    r.check("heating_state_and_thermo_consistency", 25, endpoint_state)
    r.unverified(
        "execution_provenance",
        "Saved inputs and states establish internal consistency, not the identity of the executed potential or the absence of undisclosed resets.",
    )


def _task_2(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def trace():
        table = task_2._trace(e, {"cooling"})
        if len(table) < 2 or np.any(np.diff(table.time_ps) <= 0):
            raise EvidenceError("Cooling trace needs chronological endpoint samples")
        elapsed = table.time_ps.to_numpy() - table.time_ps.iloc[0]
        if not task_2._close(elapsed[[0, -1]], [0, 370], atol=0.5, rtol=0.01):
            raise EvidenceError("Cooling must run for 370 ps")
        target = 4000 - 10 * elapsed
        if not task_2._close(table.target_temperature_K, target, atol=5, rtol=0.01):
            raise EvidenceError("Target temperatures do not reproduce 10 K/ps cooling")
        return table

    def boundary_states():
        states = e.json("boundary_states", "stage_boundaries")
        initial = task_2._state(states["initial"])
        final = task_2._state(states["cooling_end"])
        if not (
            np.array_equal(initial["ids"], final["ids"])
            and np.array_equal(initial["species"], final["species"])
            and task_2._close(initial["charges"], final["charges"], atol=1e-7, rtol=0)
            and task_2._close(
                final["time_ps"] - initial["time_ps"], 370, atol=0.5, rtol=0.01
            )
        ):
            return False
        table = trace()
        for state, row in ((initial, table.iloc[0]), (final, table.iloc[-1])):
            mass = sum(
                atomic_masses[atomic_numbers[str(symbol)]]
                for symbol in state["species"]
            )
            density = mass * 1.66053906660 / np.linalg.det(state["cell"])
            if not task_2._close(density, row.density_g_cm3, atol=1e-5):
                return False
        return True

    r.check("cooling_schedule_and_observables", 25, lambda: bool(len(trace())))
    r.check(
        "bks_pppm_npt_physics",
        25,
        _supported(lambda: task_2._physics(e)),
    )
    r.check(
        "cooling_input_cycle",
        20,
        _supported(lambda: task_2._input_cycle(e, 370)),
    )
    r.check(
        "cooling_trace_matches_raw_log",
        10,
        _supported(lambda: task_2._logged_trace(e, {"cooling"})),
    )
    r.check("cooled_state_identity_and_density", 10, boundary_states)
    r.unverified(
        "execution_provenance",
        "Saved inputs, logs, and states cannot prove which supplied potential was executed or exclude an undisclosed reset.",
    )


def _task_3(e: Evidence, r: Rubric) -> None:
    expected = np.round(np.arange(0.6, 5.01, 0.1), 8)

    @lru_cache(None)
    def dataset():
        frames = e.trajectory("dataset", "labeled_dataset", "training_dataset")
        if len(frames) != 45:
            raise EvidenceError("Expected exactly 45 silver-dimer structures")
        distances = []
        for atoms in frames:
            if (
                len(atoms) != 2
                or atoms.get_chemical_symbols() != ["Ag", "Ag"]
                or np.any(atoms.pbc)
            ):
                raise EvidenceError("Each dataset row must be an isolated Ag2 dimer")
            distances.append(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
        order = np.argsort(distances)
        if not np.allclose(np.asarray(distances)[order], expected, rtol=0, atol=1e-5):
            raise EvidenceError("Dimer separations must cover 0.6-5.0 Angstrom")
        return [frames[index] for index in order]

    def labels():
        return all(
            task_3._stored(atoms, "energy").shape == ()
            and task_3._stored(atoms, "forces").shape == (2, 3)
            for atoms in dataset()
        )

    def settings():
        value = e.settings
        model = str(value["teacher_model"]).split("/")[-1]
        digest = value.get("teacher_sha256")
        return (
            model == "teacher.model"
            and value["dispersion"] is True
            and value["energy_unit"] == "eV"
            and value["force_unit"] in {"eV/Angstrom", "eV/Å", "eV/A"}
            and (
                digest is None
                or (
                    isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
                )
            )
        )

    r.check("dimer_separation_dataset", 40, lambda: bool(dataset()))
    r.check("teacher_energy_and_force_labels", 35, labels)
    r.check("teacher_identity_dispersion_and_units", 15, settings)
    r.unverified(
        "execution_provenance",
        "Stored labels and settings cannot prove that the recorded teacher checkpoint produced them.",
    )


def _task_4(e: Evidence, r: Rubric) -> None:
    task_4.evaluate(e, r, level=1)


def _task_5(e: Evidence, r: Rubric) -> None:
    reference_cell = 5.43 / 2 * (np.ones((3, 3)) - np.eye(3))

    @lru_cache(None)
    def rows():
        records = task_5._records(e, "strain_calculations")
        frames = e.trajectory("strained_structures")
        if len(records) != 5 or len(frames) != 5:
            raise EvidenceError("Level 1 requires the five isotropic strain cases")
        indices = [record["structure_index"] for record in records]
        if sorted(indices) != list(range(5)) or any(
            type(index) is not int for index in indices
        ):
            raise EvidenceError("Each strain record must map to one saved structure")
        strains = np.asarray([record["isotropic_strain"] for record in records])
        if not np.allclose(
            np.sort(strains), np.arange(-2, 3) / 100, rtol=0, atol=1e-10
        ):
            raise EvidenceError("The isotropic strain grid is incomplete")
        if any(
            abs(float(record.get("uniaxial_strain", 0))) > 1e-12 for record in records
        ):
            raise EvidenceError("Level 1 must not add a Cartesian z strain")
        return records, frames

    def reference():
        frames = e.trajectory("reference_structure")
        return len(frames) == 1 and task_5._geometry(frames[0], reference_cell)

    def geometries():
        records, frames = rows()
        return all(
            task_5._geometry(
                frames[record["structure_index"]],
                reference_cell * (1 + float(record["isotropic_strain"])),
            )
            for record in records
        )

    @lru_cache(None)
    def masses():
        values = finite_array(e.settings["masses_amu"], shape=(2,))
        if not np.all(np.abs(values - 28.0855) <= 0.001):
            raise EvidenceError("Natural isotope-averaged silicon masses are required")
        for atoms in rows()[1] + e.trajectory("reference_structure"):
            if not close(atoms.get_masses(), values, rtol=0, atol=1e-5):
                raise EvidenceError("Saved and declared masses differ")
        return values

    @lru_cache(None)
    def modes():
        if not reference() or not geometries():
            raise EvidenceError("Valid unrelaxed strained structures are prerequisites")
        return {
            record["id"]: task_5._modes(e, record, masses()) for record in rows()[0]
        }

    def settings():
        value = e.settings
        return (
            "mace-mp-0" in str(value["model"]).lower().replace("_", "-")
            and value["energy_unit"] == "eV"
            and value["force_unit"] == "eV/Angstrom"
            and value["force_constant_unit"] == "eV/Angstrom^2"
        )

    def force_constants():
        return all(
            result_close(
                task_5._reconstruct(e, record)[0],
                task_5._array(e, record["force_constants_eV_A2"], (6, 6)),
                atol=1e-7,
            )
            for record in rows()[0]
        )

    def reported_modes():
        return all(
            task_5._saved_modes_match(e, record, modes()[record["id"]])
            and result_close(
                record["highest_positive_energy_eV"], modes()[record["id"]][2]
            )
            for record in rows()[0]
        )

    r.check("reference_primitive_structure", 10, reference)
    r.check("five_isotropic_strains_without_relaxation", 20, geometries)
    r.check("natural_silicon_masses", 10, lambda: len(masses()) == 2)
    r.check("recorded_model_units_and_conventions", 10, settings)
    r.check("force_constants_from_raw_evidence", 25, _supported(force_constants))
    r.check("all_modes_and_highest_positive_energy", 15, _supported(reported_modes))
    r.unverified(
        "execution_provenance",
        "Saved arrays establish consistency, not the executed model or absence of an intermediate relaxation.",
    )


def _task_6(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def stage():
        frames = e.trajectory(
            "equilibration_trajectory", "nvt_trajectory", "equilibration"
        )
        if len(frames) < 2:
            raise EvidenceError(
                "Save at least the initial and final equilibrated states"
            )
        for atoms in frames:
            if (
                len(atoms) != 64
                or set(atoms.get_chemical_symbols()) != {"Al"}
                or not np.all(atoms.pbc)
            ):
                raise EvidenceError("Expected periodic Al64 states")
            finite_array(atoms.positions, shape=(64, 3))
            finite_array(atoms.cell.array, shape=(3, 3))
            finite_array(atoms.arrays["momenta"], shape=(64, 3))
            if not np.allclose(atoms.get_masses(), 26.9815385, atol=1e-3, rtol=0):
                raise EvidenceError("Unexpected aluminum masses")
        times = finite_array([_time_fs(atoms) for atoms in frames])
        if abs(times[0]) > 1e-6 or np.any(np.diff(times) <= 0):
            raise EvidenceError(
                "Equilibration states must have increasing elapsed time"
            )
        return frames, times

    def geometry():
        atoms = stage()[0][0]
        gram = atoms.cell.array @ atoms.cell.array.T
        expected = np.full((3, 3), (2 * 4.05) ** 2)
        np.fill_diagonal(expected, 2 * (2 * 4.05) ** 2)
        distances = np.sort(atoms.get_all_distances(mic=True), axis=1)
        return close(np.abs(gram), expected, rtol=0, atol=1e-4) and close(
            distances[:, 1:13],
            np.full((64, 12), 4.05 / np.sqrt(2)),
            rtol=0,
            atol=1e-4,
        )

    def fixed_cell():
        frames = stage()[0]
        return all(
            close(atoms.cell.array, frames[0].cell.array, rtol=0, atol=1e-5)
            for atoms in frames
        )

    def settings():
        config = e.settings["md"]
        first = stage()[0][0]
        dof = config["temperature_dof"]
        temperature = _kinetic_temperature(first, dof)
        return (
            dof in (189, 192)
            and config["target_temperature_K"] == 300
            and config["initial_temperature_K"] == 300
            and type(config["random_seed"]) is int
            and config["remove_com_once"] is True
            and bool(config["thermostat"])
            and bool(config["equilibration_integrator"])
            and 0 < float(config["timestep_fs"]) <= np.min(np.diff(stage()[1]))
            and "mace-mp-0" in str(e.settings["model"]).lower().replace("_", "-")
            and isinstance(e.settings["model_settings"], dict)
            and close(first.get_momenta().sum(axis=0), np.zeros(3), atol=1e-7)
            and abs(temperature - 300) <= 6 * 300 * np.sqrt(2 / 189)
        )

    @lru_cache(None)
    def temperatures():
        dof = e.settings["md"]["temperature_dof"]
        return np.asarray([_kinetic_temperature(atoms, dof) for atoms in stage()[0]])

    def trace():
        table = e.table("thermal_trace", "thermal_log", "md_log")
        if "stage" in table:
            table = table[
                table.stage.astype(str).str.lower().isin({"equilibration", "nvt", "eq"})
            ]
        times = finite_array(table["time_fs"])
        if np.any(np.diff(times) <= 0):
            return False
        indices = np.searchsorted(times, stage()[1])
        if np.any(indices >= len(times)) or not close(
            times[indices], stage()[1], rtol=0, atol=1e-6
        ):
            return False
        expected = temperatures()
        if not close(
            finite_array(table["temperature_K"])[indices],
            expected,
            rtol=1e-4,
            atol=1e-7,
        ):
            return False
        return scientific_screen(
            abs(expected.mean() - 300)
            <= 6 * 300 * np.sqrt(2 / e.settings["md"]["temperature_dof"]),
            "Saved equilibration temperatures are inconclusive for a near-300 K state",
        )

    r.check("initial_fcc_geometry", 20, geometry)
    r.check("fixed_cell_sampled_trajectory", 15, fixed_cell)
    r.check("initialization_and_equilibration_settings", 20, settings)
    r.check("thermal_trace_from_saved_momenta", 20, trace)
    r.check(
        "restartable_final_state",
        15,
        lambda: np.isfinite(stage()[0][-1].get_momenta()).all()
        and stage()[0][-1].get_volume() > 0,
    )
    r.unverified(
        "execution_provenance",
        "Saved states cannot prove model execution, random velocity generation, or absence of unrecorded rescaling.",
    )


def _task_7(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def record():
        value = e.json("stages", "stage_map")
        records = value["stages"] if isinstance(value, dict) else value
        if not isinstance(records, list) or len(records) != 1:
            raise EvidenceError("Level 1 requires one identifiable 300 K NPT stage")
        item = records[0]
        if item["ensemble"] != "NPT" or float(item["target_temperature_K"]) != 300:
            raise EvidenceError("The saved stage must be 300 K NPT")
        return item

    @lru_cache(None)
    def trajectory():
        item = record()
        frames = e.trajectory(f"trajectories.{item['id']}")
        if len(frames) < 2:
            raise EvidenceError("Save initial and final NPT states")
        times = finite_array([_time_fs(atoms) for atoms in frames])
        if np.any(np.diff(times) <= 0):
            raise EvidenceError("NPT samples must be chronological")
        for atoms in frames:
            if (
                len(atoms) != 108
                or set(atoms.get_chemical_symbols()) != {"Al"}
                or not np.all(atoms.pbc)
            ):
                raise EvidenceError("Expected periodic Al108 states")
            finite_array(atoms.positions, shape=(108, 3))
            finite_array(atoms.arrays["momenta"], shape=(108, 3))
            if atoms.info.get("stage") not in (None, item["id"]):
                raise EvidenceError("Trajectory stage identity is inconsistent")
        interval = item["production"]
        if (
            len(interval) != 2
            or any(type(index) is not int for index in interval)
            or not 0 <= interval[0] < interval[1] <= len(frames)
            or interval[1] - interval[0] < 2
        ):
            raise EvidenceError("Production interval must select at least two frames")
        return frames, times

    def geometry():
        first = trajectory()[0][0]
        reference = e.trajectory("initial_structure")
        if len(reference) != 1 or not task_7._same_geometry(first, reference[0]):
            return False
        lattice = (first.get_volume() / 27) ** (1 / 3)
        distances = np.sort(first.get_all_distances(mic=True), axis=1)
        return close(
            distances[:, 1:13],
            np.full((108, 12), lattice / np.sqrt(2)),
            rtol=0,
            atol=1e-4,
        )

    def settings():
        value = e.settings
        return (
            "mace-mp-0" in str(value["model"]).lower().replace("_", "-")
            and bool(value["input_structure"])
            and value["temperature_dof"] in (321, 324)
            and type(value["random_seed"]) is int
            and value["velocity_initializations"] == 1
            and value["remove_com"] is True
            and float(value["timestep_fs"]) > 0
            and bool(value["thermostat"])
            and bool(value["barostat"])
            and abs(float(value["target_pressure_bar"]) - 1.01325) <= 1e-6
            and value["isotropic"] is True
            and close(
                trajectory()[0][0].get_momenta().sum(axis=0), np.zeros(3), atol=1e-7
            )
        )

    def isotropic_cells():
        frames = trajectory()[0]
        first = frames[0].cell.array
        return all(
            close(
                atoms.cell.array,
                first * (atoms.get_volume() / frames[0].get_volume()) ** (1 / 3),
                rtol=0,
                atol=1e-5,
            )
            for atoms in frames
        )

    @lru_cache(None)
    def production_values():
        frames = trajectory()[0]
        start, end = record()["production"]
        selected = frames[start:end]
        dof = e.settings["temperature_dof"]
        temperature = np.asarray(
            [_kinetic_temperature(atoms, dof) for atoms in selected]
        )
        volume = np.asarray([atoms.get_volume() for atoms in selected])
        return selected, temperature, volume

    def trace_and_results():
        table = e.table("thermal_trace", "raw_trace")
        table = table[table.stage == record()["id"]]
        if (
            len(table) < len(trajectory()[0])
            or not np.isfinite(
                table[
                    ["time_fs", "temperature_K", "pressure_GPa", "volume_A3"]
                ].to_numpy(dtype=float)
            ).all()
        ):
            return False
        selected, temperature, volume = production_values()
        selected_times = np.asarray([_time_fs(atoms) for atoms in selected])
        trace_times = table.time_fs.to_numpy(dtype=float)
        matched = []
        for time in selected_times:
            candidates = np.flatnonzero(
                np.isclose(trace_times, time, rtol=0, atol=1e-6)
            )
            if len(candidates) != 1:
                return False
            matched.append(candidates[0])
        rows = table.iloc[matched]
        if not (
            close(rows.temperature_K, temperature, rtol=1e-4, atol=1e-7)
            and close(rows.volume_A3, volume, rtol=1e-4, atol=1e-7)
        ):
            return False
        result = e.result("stages")[record()["id"]]
        return (
            result_close(result["temperature_K"], temperature.mean(), atol=1e-7)
            and result_close(result["volume_A3"], volume.mean(), atol=1e-7)
            and result_close(
                result["pressure_GPa"], rows.pressure_GPa.mean(), atol=1e-7
            )
        )

    r.check("initial_fcc_geometry", 15, geometry)
    r.check("recorded_npt_model_and_initialization", 15, settings)
    r.check("continuous_isotropic_npt_state", 20, isotropic_cells)
    r.check("measured_thermal_trace_and_means", 30, trace_and_results)
    r.check("restartable_final_state", 10, lambda: trajectory()[0][-1].get_volume() > 0)
    r.unverified(
        "execution_provenance",
        "Saved states and traces cannot prove checkpoint execution or absence of an undisclosed velocity reset.",
    )


def _task_8(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def dataset():
        data = e.json("regression_data", "datasets", "dataset_index")["train"]
        frames = e.trajectory("train_structures", "training_structures")
        if len(frames) != 100:
            raise EvidenceError("Expected 100 training structures")
        row_ids = data["row_ids"]
        if (
            not isinstance(row_ids, list)
            or len(row_ids) != 100
            or len(set(row_ids)) != 100
            or any(not isinstance(value, str) or not value for value in row_ids)
        ):
            raise EvidenceError("Training row IDs must be 100 unique strings")
        frame_ids = [atoms.info["row_id"] for atoms in frames]
        if set(frame_ids) != set(row_ids):
            raise EvidenceError("Structure and label row IDs differ")
        lookup = dict(zip(frame_ids, frames, strict=True))
        ordered = [lookup[value] for value in row_ids]
        energy = finite_array(data["energy_eV"], shape=(100,))
        sigma = finite_array(data["sigma_A"], shape=(100,))
        indices = finite_array(data["generation_index"], shape=(100,))
        if not np.array_equal(np.sort(indices), np.arange(100)):
            raise EvidenceError("Generation indices must be a permutation of 0..99")
        if not close(sigma, 0.01 + 0.001 * indices):
            raise EvidenceError("The displacement schedule is incorrect")
        if "strain" in data and not close(data["strain"], np.zeros(100)):
            raise EvidenceError("Level 1 training cells must be unstrained")
        return data, ordered, energy, sigma, indices

    @lru_cache(None)
    def displacements():
        _, frames, energy, sigma, _ = dataset()
        ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        result = []
        for atoms, value, scale in zip(frames, energy, sigma, strict=True):
            if len(atoms) != 64 or set(atoms.numbers) != {14} or not np.all(atoms.pbc):
                raise EvidenceError("Expected periodic Si64 training structures")
            if not close(atoms.cell.array, ideal.cell.array, rtol=0, atol=1e-5):
                raise EvidenceError("Training cells must preserve the ideal cell")
            stored = atoms.info.get(
                "energy",
                getattr(getattr(atoms, "calc", None), "results", {}).get("energy"),
            )
            if stored is None or not close(stored, value):
                raise EvidenceError("Stored structure energies and labels differ")
            delta = atoms.positions[:, None, :] - ideal.positions[None, :, :]
            length = ideal.cell.lengths()
            delta -= np.rint(delta / length) * length
            mapping = np.argmin(np.linalg.norm(delta, axis=2), axis=1)
            if len(set(mapping)) != 64:
                raise EvidenceError("Distortions do not preserve distinct ideal sites")
            result.append(delta[np.arange(64), mapping] / scale)
        return np.asarray(result)

    def random_distribution():
        values = displacements()
        unique = {
            tuple(np.round(atoms.get_scaled_positions(wrap=True).ravel(), 8))
            for atoms in dataset()[1]
        }
        return scientific_screen(
            len(unique) == 100
            and np.all(np.abs(values.mean(axis=(0, 1))) <= 0.08)
            and 0.85 <= np.sqrt(np.mean(values**2)) <= 1.15,
            "Saved displacement statistics are inconclusive for independent Gaussian draws",
        )

    def settings():
        value = e.settings
        generation = value["generation"]["train"]
        digest = value.get("teacher_sha256")
        return (
            str(value["teacher_model"]).split("/")[-1] == "teacher.model"
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
            and value["energy_unit"] == "eV"
            and value["length_unit"] in {"Angstrom", "A", "Å"}
            and type(generation["seed"]) is int
            and all(
                isinstance(generation[key], str) and generation[key].strip()
                for key in ("library", "method")
            )
        )

    r.check(
        "training_structure_geometry", 25, lambda: displacements().shape == (100, 64, 3)
    )
    r.check("distortion_schedule_and_row_identity", 20, lambda: bool(dataset()[0]))
    r.check(
        "aligned_teacher_energy_labels", 20, lambda: np.isfinite(dataset()[2]).all()
    )
    r.check("recorded_rng_teacher_digest_and_units", 15, settings)
    r.check("independent_gaussian_distortions", 10, random_distribution)
    r.unverified(
        "execution_and_teacher_provenance",
        "Saved labels, seeds, and a digest cannot prove teacher or RNG execution or absence of relaxation.",
    )


def _task_9(e: Evidence, r: Rubric) -> None:
    run = "main"

    @lru_cache(None)
    def frames(stage="trajectory"):
        result = e.trajectory(f"runs.{run}.{stage}")
        expected = 200 if stage == "trajectory" else 2 if stage == "boundary" else None
        if (expected is not None and len(result) != expected) or (
            stage == "equilibration" and len(result) < 2
        ):
            raise EvidenceError(
                "Expected 200 production, two boundary, and at least two equilibration states"
            )
        cell = finite_array(result[0].cell.array, shape=(3, 3))
        for atoms in result:
            if (
                len(atoms) != 32
                or set(atoms.get_chemical_symbols()) != {"Cu"}
                or not np.all(atoms.pbc)
            ):
                raise EvidenceError("Expected periodic Cu32 states")
            if not close(atoms.cell.array, cell, rtol=1e-6, atol=1e-7):
                raise EvidenceError("The fixed cell changed")
            finite_array(atoms.arrays["momenta"], shape=(32, 3))
            _stored_energy(atoms)
            if stage == "trajectory":
                task_3._stored(atoms, "forces")
        times = finite_array([_time_fs(atoms) for atoms in result])
        if stage != "boundary" and np.any(np.diff(times) <= 0):
            raise EvidenceError("Saved states must be chronological")
        return result

    def geometry():
        supplied = e.trajectory("input_structure")
        return (
            len(supplied) == 1
            and len(supplied[0]) == 32
            and set(supplied[0].get_chemical_symbols()) == {"Cu"}
            and np.all(supplied[0].pbc)
            and task_9._same(frames("equilibration")[0], supplied[0])
            and close(frames()[0].cell.array, supplied[0].cell.array)
        )

    def settings():
        value = e.settings["runs"][run]
        initial = frames("equilibration")[0]
        return (
            value["temperature_dof"] in (93, 96)
            and value["target_temperature_K"] == 800
            and type(value["seed"]) is int
            and value["remove_com"] is True
            and float(value["timestep_fs"]) > 0
            and bool(value["integrator"])
            and bool(value["thermostat"])
            and "mace-mp-0" in str(e.settings["teacher"]).lower().replace("_", "-")
            and close(initial.get_momenta().sum(axis=0), np.zeros(3), atol=1e-7)
            and abs(_kinetic_temperature(initial, value["temperature_dof"]) - 800)
            <= 6 * 800 * np.sqrt(2 / value["temperature_dof"])
        )

    def continuity():
        end, start = frames("boundary")
        equilibration, production = frames("equilibration"), frames()
        return (
            task_9._state_same(end, start)
            and task_9._state_same(equilibration[-1], end)
            and close(start.cell.array, production[0].cell.array)
            and _time_fs(production[0]) >= _time_fs(start)
        )

    @lru_cache(None)
    def energies_times():
        production = frames()
        return (
            np.asarray([_stored_energy(atoms) for atoms in production]),
            np.asarray([_time_fs(atoms) for atoms in production]),
        )

    def trace():
        table = e.table(f"runs.{run}.log")
        times = finite_array(table["time_fs"])
        if np.any(np.diff(times) <= 0):
            return False
        production_times = energies_times()[1]
        indices = np.searchsorted(times, production_times)
        if np.any(indices >= len(times)) or not close(
            times[indices], production_times, rtol=0, atol=1e-6
        ):
            return False
        dof = e.settings["runs"][run]["temperature_dof"]
        temperature = np.asarray(
            [_kinetic_temperature(atoms, dof) for atoms in frames()]
        )
        if not (
            close(
                table.temperature_K.to_numpy()[indices],
                temperature,
                rtol=1e-4,
                atol=1e-7,
            )
            and close(
                table.potential_energy_eV.to_numpy()[indices],
                energies_times()[0],
                atol=1e-7,
            )
        ):
            return False
        claimed = e.results["runs"][run]
        return result_close(
            claimed["production_temperature_mean_K"], temperature.mean(), atol=1e-7
        )

    @lru_cache(None)
    def correlation():
        doc = e.json("correlation")
        if doc["estimator"] != "acf":
            raise UnsupportedEvidence(
                "Only a saved autocorrelation estimator has an automatic adapter"
            )
        energy, times = energies_times()
        delta = np.diff(times)
        if not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-8):
            raise UnsupportedEvidence(
                "Nonuniform correlation sampling needs independent review"
            )
        lags = finite_array(doc["lag_indices"], ndim=1).astype(int)
        if (
            not np.array_equal(lags, doc["lag_indices"])
            or not len(lags)
            or np.any(lags < 0)
            or np.any(np.diff(lags) <= 0)
            or lags[-1] >= len(energy)
        ):
            raise EvidenceError("Invalid correlation lags")
        centered = energy - energy.mean()
        denominator = np.dot(centered, centered)
        if denominator <= 0:
            raise EvidenceError("Constant energy has no normalized autocorrelation")
        values = np.asarray(
            [
                np.dot(centered[: len(centered) - lag], centered[lag:]) / denominator
                for lag in lags
            ]
        )
        if doc["normalization"] == "unbiased":
            values *= len(energy) / (len(energy) - lags)
        elif doc["normalization"] != "biased":
            raise UnsupportedEvidence("Unsupported autocorrelation normalization")
        factor = {"fs": 1, "ps": 1000}.get(doc["time_unit"])
        if factor is None:
            raise EvidenceError("Correlation time unit must be fs or ps")
        if not (
            result_close(doc["values"], values, atol=1e-7)
            and result_close(doc["lag_times"], lags * delta[0] / factor, atol=1e-7)
        ):
            return False
        item = doc["characteristic"]
        item_factor = {"fs": 1, "ps": 1000}.get(item["time_unit"])
        if item_factor is None:
            return False
        if item["method"] != "first_crossing" or not np.array_equal(
            lags, np.arange(len(lags))
        ):
            raise UnsupportedEvidence(
                "The characteristic-time estimator needs independent review"
            )
        threshold = float(item["threshold"])
        if not 0 < threshold < 1:
            return False
        indices = np.flatnonzero(values <= threshold)
        value, bound = (
            (lags[indices[0]] * delta[0], "estimate")
            if len(indices)
            else (lags[-1] * delta[0], "lower")
        )
        return item["bound"] == bound and result_close(
            item["value"], value / item_factor, atol=1e-7
        )

    r.check("fixed_cell_cu32_chronological_frames", 20, geometry)
    r.check("recorded_initialization_and_teacher", 15, settings)
    r.check("equilibration_to_production_continuity", 15, continuity)
    r.check("complete_energy_force_dataset", 15, lambda: len(frames()) == 200)
    r.check("thermal_log_and_production_temperature", 10, trace)
    r.check("temporal_correlation_and_characteristic_time", 15, _supported(correlation))
    r.unverified(
        "execution_and_information_access_provenance",
        "Artifacts establish numerical consistency, not actual teacher execution or absence of an unrecorded reset.",
    )


def _task_10(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def stage():
        records = e.settings["stages"]
        if not isinstance(records, list) or len(records) != 1:
            raise EvidenceError("Level 1 requires exactly one 300 K stage")
        item = records[0]
        if (
            float(item["target_temperature_K"]) != 300
            or not isinstance(item["id"], str)
            or not item["id"]
        ):
            raise EvidenceError("The stage must have a distinct ID and a 300 K target")
        return item

    @lru_cache(None)
    def initial():
        frames = e.trajectory("initial_state", "initial")
        if len(frames) != 1:
            raise EvidenceError("Save exactly one initial state")
        return frames[0]

    @lru_cache(None)
    def boundaries():
        frames = e.trajectory("boundary_states", "stage_boundaries")
        if len(frames) != 2:
            raise EvidenceError("Save the start and end state of the initial stage")
        return frames

    @lru_cache(None)
    def production():
        frames = [
            atoms
            for atoms in e.trajectory("production_trajectory", "trajectory")
            if atoms.info.get("stage") == stage()["id"]
        ]
        if len(frames) < 2:
            raise EvidenceError("Save at least two production frames")
        times = finite_array([_time_fs(atoms) for atoms in frames])
        if np.any(np.diff(times) <= 0):
            raise EvidenceError("Production times must increase")
        return frames, times

    def valid_frame(atoms):
        return (
            len(atoms) == 108
            and np.all(atoms.numbers == 13)
            and bool(atoms.pbc.all())
            and close(atoms.get_masses(), np.full(108, 26.9815385), atol=1e-3)
            and finite_array(atoms.positions, shape=(108, 3)).shape == (108, 3)
            and finite_array(atoms.get_momenta(), shape=(108, 3)).shape == (108, 3)
        )

    def geometry():
        atoms = initial()
        if not valid_frame(atoms):
            return False
        length = 3 * 4.05
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        shells = np.sort(distances, axis=1)
        dof = e.settings["md"]["temperature_dof"]
        return (
            close(
                atoms.cell.array @ atoms.cell.array.T, np.eye(3) * length**2, atol=1e-4
            )
            and np.allclose(shells[:, :12], 4.05 / np.sqrt(2), atol=1e-4, rtol=0)
            and np.allclose(shells[:, 12:18], 4.05, atol=1e-4, rtol=0)
            and abs(task_10._thermal(atoms, dof)[0][0] - 300)
            <= 6 * 300 * np.sqrt(2 / dof)
        )

    def settings():
        value = e.settings["md"]
        return (
            value["temperature_dof"] in (321, 324)
            and bool(value["thermostat"])
            and float(value["timestep_fs"]) > 0
            and type(value["random_seed"]) is int
            and value["initial_temperature_K"] == 300
            and value["velocity_initializations"] == 1
            and value["manual_velocity_resets"] == 0
            and str(value["ensemble"]).upper() == "NVT"
            and "mace-mp-0" in str(e.settings["model"]).lower().replace("_", "-")
            and isinstance(e.settings["model_settings"], dict)
        )

    def fixed_cell():
        frames = [initial(), *boundaries(), *production()[0]]
        return all(
            valid_frame(atoms)
            and close(atoms.cell.array, initial().cell.array, rtol=0, atol=1e-6)
            and np.array_equal(atoms.numbers, initial().numbers)
            for atoms in frames
        )

    def same_state(left, right):
        difference = (left.positions - right.positions) @ np.linalg.inv(left.cell.array)
        difference -= np.rint(difference)
        return (
            close(difference @ left.cell.array, np.zeros((108, 3)), rtol=0, atol=1e-6)
            and close(left.get_momenta(), right.get_momenta(), rtol=0, atol=1e-7)
            and close(left.cell.array, right.cell.array, rtol=0, atol=1e-6)
            and close(_time_fs(left), _time_fs(right), rtol=0, atol=1e-7)
        )

    def continuity():
        item = stage()
        start, end = boundaries()
        frames, times = production()
        return (
            same_state(initial(), start)
            and close(
                [_time_fs(start), _time_fs(end)],
                [item["start_time_fs"], item["end_time_fs"]],
            )
            and close(
                times[[0, -1]],
                [item["production_start_time_fs"], item["production_end_time_fs"]],
            )
            and item["start_time_fs"]
            < item["production_start_time_fs"]
            < item["production_end_time_fs"]
            <= item["end_time_fs"]
        )

    @lru_cache(None)
    def values():
        dof = e.settings["md"]["temperature_dof"]
        return np.asarray(
            [task_10._thermal(atoms, dof)[0] for atoms in production()[0]]
        )

    def trace_and_results():
        table = e.table("thermal_trace", "thermal_traces")
        rows = table[table.stage == stage()["id"]]
        times = finite_array(rows["time_fs"])
        item = stage()
        if np.any(np.diff(times) <= 0) or not close(
            times[[0, -1]],
            [item["start_time_fs"], item["end_time_fs"]],
            rtol=0,
            atol=1e-6,
        ):
            return False
        indices = np.searchsorted(times, production()[1])
        if np.any(indices >= len(times)) or not close(
            times[indices], production()[1], rtol=0, atol=1e-6
        ):
            return False
        expected = values()
        columns = [
            "temperature_K",
            "temperature_x_K",
            "temperature_y_K",
            "temperature_z_K",
        ]
        if not close(
            rows[columns].to_numpy(dtype=float)[indices],
            expected[:, :4],
            rtol=1e-4,
            atol=1e-7,
        ):
            return False
        for atoms, index in zip(boundaries(), (0, -1), strict=True):
            thermal = task_10._thermal(atoms, e.settings["md"]["temperature_dof"])[0]
            if not close(
                rows.iloc[index][columns].to_numpy(dtype=float),
                thermal[:4],
                rtol=1e-4,
                atol=1e-7,
            ):
                return False
        claim = e.results["stages"][stage()["id"]]
        return (
            result_close(claim["temperature_mean_K"], expected[:, 0].mean(), atol=1e-7)
            and result_close(
                claim["directional_temperature_mean_K"],
                expected[:, 1:4].mean(axis=0),
                atol=1e-7,
            )
            and result_close(
                claim["energy_mean_eV_per_atom"], expected[:, 4].mean(), atol=1e-7
            )
        )

    r.check("initial_fcc_and_temperature", 15, geometry)
    r.check("recorded_nvt_model_and_initialization", 10, settings)
    r.check("fixed_cell_and_ordered_atoms", 15, fixed_cell)
    r.check("initial_stage_timing_and_continuity", 20, continuity)
    r.check("thermal_trace_and_reported_means", 20, trace_and_results)
    r.check(
        "production_temperature_sanity",
        10,
        lambda: scientific_screen(
            abs(values()[:, 0].mean() - 300)
            <= 6 * 300 * np.sqrt(2 / e.settings["md"]["temperature_dof"]),
            "Saved production temperature is inconclusive for the 300 K target",
        ),
    )
    r.unverified(
        "execution_provenance",
        "Saved artifacts cannot establish actual MACE execution or absence of an undisclosed velocity reset.",
    )


_EVALUATORS = {
    1: _task_1,
    2: _task_2,
    3: _task_3,
    4: _task_4,
    5: _task_5,
    6: _task_6,
    7: _task_7,
    8: _task_8,
    9: _task_9,
    10: _task_10,
}


def evaluate(evidence: Evidence, rubric: Rubric, task_number: int) -> None:
    """Evaluate only the preparation work requested by a Level 1 task."""
    _EVALUATORS[task_number](evidence, rubric)
