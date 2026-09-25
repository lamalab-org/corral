"""Artifact-only checks for dimer fitting and bulk silver MD.

Reading a stored calculator result is deliberately different from evaluating a
calculator. This module never loads model objects or calls an energy calculator.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import struct
from functools import lru_cache
from typing import TYPE_CHECKING
from zipfile import ZipFile, is_zipfile

import numpy as np
from ase import units
from ase.build import bulk

from .common import EvidenceError, UnsupportedEvidence, result_close

if TYPE_CHECKING:
    from .common import Evidence, Rubric


def _finite(value, shape=None):
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Non-finite numerical evidence")
    return array


def _stored(atoms, name):
    """Access serialized results without triggering any calculator evaluation."""
    if name == "forces" and "forces" in atoms.arrays:
        return _finite(atoms.arrays["forces"], (len(atoms), 3))
    for key in (name, "potential_energy" if name == "energy" else name):
        if key in atoms.info:
            return _finite(atoms.info[key], () if name == "energy" else (len(atoms), 3))
    cached = getattr(getattr(atoms, "calc", None), "results", {})
    if name not in cached:
        raise ValueError(f"Missing stored {name}; evaluation is not permitted")
    return _finite(cached[name], () if name == "energy" else (len(atoms), 3))


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_container(path):
    """Check bounded container metadata; never read pickle or tensor payloads."""
    if is_zipfile(path):
        with ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            return (
                any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
                and any(
                    name.endswith("/version") or name == "version" for name in names
                )
                and any(
                    re.search(r"(^|/)data/[^/]+$", item.filename) and item.file_size > 0
                    for item in entries
                )
            )
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            return False
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, path.stat().st_size - 8):
            return False
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        return False
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    payload_size = path.stat().st_size - 8 - length
    sizes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    spans = []
    for tensor in tensors:
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        if any(type(v) is not int or v < 0 for v in shape):
            return False
        start, end = tensor["data_offsets"]
        if not (
            type(start) is int
            and type(end) is int
            and 0 <= start <= end <= payload_size
        ):
            return False
        if dtype not in sizes:
            raise UnsupportedEvidence(
                "Checkpoint tensor dtype requires independent review"
            )
        if end - start != int(np.prod(shape, dtype=object)) * sizes[dtype]:
            return False
        spans.append((start, end))
    spans.sort()
    return (
        bool(spans)
        and spans[0][0] == 0
        and spans[-1][1] == payload_size
        and all(left[1] == right[0] for left, right in itertools.pairwise(spans))
    )


def _close(actual, claimed, atol=1e-7):
    actual = _finite(actual)
    claimed = _finite(claimed, actual.shape)
    return bool(np.allclose(actual, claimed, rtol=1e-4, atol=atol))


def evaluate(e: Evidence, r: Rubric) -> None:
    """Award 90 task-specific points; the shared rubric supplies the other 10."""
    expected_separations = np.round(np.arange(0.6, 5.01, 0.1), 8)

    @lru_cache(None)
    def dataset():
        frames = e.trajectory("dataset", "labeled_dataset", "training_dataset")
        if len(frames) != 45 or any(
            len(a) != 2 or a.get_chemical_symbols() != ["Ag", "Ag"] or np.any(a.pbc)
            for a in frames
        ):
            raise ValueError("Expected 45 isolated nonperiodic Ag2 structures")
        distances = np.array(
            [
                np.linalg.norm(_finite(a.positions, (2, 3))[1] - a.positions[0])
                for a in frames
            ]
        )
        order = np.argsort(distances)
        if not np.allclose(distances[order], expected_separations, atol=1e-5, rtol=0):
            raise ValueError("Dimer separation grid is not 0.6..5.0 Angstrom")
        return [frames[index] for index in order]

    @lru_cache(None)
    def labels():
        frames = dataset()
        return (
            np.array([_stored(a, "energy") for a in frames]),
            np.array([_stored(a, "forces") for a in frames]),
        )

    def identities():
        settings = e.settings
        return (
            str(settings["teacher_model"]).split("/")[-1] == "teacher.model"
            and str(settings["student_model"]).split("/")[-1] == "student.model"
            and all(
                re.fullmatch(r"[0-9a-fA-F]{64}", settings[key])
                for key in ("teacher_sha256", "student_sha256")
                if key in settings
            )
            and settings["dispersion"] is True
            and settings["energy_unit"] == "eV"
            and settings["force_unit"] in {"eV/Angstrom", "eV/Å", "eV/A"}
        )

    r.check(
        "dataset_geometry",
        8,
        lambda: bool(dataset()),
        "45 isolated Ag2 structures on the required separation grid",
    )
    r.check(
        "dataset_energy_force_labels",
        6,
        lambda: labels()[1].shape == (45, 2, 3),
        "Finite total energies and complete atomic forces",
    )
    r.check(
        "reference_identities_and_units",
        6,
        identities,
        "Recorded input identities, dispersion setting, and label units",
    )

    @lru_cache(None)
    def prediction_errors(which):
        target_energy, target_forces = labels()
        data = e.json(f"predictions_{which}", f"{which}_predictions")
        distances = _finite(data["separation_A"], (45,))
        order = np.argsort(distances)
        if not np.allclose(distances[order], expected_separations, rtol=0, atol=1e-5):
            raise ValueError("Prediction separation IDs do not match labeled data")
        energy = _finite(data["energy_eV"], (45,))[order]
        forces = _finite(data["forces_eV_A"], (45, 2, 3))[order]
        return (energy - target_energy) / 2.0, forces - target_forces

    def predictions_valid():
        return all(
            prediction_errors(which)[1].shape == (45, 2, 3)
            for which in ("before", "after")
        )

    def aggregate_errors():
        for which in ("before", "after"):
            energies, forces = prediction_errors(which)
            claimed = e.results["fitting"][which]
            for candidates in (
                {
                    "energy_rmse_per_atom_eV": np.sqrt(np.mean(energies**2)),
                    "energy_mae_per_atom_eV": np.mean(np.abs(energies)),
                },
                {
                    "force_rmse_eV_A": np.sqrt(np.mean(forces**2)),
                    "force_mae_eV_A": np.mean(np.abs(forces)),
                },
            ):
                names = candidates.keys() & claimed.keys()
                if not names:
                    if claimed.get("metrics"):
                        raise UnsupportedEvidence(
                            "Documented error metric needs independent review"
                        )
                    raise EvidenceError("Missing aggregate energy or force errors")
                if any(
                    not result_close(candidates[k], claimed[k], atol=1e-7)
                    for k in names
                ):
                    return False
        return True

    def separation_errors():
        for which in ("before", "after"):
            energies, forces = prediction_errors(which)
            claimed = e.results["fitting"][which]
            separations = _finite(claimed["separation_A"], (45,))
            order = np.argsort(separations)
            if not _close(expected_separations, separations[order], atol=1e-7):
                return False
            for candidates in (
                {
                    "energy_abs_error_per_atom_eV": np.abs(energies),
                    "energy_error_per_atom_eV": energies,
                },
                {
                    "force_rmse_by_separation_eV_A": np.sqrt(
                        np.mean(forces**2, axis=(1, 2))
                    ),
                    "force_mae_by_separation_eV_A": np.mean(
                        np.abs(forces), axis=(1, 2)
                    ),
                },
            ):
                names = candidates.keys() & claimed.keys()
                if not names:
                    if claimed.get("separation_metrics"):
                        raise UnsupportedEvidence(
                            "Separation-resolved metric needs independent review"
                        )
                    raise EvidenceError(
                        "Missing separation-resolved energy or force errors"
                    )
                if any(
                    not result_close(
                        candidates[k], _finite(claimed[k], (45,))[order], atol=1e-7
                    )
                    for k in names
                ):
                    return False
        return True

    def training_history():
        labels()
        history = e.table("training_history", "training_log")
        if (
            history.empty
            or not bool(e.settings["training"])
            or not bool(e.settings["validation_strategy"])
            or not isinstance(e.settings["training_precision"], str)
            or not e.settings["training_precision"].strip()
        ):
            return False
        if (
            "training_dataset_sha256" in e.settings
            and _hash(e.artifact("dataset", "labeled_dataset", "training_dataset"))
            != e.settings["training_dataset_sha256"].lower()
        ):
            return False
        if not {"step", "loss", "learning_rate"}.issubset(history.columns):
            numeric = history.select_dtypes(include="number").to_numpy()
            if not numeric.size or not np.isfinite(numeric).all():
                raise EvidenceError(
                    "Training history needs finite numerical progress evidence"
                )
            raise UnsupportedEvidence(
                "Documented alternative training history requires review"
            )
        step = _finite(history["step"])
        _finite(history["loss"])
        lr = _finite(history["learning_rate"])
        return (
            len(step) >= 2
            and np.all(np.diff(step) > 0)
            and step[0] >= 0
            and np.all(lr >= 0)
            and np.any(lr > 0)
        )

    def trained_checkpoint():
        path = e.artifact("trained_checkpoint", "checkpoint", "trained_model")
        if path.stat().st_size == 0:
            return False
        if (
            "trained_checkpoint_sha256" in e.settings
            and _hash(path) != e.settings["trained_checkpoint_sha256"].lower()
        ):
            return False
        if e.settings.get("checkpoint_format", "pytorch") not in {
            "pytorch",
            "safetensors",
        }:
            raise UnsupportedEvidence(
                "Retained checkpoint format needs independent loadability review"
            )
        with path.open("rb") as handle:
            prefix = handle.read(2)
        if prefix[:1] == b"\x80" and len(prefix) == 2 and prefix[1] <= 5:
            raise UnsupportedEvidence(
                "Legacy pickle checkpoint needs independent loadability review"
            )
        return _checkpoint_container(path)

    r.check(
        "before_after_predictions",
        5,
        predictions_valid,
        "Complete saved predictions aligned to the labeled structures",
    )
    r.check(
        "aggregate_fitting_errors",
        5,
        aggregate_errors,
        "Recomputed energy-per-atom and force RMSE before and after fitting",
    )
    r.check(
        "separation_resolved_errors",
        5,
        separation_errors,
        "Recomputed separation-dependent fitting diagnostics",
    )
    r.check(
        "training_history_and_dataset_identity",
        5,
        training_history,
        "Finite optimization records and correct dataset digest; records do not prove execution",
    )
    r.check(
        "checkpoint_container_and_digest",
        5,
        trained_checkpoint,
        "Safe container metadata and SHA256 check; no model is deserialized",
    )

    @lru_cache(None)
    def md():
        frames = e.trajectory("md_trajectory", "bulk_trajectory", "trajectory")
        if len(frames) < 2:
            raise ValueError("Initial and final MD evidence is required")
        for frame in frames:
            if (
                len(frame) != 108
                or set(frame.get_chemical_symbols()) != {"Ag"}
                or not np.all(frame.pbc)
                or not np.allclose(
                    frame.cell @ frame.cell.T, np.eye(3) * 12.27**2, atol=1e-4, rtol=0
                )
                or not np.allclose(
                    frame.cell.array, frames[0].cell.array, atol=1e-5, rtol=0
                )
            ):
                raise ValueError(
                    "MD frames must preserve the specified periodic Ag108 cell"
                )
            _finite(frame.positions, (108, 3))
        times = _finite(
            [
                a.info["time_fs"] if "time_fs" in a.info else 1000 * a.info["time_ps"]
                for a in frames
            ]
        )
        if abs(times[0]) > 1e-6 or np.any(np.diff(times) <= 0):
            raise ValueError(
                "Trajectory must start at time zero and be strictly chronological"
            )
        return frames, times

    def initial_geometry():
        frames, _ = md()
        actual = (
            frames[0].get_scaled_positions(wrap=False)
            - frames[0].get_scaled_positions(wrap=False)[0]
        ) * 12.27
        reference = bulk("Ag", "fcc", a=4.09, cubic=True).repeat((3, 3, 3)).positions
        delta = actual[:, None, :] - reference[None, :, :]
        delta -= np.rint(delta / 12.27) * 12.27
        distance = np.linalg.norm(delta, axis=2)
        return bool(
            np.all(np.min(distance, axis=1) < 1e-5)
            and len(np.unique(np.argmin(distance, axis=1))) == 108
        )

    @lru_cache(None)
    def md_values():
        frames, times = md()
        dof = e.settings["md"]["temperature_dof"]
        if dof not in (321, 324):
            raise ValueError("temperature_dof must declare 321 or 324")
        momenta = np.array([_finite(a.arrays["momenta"], (108, 3)) for a in frames])
        kinetic = np.array(
            [
                0.5 * np.sum(p**2 / a.get_masses()[:, None])
                for a, p in zip(frames, momenta, strict=False)
            ]
        )
        energy = np.array([_stored(a, "energy") for a in frames])
        force_max = np.array(
            [np.max(np.linalg.norm(_stored(a, "forces"), axis=1)) for a in frames]
        )
        temperature = 2 * kinetic / (dof * units.kB)
        return times, temperature, energy, force_max, kinetic, momenta

    def initialization():
        _, temperature, _, _, _, momenta = md_values()
        settings = e.settings["md"]
        return (
            float(settings["target_temperature_K"]) == 300
            and float(settings["initial_temperature_K"]) == 300
            and type(settings["random_seed"]) is int
            and settings["remove_com_once"] is True
            and isinstance(settings["thermostat"], str)
            and bool(settings["thermostat"].strip())
            and np.isfinite(float(settings["timestep_fs"]))
            and 0 < float(settings["timestep_fs"]) <= np.min(np.diff(md()[1]))
            and abs(temperature[0] - 300) <= 6 * 300 * np.sqrt(2 / 321)
            and np.allclose(momenta[0].sum(axis=0), 0, atol=1e-7)
        )

    @lru_cache(None)
    def log_values():
        table = e.table("md_log", "md_history")
        times = _finite(table["time_fs"])
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("MD log must have chronological initial and final rows")
        return table, times

    def logged_measurements():
        table, times = log_values()
        sample_times, temperature, energy, force_max, kinetic, _ = md_values()
        indices = np.searchsorted(times, sample_times)
        if np.any(indices >= len(times)) or not _close(sample_times, times[indices]):
            return False
        return all(
            _close(
                expected,
                _finite(table[column])[indices],
                atol=1e-4 if column == "temperature_K" else 1e-7,
            )
            for column, expected in (
                ("temperature_K", temperature),
                ("potential_energy_eV", energy),
                ("force_max_eV_A", force_max),
                ("kinetic_energy_eV", kinetic),
            )
        )

    def complete_log():
        _, times = log_values()
        _, sample_times = md()
        return (
            abs(times[0]) < 1e-6
            and abs(times[-1] - 5000) < 1e-4
            and _close(sample_times[[0, -1]], times[[0, -1]])
        )

    r.check(
        "initial_bulk_geometry",
        8,
        initial_geometry,
        "Periodic 3x3x3 conventional FCC Ag cell, allowing atom permutations and global translations",
    )
    r.check(
        "fixed_cell_and_five_ps_trajectory",
        6,
        lambda: abs(md()[1][-1] - 5000) < 1e-4,
        "Chronological 5 ps trajectory in the required fixed cell",
    )
    r.check(
        "recorded_md_initialization",
        3,
        initialization,
        "Recorded numerical settings and initial zero center-of-mass momentum",
    )
    r.check(
        "md_log_matches_trajectory",
        4,
        logged_measurements,
        "Temperatures, kinetic/potential energies and maximum forces agree with stored frames",
    )
    r.check(
        "md_log_covers_complete_run",
        4,
        complete_log,
        "Logs and sampled trajectory cover the same 0..5000 fs interval",
    )

    def temperature_assessment():
        values = md_values()[1]
        claimed = e.results["bulk"]
        if "temperature_mean_K" not in claimed:
            if claimed.get("temperature_analysis"):
                raise UnsupportedEvidence(
                    "Documented temperature analysis requires review"
                )
            raise EvidenceError("Missing quantitative temperature assessment")
        return all(
            result_close(value, claimed[key], atol=0.0001)
            for key, value in (
                ("temperature_mean_K", np.mean(values)),
                ("temperature_std_K", np.std(values)),
                ("temperature_min_K", np.min(values)),
                ("temperature_max_K", np.max(values)),
            )
            if key in claimed
        )

    def energy_force_assessment():
        _, _, energy, force_max, _, _ = md_values()
        claimed = e.results["bulk"]
        for candidates in (
            {
                "potential_energy_range_eV": np.ptp(energy),
                "potential_energy_std_eV": np.std(energy),
            },
            {
                "force_max_eV_A": np.max(force_max),
                "force_mean_max_eV_A": np.mean(force_max),
            },
        ):
            names = candidates.keys() & claimed.keys()
            if not names:
                if claimed.get("energy_force_analysis"):
                    raise UnsupportedEvidence(
                        "Documented energy/force diagnostics require review"
                    )
                raise EvidenceError("Missing energy or force assessment")
            if any(
                not result_close(candidates[k], claimed[k], atol=1e-7) for k in names
            ):
                return False
        return True

    def structural_assessment():
        frames, _ = md()
        displacement = frames[-1].get_scaled_positions(wrap=False) - frames[
            0
        ].get_scaled_positions(wrap=False)
        displacement = (displacement - np.rint(displacement)) @ frames[0].cell.array
        rms = np.sqrt(np.mean(np.sum(displacement**2, axis=1)))
        distance = frames[-1].get_all_distances(mic=True)
        np.fill_diagonal(distance, np.inf)
        claimed = e.results["bulk"]
        candidates = {
            "final_displacement_rms_A": rms,
            "final_min_pair_distance_A": np.min(distance),
        }
        names = candidates.keys() & claimed.keys()
        if not names:
            if claimed.get("structural_analysis"):
                raise UnsupportedEvidence(
                    "Documented structural analysis requires review"
                )
            raise EvidenceError("Missing structural evolution assessment")
        return all(result_close(candidates[k], claimed[k], atol=1e-7) for k in names)

    def diagnostic_scope():
        # Scope is a factual property of the compared arrays, not a model-quality claim.
        return (
            predictions_valid()
            and e.results["fitting"]["evaluation_domain"] == "training"
        )

    r.check(
        "temperature_assessment",
        5,
        temperature_assessment,
        "Recomputed full-trajectory temperature statistics; no arbitrary target-temperature success gate",
    )
    r.check(
        "energy_and_force_assessment",
        5,
        energy_force_assessment,
        "Recomputed energy excursion and force extrema, including unstable trajectories",
    )
    r.check(
        "structural_evolution_assessment",
        5,
        structural_assessment,
        "Recomputed final displacement and minimum pair distance",
    )
    r.check(
        "diagnostic_scope",
        5,
        diagnostic_scope,
        "Fitting metrics identified as training-set diagnostics",
    )
    r.unverified(
        "execution_provenance",
        "Submitted records cannot establish which teacher/student checkpoint was evaluated, whether model parameters were updated, which checkpoint powered MD, or whether velocities were reset between stored frames.",
    )
    r.unverified(
        "checkpoint_loadability",
        "Container metadata and digest are checked without loading the model; semantic loadability is unverified.",
    )
