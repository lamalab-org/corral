"""Reproduce fixed-volume heat-capacity and return-state analysis from saved data."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.geometry import find_mic

from .common import (
    EvidenceError,
    UnsupportedEvidence,
    close,
    finite_array,
    is_teacher_model,
    result_close,
    scientific_screen,
)

if TYPE_CHECKING:
    from .common import Evidence, Rubric


class UnsupportedAnalysis(UnsupportedEvidence):
    """The chosen method has no safe numerical adapter."""


def _supported(function):
    def run():
        try:
            return function()
        except UnsupportedAnalysis as exc:
            return None, str(exc)

    return run


def _number(value):
    return float(finite_array(value, shape=()))


def _same(a, b, atol=1e-7):
    return close(a, b, rtol=1e-4, atol=atol)


def _time(atoms):
    return _number(
        atoms.info["time_fs"]
        if "time_fs" in atoms.info
        else 1000 * atoms.info["time_ps"]
    )


def _energy(atoms):
    if "potential_energy_eV" in atoms.info:
        return _number(atoms.info["potential_energy_eV"])
    stored = getattr(getattr(atoms, "calc", None), "results", {})
    if "energy" not in stored:
        raise EvidenceError("Potential energy must be stored; no calculator is run")
    return _number(stored["energy"])


def _thermal(atoms, dof):
    if "momenta" not in atoms.arrays:
        raise EvidenceError("Saved momenta are required")
    mass = finite_array(atoms.get_masses(), shape=(108,))
    if np.any(mass <= 0):
        raise EvidenceError("Masses must be positive")
    p = finite_array(atoms.get_momenta(), shape=(108, 3))
    kinetic_axes = np.sum(p**2 / mass[:, None], axis=0) / 2
    directional = 2 * kinetic_axes / ((dof / 3) * units.kB)
    kinetic = float(kinetic_axes.sum())
    potential = _energy(atoms)
    return np.r_[directional.mean(), directional, (potential + kinetic) / 108], (
        potential,
        kinetic,
    )


def _fit(x, y, weights):
    x, y, weights = finite_array(x), finite_array(y), finite_array(weights)
    if x.shape != (5,) or y.shape != (5,) or weights.shape != (5,):
        raise EvidenceError("Fit must use exactly the five forward 300-700 K stages")
    if np.any(weights <= 0):
        raise EvidenceError("Fit weights must be positive")
    weight_sum = weights.sum()
    xmean, ymean = np.dot(weights, x) / weight_sum, np.dot(weights, y) / weight_sum
    dx, dy = x - xmean, y - ymean
    denominator = np.dot(weights, dx**2)
    if denominator <= 1e-12:
        raise EvidenceError("Measured temperatures do not identify a slope")
    slope = np.dot(weights, dx * dy) / denominator
    intercept = ymean - slope * xmean
    return float(slope), float(intercept)


def evaluate(e: Evidence, r: Rubric):
    def prerequisites(return_state=False):
        names = ["initial_fcc_and_temperature", "fixed_cell_and_ordered_atoms"]
        if return_state:
            names.extend(
                [
                    "eight_stage_cycle_and_cumulative_time",
                    "boundary_position_momentum_continuity",
                ]
            )
        checks = {item["name"]: item for item in r.checks}
        for name in names:
            if checks[name]["status"] == "unverified":
                raise UnsupportedAnalysis(
                    f"Scientific check needs verified prerequisite: {name}"
                )
            if checks[name]["status"] != "passed":
                raise EvidenceError(
                    f"Scientific check needs valid prerequisite: {name}"
                )

    @lru_cache(None)
    def stages():
        data = e.settings["stages"]
        if len(data) != 8:
            raise EvidenceError("Expected eight distinctly identified thermal stages")
        ids = [item["id"] for item in data]
        if any(not isinstance(key, str) or not key.strip() for key in ids):
            raise EvidenceError("Each stage needs a nonempty string id")
        if len(set(ids)) != 8:
            raise EvidenceError("Initial and returning 300 K stages need unique ids")
        if not close(
            [item["target_temperature_K"] for item in data],
            [300, 400, 500, 600, 700, 800, 900, 300],
        ):
            raise EvidenceError("Thermal targets or order differ from the task")
        return data

    @lru_cache(None)
    def initial():
        frames = e.trajectory("initial_state", "initial")
        if len(frames) != 1:
            raise EvidenceError("initial_state must contain exactly one frame")
        return frames[0]

    @lru_cache(None)
    def boundaries():
        frames = e.trajectory("boundary_states", "stage_boundaries")
        if len(frames) != 16:
            raise EvidenceError("Save both start and end of every stage")
        return frames

    @lru_cache(None)
    def production():
        return e.trajectory("production_trajectory", "trajectory")

    @lru_cache(None)
    def selected(index):
        config = stages()[index]
        frames = [a for a in production() if a.info["stage"] == config["id"]]
        if len(frames) < 2:
            raise EvidenceError("Save the selected production interval endpoints")
        times = finite_array([_time(a) for a in frames])
        delta = np.diff(times)
        if np.any(delta <= 0):
            raise EvidenceError("Production times must strictly increase")
        if not close(
            times[[0, -1]],
            [config["production_start_time_fs"], config["production_end_time_fs"]],
            rtol=1e-8,
        ):
            raise EvidenceError(
                "Saved frames do not cover the declared production window"
            )
        return frames

    @lru_cache(None)
    def dof():
        value = e.settings["md"]["temperature_dof"]
        if value not in (321, 324):
            raise EvidenceError(
                "Use the declared 3N or 3N-3 kinetic degrees of freedom"
            )
        return value

    @lru_cache(None)
    def values(index):
        return np.array([_thermal(a, dof())[0] for a in selected(index)])

    def result(index):
        return e.results["stages"][stages()[index]["id"]]

    def valid_frame(atoms):
        return (
            len(atoms) == 108
            and np.all(atoms.numbers == 13)
            and bool(atoms.pbc.all())
            and close(atoms.get_masses(), np.full(108, 26.9815385), atol=1e-3)
            and finite_array(atoms.positions, shape=(108, 3)).shape == (108, 3)
            and "momenta" in atoms.arrays
            and finite_array(atoms.get_momenta(), shape=(108, 3)).shape == (108, 3)
        )

    def geometry():
        atoms = initial()
        if not valid_frame(atoms):
            return False
        cell = np.asarray(atoms.cell)
        length = 3 * 4.05
        if not close(cell @ cell.T, np.eye(3) * length**2, atol=1e-4):
            return False
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        # FCC first two neighbour shells verify the conventional repetition,
        # independently of atom order, global origin, or rigid cell rotation.
        shells = np.sort(distances, axis=1)
        if not np.allclose(shells[:, :12], 4.05 / np.sqrt(2), atol=1e-4, rtol=0):
            return False
        if not np.allclose(shells[:, 12:18], 4.05, atol=1e-4, rtol=0):
            return False
        t = _thermal(atoms, dof())[0][0]
        return abs(t - 300) <= 6 * 300 * np.sqrt(2 / dof())

    def protocol():
        md = e.settings["md"]
        return (
            bool(str(md["thermostat"]).strip())
            and _number(md["timestep_fs"]) > 0
            and type(md["random_seed"]) is int
            and md["initial_temperature_K"] == 300
            and type(md["velocity_initializations"]) is int
            and md["velocity_initializations"] == 1
            and type(md["manual_velocity_resets"]) is int
            and md["manual_velocity_resets"] == 0
            and str(md["ensemble"]).upper() == "NVT"
            and dof() in (321, 324)
            and is_teacher_model(e.settings["model"])
            and isinstance(e.settings["model_settings"], dict)
        )

    def identities():
        ref = initial()
        ids = [a.info["stage"] for a in production()]
        ordered = [s["id"] for s in stages()]
        if any(key not in ordered for key in ids):
            return False
        if [ordered.index(key) for key in ids] != sorted(
            ordered.index(key) for key in ids
        ):
            return False
        return all(
            valid_frame(a)
            and close(a.cell.array, ref.cell.array, rtol=0, atol=1e-6)
            and np.array_equal(a.numbers, ref.numbers)
            and close(a.get_masses(), ref.get_masses(), rtol=0, atol=1e-6)
            for a in boundaries() + production()
        )

    def timing():
        previous = 0.0
        step = _number(e.settings["md"]["timestep_fs"])
        if step <= 0:
            return False
        for i, config in enumerate(stages()):
            start, end, prod_start, prod_end = finite_array(
                [
                    config[key]
                    for key in (
                        "start_time_fs",
                        "end_time_fs",
                        "production_start_time_fs",
                        "production_end_time_fs",
                    )
                ]
            )
            if not close(start, previous, rtol=0, atol=1e-7):
                return False
            if not start < prod_start < prod_end <= end:
                return False
            for atoms, time in zip(
                boundaries()[2 * i : 2 * i + 2], (start, end), strict=True
            ):
                if atoms.info["stage"] != config["id"] or not close(_time(atoms), time):
                    return False
            times = np.r_[[start, end], [_time(a) for a in selected(i)]]
            if not np.allclose(times / step, np.rint(times / step), rtol=0, atol=1e-5):
                return False
            previous = end
        return True

    def same_state(a, b):
        displacement, _ = find_mic(a.positions - b.positions, a.cell, pbc=True)
        return (
            close(displacement, np.zeros((108, 3)), rtol=0, atol=1e-6)
            and close(a.get_momenta(), b.get_momenta(), rtol=0, atol=1e-7)
            and close(a.cell.array, b.cell.array, rtol=0, atol=1e-6)
            and close(_energy(a), _energy(b), rtol=0, atol=1e-6)
            and close(_time(a), _time(b), rtol=0, atol=1e-7)
        )

    def continuity():
        b = boundaries()
        if not same_state(initial(), b[0]):
            return False
        if not all(same_state(b[2 * i - 1], b[2 * i]) for i in range(1, 8)):
            return False
        for i in range(8):
            for frame in selected(i):
                for edge in b[2 * i : 2 * i + 2]:
                    if close(
                        _time(frame), _time(edge), rtol=0, atol=1e-7
                    ) and not same_state(frame, edge):
                        return False
        return True

    r.check("initial_fcc_and_temperature", 6, geometry)
    r.check("recorded_nvt_model_and_initialization", 4, protocol)
    r.check("fixed_cell_and_ordered_atoms", 5, identities)
    r.check("eight_stage_cycle_and_cumulative_time", 5, _supported(timing))
    r.check("boundary_position_momentum_continuity", 6, _supported(continuity))

    @lru_cache(None)
    def trace(index):
        table = e.table("thermal_trace", "thermal_traces")
        rows = table[table["stage"] == stages()[index]["id"]]
        times = finite_array(rows["time_fs"], ndim=1)
        if np.any(np.diff(times) <= 0):
            raise EvidenceError(
                "Thermal trace time must strictly increase within a stage"
            )
        config = stages()[index]
        if not close(times[[0, -1]], [config["start_time_fs"], config["end_time_fs"]]):
            raise EvidenceError("Thermal trace must cover equilibration and stage end")
        return rows

    def logs():
        names = [
            "temperature_K",
            "temperature_x_K",
            "temperature_y_K",
            "temperature_z_K",
        ]
        for i in range(8):
            rows = trace(i)
            temperatures = finite_array(rows[names].to_numpy(), ndim=2)
            pe = finite_array(rows["potential_energy_eV"])
            ke = finite_array(rows["kinetic_energy_eV"])
            if np.any(ke < 0) or np.any(temperatures < 0):
                return False
            if not _same(temperatures[:, 0], temperatures[:, 1:].mean(axis=1)):
                return False
            if not _same(2 * ke / (dof() * units.kB), temperatures[:, 0]):
                return False
            if not _same(pe + ke, rows["total_energy_eV"].to_numpy()):
                return False
            for frame in boundaries()[2 * i : 2 * i + 2] + selected(i):
                match = rows[
                    np.isclose(rows["time_fs"], _time(frame), rtol=0, atol=1e-7)
                ]
                if len(match) != 1:
                    return False
                row = match.iloc[0]
                thermal, energy = _thermal(frame, dof())
                if not _same(thermal[:4], row[names].to_numpy(dtype=float)):
                    return False
                if not _same(
                    energy,
                    row[["potential_energy_eV", "kinetic_energy_eV"]].to_numpy(
                        dtype=float
                    ),
                ):
                    return False
        return True

    def temperatures():
        for i in range(8):
            data, claim = values(i), result(i)
            if not (
                result_close(data[:, 0].mean(), claim["temperature_mean_K"], atol=1e-07)
                and (
                    "temperature_std_K" not in claim
                    or result_close(
                        data[:, 0].std(), claim["temperature_std_K"], atol=1e-07
                    )
                )
                and result_close(
                    data[:, 1:4].mean(axis=0),
                    claim["directional_temperature_mean_K"],
                    atol=1e-07,
                )
            ):
                return False
        return True

    def target_sanity():
        compatible = []
        for i, config in enumerate(stages()):
            target = config["target_temperature_K"]
            mean = values(i)[:, :4].mean(axis=0)
            compatible.append(abs(mean[0] - target) <= 6 * target * np.sqrt(2 / dof()))
            compatible.append(
                np.all(abs(mean[1:] - target) <= 6 * target * np.sqrt(6 / dof()))
            )
        return scientific_screen(
            all(compatible),
            "Overall or directional production temperature is inconclusive",
        )

    def equilibration():
        for i, config in enumerate(stages()):
            rows = trace(i)
            rows = rows[rows["time_fs"] < config["production_start_time_fs"]]
            if len(rows) < 2:
                return False
            temp = finite_array(rows["temperature_K"])
            times = finite_array(rows["time_fs"]) / 1000
            half = len(temp) // 2
            expected = [
                temp[:half].mean(),
                temp[half:].mean(),
                np.polyfit(times - times[0], temp, 1)[0],
            ]
            claim = result(i).get("equilibration", {})
            if not isinstance(claim, dict):
                raise EvidenceError("Equilibration analysis must be a JSON object")
            keys = (
                "temperature_first_half_mean_K",
                "temperature_second_half_mean_K",
                "temperature_drift_K_per_ps",
            )
            if any(
                key in claim and not result_close(value, claim[key], atol=1e-7)
                for key, value in zip(keys, expected, strict=True)
            ):
                return False
        return True

    r.check("thermal_trace_from_saved_momenta_and_energy", 7, _supported(logs))
    r.check("total_and_directional_temperature_means", 6, _supported(temperatures))
    r.check("production_target_temperature_sanity", 4, _supported(target_sanity))
    r.check("equilibration_thermal_diagnostics", 5, equilibration)

    def energy_means():
        return all(
            result_close(
                values(i)[:, 4].mean(), result(i)["energy_mean_eV_per_atom"], atol=1e-07
            )
            for i in range(8)
        )

    def weights():
        config = e.settings["fit"]
        if config["method"] == "ols":
            return np.ones(5)
        if config["method"] == "wls":
            values = finite_array(config["weights"], shape=(5,))
            if np.any(values <= 0):
                raise EvidenceError("WLS requires five positive declared weights")
            return values
        raise UnsupportedAnalysis("Automatic regression check supports ols and wls")

    @lru_cache(None)
    def fitted():
        prerequisites()
        means = np.array([values(i).mean(axis=0) for i in range(5)])
        if e.settings["fit"]["method"] == "theil_sen":
            x, y = means[:, 0], means[:, 4]
            pairs = [(i, j) for i in range(5) for j in range(i + 1, 5) if x[i] != x[j]]
            if not pairs:
                raise EvidenceError(
                    "Heat-capacity fit needs distinct measured temperatures"
                )
            slope = np.median([(y[j] - y[i]) / (x[j] - x[i]) for i, j in pairs])
            return slope, np.median(y - slope * x)
        return _fit(means[:, 0], means[:, 4], weights())

    def fit_result():
        claim = e.results["heat_capacity"]
        return (
            fit_stage_selection()
            and result_close(fitted()[0], claim["slope_eV_per_atom_K"], atol=1e-10)
            and result_close(fitted()[1], claim["intercept_eV_per_atom"], atol=1e-07)
        )

    def fit_stage_selection():
        ids = e.results["heat_capacity"]["fit_stage_ids"]
        return (
            isinstance(ids, list)
            and len(ids) == 5
            and set(ids) == {item["id"] for item in stages()[:5]}
        )

    def consistent_fit_claim():
        """Keep arithmetic independent of adapter coverage, not of known errors."""
        prerequisites()
        claim = e.results["heat_capacity"]
        if not fit_stage_selection():
            raise EvidenceError("Heat-capacity fit must use the five requested stages")
        slope = _number(claim["slope_eV_per_atom_K"])
        intercept = _number(claim["intercept_eV_per_atom"])
        try:
            actual = fitted()
        except UnsupportedAnalysis:
            return slope, intercept
        if not result_close(slope, actual[0], atol=1e-10) or not result_close(
            intercept, actual[1], atol=1e-7
        ):
            raise EvidenceError("Reported fit contradicts the supported calculation")
        return slope, intercept

    def capacity_units():
        slope, _ = consistent_fit_claim()
        claim = e.results["heat_capacity"]
        return (
            claim["units"] == "eV/(atom K)"
            and claim["scaled_units"] == "k_B/atom"
            and result_close(slope / units.kB, claim["value_kB_per_atom"], atol=1e-07)
        )

    r.check("total_energy_per_atom_means", 8, _supported(energy_means))
    r.check("measured_temperature_heat_capacity_fit", 10, _supported(fit_result))
    r.check("heat_capacity_units_and_kB_conversion", 4, _supported(capacity_units))

    def residuals():
        slope, intercept = consistent_fit_claim()
        for i in (5, 6):
            mean = values(i).mean(axis=0)
            residual = mean[4] - (intercept + slope * mean[0])
            if not result_close(
                residual,
                e.results["high_temperature_residuals"][stages()[i]["id"]],
                atol=1e-07,
            ):
                return False
        return True

    @lru_cache(None)
    def energy_return():
        prerequisites(return_state=True)
        return values(7)[:, 4].mean() - values(0)[:, 4].mean()

    def energy_comparison():
        return result_close(
            energy_return(),
            e.results["return_state"]["energy_difference_eV_per_atom"],
            atol=1e-7,
        )

    def structural_values(index):
        config = e.settings["structure"]
        method, output = config["method"], []
        for frame in selected(index):
            if method == "coordination":
                cutoff = _number(config["cutoff_A"])
                if not 0 < cutoff < min(frame.cell.lengths()) / 2:
                    raise EvidenceError(
                        "Coordination cutoff must be positive and below half the shortest cell vector"
                    )
                initial_distances = initial().get_all_distances(mic=True)
                np.fill_diagonal(initial_distances, np.inf)
                initial_coordination = np.mean(
                    np.sum(initial_distances < cutoff, axis=1)
                )
                if not 0 < initial_coordination < 107:
                    raise EvidenceError(
                        "Coordination cutoff yields a vacuous initial-neighbour comparison"
                    )
                distances = frame.get_all_distances(mic=True)
                np.fill_diagonal(distances, np.inf)
                output.append(float(np.mean(np.sum(distances < cutoff, axis=1))))
            elif method == "rms_displacement":
                if type(config["remove_translation"]) is not bool:
                    raise EvidenceError("remove_translation must be boolean")
                displacement, _ = find_mic(
                    frame.positions - initial().positions, frame.cell, pbc=True
                )
                if config["remove_translation"]:
                    displacement -= displacement.mean(axis=0)
                output.append(float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))))
            elif method == "nearest_neighbor_distance":
                distance = frame.get_all_distances(mic=True)
                np.fill_diagonal(distance, np.inf)
                output.append(float(np.min(distance, axis=1).mean()))
            else:
                raise UnsupportedAnalysis(
                    "The documented structural metric requires independent review"
                )
        return np.array(output)

    @lru_cache(None)
    def structure_return():
        prerequisites(return_state=True)
        first, last = (structural_values(i).mean() for i in (0, 7))
        return np.array([first, last, last - first])

    def structure_comparison():
        claim = e.results["return_state"]
        # Difference arithmetic is required for every metric, even when the
        # structural calculation itself needs independent review.
        if not result_close(
            _number(claim["return_structural_mean"])
            - _number(claim["initial_structural_mean"]),
            claim["structural_difference"],
            atol=1e-7,
        ):
            return False
        return result_close(
            structure_return(),
            [
                claim[key]
                for key in (
                    "initial_structural_mean",
                    "return_structural_mean",
                    "structural_difference",
                )
            ],
            atol=1e-07,
        )

    r.check("high_temperature_energy_residuals", 7, _supported(residuals))
    r.check("return_minus_initial_energy", 6, _supported(energy_comparison))
    r.check(
        "consistent_structural_return_comparison", 7, _supported(structure_comparison)
    )
    r.unverified(
        "execution_provenance",
        "Saved artifacts cannot establish actual MACE loading, thermostat execution, single random initialization, or absence of resets between saved frames.",
    )
    r.unverified(
        "scientific_sampling_adequacy",
        "Finite traces do not prove equilibration or a scientifically sufficient structural metric.",
    )
