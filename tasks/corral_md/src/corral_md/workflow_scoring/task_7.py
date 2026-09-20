"""Check two thermal-expansion routes using saved states and raw records only."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from scipy.spatial import ConvexHull

from .common import (
    EvidenceError,
    UnsupportedEvidence,
    close,
    finite_array,
    result_close,
    scientific_screen,
)

if TYPE_CHECKING:
    from .common import Evidence, Rubric


EV_A3_TO_GPA = 160.21766208


class UnsupportedMethod(UnsupportedEvidence):
    """A submitted analysis convention needs a separately reviewed adapter."""


def _number(value):
    return float(finite_array(value, shape=()))


def _matches(actual, expected, atol=1e-7):
    return close(actual, expected, rtol=1e-4, atol=atol)


def _checked(fn):
    def check():
        try:
            return fn()
        except UnsupportedMethod as exc:
            return None, str(exc)

    return check


def _time(a):
    return _number(
        a.info["time_fs"] if "time_fs" in a.info else 1000 * a.info["time_ps"]
    )


def _energy(a):
    stored = getattr(getattr(a, "calc", None), "results", {})
    for name in ("potential_energy_eV", "potential_energy", "energy"):
        if name in a.info:
            return _number(a.info[name])
    if "energy" not in stored:
        raise EvidenceError(
            "Potential energy must be saved; calculator calls are forbidden"
        )
    return _number(stored["energy"])


def _same_state(a, b, kinetic=True):
    if (
        len(a) != len(b)
        or not np.array_equal(a.numbers, b.numbers)
        or not _matches(a.cell.array, b.cell.array, 1e-5)
    ):
        return False
    difference = a.get_scaled_positions(wrap=False) - b.get_scaled_positions(wrap=False)
    difference -= np.rint(difference)
    if np.max(np.abs(difference @ a.cell.array)) > 1e-5:
        return False
    if kinetic:
        return (
            _matches(a.arrays["momenta"], b.arrays["momenta"], 1e-6)
            and abs(_time(a) - _time(b)) <= 1e-6
        )
    return True


def _same_geometry(a, b):
    """Compare supplied FCC cells while allowing origin, rotation and atom order."""
    if (
        len(b) != 108
        or set(b.get_chemical_symbols()) != {"Al"}
        or not np.all(b.pbc)
        or not close(
            a.cell.array @ a.cell.array.T,
            b.cell.array @ b.cell.array.T,
            rtol=0,
            atol=1e-4,
        )
    ):
        return False
    left = finite_array(a.get_scaled_positions(wrap=False), shape=(108, 3))
    right = finite_array(b.get_scaled_positions(wrap=False), shape=(108, 3))
    right = right + (left[0] - right[0])
    delta = left[:, None, :] - right[None, :, :]
    delta -= np.rint(delta)
    distance = np.linalg.norm(delta @ a.cell.array, axis=-1)
    nearest = np.argmin(distance, axis=1)
    return (
        np.max(distance[np.arange(108), nearest]) <= 1e-5 and len(set(nearest)) == 108
    )


def _fit(x, y):
    x, y = finite_array(x, ndim=2), finite_array(y, ndim=1)
    # Column scaling avoids artificial rank loss from Kelvin/Angstrom units.
    scale = np.max(np.abs(x), axis=0)
    scale[scale == 0] = 1
    if len(y) < x.shape[1] or np.linalg.matrix_rank(x / scale) != x.shape[1]:
        raise EvidenceError("Local fit is underdetermined or rank deficient")
    coeff = np.linalg.lstsq(x / scale, y, rcond=None)[0] / scale
    return coeff, y - x @ coeff


def _interval_fit(data, config):
    """Reproduce a declared line estimator without prescribing a workflow."""
    if not isinstance(config, dict):
        raise EvidenceError("Volume-fit settings must be an object")
    method = config.get("method", "ols")
    if not isinstance(method, str) or not method.strip():
        raise EvidenceError("Volume-fit method must be a nonempty string")
    temperature, volume = data[:, 0], data[:, 1]
    design = np.column_stack([np.ones(len(data)), temperature])
    if method in (
        "ols",
        "linear",
        "ordinary_least_squares",
        "unweighted_least_squares",
    ):
        return _fit(design, volume)
    if method in ("wls", "weighted_least_squares"):
        weights = finite_array(config["weights"], shape=(len(data),))
        if np.any(weights <= 0):
            raise EvidenceError("Volume-fit weights must be finite and positive")
        # Normalization preserves the objective while avoiding overflow when
        # uncertainty-derived weights have a large common scale.
        root = np.sqrt(weights / weights.max())
        coefficients, _ = _fit(design * root[:, None], volume * root)
        return coefficients, volume - design @ coefficients
    if method == "theil_sen":
        left, right = np.triu_indices(len(data), k=1)
        distinct = temperature[left] != temperature[right]
        left, right = left[distinct], right[distinct]
        if not len(left):
            raise EvidenceError("Volume-fit temperatures do not identify a slope")
        slope = np.median(
            (volume[right] - volume[left]) / (temperature[right] - temperature[left])
        )
        intercept_method = config.get("intercept_method", "joint")
        if intercept_method == "joint":
            intercept = np.median(volume - slope * temperature)
        elif intercept_method == "separate":
            intercept = np.median(volume) - slope * np.median(temperature)
        else:
            raise UnsupportedMethod(
                f"Theil-Sen intercept convention {intercept_method!r} needs independent review"
            )
        coefficients = np.array([intercept, slope])
        return coefficients, volume - design @ coefficients
    raise UnsupportedMethod(
        f"Volume line estimator {method!r} needs independent review"
    )


def _npt_fit(means, config, center, volume):
    method = config["method"]
    if method not in ("linear", "quadratic", "finite_difference"):
        raise UnsupportedMethod(f"Local NPT estimator {method!r} is not supported")
    selected = config["stage_ids"]
    if len(set(selected)) != len(selected):
        raise EvidenceError("Repeated fit stage identity")
    data = finite_array([means[sid] for sid in selected])
    dt = data[:, 0] - center
    if not min(dt) <= 1e-7 or not max(dt) >= -1e-7:
        raise UnsupportedMethod(
            "A one-sided local NPT derivative needs independent assessment"
        )
    if method == "finite_difference":
        weights = finite_array(config["derivative_weights_K_inv"], shape=(len(dt),))
        if not _matches(weights.sum(), 0, 1e-10) or not _matches(weights @ dt, 1, 1e-8):
            raise EvidenceError(
                "Local derivative weights must cancel constants and differentiate temperature"
            )
        return float(weights @ data[:, 1]) / volume, None
    columns = [np.ones(len(dt)), dt]
    if method == "quadratic":
        columns.append(dt * dt)
    coef, residual = _fit(np.column_stack(columns), data[:, 1])
    return coef[1] / volume, residual


def _pressure_fit(means, config, volume, require_hull=True):
    method = config["method"]
    if method not in ("linear", "quadratic", "finite_difference"):
        raise UnsupportedMethod(f"Pressure estimator {method!r} is not supported")
    selected = config["stage_ids"]
    if len(set(selected)) != len(selected):
        raise EvidenceError("Repeated fit stage identity")
    data = finite_array([means[sid] for sid in selected])
    dt, dv = data[:, 0] - 400, data[:, 1] - volume
    points = np.column_stack([dt, dv])
    scale = np.max(np.abs(points), axis=0)
    if np.any(scale <= 1e-12):
        raise EvidenceError("Pressure fit must vary both temperature and volume")
    columns = [np.ones(len(dt)), dt, dv]
    if method == "quadratic":
        columns.extend([dt * dt, dt * dv, dv * dv])
    if method == "finite_difference":
        weights = finite_array(config["derivative_weights"], shape=(2, len(data)))
        if not _matches(weights.sum(axis=1), np.zeros(2), 1e-10) or not _matches(
            weights @ points, np.eye(2), 1e-8
        ):
            raise EvidenceError(
                "Pressure derivative weights must cancel constants and differentiate T,V"
            )
        derivatives = weights @ data[:, 2]
        intercept = np.mean(data[:, 2] - points @ derivatives)
        coef = np.r_[intercept, derivatives]
        residual = data[:, 2] - np.column_stack(columns) @ coef
    else:
        coef, residual = _fit(np.column_stack(columns), data[:, 2])
    modulus = -volume * coef[2]
    if abs(modulus) < 1e-14:
        raise EvidenceError("Zero fitted bulk modulus prevents a finite beta")
    if require_hull:
        hull = ConvexHull(points / scale)
        if np.max(hull.equations[:, -1]) > 1e-7:
            raise UnsupportedMethod(
                "The declared one-sided pressure derivative needs independent locality review"
            )
    return modulus, coef[1] / modulus, coef, residual


def evaluate(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def stages():
        raw = e.json("stages", "stage_map")
        records = raw["stages"] if isinstance(raw, dict) else raw
        if not isinstance(records, list) or not records:
            raise EvidenceError("Expected nonempty stage identity map")
        ids = [s["id"] for s in records]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(
            ids
        ):
            raise EvidenceError("Stage identities must be unique nonempty strings")
        if any(s["ensemble"] not in ("NPT", "NVT") for s in records):
            raise EvidenceError("Supported ensembles are NPT and NVT")
        return records

    @lru_cache(None)
    def stage(sid):
        info = next(s for s in stages() if s["id"] == sid)
        frames = e.trajectory(f"trajectories.{sid}")
        if len(frames) < 2:
            raise EvidenceError("A stage needs saved start and end states")
        times = finite_array([_time(a) for a in frames])
        if np.any(np.diff(times) <= 0):
            raise EvidenceError("Stage sample times must strictly increase")
        interval = info["production"]
        if (
            len(interval) != 2
            or any(type(i) is not int for i in interval)
            or not 0 <= interval[0] < interval[1] <= len(frames)
            or interval[1] - interval[0] < 2
        ):
            raise EvidenceError("Invalid production sample interval")
        for a in frames:
            if (
                len(a) != 108
                or set(a.get_chemical_symbols()) != {"Al"}
                or not np.all(a.pbc)
            ):
                raise EvidenceError("Expected periodic Al108")
            finite_array(a.positions, shape=(108, 3))
            finite_array(a.cell.array, shape=(3, 3))
            finite_array(a.arrays["momenta"], shape=(108, 3))
            _energy(a)
            if a.info.get("stage") != sid or not close(
                a.get_masses(), np.full(108, 26.9815385), rtol=0, atol=1e-3
            ):
                raise EvidenceError("Inconsistent stage identity or Al masses")
            if a.get_volume() <= 0:
                raise EvidenceError("Invalid cell volume")
        return info, frames, times

    @lru_cache(None)
    def raw(sid):
        info, frames, times = stage(sid)
        dof = e.settings["temperature_dof"]
        if dof not in (321, 324):
            raise UnsupportedMethod("Temperature DOF must be documented as 321 or 324")
        kinetic = np.array(
            [
                np.sum(a.arrays["momenta"] ** 2 / a.get_masses()[:, None]) / 2
                for a in frames
            ]
        )
        temperature = 2 * kinetic / (dof * units.kB)
        volume = np.array([a.get_volume() for a in frames])
        potential = np.array([_energy(a) for a in frames])
        return np.column_stack([temperature, volume]), kinetic, potential

    @lru_cache(None)
    def trace():
        table = e.table("thermal_trace", "raw_trace")
        if table.duplicated(["stage", "time_fs"]).any():
            raise EvidenceError("Duplicate raw stage/time record")
        return table

    @lru_cache(None)
    def pressure(sid):
        _, _, times = stage(sid)
        values, kinetic, potential = raw(sid)
        table = trace()
        chosen = table.loc[table["stage"] == sid]
        rows = []
        for time in times:
            match = chosen.loc[np.abs(chosen["time_fs"] - time) <= 1e-6]
            if len(match) != 1:
                raise EvidenceError(
                    f"Missing or ambiguous raw stress/thermal row for {sid}"
                )
            rows.append(match.iloc[0])
        stress = finite_array(
            [
                [row[k] for k in ("sxx", "syy", "szz", "syz", "sxz", "sxy")]
                for row in rows
            ]
        )
        factors = {"eV/Angstrom^3": EV_A3_TO_GPA, "GPa": 1.0, "bar": 1e-4}
        if e.settings["stress_unit"] not in factors or e.settings[
            "stress_sign"
        ] not in ("tensile_positive", "compression_positive"):
            raise UnsupportedMethod("Unsupported stress unit/sign convention")
        sign = -1 if e.settings["stress_sign"] == "tensile_positive" else 1
        p = sign * stress[:, :3].mean(axis=1) * factors[e.settings["stress_unit"]]
        if e.settings["stress_kind"] == "potential_only":
            p += 2 * kinetic / (3 * values[:, 1]) * EV_A3_TO_GPA
        elif e.settings["stress_kind"] not in {"total", "including_kinetic"}:
            raise UnsupportedMethod("The declared stress convention needs review")
        comparisons = {
            "temperature_K": values[:, 0],
            "volume_A3": values[:, 1],
            "kinetic_energy_eV": kinetic,
            "potential_energy_eV": potential,
            "pressure_GPa": p,
        }
        for key, expected in comparisons.items():
            if not _matches([row[key] for row in rows], expected):
                raise EvidenceError(
                    f"{sid}: raw {key} does not agree with saved states"
                )
        return p

    def npt():
        return [s for s in stages() if s["ensemble"] == "NPT"]

    def find(branch, temp):
        found = [
            s["id"]
            for s in npt()
            if s["branch"] == branch and abs(s["target_temperature_K"] - temp) < 1e-6
        ]
        if len(found) != 1:
            raise EvidenceError(f"Need one identifiable {branch} {temp} K stage")
        return found[0]

    def config_valid(config, ensemble, heating=False):
        records = {s["id"]: s for s in stages()}
        return bool(config["stage_ids"]) and all(
            records[sid]["ensemble"] == ensemble
            and (not heating or records[sid]["branch"] == "heating")
            for sid in config["stage_ids"]
        )

    @lru_cache(None)
    def means(ensemble=None):
        result = {}
        for info in stages():
            if ensemble is not None and info["ensemble"] != ensemble:
                continue
            sid = info["id"]
            values, _, _ = raw(sid)
            a, b = info["production"]
            result[sid] = np.r_[values[a:b].mean(axis=0), pressure(sid)[a:b].mean()]
        return result

    def temperature_control(info):
        a, b = info["production"]
        temperature = raw(info["id"])[0][a:b, 0]
        target = _number(info["target_temperature_K"])
        allowance = max(
            0.25 * target, 6 * np.std(temperature, ddof=1) / np.sqrt(len(temperature))
        )
        return target > 0 and abs(temperature.mean() - target) <= allowance

    @lru_cache(None)
    def geometry():
        _, frames, _ = stage(npt()[0]["id"])
        a = frames[0]
        reference = e.trajectory("initial_structure")[0]
        if not _same_geometry(a, reference):
            return False
        lattice = (a.get_volume() / 27) ** (1 / 3)
        gram = a.cell.array @ a.cell.array.T
        distances = np.sort(a.get_all_distances(mic=True), axis=1)
        return (
            close(gram, np.eye(3) * (3 * lattice) ** 2, rtol=0, atol=0.0001)
            and close(
                distances[:, 1:13],
                np.full((108, 12), lattice / np.sqrt(2)),
                rtol=0,
                atol=0.0001,
            )
            and close(
                distances[:, 13:19], np.full((108, 6), lattice), rtol=0, atol=0.0001
            )
        )

    def settings():
        s = e.settings
        return (
            "MACE-MP-0" in str(s["model"]).upper().replace("_", "-")
            and bool(s["input_structure"])
            and type(s["random_seed"]) is int
            and s["velocity_initializations"] == 1
            and s["remove_com"] is True
            and s["timestep_fs"] > 0
            and bool(s["thermostat"])
            and bool(s["barostat"])
            and abs(s["target_pressure_bar"] - 1.01325) <= 1e-6
            and s["isotropic"] is True
        )

    @lru_cache(None)
    def sequence():
        records = npt()
        for branch in ("heating", "cooling"):
            for temperature in (300, 400, 500) if branch == "heating" else (400, 300):
                find(branch, temperature)
        branches = [s["branch"] for s in records]
        split = branches.index("cooling")
        if branches != ["heating"] * split + ["cooling"] * (len(records) - split):
            return False
        temps = [s["target_temperature_K"] for s in records]
        if (
            temps[0] != 300
            or temps[split - 1] != 500
            or temps[-1] != 300
            or np.any(np.diff(temps[:split]) <= 0)
            or np.any(np.diff(temps[split - 1 :]) >= 0)
        ):
            return False
        first = stage(records[0]["id"])[1][0]
        if np.linalg.norm(first.arrays["momenta"].sum(axis=0)) > 1e-6:
            return False
        previous = None
        for info in records:
            _, frames, _ = stage(info["id"])
            if previous is not None and not _same_state(previous, frames[0]):
                return False
            for a in frames:
                ratio = (a.get_volume() / first.get_volume()) ** (1 / 3)
                if not _matches(a.cell.array, first.cell.array * ratio, 1e-5):
                    return False
            previous = frames[-1]
        return True

    def measured_targets():
        compatible = []
        for info in stages():
            compatible.append(temperature_control(info))
            if info["ensemble"] == "NPT":
                a, b = info["production"]
                measured_pressure = pressure(info["id"])[a:b]
                allowance = max(
                    1.0,
                    6
                    * np.std(measured_pressure, ddof=1)
                    / np.sqrt(len(measured_pressure)),
                )
                compatible.append(
                    abs(measured_pressure.mean() - 1.01325e-4) <= allowance
                )
        return scientific_screen(
            all(compatible), "Measured temperature or pressure control is inconclusive"
        )

    def volume_summaries():
        expected_units = {
            "temperature": "K",
            "volume": "Angstrom^3",
            "pressure": "GPa",
            "bulk_modulus": "GPa",
            "beta": "K^-1",
        }
        if any(e.result("units")[k] != v for k, v in expected_units.items()):
            return False
        for info in npt():
            sid = info["id"]
            values, _, _ = raw(sid)
            a, b = info["production"]
            selected = values[a:b]
            time = stage(sid)[2][a:b] / 1000
            drift, _ = _fit(
                np.column_stack([np.ones(len(time)), time - time.mean()]),
                selected[:, 0],
            )
            vdrift, _ = _fit(
                np.column_stack([np.ones(len(time)), time - time.mean()]),
                selected[:, 1],
            )
            result = e.result("stages")[sid]
            expected = {
                "temperature_K": selected[:, 0].mean(),
                "volume_A3": selected[:, 1].mean(),
                "pressure_GPa": pressure(sid)[a:b].mean(),
                "temperature_drift_K_per_ps": drift[1],
                "volume_drift_A3_per_ps": vdrift[1],
            }
            if any(
                not result_close(result[k], v, atol=1e-07)
                for k, v in expected.items()
                if k in {"temperature_K", "volume_A3", "pressure_GPa"} or k in result
            ):
                return False
            middle = len(selected) // 2
            for name, part in [
                ("first_half", selected[:middle]),
                ("second_half", selected[middle:]),
            ]:
                if name not in result:
                    continue
                if not result_close(
                    [result[name]["temperature_K"], result[name]["volume_A3"]],
                    part.mean(axis=0),
                    atol=1e-07,
                ):
                    return False
        return True

    def volume_fit():
        if not geometry() or not sequence():
            return False
        averages = means("NPT")
        heating = [
            s["id"]
            for s in npt()
            if s["branch"] == "heating" and 300 <= s["target_temperature_K"] <= 500
        ]
        data = np.array([averages[sid] for sid in heating])
        claims = e.result("volume_fit")
        slope = _number(claims["slope_A3_per_K"])
        residuals = finite_array(claims["residuals_A3"], shape=(len(data),))
        if not result_close(
            e.result("reference_volume_A3"),
            averages[find("heating", 400)][1],
            atol=1e-07,
        ):
            return False
        try:
            coef, residual = _interval_fit(data, e.settings.get("volume_fit", {}))
        except UnsupportedMethod:
            # Residuals identify the submitted fitted values. Verify that those
            # values form a line independently of how that line was estimated.
            design = np.column_stack([np.ones(len(data)), data[:, 0]])
            claimed_line, _ = _fit(design, data[:, 1] - residuals)
            if not (
                result_close(slope, claimed_line[1], atol=1e-07)
                and result_close(
                    residuals, data[:, 1] - design @ claimed_line, atol=1e-07
                )
                and result_close(
                    e.result("beta_volume"),
                    claimed_line[1] / averages[find("heating", 300)][1],
                    atol=1e-10,
                )
            ):
                return False
            raise
        return (
            result_close(slope, coef[1], atol=1e-07)
            and result_close(residuals, residual, atol=1e-07)
            and result_close(
                e.result("beta_volume"),
                coef[1] / averages[find("heating", 300)][1],
                atol=1e-10,
            )
        )

    def local_fit():
        if (
            not geometry()
            or not sequence()
            or not config_valid(e.settings["local_npt"], "NPT", True)
        ):
            return False
        av = means("NPT")
        value, residual = _npt_fit(
            av,
            e.settings["local_npt"],
            av[find("heating", 400)][0],
            av[find("heating", 400)][1],
        )
        if not result_close(e.result("beta_npt_local"), value, atol=1e-10):
            return False
        if residual is None:
            return result_close(
                e.result("local_npt_fit")["derivative_A3_per_K"],
                value * av[find("heating", 400)][1],
                atol=1e-7,
            )
        return result_close(
            e.result("local_npt_fit")["residuals_A3"], residual, atol=1e-07
        )

    @lru_cache(None)
    def nvt_states():
        records = [s for s in stages() if s["ensemble"] == "NVT"]
        if not records:
            return False
        for info in records:
            _, frames, _ = stage(info["id"])
            if any(
                not _matches(a.cell.array, frames[0].cell.array, 1e-5) for a in frames
            ):
                return False
            a, b = info["production"]
            expected = np.r_[
                raw(info["id"])[0][a:b].mean(axis=0), pressure(info["id"])[a:b].mean()
            ]
            result = e.result("stages")[info["id"]]
            if not result_close(
                [result["temperature_K"], result["volume_A3"], result["pressure_GPa"]],
                expected,
                atol=1e-07,
            ):
                return False
        return True

    def raw_pressure():
        for info in stages():
            pressure(info["id"])
        return True

    def pressure_fit():
        if (
            not nvt_states()
            or not geometry()
            or not sequence()
            or not config_valid(e.settings["pressure_fit"], "NVT")
        ):
            return False
        av = means()
        modulus, beta, coef, residual = _pressure_fit(
            av, e.settings["pressure_fit"], av[find("heating", 400)][1]
        )
        claims = e.result("pressure_fit")
        if e.settings["pressure_fit"]["method"] == "finite_difference":
            derivative_match = result_close(
                [claims["dP_dT_GPa_per_K"], claims["dP_dV_GPa_per_A3"]],
                coef[1:],
                atol=1e-7,
            )
        else:
            derivative_match = result_close(claims["coefficients"], coef, atol=1e-7)
        return (
            result_close(e.result("bulk_modulus_GPa"), modulus, atol=1e-07)
            and result_close(e.result("beta_pressure"), beta, atol=1e-10)
            and derivative_match
            and result_close(
                e.result("pressure_fit")["residuals_GPa"], residual, atol=1e-07
            )
        )

    def volume_differences():
        if not geometry() or not sequence():
            return False
        av = means("NPT")
        return all(
            result_close(
                e.result("volume_differences")[str(t)]["value_A3"],
                av[find("cooling", t)][1] - av[find("heating", t)][1],
                atol=1e-07,
            )
            for t in (300, 400)
        )

    def comparison():
        for check in (local_fit, pressure_fit):
            try:
                if not check():
                    return False
            except UnsupportedMethod:
                # Method validation stays pending independently; subtraction
                # of the requested reported estimates remains reproducible.
                pass
        return result_close(
            e.result("comparison")["difference_K_inv"],
            _number(e.result("beta_pressure")) - _number(e.result("beta_npt_local")),
            atol=1e-10,
        )

    def stability():
        try:
            if not pressure_fit():
                return False
        except UnsupportedMethod:
            pass
        modulus = _number(e.result("bulk_modulus_GPa"))
        return type(e.result("stable")) is bool and e.result("stable") == (modulus > 0)

    r.check("initial_fcc_geometry", 5, geometry)
    r.check("physical_settings_record", 3, settings)
    r.check("continuous_isotropic_npt_cycle", 10, _checked(sequence))
    r.check("measured_npt_summaries_and_drift", 6, _checked(volume_summaries))
    r.check("measured_temperature_and_pressure_targets", 2, measured_targets)
    r.check("interval_expansion_fit", 10, _checked(volume_fit))
    r.check("local_npt_expansion_fit", 6, _checked(local_fit))
    r.check("fixed_cell_nvt_evidence", 8, nvt_states)
    r.check("raw_stress_and_kinetic_pressure", 10, _checked(raw_pressure))
    r.check("local_pressure_derivatives", 12, _checked(pressure_fit))
    r.check("heating_cooling_differences", 8, _checked(volume_differences))
    r.check("local_route_comparison", 8, _checked(comparison))
    r.check(
        "bulk_modulus_stability_diagnostic",
        2,
        _checked(stability),
    )
    r.unverified(
        "execution_and_input_provenance",
        "Saved data cannot prove actual checkpoint usage, input provenance, or absence of unrecorded state/velocity resets.",
    )
    r.unverified(
        "scientific_sampling_adequacy",
        "Numerical agreement alone does not establish equilibration, block independence, or adequacy of local perturbations.",
    )
