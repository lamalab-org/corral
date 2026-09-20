"""Reproduce Al velocity spectra and MD diagnostics from saved data only."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from scipy import signal
from scipy.ndimage import gaussian_filter1d

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


class UnsupportedEstimator(UnsupportedEvidence):
    """A supported numerical check cannot reproduce the chosen estimator."""


def _number(value):
    return float(finite_array(value, shape=()))


def _time(atoms):
    return _number(
        atoms.info["time_fs"]
        if "time_fs" in atoms.info
        else 1000 * atoms.info["time_ps"]
    )


def _energy(atoms):
    for key in ("potential_energy_eV", "potential_energy", "energy"):
        if key in atoms.info:
            return _number(atoms.info[key])
    cached = getattr(getattr(atoms, "calc", None), "results", {})
    if "energy" not in cached:
        raise EvidenceError(
            "Missing stored potential energy; calculator evaluation is prohibited"
        )
    return _number(cached["energy"])


def _area(y, x):
    return float(np.trapezoid(y, x))


UNIFORM_ESTIMATORS = {"periodogram", "fft", "welch", "vacf", "multitaper"}


def _spectral_integral(y, frequency, config):
    """Apply a declared quadrature consistently to saved and reproduced spectra."""
    rule = config.get("integration_rule", "trapezoid")
    if rule == "trapezoid":
        return _area(y, frequency)
    if rule == "bin_sum":
        widths = None
        if "bin_edges_THz" in config:
            edges = finite_array(config["bin_edges_THz"], shape=(len(frequency) + 1,))
            widths = np.diff(edges)
            if (
                np.any(widths <= 0)
                or np.any(frequency < edges[:-1])
                or np.any(frequency > edges[1:])
            ):
                raise EvidenceError(
                    "Spectral bin edges must increase and contain their frequencies"
                )
        if "bin_widths_THz" in config:
            declared = finite_array(config["bin_widths_THz"], shape=(len(frequency),))
            if np.any(declared <= 0):
                raise EvidenceError("Spectral bin widths must be positive")
            if widths is not None and not close(
                widths, declared, rtol=1e-7, atol=1e-10
            ):
                raise EvidenceError("Spectral bin widths disagree with bin edges")
            widths = declared
        if widths is None:
            raise EvidenceError("bin_sum needs spectral bin widths or bin edges")
        return float(np.dot(y, widths))
    if not isinstance(rule, str) or not rule.strip():
        raise EvidenceError("The integration rule must be a nonempty method identifier")
    raise UnsupportedEstimator(
        f"Declared spectral integration rule {rule!r} needs independent review"
    )


def _spectrum(velocity, masses, dt, config):
    """Recompute a declared estimator; never call submitted analysis code."""
    method = config["estimator"].lower()
    if method not in UNIFORM_ESTIMATORS:
        raise UnsupportedEstimator(
            f"Unsupported estimator {method}; retain diagnostic credit"
        )
    detrend = config["detrend"]
    if detrend not in (False, "constant", "linear"):
        raise UnsupportedEstimator("Supported detrending: false, constant, linear")
    data = finite_array(velocity, ndim=3).copy()
    n = len(data)
    nperseg = int(config["nperseg"]) if method == "welch" else n
    nfft = int(config.get("nfft", nperseg))
    if nfft > 1048576:
        raise UnsupportedEstimator(
            "FFT exceeds the documented artifact-analysis resource bound"
        )
    if nperseg < 4 or nperseg > n or nfft < nperseg:
        raise EvidenceError("Invalid transform or Welch segment length")
    if (
        type(config["mass_weighted"]) is not bool
        or type(config["remove_com"]) is not bool
    ):
        raise EvidenceError("mass_weighted and remove_com must be booleans")
    if config["remove_com"]:
        data -= (
            np.sum(data * masses[None, :, None], axis=1, keepdims=True) / masses.sum()
        )
    if config["mass_weighted"]:
        data *= np.sqrt(masses)[None, :, None]
    window = (
        config.get("window", "boxcar") if method == "multitaper" else config["window"]
    )
    if isinstance(window, list):
        window = tuple(window)
    if method == "vacf" and (window != "boxcar" or detrend == "linear" or nfft != n):
        raise UnsupportedEstimator(
            "Full biased VACF supports boxcar, false/constant detrending and nfft=N"
        )
    try:
        signal.get_window(window, nperseg)
    except (ValueError, TypeError) as exc:
        raise UnsupportedEstimator(f"Unsupported window: {window}") from exc
    fs = 1000.0 / dt  # samples/ps = THz
    power = np.zeros(nfft // 2 + 1)
    tapers = None
    if method == "multitaper":
        bandwidth = _number(config["time_bandwidth"])
        count = config["n_tapers"]
        if not 0 < bandwidth < n / 2 or type(count) is not int or not 1 <= count <= n:
            raise EvidenceError("Invalid DPSS time bandwidth or taper count")
        if n * count > 1048576:
            raise UnsupportedEstimator(
                "DPSS array exceeds automatic analysis resource bound"
            )
        tapers = signal.windows.dpss(n, bandwidth, Kmax=count, sym=False, norm=2)
    # Work one component at a time, bounding memory even for large grids.
    for column in data.reshape(n, -1).T:
        if method == "multitaper":
            centered = signal.detrend(column, type=detrend) if detrend else column
            frequency, tapered = signal.periodogram(
                centered[None, :] * tapers,
                fs=fs,
                window="boxcar",
                nfft=nfft,
                detrend=False,
                return_onesided=True,
                scaling="density",
                axis=-1,
            )
            # Dividing by taper energy gives every taper equal density weight.
            part = np.mean(tapered * n / np.sum(tapers**2, axis=1)[:, None], axis=0)
        elif method == "welch":
            overlap = int(config["noverlap"])
            if overlap < 0 or overlap >= nperseg:
                raise EvidenceError("Welch requires 0 <= noverlap < nperseg")
            frequency, part = signal.welch(
                column,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=overlap,
                nfft=nfft,
                detrend=detrend,
                return_onesided=True,
                scaling="density",
            )
        else:
            # Full biased VACF is Wiener-Khinchin-equivalent to this boxcar PSD.
            frequency, part = signal.periodogram(
                column,
                fs=fs,
                window=window,
                nfft=nfft,
                detrend=detrend,
                return_onesided=True,
                scaling="density",
            )
        power += part
    sigma = _number(config["smoothing_sigma_bins"])
    if sigma < 0:
        raise EvidenceError("Negative smoothing width")
    if sigma:
        power = gaussian_filter1d(power, sigma, mode="nearest")
    target = _number(config["normalization_area"])
    integral = _spectral_integral(power, frequency, config)
    if target <= 0 or integral <= 0 or not str(config["normalization_unit"]).strip():
        raise EvidenceError("Positive spectral area and declared units are required")
    resolution = fs / nperseg
    if method == "multitaper":
        resolution *= 2 * bandwidth
    return frequency, power * target / integral, resolution


def _spectral_close(f, y, expected_f, expected_y):
    return result_close(f, expected_f, atol=1e-08) and result_close(
        y, expected_y, atol=max(1e-10, float(np.max(expected_y)) * 1e-06)
    )


def _supported(condition):
    """Unsupported numerical methods are unverified, never claimed to fail physics."""

    def checked():
        try:
            return condition()
        except UnsupportedEstimator as exc:
            return None, str(exc)

    return checked


def evaluate(e: Evidence, r: Rubric) -> None:
    @lru_cache(None)
    def stage(which):
        names = (
            ("equilibration_trajectory", "nvt_trajectory", "equilibration")
            if which == "eq"
            else ("nve_trajectory", "production_trajectory", "trajectory")
        )
        frames = e.trajectory(*names)
        minimum = 2 if which == "eq" else 4
        if len(frames) < minimum:
            raise EvidenceError(f"Need at least {minimum} saved {which} states")
        for a in frames:
            if (
                len(a) != 64
                or set(a.get_chemical_symbols()) != {"Al"}
                or not np.all(a.pbc)
            ):
                raise EvidenceError("Expected periodic Al64 states")
            finite_array(a.positions, shape=(64, 3))
            finite_array(a.cell.array, shape=(3, 3))
            if not np.allclose(a.get_masses(), 26.9815385, atol=1e-3, rtol=0):
                raise EvidenceError("Unexpected Al masses")
        times = finite_array([_time(a) for a in frames])
        if np.any(np.diff(times) <= 0):
            raise EvidenceError("Saved states must have increasing global times")
        return frames, times

    @lru_cache(None)
    def values(which):
        frames, times = stage(which)
        masses = frames[0].get_masses()
        momenta = []
        for a in frames:
            if "momenta" not in a.arrays:
                break
            momenta.append(finite_array(a.arrays["momenta"], shape=(64, 3)))
        velocity = None
        if which == "prod" and "velocities" in e._artifacts:
            velocity = finite_array(e.array("velocities"), shape=(len(frames), 64, 3))
            unit = e.settings["velocity_unit"]
            factors = {
                "ASE": 1.0,
                "Angstrom/fs": 1 / units.fs,
                "Angstrom/ps": 1 / (1000 * units.fs),
            }
            if unit not in factors:
                raise EvidenceError("Unknown velocity unit")
            velocity = velocity * factors[unit]
            for i, a in enumerate(frames):
                if "momenta" in a.arrays and not close(
                    velocity[i] * masses[:, None], a.arrays["momenta"]
                ):
                    raise EvidenceError("Saved velocities contradict saved momenta")
        elif len(momenta) == len(frames):
            velocity = np.asarray(momenta) / masses[None, :, None]
        if velocity is None:
            raise EvidenceError(
                "Complete momenta or explicitly unit-labeled velocities are required"
            )
        dof = e.settings["md"]["temperature_dof"]
        if dof not in (189, 192):
            raise EvidenceError("temperature_dof must declare 189 or 192")
        kinetic = 0.5 * np.sum(velocity**2 * masses[None, :, None], axis=(1, 2))
        temperature = 2 * kinetic / (dof * units.kB)
        return times, velocity, kinetic, temperature

    @lru_cache(None)
    def energies(which):
        return finite_array([_energy(a) for a in stage(which)[0]])

    def geometry():
        initial = stage("eq")[0][0]
        gram = initial.cell.array @ initial.cell.array.T
        expected = np.full((3, 3), (2 * 4.05) ** 2)
        np.fill_diagonal(expected, 2 * (2 * 4.05) ** 2)
        if not close(np.abs(gram), expected, rtol=0, atol=1e-4):
            return False
        distances = np.sort(initial.get_all_distances(mic=True), axis=1)
        return close(
            distances[:, 1:13], np.full((64, 12), 4.05 / np.sqrt(2)), rtol=0, atol=1e-4
        ) and close(distances[:, 13:19], np.full((64, 6), 4.05), rtol=0, atol=1e-4)

    def fixed_cell():
        eq, _ = stage("eq")
        prod, _ = stage("prod")
        return all(
            close(a.cell.array, eq[0].cell.array, rtol=0, atol=1e-5)
            and close(a.get_masses(), eq[0].get_masses(), rtol=0, atol=1e-7)
            for a in eq + prod
        )

    def initialization():
        config = e.settings["md"]
        initial = stage("eq")[0][0]
        p = finite_array(initial.arrays["momenta"], shape=(64, 3))
        t = (
            2
            * (0.5 * np.sum(p**2 / initial.get_masses()[:, None]))
            / (config["temperature_dof"] * units.kB)
        )
        return (
            config["target_temperature_K"] == 300
            and config["initial_temperature_K"] == 300
            and type(config["random_seed"]) is int
            and config["remove_com_once"] is True
            and close(p.sum(axis=0), np.zeros(3), atol=1e-7)
            and abs(t - 300) <= 6 * 300 * np.sqrt(2 / 189)
            and abs(stage("eq")[1][0]) < 1e-6
            and 0
            < _number(config["timestep_fs"])
            <= min(np.min(np.diff(stage(k)[1])) for k in ("eq", "prod"))
            and all(
                isinstance(config[k], str) and bool(config[k].strip())
                for k in (
                    "thermostat",
                    "equilibration_integrator",
                    "production_integrator",
                )
            )
            and config["production_ensemble"].upper() == "NVE"
            and "mace-mp-0" in str(e.settings["model"]).lower()
            and isinstance(e.settings["model_settings"], dict)
        )

    def continuity():
        eq, eq_time = stage("eq")
        prod, prod_time = stage("prod")
        delta = (prod[0].positions - eq[-1].positions) @ np.linalg.inv(
            eq[-1].cell.array
        )
        delta -= np.round(delta)
        return (
            abs(eq_time[-1] - prod_time[0]) < 1e-6
            and close(eq[-1].cell.array, prod[0].cell.array, rtol=0, atol=1e-5)
            and close(delta @ eq[-1].cell.array, np.zeros((64, 3)), rtol=0, atol=1e-6)
            and close(values("eq")[1][-1], values("prod")[1][0], rtol=1e-5, atol=1e-8)
            and close(_energy(eq[-1]), _energy(prod[0]), rtol=1e-5, atol=1e-7)
        )

    r.check("initial_fcc_geometry", 8, geometry)
    r.check("fixed_cell_and_saved_states", 6, fixed_cell)
    r.check("initialization_and_recorded_settings", 6, initialization)
    r.check("equilibration_to_nve_continuity", 8, continuity)

    @lru_cache(None)
    def trace():
        return e.table("thermal_trace", "thermal_log", "md_log")

    def logged(columns, stages=("eq", "prod")):
        table = trace()
        aliases = {"eq": ["equilibration", "nvt"], "prod": ["production", "nve"]}
        for which in stages:
            part = table[table["stage"].str.lower().isin(aliases[which])]
            times, _, kinetic, temperature = values(which)
            stored_times = finite_array(part["time_fs"])
            if np.any(np.diff(stored_times) <= 0):
                return False
            indices = np.searchsorted(stored_times, times)
            if np.any(indices >= len(stored_times)) or not close(
                times, stored_times[indices], rtol=0, atol=1e-6
            ):
                return False
            quantities = {"temperature_K": temperature, "kinetic_energy_eV": kinetic}
            if any("potential" in c or "total" in c for c in columns):
                quantities.update(
                    potential_energy_eV=energies(which),
                    total_energy_eV=energies(which) + kinetic,
                )
            if not all(
                close(
                    quantities[c], finite_array(part[c])[indices], rtol=1e-4, atol=1e-7
                )
                for c in columns
            ):
                return False
        return True

    def production_temperature():
        t = values("prod")[3]
        claims = e.results["production"]
        return result_close(t.mean(), claims["temperature_mean_K"], atol=0.0001) and (
            "temperature_std_K" not in claims
            or result_close(t.std(), claims["temperature_std_K"], atol=0.0001)
        )

    def energy_diagnostics():
        times, _, kinetic, _ = values("prod")
        energy = energies("prod") + kinetic
        claims = e.results["production"]
        slope = np.polyfit((times - times[0]) / 1000, energy, 1)[0]
        return logged(["potential_energy_eV", "total_energy_eV"]) and all(
            result_close(value, claims[key], atol=1e-07)
            for key, value in [
                ("total_energy_mean_eV", energy.mean()),
                ("total_energy_range_eV", np.ptp(energy)),
                ("energy_drift_eV_per_ps", slope),
            ]
            if key in claims
        )

    def equilibration_diagnostics():
        times, _, _, t = values("eq")
        cut = len(t) // 2
        claims = e.results.get("equilibration", {})
        return logged(["temperature_K"], stages=("eq",)) and all(
            result_close(value, claims[key], atol=0.0001)
            for key, value in [
                ("temperature_first_half_mean_K", t[:cut].mean()),
                ("temperature_second_half_mean_K", t[cut:].mean()),
                ("temperature_std_K", t.std()),
                (
                    "temperature_drift_K_per_ps",
                    np.polyfit((times - times[0]) / 1000, t, 1)[0],
                ),
            ]
            if key in claims
        )

    r.check(
        "kinetic_temperature_trace",
        6,
        lambda: logged(["temperature_K", "kinetic_energy_eV"]),
    )
    r.check(
        "energy_trace_and_drift",
        8,
        energy_diagnostics,
        "Reproduce the energy trace and supplied diagnostics without requiring a particular drift magnitude.",
    )
    r.check("production_temperature_statistics", 3, production_temperature)
    r.check(
        "production_temperature_sanity",
        2,
        lambda: scientific_screen(
            abs(values("prod")[3].mean() - 300)
            <= 6 * 300 * np.sqrt(2 / e.settings["md"]["temperature_dof"]),
            "Production temperature evidence is inconclusive for the near-300 K target",
        ),
    )
    r.check("equilibration_thermal_diagnostics", 5, equilibration_diagnostics)

    @lru_cache(None)
    def submitted_spectrum():
        table = e.table("spectrum", "vdos")
        return finite_array(table["frequency_THz"]), finite_array(table["vdos"])

    @lru_cache(None)
    def dt():
        step = np.diff(stage("prod")[1])
        if not np.allclose(step, step[0], rtol=1e-6, atol=1e-8):
            estimator = e.settings["spectrum"]["estimator"]
            if not isinstance(estimator, str) or not estimator.strip():
                raise EvidenceError(
                    "A nonempty spectral estimator identifier is required"
                )
            if estimator.lower() not in UNIFORM_ESTIMATORS:
                raise UnsupportedEstimator(
                    f"Nonuniform sampling under declared estimator {estimator!r} needs independent review"
                )
            raise EvidenceError(
                "The declared FFT-based spectral estimator requires uniform saved time intervals"
            )
        return float(step[0])

    @lru_cache(None)
    def reproduced():
        return _spectrum(
            values("prod")[1],
            stage("prod")[0][0].get_masses(),
            dt(),
            e.settings["spectrum"],
        )

    def normalization():
        f, y = submitted_spectrum()
        config = e.settings["spectrum"]
        target = _number(config["normalization_area"])
        return (
            len(f) >= 3
            and f[0] >= 0
            and np.all(np.diff(f) > 0)
            and np.all(y >= 0)
            and target > 0
            and result_close(_spectral_integral(y, f, config), target, atol=1e-07)
        )

    def sampling_metadata():
        f, _ = submitted_spectrum()
        claims = e.results["spectrum"]
        return (
            result_close(f[[0, -1]], claims["usable_frequency_range_THz"], atol=1e-07)
            and f[0] >= 0
            and result_close(dt(), claims["sampling_interval_fs"], atol=1e-07)
            and result_close(500 / dt(), claims["nyquist_THz"], atol=1e-07)
            and f[-1] <= 500 / dt() + 1e-8
        )

    def conventions():
        config = e.settings["spectrum"]
        claims = e.results["spectrum"]
        return (
            result_close(
                config["normalization_area"], claims["normalization_area"], atol=1e-07
            )
            and isinstance(config["normalization_unit"], str)
            and bool(config["normalization_unit"].strip())
            and config["normalization_unit"] == claims["normalization_unit"]
        )

    r.check("finite_nonnegative_normalized_spectrum", 6, normalization)
    r.check("sampling_frequency_range_and_nyquist", 8, sampling_metadata)
    r.check(
        "velocity_spectrum_reproduction",
        18,
        _supported(lambda: _spectral_close(*submitted_spectrum(), *reproduced()[:2])),
    )
    r.check("spectral_normalization_conventions", 3, conventions)
    r.check(
        "spectral_resolution_and_conventions",
        3,
        _supported(
            lambda: result_close(
                reproduced()[2],
                e.results["spectrum"]["frequency_resolution_THz"],
                atol=1e-07,
            )
        ),
    )

    r.unverified(
        "execution_provenance",
        "Saved states check consistency, not which model actually ran, Maxwell-Boltzmann random generation, thermostat execution, or absence of resets between saved frames.",
    )
    r.unverified(
        "scientific_sampling_adequacy",
        "Truthful numerical diagnostics alone do not establish equilibration sufficiency, acceptable NVE drift, or adequate finite sampling.",
    )
