"""Task 6 uses saved numerical evidence, including adversarial inconsistencies."""

import json
import pickle

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.score import check_level2_workflow
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.task_6 import evaluate
from scipy import signal


def _write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def _read(path):
    return json.loads(path.read_text())


def _score(path):
    r = Rubric(6)
    evaluate(Evidence(path), r)
    return r


def _check(rubric, name):
    return next(c for c in rubric.checks if c["name"] == name)


def test_off_target_production_temperature_requires_independent_review(submission):
    path = submission.parent / "prod.json"
    frames = _read(path)
    for frame in frames:
        frame["momenta"] = (np.asarray(frame["momenta"]) * 0.1).tolist()
    _write(path, frames)
    assert (
        _check(_score(submission), "production_temperature_sanity")["status"]
        == "unverified"
    )


def test_energy_drift_with_stale_evidence_loses_diagnostic_credit(submission):
    path = submission.parent / "prod.json"
    frames = _read(path)
    for i, frame in enumerate(frames):
        frame["energy"] += i
    _write(path, frames)
    assert _check(_score(submission), "energy_trace_and_drift")["status"] == "failed"


@pytest.mark.parametrize("increment_eV", [-0.005, 0.005, 1.0])
def test_truthfully_assessed_drift_keeps_full_credit(submission, increment_eV):
    directory = submission.parent
    frames = _read(directory / "prod.json")
    trace = _read(directory / "trace.json")
    production = [row for row in trace if row["stage"] == "production"]
    total_energy = []
    for index, (frame, row) in enumerate(zip(frames, production, strict=True)):
        # Keep the shared equilibration/production boundary unchanged.
        frame["energy"] += increment_eV * index
        momenta = np.asarray(frame["momenta"])
        masses = np.asarray(frame["masses"])
        kinetic = 0.5 * np.sum(momenta**2 / masses[:, None])
        row["potential_energy_eV"] = frame["energy"]
        row["total_energy_eV"] = float(frame["energy"] + kinetic)
        total_energy.append(row["total_energy_eV"])
    times = np.array([frame["time_fs"] for frame in frames]) / 1000
    energy = np.asarray(total_energy)
    manifest = _read(submission)
    manifest["results"]["production"].update(
        total_energy_mean_eV=float(energy.mean()),
        total_energy_range_eV=float(np.ptp(energy)),
        energy_drift_eV_per_ps=float(np.polyfit(times - times[0], energy, 1)[0]),
    )
    _write(directory / "prod.json", frames)
    _write(directory / "trace.json", trace)
    _write(submission, manifest)
    grader = check_level2_workflow(6)
    report = grader.evaluate(submission)
    assert report["status"] == "complete"
    assert report["score"] == pytest.approx(1), report
    assert sum(check["points"] for check in report["checks"]) == 100
    assert (
        next(
            check
            for check in report["checks"]
            if check["name"] == "energy_trace_and_drift"
        )["points"]
        == 8
    )

    # Removing the acceptance threshold must not hide incorrect diagnostics.
    manifest["results"]["production"]["energy_drift_eV_per_ps"] = 0
    _write(submission, manifest)
    report = grader.evaluate(submission)
    assert report["score"] == pytest.approx(0.92), report
    assert (
        next(
            check
            for check in report["checks"]
            if check["name"] == "energy_trace_and_drift"
        )["status"]
        == "failed"
    )


def _reference(velocities, dt, config):
    method = config["estimator"]
    kwargs = {
        "fs": 1000 / dt,
        "axis": 0,
        "detrend": config["detrend"],
        "window": config["window"],
        "scaling": "density",
    }
    if "nfft" in config:
        kwargs["nfft"] = config["nfft"]
    if method == "welch":
        f, y = signal.welch(
            velocities, nperseg=config["nperseg"], noverlap=config["noverlap"], **kwargs
        )
    else:
        f, y = signal.periodogram(velocities, **kwargs)
    y = np.sum(y, axis=(1, 2))
    return f, y / np.trapezoid(y, f) * config["normalization_area"]


def _replace_spectra(path, config):
    manifest = _read(path)
    settings = _read(path.parent / "settings.json")
    settings["spectrum"] = config
    _write(path.parent / "settings.json", settings)
    frames = _read(path.parent / "prod.json")
    velocities = (
        np.array([a["momenta"] for a in frames])
        / np.array(frames[0]["masses"])[None, :, None]
    )
    dt = frames[1]["time_fs"] - frames[0]["time_fs"]
    f, y = _reference(velocities, dt, config)
    _write(
        path.parent / "spectrum.json", {"frequency_THz": f.tolist(), "vdos": y.tolist()}
    )
    manifest["results"]["spectrum"].update(
        frequency_resolution_THz=1000
        / (dt * (config["nperseg"] if config["estimator"] == "welch" else len(frames))),
        usable_frequency_range_THz=[float(f[0]), float(f[-1])],
        normalization_area=config["normalization_area"],
        normalization_unit=config["normalization_unit"],
    )
    comparisons = []
    for start, stop in [
        (0, len(velocities) // 2),
        (len(velocities) // 2, len(velocities)),
    ]:
        f, y = _reference(velocities[start:stop], dt, config)
        comparisons.append(
            {
                "start": start,
                "stop": stop,
                "settings": {},
                "frequency_THz": f.tolist(),
                "vdos": y.tolist(),
            }
        )
    _write(path.parent / "sensitivity.json", {"comparisons": comparisons})
    fa, fb = [np.array(c["frequency_THz"]) for c in comparisons]
    ya, yb = [np.array(c["vdos"]) for c in comparisons]
    lo, hi = max(fa[0], fb[0]), min(fa[-1], fb[-1])
    grid = np.unique(np.concatenate((fa, fb)))
    a, b = np.interp(grid, fa, ya), np.interp(grid, fb, yb)
    a, b = a / np.trapezoid(a, grid), b / np.trapezoid(b, grid)
    manifest["results"]["stability"] = {
        "l1_distance": float(np.trapezoid(np.abs(a - b), grid)),
        "frequency_range_THz": [float(lo), float(hi)],
    }
    _write(path, manifest)


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05).repeat((4, 4, 4))
    mass = atoms.get_masses()
    rng = np.random.default_rng(74)
    p0 = rng.normal(size=(64, 3))
    p0 -= p0.mean(axis=0)
    p0 *= np.sqrt(189 * units.kB * 300 / np.sum(p0**2 / mass[:, None]))
    dt, count = 5.0, 128
    production_time = 1000 + dt * np.arange(count)
    aa = rng.normal(size=(64, 3))
    bb = rng.normal(size=(64, 3))
    aa -= aa.mean(axis=0)
    bb -= bb.mean(axis=0)
    # Smooth stationary harmonic velocities with frequencies in Al's acoustic range.
    phase = production_time[:, None, None] / 1000
    velocity = aa * np.cos(2 * np.pi * 6 * phase) + bb * np.sin(2 * np.pi * 9.5 * phase)
    velocity *= np.sqrt(
        189
        * units.kB
        * 300
        / np.mean(np.sum(velocity**2 * mass[None, :, None], axis=(1, 2)))
    )
    displacement = np.cumsum(velocity, axis=0) * dt * units.fs
    displacement -= displacement[0]
    eq_times = np.array([0, 250, 500, 750, 1000])
    equil, prod, trace = [], [], []
    for i, time in enumerate(eq_times):
        fraction = i / (len(eq_times) - 1)
        p = (1 - fraction) * p0 + fraction * velocity[0] * mass[:, None]
        equil.append(
            {
                "symbols": ["Al"] * 64,
                "positions": atoms.positions.tolist(),
                "momenta": p.tolist(),
                "masses": mass.tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": [True] * 3,
                "time_fs": float(time),
                "energy": -220.0,
            }
        )
    for i, time in enumerate(production_time):
        kinetic = np.sum(velocity[i] ** 2 * mass[:, None]) / 2
        prod.append(
            {
                "symbols": ["Al"] * 64,
                "positions": (atoms.positions + displacement[i]).tolist(),
                "momenta": (velocity[i] * mass[:, None]).tolist(),
                "masses": mass.tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": [True] * 3,
                "time_fs": float(time),
                "energy": float(
                    -215.0 - kinetic + 0.001 * (time - production_time[0]) / 1000
                ),
            }
        )
    equil[-1]["energy"] = prod[0]["energy"]
    temperatures = {}
    for label, rows in [("equilibration", equil), ("production", prod)]:
        values = []
        for frame in rows:
            kinetic = float(
                0.5 * np.sum(np.array(frame["momenta"]) ** 2 / mass[:, None])
            )
            t = 2 * kinetic / (189 * units.kB)
            values.append(t)
            trace.append(
                {
                    "stage": label,
                    "time_fs": frame["time_fs"],
                    "temperature_K": t,
                    "kinetic_energy_eV": kinetic,
                    "potential_energy_eV": frame["energy"],
                    "total_energy_eV": kinetic + frame["energy"],
                }
            )
        temperatures[label] = np.array(values)
    paths = {
        "equilibration_trajectory": _write(tmp_path / "eq.json", equil),
        "nve_trajectory": _write(tmp_path / "prod.json", prod),
        "thermal_trace": _write(tmp_path / "trace.json", trace),
        "spectrum": str(tmp_path / "spectrum.json"),
        "sensitivity": str(tmp_path / "sensitivity.json"),
    }
    eq_t, prod_t = temperatures["equilibration"], temperatures["production"]
    energy = np.array(
        [row["total_energy_eV"] for row in trace if row["stage"] == "production"]
    )
    results = {
        "equilibration": {
            "temperature_first_half_mean_K": float(eq_t[:2].mean()),
            "temperature_second_half_mean_K": float(eq_t[2:].mean()),
            "temperature_std_K": float(eq_t.std()),
            "temperature_drift_K_per_ps": float(
                np.polyfit(eq_times / 1000, eq_t, 1)[0]
            ),
        },
        "production": {
            "temperature_mean_K": float(prod_t.mean()),
            "temperature_std_K": float(prod_t.std()),
            "total_energy_mean_eV": float(energy.mean()),
            "total_energy_range_eV": float(np.ptp(energy)),
            "energy_drift_eV_per_ps": float(
                np.polyfit((production_time - production_time[0]) / 1000, energy, 1)[0]
            ),
        },
        "spectrum": {"sampling_interval_fs": dt, "nyquist_THz": 500 / dt},
    }
    settings = {
        "md": {
            "timestep_fs": 1,
            "temperature_dof": 189,
            "target_temperature_K": 300,
            "initial_temperature_K": 300,
            "random_seed": 74,
            "remove_com_once": True,
            "thermostat": "Langevin",
            "equilibration_integrator": "Langevin",
            "production_integrator": "VelocityVerlet",
            "production_ensemble": "NVE",
        },
        "model": "MACE-MP-0",
        "model_settings": {"dtype": "float64"},
    }
    report = tmp_path / "report.md"
    report.write_text(
        "Assess finite sampling, thermal equilibration and energy drift using the saved traces."
    )
    script = tmp_path / "run.py"
    script.write_text("raise RuntimeError('Scoring must never execute this script')\n")
    path = tmp_path / "manifest.json"
    _write(
        path,
        {
            "artifacts": paths,
            "results": results,
            "settings": _write(tmp_path / "settings.json", settings),
            "report": str(report),
            "scripts": [str(script)],
        },
    )
    _replace_spectra(
        path,
        {
            "estimator": "periodogram",
            "detrend": "constant",
            "window": "hann",
            "mass_weighted": False,
            "remove_com": False,
            "normalization_area": 1,
            "normalization_unit": "1/THz",
            "smoothing_sigma_bins": 0,
        },
    )
    return path


def test_consistent_oscillatory_evidence_gets_full_credit_without_execution(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Attempted execution or deserialization")

    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(c["points"] for c in result.checks) == 90
    assert _check(result, "execution_provenance")["status"] == "unverified"


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ("dt", "sampling_frequency_range_and_nyquist"),
        ("frequency", "velocity_spectrum_reproduction"),
        ("normalization", "finite_nonnegative_normalized_spectrum"),
        ("energy", "energy_trace_and_drift"),
        ("momenta", "kinetic_temperature_trace"),
        ("boundary", "equilibration_to_nve_continuity"),
        ("cell", "fixed_cell_and_saved_states"),
        ("com", "initialization_and_recorded_settings"),
        ("geometry", "initial_fcc_geometry"),
        ("resolution", "spectral_resolution_and_conventions"),
        ("temperature", "production_temperature_statistics"),
    ],
)
def test_inconsistent_evidence_is_rejected(submission, change, failed):
    directory = submission.parent
    manifest = _read(submission)
    if change in {"dt", "stability", "resolution", "temperature"}:
        if change == "dt":
            manifest["results"]["spectrum"]["sampling_interval_fs"] = 10
        elif change == "stability":
            manifest["results"]["stability"]["l1_distance"] += 0.5
        elif change == "temperature":
            manifest["results"]["production"]["temperature_mean_K"] = 30000
        else:
            manifest["results"]["spectrum"]["frequency_resolution_THz"] *= 2
        _write(submission, manifest)
    elif change in {"frequency", "normalization"}:
        data = _read(directory / "spectrum.json")
        data["frequency_THz" if change == "frequency" else "vdos"] = [
            2 * x for x in data["frequency_THz" if change == "frequency" else "vdos"]
        ]
        _write(directory / "spectrum.json", data)
    elif change == "sensitivity":
        data = _read(directory / "sensitivity.json")
        data["comparisons"][0]["vdos"][2] += 0.2
        _write(directory / "sensitivity.json", data)
    else:
        name = "eq.json" if change in {"com", "geometry"} else "prod.json"
        data = _read(directory / name)
        if change == "energy":
            data[4]["energy"] += 10
        elif change == "momenta":
            data[7]["momenta"][0][0] += 3
        elif change == "boundary":
            data[0]["positions"][0][0] += 0.1
        elif change == "cell":
            data[-1]["cell"][0][1] += 0.1
        elif change == "geometry":
            data[0]["positions"][0][0] += 0.5
        else:
            data[0]["momenta"][0][0] += 1
        _write(directory / name, data)
    result = _score(submission)
    assert _check(result, failed)["status"] == "failed", result.checks
    assert sum(c["points"] for c in result.checks) == 90


@pytest.mark.parametrize("method", ["welch", "vacf"])
def test_alternative_estimators_receive_full_credit(submission, method):
    config = _read(submission.parent / "settings.json")["spectrum"]
    config["estimator"] = method
    if method == "welch":
        config.update(nperseg=32, noverlap=16)
    else:
        config["window"] = "boxcar"
    _replace_spectra(submission, config)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("ASE", 1), ("Angstrom/fs", units.fs), ("Angstrom/ps", 1000 * units.fs)],
)
def test_velocity_units_and_separate_velocity_evidence(submission, unit, factor):
    directory = submission.parent
    frames = _read(directory / "prod.json")
    velocity = (
        np.array([a.pop("momenta") for a in frames])
        / np.array(frames[0]["masses"])[None, :, None]
    )
    _write(directory / "prod.json", frames)
    _write(directory / "velocities.json", (velocity * factor).tolist())
    manifest = _read(submission)
    manifest["artifacts"]["velocities"] = str(directory / "velocities.json")
    _write(submission, manifest)
    settings = _read(directory / "settings.json")
    settings["velocity_unit"] = unit
    _write(directory / "settings.json", settings)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_unsupported_estimator_preserves_independent_points(submission):
    settings = _read(submission.parent / "settings.json")
    settings["spectrum"]["estimator"] = "maximum_entropy"
    _write(submission.parent / "settings.json", settings)
    result = _score(submission)
    assert _check(result, "velocity_spectrum_reproduction")["status"] == "unverified"

    assert _check(result, "sampling_frequency_range_and_nyquist")["status"] == "passed"
    assert _check(result, "energy_trace_and_drift")["status"] == "passed"
    assert _check(result, "spectral_normalization_conventions")["status"] == "passed"
    assert result.score is None


def test_multitaper_spectrum_is_reproduced_independently(submission):
    config = _read(submission.parent / "settings.json")
    config["spectrum"].update(estimator="multitaper", time_bandwidth=3.5, n_tapers=5)
    del config["spectrum"]["window"]
    frames = _read(submission.parent / "prod.json")
    velocity = (
        np.array([a["momenta"] for a in frames])
        / np.array(frames[0]["masses"])[None, :, None]
    )
    n = len(velocity)
    dt = frames[1]["time_fs"] - frames[0]["time_fs"]
    # Independent vectorized transform; equal taper energy weighting is checked
    # against the implementation's per-component spectral calculation.
    tapers = signal.windows.dpss(n, 3.5, Kmax=5, sym=False, norm=2)
    centered = velocity - velocity.mean(axis=0)
    transform = np.fft.rfft(centered[None] * tapers[:, :, None, None], axis=1)
    power = np.abs(transform) ** 2 / np.sum(tapers**2, axis=1)[:, None, None, None]
    power[:, 1:-1] *= 2
    power = power.mean(axis=0).sum(axis=(1, 2))
    frequency = np.fft.rfftfreq(n, d=dt / 1000)
    power /= np.trapezoid(power, frequency)
    _write(
        submission.parent / "spectrum.json",
        {"frequency_THz": frequency.tolist(), "vdos": power.tolist()},
    )
    _write(submission.parent / "settings.json", config)
    manifest = _read(submission)
    manifest["results"]["spectrum"]["frequency_resolution_THz"] = 7 * 1000 / (dt * n)
    _write(submission, manifest)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    power[3] *= 2
    power /= np.trapezoid(power, frequency)
    _write(
        submission.parent / "spectrum.json",
        {"frequency_THz": frequency.tolist(), "vdos": power.tolist()},
    )
    assert (
        _check(_score(submission), "velocity_spectrum_reproduction")["status"]
        == "failed"
    )


def test_thermal_evidence_does_not_require_half_or_ols_reporting(submission):
    manifest = _read(submission)
    del manifest["results"]["equilibration"]
    manifest["results"]["production"] = {
        "temperature_mean_K": manifest["results"]["production"]["temperature_mean_K"]
    }
    _write(submission, manifest)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks


def test_missing_artifacts_keep_all_rubric_points(submission):
    (submission.parent / "prod.json").unlink()
    result = _score(submission)
    assert sum(c["points"] for c in result.checks) == 90
    assert _check(result, "initial_fcc_geometry")["status"] == "passed"
    assert _check(result, "equilibration_thermal_diagnostics")["status"] == "passed"
    assert _check(result, "velocity_spectrum_reproduction")["status"] == "failed"


def test_empty_submission_has_zero_points(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, {"results": {}, "artifacts": {}})
    result = _score(path)
    assert result.score == 0
    assert sum(c["points"] for c in result.checks) == 90


def test_inconsistent_boundary_momenta_are_rejected(submission):
    path = submission.parent / "eq.json"
    data = _read(path)
    data[-1]["momenta"][0][0] += 0.2
    _write(path, data)
    assert (
        _check(_score(submission), "equilibration_to_nve_continuity")["status"]
        == "failed"
    )


def test_nonuniform_saved_times_do_not_silently_use_declared_dt(submission):
    path = submission.parent / "prod.json"
    data = _read(path)
    data[4]["time_fs"] += 0.2
    _write(path, data)
    result = _score(submission)
    assert _check(result, "sampling_frequency_range_and_nyquist")["status"] == "failed"
    assert _check(result, "velocity_spectrum_reproduction")["status"] == "failed"


def test_nonuniform_lomb_scargle_retains_evidence_for_review(submission):
    frames = _read(submission.parent / "prod.json")
    original_time = frames[4]["time_fs"]
    frames[4]["time_fs"] += 0.2
    times = np.array([frame["time_fs"] for frame in frames])
    velocities = (
        np.array([frame["momenta"] for frame in frames])
        / np.array(frames[0]["masses"])[None, :, None]
    )
    frequency = np.linspace(0.25, 80, 160)
    power = sum(
        signal.lombscargle(
            (times - times[0]) / 1000, column, 2 * np.pi * frequency, precenter=True
        )
        for column in velocities.reshape(len(times), -1).T
    )
    power /= np.trapezoid(power, frequency)
    _write(submission.parent / "prod.json", frames)
    trace = _read(submission.parent / "trace.json")
    for row in trace:
        if row["stage"] == "production" and row["time_fs"] == original_time:
            row["time_fs"] += 0.2
    _write(submission.parent / "trace.json", trace)
    _write(
        submission.parent / "spectrum.json",
        {"frequency_THz": frequency.tolist(), "vdos": power.tolist()},
    )
    settings = _read(submission.parent / "settings.json")
    settings["spectrum"].update(
        estimator="lomb_scargle",
        sampling="nonuniform",
        detrend="constant",
        window="none",
    )
    _write(submission.parent / "settings.json", settings)
    manifest = _read(submission)
    claims = manifest["results"]["spectrum"]
    del claims["sampling_interval_fs"], claims["nyquist_THz"]
    claims.update(
        sampling_intervals_fs=np.diff(times).tolist(),
        usable_frequency_range_THz=frequency[[0, -1]].tolist(),
        frequency_resolution_THz=1000 / (times[-1] - times[0]),
    )
    manifest["results"]["production"].pop("energy_drift_eV_per_ps")
    _write(submission, manifest)
    result = _score(submission)
    assert result.score is None
    assert not [check for check in result.checks if check["status"] == "failed"]
    for name in (
        "sampling_frequency_range_and_nyquist",
        "velocity_spectrum_reproduction",
        "spectral_resolution_and_conventions",
    ):
        assert _check(result, name)["status"] == "unverified"
    for name in (
        "finite_nonnegative_normalized_spectrum",
        "kinetic_temperature_trace",
        "energy_trace_and_drift",
    ):
        assert _check(result, name)["status"] == "passed"


@pytest.mark.parametrize("bins", ["widths", "edges", "both"])
def test_discrete_bin_normalization_is_reproduced(submission, bins):
    settings = _read(submission.parent / "settings.json")
    config = settings["spectrum"]
    config.update(detrend=False, window="boxcar")
    _replace_spectra(submission, config)
    spectrum = _read(submission.parent / "spectrum.json")
    frequency = np.array(spectrum["frequency_THz"])
    power = np.array(spectrum["vdos"])
    width = frequency[1] - frequency[0]
    config["integration_rule"] = "bin_sum"
    if bins in ("widths", "both"):
        config["bin_widths_THz"] = np.full(len(frequency), width).tolist()
    if bins in ("edges", "both"):
        config["bin_edges_THz"] = np.r_[frequency, frequency[-1] + width].tolist()
    power /= np.sum(power) * width
    assert not np.isclose(np.trapezoid(power, frequency), 1, atol=1e-7, rtol=1e-7)
    spectrum["vdos"] = power.tolist()
    _write(submission.parent / "spectrum.json", spectrum)
    _write(submission.parent / "settings.json", settings)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    spectrum["vdos"] = (2 * power).tolist()
    _write(submission.parent / "spectrum.json", spectrum)
    assert (
        _check(_score(submission), "finite_nonnegative_normalized_spectrum")["status"]
        == "failed"
    )


@pytest.mark.parametrize("corrupt", ["negative_width", "edge_mismatch", "missing_bins"])
def test_invalid_bin_quadrature_is_failed_evidence(submission, corrupt):
    settings = _read(submission.parent / "settings.json")
    frequency = np.array(_read(submission.parent / "spectrum.json")["frequency_THz"])
    width = frequency[1] - frequency[0]
    config = settings["spectrum"]
    config["integration_rule"] = "bin_sum"
    if corrupt == "negative_width":
        config["bin_widths_THz"] = np.full(len(frequency), -width).tolist()
    elif corrupt == "edge_mismatch":
        config["bin_widths_THz"] = np.full(len(frequency), 2 * width).tolist()
        config["bin_edges_THz"] = np.r_[frequency, frequency[-1] + width].tolist()
    _write(submission.parent / "settings.json", settings)
    result = _score(submission)
    assert (
        _check(result, "finite_nonnegative_normalized_spectrum")["status"] == "failed"
    )
    assert _check(result, "velocity_spectrum_reproduction")["status"] == "failed"


def test_other_quadrature_is_reviewed_without_hiding_invalid_spectrum(submission):
    settings = _read(submission.parent / "settings.json")
    settings["spectrum"]["integration_rule"] = "simpson"
    _write(submission.parent / "settings.json", settings)
    result = _score(submission)
    assert result.score is None
    assert (
        _check(result, "finite_nonnegative_normalized_spectrum")["status"]
        == "unverified"
    )
    data = _read(submission.parent / "spectrum.json")
    data["vdos"][0] = -1
    _write(submission.parent / "spectrum.json", data)
    assert (
        _check(_score(submission), "finite_nonnegative_normalized_spectrum")["status"]
        == "failed"
    )


def test_contradictory_velocity_and_momentum_evidence_is_rejected(submission):
    manifest = _read(submission)
    data = _read(submission.parent / "prod.json")
    velocity = (
        np.array([a["momenta"] for a in data])
        / np.array(data[0]["masses"])[None, :, None]
    )
    manifest["artifacts"]["velocities"] = _write(
        submission.parent / "velocities.json", (2 * velocity).tolist()
    )
    _write(submission, manifest)
    settings = _read(submission.parent / "settings.json")
    settings["velocity_unit"] = "ASE"
    _write(submission.parent / "settings.json", settings)
    assert _check(_score(submission), "kinetic_temperature_trace")["status"] == "failed"


def test_zero_padding_does_not_claim_improved_frequency_resolution(submission):
    config = _read(submission.parent / "settings.json")["spectrum"]
    config["nfft"] = 512
    _replace_spectra(submission, config)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    manifest = _read(submission)
    manifest["results"]["spectrum"]["frequency_resolution_THz"] = 1000 / (5 * 512)
    _write(submission, manifest)
    assert (
        _check(_score(submission), "spectral_resolution_and_conventions")["status"]
        == "failed"
    )


def test_equivalent_rotated_cell_and_translated_frames_are_accepted(submission):
    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    # A rotated system has the same scalar VDOS after Cartesian components are summed.
    for name in ("eq.json", "prod.json"):
        path = submission.parent / name
        data = _read(path)
        for frame in data:
            frame["cell"] = (np.array(frame["cell"]) @ rotation).tolist()
            frame["positions"] = (
                np.array(frame["positions"]) @ rotation + np.array([0.7, 1.4, -0.2])
            ).tolist()
            frame["momenta"] = (np.array(frame["momenta"]) @ rotation).tolist()
        _write(path, data)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
