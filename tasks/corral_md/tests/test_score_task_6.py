"""Independent harmonic signals exercise log analysis without running MACE."""

import copy
import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import check_level1_workflow, check_level2_workflow
from corral_md.scoring.evaluator import PendingReviewError
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.task_6 import Task6Scorer, spectral_summaries
from corral_md.scoring.verification import ModalVerifier
from scipy import signal
from scipy.integrate import cumulative_trapezoid


def write_log(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05).repeat((4, 4, 4))
    rng = np.random.default_rng(42)
    omega = 2 * np.pi * rng.choice([4, 6, 8, 10], size=(32, 3)) / 1000
    phase = rng.uniform(0, 2 * np.pi, size=(32, 3))
    amplitude = np.sqrt(2 * units.kB * 300 / atoms.get_masses()[0])

    def frame(time, step):
        angle = omega * time + phase
        v = amplitude * np.cos(angle)
        v = np.concatenate([v, -v])
        displacement = amplitude * units.fs / omega * (np.sin(angle) - np.sin(phase))
        position = atoms.positions + np.concatenate([displacement, -displacement])
        p = v * atoms.get_masses()[:, None]
        kinetic = (p**2 / atoms.get_masses()[:, None]).sum() / 2
        return {
            "event": "sample",
            "step": step,
            "time_fs": time,
            "cell_A": atoms.cell.array.tolist(),
            "positions_A": position.tolist(),
            "momenta": p.tolist(),
            "potential_energy_eV": -220 - kinetic,
        }

    records = [
        {
            "event": "start",
            "model_sha256": PINNED_TEACHER_SHA256,
            "random_seed": 42,
            "numbers": atoms.numbers.tolist(),
            "masses_amu": atoms.get_masses().tolist(),
            "units": {
                "time": "fs",
                "position": "Angstrom",
                "energy": "eV",
                "momenta": "ASE",
            },
        }
    ]
    stages = {}
    for name, offset, steps, interval in (
        ("equilibration", 0, 4000, 100),
        ("production", 4000, 16000, 10),
    ):
        records.append(
            {
                "event": "stage",
                "name": name,
                "steps": steps,
                "interval": interval,
                "timestep_fs": 1,
                "integrator": "Langevin" if offset == 0 else "VelocityVerlet",
                "target_temperature_K": 300 if offset == 0 else None,
            }
        )
        rows = [frame(offset + step, step) for step in range(0, steps + 1, interval)]
        stages[name] = rows
        records.extend(rows)
        records.append({"event": "end", "name": name, "steps": steps})
    write_log(tmp_path / "run.jsonl", records)
    temperatures = {}
    for name, rows in stages.items():
        p = np.array([row["momenta"] for row in rows])
        temperatures[name] = (p**2 / atoms.get_masses()[None, :, None]).sum(
            axis=(1, 2)
        ) / (192 * units.kB)
    velocity = (
        np.array([row["momenta"] for row in stages["production"]])
        / atoms.get_masses()[None, :, None]
    )
    velocity -= velocity.mean(axis=1, keepdims=True)
    frequency, power = signal.welch(
        velocity,
        fs=100,
        window="hann",
        nperseg=400,
        noverlap=200,
        nfft=400,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    power = power.sum(axis=(1, 2))
    power /= np.trapezoid(power, frequency)
    mean = np.trapezoid(frequency * power, frequency)
    std = np.sqrt(np.trapezoid((frequency - mean) ** 2 * power, frequency))
    percentile = np.interp(
        0.95, cumulative_trapezoid(power, frequency, initial=0), frequency
    )
    results = {
        "equilibration_temperature_K": temperatures["equilibration"][20:].mean(),
        "production_temperature_K": temperatures["production"].mean(),
        "sampling_interval_fs": 10,
        "frequency_resolution_THz": 0.5,
        "energy_drift_eV_atom_ps": 0,
        "energy_span_eV_atom": 0,
        "mean_frequency_THz": float(mean),
        "std_frequency_THz": float(std),
        "frequency_95pct_THz": float(percentile),
    }
    path = tmp_path / "answer.json"
    path.write_text(
        json.dumps({"results": results, "artifacts": {"run_log": "run.jsonl"}})
    )
    return path


def verification_response(plan, fingerprint):
    assert all("energy" not in frame for job in plan.jobs for frame in job["frames"])
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "jobs": [
            {
                "id": job["id"],
                "status": "complete",
                "value": copy.deepcopy(plan.expected[job["id"]]["values"]),
            }
            for job in plan.jobs
        ],
    }


def scorer(level=2, transport=verification_response):
    return Task6Scorer(level=level, verifier=ModalVerifier(transport=transport))


def grade(path, level=2):
    return scorer(level).evaluate(path)


def test_recomputes_correct_answers_from_one_log(submission):
    report = grade(submission)
    assert report["score"] == 1, report
    assert report["computed"]["production_temperature_K"] == pytest.approx(300, abs=1)
    assert report["computed"]["energy_span_eV_atom"] < 1e-12
    assert report["pending_checks"] == []


def test_two_summary_answers_keep_dynamics_checks(submission):
    answer = json.loads(submission.read_text())
    answer["results"] = {
        key: answer["results"][key]
        for key in ("mean_frequency_THz", "std_frequency_THz")
    }
    submission.write_text(json.dumps(answer))
    report = scorer().evaluate(submission)
    assert report["score"] == 1, report
    for key in (
        "equilibration_temperature_K",
        "production_temperature_K",
        "sampling_interval_fs",
        "frequency_resolution_THz",
        "energy_drift_eV_atom_ps",
        "energy_span_eV_atom",
        "frequency_95pct_THz",
    ):
        assert key in report["computed"]
    log_path = submission.parent / "run.jsonl"
    records = read_log(log_path)
    write_log(log_path, records[:-1])
    assert scorer().evaluate(submission)["score"] == 0


def test_level1_needs_only_equilibrated_restartable_log(submission):
    path = submission.parent / "run.jsonl"
    records = read_log(path)
    stop = next(i for i, row in enumerate(records) if row["event"] == "end")
    write_log(path, records[: stop + 1])
    answer = json.loads(submission.read_text())
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    assert grade(submission, 1)["score"] == 1
    assert grade(submission, 2)["score"] == 0


@pytest.mark.parametrize(
    "key",
    [
        "mean_frequency_THz",
        "std_frequency_THz",
        "frequency_95pct_THz",
        "equilibration_temperature_K",
        "production_temperature_K",
        "sampling_interval_fs",
        "frequency_resolution_THz",
        "energy_drift_eV_atom_ps",
        "energy_span_eV_atom",
    ],
)
def test_altered_answers_fail(submission, key):
    answer = json.loads(submission.read_text())
    answer["results"][key] += 10
    submission.write_text(json.dumps(answer))
    report = grade(submission)
    assert report["score"] == 0
    assert key in report["checks"][0]["detail"]


@pytest.mark.parametrize(
    "change",
    [
        "truncated",
        "model",
        "units",
        "time",
        "boundary",
        "reset",
        "cell",
        "mass",
        "energy",
        "nonfinite",
        "missing_momenta",
        "thermostat",
        "cold",
        "geometry",
    ],
)
def test_bad_recorded_computations_fail(submission, change):
    path = submission.parent / "run.jsonl"
    records = read_log(path)
    samples = [row for row in records if row["event"] == "sample"]
    production = samples[41:]
    if change == "truncated":
        records.pop()
    elif change == "model":
        records[0]["model_sha256"] = "wrong"
    elif change == "units":
        records[0]["units"]["time"] = "ps"
    elif change == "time":
        production[10]["time_fs"] += 1
    elif change == "boundary":
        production[0]["momenta"][0][0] += 1
    elif change == "reset":
        production[50]["momenta"] = (-np.array(production[50]["momenta"])).tolist()
    elif change == "cell":
        production[-1]["cell_A"][0][0] += 0.1
    elif change == "mass":
        records[0]["masses_amu"][0] = 1
    elif change == "energy":
        production[20]["potential_energy_eV"] += 1
    elif change == "nonfinite":
        production[-1]["potential_energy_eV"] = float("nan")
    elif change == "missing_momenta":
        del production[-1]["momenta"]
    elif change == "thermostat":
        next(row for row in records if row.get("name") == "production")[
            "target_temperature_K"
        ] = 300
    elif change == "geometry":
        samples[0]["positions_A"][0][0] += 0.3
    else:
        for row in samples[:41]:
            row["momenta"] = (np.array(row["momenta"]) / 10).tolist()
    write_log(path, records)
    assert grade(submission)["score"] == 0


@pytest.mark.parametrize("key", ["mean_frequency_THz", "std_frequency_THz"])
def test_every_spectral_summary_is_required(submission, key):
    answer = json.loads(submission.read_text())
    del answer["results"][key]
    submission.write_text(json.dumps(answer))
    report = grade(submission)
    assert report["score"] == 0
    assert key in report["checks"][0]["detail"]


@pytest.mark.parametrize("value", [None, True, "6.1", [], float("nan"), float("inf")])
def test_spectral_summaries_must_be_finite_numbers(submission, value):
    answer = json.loads(submission.read_text())
    answer["results"]["mean_frequency_THz"] = value
    submission.write_text(json.dumps(answer))
    report = grade(submission)
    assert report["score"] == 0
    assert "finite" in report["checks"][0]["detail"]


@pytest.mark.parametrize("linked", [False, True])
def test_legacy_spectrum_cannot_replace_required_summaries(submission, linked):
    answer = json.loads(submission.read_text())
    spectrum = {"frequency_THz": [0, 25, 50], "vdos": [0, 0.04, 0]}
    answer["results"] = {}
    if linked:
        (submission.parent / "spectrum.json").write_text(json.dumps(spectrum))
        answer["artifacts"]["spectrum"] = "spectrum.json"
    else:
        answer["results"]["spectrum"] = spectrum
    submission.write_text(json.dumps(answer))
    assert grade(submission)["score"] == 0


def test_velocity_units_produce_same_summaries(submission):
    original = grade(submission)["computed"]
    path = submission.parent / "run.jsonl"
    records = read_log(path)
    masses = np.array(records[0]["masses_amu"])
    for row in records:
        if row["event"] == "sample":
            row["velocities_A_fs"] = (
                np.array(row.pop("momenta")) / masses[:, None] * units.fs
            ).tolist()
    write_log(path, records)
    report = grade(submission)
    assert report["score"] == 1
    for key in ("mean_frequency_THz", "std_frequency_THz", "frequency_95pct_THz"):
        assert report["computed"][key] == pytest.approx(original[key])


@pytest.mark.parametrize("area", [1, 3])
def test_spectral_summaries_of_uniform_density(area):
    frequency = np.linspace(0, 10, 1001)
    density = np.full_like(frequency, area / 10)
    result = spectral_summaries(frequency, density)
    assert result["mean_frequency_THz"] == pytest.approx(5)
    assert result["std_frequency_THz"] == pytest.approx(10 / np.sqrt(12), rel=2e-6)
    assert result["frequency_95pct_THz"] == pytest.approx(9.5)


def test_percentile_interpolates_first_crossing_before_a_plateau():
    result = spectral_summaries([0, 1, 2, 3, 4], [19, 0, 0, 0, 1])
    assert result["frequency_95pct_THz"] == pytest.approx(1)


@pytest.mark.parametrize("change", ["undersampled", "short"])
def test_inadequate_production_cannot_pass_with_correct_summaries(submission, change):
    path = submission.parent / "run.jsonl"
    records = read_log(path)
    start = next(i for i, row in enumerate(records) if row.get("name") == "production")
    stage = records[start]
    samples = records[start + 1 : -1]
    end = records[-1]
    if change == "undersampled":
        stage["interval"] *= 2
        samples = samples[::2]
        expected = "at most every 10 fs"
    else:
        stage["steps"] = end["steps"] = 4000
        samples = samples[:401]
        expected = "too short"
    write_log(path, [*records[:start], stage, *samples, end])
    report = grade(submission)
    assert report["score"] == 0
    assert expected in report["checks"][0]["detail"]


def test_changing_spectral_features_fail_block_stability(submission):
    from corral_md.scoring.common import Evidence, EvidenceError
    from corral_md.scoring.task_6 import production_statistics
    from corral_md.scoring.task_6 import read_log as load_log

    stage = load_log(Evidence(submission), 2)[1]
    t = (stage["time"] - stage["time"][0]) / 1000
    rng = np.random.default_rng(123)
    frequency = rng.choice([4, 6, 8, 10], size=(32, 3))
    phase = (
        2
        * np.pi
        * (t[:, None, None] * frequency + 10 * np.maximum(t[:, None, None] - 8, 0))
    )
    phase += rng.uniform(0, 2 * np.pi, size=(32, 3))
    half = np.sqrt(2 * units.kB * 300 / 26.9815385) * np.cos(phase)
    v = np.concatenate([half, -half], axis=1)
    stage["velocity"] = v
    stage["momenta"] = v * 26.9815385
    stage["kinetic"] = (v**2 * 26.9815385).sum(axis=(1, 2)) / 2
    stage["temperature"] = stage["kinetic"] * 2 / (192 * units.kB)
    stage["positions"][1:] = stage["positions"][0] + np.cumsum(
        (v[1:] + v[:-1]) * 5 * units.fs, axis=0
    )
    with pytest.raises(EvidenceError, match="not stable across trajectory blocks"):
        production_statistics(stage)


def test_restored_workspace_resolves_log_without_extra_documents(submission):
    from corral_md.submission import resolve_submission

    resolved = resolve_submission(submission.read_text(), submission.parent)
    assert scorer().evaluate_submission(resolved).score == 1


@pytest.mark.parametrize("backend", ["offline", "local", "modal"])
def test_factories_use_only_the_new_scorer(backend, md_assets):
    for level, factory in enumerate((check_level1_workflow, check_level2_workflow), 1):
        scorer = factory(6, verification_backend=backend, assets_dir=str(md_assets))
        assert isinstance(scorer, Task6Scorer)
        assert scorer.level == level
        assert (scorer.verifier is None) == (backend == "offline")
        assert scorer.evaluate_submission("{}").score == 0


@pytest.mark.parametrize("change", ["potential_drift", "anisotropic", "collapse"])
def test_level1_rejects_bad_preparation_with_controlled_temperature(submission, change):
    log = submission.parent / "run.jsonl"
    records = read_log(log)
    end = next(i for i, row in enumerate(records) if row["event"] == "end")
    records = records[: end + 1]
    samples = [r for r in records if r["event"] == "sample"]
    if change == "potential_drift":
        for row in samples:
            row["potential_energy_eV"] += 1e6 * row["time_fs"]
    elif change == "anisotropic":
        for row in samples[1:]:
            p = np.asarray(row["momenta"])
            q = np.zeros_like(p)
            q[:, 0] = np.sqrt((p * p).sum(axis=1)) * np.sign(p[:, 0])
            row["momenta"] = q.tolist()
    else:
        samples[-1]["positions_A"][1] = samples[-1]["positions_A"][0]
    write_log(log, records)
    submission.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    result = scorer(1).evaluate_submission(submission)
    assert result.score == 0
    assert {
        "potential_drift": "energy",
        "anisotropic": "directional",
        "collapse": "overlapping",
    }[change] in result.feedback


@pytest.mark.parametrize("level", [1, 2])
def test_model_verification_is_required_and_samples_endpoints(submission, level):
    if level == 1:
        log = submission.parent / "run.jsonl"
        records = read_log(log)
        end = next(i for i, r in enumerate(records) if r["event"] == "end")
        write_log(log, records[: end + 1])
        submission.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    offline = Task6Scorer(level=level)
    report = offline.evaluate(submission)
    assert report["score"] is None
    assert report["pending_checks"] == ["independent_model_calculation"]
    with pytest.raises(PendingReviewError):
        offline(submission)
    verified = scorer(level).evaluate(submission)
    assert verified["score"] == 1, verified
    checks = verified["verification"]["checks"]
    assert len(checks) == level
    for check in checks:
        assert check["sample_indices"][0] == 0
        assert check["sample_indices"][-1] == (
            40 if check["id"] == "equilibration" else 1600
        )


@pytest.mark.parametrize(
    "change", ["wrong_energy", "unavailable", "identity", "missing"]
)
def test_model_verification_cannot_silently_pass(submission, change):
    def transport(plan, fingerprint):
        result = verification_response(plan, fingerprint)
        if change == "wrong_energy":
            result["jobs"][0]["value"]["frames"][0]["energy"] += 1
        elif change == "unavailable":
            raise RuntimeError("backend unavailable")
        elif change == "identity":
            result["challenge"] = "different evidence"
        else:
            result["jobs"] = []
        return result

    report = scorer(transport=transport).evaluate(submission)
    assert report["score"] == (0 if change == "wrong_energy" else None), report


def test_offline_cli_reports_pending_verification(submission, monkeypatch, capsys):
    from corral_md.scoring.evaluator import main

    monkeypatch.setattr(
        "sys.argv",
        ["scoring", "6", str(submission), "--verification", "offline"],
    )
    assert main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["score"] is None
    assert report["pending_checks"] == ["independent_model_calculation"]


def test_equilibration_diagnostics_are_recomputed_and_optional_claims_checked(
    submission,
):
    report = grade(submission)
    key = "equilibration_directional_temperature_K"
    assert np.mean(report["computed"][key]) == pytest.approx(
        report["computed"]["equilibration_temperature_K"]
    )
    answer = json.loads(submission.read_text())
    answer["results"][key] = [300, 300, 900]
    submission.write_text(json.dumps(answer))
    assert grade(submission)["score"] == 0


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("representation", ["nested", "aliased"])
def test_scientific_score_is_independent_of_md_log_layout(
    submission, rewrite_md_recording, level, representation
):
    path = submission[0] if isinstance(submission, tuple) else submission
    answer = json.loads(path.read_text())
    log = path.parent / answer["artifacts"]["run_log"]
    if level == 1:
        answer.pop("results", None)
        path.write_text(json.dumps(answer))
    rewrite_md_recording(log, representation, first_stage=level == 1)
    report = scorer(level=level).evaluate(path)
    assert report["score"] == 1, report
