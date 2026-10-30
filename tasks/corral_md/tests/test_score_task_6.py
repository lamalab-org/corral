"""Level 1 grades the sampled thermal outcome without workflow metadata."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import Task6Scorer, check_level1_workflow, check_level2_workflow
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.level2_quantities import Level2QuantityScorer
from corral_md.scoring.task_6 import spectral_summaries
from scipy import signal
from scipy.integrate import cumulative_trapezoid


def submission(tmp_path, rows):
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(row) for row in rows))
    return {"artifacts": {"run_log": str(log)}}


def temperatures():
    return [
        {"time_fs": time, "temperature_K": 300 + 4 * np.sin(time / 100)}
        for time in range(0, 3001, 50)
    ]


@pytest.mark.parametrize("backend", ["offline", "local", "modal"])
def test_saved_task6_temperatures_pass_without_restart_or_verifier(backend):
    # Original saved run: scalar samples and descriptive NVT metadata, no
    # separate integrator/com_removed_once fields. Its restart is not supplied.
    log = Path(__file__).parent / "data/task6_equilibration.jsonl"
    scorer = check_level1_workflow(6, verification_backend=backend)
    report = scorer.evaluate({"artifacts": {"run_log": str(log)}})
    assert isinstance(scorer, Task6Scorer)
    assert scorer.verifier is None
    assert report["score"] == 1, report
    assert report["computed"]["equilibration_temperature_K"] == pytest.approx(
        306.44568006041794
    )


@pytest.mark.parametrize("layout", ["bare", "descriptive", "nested", "picoseconds"])
def test_only_temperature_and_time_are_required(tmp_path, layout):
    rows = temperatures()
    if layout == "descriptive":
        rows.insert(0, {"type": "metadata", "ensemble": "NVT Langevin, fixed cell"})
    elif layout == "picoseconds":
        rows = [
            {"time_ps": row["time_fs"] / 1000, "T_K": row["temperature_K"]}
            for row in rows
        ]
    answer = submission(tmp_path, rows)
    if layout == "nested":
        Path(answer["artifacts"]["run_log"]).write_text(json.dumps({"samples": rows}))
    result = Task6Scorer().evaluate_submission(answer)
    assert result.score == 1
    assert result.metadata["workflow_evaluation"]["computed"][
        "equilibration_temperature_K"
    ] == pytest.approx(np.mean([row["temperature_K"] for row in temperatures()][30:]))


@pytest.mark.parametrize("representation", ["kinetic", "momenta"])
def test_temperature_can_be_derived_from_retained_kinetics(tmp_path, representation):
    rows = temperatures()
    for row in rows:
        t = row.pop("temperature_K")
        if representation == "kinetic":
            row["ekin_eV"] = 192 * units.kB * t / 2
        else:
            row["momenta"] = np.full(
                (64, 3), np.sqrt(26.9815385 * units.kB * t)
            ).tolist()
    report = check_level1_workflow(6).evaluate(submission(tmp_path, rows))
    assert report["score"] == 1, report


def test_production_and_summary_do_not_enter_equilibration_mean(tmp_path):
    rows = [{"event": "stage", "name": "equilibration"}, *temperatures()]
    rows += [
        {"event": "summary", "temperature_K": 9999},
        {"event": "stage", "name": "production"},
        {"event": "sample", "time_fs": 3000, "temperature_K": 9999},
    ]
    assert check_level1_workflow(6)(submission(tmp_path, rows)) == 1


@pytest.mark.parametrize(
    "change",
    [
        "cold",
        "drift",
        "short",
        "sparse",
        "few",
        "unordered",
        "duplicate",
        "nonfinite",
        "missing_time",
        "missing_temperature",
        "conflicting_time",
    ],
)
def test_invalid_or_unsettled_temperatures_fail(tmp_path, change):
    rows = temperatures()
    if change == "cold":
        for row in rows:
            row["temperature_K"] = 100
    elif change == "drift":
        # Mean of the final half is exactly 300 K, but it has not settled.
        for row, t in zip(rows[30:], np.linspace(200, 400, 31), strict=True):
            row["temperature_K"] = float(t)
    elif change == "short":
        rows = rows[:32]
    elif change == "sparse":
        for row in rows:
            row["time_fs"] *= 3
    elif change == "few":
        rows = rows[::3]
    elif change == "unordered":
        rows.reverse()
    elif change == "duplicate":
        rows[4]["time_fs"] = rows[3]["time_fs"]
    elif change == "nonfinite":
        rows[4]["temperature_K"] = float("nan")
    elif change == "missing_time":
        del rows[4]["time_fs"]
    elif change == "missing_temperature":
        del rows[4]["temperature_K"]
    else:
        rows[4]["time_ps"] = 99
    result = check_level1_workflow(6).evaluate_submission(submission(tmp_path, rows))
    assert result.score == 0
    assert result.feedback


def test_initial_transient_does_not_fail_settled_terminal_half(tmp_path):
    rows = temperatures()
    for row in rows[:30]:
        row["temperature_K"] = 100
    assert check_level1_workflow(6)(submission(tmp_path, rows)) == 1


def test_incorrect_optional_mean_is_rejected(tmp_path):
    answer = submission(tmp_path, temperatures())
    answer["results"] = {"equilibration_temperature_K": 600}
    assert check_level1_workflow(6)(answer) == 0


def test_level2_keeps_its_existing_quantity_policy():
    assert isinstance(check_level2_workflow(6), Level2QuantityScorer)


def write_log(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def trajectory_submission(tmp_path):
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


def test_restored_workspace_resolves_log_without_extra_documents(
    trajectory_submission, monkeypatch
):
    from corral_md.submission import resolve_submission

    # This fixture tests path restoration and velocity-to-spectrum mathematics.
    # Its four discrete harmonic frequencies are not a physical Al reference.
    monkeypatch.setattr(
        "corral_md.scoring.level2_quantities.require_reference_spectrum",
        lambda frequency, power: {"test_only": "synthetic harmonic spectrum"},
    )
    resolved = resolve_submission(
        trajectory_submission.read_text(), trajectory_submission.parent
    )
    assert check_level2_workflow(6).evaluate_submission(resolved).score == 1


@pytest.mark.parametrize("backend", ["offline", "local", "modal"])
def test_factories_use_only_the_new_scorer(backend, md_assets):
    for level, factory in enumerate((check_level1_workflow, check_level2_workflow), 1):
        scorer = factory(6, verification_backend=backend, assets_dir=str(md_assets))
        assert isinstance(scorer, Task6Scorer if level == 1 else Level2QuantityScorer)
        assert scorer.level == level
        if level == 1:
            assert scorer.verifier is None
        assert scorer.evaluate_submission("{}").score == 0


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("representation", ["nested", "aliased"])
def test_scientific_score_is_independent_of_md_log_layout(
    trajectory_submission, rewrite_md_recording, level, representation, monkeypatch
):
    # Exercise the same reconstructed moments in every layout; the actual
    # physical-spectrum gate is covered by the separate rejection test below.
    monkeypatch.setattr(
        "corral_md.scoring.level2_quantities.require_reference_spectrum",
        lambda frequency, power: {"test_only": "synthetic harmonic spectrum"},
    )
    path = trajectory_submission
    answer = json.loads(path.read_text())
    log = path.parent / answer["artifacts"]["run_log"]
    if level == 1:
        answer.pop("results", None)
        path.write_text(json.dumps(answer))
    rewrite_md_recording(log, representation, first_stage=level == 1)
    factory = check_level1_workflow if level == 1 else check_level2_workflow
    report = factory(6).evaluate(path)
    assert report["score"] == 1, report


def test_default_level2_rejects_unphysical_synthetic_spectrum(
    trajectory_submission,
):
    from corral_md.scoring import check_level2_workflow

    report = check_level2_workflow(6, verification_backend="offline").evaluate(
        trajectory_submission
    )
    assert report["score"] == 0, report
    assert "cosine similarity" in report["checks"][0]["detail"]
    assert report["scoring_policy"] == "level2-final-quantities-v2"
