"""Known equations of state exercise Task 7 without rerunning MACE."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import (
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.scoring.task_7 import Task7Scorer, stage_statistics


def write_log(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def submission(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    v0 = atoms.get_volume()
    rng = np.random.default_rng(42)
    p = rng.normal(size=(108, 3))
    p -= p.mean(axis=0)
    p *= np.sqrt(324 * units.kB * 300 / (p**2 / atoms.get_masses()[:, None]).sum())

    def state(volume, temperature, step):
        scale = (volume / v0) ** (1 / 3)
        displacement = 0.01 * np.sin(step + np.arange(324).reshape(108, 3))
        return {
            "cell_A": (atoms.cell.array * scale).tolist(),
            "positions_A": (
                atoms.positions * scale + (displacement if step else 0)
            ).tolist(),
            "momenta": (p * np.sqrt(temperature / 300)).tolist(),
        }

    def volume(temperature):
        return v0 * (1 + 1.5e-4 * (temperature - 300) + 1e-7 * (temperature - 300) ** 2)

    tc, vc = 401.0, volume(401)
    modulus, pressure_beta = 70, 1.5e-4
    specs = [("heating", t, volume(t + 1)) for t in (300, 380, 400, 420, 500)]
    specs += [("cooling", t, volume(t + 1) + 0.5) for t in (400, 300)]
    specs += [
        ("pressure", t, v)
        for t, v in (
            (380, vc),
            (400, vc),
            (420, vc),
            (400, vc * 0.99),
            (400, vc * 1.01),
        )
    ]
    rows = [
        {
            "event": "start",
            "model_sha256": PINNED_TEACHER_SHA256,
            "random_seed": 42,
            "numbers": atoms.numbers.tolist(),
            "masses_amu": atoms.get_masses().tolist(),
            "units": {
                "time": "fs",
                "cell": "Angstrom",
                "energy": "eV",
                "pressure": "GPa",
                "momenta": "ASE",
            },
        }
    ]
    previous, offset = state(v0, 300, 0), 0
    for index, (branch, target, mean_volume) in enumerate(specs):
        npt = branch != "pressure"
        start = previous if npt else state(mean_volume, target + 1, index)
        rows.append(
            {
                "event": "stage",
                "ensemble": "NPT" if npt else "NVT",
                "branch": branch,
                "integrator": "NPTBerendsen" if npt else "Langevin",
                "target_temperature_K": target,
                "target_pressure_GPa": 1.01325e-4 if npt else None,
                "timestep_fs": 1,
                "equilibration_steps": 2000,
                "production_steps": 4000,
                "interval": 50,
                "state": start,
            }
        )
        # Independent, bounded fluctuations; each production mean is exact.
        noise = rng.normal(size=(80, 3))
        noise -= noise.mean(axis=0)
        for step in range(0, 6001, 50):
            fluctuation = noise[(step // 50 - 41) % 80] if step > 2000 else np.zeros(3)
            temperature = target + 1 + fluctuation[0] * 0.2
            vol = mean_volume + (fluctuation[1] * 0.02 if npt else 0)
            pressure = (
                1.01325e-4
                if npt
                else modulus * pressure_beta * (target + 1 - tc)
                - modulus * (vol / vc - 1)
            )
            pressure += fluctuation[2] * 0.0001
            kinetic = 324 * units.kB * temperature / 2
            cell = state(vol, temperature, step)["cell_A"]
            if step == 0:
                cell = start["cell_A"]
                kinetic = (
                    np.array(start["momenta"]) ** 2 / atoms.get_masses()[:, None]
                ).sum() / 2
                temperature = 2 * kinetic / (324 * units.kB)
                vol = np.linalg.det(cell)
            rows.append(
                {
                    "event": "sample",
                    "step": step,
                    "time_fs": offset + step,
                    "cell_A": cell,
                    "kinetic_energy_eV": kinetic,
                    "temperature_K": temperature,
                    "potential_pressure_GPa": pressure
                    - 2 * kinetic / (3 * vol * units.GPa),
                    "pressure_GPa": pressure,
                }
            )
        previous = state(vol, temperature, index + 1)
        offset += 6000
        rows.append(
            {"event": "end", "step": 6000, "time_fs": offset, "state": previous}
        )
    write_log(tmp_path / "run.jsonl", rows)
    # Independent expected answers: polynomial mean volumes and analytic EOS.
    heating_t = np.array([301, 381, 401, 421, 501])
    slope = np.polyfit(heating_t, volume(heating_t), 1)[0]
    local = v0 * (1.5e-4 + 2e-7 * 101) / vc
    answer = {
        "artifacts": {"run_log": "run.jsonl"},
        "results": {
            "beta_volume": slope / volume(301),
            "beta_npt_local": local,
            "bulk_modulus_GPa": modulus,
            "beta_pressure": pressure_beta,
            "local_difference_K_inv": pressure_beta - local,
            "volume_differences_A3": {"300": 0.5, "400": 0.5},
        },
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(answer))
    return path


def test_recomputes_both_routes_and_preserves_nonzero_differences(submission):
    report = Task7Scorer().evaluate(submission)
    assert report["score"] == 1, report
    calculated = report["computed"]
    assert calculated["bulk_modulus_GPa"] == pytest.approx(70, rel=1e-5)
    assert calculated["beta_pressure"] == pytest.approx(1.5e-4, rel=1e-5)
    assert calculated["beta_volume"] != pytest.approx(
        calculated["beta_npt_local"], rel=0.01
    )
    assert not calculated["local_routes_consistent"]
    assert calculated["local_difference_se_K_inv"] > 0
    assert calculated["volume_differences_A3"] == pytest.approx(
        {"300": 0.5, "400": 0.5}
    )
    assert report["pending_checks"] == []
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_one_expansion_answer_keeps_all_diagnostics(submission):
    answer = json.loads(submission.read_text())
    answer["results"] = {"beta_volume": answer["results"]["beta_volume"]}
    submission.write_text(json.dumps(answer))
    report = Task7Scorer().evaluate(submission)
    assert report["score"] == 1, report
    assert "bulk_modulus_GPa" in report["computed"]
    assert "volume_differences_A3" in report["computed"]
    answer["results"]["volume_differences_A3"] = []
    submission.write_text(json.dumps(answer))
    assert Task7Scorer().evaluate(submission)["score"] == 0
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    assert Task7Scorer().evaluate(submission)["score"] == 0


def test_level1_only_needs_reference_stage(submission):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    stop = next(i for i, r in enumerate(rows) if r["event"] == "end")
    write_log(log, rows[: stop + 1])
    answer = json.loads(submission.read_text())
    answer.pop("results")
    submission.write_text(json.dumps(answer))
    assert check_level1_workflow(7)(submission) == 1
    assert check_level2_workflow(7)(submission) == 0
    answer["results"] = {"pressure_GPa": 1.01325}
    submission.write_text(json.dumps(answer))
    assert check_level1_workflow(7)(submission) == 0


@pytest.fixture
def reference_submission(submission):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    end = next(i for i, row in enumerate(rows) if row["event"] == "end")
    write_log(log, rows[: end + 1])
    submission.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    return submission


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("timestep_fs", [0.5, 1.0, 2.0])
def test_requires_one_fs_timestep_with_consistent_sampling(
    submission, level, timestep_fs
):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    if level == 1:
        end = next(i for i, row in enumerate(rows) if row["event"] == "end")
        rows = rows[: end + 1]
        submission.write_text(json.dumps({"artifacts": {"run_log": "run.jsonl"}}))
    # Preserve elapsed times, durations, and sample counts at each timestep.
    for row in rows:
        if row["event"] == "stage":
            row["timestep_fs"] = timestep_fs
            for key in ("equilibration_steps", "production_steps", "interval"):
                row[key] = int(row[key] / timestep_fs)
        elif row["event"] in {"sample", "end"}:
            row["step"] = int(row["step"] / timestep_fs)
    write_log(log, rows)

    result = Task7Scorer(level=level).evaluate_submission(submission)
    assert result.score == (1 if timestep_fs == 1 else 0), result.feedback
    if timestep_fs != 1:
        assert "Use an integration timestep of 1 fs" in result.feedback


def replace_reference_measurements(submission, data):
    """Change thermal evidence while preserving its units and boundary checks."""
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    samples = [r for r in rows if r["event"] == "sample" and r["step"] > 2000]
    for row, (temperature, volume, pressure) in zip(samples, data, strict=True):
        cell = np.asarray(row["cell_A"])
        row["cell_A"] = (cell * (volume / np.linalg.det(cell)) ** (1 / 3)).tolist()
        row["temperature_K"] = float(temperature)
        row["kinetic_energy_eV"] = float(324 * units.kB * temperature / 2)
        row["pressure_GPa"] = float(pressure)
        row["potential_pressure_GPa"] = float(
            pressure - 2 * row["kinetic_energy_eV"] / (3 * volume * units.GPa)
        )
    state = rows[-1]["state"]
    scale = (data[-1, 1] / np.linalg.det(state["cell_A"])) ** (1 / 3)
    state["cell_A"] = samples[-1]["cell_A"]
    state["positions_A"] = (np.array(state["positions_A"]) * scale).tolist()
    momenta = np.array(state["momenta"])
    kinetic = (momenta**2 / np.array(rows[0]["masses_amu"])[:, None]).sum() / 2
    state["momenta"] = (
        momenta * np.sqrt(samples[-1]["kinetic_energy_eV"] / kinetic)
    ).tolist()
    write_log(log, rows)


def reference_signal(n=80):
    rng = np.random.default_rng(42)
    return np.array([300, 1800, 1.01325e-4]) + rng.normal(size=(n, 3)) * [
        0.2,
        0.02,
        0.0001,
    ]


def test_level1_reconstructs_reference_and_reports_window_precision(
    reference_submission,
):
    data = reference_signal()
    replace_reference_measurements(reference_submission, data)
    report = Task7Scorer(level=1).evaluate(reference_submission)
    assert report["score"] == 1, report
    reference = report["computed"]["stages"][0]
    keys = ("temperature_K", "volume_A3", "pressure_GPa")
    assert [reference[k] for k in keys] == pytest.approx(data.mean(axis=0))
    assert reference["sample_count"] == reference["distinct_thermal_samples"] == 80
    assert set(reference["block_standard_errors"]) == {"4", "8", "16"}
    for window, start in zip(reference["production_windows"], (20, 40), strict=True):
        assert window["discarded_samples"] == start
        assert window["sample_count"] == 80 - start
        assert [window[k] for k in keys] == pytest.approx(data[start:].mean(axis=0))
    assert "beta_volume" not in report["computed"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("column", range(3))
def test_level1_checks_optional_reference_answers(reference_submission, column):
    data = reference_signal()
    replace_reference_measurements(reference_submission, data)
    key = ("temperature_K", "volume_A3", "pressure_GPa")[column]
    answer = json.loads(reference_submission.read_text())
    answer["results"] = {key: float(data[:, column].mean())}
    reference_submission.write_text(json.dumps(answer))
    assert Task7Scorer(level=1)(reference_submission) == 1
    answer["results"][key] += (50, 100, 1)[column]
    answer["results"]["standard_errors"] = [1e9] * 3
    reference_submission.write_text(json.dumps(answer))
    result = Task7Scorer(level=1).evaluate_submission(reference_submission)
    assert result.score == 0
    assert key in result.feedback


@pytest.mark.parametrize("column", range(3))
def test_level1_constant_observable_does_not_establish_precision(
    reference_submission, column
):
    data = reference_signal()
    data[:, column] = data[:, column].mean()
    replace_reference_measurements(reference_submission, data)
    result = Task7Scorer(level=1).evaluate_submission(reference_submission)
    assert result.score == 0
    assert "independent sampling" in result.feedback


def test_level1_repeated_records_do_not_count_as_new_samples(reference_submission):
    data = np.tile(reference_signal(8), (10, 1))
    replace_reference_measurements(reference_submission, data)
    result = Task7Scorer(level=1).evaluate_submission(reference_submission)
    assert result.score == 0
    assert "independent sampling" in result.feedback


def test_level1_checks_sampling_in_later_windows(reference_submission):
    data = reference_signal()
    data[40:, 1] = 1800
    replace_reference_measurements(reference_submission, data)
    result = Task7Scorer(level=1).evaluate_submission(reference_submission)
    assert result.score == 0
    assert "Later production window" in result.feedback


def test_level1_unsettled_volume_fails_even_without_reported_results(
    reference_submission,
):
    data = reference_signal()
    # Equal half means hide a substantial early-production excursion.
    data[:, 1] += np.repeat([10, -10, 0, 0], 20)
    replace_reference_measurements(reference_submission, data)
    assert Task7Scorer(level=1)(reference_submission) == 0


@pytest.mark.parametrize("count", [32, 37, 80])
def test_stationary_reference_accepts_minimum_and_uneven_sample_counts(count):
    report = stage_statistics(
        {
            "data": reference_signal(count),
            "ensemble": "NPT",
            "target_temperature_K": 300,
        }
    )
    assert report["sample_count"] == count
    assert min(report["effective_samples"]) >= 16


def test_fixed_nvt_volume_needs_no_statistical_sampling():
    data = reference_signal()
    data[:, 1] = 1800
    report = stage_statistics(
        {"data": data, "ensemble": "NVT", "target_temperature_K": 300}
    )
    assert report["standard_errors"][1] == 0
    assert report["effective_samples"][1] is None


def test_correlated_reference_uncertainty_matches_known_ar1_scale():
    rng = np.random.default_rng(42)
    noise = np.zeros((5120, 3))
    for i in range(1, len(noise)):
        noise[i] = 0.8 * noise[i - 1] + 0.6 * rng.normal(size=3)
    amplitudes = np.array([0.5, 0.1, 0.002])
    data = [300, 1800, 1.01325e-4] + noise[-4096:] * amplitudes
    report = stage_statistics(
        {"data": data, "ensemble": "NPT", "target_temperature_K": 300}
    )
    # Stationary AR(1): variance of the mean approaches sigma^2/N * (1+rho)/(1-rho).
    expected_se = amplitudes * np.sqrt((1 + 0.8) / (1 - 0.8) / 4096)
    assert np.all(np.array(report["standard_errors"]) > 0.5 * expected_se)
    assert np.all(np.array(report["standard_errors"]) < 2 * expected_se)
    assert max(report["effective_samples"]) < 4096 / 4


@pytest.mark.parametrize(
    "key",
    [
        "beta_volume",
        "beta_npt_local",
        "bulk_modulus_GPa",
        "beta_pressure",
        "local_difference_K_inv",
        "volume_differences_A3",
    ],
)
def test_altered_answers_fail(submission, key):
    answer = json.loads(submission.read_text())
    if key == "volume_differences_A3":
        answer["results"][key]["300"] *= -1
    else:
        answer["results"][key] *= 2
    submission.write_text(json.dumps(answer))
    result = Task7Scorer().evaluate_submission(submission)
    assert result.score == 0
    assert key in result.feedback


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "missing_sample",
        "model",
        "units",
        "mass",
        "time",
        "pressure",
        "kinetic",
        "reset",
        "cell",
        "order",
        "target",
        "short",
        "no_local",
        "drift",
        "noise",
        "nonlocal",
        "unresolved",
        "nonlinear",
    ],
)
def test_bad_evidence_fails(submission, mutation):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    stages = [r for r in rows if r["event"] == "stage"]
    samples = [r for r in rows if r["event"] == "sample"]
    if mutation == "truncated":
        rows.pop()
    elif mutation == "missing_sample":
        rows.remove(samples[50])
    elif mutation == "model":
        rows[0]["model_sha256"] = "0" * 64
    elif mutation == "units":
        rows[0]["units"]["pressure"] = "bar"
    elif mutation == "mass":
        rows[0]["masses_amu"][0] *= 2
    elif mutation == "time":
        samples[50]["time_fs"] += 1
    elif mutation == "pressure":
        samples[50]["pressure_GPa"] = samples[50]["potential_pressure_GPa"]
    elif mutation == "kinetic":
        samples[50]["temperature_K"] *= 2
    elif mutation == "reset":
        # Preserve kinetic energy to isolate the boundary continuity check.
        stages[1]["state"]["momenta"][0][0] *= -1
    elif mutation == "cell":
        samples[-50]["cell_A"][0][0] *= 1.01
    elif mutation == "order":
        stages[1]["branch"] = "cooling"
    elif mutation == "target":
        stages[0]["target_pressure_GPa"] = 1.01325
    elif mutation == "short":
        stages[0]["production_steps"] = 10
    elif mutation == "no_local":
        stages[1]["target_temperature_K"] = 350
    else:
        stage_index = -1
        for row in rows:
            if row["event"] == "stage":
                stage_index += 1
            if row["event"] != "sample" or row["step"] <= 2000:
                continue
            delta = 0
            if mutation == "drift" and stage_index == 0:
                delta = (row["step"] - 2000) * 0.001
            elif mutation == "noise" and stage_index == 0:
                delta = (-1) ** (row["step"] // 50) * 10
            elif mutation == "unresolved" and stages[stage_index]["ensemble"] == "NVT":
                delta = -row["pressure_GPa"]
            elif mutation == "nonlinear" and stage_index == 8:
                delta = 2
            elif mutation == "nonlocal" and stage_index == 9:
                stages[stage_index]["target_temperature_K"] = 480
            row["pressure_GPa"] += delta
            row["potential_pressure_GPa"] += delta
    write_log(log, rows)
    report = Task7Scorer().evaluate(submission)
    assert report["score"] == 0, (mutation, report)


def test_huge_claimed_uncertainty_does_not_hide_wrong_answer(submission):
    answer = json.loads(submission.read_text())
    answer["results"].update(beta_pressure=1, standard_errors={"beta_pressure": 1e9})
    submission.write_text(json.dumps(answer))
    assert Task7Scorer()(submission) == 0


def test_block_cancellation_does_not_hide_unresolved_pressure_derivative(submission):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    nvt = False
    for row in rows:
        if row["event"] == "stage":
            nvt = row["ensemble"] == "NVT"
        if nvt and row["event"] == "sample" and row["step"] > 2000:
            delta = 0.8 * (-1) ** (row["step"] // 50)
            row["pressure_GPa"] += delta
            row["potential_pressure_GPa"] += delta
    write_log(log, rows)
    report = Task7Scorer().evaluate(submission)
    assert report["score"] == 0, report
    assert "derivatives are not resolved" in report["checks"][0]["detail"]


def test_cli_scores_log_submission(submission, tmp_path, monkeypatch):
    import sys

    from corral_md.scoring.evaluator import main

    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys, "argv", ["score", "7", str(submission), "--output", str(output)]
    )
    assert main() == 0
    assert json.loads(output.read_text())["score"] == 1


def test_thermostat_temperature_roundoff_is_accepted(submission):
    log = submission.parent / "run.jsonl"
    rows = read_log(log)
    for row in rows:
        if row["event"] == "stage":
            row["target_temperature_K"] -= 1e-12
    write_log(log, rows)
    assert Task7Scorer()(submission) == 1


@pytest.mark.parametrize(
    "value", [None, {}, [], "", 123, {"results": {"beta_volume": float("nan")}}]
)
def test_malformed_submissions_score_zero(value):
    assert Task7Scorer()(value) == 0


@pytest.mark.parametrize("level", [1, 2])
def test_new_scorer_and_no_legacy_examples(level, tmp_path):
    factory = check_level1_workflow if level == 1 else check_level2_workflow
    assert isinstance(factory(7), Task7Scorer)
    with pytest.raises(ValueError, match="Task7Scorer"):
        WorkflowScorer(7, level=level)
    assert not (tmp_path / "submission_examples").exists()


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
    report = Task7Scorer(level=level).evaluate(path)
    assert report["score"] == 1, report
