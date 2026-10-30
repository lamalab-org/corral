"""Task 1 saved-record tests; synthetic Brownian paths are not calibration runs."""

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from corral_md.scoring import (
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring import task_1 as scoring
from corral_md.scoring.common import EvidenceError
from corral_md.submission import resolve_submission


def _save(path, data):
    path.write_text(json.dumps(data))


def _write_dump(path, data):
    with path.open("w") as stream:
        for i, step in enumerate(data.steps):
            if data.times_ps and data.times_ps[i] is not None:
                stream.write(f"ITEM: TIME\n{data.times_ps[i]:.12g}\n")
            stream.write(
                f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n216\nITEM: BOX BOUNDS pp pp pp\n"
            )
            np.savetxt(
                stream,
                np.column_stack([np.zeros(3), np.diag(data.cells[i])]),
                fmt="%.12g",
            )
            stream.write("ITEM: ATOMS " + " ".join(scoring.DUMP_COLUMNS) + "\n")
            rows = np.column_stack(
                [np.arange(1, 217), np.ones(216), data.positions[i], data.velocities[i]]
            )
            # LAMMPS may reorder atoms between frames. Identity, not row order, matters.
            np.savetxt(stream, rows[::-1] if i % 2 else rows, fmt="%.12g")


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("silicon-records")
    rng = np.random.default_rng(17)
    times = np.arange(0, 1501, 2)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    # Synthetic D = 1.74e-8 m2/s = 1.74 A2/ps, so each coordinate has
    # variance 2 * D * dt = 2 * 1.74 * 2 A2. This is not a calibration run.
    motion = rng.normal(scale=np.sqrt(6.96), size=(len(times), 216, 3))
    motion[times < 800] = 0
    positions = atoms.positions + np.cumsum(motion, axis=0)
    velocities = rng.normal(size=positions.shape)
    velocities -= velocities.mean(axis=1, keepdims=True)
    target = np.minimum(2500, 300 + 2.2 * times)
    temperature = (
        28.0855
        * np.sum((velocities / (1000 * units.fs)) ** 2, axis=(1, 2))
        / (645 * units.kB)
    )
    velocities *= np.sqrt(target / temperature)[:, None, None]
    data = scoring.Trajectory(
        (times * 1000).astype(int),
        np.tile(atoms.cell.array, (len(times), 1, 1)),
        positions,
        velocities,
    )
    _write_dump(root / "run.lammpstrj", data)
    density = 216 * 28.0855 * 1.66053906660 / atoms.get_volume()
    commands = """units metal
atom_style full
read_data silicon.data
replicate 3 3 3
pair_style sw
pair_coeff * * /workspace/resources/md-potentials-SW-Si/Si.sw Si
timestep 0.001
velocity all create 300 17 mom yes
fix heat all npt temp 300 2500 0.1 iso 0 0 1
run 1000000
unfix heat
fix production all npt temp 2500 2500 0.1 iso 0 0 1
run 500000
"""
    with (root / "run.log").open("w") as stream:
        stream.write(commands + "Step Time Temp Press Density\n")
        np.savetxt(
            stream,
            np.column_stack(
                [
                    data.steps,
                    times,
                    target,
                    np.zeros(len(times)),
                    np.full(len(times), density),
                ]
            ),
            fmt="%.12g",
        )
    _save(
        root / "manifest.json",
        {
            "artifacts": {"lammps_log": "run.log", "trajectory": "run.lammpstrj"},
            "results": {"diffusion_m2_s": 1.74e-8},
        },
    )
    return root


@pytest.fixture
def submission(recorded_run, tmp_path):
    for path in recorded_run.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    return tmp_path / "manifest.json"


def _grader(level=2):
    return WorkflowScorer(1, level=level)


def test_short_diffusion_result_keeps_numerical_checks(submission):
    manifest = json.loads(submission.read_text())
    results = manifest["results"]
    results["diffusion"] = results.pop("diffusion_m2_s")
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 1
    results["diffusion"] *= 2
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 0


@pytest.mark.parametrize("time_headers", [False, True])
def test_reconstructs_known_diffusion_with_only_two_records(submission, time_headers):
    if time_headers:
        path = submission.parent / "run.lammpstrj"
        data = scoring.read_trajectory(path)
        _write_dump(path, replace(data, times_ps=tuple(data.steps * 0.001)))
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report
    assert report["calculated"]["diffusion_m2_s"] == pytest.approx(1.74e-8, rel=0.1)
    assert report["calculated"]["analysis_interval_ps"] == [1000, 1500]
    assert 0 < report["calculated"]["uncertainty_95_m2_s"] < 4e-9
    assert sum(c["points"] for c in report["checks"]) == 100
    assert report["calculated"]["reference_diffusion"] == {
        "diffusion_m2_s": 1.74e-8,
        "relative_tolerance": 0.1,
        "accepted_range_m2_s": pytest.approx([1.566e-8, 1.914e-8], abs=1e-20),
    }


@pytest.fixture
def short_dump(tmp_path):
    path = tmp_path / "short.lammpstrj"
    _write_dump(
        path,
        scoring.Trajectory(
            np.array([0, 2]),
            np.tile(np.eye(3), (2, 1, 1)),
            np.zeros((2, 216, 3)),
            np.zeros((2, 216, 3)),
        ),
    )
    return path


@pytest.mark.parametrize("times", [(0.0, 0.002), (None, 0.002), (0.0, None)])
def test_optional_time_headers_preserve_trajectory(short_dump, times):
    original = scoring.read_trajectory(short_dump)
    _write_dump(short_dump, replace(original, times_ps=times))
    actual = scoring.read_trajectory(short_dump)
    assert actual.times_ps == times
    for field in ("steps", "cells", "positions", "velocities"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "invalid", ""])
def test_malformed_time_headers_are_rejected(short_dump, value):
    short_dump.write_text(f"ITEM: TIME\n{value}\n" + short_dump.read_text())
    with pytest.raises(EvidenceError, match="Invalid trajectory TIME value"):
        scoring.read_trajectory(short_dump)


@pytest.mark.parametrize("tail", ["", "0\n", "0\nITEM: TIME\n0\n"])
def test_incomplete_or_duplicate_time_headers_are_rejected(short_dump, tail):
    short_dump.write_text("ITEM: TIME\n" + tail)
    with pytest.raises(EvidenceError):
        scoring.read_trajectory(short_dump)


@pytest.mark.parametrize("times", [(0, 0), (None, 0), (0, None), (0, 1)])
def test_time_headers_at_repeated_frame_boundaries(short_dump, times):
    _, first, second = short_dump.read_text().split("ITEM: TIMESTEP\n")
    repeated = "".join(
        (f"ITEM: TIME\n{time}\n" if time is not None else "")
        + "ITEM: TIMESTEP\n"
        + first
        for time in times
    )
    short_dump.write_text(repeated + "ITEM: TIMESTEP\n" + second)
    if times == (0, 1):
        with pytest.raises(EvidenceError, match="Conflicting TIME headers"):
            scoring.read_trajectory(short_dump)
    else:
        actual = scoring.read_trajectory(short_dump)
        assert actual.times_ps == (0, None)
        assert actual.steps.tolist() == [0, 2]


def test_time_headers_must_match_recorded_run_clock(submission):
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    # Positive, increasing times are still invalid if they contradict the log.
    _write_dump(path, replace(data, times_ps=tuple(data.steps * 0.002)))
    report = _grader().evaluate(submission)
    assert report["score"] == 0, report
    assert "Trajectory TIME headers contradict recorded run times" in str(report)


def test_default_scorer_completes_from_recorded_evidence(submission):
    result = check_level2_workflow(1).evaluate_submission(submission)
    report = result.metadata["workflow_evaluation"]
    assert result.score == 1, report
    assert report["status"] == "complete"
    assert "pending_checks" not in report
    assert report["earned_points"] == report["possible_points"] == 100
    assert report["scoring_policy"] == scoring.DIFFUSION_POLICY
    assert report["mode"] == "final_quantities"


@pytest.mark.parametrize("clock_source", ["log", "dump", "both", "coarse_log"])
@pytest.mark.parametrize("local_clock", [False, True])
def test_diffusion_needs_only_production_times_and_positions(
    submission, clock_source, local_clock
):
    """Regression for a log switched after setup, plus production-only output."""
    log = submission.parent / "run.log"
    _, thermal = scoring.read_log(log)
    path = submission.parent / "run.lammpstrj"
    original = scoring.read_trajectory(path)
    selected = original.steps >= 1000000
    steps = original.steps[selected] - (1000000 if local_clock else 0)
    times = thermal[thermal[:, 1] >= 1000, 1] - (1000 if local_clock else 0)
    data = scoring.Trajectory(
        steps,
        original.cells[selected],
        original.positions[selected],
        original.velocities[selected],
        times_ps=tuple(times) if clock_source in {"dump", "both"} else (),
    )
    _write_dump(path, data)
    if clock_source == "dump":
        log.write_text("LAMMPS retained production output\n")
    else:
        stride = 5 if clock_source == "coarse_log" else 1
        with log.open("w") as stream:
            stream.write("Step Time\n")
            np.savetxt(stream, np.column_stack([steps, times])[::stride], fmt="%.12g")
    # No velocity information is needed, even when its optional columns exist.
    text = path.read_text().replace(" vx ", " unused ")
    path.write_text(text)
    report = check_level2_workflow(1).evaluate(submission)
    assert report["score"] == 1, report
    assert report["calculated"]["analysis_interval_ps"] == (
        [0, 500] if local_clock else [1000, 1500]
    )
    assert report["calculated"]["diffusion_m2_s"] == pytest.approx(1.74e-8, rel=0.1)
    assert not {
        "heating_and_production_protocol",
        "state_and_thermal_continuity",
        "liquid_state",
        "stable_production",
    } & {c["name"] for c in report["checks"]}


@pytest.mark.parametrize("source", ["log", "dump"])
def test_explicit_incompatible_units_cannot_be_silently_assumed_metal(
    submission, source
):
    path = submission.parent / ("run.log" if source == "log" else "run.lammpstrj")
    if source == "log":
        path.write_text(path.read_text().replace("units metal", "units real"))
    else:
        path.write_text("ITEM: UNITS\nreal\n" + path.read_text())
    report = _grader().evaluate(submission)
    assert report["score"] == 0
    assert "units" in str(report)


def test_dump_units_header_is_supported(submission):
    path = submission.parent / "run.lammpstrj"
    path.write_text("ITEM: UNITS\nmetal\n" + path.read_text())
    assert _grader().evaluate(submission)["score"] == 1


def test_missing_physical_time_does_not_guess_a_timestep(submission):
    log = submission.parent / "run.log"
    log.write_text("Step Temp\n1000000 2500\n1500000 2500\n")
    report = _grader().evaluate(submission)
    assert report["score"] == 0
    assert "physical times" in str(report)


@pytest.mark.parametrize("mutation", ["reversed", "reset", "nonfinite", "conflicting"])
def test_invalid_output_clock_is_rejected(submission, mutation):
    log = submission.parent / "run.log"
    _, thermal = scoring.read_log(log)
    clock = thermal[:, :2].copy()
    if mutation == "reversed":
        clock = clock[::-1]
    elif mutation == "reset":
        clock[clock[:, 1] >= 1100, 1] -= 1100
    elif mutation == "nonfinite":
        clock[-1, 1] = np.nan
    else:
        duplicate = clock[-1].copy()
        duplicate[1] += 1
        clock = np.vstack([clock, duplicate])
    with log.open("w") as stream:
        stream.write("Step Time\n")
        np.savetxt(stream, clock, fmt="%.12g")
    assert _grader().evaluate(submission)["score"] == 0


def test_production_clock_can_be_used_with_a_full_trajectory(submission):
    log = submission.parent / "run.log"
    _, thermal = scoring.read_log(log)
    with log.open("w") as stream:
        stream.write("Step Time\n")
        np.savetxt(stream, thermal[thermal[:, 1] >= 1000, :2], fmt="%.12g")
    assert _grader().evaluate(submission)["score"] == 1


@pytest.mark.parametrize(
    "value", [1e-4, 2.0, -2e-8, 0, True, "2e-8", None, float("nan")]
)
def test_wrong_answer_or_units_cannot_pass(submission, value):
    manifest = json.loads(submission.read_text())
    manifest["results"]["diffusion_m2_s"] = value
    manifest["results"]["uncertainty_m2_s"] = 1e9
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 0


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_consistent_but_wrong_diffusion_fails_reference(submission, factor):
    original = _grader().evaluate(submission)["calculated"]["diffusion_m2_s"]
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    # Change the actual displacements, preserving linear MSD and its stability.
    data.positions[:] = data.positions[0] + np.sqrt(factor) * (
        data.positions - data.positions[0]
    )
    _write_dump(path, data)
    manifest = json.loads(submission.read_text())
    manifest["results"]["diffusion_m2_s"] = factor * original
    # Submission metadata cannot replace the evaluator's reference or tolerance.
    manifest["results"]["uncertainty_m2_s"] = 1e9
    manifest["reference_diffusion"] = {
        "diffusion_m2_s": factor * original,
        "relative_tolerance": 1e9,
    }
    _save(submission, manifest)

    result = check_level2_workflow(1).evaluate_submission(submission)
    report = result.metadata["workflow_evaluation"]
    assert result.score == 0, report
    assert report["status"] == "complete"
    assert "pending_checks" not in report
    assert report["calculated"]["diffusion_m2_s"] == pytest.approx(
        factor * original, rel=1e-8, abs=0
    )
    assert [c["name"] for c in report["checks"] if c["status"] != "passed"] == [
        "reference_diffusion"
    ]
    assert "Reference: 1.74e-08" in result.feedback


@pytest.mark.parametrize(
    ("reported", "reconstructed", "expected"),
    [
        (1.566e-8, 1.566e-8, 1),
        (1.914e-8, 1.914e-8, 1),
        (1.565999e-8, 1.566e-8, 0),
        (1.914001e-8, 1.914e-8, 0),
        (1.566e-8, 1.565999e-8, 0),
        (1.914e-8, 1.914001e-8, 0),
    ],
)
def test_reference_bounds_apply_to_both_coefficients(
    submission, monkeypatch, reported, reconstructed, expected
):
    # Isolate exact boundary decisions from trajectory serialization/fit rounding.
    diagnostics = _grader().evaluate(submission)["calculated"]
    monkeypatch.setattr(
        scoring,
        "diffusion",
        lambda *_: {**diagnostics, "diffusion_m2_s": reconstructed},
    )
    manifest = json.loads(submission.read_text())
    manifest["results"]["diffusion_m2_s"] = reported
    _save(submission, manifest)
    report = _grader().evaluate(submission)
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["reported_diffusion"]["status"] == "passed", report
    assert report["score"] == expected, report
    assert checks["reference_diffusion"]["status"] == (
        "passed" if expected else "failed"
    )


def test_reference_answer_still_requires_trajectory_consistency(submission):
    original = _grader().evaluate(submission)["calculated"]["diffusion_m2_s"]
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    # Keep the reconstructed value inside the reference range, but far enough
    # from the reported value to fail the independently bounded MSD comparison.
    data.positions[:] = data.positions[0] + np.sqrt(1.58e-8 / original) * (
        data.positions - data.positions[0]
    )
    _write_dump(path, data)
    manifest = json.loads(submission.read_text())
    manifest["results"]["diffusion_m2_s"] = 1.9e-8
    _save(submission, manifest)
    report = _grader().evaluate(submission)
    assert report["score"] == 0, report
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["reported_diffusion"]["status"] == "failed", report


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("temp 300 2500", "temp 300 2000"),
        ("run 500000", "run 400000"),
        ("pair_style sw", "pair_style tersoff"),
        ("SW-Si/Si.sw", "SW-Si/other.sw"),
        ("unfix heat", "unfix heat\nvelocity all create 2500 999"),
        ("unfix heat", "unfix heat\nreset_timestep 0"),
        ("unfix heat", "unfix heat\nread_restart another.restart"),
        ("iso 0 0", "iso 10 10"),
    ],
)
def test_setup_commands_do_not_grade_level2_diffusion(submission, old, new):
    log = submission.parent / "run.log"
    log.write_text(log.read_text().replace(old, new))
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "nonfinite",
        "identity",
        "frozen",
        "singular_cell",
        "wrapped",
        "jump",
    ],
)
def test_bad_trajectory_fails(submission, mutation):
    path = submission.parent / "run.lammpstrj"
    if mutation == "identity":
        path.write_text(path.read_text().replace("\n216 1 ", "\n215 1 ", 1))
    elif mutation == "nonfinite":
        path.write_text(path.read_text().replace("0 16.29", "0 nan", 1))
    else:
        data = scoring.read_trajectory(path)
        if mutation == "truncated":
            data = scoring.Trajectory(
                data.steps[:-1],
                data.cells[:-1],
                data.positions[:-1],
                data.velocities[:-1],
            )
        elif mutation == "frozen":
            data.positions[:] = data.positions[0]
        elif mutation == "singular_cell":
            data.cells[-1, 0] = 0
        elif mutation == "wrapped":
            data.positions %= np.diag(data.cells[0])
        elif mutation == "jump":
            data.positions[-1, 0] += 1000
        _write_dump(path, data)
    assert _grader().evaluate(submission)["score"] == 0


def test_diffusion_does_not_require_velocity_columns(submission):
    path = submission.parent / "run.lammpstrj"
    path.write_text(path.read_text().replace(" vx ", " wrong "))
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report


def test_thermal_diagnostics_and_initial_crystal_are_not_diffusion_gates(submission):
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    # Preserve every production displacement while changing irrelevant initial
    # geometry and optional velocities. No full-cycle state check should run.
    data.positions[data.steps < 1000000] = 0
    data.velocities[:] = np.nan
    _write_dump(path, data)
    log = submission.parent / "run.log"
    text = log.read_text().split("Step Time Temp Press Density\n", 1)[1]
    rows = np.loadtxt(text.splitlines())
    rows[:, 2:] = np.nan
    with log.open("w") as stream:
        stream.write("Step Time Temp Press Density\n")
        np.savetxt(stream, rows, fmt="%.12g")
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report


def test_production_only_trajectory_is_sufficient_for_diffusion(submission):
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    selected = data.steps >= 1000000
    _write_dump(
        path,
        scoring.Trajectory(
            data.steps[selected],
            data.cells[selected],
            data.positions[selected],
            data.velocities[selected],
        ),
    )
    assert _grader().evaluate(submission)["score"] == 1


@pytest.mark.parametrize("phase", ["crystalline", "liquid"])
@pytest.mark.parametrize("mutation", [None, "wrong_protocol", "truncated", "hot"])
@pytest.mark.parametrize("time_headers", [False, True])
def test_level1_scores_final_temperature_without_results(
    submission, phase, mutation, time_headers
):
    log = submission.parent / "run.log"
    text = log.read_text().replace(
        "unfix heat\nfix production all npt temp 2500 2500 0.1 iso 0 0 1\nrun 500000\n",
        "",
    )
    lines = text.splitlines()
    log.write_text(
        "\n".join(
            line
            for line in lines
            if not line[0].isdigit() or float(line.split()[1]) <= 1000
        )
        + "\n"
    )
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    if phase == "crystalline":
        vibration = np.random.default_rng(23).normal(
            scale=0.1, size=data.positions.shape
        )
        vibration[0] = 0
        data.positions[:] = data.positions[0] + vibration
    mask = data.steps <= 1000000
    if mutation == "wrong_protocol":
        log.write_text(log.read_text().replace("temp 300 2500", "temp 300 2000"))
    elif mutation == "truncated":
        mask = data.steps < 1000000
    elif mutation == "hot":
        data.velocities[data.steps == 1000000] *= 3
        lines = log.read_text().splitlines()
        final = lines[-1].split()
        final[2] = "8000"
        lines[-1] = " ".join(final)
        log.write_text("\n".join(lines) + "\n")
    _write_dump(
        path,
        scoring.Trajectory(
            data.steps[mask],
            data.cells[mask],
            data.positions[mask],
            data.velocities[mask],
            times_ps=tuple(data.steps[mask] * 0.001) if time_headers else (),
        ),
    )
    manifest = json.loads(submission.read_text())
    del manifest["results"]
    _save(submission, manifest)
    report = check_level1_workflow(1).evaluate(submission)
    if mutation == "hot":
        assert report["score"] == 0, report
        return
    assert report["score"] == 1, report
    assert report["earned_points"] == report["possible_points"] == 100
    assert report["calculated"]["final_temperature_K"] == 2500
    assert report["mode"] == "final_temperature"
    assert report["scoring_policy"] == "level1-task1-temperature-v1"
    assert "phase" not in report["calculated"]
    assert "liquid_state" not in {c["name"] for c in report["checks"]}
    assert "diffusion_m2_s" not in report["calculated"]
    assert "reference_diffusion" not in {c["name"] for c in report["checks"]}
    assert _grader().evaluate(submission)["score"] == 0


def test_level2_still_requires_resolved_diffusive_motion(submission):
    path = submission.parent / "run.lammpstrj"
    data = scoring.read_trajectory(path)
    vibration = np.random.default_rng(23).normal(scale=0.1, size=data.positions.shape)
    vibration[0] = 0
    data.positions[:] = data.positions[0] + vibration
    _write_dump(path, data)
    report = _grader().evaluate(submission)
    assert report["score"] == 0, report
    assert "diffusive MSD regime" in str(report)


def test_collective_translation_and_cell_breathing_are_removed(recorded_run):
    data = scoring.read_trajectory(recorded_run / "run.lammpstrj")
    commands, clock = scoring.read_log(
        recorded_run / "run.log", columns=("Step", "Time"), allow_missing=True
    )
    times, positions, _ = scoring._diffusion_states(data, commands, clock)
    recorded_times = np.interp(data.steps, clock[:, 0], clock[:, 1])
    reference = scoring.diffusion(times, positions)["diffusion_m2_s"]
    # A large constant translation changes no relative motion.
    shifted = replace(
        data,
        positions=data.positions + recorded_times[:, None, None] * np.array([1, 2, 3]),
    )
    _, motion, _ = scoring._diffusion_states(shifted, commands, clock)
    assert scoring.diffusion(times, motion)["diffusion_m2_s"] == pytest.approx(
        reference, rel=1e-10
    )
    scale = 1 + 0.005 * np.sin(recorded_times / 10)
    breathing = replace(
        data,
        cells=data.cells * scale[:, None, None],
        positions=data.positions[0] * scale[:, None, None],
    )
    _, motion, _ = scoring._diffusion_states(breathing, commands, clock)
    assert np.max(np.abs(motion)) < 1e-10


def test_ballistic_and_insufficient_sampling_are_rejected():
    times = np.arange(1000, 1501, 2)
    velocity = np.random.default_rng(5).normal(size=(216, 3))
    with pytest.raises(EvidenceError, match="diffusive"):
        scoring.diffusion(times, times[:, None, None] * velocity)
    with pytest.raises(EvidenceError, match="Insufficient"):
        scoring.diffusion(times[:10], np.zeros((10, 216, 3)))


def test_diffusion_changing_during_production_is_rejected():
    times = np.arange(1000, 1501, 2)
    motion = np.random.default_rng(9).normal(size=(len(times), 216, 3))
    motion[times >= 1250] *= 4
    with pytest.raises(EvidenceError, match=r"diffusive|stable|correlation"):
        scoring.diffusion(times, np.cumsum(motion, axis=0))


def test_first_100_ps_at_2500_k_contributes_to_diffusion():
    times = np.arange(1000, 1501, 2)
    motion = np.random.default_rng(17).normal(
        scale=np.sqrt(8), size=(len(times), 216, 3)
    )
    original = scoring.diffusion(times, np.cumsum(motion, axis=0))
    # Reverse early increments while preserving their magnitudes and all later
    # relative motion. An estimator discarding the first 100 ps would not change.
    motion[(times > 1000) & (times <= 1100)] *= -1
    changed = scoring.diffusion(times, np.cumsum(motion, axis=0))
    assert changed["diffusion_m2_s"] != pytest.approx(
        original["diffusion_m2_s"], rel=1e-4, abs=0
    )


def test_cli_completes_from_recorded_evidence(submission, monkeypatch):
    from corral_md.scoring.evaluator import main

    output = submission.parent / "evaluation.json"
    args = ["score", "1", str(submission), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["score"] == 1
    assert "pending_checks" not in report
    assert report["calculated"]["diffusion_m2_s"] > 0
    assert report["calculated"]["analysis_interval_ps"] == [1000, 1500]


def test_workspace_restoration_confines_evidence(submission, tmp_path):
    resolved = resolve_submission(submission.read_text(), tmp_path)
    assert _grader().evaluate(resolved)["score"] == 1
    manifest = json.loads(submission.read_text())
    manifest["artifacts"]["lammps_log"] = "../outside.log"
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 0


@pytest.mark.parametrize("level", [1, 2])
def test_recording_adapter_preserves_run_and_physics_commands(level):
    source = "timestep 0.002\nfix heat all npt temp 300 2500 0.1 iso 0 0 1\nrun 500000\nunfix heat\nrun 250000\n"
    recorded = scoring.recording_input(
        source, "/workspace/output/run.lammpstrj", level=level
    )
    assert all(line in recorded.splitlines() for line in source.splitlines())
    if level == 1:
        assert recorded.splitlines() == [
            "timestep 0.002",
            "fix heat all npt temp 300 2500 0.1 iso 0 0 1",
            "thermo_style custom step temp",
            "thermo_modify format float %.16g",
            "run 500000",
            "unfix heat",
            "thermo_style custom step temp",
            "thermo_modify format float %.16g",
            "run 250000",
        ]
    else:
        assert recorded == scoring.recording_input(
            source, "/workspace/output/run.lammpstrj"
        )
        assert recorded.count("dump corral_task1") == 1
        assert recorded.count("thermo_style custom step time temp press density") == 2
        assert "id type xu yu zu vx vy vz" in recorded


@pytest.mark.parametrize("level", [1, 2])
def test_environment_records_task1_and_seeds_no_examples(
    level, tmp_path, md_assets, monkeypatch
):
    from corral_md import env, tools

    environments = env.create_environments(
        str(tmp_path), level=level, assets_dir=str(md_assets)
    )
    environment = environments[f"level_{level}_task_1"].for_task("recorded-silicon")
    environment._ensure_seed_directories()
    root = environment.workspace_path
    root = Path(root)
    assert not (root / "submission_examples").exists()
    source = root / "input" / "heat.in"
    source.write_text("units metal\nrun 1000\n")
    original = source.read_text()

    def simulate(workspace, input_file, *, assets_dir):
        assert str(workspace) == str(root)
        assert assets_dir == str(md_assets)
        recorded = root / input_file.removeprefix("/workspace/")
        if level == 1:
            assert recorded.read_text() == (
                "units metal\nthermo_style custom step temp\n"
                "thermo_modify format float %.16g\nrun 1000\n"
            )
        else:
            assert "dump corral_task1" in recorded.read_text()
        log = root / "output" / "heat.recorded.log"
        log.write_text("Step Temp\n1000 2500\nLoop time of 1\n")
        return log

    monkeypatch.setattr(tools, "run_lammps_locally", simulate)
    result = environment.tools["run_lammps"].execute(
        input_file="/workspace/input/heat.in"
    )
    if level == 1:
        assert result == (
            "Simulation completed locally; log: /workspace/output/heat.recorded.log"
        )
        assert list((root / "output").iterdir()) == [
            root / "output" / "heat.recorded.log"
        ]
    else:
        assert "trajectory: /workspace/output/heat.lammpstrj" in result
    assert "log: /workspace/output/heat.recorded.log" in result
    assert source.read_text() == original


@pytest.mark.parametrize("level", [1, 2])
def test_recording_wrapper_preserves_remote_dispatch(tmp_path, monkeypatch, level):
    from corral_md import tools

    source = tmp_path / "heat.in"
    source.write_text("units metal\nrun 1000\n")
    calls = []

    def simulate(workspace, input_file, *, action_id):
        calls.append((workspace, input_file, action_id))
        return "Simulation completed; log: /workspace/output/heat.recorded.log"

    monkeypatch.setattr(tools, "_run_lammps_for_workspace", simulate)
    simulator = tools.build_run_lammps_tool(tmp_path)
    original_execute = simulator.execute
    recorded = tools.with_task1_recording(simulator, tmp_path, level=level)
    assert recorded.trusted
    assert recorded.hidden_args == simulator.hidden_args
    assert recorded.workspace_args == simulator.workspace_args
    result = recorded.execute(
        input_file="/workspace/heat.in", corral_action_id="preserved-action-id"
    )
    assert calls == [
        (tmp_path, str(tmp_path / "input/heat.recorded.in"), "preserved-action-id")
    ]
    assert simulator.execute == original_execute
    assert source.read_text() == "units metal\nrun 1000\n"
    assert ("trajectory:" in result) == (level == 2)


def test_orthogonal_cell_written_in_triclinic_dump_format(submission):
    path = submission.parent / "run.lammpstrj"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line == "ITEM: BOX BOUNDS pp pp pp":
            lines[i] = "ITEM: BOX BOUNDS xy xz yz pp pp pp"
            for offset in (1, 2, 3):
                lines[i + offset] += " 0.0"
    path.write_text("\n".join(lines) + "\n")
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report
