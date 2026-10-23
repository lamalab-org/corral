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
from corral_md.scoring import WorkflowScorer, check_level1_workflow, check_level2_workflow
from corral_md.submission import resolve_submission
from corral_md.scoring import task_1 as scoring
from corral_md.scoring.common import EvidenceError


def _save(path, data):
    path.write_text(json.dumps(data))


def _write_dump(path, data):
    with path.open("w") as stream:
        for i, step in enumerate(data.steps):
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
    motion = rng.normal(scale=np.sqrt(8), size=(len(times), 216, 3))
    motion[times < 800] = 0
    positions = atoms.positions + np.cumsum(motion, axis=0)
    velocities = rng.normal(size=positions.shape)
    velocities -= velocities.mean(axis=1, keepdims=True)
    target = np.minimum(2500, 300 + 2.2 * times)
    temperature = (
        scoring.SILICON_MASS
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
    density = 216 * scoring.SILICON_MASS * 1.66053906660 / atoms.get_volume()
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
            "results": {"diffusion_m2_s": 2e-8},
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


def test_reconstructs_known_diffusion_with_only_two_records(submission):
    report = _grader().evaluate(submission)
    assert report["score"] == 1, report
    assert report["calculated"]["diffusion_m2_s"] == pytest.approx(2e-8, rel=0.1)
    assert report["calculated"]["analysis_interval_ps"] == [1000, 1500]
    assert 0 < report["calculated"]["uncertainty_95_m2_s"] < 4e-9
    assert sum(c["points"] for c in report["checks"]) == 100


def test_default_scorer_completes_from_recorded_evidence(submission):
    result = check_level2_workflow(1).evaluate_submission(submission)
    report = result.metadata["workflow_evaluation"]
    assert result.score == 1, report
    assert report["status"] == "complete"
    assert report["pending_checks"] == []
    assert report["earned_points"] == report["possible_points"] == 100


@pytest.mark.parametrize(
    "value", [1e-4, 2.0, -2e-8, 0, True, "2e-8", None, float("nan")]
)
def test_wrong_answer_or_units_cannot_pass(submission, value):
    manifest = json.loads(submission.read_text())
    manifest["results"]["diffusion_m2_s"] = value
    manifest["results"]["uncertainty_m2_s"] = 1e9
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 0


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
def test_wrong_protocol_and_state_resets_fail(submission, old, new):
    log = submission.parent / "run.log"
    log.write_text(log.read_text().replace(old, new))
    assert _grader().evaluate(submission)["score"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "missing_velocities",
        "nonfinite",
        "identity",
        "frozen",
        "hot",
        "wrong_cell",
        "wrapped",
        "jump",
    ],
)
def test_bad_trajectory_fails(submission, mutation):
    path = submission.parent / "run.lammpstrj"
    if mutation == "missing_velocities":
        path.write_text(path.read_text().replace(" vx ", " wrong "))
    elif mutation == "identity":
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
        elif mutation == "hot":
            data.velocities[-1] *= 3
        elif mutation == "wrong_cell":
            data.cells[-1] *= 1.1
        elif mutation == "wrapped":
            data.positions %= np.diag(data.cells[0])
        elif mutation == "jump":
            data.positions[-1, 0] += 1000
        _write_dump(path, data)
    assert _grader().evaluate(submission)["score"] == 0


def test_level1_stops_at_preparation_without_results(submission):
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
    mask = data.steps <= 1000000
    _write_dump(
        path,
        scoring.Trajectory(
            data.steps[mask],
            data.cells[mask],
            data.positions[mask],
            data.velocities[mask],
        ),
    )
    manifest = json.loads(submission.read_text())
    del manifest["results"]
    _save(submission, manifest)
    report = check_level1_workflow(1).evaluate(submission)
    assert report["score"] == 1, report
    assert "diffusion_m2_s" not in report["calculated"]
    assert _grader().evaluate(submission)["score"] == 0


def test_collective_translation_and_cell_breathing_are_removed(recorded_run):
    data = scoring.read_trajectory(recorded_run / "run.lammpstrj")
    commands, thermal = scoring.read_log(recorded_run / "run.log")
    segments = scoring._protocol(commands, thermal, 1500)
    times, positions = scoring._states(data, thermal, segments, 1500)
    reference = scoring.diffusion(times, positions)["diffusion_m2_s"]
    # A large constant translation changes no relative motion.
    shifted = replace(
        data, positions=data.positions + times[:, None, None] * np.array([1, 2, 3])
    )
    _, motion = scoring._states(shifted, thermal, segments, 1500)
    assert scoring.diffusion(times, motion)["diffusion_m2_s"] == pytest.approx(
        reference, rel=1e-10
    )
    scale = 1 + 0.005 * np.sin(times / 10)
    breathing = replace(
        data,
        cells=data.cells * scale[:, None, None],
        positions=data.positions[0] * scale[:, None, None],
    )
    thermal[:, 4] /= scale**3
    _, motion = scoring._states(breathing, thermal, segments, 1500)
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
    assert report["pending_checks"] == []
    assert report["calculated"]["diffusion_m2_s"] > 0
    assert report["calculated"]["analysis_interval_ps"] == [1000, 1500]


def test_workspace_restoration_confines_evidence(submission, tmp_path):
    resolved = resolve_submission(submission.read_text(), tmp_path)
    assert _grader().evaluate(resolved)["score"] == 1
    manifest = json.loads(submission.read_text())
    manifest["artifacts"]["lammps_log"] = "../outside.log"
    _save(submission, manifest)
    assert _grader().evaluate(submission)["score"] == 0


def test_recording_adapter_preserves_run_and_physics_commands():
    source = "timestep 0.002\nfix heat all npt temp 300 2500 0.1 iso 0 0 1\nrun 500000\nunfix heat\nrun 250000\n"
    recorded = scoring.recording_input(source, "/workspace/output/run.lammpstrj")
    assert all(line in recorded.splitlines() for line in source.splitlines())
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
        assert "dump corral_task1" in recorded.read_text()
        log = root / "output" / "heat.recorded.log"
        log.write_text("retained log")
        return log

    monkeypatch.setattr(tools, "run_lammps_locally", simulate)
    result = environment.tools["run_lammps"].execute(
        input_file="/workspace/input/heat.in"
    )
    assert "trajectory: /workspace/output/heat.lammpstrj" in result
    assert "log: /workspace/output/heat.recorded.log" in result
    assert source.read_text() == original
