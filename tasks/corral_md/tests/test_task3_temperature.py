"""Level 2 Task 3 grades the final kinetic-temperature window only."""

import json

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.io import read, write
from ase.io.trajectory import Trajectory
from corral_md.scoring import Task3TemperatureScorer, check_level2_workflow
from corral_md.scoring.evaluator import main
from corral_md.scoring.task_3_temperature import VERSION
from corral_md.submission import resolve_local_submission


def trajectory_answer(tmp_path, temperatures):
    path = tmp_path / "bulk.traj"
    atoms = bulk("Ag", "fcc", a=4.09, cubic=True).repeat((3, 3, 3))
    with Trajectory(path, "w") as trajectory:
        for temperature in temperatures:
            momenta = np.ones((len(atoms), 3)) * np.sqrt(
                atoms.get_masses()[:, None] * units.kB * temperature
            )
            atoms.set_momenta(momenta)
            trajectory.write(atoms)
    manifest = tmp_path / "answer.json"
    manifest.write_text(json.dumps({"artifacts": {"md_trajectory": "bulk.traj"}}))
    return manifest


@pytest.mark.parametrize("backend", [None, "offline", "local", "modal"])
def test_default_factory_needs_no_verifier_or_assets(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("CORRAL_MD_VERIFICATION", "unavailable")
    scorer = check_level2_workflow(
        3, verification_backend=backend, assets_dir=str(tmp_path / "absent-assets")
    )
    assert isinstance(scorer, Task3TemperatureScorer)
    assert scorer(trajectory_answer(tmp_path, [300] * 10)) == 1


@pytest.mark.parametrize(
    "temperature,expected", [(269.9, 0), (270, 1), (300, 1), (330, 1), (330.1, 0)]
)
def test_temperature_bounds_and_diagnostics(tmp_path, temperature, expected):
    path = trajectory_answer(tmp_path, [temperature] * 12)
    result = Task3TemperatureScorer().evaluate_submission(path)
    assert result.score == expected
    report = result.metadata["workflow_evaluation"]
    assert report["scoring_policy"] == VERSION
    assert report["computed"]["mean_temperature_K"] == pytest.approx(temperature)
    assert report["computed"]["allowed_range_K"] == [270, 330]
    assert report["computed"]["averaged_frames"] == 10
    assert report["computed"]["total_frames"] == 12
    assert bool(result.feedback) == (expected == 0)
    json.dumps(report, allow_nan=False)


def test_averages_only_last_ten_frames_without_per_frame_temperature_gates(tmp_path):
    path = trajectory_answer(tmp_path, [1000] * 20 + [200, 400] * 5)
    assert Task3TemperatureScorer()(path) == 1


@pytest.mark.parametrize("count", [1, 3, 9])
def test_short_trajectory_uses_all_available_frames(tmp_path, count):
    path = trajectory_answer(tmp_path, [300] * count)
    report = Task3TemperatureScorer().evaluate(path)
    assert report["score"] == 1
    assert report["computed"]["averaged_frames"] == count


def test_existing_manifest_ignores_dataset_model_and_log_contents(tmp_path):
    path = trajectory_answer(tmp_path, [300] * 10)
    manifest = json.loads(path.read_text())
    for role in ("dataset", "trained_checkpoint", "run_log"):
        artifact = tmp_path / role
        artifact.write_text("Not a readable dataset, model or calculation log")
        manifest["artifacts"][role] = role
    path.write_text(json.dumps(manifest))
    assert Task3TemperatureScorer()(path) == 1


def test_reported_temperature_cannot_override_kinetic_temperature(tmp_path):
    path = trajectory_answer(tmp_path, [600] * 10)
    manifest = json.loads(path.read_text())
    manifest["results"] = {"mean_temperature_K": 300}
    path.write_text(json.dumps(manifest))
    assert Task3TemperatureScorer()(path) == 0


@pytest.mark.parametrize("defect", ["missing", "empty", "corrupt", "no_link"])
def test_unreadable_or_absent_trajectory_fails(tmp_path, defect):
    path = trajectory_answer(tmp_path, [])
    trajectory = tmp_path / "bulk.traj"
    if defect == "missing":
        trajectory.unlink()
    elif defect == "corrupt":
        trajectory.write_bytes(b"not an ASE trajectory")
    elif defect == "no_link":
        path.write_text("{}")
    result = Task3TemperatureScorer().evaluate_submission(path)
    assert result.score == 0
    assert result.feedback


@pytest.mark.parametrize("defect", ["missing_momenta", "nan_momenta", "zero_mass"])
def test_invalid_kinetic_data_fails(tmp_path, defect):
    path = trajectory_answer(tmp_path, [300])
    trajectory = tmp_path / "bulk.traj"
    atoms = read(trajectory)
    if defect == "missing_momenta":
        del atoms.arrays["momenta"]
    elif defect == "nan_momenta":
        atoms.arrays["momenta"][0, 0] = np.nan
    else:
        atoms.set_masses(np.zeros(len(atoms)))
    write(trajectory, atoms)
    result = Task3TemperatureScorer().evaluate_submission(path)
    assert result.score == 0
    assert result.feedback


def test_workspace_paths_are_resolved_and_confined(tmp_path):
    trajectory_answer(tmp_path, [300])
    answer = json.dumps({"artifacts": {"md_trajectory": "/workspace/bulk.traj"}})
    resolved = resolve_local_submission(answer, tmp_path)
    assert Task3TemperatureScorer()(resolved) == 1
    answer = json.dumps({"artifacts": {"md_trajectory": "../bulk.traj"}})
    assert Task3TemperatureScorer()(resolve_local_submission(answer, tmp_path)) == 0


def test_cli_uses_temperature_policy_without_backend(tmp_path, monkeypatch, capsys):
    path = trajectory_answer(tmp_path, [300])
    monkeypatch.setattr("sys.argv", ["corral-md-score", "3", str(path), "--level", "2"])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["score"] == 1
    assert report["scoring_policy"] == VERSION
