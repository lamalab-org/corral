"""Artifact-only task 3 scoring, including truthful unsuccessful transfer."""

import hashlib
import json
import pickle
import struct

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from corral_md.score import (
    PendingReviewError,
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1_trusted import PINNED_TEACHER_SHA256
from corral_md.workflow_scoring.task_3 import evaluate


def _write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_level1_accepts_unit_labeled_dataset_records(submission):
    settings_path = submission.parent / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["teacher_sha256"] = PINNED_TEACHER_SHA256
    _write(settings_path, settings)
    dataset_path = submission.parent / "dataset.json"
    records = json.loads(dataset_path.read_text())
    for record in records:
        record["positions_angstrom"] = record.pop("positions")
        record["energy_eV"] = record.pop("energy")
        record["forces_eV_per_angstrom"] = record.pop("forces")
    _write(dataset_path, {"records": records})

    result = check_level1_workflow(3, verification_backend="offline").evaluate(submission)
    assert result["status"] == "pending_review", result["checks"]
    assert _check(result, "teacher_energy_and_force_labels")["status"] == "passed"
    assert _check(result, "independent_model_calculation")["status"] == "unverified"

    records[0]["energy"] = records[0]["energy_eV"] + 1
    _write(dataset_path, {"records": records})
    result = check_level1_workflow(3, verification_backend="offline").evaluate(submission)
    assert result["score"] == 0


def test_level1_binary_score_waits_for_independent_model_result(submission):
    settings_path = submission.parent / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["teacher_sha256"] = PINNED_TEACHER_SHA256
    _write(settings_path, settings)

    class StubVerifier:
        def __init__(self, status):
            self.status = status

        def evaluate(self, evidence, task_number):
            assert task_number == 3
            return {
                "evidence_sha256": evidence.fingerprint(),
                "backend": {"release_id": "test-release"},
                "checks": [
                    {
                        "id": "teacher_dimer_labels",
                        "status": self.status,
                        "targets": ["independent_model_calculation"],
                        "detail": "Controlled test response",
                    }
                ],
            }

    for status, expected in (("passed", 1.0), ("failed", 0.0), ("unverified", None)):
        result = WorkflowScorer(3, level=1, verifier=StubVerifier(status)).evaluate(submission)
        assert result["score"] == expected


@pytest.fixture
def submission(tmp_path):
    separations = np.round(np.arange(0.6, 5.01, 0.1), 8)
    energies = -1 / separations
    forces = np.zeros((45, 2, 3))
    forces[:, 0, 0] = 1 / separations**2
    forces[:, 1, 0] = -1 / separations**2
    dataset = [
        {
            "symbols": ["Ag", "Ag"],
            "positions": [[0, 0, 0], [float(d), 0, 0]],
            "pbc": False,
            "energy": float(energy),
            "forces": force.tolist(),
        }
        for d, energy, force in zip(separations, energies, forces, strict=False)
    ]
    paths = {"dataset": _write(tmp_path / "dataset.json", dataset)}
    fitting = {"evaluation_domain": "training"}
    for stage, error in (("before", 2), ("after", 0.1)):
        predictions = {
            "separation_A": separations.tolist(),
            "energy_eV": (energies + error).tolist(),
            "forces_eV_A": (forces + error).tolist(),
        }
        paths[f"predictions_{stage}"] = _write(tmp_path / f"{stage}.json", predictions)
        fitting[stage] = {
            "energy_rmse_per_atom_eV": error / 2,
            "force_rmse_eV_A": error,
            "separation_A": separations.tolist(),
            "energy_abs_error_per_atom_eV": [error / 2] * 45,
            "force_rmse_by_separation_eV_A": [error] * 45,
        }

    paths["training_history"] = _write(
        tmp_path / "training.json",
        [
            {"step": 0, "loss": 4.0, "learning_rate": 0.01},
            {"step": 1, "loss": 0.01, "learning_rate": 0.01},
        ],
    )
    checkpoint = tmp_path / "model.safetensors"
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    checkpoint.write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1.0, 2.0)
    )
    paths["trained_checkpoint"] = str(checkpoint)
    initial = bulk("Ag", "fcc", a=4.09, cubic=True).repeat((3, 3, 3))
    random = np.random.default_rng(1337)
    momenta = random.normal(size=(108, 3))
    momenta -= momenta.mean(axis=0)
    kinetic = np.sum(momenta**2 / initial.get_masses()[:, None]) / 2
    momenta *= np.sqrt(321 * units.kB * 300 / (2 * kinetic))
    frames, log = [], []
    last_displacement = random.normal(size=(108, 3)) * 0.03
    for fraction, temperature in ((0, 300), (0.5, 330), (1, 400)):
        positions = initial.positions + fraction * last_displacement
        frame = {
            "symbols": initial.get_chemical_symbols(),
            "positions": positions.tolist(),
            "cell": initial.cell.array.tolist(),
            "pbc": [True] * 3,
            "time_fs": 5000 * fraction,
            "momenta": (momenta * np.sqrt(temperature / 300)).tolist(),
            "energy": -400 + fraction,
            "forces": (np.ones((108, 3)) * fraction).tolist(),
        }
        frames.append(frame)
        log.append(
            {
                "time_fs": 5000 * fraction,
                "temperature_K": temperature,
                "potential_energy_eV": -400 + fraction,
                "force_max_eV_A": np.sqrt(3) * fraction,
                "kinetic_energy_eV": 0.5 * 321 * units.kB * temperature,
            }
        )
    paths["md_trajectory"] = _write(tmp_path / "md.json", frames)
    paths["md_log"] = _write(tmp_path / "mdlog.json", log)
    displacement_rms = np.sqrt(np.mean(np.sum(last_displacement**2, axis=1)))
    final_positions = np.array(frames[-1]["positions"])
    differences = final_positions[:, None, :] - final_positions[None, :, :]
    differences -= np.rint(differences / 12.27) * 12.27
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    settings = {
        "teacher_model": "teacher.model",
        "student_model": "student.model",
        "teacher_sha256": "a" * 64,
        "student_sha256": "b" * 64,
        "dispersion": True,
        "energy_unit": "eV",
        "force_unit": "eV/Angstrom",
        "training_precision": "float64",
        "training": {"optimizer": "Adam", "epochs": 2},
        "validation_strategy": "Fitting diagnostics on all training structures; no held-out generalization claim.",
        "training_dataset_sha256": _digest(tmp_path / "dataset.json"),
        "trained_checkpoint_sha256": _digest(checkpoint),
        "md": {
            "target_temperature_K": 300,
            "initial_temperature_K": 300,
            "random_seed": 1337,
            "thermostat": "Langevin",
            "remove_com_once": True,
            "timestep_fs": 1.0,
            "temperature_dof": 321,
        },
    }
    results = {
        "fitting": fitting,
        "bulk": {
            "temperature_mean_K": np.mean([300, 330, 400]),
            "temperature_std_K": np.std([300, 330, 400]),
            "temperature_min_K": 300,
            "temperature_max_K": 400,
            "potential_energy_range_eV": 1.0,
            "force_max_eV_A": np.sqrt(3),
            "final_displacement_rms_A": float(displacement_rms),
            "final_min_pair_distance_A": float(np.min(distances)),
        },
    }
    report = tmp_path / "report.md"
    report.write_text(
        "Dimer fitting errors are training diagnostics. Bulk temperature increases; dimer fitting cannot establish bulk generalization."
    )
    script = tmp_path / "script.py"
    script.write_text(
        "raise RuntimeError('Submitted code must never be executed by scoring')\n"
    )
    manifest = {
        "artifacts": paths,
        "results": results,
        "settings": _write(tmp_path / "settings.json", settings),
        "report": str(report),
        "scripts": [str(script)],
    }
    _write(tmp_path / "manifest.json", manifest)
    return tmp_path / "manifest.json"


def _score(path):
    rubric = Rubric(3)
    evaluate(Evidence(path), rubric)
    return rubric


def _check(rubric, name):
    checks = rubric["checks"] if isinstance(rubric, dict) else rubric.checks
    return next(check for check in checks if check["name"] == name)


def test_level1_scores_only_the_shared_teacher_dataset(submission):
    manifest = json.loads(submission.read_text())
    manifest["artifacts"] = {"dataset": manifest["artifacts"]["dataset"]}
    manifest["results"] = {}
    settings_path = submission.parent / "level1-settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "teacher_model": "teacher.model",
                "teacher_sha256": PINNED_TEACHER_SHA256,
                "dispersion": True,
                "energy_unit": "eV",
                "force_unit": "eV/Angstrom",
                "calculator": {"default_dtype": "float64"},
            }
        )
    )
    manifest["settings"] = str(settings_path)
    level1_manifest = submission.parent / "level1-manifest.json"
    level1_manifest.write_text(json.dumps(manifest))

    report = check_level1_workflow(3, verification_backend="offline").evaluate(level1_manifest)
    assert report["status"] == "pending_review"
    assert report["level"] == 1
    assert {check["name"] for check in report["checks"] if check["points"]} == {
        "manifest_and_linked_artifacts",
        "recorded_settings",
        "saved_scripts",
        "dimer_separation_dataset",
        "teacher_energy_and_force_labels",
        "teacher_identity_dispersion_and_units",
        "pinned_teacher_digest",
        "independent_model_calculation",
    }
    assert check_level2_workflow(3).evaluate(level1_manifest)["score"] == 0


def test_task3_diagnostic_scope_does_not_require_a_report(submission):
    manifest = json.loads(submission.read_text())
    manifest.pop("report")
    _write(submission, manifest)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert _check(result, "diagnostic_scope")["status"] == "passed"


def test_task3_consistent_artifacts_receive_all_task_points_without_execution(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "Scoring attempted model evaluation or unsafe deserialization"
        )

    monkeypatch.setattr(SinglePointCalculator, "calculate", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    result = _score(submission)
    assert result.score == pytest.approx(0.9), result.checks
    assert sum(check["points"] for check in result.checks) == 90
    assert _check(result, "execution_provenance")["status"] == "unverified"
    assert _check(result, "checkpoint_loadability")["status"] == "unverified"


def test_silver_checkpoint_requires_isolated_inference_for_full_score(submission):
    offline = check_level2_workflow(3, verification_backend="offline")
    pending = offline.evaluate(submission)
    assert pending["score"] is None
    assert pending["pending_checks"] == ["checkpoint_container_and_digest"]
    with pytest.raises(PendingReviewError):
        offline(submission)

    class Verifier:
        def __init__(self, statuses):
            self.statuses = statuses

        def evaluate(self, evidence, task_number):
            assert task_number == 3
            return {
                "evidence_sha256": evidence.fingerprint(),
                "checks": [
                    {"id": name, "status": status, "targets": [], "detail": status}
                    for name, status in self.statuses.items()
                ],
            }

    for statuses, expected, score in (
        ({"student_after": "passed", "trained_bulk_model": "passed"}, "complete", 1),
        (
            {"student_after": "unverified", "trained_bulk_model": "passed"},
            "pending_review",
            None,
        ),
        ({"student_after": "failed", "trained_bulk_model": "passed"}, "complete", 0),
        ({"student_after": "passed"}, "pending_review", None),
    ):
        report = WorkflowScorer(3, verifier=Verifier(statuses)).evaluate(submission)
        assert report["status"] == expected
        assert (
            _check(report, "checkpoint_loadability")["status"]
            == _check(report, "checkpoint_container_and_digest")["status"]
        )
        assert report["score"] == score


def test_silver_uses_modal_verifier_by_default(submission, monkeypatch):
    monkeypatch.delenv("CORRAL_MD_VERIFICATION", raising=False)
    grader = check_level2_workflow(3)
    assert grader.verifier is not None
    assert check_level2_workflow(3, verification_backend="offline").verifier is None

    def unavailable(*_args):
        raise RuntimeError("verifier unavailable")

    monkeypatch.setattr(grader.verifier, "_calculate", unavailable)
    report = grader.evaluate(submission)
    assert report["score"] is None
    assert "checkpoint_container_and_digest" in report["pending_checks"]


@pytest.mark.parametrize(
    ("change", "failed_check"),
    [
        ("dimer_geometry", "dataset_geometry"),
        ("force_shape", "dataset_energy_force_labels"),
        ("wrong_rmse", "aggregate_fitting_errors"),
        ("wrong_separation_error", "separation_resolved_errors"),
        ("checkpoint_digest", "checkpoint_container_and_digest"),
        ("bad_checkpoint", "checkpoint_container_and_digest"),
        ("truncated_md", "fixed_cell_and_five_ps_trajectory"),
        ("bulk_geometry", "initial_bulk_geometry"),
        ("missing_momenta", "temperature_assessment"),
        ("wrong_temperature", "md_log_matches_trajectory"),
        ("training_history", "training_history_and_dataset_identity"),
    ],
)
def test_task3_rejects_inconsistent_evidence(submission, change, failed_check):
    directory = submission.parent
    manifest = json.loads(submission.read_text())
    if change in {"dimer_geometry", "force_shape"}:
        data = json.loads((directory / "dataset.json").read_text())
        if change == "dimer_geometry":
            data[0]["positions"][1][0] = 0.65
        else:
            data[0]["forces"] = [[0, 0, 0]]
        _write(directory / "dataset.json", data)
    elif change in {"wrong_rmse", "wrong_separation_error"}:
        if change == "wrong_rmse":
            manifest["results"]["fitting"]["after"]["force_rmse_eV_A"] = 0.001
        else:
            manifest["results"]["fitting"]["after"]["energy_abs_error_per_atom_eV"][
                2
            ] = 50
        _write(submission, manifest)
    elif change == "checkpoint_digest":
        settings = json.loads((directory / "settings.json").read_text())
        settings["trained_checkpoint_sha256"] = "0" * 64
        _write(directory / "settings.json", settings)
    elif change == "bad_checkpoint":
        (directory / "model.safetensors").write_bytes(b"not a checkpoint")
        settings = json.loads((directory / "settings.json").read_text())
        settings["trained_checkpoint_sha256"] = _digest(directory / "model.safetensors")
        _write(directory / "settings.json", settings)
    elif change in {"truncated_md", "bulk_geometry", "missing_momenta"}:
        frames = json.loads((directory / "md.json").read_text())
        if change == "truncated_md":
            frames[-1]["time_fs"] = 4000
        elif change == "bulk_geometry":
            frames[0]["positions"][0][0] += 0.1
        else:
            del frames[1]["momenta"]
        _write(directory / "md.json", frames)
    elif change == "wrong_temperature":
        log = json.loads((directory / "mdlog.json").read_text())
        log[1]["temperature_K"] = 300
        _write(directory / "mdlog.json", log)
    elif change == "training_history":
        _write(
            directory / "training.json", [{"step": 0, "loss": 1, "learning_rate": 0}]
        )
    result = _score(submission)
    assert _check(result, failed_check)["status"] == "failed", result.checks
    assert result.score < 0.9


def test_task3_accepts_dimer_frame_and_prediction_permutations(submission):
    directory = submission.parent
    dataset = json.loads((directory / "dataset.json").read_text())
    _write(directory / "dataset.json", dataset[::-1])
    settings = json.loads((directory / "settings.json").read_text())
    settings["training_dataset_sha256"] = _digest(directory / "dataset.json")
    _write(directory / "settings.json", settings)
    predictions = json.loads((directory / "after.json").read_text())
    _write(
        directory / "after.json",
        {key: values[::-1] for key, values in predictions.items()},
    )
    assert _score(submission).score == pytest.approx(0.9)


def test_task3_honest_incomplete_md_keeps_diagnostic_credit(submission):
    directory = submission.parent
    for file in ("md.json", "mdlog.json"):
        rows = json.loads((directory / file).read_text())
        for row in rows:
            row["time_fs"] /= 2
        _write(directory / file, rows)
    result = _score(submission)
    assert _check(result, "fixed_cell_and_five_ps_trajectory")["status"] == "failed"
    assert _check(result, "md_log_covers_complete_run")["status"] == "failed"
    assert _check(result, "temperature_assessment")["status"] == "passed"
    assert _check(result, "energy_and_force_assessment")["status"] == "passed"
    assert result.score == pytest.approx(0.8)


def test_task3_missing_and_malformed_artifacts_are_failed_checks(submission):
    (submission.parent / "after.json").write_text("not JSON")
    (submission.parent / "mdlog.json").unlink()
    result = _score(submission)
    assert _check(result, "before_after_predictions")["status"] == "failed"
    assert _check(result, "md_log_matches_trajectory")["status"] == "failed"


def test_task3_empty_submission_awards_no_scientific_points(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, {"results": {}, "artifacts": {}})
    assert _score(path).score == 0


def test_alternative_error_and_stability_summaries_receive_full_credit(submission):
    manifest = json.loads(submission.read_text())
    for phase in ("before", "after"):
        claim = manifest["results"]["fitting"][phase]
        reference = json.loads((submission.parent / "dataset.json").read_text())
        predicted = json.loads((submission.parent / f"{phase}.json").read_text())
        energies = (
            np.asarray(predicted["energy_eV"])
            - np.array([r["energy"] for r in reference])
        ) / 2
        forces = np.asarray(predicted["forces_eV_A"]) - np.array(
            [r["forces"] for r in reference]
        )
        claim.pop("energy_rmse_per_atom_eV")
        claim.pop("force_rmse_eV_A")
        claim.pop("force_rmse_by_separation_eV_A")
        claim.update(
            energy_mae_per_atom_eV=np.abs(energies).mean(),
            force_mae_eV_A=np.abs(forces).mean(),
            force_mae_by_separation_eV_A=np.abs(forces).mean(axis=(1, 2)).tolist(),
        )
    bulk_results = manifest["results"]["bulk"]
    for key in (
        "temperature_min_K",
        "temperature_max_K",
        "temperature_std_K",
        "final_min_pair_distance_A",
    ):
        bulk_results.pop(key)
    bulk_results.pop("potential_energy_range_eV")
    bulk_results["potential_energy_std_eV"] = np.std([-400, -399.5, -399])
    settings = json.loads((submission.parent / "settings.json").read_text())
    for key in (
        "teacher_sha256",
        "student_sha256",
        "training_dataset_sha256",
        "trained_checkpoint_sha256",
    ):
        settings.pop(key)
    _write(submission.parent / "settings.json", settings)
    _write(submission, manifest)
    assert _score(submission).score == pytest.approx(0.9)
    manifest["results"]["fitting"]["after"]["force_mae_eV_A"] += 2
    _write(submission, manifest)
    assert _check(_score(submission), "aggregate_fitting_errors")["status"] == "failed"


def test_rotated_bulk_geometry_is_equivalent(submission):
    frames = json.loads((submission.parent / "md.json").read_text())
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    for frame in frames:
        for field in ("positions", "cell", "momenta", "forces"):
            frame[field] = (np.asarray(frame[field]) @ rotation).tolist()
    _write(submission.parent / "md.json", frames)
    assert _score(submission).score == pytest.approx(0.9)


def test_alternate_numerical_training_history_is_reviewable(submission):
    _write(
        submission.parent / "training.json",
        [
            {"iteration": 0, "objective": -2.0, "gradient_norm": 0.5},
            {"iteration": 1, "objective": -3.0, "gradient_norm": 0.01},
        ],
    )
    rubric = _score(submission)
    assert (
        _check(rubric, "training_history_and_dataset_identity")["status"]
        == "unverified"
    )
    assert _check(rubric, "aggregate_fitting_errors")["status"] == "passed"
    assert rubric.score is None
    settings = json.loads((submission.parent / "settings.json").read_text())
    settings["training_dataset_sha256"] = "0" * 64
    _write(submission.parent / "settings.json", settings)
    assert (
        _check(_score(submission), "training_history_and_dataset_identity")["status"]
        == "failed"
    )


def test_legacy_checkpoint_needs_review_without_deserialization(
    submission, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Checkpoint must not be deserialized")

    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    path = submission.parent / "model.safetensors"
    path.write_bytes(pickle.dumps({"weights": [1, 2, 3]}))
    settings = json.loads((submission.parent / "settings.json").read_text())
    settings["trained_checkpoint_sha256"] = _digest(path)
    _write(submission.parent / "settings.json", settings)
    assert (
        _check(_score(submission), "checkpoint_container_and_digest")["status"]
        == "unverified"
    )
