"""Focused checks for Level 1's evaluator-owned MACE verification plan."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.level1_trusted import (
    MODEL_TARGET,
    PINNED_TEACHER_SHA256,
    Level1ModalVerifier,
    build_level1_plan,
    replay_task8_default_rng,
    teacher_digest_matches,
)
from corral_md.workflow_scoring.verification import apply_verification


def _document(tmp_path, settings, artifacts, *, results=None):
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    manifest = {
        "settings": "settings.json",
        "artifacts": artifacts,
        "results": results or {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return Evidence(path)


def _dimers(tmp_path):
    frames = []
    for distance in np.round(np.arange(0.6, 5.01, 0.1), 8):
        atoms = Atoms("Ag2", positions=[[-distance / 2, 0, 0], [distance / 2, 0, 0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(distance),
            forces=np.zeros((2, 3)),
        )
        frames.append(atoms)
    write(tmp_path / "dimers.extxyz", frames)
    return _document(
        tmp_path,
        {
            "teacher_model": "teacher.model",
            "teacher_model_sha256": PINNED_TEACHER_SHA256,
            "dispersion": True,
            "default_dtype": "float64",
        },
        {"dataset": "dimers.extxyz"},
    )


def test_level1_dimer_plan_contains_only_geometry_and_pinned_model(tmp_path):
    evidence = _dimers(tmp_path)
    plan = build_level1_plan(evidence, 3, "trusted-challenge")
    assert not plan.notes
    assert len(plan.jobs) == 1
    job = plan.jobs[0]
    assert job["operation"] == "reference"
    assert job["parameters"] == {
        "model": "teacher.model",
        "default_dtype": "float64",
        "dispersion": True,
    }
    assert len(job["frames"]) == 45
    assert all(
        "energy" not in frame and "forces" not in frame for frame in job["frames"]
    )
    assert plan.expected[job["id"]]["targets"] == [MODEL_TARGET]


def test_level1_verifier_passes_only_matching_independent_values(tmp_path):
    evidence = _dimers(tmp_path)

    def transport(plan, fingerprint):
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

    passing = Level1ModalVerifier(transport=transport).evaluate(evidence, 3)
    assert [(item["id"], item["status"]) for item in passing["checks"]] == [
        ("teacher_dimer_labels", "passed")
    ]

    def altered(plan, fingerprint):
        result = transport(plan, fingerprint)
        result["jobs"][0]["value"]["frames"][0]["energy"] += 1
        return result

    failing = Level1ModalVerifier(transport=altered).evaluate(evidence, 3)
    assert failing["checks"][0]["status"] == "failed"


def test_digest_rejects_wrong_or_conflicting_claims(tmp_path):
    assets = json.loads(
        (Path(__file__).resolve().parents[1] / "modal_app/assets.json").read_text()
    )
    assert assets["models"]["teacher.model"]["sha256"] == PINNED_TEACHER_SHA256
    for number, field in (
        (3, "teacher_model_sha256"),
        (4, "checkpoint_sha256"),
        (5, "model_sha256"),
        (8, "teacher_sha256"),
    ):
        settings = {field: PINNED_TEACHER_SHA256}
        if number == 4:
            settings["checkpoint"] = "/workspace/models/teacher.model"
        evidence = _document(tmp_path, settings, {})
        assert teacher_digest_matches(evidence, number)
        evidence.settings[field] = "a" * 64
        assert not teacher_digest_matches(evidence, number)
    evidence = _document(
        tmp_path,
        {
            "checkpoint": "/workspace/models/teacher.model",
            "checkpoint_sha256": PINNED_TEACHER_SHA256,
        },
        {},
        results={"model_sha256": "a" * 64},
    )
    assert not teacher_digest_matches(evidence, 4)
    evidence.settings["checkpoint"] = "/workspace/models/student.model"
    evidence.results["model_sha256"] = PINNED_TEACHER_SHA256
    assert not teacher_digest_matches(evidence, 4)


def test_task8_recorded_default_rng_replays_all_structures(tmp_path):
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(20250308)
    frames = []
    for index in range(100):
        atoms = ideal.copy()
        atoms.positions += rng.normal(0, 0.01 + 0.001 * index, size=(64, 3))
        atoms.info["row_id"] = f"row_{index}"
        frames.append(atoms)
    write(tmp_path / "frames.extxyz", frames)
    data = {
        "train": {
            "row_ids": [f"row_{index}" for index in range(100)],
            "generation_index": list(range(100)),
            "energy_eV": [0] * 100,
        }
    }
    (tmp_path / "regression.json").write_text(json.dumps(data))
    evidence = _document(
        tmp_path,
        {"generation": {"train": {"seed": 20250308, "library": "NumPy default_rng"}}},
        {"train_structures": "frames.extxyz", "regression_data": "regression.json"},
    )
    assert replay_task8_default_rng(evidence)
    evidence.trajectory("train_structures")[37].positions[3, 1] += 0.001
    assert not replay_task8_default_rng(evidence)


def test_task6_unlabeled_trajectory_uses_aligned_raw_trace_energy(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05).repeat((4, 4, 4))
    frames = [atoms.copy() for _ in range(3)]
    write(tmp_path / "trajectory.extxyz", frames)
    rows = [
        {"stage": "nvt", "step": step, "potential_energy_eV": energy}
        for step, energy in ((0, -238.0), (50, -237.5), (100, -237.0))
    ]
    (tmp_path / "trace.json").write_text(json.dumps(rows))
    evidence = _document(
        tmp_path,
        {"model": "teacher.model", "md": {"sample_interval_steps": 50}},
        {
            "equilibration_trajectory": "trajectory.extxyz",
            "thermal_trace": "trace.json",
        },
    )
    plan = build_level1_plan(evidence, 6, "trusted-challenge")
    assert not plan.notes
    assert len(plan.jobs) == 1
    assert plan.jobs[0]["properties"] == ["energy"]
    assert len(plan.jobs[0]["frames"]) == 3
    expected = plan.expected["aluminum_equilibration"]["values"]["frames"]
    assert [item["energy"] for item in expected] == [-238.0, -237.5, -237.0]


def test_task6_trace_skips_duplicate_initial_ase_frame(tmp_path):
    initial = bulk("Al", "fcc", a=4.05).repeat((4, 4, 4))
    frames = [initial.copy()]
    for index, energy in enumerate((-238.0, -237.5, -237.0)):
        atoms = initial.copy()
        atoms.positions[0, 0] += 0.01 * index
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        frames.append(atoms)
    write(tmp_path / "trajectory.traj", frames)
    rows = [
        {"stage": "nvt", "step": step, "potential_energy_eV": energy}
        for step, energy in ((0, -238.0), (50, -237.5), (100, -237.0))
    ]
    (tmp_path / "trace.json").write_text(json.dumps(rows))
    evidence = _document(
        tmp_path,
        {"model": "teacher.model", "md": {"sample_interval_steps": 50, "steps": 100}},
        {
            "equilibration_trajectory": "trajectory.traj",
            "thermal_trace": "trace.json",
        },
    )
    plan = build_level1_plan(evidence, 6, "trusted-challenge")
    assert not plan.notes
    assert len(plan.jobs[0]["frames"]) == 3
    assert [row["positions"][0][0] for row in plan.jobs[0]["frames"]] == [
        initial.positions[0, 0] + 0.01 * index for index in range(3)
    ]
    expected = plan.expected["aluminum_equilibration"]["values"]["frames"]
    assert [item["energy"] for item in expected] == [-238.0, -237.5, -237.0]


def test_required_independent_calculation_keeps_binary_score_pending_until_passed():
    def rubric():
        result = Rubric(3, binary=True)
        result.check("saved_evidence", 99, True)
        result.unverified(MODEL_TARGET, "Awaiting trusted calculation", points=1)
        return result

    pending = rubric()
    apply_verification(pending, None)
    assert pending.score is None

    passed = rubric()
    apply_verification(
        passed,
        {"checks": [{"id": "spot", "status": "passed", "targets": [MODEL_TARGET]}]},
    )
    assert passed.score == 1.0

    failed = rubric()
    apply_verification(
        failed,
        {"checks": [{"id": "spot", "status": "failed", "targets": [MODEL_TARGET]}]},
    )
    assert failed.score == 0.0

    unavailable = rubric()
    apply_verification(
        unavailable,
        {"checks": [{"id": "spot", "status": "unverified", "targets": [MODEL_TARGET]}]},
    )
    assert unavailable.score is None
    with pytest.raises(ValueError, match="trusted verification"):
        unavailable.apply_review(
            {
                "evidence_sha256": "frozen-evidence",
                "decisions": {
                    MODEL_TARGET: {
                        "passed": True,
                        "reviewer": "evaluator",
                        "reason": "No remote calculation was available",
                    }
                },
            },
            "frozen-evidence",
        )
