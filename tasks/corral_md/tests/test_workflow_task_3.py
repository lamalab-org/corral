"""Level 1 verifies every silver dimer label against the reference model."""

import json
from pathlib import Path

import numpy as np
import pytest
from corral_md.scoring.common import Evidence
from corral_md.scoring.evaluator import (
    EvaluationError,
    WorkflowScorer,
    check_level1_workflow,
)
from corral_md.scoring.task_3 import build_plan


def write(path, value):
    Path(path).write_text(json.dumps(value))
    return str(path)


class SyntheticVerifier:
    """Controlled teacher responses derived from the submitted geometry."""

    def _calculate(self, plan, fingerprint):
        (job,) = plan.jobs
        assert job["id"] == "teacher"
        rows = []
        for frame in job["frames"]:
            distance = np.linalg.norm(np.diff(frame["positions"], axis=0))
            forces = np.zeros((2, 3))
            forces[:, 0] = [1 / distance**2, -1 / distance**2]
            rows.append({"energy": -1 / distance, "forces": forces.tolist()})
        return {
            "evidence_sha256": fingerprint,
            "challenge": plan.challenge,
            "jobs": [
                {"id": "teacher", "status": "complete", "value": {"frames": rows}}
            ],
        }


@pytest.fixture
def submission(tmp_path):
    reference = []
    for distance in np.linspace(0.6, 5, 45):
        forces = np.zeros((2, 3))
        forces[:, 0] = [1 / distance**2, -1 / distance**2]
        reference.append(
            {
                "symbols": ["Ag", "Ag"],
                "positions": [[0, 0, 0], [distance, 0, 0]],
                "pbc": False,
                "energy": -1 / distance,
                "forces": forces.tolist(),
            }
        )
    write(tmp_path / "dataset.json", reference)
    write(
        tmp_path / "run.json",
        {
            "teacher_model": "teacher.model",
            "dispersion": True,
            "default_dtype": "float64",
            "energy_unit": "eV",
            "force_unit": "eV/Angstrom",
        },
    )
    path = tmp_path / "answer.json"
    write(path, {"artifacts": {"dataset": "dataset.json", "run_log": "run.json"}})
    return path


def score(path, verifier=None):
    return WorkflowScorer(
        3, level=1, verifier=verifier or SyntheticVerifier()
    ).evaluate(path)


def test_level1_needs_only_reference_calculations(submission):
    report = score(submission)
    assert report["score"] == 1, report
    assert sum(c["points"] for c in report["checks"]) == 100
    plan = build_plan(Evidence(submission), "challenge")
    assert [job["id"] for job in plan.jobs] == ["teacher"]
    assert len(plan.jobs[0]["frames"]) == 45
    assert not plan.files


@pytest.mark.parametrize(
    "change",
    [
        "grid",
        "labels",
        "missing_forces",
        "wrong_teacher",
        "no_dispersion",
        "units",
        "nan",
    ],
)
def test_bad_reference_calculations_fail(submission, change):
    root = submission.parent
    log = json.loads((root / "run.json").read_text())
    reference = json.loads((root / "dataset.json").read_text())
    if change == "grid":
        reference[0]["positions"][1][0] += 0.1
    elif change == "labels":
        reference[0]["energy"] += 100
    elif change == "missing_forces":
        del reference[0]["forces"]
    elif change == "wrong_teacher":
        log["teacher_model"] = "student.model"
    elif change == "no_dispersion":
        log["dispersion"] = False
    elif change == "units":
        log["energy_unit"] = "kcal/mol"
    else:
        reference[0]["positions"][0][0] = float("nan")
    write(root / "run.json", log)
    write(root / "dataset.json", reference)
    assert score(submission)["score"] == 0


def test_unavailable_backend_raises_evaluation_error(submission):
    scorer = check_level1_workflow(3, verification_backend="offline")
    with pytest.raises(EvaluationError):
        scorer.evaluate(submission)
    with pytest.raises(EvaluationError):
        scorer(submission)


def test_frozen_evidence_is_checked(submission):
    class Mutating(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            response = super()._calculate(plan, fingerprint)
            (submission.parent / "dataset.json").write_text("[]")
            return response

    with pytest.raises(RuntimeError, match="changed during grading"):
        score(submission, verifier=Mutating())


def test_dimer_order_does_not_change_score(submission):
    path = submission.parent / "dataset.json"
    write(path, json.loads(path.read_text())[::-1])
    assert score(submission)["score"] == 1


@pytest.mark.parametrize(
    "failure", ["identity", "missing", "unavailable", "failed", "nonfinite"]
)
def test_independent_reference_failure_is_not_credited(submission, failure):
    class Broken(SyntheticVerifier):
        def _calculate(self, plan, fingerprint):
            result = super()._calculate(plan, fingerprint)
            if failure == "identity":
                result["evidence_sha256"] = "another-submission"
            elif failure == "missing":
                result["jobs"] = []
            elif failure == "unavailable":
                result["jobs"][0]["status"] = "unsupported"
            elif failure == "failed":
                result["jobs"][0]["status"] = "error"
            else:
                result["jobs"][0]["value"]["frames"][0]["energy"] = float("nan")
            return result

    if failure == "unavailable":
        with pytest.raises(EvaluationError):
            score(submission, verifier=Broken())
    else:
        report = score(submission, verifier=Broken())
        assert report["score"] == 0
        json.dumps(report, allow_nan=False)


def test_reference_settings_use_aliases_and_explicit_calculator_description(submission):
    path = submission.parent / "run.json"
    log = json.loads(path.read_text())
    log["teacher_model"] = "/workspace/resources/models/teacher.model"
    log["energy_units"] = log.pop("energy_unit")
    log["force_units"] = "eV/A"
    log.pop("force_unit")
    log["calculator"] = (
        f"MACECalculator(dispersion=True, default_dtype={log.pop('default_dtype')})"
    )
    log.pop("dispersion")
    write(path, log)
    assert score(submission)["score"] == 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reference_plan_or_live_fallback(submission, dtype):
    path = submission.parent / "run.json"
    log = json.loads(path.read_text())
    log["default_dtype"] = dtype
    write(path, log)
    (job,) = build_plan(Evidence(submission), "challenge").jobs
    assert job["operation"] == ("reference" if dtype == "float64" else "mace")
    assert job["parameters"]["dispersion"] is True
