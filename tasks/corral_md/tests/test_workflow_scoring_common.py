"""Submission, diagnostic and no-execution guarantees shared by all workflows."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from corral_md.score import (
    PendingReviewError,
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
    main,
)
from corral_md.submission import resolve_submission
from corral_md.workflow_scoring.common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
    is_teacher_model,
    level1_reproducibility,
    reproducibility,
)


@pytest.mark.parametrize(
    "value",
    [
        "MACE-MP-0",
        "/workspace/models/teacher.model",
        "teacher.model",
        {"identity": "MACE-MP-0", "path": "/workspace/models/teacher.model"},
    ],
)
def test_teacher_declaration_accepts_pinned_filename(value):
    assert is_teacher_model(value)


@pytest.mark.parametrize(
    "value",
    [
        "MACE-MP-0b",
        "student.model",
        "not-mace-mp-01",
        {},
        {"identity": "MACE-MP-0", "path": "student.model"},
    ],
)
def test_teacher_declaration_rejects_other_or_conflicting_models(value):
    assert not is_teacher_model(value)


def test_columnar_json_frames_keep_labels_geometry_and_time_aligned(tmp_path):
    data = {
        "symbols": ["Ag", "Ag"],
        "pbc": False,
        "cell_A": np.eye(3).tolist(),
        "positions_A": [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]],
        "energy_eV": [2, 3],
        "forces_eV_A": np.zeros((2, 2, 3)).tolist(),
        "elapsed_time_fs": [0, 10],
        "stage": ["equilibration", "production"],
        "info": [{"row_id": "a"}, {"row_id": "b"}],
    }
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(data))
    frames = Evidence({"artifacts": {"dataset": str(path)}}).trajectory("dataset")
    assert len(frames) == 2
    assert all(len(a) == 2 for a in frames)
    assert [a.calc.results["energy"] for a in frames] == [2, 3]
    assert [a.info["time_fs"] for a in frames] == [0, 10]
    assert [a.info["stage"] for a in frames] == ["equilibration", "production"]
    assert [a.info["row_id"] for a in frames] == ["a", "b"]
    assert frames[1].positions[1, 0] == 2
    data["energy_eV"] = [2]
    path.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match="one entry per frame"):
        Evidence({"artifacts": {"dataset": str(path)}}).trajectory("dataset")


def test_conflicting_frame_time_aliases_are_not_silently_selected(tmp_path):
    path = tmp_path / "frames.json"
    path.write_text(
        json.dumps(
            [
                {
                    "symbols": ["Ag"],
                    "positions": [[0, 0, 0]],
                    "pbc": False,
                    "time_fs": 1,
                    "elapsed_time_fs": 2,
                }
            ]
        )
    )
    with pytest.raises(EvidenceError, match="Conflicting evidence fields"):
        Evidence({"artifacts": {"dataset": str(path)}}).trajectory("dataset")


def test_document_sidecars_must_be_confined_and_manifest_linked(manifest, tmp_path):
    root, path, _ = manifest
    e = Evidence(resolve_submission(str(path), root))
    assert e.linked_path("data.json", path) == path.parent / "data.json"
    (path.parent / "unlisted.npz").write_bytes(b"not evidence")
    with pytest.raises(EvidenceError, match="listed in manifest.artifacts"):
        e.linked_path("unlisted.npz", path)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    with pytest.raises(EvidenceError, match="escapes"):
        e.linked_path(str(outside), path)


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "workspace"
    output = root / "outputs"
    output.mkdir(parents=True)
    (output / "settings.json").write_text(
        json.dumps({"model": "MACE-MP-0", "unit": "eV"})
    )
    (output / "analysis.json").write_text(
        json.dumps({"interpretation": "A supported conclusion"})
    )
    (output / "run.py").write_text(
        "raise RuntimeError('submitted scripts must never run')"
    )
    (output / "data.json").write_text(json.dumps([1, 2, 3]))
    data = {
        "results": {"value": 3.0, "unit": "eV", "status": "measured"},
        "settings": "settings.json",
        "report": "analysis.json",
        "scripts": ["run.py"],
        "artifacts": {"data": "data.json"},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(data))
    return root, path, data


def test_shared_manifest_preserves_metadata_and_resolves_only_paths(manifest):
    root, path, data = manifest
    original = path.read_bytes()
    resolved = resolve_submission(str(path), root)
    assert json.loads(resolved)["results"] == data["results"]
    evidence = Evidence(resolved)
    assert evidence.root == root
    assert evidence.settings["model"] == "MACE-MP-0"
    assert evidence.json("data") == [1, 2, 3]
    assert evidence.result("unit") == "eV"
    assert path.read_bytes() == original


@pytest.mark.parametrize("report", [None, "missing.md", "analysis.json"])
def test_shared_credit_does_not_require_a_narrative_report(manifest, report):
    root, path, data = manifest
    if report is None:
        data.pop("report")
    else:
        data["report"] = report
    path.write_text(json.dumps(data))
    rubric = Rubric()
    reproducibility(Evidence(resolve_submission(str(path), root)), rubric)
    assert sum(check["points"] for check in rubric.checks) == 10
    assert sum(check["earned"] for check in rubric.checks) == 10
    assert all(check["name"] != "analysis_report" for check in rubric.checks)


def test_level1_reproducibility_ignores_unused_level2_slots(manifest):
    root, path, data = manifest
    data["results"] = {}
    data["artifacts"].update(
        {
            "later_scalar": "",
            "later_list": [],
            "later_nested": {"level2_only": ""},
        }
    )
    path.write_text(json.dumps(data))
    rubric = Rubric()
    level1_reproducibility(Evidence(resolve_submission(str(path), root)), rubric)
    assert sum(check["points"] for check in rubric.checks) == 10
    assert sum(check["earned"] for check in rubric.checks) == 10


def test_missing_link_preserves_independent_evidence(manifest):
    root, path, data = manifest
    data["artifacts"]["missing"] = "missing.csv"
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(str(path), root))
    assert evidence.json("data") == [1, 2, 3]
    with pytest.raises(EvidenceError, match="Missing artifact"):
        evidence.table("missing")


@pytest.mark.parametrize("relocated", [False, True])
@pytest.mark.parametrize("other_runs", [1, 2])
def test_missing_run_artifact_cannot_bind_to_another_run(
    manifest, relocated, other_runs
):
    root, path, data = manifest
    for i in range(other_runs):
        run = root / f"run{i}"
        run.mkdir()
        (run / "trajectory.json").write_text("[99]")
    data["artifacts"]["missing"] = (
        "/old/workspace/runA/trajectory.json" if relocated else "runA/trajectory.json"
    )
    path.write_text(json.dumps(data))
    answer = "/old/workspace/outputs/manifest.json" if relocated else str(path)
    evidence = Evidence(resolve_submission(answer, root))
    assert evidence.json("data") == [1, 2, 3]
    with pytest.raises(EvidenceError, match="Missing artifact"):
        evidence.json("missing")


def test_missing_absolute_artifact_does_not_bind_to_same_basename(manifest):
    root, path, data = manifest
    data["artifacts"]["missing"] = str(root / "other/data.json")
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(str(path), root))
    assert evidence.json("data") == [1, 2, 3]
    with pytest.raises(EvidenceError, match="Missing artifact"):
        evidence.json("missing")


@pytest.mark.parametrize(
    "answer", ["runA/manifest.json", "/old/workspace/runA/manifest.json"]
)
def test_missing_manifest_does_not_bind_to_another_run(tmp_path, answer):
    other = tmp_path / "runB"
    other.mkdir()
    (other / "manifest.json").write_text('{"results": {"value": 99}}')
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_submission(answer, tmp_path)


def test_result_strings_are_not_paths_even_when_they_look_like_paths(manifest):
    root, path, data = manifest
    data["results"]["unit"] = "m2/s"
    data["results"]["explanation"] = "../this is prose"
    path.write_text(json.dumps(data))
    assert Evidence(resolve_submission(str(path), root)).result("unit") == "m2/s"


@pytest.mark.parametrize("link", ["../../outside.json", "/external/missing.json"])
def test_workspace_confinement(manifest, link):
    root, path, data = manifest
    data["artifacts"]["data"] = link
    path.write_text(json.dumps(data))
    if ".." in link:
        with pytest.raises(ValueError):
            resolve_submission(str(path), root)
    else:
        evidence = Evidence(resolve_submission(str(path), root))
        with pytest.raises(EvidenceError):
            evidence.json("data")


def test_object_arrays_cannot_be_deserialized(manifest):
    root, path, data = manifest
    np.save(path.parent / "objects.npy", np.array([{"not": "numeric"}], dtype=object))
    data["artifacts"]["objects"] = "objects.npy"
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(str(path), root))
    with pytest.raises(ValueError):
        evidence.array("objects")


@pytest.mark.parametrize("pbc", [None, "true", [1, 1, 1]])
def test_json_frames_require_boundary_condition_evidence(manifest, pbc):
    root, path, _ = manifest
    frame = {"symbols": ["Al"], "positions": [[0, 0, 0]], "cell": [4, 4, 4]}
    if pbc is not None:
        frame["pbc"] = pbc
    (path.parent / "data.json").write_text(json.dumps([frame]))
    evidence = Evidence(resolve_submission(str(path), root))
    with pytest.raises(EvidenceError, match="explicit boolean pbc"):
        evidence.trajectory("data")


def test_rubric_distinguishes_failure_and_unverified():
    rubric = Rubric(1)
    rubric.check("valid", 25, True)
    rubric.check("invalid", 25, lambda: (False, "wrong value"))
    rubric.check("unsupported", 25, None)
    rubric.check("missing", 25, lambda: {}["missing"])
    assert rubric.score is None
    assert rubric.as_dict()["score_bounds"] == [0.25, 0.5]
    assert rubric.as_dict()["pending_points"] == 25
    assert [c["status"] for c in rubric.checks] == [
        "passed",
        "failed",
        "unverified",
        "failed",
    ]
    assert "KeyError" in rubric.checks[-1]["detail"]


def test_binary_rubric_is_terminal_on_failure_and_keeps_pending_distinct():
    failed = Rubric(1, binary=True)
    failed.check("valid", 25, True)
    failed.check("invalid", 25, False)
    failed.check("unsupported", 50, None)
    assert failed.score == 0
    assert failed.as_dict()["score_bounds"] == [0, 0]
    assert failed.as_dict()["status"] == "complete"

    pending = Rubric(1, binary=True)
    pending.check("valid", 50, True)
    pending.check("unsupported", 50, None)
    assert pending.score is None
    assert pending.as_dict()["score_bounds"] == [0, 1]


def test_workflow_failure_skips_later_checks_and_modal_verification(
    manifest, monkeypatch
):
    _, path, _ = manifest
    evaluated = []

    def evaluate(_evidence, rubric):
        rubric.check("reported_diffusion", 45, False, "first failure")
        rubric.check(
            "diffusion_estimator_and_units",
            45,
            lambda: evaluated.append("expensive") or True,
        )

    class Verifier:
        def evaluate(self, *_args):
            raise AssertionError("Modal verification must not run after a failure")

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(lambda _self: SimpleNamespace(evaluate=evaluate)),
    )
    report = WorkflowScorer(1, verifier=Verifier()).evaluate(path)
    assert report["score"] == 0
    assert evaluated == []
    assert report["checks"][-2]["status"] == "skipped"


def test_failed_reproducibility_check_skips_task_evaluation(manifest, monkeypatch):
    _, path, data = manifest
    data["artifacts"]["missing"] = "missing.json"
    path.write_text(json.dumps(data))

    def evaluate(*_args):
        raise AssertionError("Task checks must not run after reproducibility fails")

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(lambda _self: SimpleNamespace(evaluate=evaluate)),
    )
    report = WorkflowScorer(1).evaluate(path)
    assert report["score"] == 0
    assert report["checks"][-2]["name"] == "remaining_evidence"


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({"status": "finished", "units": "eV"}, 0),
        ({"completed": True}, 0),
        ({"energy": {"mean": -3.5, "units": "eV"}}, 1),
        ({"energy": float("nan")}, 0),
    ],
)
def test_result_credit_requires_finite_numerical_results(results, expected):
    rubric = Rubric()
    reproducibility(Evidence({"results": results}), rubric)
    check = next(c for c in rubric.checks if c["name"] == "reported_results")
    assert check["earned"] == expected


def test_factory_returns_scalar_and_keeps_json_checks_task_local(manifest, monkeypatch):
    root, path, _ = manifest

    def evaluate(e, r):
        r.check("reported_diffusion", 90, lambda: e.json("data") == [1, 2, 3])

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(lambda _self: SimpleNamespace(evaluate=evaluate)),
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    scorer = check_level2_workflow(1)
    submission = resolve_submission(str(path), root)
    good = scorer(submission)
    assert type(good) is float
    assert good == 1.0
    report = scorer.evaluate(submission)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["mode"] == "artifact_only"
    assert report["checks"][-1]["status"] == "unverified"
    bad = scorer(3.5)
    assert bad == 0
    assert good == 1
    assert report["score"] == 1
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("number", [0, 11, True, 1.5])
def test_invalid_task_number(number):
    with pytest.raises(ValueError):
        check_level2_workflow(number)
    with pytest.raises(ValueError):
        check_level1_workflow(number)


def test_level1_factory_uses_independent_model_verification():
    scorer = check_level1_workflow(3)
    assert scorer.task_number == 3
    assert scorer.level == 1
    assert scorer.verifier is not None
    assert check_level1_workflow(3, verification_backend="modal").verifier is not None
    assert check_level1_workflow(3, verification_backend="offline").verifier is None
    assert check_level1_workflow(2).verifier is None


def _pending_scorer(monkeypatch):
    def evaluate(e, r):
        def alternative():
            assert e.json("data") == [1, 2, 3]
            raise UnsupportedEvidence("Retained alternate calculation needs review")

        r.check("reported_diffusion", 90, alternative)

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(lambda _: SimpleNamespace(evaluate=evaluate)),
    )
    return check_level2_workflow(1)


def _review(report, *, passed=True):
    return {
        "evidence_sha256": report["evidence_sha256"],
        "decisions": {
            "reported_diffusion": {
                "passed": passed,
                "reviewer": "independent evaluator",
                "reason": "Checked retained data and calculation against the diffusion requirement",
            }
        },
    }


def test_pending_score_is_not_zero_and_trusted_review_completes_it(
    manifest, monkeypatch
):
    _, path, _ = manifest
    scorer = _pending_scorer(monkeypatch)
    pending = scorer.evaluate(path)
    assert pending["score"] is None
    assert pending["status"] == "pending_review"
    assert pending["score_bounds"] == [0.0, 1.0]
    assert pending["possible_points"] == 100
    assert len(pending["evidence_sha256"]) == 64
    json.dumps(pending, allow_nan=False)
    with pytest.raises(PendingReviewError) as error:
        scorer(path)
    assert error.value.report["pending_checks"] == ["reported_diffusion"]
    reviewed = scorer.evaluate(path, review=_review(pending))
    assert reviewed["score"] == 1
    assert reviewed["status"] == "complete"
    assert reviewed["pending_checks"] == []
    assert scorer.score_submission(path, review=_review(pending, passed=False)) == 0


@pytest.mark.parametrize(
    "changed",
    ["manifest.json", "data.json", "settings.json", "run.py", "analysis.json"],
)
def test_review_is_bound_to_all_evidence_bytes(manifest, monkeypatch, changed):
    _, path, _ = manifest
    scorer = _pending_scorer(monkeypatch)
    review = _review(scorer.evaluate(path))
    artifact = path.parent / changed
    artifact.write_text(artifact.read_text() + "\n")
    # JSON whitespace in the manifest leaves its semantic identity unchanged;
    # change an actual result as well for this case.
    if changed == "manifest.json":
        data = json.loads(path.read_text())
        data["results"]["value"] = 4
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evidence_sha256"):
        scorer.evaluate(path, review=review)


def test_submission_authored_review_is_ignored(manifest, monkeypatch):
    _, path, data = manifest
    scorer = _pending_scorer(monkeypatch)
    data["review"] = _review(scorer.evaluate(path))
    path.write_text(json.dumps(data))
    assert scorer.evaluate(path)["status"] == "pending_review"


def test_review_cannot_override_failed_checks_or_partially_apply_invalid_bundle():
    rubric = Rubric(1)
    rubric.check("pending", 40, None)
    rubric.check("failed", 60, False)
    decision = {"passed": True, "reviewer": "reviewer", "reason": "reason"}
    with pytest.raises(ValueError, match="only pending"):
        rubric.apply_review(
            {
                "evidence_sha256": "digest",
                "decisions": {
                    "pending": decision,
                    "failed": decision,
                },
            },
            "digest",
        )
    assert rubric.checks[0]["status"] == "unverified"
    assert rubric.checks[1]["status"] == "failed"


def test_zero_point_provenance_does_not_block_completed_score():
    rubric = Rubric(1)
    rubric.check("complete", 100, True)
    rubric.unverified("execution", "Saved data cannot establish actual execution")
    assert rubric.score == 1
    assert rubric.as_dict()["status"] == "complete"


def test_cli_writes_pending_report_then_resolves_trusted_review(
    manifest, monkeypatch, tmp_path
):
    _, path, _ = manifest
    _pending_scorer(monkeypatch)
    output, review_path = tmp_path / "evaluation.json", tmp_path / "review.json"
    argv = ["workflow_scoring", "1", str(path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 2
    pending = json.loads(output.read_text())
    review_path.write_text(json.dumps(_review(pending)))
    monkeypatch.setattr(sys, "argv", [*argv, "--review", str(review_path)])
    assert main() == 0
    assert json.loads(output.read_text())["score"] == 1


def test_malformed_direct_manifest_and_nonfinite_pending_result_do_not_crash(
    manifest, monkeypatch
):
    root, path, _ = manifest
    scorer = _pending_scorer(monkeypatch)
    assert scorer.evaluate({"results": {"value": object()}})["score"] == 0
    data = json.loads(resolve_submission(str(path), root))
    data["results"]["value"] = float("nan")
    report = scorer.evaluate(data)
    assert report["score"] == 0
    assert report["status"] == "complete"
    assert (
        next(c for c in report["checks"] if c["name"] == "reported_results")["status"]
        == "failed"
    )
    json.dumps(report, allow_nan=False)
