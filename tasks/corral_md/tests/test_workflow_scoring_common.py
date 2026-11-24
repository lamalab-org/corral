"""Submission, diagnostic and no-execution guarantees shared by all workflows."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from corral_md.scoring.common import (
    Evidence,
    EvidenceError,
    Rubric,
    UnsupportedEvidence,
)
from corral_md.scoring.evaluator import (
    EvaluationError,
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
    main,
)
from corral_md.submission import resolve_submission


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
    e = Evidence(resolve_submission(path.read_text(), root))
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
        "settings": str(output / "settings.json"),
        "report": str(output / "analysis.json"),
        "scripts": [str(output / "run.py")],
        "artifacts": {"data": str(output / "data.json")},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(data))
    return root, path, data


def test_shared_manifest_preserves_metadata_and_resolves_only_paths(manifest):
    root, path, data = manifest
    original = path.read_bytes()
    resolved = resolve_submission(path.read_text(), root)
    assert json.loads(resolved)["results"] == data["results"]
    evidence = Evidence(resolved)
    assert evidence.root == root
    assert evidence.settings["model"] == "MACE-MP-0"
    assert evidence.json("data") == [1, 2, 3]
    assert evidence.result("unit") == "eV"
    assert path.read_bytes() == original


def test_missing_link_preserves_independent_evidence(manifest):
    root, path, data = manifest
    data["artifacts"]["missing"] = "missing.csv"
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(path.read_text(), root))
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
    answer = path.read_text()
    evidence = Evidence(resolve_submission(answer, root))
    assert evidence.json("data") == [1, 2, 3]
    with pytest.raises(EvidenceError, match="Missing artifact|escapes"):
        evidence.json("missing")


def test_missing_absolute_artifact_does_not_bind_to_same_basename(manifest):
    root, path, data = manifest
    data["artifacts"]["missing"] = str(root / "other/data.json")
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(path.read_text(), root))
    assert evidence.json("data") == [1, 2, 3]
    with pytest.raises(EvidenceError, match="Missing artifact|escapes"):
        evidence.json("missing")


@pytest.mark.parametrize(
    "answer", ["runA/manifest.json", "/old/workspace/runA/manifest.json"]
)
def test_missing_manifest_does_not_bind_to_another_run(tmp_path, answer):
    other = tmp_path / "runB"
    other.mkdir()
    (other / "manifest.json").write_text('{"results": {"value": 99}}')
    with pytest.raises(ValueError, match="JSON object directly"):
        resolve_submission(answer, tmp_path)


def test_result_strings_are_not_paths_even_when_they_look_like_paths(manifest):
    root, path, data = manifest
    data["results"]["unit"] = "m2/s"
    data["results"]["explanation"] = "../this is prose"
    path.write_text(json.dumps(data))
    assert Evidence(resolve_submission(path.read_text(), root)).result("unit") == "m2/s"


@pytest.mark.parametrize("link", ["../../outside.json", "/external/missing.json"])
def test_workspace_confinement(manifest, link):
    root, path, data = manifest
    data["artifacts"]["data"] = link
    path.write_text(json.dumps(data))
    if ".." in link:
        with pytest.raises(ValueError):
            resolve_submission(path.read_text(), root)
    else:
        evidence = Evidence(resolve_submission(path.read_text(), root))
        with pytest.raises(EvidenceError):
            evidence.json("data")


def test_object_arrays_cannot_be_deserialized(manifest):
    root, path, data = manifest
    np.save(path.parent / "objects.npy", np.array([{"not": "numeric"}], dtype=object))
    data["artifacts"]["objects"] = str(path.parent / "objects.npy")
    path.write_text(json.dumps(data))
    evidence = Evidence(resolve_submission(path.read_text(), root))
    with pytest.raises(ValueError):
        evidence.array("objects")


@pytest.mark.parametrize("pbc", [None, "true", [1, 1, 1]])
def test_json_frames_require_boundary_condition_evidence(manifest, pbc):
    root, path, _ = manifest
    frame = {"symbols": ["Al"], "positions": [[0, 0, 0]], "cell": [4, 4, 4]}
    if pbc is not None:
        frame["pbc"] = pbc
    (path.parent / "data.json").write_text(json.dumps([frame]))
    evidence = Evidence(resolve_submission(path.read_text(), root))
    with pytest.raises(EvidenceError, match="explicit boolean pbc"):
        evidence.trajectory("data")


def test_rubric_distinguishes_failure_and_unverified():
    rubric = Rubric(1)
    rubric.check("valid", 25, True)
    rubric.check("invalid", 25, lambda: (False, "wrong value"))
    rubric.check("unsupported", 25, None)
    rubric.error("missing", 25, EvidenceError("Missing required scientific quantity"))
    assert rubric.score == 0
    assert rubric.as_dict()["score_bounds"] == [0, 0]
    assert [c["status"] for c in rubric.checks] == [
        "passed",
        "failed",
        "unverified",
        "failed",
    ]
    assert "EvidenceError" in rubric.checks[-1]["detail"]


def test_binary_rubric_is_terminal_on_failure_and_keeps_errors_distinct():
    failed = Rubric(1)
    failed.check("valid", 25, True)
    failed.check("invalid", 25, False)
    failed.check("unsupported", 50, None)
    assert failed.score == 0
    assert failed.as_dict()["score_bounds"] == [0, 0]
    assert failed.as_dict()["status"] == "complete"

    pending = Rubric(1)
    pending.check("valid", 50, True)
    pending.check("unsupported", 50, None)
    with pytest.raises(EvaluationError, match="unsupported"):
        _ = pending.score
    with pytest.raises(EvaluationError, match="unsupported") as caught:
        pending.as_dict()
    assert "score" not in caught.value.report
    assert caught.value.report["unsupported_checks"] == ["unsupported"]


@pytest.mark.parametrize("earned", [0, 25, 99, 100])
def test_rubric_never_awards_partial_credit(earned):
    rubric = Rubric(1)
    rubric.check("verified_evidence", earned, True)
    assert rubric.score == (1 if earned == 100 else 0)


def test_retired_partial_credit_policy_cannot_be_selected():
    with pytest.raises(TypeError, match="binary"):
        Rubric(1, binary=False)


def test_workflow_failure_skips_later_checks_and_modal_verification(manifest):
    _, path, _ = manifest

    class Verifier:
        def _calculate(self, *_args):
            raise AssertionError("Model verification must not run after a log failure")

    report = WorkflowScorer(5, verifier=Verifier()).evaluate(path)
    assert report["score"] == 0
    assert report["checks"][0]["name"] == "calculation_log"
    assert report["checks"][0]["status"] == "failed"
    assert all(check["status"] != "passed" for check in report["checks"])


def test_missing_log_fails_before_task_evaluation(manifest):
    _, path, data = manifest
    data["artifacts"] = {"run_log": "missing.json"}
    path.write_text(json.dumps(data))
    report = WorkflowScorer(5).evaluate(path)
    assert report["score"] == 0
    assert report["checks"][0]["status"] == "failed"


def test_factory_returns_scalar_and_keeps_json_checks_task_local(manifest, monkeypatch):
    root, path, _ = manifest

    def evaluate_submission(value, **_kwargs):
        rubric = Rubric(5)
        rubric.check(
            "saved_value", 100, lambda: Evidence(value).json("data") == [1, 2, 3]
        )
        return {**rubric.as_dict(), "mode": "saved_logs"}

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(
            lambda _self: SimpleNamespace(evaluate_submission=evaluate_submission)
        ),
    )
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    scorer = WorkflowScorer(5)
    submission = resolve_submission(path.read_text(), root)
    good = scorer(submission)
    assert type(good) is float
    assert good == 1.0
    report = scorer.evaluate(submission)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["mode"] == "saved_logs"
    assert report["checks"][-1]["status"] == "passed"
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


def _unsupported_scorer(monkeypatch):
    def evaluate_submission(submission, **_kwargs):
        r = Rubric(5, fail_fast=True)
        try:
            e = Evidence(submission)
        except Exception as exc:
            r.check("readable_manifest", 100, False, str(exc))
            return r.as_dict()
        r.check("reported_results", 10, lambda: bool(np.isfinite(e.results["value"])))

        def alternative():
            assert e.json("data") == [1, 2, 3]
            raise UnsupportedEvidence("Retained alternate calculation is unsupported")

        r.check("reported_diffusion", 90, alternative)
        fingerprint = e.fingerprint() if not r.failed else None
        return {**r.as_dict(), "evidence_sha256": fingerprint}

    monkeypatch.setattr(
        WorkflowScorer,
        "module",
        property(lambda _: SimpleNamespace(evaluate_submission=evaluate_submission)),
    )
    return WorkflowScorer(5)


@pytest.mark.parametrize(
    "entry", ["evaluate", "evaluate_submission", "score_submission", "__call__"]
)
def test_unsupported_calculation_is_an_explicit_error(manifest, monkeypatch, entry):
    _, path, _ = manifest
    scorer = _unsupported_scorer(monkeypatch)
    with pytest.raises(
        EvaluationError, match="Retained alternate calculation"
    ) as caught:
        getattr(scorer, entry)(path)
    report = caught.value.report
    assert report["status"] == "evaluation_error"
    assert report["unsupported_checks"] == ["reported_diffusion"]
    assert "score" not in report
    assert "pending_checks" not in report
    json.dumps(report, allow_nan=False)


def test_submission_authored_review_cannot_create_a_score(manifest, monkeypatch):
    _, path, data = manifest
    scorer = _unsupported_scorer(monkeypatch)
    data["review"] = {"decisions": {"reported_diffusion": {"passed": True}}}
    path.write_text(json.dumps(data))
    with pytest.raises(EvaluationError, match="reported_diffusion"):
        scorer.evaluate(path)


@pytest.mark.parametrize(
    "entry", ["evaluate", "evaluate_submission", "score_submission"]
)
def test_review_argument_is_removed(manifest, monkeypatch, entry):
    _, path, _ = manifest
    scorer = _unsupported_scorer(monkeypatch)
    with pytest.raises(TypeError, match="review"):
        getattr(scorer, entry)(path, review={"decisions": {}})
    assert not hasattr(Rubric(), "apply_review")


def test_zero_point_provenance_does_not_block_completed_score():
    rubric = Rubric(1)
    rubric.check("complete", 100, True)
    rubric.unverified("execution", "Saved data cannot establish actual execution")
    assert rubric.score == 1
    assert rubric.as_dict()["status"] == "complete"


def test_cli_writes_terminal_error_and_rejects_review(manifest, monkeypatch, tmp_path):
    _, path, _ = manifest
    scorer = _unsupported_scorer(monkeypatch)
    monkeypatch.setattr(
        "corral_md.scoring.evaluator.check_level2_workflow", lambda *_a, **_kw: scorer
    )
    output = tmp_path / "evaluation.json"
    argv = [
        "corral_md.scoring",
        "10",
        str(path),
        "--output",
        str(output),
        "--verification",
        "offline",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 2
    report = json.loads(output.read_text())
    assert report["status"] == "evaluation_error"
    assert report["level"] == 2
    assert report["unsupported_checks"] == ["reported_diffusion"]
    assert "score" not in report
    monkeypatch.setattr(sys, "argv", [*argv, "--review", "unused.json"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2


def test_malformed_direct_manifest_and_nonfinite_pending_result_do_not_crash(
    manifest, monkeypatch
):
    root, path, _ = manifest
    scorer = _unsupported_scorer(monkeypatch)
    assert scorer.evaluate({"results": {"value": object()}})["score"] == 0
    data = json.loads(resolve_submission(path.read_text(), root))
    data["results"]["value"] = float("nan")
    report = scorer.evaluate(data)
    assert report["score"] == 0
    assert report["status"] == "complete"
    assert (
        next(c for c in report["checks"] if c["name"] == "reported_results")["status"]
        == "failed"
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("task", "level"),
    [(task, level) for task in (2, 6, 7, 9, 10) for level in (1, 2)] + [(3, 2), (5, 1)],
)
def test_workflow_constructor_cannot_select_a_retired_policy(task, level):
    with pytest.raises(ValueError, match=f"check_level{level}_workflow"):
        WorkflowScorer(task, level=level)


@pytest.mark.parametrize(("task", "level"), [(3, 2), (5, 1)])
def test_task_module_cannot_select_a_retired_policy(task, level):
    from importlib import import_module

    module = import_module(f"corral_md.scoring.task_{task}")
    with pytest.raises(ValueError, match="uses"):
        module.evaluate_submission({}, level=level)
