"""The shipped definitions connect task prompts, retained evidence, and scoring."""

import json

import pytest
from corral_md import env
from corral_md.scoring import EvaluationError

from corral.core.state import ExecutionState, RuntimeState
from corral.evaluation import TaskScorer


def _state(submission):
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="offline-check",
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission=submission,
    )


@pytest.mark.parametrize("level", [1, 2])
def test_task1_prompt_describes_records_without_examples(level):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_1.json"
    )[f"level_{level}_task_1"]
    environment = env.MolecularDynamicsEnvironment("task", task)
    prompt = env._md_task_prompt(environment, _state("{}"))
    assert "Blank manifest" not in prompt
    assert "submission_examples/" not in prompt
    assert "lammps_log" in prompt
    assert (
        '"trajectory"' in task.submission_format
        if level == 2
        else '"trajectory"' not in task.submission_format
    )
    assert '"diffusion"' in prompt if level == 2 else '"diffusion"' not in prompt


def test_manifest_path_is_rejected_as_a_final_answer(tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_1.json"
    )["level_2_task_1"]
    with pytest.raises(ValueError, match="JSON object directly"):
        TaskScorer(task, workspace=tmp_path).evaluate(_state("missing.json"))


def test_shared_scorer_propagates_pending_instead_of_recording_zero(
    tmp_path, monkeypatch
):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_10.json"
    )["level_2_task_10"]
    from corral_md.scoring.common import Rubric

    rubric = Rubric(10)
    rubric.check("independent_model_calculation", 100, None, "Backend unavailable")
    monkeypatch.setattr(
        task.scoring_fn,
        "evaluate",
        lambda *_a, **_kw: rubric.as_dict(),
    )
    state = _state("{}")
    before = state.model_dump()
    with pytest.raises(EvaluationError) as pending:
        TaskScorer(task, workspace=tmp_path).evaluate(state)
    assert "score" not in pending.value.report
    assert pending.value.report["status"] == "evaluation_error"
    assert state.model_dump() == before


def test_partial_answer_survives_workspace_restoration(tmp_path):
    restored = tmp_path / "restored"
    outputs = restored / "results"
    outputs.mkdir(parents=True)
    (outputs / "settings.json").write_text(
        json.dumps({"model": "MACE-MP-0", "units": "eV"})
    )
    (outputs / "report.json").write_text(
        json.dumps({"limitations": "Calculation incomplete"})
    )
    (outputs / "run.py").write_text("raise RuntimeError('must never run')")
    (outputs / "data.json").write_text("[1, 2, 3]")
    manifest = {
        "results": {"status": "incomplete", "units": "eV"},
        "settings": "/workspace/results/settings.json",
        "report": "/workspace/results/report.json",
        "scripts": ["/workspace/results/run.py"],
        "artifacts": {"data": "/workspace/results/data.json"},
    }
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_1.json"
    )["level_2_task_1"]
    result = TaskScorer(task, workspace=restored).evaluate(_state(json.dumps(manifest)))
    assert result.score == 0
    assert result.metrics == {"score": result.score}
    assert result.metadata["workflow_evaluation"]["score"] == 0
    assert "Missing requested submission field: artifacts.lammps_log" in result.feedback
    assert not (outputs / "manifest.json").exists()
