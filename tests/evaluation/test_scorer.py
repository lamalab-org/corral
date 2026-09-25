from dataclasses import replace
from pathlib import Path

import pytest

from corral.core.state import EnvironmentState, ExecutionState, RuntimeState
from corral.core.task import TaskDefinition
from corral.evaluation import SubmissionScore, TaskScorer


def _submitted_state(answer: str) -> ExecutionState:
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="execution",
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission=answer,
    )


def _task(scoring_fn, *, resolve_answer: bool = False) -> TaskDefinition:
    return TaskDefinition(
        name="task",
        description="task",
        tools=[],
        scoring_fn=scoring_fn,
        submission_format={},
        resolve_answer=resolve_answer,
    )


def test_scorer_returns_sibling_result_without_changing_state():
    state = _submitted_state("42")
    commit_hash = state.through_commit_hash

    result = TaskScorer(_task(lambda answer: answer == "42")).evaluate(state)

    assert result.score == 1.0
    assert result.metrics == {"score": 1.0}
    assert result.commit_hash == commit_hash
    assert state.through_commit_hash == commit_hash
    assert "score" not in state.model_dump()
    assert "submitted_answer" not in state.model_dump()


def test_detailed_submission_score_is_retained_without_double_evaluation():
    class DetailedScorer:
        calls = 0

        def __call__(self, _answer):
            raise AssertionError(
                "Scalar fallback must not run after detailed evaluation"
            )

        def evaluate_submission(self, answer):
            self.calls += 1
            assert answer == "42"
            return SubmissionScore(
                score=0,
                feedback="missing required evidence",
                metadata={"checks": [{"name": "saved_state", "status": "failed"}]},
            )

    scorer = DetailedScorer()
    state = _submitted_state("42")
    before = state.model_dump_json()
    result = TaskScorer(_task(scorer)).evaluate(state)
    assert result.feedback == "missing required evidence"
    assert result.metadata["checks"][0]["status"] == "failed"
    assert scorer.calls == 1
    assert state.model_dump_json() == before


def test_file_submission_is_resolved_only_during_evaluation(tmp_path):
    answer_path = tmp_path / "answer.txt"
    answer_path.write_text("7", encoding="utf-8")
    seen: list[str] = []

    def score(path: str) -> float:
        seen.append(path)
        return float(Path(path).read_text(encoding="utf-8"))

    state = _submitted_state("/workspace/answer.txt")
    result = TaskScorer(_task(score, resolve_answer=True), workspace=tmp_path).evaluate(
        state
    )

    assert result.score == 7.0
    assert Path(seen[0]) == answer_path
    assert state.submission == "/workspace/answer.txt"


@pytest.mark.parametrize("submission_kind", ["absolute", "traversal", "symlink"])
def test_file_submission_cannot_read_a_sibling_task_workspace(
    tmp_path, submission_kind
):
    workspace = tmp_path / "task-a"
    sibling = tmp_path / "task-b"
    workspace.mkdir()
    sibling.mkdir()
    secret = sibling / "answer.txt"
    secret.write_text("99", encoding="utf-8")

    if submission_kind == "absolute":
        submission = str(secret)
    elif submission_kind == "traversal":
        submission = "../task-b/answer.txt"
    else:
        link = workspace / "answer-link.txt"
        link.symlink_to(secret)
        submission = link.name

    called = False

    def score(_path: str) -> float:
        nonlocal called
        called = True
        return 1.0

    with pytest.raises(ValueError, match="workspace"):
        TaskScorer(_task(score, resolve_answer=True), workspace=workspace).evaluate(
            _submitted_state(submission)
        )
    assert called is False


def test_scorer_rejects_state_without_runtime_output():
    with pytest.raises(ValueError, match="completed"):
        TaskScorer(_task(lambda answer: 1.0)).evaluate(
            ExecutionState(
                through_commit_hash="a" * 64,
                execution_id="execution",
                branch_id="main",
            )
        )


def test_interactive_scorer_reads_committed_state_without_resolving_final_text(
    tmp_path,
):
    def answer_scorer(_answer):
        raise AssertionError(
            "interactive scoring must not reinterpret the final answer"
        )

    def trajectory_scorer(state):
        return float(state.environment.values["successful_submission"])

    task = replace(
        _task(answer_scorer, resolve_answer=True), state_scoring_fn=trajectory_scorer
    )
    state = ExecutionState.model_validate(
        {
            **_submitted_state("../not-a-file").model_dump(),
            "environment": EnvironmentState(values={"successful_submission": True}),
        }
    )
    before = state.model_dump_json()
    result = TaskScorer(task, workspace=tmp_path).evaluate(state)
    assert result.score == 1.0
    assert "trajectory_scorer" in result.scorer_version
    assert state.model_dump_json() == before
