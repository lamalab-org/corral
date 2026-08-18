from pathlib import Path

import pytest

from corral.core.action import submit_answer_action
from corral.core.state import RuntimeState, State
from corral.core.task import TaskDefinition
from corral.evaluation import TaskScorer


def _submitted_state(answer: str) -> State:
    action = submit_answer_action(answer, action_id="submission-1")
    return State(
        messages=(
            action.to_message(),
            {
                "role": "tool",
                "tool_call_id": action.id,
                "name": action.name,
                "content": "answer accepted",
                "metadata": {"status": "success", "success": True},
            },
        ),
        runtime=RuntimeState(status="submitted"),
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
    state_hash = state.state_hash

    result = TaskScorer(_task(lambda answer: answer == "42")).evaluate(state)

    assert result.score == 1.0
    assert result.metrics == {"score": 1.0}
    assert result.state_hash == state_hash
    assert state.state_hash == state_hash
    assert "score" not in state.model_dump()
    assert "submitted_answer" not in state.model_dump()


def test_file_submission_is_resolved_only_during_evaluation(tmp_path):
    answer_path = tmp_path / "answer.txt"
    answer_path.write_text("7", encoding="utf-8")
    seen: list[str] = []

    def score(path: str) -> float:
        seen.append(path)
        return float(Path(path).read_text(encoding="utf-8"))

    state = _submitted_state("answer.txt")
    result = TaskScorer(_task(score, resolve_answer=True), workspace=tmp_path).evaluate(
        state
    )

    assert result.score == 7.0
    assert Path(seen[0]) == answer_path
    assert state.submission == "answer.txt"


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
        TaskScorer(_task(lambda answer: 1.0)).evaluate(State())
