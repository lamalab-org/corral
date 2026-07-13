"""Tests for `execute_single_trial` routing of harness infrastructure failures.

A black-box harness agent surfaces a timeout / SDK crash / MCP transport failure
as an :class:`AgentRunResult` with a non-submit-worthy ``status`` and an error
string for ``answer``. `execute_single_trial` must record that as a trial error
(via ``get_task_status`` score) instead of submitting the error string to the
task scorer.
"""

from corral.agents.schema import SURRENDER_SENTINEL, AgentRunResult
from corral.report import TaskTrialResult
from corral.run import execute_single_trial


class FakeInterface:
    def __init__(self):
        self.submit_calls = []
        self.surrender_calls = []

    def configure_additional_apps(self, task_id, timeout=None):
        return "configured"

    def get_task_status(self, task_id):
        return {"score": 0.0}

    def submit_answer(self, task_id, answer):
        self.submit_calls.append((task_id, answer))
        return TaskTrialResult(
            task_id=task_id,
            trial_id="attempt_1",
            score=1.0,
            state={},
            tool_statistics={},
        )

    def surrender_task(self, task_id):
        self.surrender_calls.append(task_id)
        return TaskTrialResult(
            task_id=task_id,
            trial_id="attempt_1",
            score=0.0,
            state={},
            tool_statistics={},
            surrendered=True,
        )


class FakeAgent:
    def __init__(self, result: AgentRunResult):
        self._result = result

    def run_agent(self, interface, task_id, **kwargs):
        return self._result

    def get_total_token_usage(self):
        return {}


def _run(result: AgentRunResult):
    interface = FakeInterface()
    agent = FakeAgent(result)
    trial = execute_single_trial(
        task_id="task-1", trial_index=0, interface=interface, agent=agent
    )
    return interface, trial


def test_success_status_is_submitted():
    interface, trial = _run(AgentRunResult(answer="42", status="success"))
    assert interface.submit_calls == [("task-1", "42")]
    assert trial.score == 1.0


def test_timeout_status_is_not_submitted():
    interface, trial = _run(
        AgentRunResult(
            answer="Error solving the task: timed out",
            status="timeout",
            error_message="harness timed out",
        )
    )
    # The error string is never submitted as if it were a model answer.
    assert interface.submit_calls == []
    assert "Agent timeout" in (trial.error_message or "")


def test_sdk_failure_status_is_not_submitted():
    interface, trial = _run(
        AgentRunResult(
            answer="Error solving the task: crash",
            status="sdk_failure",
            error_message="boom",
        )
    )
    assert interface.submit_calls == []
    assert "Agent sdk_failure" in (trial.error_message or "")


def test_surrender_is_routed_to_surrender_task():
    interface, trial = _run(
        AgentRunResult(answer=SURRENDER_SENTINEL, status="surrender")
    )
    assert interface.surrender_calls == ["task-1"]
    assert interface.submit_calls == []
    assert trial.surrendered is True
