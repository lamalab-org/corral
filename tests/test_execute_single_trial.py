"""Tests for executing a trial through the State/Action runtime."""

from corral.agents.schema import SURRENDER_SENTINEL
from corral.core.action import Action, submit_answer_action
from corral.report import TaskTrialResult
from corral.run import execute_single_trial
from corral.types import ToolResponse


class FakeInterface:
    def __init__(self):
        self.submit_calls = []
        self.surrender_calls = []
        self.tool_calls = []

    def configure_additional_apps(self, task_id, timeout=None):
        return "configured"

    def get_task_prompt(self, task_id):
        return f"solve {task_id}"

    def get_available_tools_for_task(self, task_id, verbosity="brief"):
        return {
            "tools": [
                {
                    "name": "measure",
                    "description": "Measure",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        }

    def execute_tool(self, task_id, name, arguments):
        self.tool_calls.append((task_id, name, arguments))
        return ToolResponse(result="measured", success=True, error=None)

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


class SubmitAgent:
    model = "test-model"
    max_iterations = 2

    def __init__(self, answer="42"):
        self.answer = answer

    async def step(self, state):
        return submit_answer_action(self.answer)


class ToolThenSubmitAgent:
    model = "test-model"
    max_iterations = 2

    async def step(self, state):
        if state.usage.agent_steps == 0:
            return Action(name="measure", arguments={})
        return submit_answer_action("42")


def _run(agent):
    interface = FakeInterface()
    trial = execute_single_trial(
        task_id="task-1", trial_index=0, interface=interface, agent=agent
    )
    return interface, trial


def test_submit_action_is_submitted_to_environment():
    interface, trial = _run(SubmitAgent())

    assert interface.submit_calls == [("task-1", "42")]
    assert trial.score == 1.0
    assert any(message.get("name") == "submit_answer" for message in trial.messages)


def test_tool_action_is_executed_before_submission():
    interface, trial = _run(ToolThenSubmitAgent())

    assert interface.tool_calls == [("task-1", "measure", {})]
    assert interface.submit_calls == [("task-1", "42")]
    assert trial.score == 1.0


def test_surrender_is_a_submit_action_with_the_sentinel():
    interface, trial = _run(SubmitAgent(SURRENDER_SENTINEL))

    assert interface.surrender_calls == ["task-1"]
    assert interface.submit_calls == []
    assert trial.surrendered is True
