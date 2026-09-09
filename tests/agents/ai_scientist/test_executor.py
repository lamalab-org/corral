from dataclasses import dataclass

from corral.agents.ai_scientist.search.nodes import (
    ExecutedAction,
    Observation,
    PlannedAction,
)
from corral.agents.ai_scientist.tools import CorralExecutor
from corral.core.action import Action


@dataclass
class Response:
    success: bool
    result: str | None = None
    error: str | None = None


class FakeActionExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, action: Action):
        self.calls.append(action)
        return Response(success=True, result=f"measured:{action.arguments['value']}")


TOOLS = {
    "tools": [
        {
            "name": "measure",
            "description": "Measure a value",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        }
    ]
}


def action(name="measure", arguments=None):
    return PlannedAction(
        purpose="test",
        tool_name=name,
        arguments=arguments or {"value": 2},
        expected_information="a measurement",
    )


def test_executor_validates_before_calling_and_executes_sequentially():
    execute_action = FakeActionExecutor()
    tool_calls = []
    executor = CorralExecutor(
        execute_action=execute_action,
        tools=TOOLS,
        max_tool_calls=3,
        stop_on_error=False,
        on_tool_call=tool_calls.append,
    )

    observations = executor.execute_plan(
        [
            action(arguments={"value": "wrong"}),
            action(arguments={"value": 4}),
        ]
    )

    assert [item.success for item in observations] == [False, True]
    assert "Invalid tool arguments" in observations[0].error
    assert len(execute_action.calls) == 1
    assert execute_action.calls[0].name == "measure"
    assert execute_action.calls[0].arguments == {"value": 4}
    assert executor.call_count == 1
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "measure"
    assert tool_calls[0]["status"] == "success"
    assert tool_calls[0]["result"] == "measured:4"
    assert tool_calls[0]["duration"] >= 0


def test_executor_preserves_canonical_action_failures():
    class FailedActionExecutor:
        def __call__(self, _action: Action):
            return Response(
                success=False,
                error="instrument unavailable",
            )

    tool_calls = []
    executor = CorralExecutor(
        execute_action=FailedActionExecutor(),
        tools=TOOLS,
        max_tool_calls=1,
        on_tool_call=tool_calls.append,
    )

    observation = executor.execute_plan([action()])[0]

    assert observation.success is False
    assert observation.error == "instrument unavailable"
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "measure"
    assert tool_calls[0]["arguments"] == {"value": 2}
    assert tool_calls[0]["result"] is None
    assert tool_calls[0]["status"] == "execution_error"
    assert tool_calls[0]["error_message"] == "instrument unavailable"
    assert tool_calls[0]["duration"] >= 0
    assert tool_calls[0]["timestamp"]


def test_executor_rejects_unknown_tools_without_spending_budget():
    execute_action = FakeActionExecutor()
    executor = CorralExecutor(
        execute_action=execute_action,
        tools=TOOLS,
        max_tool_calls=1,
    )

    observation = executor.execute_plan([action(name="invented")])[0]

    assert not observation.success
    assert "Unknown Corral tool" in observation.error
    assert execute_action.calls == []
    assert executor.call_count == 0


def test_executor_reports_local_budget_exhaustion_as_an_observation():
    execute_action = FakeActionExecutor()
    executor = CorralExecutor(
        execute_action=execute_action,
        tools=TOOLS,
        max_tool_calls=1,
        stop_on_error=False,
    )

    observations = executor.execute_plan([action(), action(arguments={"value": 3})])

    assert [item.success for item in observations] == [True, False]
    assert "budget exhausted" in observations[1].error
    assert len(execute_action.calls) == 1


def test_replication_plan_overrides_only_schema_declared_random_state():
    executor = CorralExecutor(
        execute_action=FakeActionExecutor(),
        tools={
            "tools": [
                {
                    "name": "measure",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"},
                            "random_state": {"type": "integer"},
                        },
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                }
            ]
        },
        max_tool_calls=1,
    )
    original_action = action(arguments={"value": 2, "random_state": 99})
    original = ExecutedAction(
        action=original_action,
        observation=Observation(
            action_index=0,
            purpose="test",
            tool_name="measure",
            arguments=original_action.arguments,
            success=True,
            result="measured:2",
        ),
    )

    plan, overrides = executor.prepare_replication_plan([original], seed=7)

    assert original_action.arguments == {"value": 2, "random_state": 99}
    assert plan[0].arguments == {"value": 2, "random_state": 7}
    assert plan[0].purpose == original_action.purpose
    assert plan[0].expected_information == original_action.expected_information
    assert overrides == ["action_0.measure.random_state"]
