"""Tests for the intervention hook system."""

import pytest

from corral.agents.hooks.core import CriticalHookError, HookContext
from corral.agents.hooks.intervention import (
    _parse_react_actions,
    create_intervention_hook,
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, class_name: str = "DefaultAgent"):
        self._class_name = class_name

    @property
    def __class__(self):
        return type(self._class_name, (), {})


class MockInterface:
    """Mock interface for testing."""

    def execute_tool(self, task_id: str, tool_name: str, arguments: dict):
        """Mock tool execution."""
        return MockToolResponse(success=True, result=f"Executed {tool_name}")


class MockToolResponse:
    """Mock tool response."""

    def __init__(self, success: bool = True, result: str = "", error: str = ""):
        self.success = success
        self.result = result
        self.error = error


def test_parse_react_actions_basic():
    """Test parsing basic ReAct actions."""
    text = """
    <thought>I need to search</thought>
    <action>search</action>
    <action_input>{"query": "test"}</action_input>
    """

    actions = _parse_react_actions(text)

    assert len(actions) == 1
    assert actions[0].tool_name == "search"
    assert actions[0].arguments == {"query": "test"}


def test_parse_react_actions_multiple():
    """Test parsing multiple actions."""
    text = """
    <action>tool1</action>
    <action_input>{"arg": "value1"}</action_input>
    <action>tool2</action>
    <action_input>{"arg": "value2"}</action_input>
    """

    actions = _parse_react_actions(text)

    assert len(actions) == 2
    assert actions[0].tool_name == "tool1"
    assert actions[1].tool_name == "tool2"


def test_parse_react_actions_with_boolean_conversion():
    """Test that Python booleans are converted to JSON booleans."""
    text = """
    <action>configure</action>
    <action_input>{"enabled": True, "disabled": False}</action_input>
    """

    actions = _parse_react_actions(text)

    assert len(actions) == 1
    assert actions[0].arguments["enabled"] is True
    assert actions[0].arguments["disabled"] is False


def test_parse_react_actions_invalid_json():
    """Test that invalid JSON is handled gracefully."""
    text = """
    <action>test</action>
    <action_input>{invalid json}</action_input>
    """

    actions = _parse_react_actions(text)

    # Should return empty list when JSON parsing fails
    assert len(actions) == 0


def test_parse_react_actions_no_input():
    """Test actions without action_input are skipped."""
    text = """
    <action>incomplete_action</action>
    """

    actions = _parse_react_actions(text)

    # Actions without action_input should be skipped
    assert len(actions) == 0


def test_create_intervention_hook_basic():
    """Test creating and using a basic intervention hook."""
    intervention_map = {"task1": "This is an intervention thought"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task1",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    # Should add a message
    assert len(context.messages) == 1
    assert context.messages[0]["content"] == "This is an intervention thought"
    assert context.metadata["intervention_applied"] is True


def test_create_intervention_hook_no_match():
    """Test that hook does nothing for unmatched task IDs."""
    intervention_map = {"task1": "Intervention"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task2",  # Different task ID
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    # Should not add any messages
    assert len(context.messages) == 0
    assert "intervention_applied" not in context.metadata


def test_react_agent_intervention_wraps_in_thought_tags():
    """Test that ReActAgent intervention wraps content in thought tags."""
    intervention_map = {"task1": "My reasoning here"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ReActAgent"),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    assert len(context.messages) == 1
    assert context.messages[0]["content"] == "<thought>My reasoning here</thought>"


def test_react_agent_intervention_strips_actions_when_not_executing():
    """Test that action tags are stripped when execute_tools=False."""
    intervention_with_action = """
    I need to search for information.
    <action>search</action>
    <action_input>{"query": "test"}</action_input>
    """
    intervention_map = {"task1": intervention_with_action}

    hook = create_intervention_hook(intervention_map, execute_tools=False)

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ReActAgent"),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    assert len(context.messages) == 1
    # Should strip action tags
    assert "<action>" not in context.messages[0]["content"]
    assert "<action_input>" not in context.messages[0]["content"]
    # Should be wrapped in thought tags
    assert context.messages[0]["content"].startswith("<thought>")


def test_react_agent_intervention_executes_tools():
    """Test that ReActAgent intervention can execute tools."""
    intervention_with_action = """
    <thought>I need to search</thought>
    <action>search</action>
    <action_input>{"query": "test"}</action_input>
    """
    intervention_map = {"task1": intervention_with_action}

    hook = create_intervention_hook(intervention_map, execute_tools=True)

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ReActAgent"),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    # Should have intervention message + observation
    assert len(context.messages) == 2
    assert context.messages[1]["content"] == "Observation: Executed search"


def test_react_agent_intervention_tool_failure():
    """Test that tool execution failure raises CriticalHookError."""
    intervention_with_action = """
    <thought>I need to search</thought>
    <action>search</action>
    <action_input>{"query": "test"}</action_input>
    """
    intervention_map = {"task1": intervention_with_action}

    hook = create_intervention_hook(intervention_map, execute_tools=True)

    class FailingInterface:
        def execute_tool(self, task_id: str, tool_name: str, arguments: dict):
            return MockToolResponse(success=False, error="Tool failed")

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ReActAgent"),
        interface=FailingInterface(),
        messages=[],
        iteration=0,
    )

    with pytest.raises(CriticalHookError) as exc_info:
        hook(context)

    assert "Tool 'search' execution failed" in str(exc_info.value)


def test_toolcalling_agent_intervention():
    """Test that ToolCallingAgent intervention adds message without tool execution."""
    intervention_map = {"task1": "Intervention thought"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ToolCallingAgent"),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    assert len(context.messages) == 1
    assert context.messages[0]["content"] == "Intervention thought"


def test_intervention_metadata():
    """Test that intervention metadata is properly set."""
    intervention_map = {"task1": "Test intervention"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task1",
        agent=MockAgent(),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    assert context.metadata["intervention_applied"] is True
    assert context.metadata["intervention_thought"] == "Test intervention"


def test_intervention_preserves_existing_thought_tags():
    """Test that intervention already wrapped in thought tags is not double-wrapped."""
    intervention_map = {"task1": "<thought>Already wrapped</thought>"}

    hook = create_intervention_hook(intervention_map)

    context = HookContext(
        task_id="task1",
        agent=MockAgent("ReActAgent"),
        interface=MockInterface(),
        messages=[],
        iteration=0,
    )

    hook(context)

    assert len(context.messages) == 1
    # Should not double-wrap
    assert context.messages[0]["content"] == "<thought>Already wrapped</thought>"
    assert "<thought><thought>" not in context.messages[0]["content"]
