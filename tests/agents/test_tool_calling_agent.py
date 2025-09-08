"""Tests for the ToolCallingAgent class."""

import importlib.resources

import pytest
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.tool_calling import Action, ToolCallingAgent
from corral.agents.utils import LiteLLMMessage
from corral.report import ToolResponse

# Import shared mock classes from conftest.py
from .conftest import MockFunction, MockLLMResponse, MockPrompt, MockToolCall


def get_messages_by_role(messages, role):
    """Helper function to get messages by role from TypedDict messages."""
    result = []
    for msg in messages:
        if hasattr(msg, "get"):
            # It's a dict/TypedDict
            if msg.get("role") == role:
                result.append(msg)
        elif hasattr(msg, "role") and msg.role == role:
            # It's an object with role attribute
            result.append(msg)
    return result


@pytest.fixture()
def tool_calling_agent():
    """Create a ToolCallingAgent instance for testing."""
    return ToolCallingAgent(
        model="openai/gpt-4o",
        max_iterations=5,
        temperature=0.7,
        system_prompt=MockPrompt("You are a helpful AI assistant."),
        user_prompt=MockPrompt(
            "Task: {{task_guide}}\n\nExamples: {{examples}}\n\nSolve this step by step."
        ),
    )


# Tests for Action dataclass


def test_action_creation():
    """Test Action dataclass creation."""
    action = Action(tool_name="test_tool", arguments={"query": "test query"})

    assert action.tool_name == "test_tool"
    assert action.arguments == {"query": "test query"}


def test_action_with_complex_arguments():
    """Test Action with complex arguments."""
    complex_args = {
        "string_arg": "test",
        "number_arg": 42,
        "list_arg": ["a", "b", "c"],
        "dict_arg": {"nested": "value"},
    }

    action = Action(tool_name="complex_tool", arguments=complex_args)

    assert action.tool_name == "complex_tool"
    assert action.arguments == complex_args


# Tests for ToolCallingAgent class


def test_tool_calling_agent_init_with_defaults():
    """Test ToolCallingAgent initialization with defaults."""
    agent = ToolCallingAgent()

    assert agent.model == "openai/gpt-4o"
    assert agent.max_iterations == 10
    assert agent.temperature == 0.7
    assert agent.api_endpoint is None


def test_tool_calling_agent_init_with_custom_params():
    """Test ToolCallingAgent initialization with custom parameters."""
    agent = ToolCallingAgent(
        model="anthropic/claude-3-5-sonnet-20241022",
        max_iterations=15,
        temperature=0.3,
        api_endpoint="https://custom.api.endpoint",
    )

    assert agent.model == "anthropic/claude-3-5-sonnet-20241022"
    assert agent.max_iterations == 15
    assert agent.temperature == 0.3
    assert agent.api_endpoint == "https://custom.api.endpoint"


def test_tool_calling_agent_init_with_prompt_store(mock_prompt_store):
    """Test ToolCallingAgent initialization with PromptStore."""
    agent = ToolCallingAgent(
        prompt_store=mock_prompt_store,
        system_prompt_id="test-system-id",
        user_prompt_id="test-user-id",
    )

    # The prompt_store is stored as 'store' in BaseAgent
    assert hasattr(agent, "store")
    assert agent.store == mock_prompt_store


def test_tool_calling_agent_run_setup(tool_calling_agent, mock_interface, monkeypatch):
    """Test that run method sets up correctly."""
    # Mock the functions using monkeypatch
    mock_create_prompt = MockFunction(
        return_value=[
            LiteLLMMessage(role="system", content="System prompt"),
            LiteLLMMessage(role="user", content="User prompt"),
        ]
    )
    mock_convert_tools = MockFunction(
        return_value=[{"type": "function", "function": {"name": "test_tool"}}]
    )

    monkeypatch.setattr("corral.agents.tool_calling.create_prompt", mock_create_prompt)
    monkeypatch.setattr(
        "corral.agents.tool_calling.convert_to_openai_tool_format", mock_convert_tools
    )

    # Mock get_llm_response to return final answer immediately
    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: Test result")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    # Verify setup calls
    assert mock_interface.call_counts.get("get_available_tools_for_task", 0) == 1
    assert mock_interface.call_counts.get("get_task_prompt", 0) == 1
    assert mock_convert_tools.call_count == 1
    assert mock_create_prompt.call_count == 1

    assert result == "Test result"


def test_tool_calling_agent_run_with_final_answer(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when LLM returns final answer immediately."""
    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: Direct answer")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Direct answer"
    assert len(tool_calling_agent.messages) == 3  # system + user + assistant
    assert tool_calling_agent.messages[-1]["role"] == "assistant"
    assert tool_calling_agent.messages[-1]["content"] == "Final Answer: Direct answer"


def test_tool_calling_agent_run_with_tool_calls(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method with tool calls."""
    # First response: tool call
    tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
    first_response = MockLLMResponse(content=None, tool_calls=[tool_call])

    # Second response: final answer
    second_response = MockLLMResponse(content="Final Answer: Tool result processed")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Tool result processed"

    # Verify tool execution
    assert len(mock_interface.tool_calls) == 1
    assert mock_interface.tool_calls[0]["task_id"] == "test_task"
    assert mock_interface.tool_calls[0]["tool_name"] == "test_tool"
    assert mock_interface.tool_calls[0]["arguments"] == {"query": "test"}

    # Verify message flow
    assert (
        len(tool_calling_agent.messages) >= 4
    )  # system + user + assistant + tool + assistant

    # Check tool message was added
    tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[0]["content"] == "Tool execution result"
    assert tool_messages[0]["name"] == "test_tool"


def test_tool_calling_agent_run_with_multiple_tool_calls(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method with multiple tool calls in one response."""
    tool_call_1 = MockToolCall("call_1", "test_tool", {"query": "first"})
    tool_call_2 = MockToolCall("call_2", "test_tool", {"query": "second"})

    first_response = MockLLMResponse(
        content=None, tool_calls=[tool_call_1, tool_call_2]
    )
    second_response = MockLLMResponse(content="Final Answer: Multiple tools processed")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Multiple tools processed"

    # Verify both tools were executed
    assert len(mock_interface.tool_calls) == 2
    assert mock_interface.tool_calls[0] == {
        "task_id": "test_task",
        "tool_name": "test_tool",
        "arguments": {"query": "first"},
    }
    assert mock_interface.tool_calls[1] == {
        "task_id": "test_task",
        "tool_name": "test_tool",
        "arguments": {"query": "second"},
    }

    # Verify two tool messages were added
    tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
    assert len(tool_messages) == 2


def test_tool_calling_agent_run_with_tool_error(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when tool execution fails."""
    # Mock tool execution failure
    mock_tool_response = ToolResponse(
        success=False, result=None, error="Tool execution failed"
    )
    mock_interface.tool_responses = [mock_tool_response]

    tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
    first_response = MockLLMResponse(content=None, tool_calls=[tool_call])
    second_response = MockLLMResponse(content="Final Answer: Handled error")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Handled error"

    # Verify error was handled - the code converts str(None) to "None"
    tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "None"  # str(None) = "None"


def test_tool_calling_agent_run_with_tool_exception(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when tool execution raises exception."""

    # Mock tool execution to raise exception
    def execute_tool_side_effect(*args):
        raise Exception("Unexpected error")

    mock_interface.execute_tool = execute_tool_side_effect

    tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
    first_response = MockLLMResponse(content=None, tool_calls=[tool_call])
    second_response = MockLLMResponse(content="Final Answer: Exception handled")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Exception handled"

    # Verify exception was handled
    tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "Unexpected error"


def test_tool_calling_agent_run_with_llm_response_exception(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when LLM response raises exception."""
    mock_llm_response = MockFunction(
        side_effect=[
            Exception("LLM error"),
            MockLLMResponse(content="Final Answer: Recovered from error"),
        ]
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Recovered from error"

    # Verify error message was added
    error_messages = get_messages_by_role(tool_calling_agent.messages, "system")
    assert len(error_messages) == 2  # initial system + error system
    assert "Error during agent iteration: LLM error" in error_messages[1]["content"]


def test_tool_calling_agent_run_with_content_but_no_final_answer(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when LLM returns content but no final answer."""
    first_response = MockLLMResponse(content="I need to think about this...")
    second_response = MockLLMResponse(content="Final Answer: Now I have the answer")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Now I have the answer"

    # Verify intermediate response was added
    assistant_messages = get_messages_by_role(tool_calling_agent.messages, "assistant")
    assert len(assistant_messages) == 2
    assert assistant_messages[0]["content"] == "I need to think about this..."


def test_tool_calling_agent_run_max_iterations_reached(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method when max iterations is reached."""
    # Set low max iterations for testing
    tool_calling_agent.max_iterations = 2

    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Still thinking...")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "Error solving the task. Maximum iterations reached."

    # Verify max iterations was reached
    assert mock_llm_response.call_count == 2

    # Verify error message was added
    error_messages = [
        msg
        for msg in tool_calling_agent.messages
        if msg.get("role") == "assistant" and msg.get("name") == "tool-calling-error"
    ]
    assert len(error_messages) == 1
    assert "Maximum iterations reached" in error_messages[0]["content"]


def test_tool_calling_agent_run_with_custom_task_prompt(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test run method with custom task prompt."""
    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: Custom task result")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(
        mock_interface, "test_task", task_prompt="Custom task prompt"
    )

    assert result == "Custom task result"

    # Verify custom task prompt was used instead of interface prompt
    assert mock_interface.call_counts.get("get_task_prompt", 0) == 0


def test_run_with_history(tool_calling_agent, mock_interface, monkeypatch):
    """Test run method with conversation history."""
    history = [
        LiteLLMMessage(role="user", content="Previous question"),
        LiteLLMMessage(role="assistant", content="Previous answer"),
    ]

    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: With history")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task", history=history)

    assert result == "With history"

    # Verify history was included in messages
    # Should have system + user + history + user (task) + assistant
    assert len(tool_calling_agent.messages) >= 4


def test_run_with_examples(tool_calling_agent, mock_interface, monkeypatch):
    """Test run method with examples."""
    examples = ["Example 1", "Example 2"]

    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: With examples")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task", examples=examples)

    assert result == "With examples"


def test_run_case_insensitive_final_answer(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test that final answer matching is case insensitive."""
    test_cases = [
        "final answer: lowercase",
        "Final Answer: normal case",
        "FINAL ANSWER: uppercase",
        "Final answer: mixed case",
    ]

    for content in test_cases:
        # Reset the agent for each test case
        tool_calling_agent.messages = []

        mock_llm_response = MockFunction(return_value=MockLLMResponse(content=content))
        monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

        result = tool_calling_agent.run(mock_interface, "test_task")

        expected = content.split(":", 1)[1].strip()
        assert result == expected


def test_run_final_answer_extraction(tool_calling_agent, mock_interface, monkeypatch):
    """Test final answer extraction from complex content."""
    content = """
    I need to solve this step by step.

    First, let me analyze the problem...

    After careful consideration, I believe the answer is:

    Final Answer: The solution is 42 with detailed explanation.
    """

    mock_llm_response = MockFunction(return_value=MockLLMResponse(content=content))
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "The solution is 42 with detailed explanation."


def test_tools_conversion_called_correctly(
    tool_calling_agent, mock_interface, monkeypatch
):
    """Test that tools are converted to OpenAI format correctly."""
    expected_tools = [{"type": "function", "function": {"name": "test_tool"}}]
    mock_convert_tools = MockFunction(return_value=expected_tools)
    monkeypatch.setattr(
        "corral.agents.tool_calling.convert_to_openai_tool_format", mock_convert_tools
    )

    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: Test")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    tool_calling_agent.run(mock_interface, "test_task")

    # Verify convert_to_openai_tool_format was called with correct tools
    assert mock_convert_tools.call_count == 1
    args, kwargs = mock_convert_tools.call_args_list[0]
    assert len(args) == 1
    # The mock_interface now returns a dict with tools key
    expected_arg = mock_interface.available_tools
    assert args[0] == expected_arg


def test_available_tools_storage(tool_calling_agent, mock_interface, monkeypatch):
    """Test that available tools are processed correctly."""
    mock_llm_response = MockFunction(
        return_value=MockLLMResponse(content="Final Answer: Test")
    )
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    # Verify the run completed successfully
    assert result == "Test"


def test_json_parsing_in_tool_calls(tool_calling_agent, mock_interface, monkeypatch):
    """Test that JSON parsing works correctly for tool arguments."""
    complex_args = {
        "string_param": "test string",
        "number_param": 42,
        "boolean_param": True,
        "array_param": [1, 2, 3],
        "object_param": {"nested": "value"},
    }

    tool_call = MockToolCall("call_1", "test_tool", complex_args)
    first_response = MockLLMResponse(content=None, tool_calls=[tool_call])
    second_response = MockLLMResponse(content="Final Answer: JSON parsed correctly")

    mock_llm_response = MockFunction(side_effect=[first_response, second_response])
    monkeypatch.setattr(tool_calling_agent, "get_llm_response", mock_llm_response)

    result = tool_calling_agent.run(mock_interface, "test_task")

    assert result == "JSON parsed correctly"

    # Verify JSON was parsed correctly
    assert len(mock_interface.tool_calls) == 1
    assert mock_interface.tool_calls[0]["task_id"] == "test_task"
    assert mock_interface.tool_calls[0]["tool_name"] == "test_tool"
    assert mock_interface.tool_calls[0]["arguments"] == complex_args


def test_docstring_requirements():
    """Test that the class docstring mentions required prompt fields."""
    docstring = ToolCallingAgent.__doc__

    # Check that docstring exists and contains required fields
    assert docstring is not None
    assert "{{task_guide}}" in docstring
    assert "{{examples}}" in docstring
    assert "Required Prompt Fields" in docstring

    # Check that it mentions function calling
    assert "function calling" in docstring or "tool calling" in docstring


def test_tool_calling_agent_inheritance_from_base_agent():
    """Test that ToolCallingAgent inherits from BaseAgent."""
    assert issubclass(ToolCallingAgent, BaseAgent)

    # Test that it has the required abstract method
    agent = ToolCallingAgent()
    assert hasattr(agent, "run")
    assert callable(agent.run)


# Tests for ToolCallingAgent default prompts


def test_tool_calling_agent_default_system_prompt_content():
    """Test that the default system prompt contains expected content."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the system prompt using the default ID
        system_prompt = store.get("400fcecf-f5f2-464b-aff5-8a4377c9685c")

        # Check that it contains expected content
        content = system_prompt.content
        assert "helpful" in content.lower()
        assert "assistant" in content.lower()


def test_tool_calling_agent_default_user_prompt_content():
    """Test that the default user prompt contains expected placeholders and instructions."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the user prompt using the default ID
        user_prompt = store.get("fe04453b-5469-4611-bba6-6d81487df787")

        # Check that it contains expected content
        content = user_prompt.content
        assert "{{task_guide}}" in content
        assert "Final Answer:" in content
        assert "tools available" in content.lower()
        assert "task is completed" in content.lower()


def test_tool_calling_agent_default_extractor_prompt_content():
    """Test that the default extractor prompt contains expected placeholders."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the extractor prompt using the default ID
        extractor_prompt = store.get("9d37e4a0-26c5-438a-ba1b-a273388fcded")

        # Check that it contains expected content
        content = extractor_prompt.content
        assert "{{message}}" in content
        assert "{{answer}}" in content
        assert "extract" in content.lower()
        assert "answer only" in content.lower()


def test_tool_calling_agent_user_prompt_template_variables():
    """Test that the user prompt template supports required variables."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the user prompt using the default ID
        user_prompt = store.get("fe04453b-5469-4611-bba6-6d81487df787")

        # Test that it can be filled with required variables
        filled_content = user_prompt.fill(
            {"task_guide": "Test task description", "examples": "Test examples"}
        )

        # Check that variables were replaced
        assert "Test task description" in filled_content
        assert "{{task_guide}}" not in filled_content

        # Check that the structure is maintained
        assert "Final Answer:" in filled_content
        assert "tools available" in filled_content.lower()


def test_tool_calling_agent_extractor_prompt_template_variables():
    """Test that the extractor prompt template supports required variables."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the extractor prompt using the default ID
        extractor_prompt = store.get("9d37e4a0-26c5-438a-ba1b-a273388fcded")

        # Test that it can be filled with required variables
        filled_content = extractor_prompt.fill(
            {"message": "Test message content", "answer": "Test answer content"}
        )

        # Check that variables were replaced
        assert "Test message content" in filled_content
        assert "Test answer content" in filled_content
        assert "{{message}}" not in filled_content
        assert "{{answer}}" not in filled_content


def test_tool_calling_agent_default_prompt_ids_match_class_defaults():
    """Test that the default prompt IDs in the class match the available prompts."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Check that all default prompt IDs exist in the store
        system_prompt_id = "400fcecf-f5f2-464b-aff5-8a4377c9685c"
        user_prompt_id = "fe04453b-5469-4611-bba6-6d81487df787"
        extractor_prompt_id = "9d37e4a0-26c5-438a-ba1b-a273388fcded"

        # These should not raise exceptions
        system_prompt = store.get(system_prompt_id)
        user_prompt = store.get(user_prompt_id)
        extractor_prompt = store.get(extractor_prompt_id)

        # Check that they have the expected interface
        assert hasattr(system_prompt, "content")
        assert hasattr(user_prompt, "fill")
        assert hasattr(extractor_prompt, "fill")


def test_tool_calling_agent_user_prompt_does_not_contain_tool_descriptions():
    """Test that the user prompt does not contain tool descriptions (as per docstring)."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the user prompt using the default ID
        user_prompt = store.get("fe04453b-5469-4611-bba6-6d81487df787")

        # Check that it doesn't contain tool descriptions
        content = user_prompt.content.lower()

        # Should not contain explicit tool descriptions or tool lists
        assert "available tools:" not in content
        assert "tool descriptions:" not in content
        assert "{{tools}}" not in content

        # But should mention tools generically
        assert "tools available" in content or "use some of the tools" in content


def test_tool_calling_agent_user_prompt_missing_examples_variable():
    """Test that the user prompt handles missing examples variable gracefully."""
    # Get the actual prompt store path
    with importlib.resources.path("corral.agents", "prompts") as prompts_path:
        store = PromptStore(prompts_path)

        # Get the user prompt using the default ID
        user_prompt = store.get("fe04453b-5469-4611-bba6-6d81487df787")

        # Fill with only task_guide, leaving examples empty
        filled_content = user_prompt.fill({"task_guide": "Test task description"})

        # Should still contain the task description
        assert "Test task description" in filled_content
        assert "Final Answer:" in filled_content

        # The {{examples}} placeholder should remain (or be empty depending on implementation)
        # This tests that the prompt doesn't break when examples is not provided
