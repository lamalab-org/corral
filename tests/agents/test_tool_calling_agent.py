"""Tests for the ToolCallingAgent class."""

import json
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from corral.agents.tool_calling import Action, ToolCallingAgent
from corral.agents.utils import LiteLLMMessage
from corral.evaluate import BenchmarkInterface
from corral.report import ToolResponse


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


class MockPrompt:
    """Mock prompt class that implements the required fill method."""

    def __init__(self, content: str):
        self.content = content

    def fill(self, replacements: dict[str, Any]) -> str:
        """Fill the prompt with replacements."""
        result = self.content
        for key, value in replacements.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


class MockLLMResponse:
    """Mock LLM response for testing."""

    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockToolCall:
    """Mock tool call for testing."""

    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.function = Mock()
        self.function.name = name
        self.function.arguments = json.dumps(arguments)


@pytest.fixture()
def mock_prompt_store():
    """Mock PromptStore for testing."""
    store = Mock()
    store.get.return_value = MockPrompt("System prompt")
    return store


@pytest.fixture()
def mock_interface():
    """Mock BenchmarkInterface for testing."""
    interface = Mock(spec=BenchmarkInterface)
    interface.get_available_tools_for_task.return_value = {
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Test query"}
                    },
                    "required": ["query"],
                },
            }
        ]
    }
    interface.get_task_prompt.return_value = "Test task prompt"

    # Mock tool execution
    mock_tool_response = Mock(spec=ToolResponse)
    mock_tool_response.result = "Tool execution result"
    mock_tool_response.error = None
    interface.execute_tool.return_value = mock_tool_response

    return interface


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


class TestAction:
    """Test the Action dataclass."""

    def test_action_creation(self):
        """Test Action dataclass creation."""
        action = Action(tool_name="test_tool", arguments={"query": "test query"})

        assert action.tool_name == "test_tool"
        assert action.arguments == {"query": "test query"}

    def test_action_with_complex_arguments(self):
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


class TestToolCallingAgent:
    """Test the ToolCallingAgent class."""

    def test_init_with_defaults(self):
        """Test ToolCallingAgent initialization with defaults."""
        agent = ToolCallingAgent()

        assert agent.model == "openai/gpt-4o"
        assert agent.max_iterations == 10
        assert agent.temperature == 0.7
        assert agent.api_endpoint is None
        assert agent._available_tools is None

    def test_init_with_custom_params(self):
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

    def test_init_with_prompt_store(self, mock_prompt_store):
        """Test ToolCallingAgent initialization with PromptStore."""
        agent = ToolCallingAgent(
            prompt_store=mock_prompt_store,
            system_prompt_id="test-system-id",
            user_prompt_id="test-user-id",
        )

        # The prompt_store is stored as 'store' in BaseAgent
        assert hasattr(agent, "store")
        assert agent.store == mock_prompt_store

    @patch("corral.agents.tool_calling.convert_to_openai_tool_format")
    @patch("corral.agents.tool_calling.create_prompt")
    def test_run_setup(
        self, mock_create_prompt, mock_convert_tools, tool_calling_agent, mock_interface
    ):
        """Test that run method sets up correctly."""
        # Mock the prompt creation
        mock_create_prompt.return_value = [
            LiteLLMMessage(role="system", content="System prompt"),
            LiteLLMMessage(role="user", content="User prompt"),
        ]

        # Mock tool conversion
        mock_convert_tools.return_value = [
            {"type": "function", "function": {"name": "test_tool"}}
        ]

        # Mock get_llm_response to return final answer immediately
        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: Test result"
            )

            result = tool_calling_agent.run(mock_interface, "test_task")

            # Verify setup calls
            mock_interface.get_available_tools_for_task.assert_called_once_with(
                "test_task"
            )
            mock_interface.get_task_prompt.assert_called_once_with("test_task")
            mock_convert_tools.assert_called_once()
            mock_create_prompt.assert_called_once()

            # Verify tools are stored
            assert tool_calling_agent._available_tools == [
                {"type": "function", "function": {"name": "test_tool"}}
            ]

            assert result == "Test result"

    def test_run_with_final_answer(self, tool_calling_agent, mock_interface):
        """Test run method when LLM returns final answer immediately."""
        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: Direct answer"
            )

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Direct answer"
            assert len(tool_calling_agent.messages) == 3  # system + user + assistant
            assert tool_calling_agent.messages[-1]["role"] == "assistant"
            assert (
                tool_calling_agent.messages[-1]["content"]
                == "Final Answer: Direct answer"
            )

    def test_run_with_tool_calls(self, tool_calling_agent, mock_interface):
        """Test run method with tool calls."""
        # First response: tool call
        tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
        first_response = MockLLMResponse(content=None, tool_calls=[tool_call])

        # Second response: final answer
        second_response = MockLLMResponse(content="Final Answer: Tool result processed")

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Tool result processed"

            # Verify tool execution
            mock_interface.execute_tool.assert_called_once_with(
                "test_task", "test_tool", {"query": "test"}
            )

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

    def test_run_with_multiple_tool_calls(self, tool_calling_agent, mock_interface):
        """Test run method with multiple tool calls in one response."""
        tool_call_1 = MockToolCall("call_1", "test_tool", {"query": "first"})
        tool_call_2 = MockToolCall("call_2", "test_tool", {"query": "second"})

        first_response = MockLLMResponse(
            content=None, tool_calls=[tool_call_1, tool_call_2]
        )
        second_response = MockLLMResponse(
            content="Final Answer: Multiple tools processed"
        )

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Multiple tools processed"

            # Verify both tools were executed
            assert mock_interface.execute_tool.call_count == 2
            mock_interface.execute_tool.assert_has_calls(
                [
                    call("test_task", "test_tool", {"query": "first"}),
                    call("test_task", "test_tool", {"query": "second"}),
                ]
            )

            # Verify two tool messages were added
            tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
            assert len(tool_messages) == 2

    def test_run_with_tool_error(self, tool_calling_agent, mock_interface):
        """Test run method when tool execution fails."""
        # Mock tool execution failure
        mock_tool_response = Mock(spec=ToolResponse)
        mock_tool_response.result = None
        mock_tool_response.error = "Tool execution failed"
        mock_interface.execute_tool.return_value = mock_tool_response

        tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
        first_response = MockLLMResponse(content=None, tool_calls=[tool_call])
        second_response = MockLLMResponse(content="Final Answer: Handled error")

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Handled error"

            # Verify error was handled - the code converts str(None) to "None"
            tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
            assert len(tool_messages) == 1
            assert tool_messages[0]["content"] == "None"  # str(None) = "None"

    def test_run_with_tool_exception(self, tool_calling_agent, mock_interface):
        """Test run method when tool execution raises exception."""
        # Mock tool execution to raise exception
        mock_interface.execute_tool.side_effect = Exception("Unexpected error")

        tool_call = MockToolCall("call_1", "test_tool", {"query": "test"})
        first_response = MockLLMResponse(content=None, tool_calls=[tool_call])
        second_response = MockLLMResponse(content="Final Answer: Exception handled")

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Exception handled"

            # Verify exception was handled
            tool_messages = get_messages_by_role(tool_calling_agent.messages, "tool")
            assert len(tool_messages) == 1
            assert tool_messages[0]["content"] == "Unexpected error"

    def test_run_with_llm_response_exception(self, tool_calling_agent, mock_interface):
        """Test run method when LLM response raises exception."""
        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [
                Exception("LLM error"),
                MockLLMResponse(content="Final Answer: Recovered from error"),
            ]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Recovered from error"

            # Verify error message was added
            error_messages = get_messages_by_role(tool_calling_agent.messages, "system")
            assert len(error_messages) == 2  # initial system + error system
            assert (
                "Error during agent iteration: LLM error"
                in error_messages[1]["content"]
            )

    def test_run_with_content_but_no_final_answer(
        self, tool_calling_agent, mock_interface
    ):
        """Test run method when LLM returns content but no final answer."""
        first_response = MockLLMResponse(content="I need to think about this...")
        second_response = MockLLMResponse(content="Final Answer: Now I have the answer")

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Now I have the answer"

            # Verify intermediate response was added
            assistant_messages = get_messages_by_role(
                tool_calling_agent.messages, "assistant"
            )
            assert len(assistant_messages) == 2
            assert assistant_messages[0]["content"] == "I need to think about this..."

    def test_run_max_iterations_reached(self, tool_calling_agent, mock_interface):
        """Test run method when max iterations is reached."""
        # Set low max iterations for testing
        tool_calling_agent.max_iterations = 2

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Still thinking..."
            )

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "Error solving the task. Maximum iterations reached."

            # Verify max iterations was reached
            assert mock_llm_response.call_count == 2

            # Verify error message was added
            error_messages = [
                msg
                for msg in tool_calling_agent.messages
                if msg.get("role") == "assistant"
                and msg.get("name") == "tool-calling-error"
            ]
            assert len(error_messages) == 1
            assert "Maximum iterations reached" in error_messages[0]["content"]

    def test_run_with_custom_task_prompt(self, tool_calling_agent, mock_interface):
        """Test run method with custom task prompt."""
        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: Custom task result"
            )

            result = tool_calling_agent.run(
                mock_interface, "test_task", task_prompt="Custom task prompt"
            )

            assert result == "Custom task result"

            # Verify custom task prompt was used instead of interface prompt
            mock_interface.get_task_prompt.assert_not_called()

    def test_run_with_history(self, tool_calling_agent, mock_interface):
        """Test run method with conversation history."""
        history = [
            LiteLLMMessage(role="user", content="Previous question"),
            LiteLLMMessage(role="assistant", content="Previous answer"),
        ]

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: With history"
            )

            result = tool_calling_agent.run(
                mock_interface, "test_task", history=history
            )

            assert result == "With history"

            # Verify history was included in messages
            # Should have system + user + history + user (task) + assistant
            assert len(tool_calling_agent.messages) >= 4

    def test_run_with_examples(self, tool_calling_agent, mock_interface):
        """Test run method with examples."""
        examples = ["Example 1", "Example 2"]

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: With examples"
            )

            result = tool_calling_agent.run(
                mock_interface, "test_task", examples=examples
            )

            assert result == "With examples"

    def test_run_case_insensitive_final_answer(
        self, tool_calling_agent, mock_interface
    ):
        """Test that final answer matching is case insensitive."""
        test_cases = [
            "final answer: lowercase",
            "Final Answer: normal case",
            "FINAL ANSWER: uppercase",
            "Final answer: mixed case",
        ]

        for content in test_cases:
            with patch.object(
                tool_calling_agent, "get_llm_response"
            ) as mock_llm_response:
                mock_llm_response.return_value = MockLLMResponse(content=content)

                result = tool_calling_agent.run(mock_interface, "test_task")

                expected = content.split(":", 1)[1].strip()
                assert result == expected

    def test_run_final_answer_extraction(self, tool_calling_agent, mock_interface):
        """Test final answer extraction from complex content."""
        content = """
        I need to solve this step by step.

        First, let me analyze the problem...

        After careful consideration, I believe the answer is:

        Final Answer: The solution is 42 with detailed explanation.
        """

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(content=content)

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "The solution is 42 with detailed explanation."

    @patch("corral.agents.tool_calling.convert_to_openai_tool_format")
    def test_tools_conversion_called_correctly(
        self, mock_convert_tools, tool_calling_agent, mock_interface
    ):
        """Test that tools are converted to OpenAI format correctly."""
        expected_tools = [{"type": "function", "function": {"name": "test_tool"}}]
        mock_convert_tools.return_value = expected_tools

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: Test"
            )

            tool_calling_agent.run(mock_interface, "test_task")

            # Verify convert_to_openai_tool_format was called with correct tools
            mock_convert_tools.assert_called_once()
            args, kwargs = mock_convert_tools.call_args
            assert len(args) == 1
            # The mock_interface now returns a dict with tools key
            expected_arg = mock_interface.get_available_tools_for_task.return_value
            assert args[0] == expected_arg

    def test_available_tools_storage(self, tool_calling_agent, mock_interface):
        """Test that available tools are stored correctly."""
        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.return_value = MockLLMResponse(
                content="Final Answer: Test"
            )

            # Initially no tools
            assert tool_calling_agent._available_tools is None

            tool_calling_agent.run(mock_interface, "test_task")

            # Tools should be stored after run
            assert tool_calling_agent._available_tools is not None
            assert isinstance(tool_calling_agent._available_tools, list)

    def test_json_parsing_in_tool_calls(self, tool_calling_agent, mock_interface):
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

        with patch.object(tool_calling_agent, "get_llm_response") as mock_llm_response:
            mock_llm_response.side_effect = [first_response, second_response]

            result = tool_calling_agent.run(mock_interface, "test_task")

            assert result == "JSON parsed correctly"

            # Verify JSON was parsed correctly
            mock_interface.execute_tool.assert_called_once_with(
                "test_task", "test_tool", complex_args
            )

    def test_docstring_requirements(self):
        """Test that the class docstring mentions required prompt fields."""
        docstring = ToolCallingAgent.__doc__

        # Check that docstring exists and contains required fields
        assert docstring is not None
        assert "{{task_guide}}" in docstring
        assert "{{examples}}" in docstring
        assert "Required Prompt Fields" in docstring

        # Check that it mentions function calling
        assert "function calling" in docstring or "tool calling" in docstring

    def test_inheritance_from_base_agent(self):
        """Test that ToolCallingAgent inherits from BaseAgent."""
        from corral.agents.base_agent import BaseAgent

        assert issubclass(ToolCallingAgent, BaseAgent)

        # Test that it has the required abstract method
        agent = ToolCallingAgent()
        assert hasattr(agent, "run")
        assert callable(agent.run)


class TestToolCallingAgentDefaultPrompts:
    """Test that the default prompts contain the expected information."""

    def test_default_system_prompt_content(self):
        """Test that the default system prompt contains expected content."""
        import importlib.resources

        from promptstore import PromptStore

        # Get the actual prompt store path
        with importlib.resources.path("corral.agents", "prompts") as prompts_path:
            store = PromptStore(prompts_path)

            # Get the system prompt using the default ID
            system_prompt = store.get("400fcecf-f5f2-464b-aff5-8a4377c9685c")

            # Check that it contains expected content
            content = system_prompt.content
            assert "autonomous agent" in content.lower()
            assert "environment" in content.lower()

    def test_default_user_prompt_content(self):
        """Test that the default user prompt contains expected placeholders and instructions."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_default_extractor_prompt_content(self):
        """Test that the default extractor prompt contains expected placeholders."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_user_prompt_template_variables(self):
        """Test that the user prompt template supports required variables."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_extractor_prompt_template_variables(self):
        """Test that the extractor prompt template supports required variables."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_default_prompt_ids_match_class_defaults(self):
        """Test that the default prompt IDs in the class match the available prompts."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_user_prompt_does_not_contain_tool_descriptions(self):
        """Test that the user prompt does not contain tool descriptions (as per docstring)."""
        import importlib.resources

        from promptstore import PromptStore

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

    def test_user_prompt_missing_examples_variable(self):
        """Test that the user prompt handles missing examples variable gracefully."""
        import importlib.resources

        from promptstore import PromptStore

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
