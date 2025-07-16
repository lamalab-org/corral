"""Comprehensive tests for the ReActAgent class."""

from typing import Any
from unittest.mock import Mock, patch

from corral.agents.react import Action, ReActAgent, Thought
from corral.agents.utils import LiteLLMMessage
from corral.report import ToolResponse

# Try to import pytest if available
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    pytest = None
    HAS_PYTEST = False

try:
    from promptstore import PromptStore
except ImportError:
    PromptStore = None


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


class MockBenchmarkInterface:
    """Mock BenchmarkInterface for testing."""

    def __init__(self):
        self.task_guide = "Test task guide"
        self.tool_responses = []
        self.tool_calls = []

    def get_task_guide(self, task_id: str) -> str:
        return self.task_guide

    def execute_tool(
        self, task_id: str, tool_name: str, arguments: dict
    ) -> ToolResponse:
        self.tool_calls.append(
            {"task_id": task_id, "tool_name": tool_name, "arguments": arguments}
        )
        if self.tool_responses:
            return self.tool_responses.pop(0)
        return ToolResponse(success=True, result="Mock tool result", error=None)


def create_mock_prompt_store():
    """Create mock PromptStore for testing."""
    if PromptStore is None:
        return None
    store = Mock(spec=PromptStore)
    store.get.return_value = MockPrompt("Test prompt: {{task_guide}}")
    return store


def create_mock_interface():
    """Create mock BenchmarkInterface for testing."""
    return MockBenchmarkInterface()


def create_react_agent():
    """Create a ReActAgent instance for testing."""
    return ReActAgent(model="test-model", max_iterations=3, temperature=0.5)


# Conditionally set up pytest fixtures if available
if HAS_PYTEST and pytest is not None:

    @pytest.fixture
    def mock_prompt_store():
        """Mock PromptStore for testing."""
        return create_mock_prompt_store()

    @pytest.fixture
    def mock_interface():
        """Mock BenchmarkInterface for testing."""
        return create_mock_interface()

    @pytest.fixture
    def react_agent():
        """Create a ReActAgent instance for testing."""
        return create_react_agent()
else:
    # Define regular functions for non-pytest usage
    def mock_prompt_store():
        return create_mock_prompt_store()

    def mock_interface():
        return create_mock_interface()

    def react_agent():
        return create_react_agent()


class TestThought:
    """Test cases for the Thought dataclass."""

    def test_thought_creation(self):
        """Test creating a Thought object."""
        content = "I need to analyze the problem"
        thought = Thought(content=content)
        assert thought.content == content
        assert isinstance(thought.content, str)

    def test_thought_empty_content(self):
        """Test creating a Thought with empty content."""
        thought = Thought(content="")
        assert thought.content == ""

    def test_thought_multiline_content(self):
        """Test creating a Thought with multiline content."""
        content = "First line\nSecond line\nThird line"
        thought = Thought(content=content)
        assert thought.content == content
        assert "\n" in thought.content


class TestAction:
    """Test cases for the Action dataclass."""

    def test_action_creation(self):
        """Test creating an Action object."""
        tool_name = "test_tool"
        arguments = {"param1": "value1", "param2": 42}
        action = Action(tool_name=tool_name, arguments=arguments)

        assert action.tool_name == tool_name
        assert action.arguments == arguments
        assert isinstance(action.arguments, dict)

    def test_action_empty_arguments(self):
        """Test creating an Action with empty arguments."""
        action = Action(tool_name="test_tool", arguments={})
        assert action.tool_name == "test_tool"
        assert action.arguments == {}

    def test_action_complex_arguments(self):
        """Test creating an Action with complex arguments."""
        arguments = {
            "string_param": "test",
            "int_param": 123,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
        }
        action = Action(tool_name="complex_tool", arguments=arguments)
        assert action.arguments == arguments


class TestReActAgentInitialization:
    """Test cases for ReActAgent initialization."""

    def test_default_initialization(self):
        """Test ReActAgent initialization with default parameters."""
        agent = ReActAgent()

        assert agent.model == "openai/gpt-4o"
        assert agent.max_iterations == 10
        assert agent.temperature == 0.7
        assert agent.api_endpoint is None

    def test_custom_initialization(self):
        """Test ReActAgent initialization with custom parameters."""
        agent = ReActAgent(
            model="custom-model",
            max_iterations=5,
            temperature=0.3,
            api_endpoint="http://custom-endpoint",
        )

        assert agent.model == "custom-model"
        assert agent.max_iterations == 5
        assert agent.temperature == 0.3
        assert agent.api_endpoint == "http://custom-endpoint"

    def test_initialization_with_custom_prompts(self):
        """Test ReActAgent initialization with custom prompts."""
        system_prompt = "Custom system prompt"
        user_prompt = "Custom user prompt: {{task_guide}}"

        agent = ReActAgent(system_prompt=system_prompt, user_prompt=user_prompt)

        assert agent.system_prompt == system_prompt
        assert agent.user_prompt == user_prompt

    def test_initialization_with_prompt_store(self, mock_prompt_store):
        """Test ReActAgent initialization with PromptStore."""
        if PromptStore is None:
            return  # Skip test if PromptStore not available
        agent = ReActAgent(prompt_store=mock_prompt_store)
        assert agent.store == mock_prompt_store

    def test_initialization_with_kwargs(self):
        """Test ReActAgent initialization with additional kwargs."""
        agent = ReActAgent(
            model="test-model", custom_param="custom_value", another_param=42
        )

        assert agent.model == "test-model"
        # Additional kwargs should be passed to parent class


class TestReActAgentParsing:
    """Test cases for ReActAgent response parsing."""

    def test_parse_llm_response_thought_only(self, react_agent):
        """Test parsing response with only a thought."""
        response = "Thought: I need to analyze this problem carefully."

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert thought.content == "I need to analyze this problem carefully."
        assert actions is None

    def test_parse_llm_response_thought_and_action(self, react_agent):
        """Test parsing response with thought and action."""
        response = """Thought: I need to search for information.
Action: search
Action Input: {"query": "test query", "limit": 10}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert thought.content == "I need to search for information."
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test query", "limit": 10}

    def test_parse_llm_response_multiple_actions(self, react_agent):
        """Test parsing response with multiple actions."""
        response = """Thought: I need to use multiple tools.
Action: search
Action Input: {"query": "test"}
Action: calculate
Action Input: {"expression": "2+2"}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 2
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test"}
        assert actions[1].tool_name == "calculate"
        assert actions[1].arguments == {"expression": "2+2"}

    def test_parse_llm_response_final_answer(self, react_agent):
        """Test parsing response with final answer."""
        response = """Thought: I have found the answer.
Final Answer: The result is 42."""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert thought.content == "I have found the answer."
        assert actions is None

    def test_parse_llm_response_invalid_json(self, react_agent):
        """Test parsing response with invalid JSON in action input."""
        response = """Thought: Testing invalid JSON.
Action: search
Action Input: {invalid json}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None  # Should be None due to JSON parsing error

    def test_parse_llm_response_no_thought(self, react_agent):
        """Test parsing response with no thought."""
        response = """Action: search
Action Input: {"query": "test"}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"

    def test_parse_llm_response_empty_response(self, react_agent):
        """Test parsing empty response."""
        response = ""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is None
        assert actions is None

    def test_parse_llm_response_multiline_thought(self, react_agent):
        """Test parsing response with multiline thought."""
        response = """Thought: This is a complex problem that requires
multiple lines of reasoning to solve properly.
Let me break it down step by step.
Action: search
Action Input: {"query": "complex problem"}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert (
            "This is a complex problem that requires\nmultiple lines of reasoning"
            in thought.content
        )
        assert actions is not None
        assert len(actions) == 1


class TestReActAgentRun:
    """Test cases for ReActAgent.run method."""

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_final_answer(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method that returns final answer immediately."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "Thought: I can answer this directly.\nFinal Answer: 42"
        mock_get_llm_response.return_value = mock_response

        # Mock create_prompt
        mock_create_prompt.return_value = []

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "42"
        mock_create_prompt.assert_called_once()
        mock_get_llm_response.assert_called_once()

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_tool_execution(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method that executes tools before finding answer."""
        # Mock the LLM responses
        response1 = Mock()
        response1.content = """Thought: I need to search for information.
Action: search
Action Input: {"query": "test"}"""

        response2 = Mock()
        response2.content = (
            "Thought: Based on the search results.\nFinal Answer: Found it!"
        )

        mock_get_llm_response.side_effect = [response1, response2]

        # Mock create_prompt
        mock_create_prompt.return_value = []

        # Set up tool response
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Search results", error=None)
        ]

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "Found it!"
        assert len(mock_interface.tool_calls) == 1
        assert mock_interface.tool_calls[0]["tool_name"] == "search"
        assert mock_interface.tool_calls[0]["arguments"] == {"query": "test"}

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_tool_error(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method with tool execution error."""
        # Mock the LLM responses
        response1 = Mock()
        response1.content = """Thought: I need to use a tool.
Action: failing_tool
Action Input: {"param": "value"}"""

        response2 = Mock()
        response2.content = "Thought: The tool failed.\nFinal Answer: Handled error"

        mock_get_llm_response.side_effect = [response1, response2]

        # Mock create_prompt
        mock_create_prompt.return_value = []

        # Set up tool error response
        mock_interface.tool_responses = [
            ToolResponse(success=False, result=None, error="Tool failed")
        ]

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "Handled error"
        assert len(mock_interface.tool_calls) == 1

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_max_iterations_exceeded(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method when max iterations is exceeded."""
        # Mock the LLM response that never returns final answer
        mock_response = Mock()
        mock_response.content = """Thought: I'm thinking about this problem.
Action: search
Action Input: {"query": "test"}"""

        mock_get_llm_response.return_value = mock_response

        # Mock create_prompt
        mock_create_prompt.return_value = []

        # Set up tool response
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Result", error=None)
        ] * 10

        result = react_agent.run(mock_interface, "test_task_id")

        assert (
            result
            == "Error solving the task: unable to complete it in the iteration limit"
        )
        assert (
            len(mock_interface.tool_calls) == 3
        )  # max_iterations is 3 for this test agent

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_multiple_tools_in_one_response(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method with multiple tools in one response."""
        # Mock the LLM response
        response1 = Mock()
        response1.content = """Thought: I need to use multiple tools.
Action: search
Action Input: {"query": "test"}
Action: calculate
Action Input: {"expression": "2+2"}"""

        response2 = Mock()
        response2.content = "Thought: Got all results.\nFinal Answer: Complete"

        mock_get_llm_response.side_effect = [response1, response2]

        # Mock create_prompt
        mock_create_prompt.return_value = []

        # Set up tool responses
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Search result", error=None),
            ToolResponse(success=True, result="4", error=None),
        ]

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "Complete"
        assert len(mock_interface.tool_calls) == 2
        assert mock_interface.tool_calls[0]["tool_name"] == "search"
        assert mock_interface.tool_calls[1]["tool_name"] == "calculate"

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_custom_task_prompt(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method with custom task prompt."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "Thought: Custom task.\nFinal Answer: Done"
        mock_get_llm_response.return_value = mock_response

        # Mock create_prompt
        mock_create_prompt.return_value = []

        custom_task_prompt = "Custom task description"
        result = react_agent.run(
            mock_interface, "test_task_id", task_prompt=custom_task_prompt
        )

        assert result == "Done"

        # Verify create_prompt was called with custom task prompt
        call_args = mock_create_prompt.call_args
        assert call_args[1]["task_guide"] == custom_task_prompt

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_history_and_examples(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method with history and examples."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = (
            "Thought: Using history and examples.\nFinal Answer: Success"
        )
        mock_get_llm_response.return_value = mock_response

        # Mock create_prompt
        mock_create_prompt.return_value = []

        history = [LiteLLMMessage(role="user", content="Previous message")]
        examples = ["Example 1", "Example 2"]

        result = react_agent.run(
            mock_interface, "test_task_id", history=history, examples=examples
        )

        assert result == "Success"

        # Verify create_prompt was called with history and examples
        call_args = mock_create_prompt.call_args
        assert call_args[1]["history"] == history
        assert call_args[1]["examples"] == examples

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_message_construction(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test that messages are constructed correctly during run."""
        # Mock the LLM responses
        response1 = Mock()
        response1.content = """Thought: I need to search.
Action: search
Action Input: {"query": "test"}"""

        response2 = Mock()
        response2.content = "Thought: Found it.\nFinal Answer: Result"

        mock_get_llm_response.side_effect = [response1, response2]

        # Mock create_prompt to return a list we can modify
        initial_messages = [LiteLLMMessage(role="system", content="System prompt")]
        mock_create_prompt.return_value = initial_messages

        # Set up tool response
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Search result", error=None)
        ]

        # Check that messages were added correctly
        assert (
            len(react_agent.messages) >= 4
        )  # Initial + assistant action + user observation + final answer

        # Check message structure
        messages = react_agent.messages
        assert any(
            msg.get("role") == "assistant"
            and "Action: search" in msg.get("content", "")
            for msg in messages
        )
        assert any(
            msg.get("role") == "user"
            and "Observation: Search result" in msg.get("content", "")
            for msg in messages
        )
        assert any(
            msg.get("role") == "assistant"
            and "Final Answer: Result" in msg.get("content", "")
            for msg in messages
        )


class TestReActAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_response_with_malformed_action(self, react_agent):
        """Test parsing response with malformed action structure."""
        response = """Thought: Testing malformed action.
Action: search
Action Input: not valid json at all"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None

    def test_parse_response_with_nested_final_answer(self, react_agent):
        """Test parsing response with nested final answer pattern."""
        response = """Thought: The answer mentions "Final Answer: not really" in the text.
Final Answer: The actual final answer is 42."""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None

    def test_parse_response_with_special_characters(self, react_agent):
        """Test parsing response with special characters in JSON."""
        response = """Thought: Testing special characters.
Action: search
Action Input: {"query": "test with \\"quotes\\" and \\n newlines", "special": "chars: !@#$%^&*()"}"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert "quotes" in actions[0].arguments["query"]
        assert actions[0].arguments["special"] == "chars: !@#$%^&*()"

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_run_with_empty_llm_response(
        self, mock_create_prompt, mock_get_llm_response, react_agent, mock_interface
    ):
        """Test run method with empty LLM response."""
        # Mock empty LLM response
        mock_response = Mock()
        mock_response.content = ""
        mock_get_llm_response.return_value = mock_response

        # Mock create_prompt
        mock_create_prompt.return_value = []

        result = react_agent.run(mock_interface, "test_task_id")

        assert (
            result
            == "Error solving the task: unable to complete it in the iteration limit"
        )

    def test_initialization_with_none_values(self):
        """Test ReActAgent initialization with None values."""
        agent = ReActAgent(
            system_prompt=None,
            user_prompt=None,
            extractor_prompt=None,
            api_endpoint=None,
        )

        # When system_prompt is None, it loads from prompt store with default ID
        assert (
            agent.system_prompt is not None
        )  # It gets a default value from prompt store
        assert (
            agent.user_prompt is not None
        )  # It gets a default value from prompt store
        assert (
            agent.extractor_prompt is not None
        )  # It gets a default value from prompt store
        assert agent.api_endpoint is None


class TestReActAgentIntegration:
    """Integration tests for ReActAgent with real-like scenarios."""

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_complete_react_cycle(
        self, mock_create_prompt, mock_get_llm_response, mock_interface
    ):
        """Test a complete ReAct cycle with realistic interaction."""
        agent = ReActAgent(model="test-model", max_iterations=5)

        # Mock realistic LLM responses
        responses = [
            # First iteration - analyze problem
            Mock(
                content="""Thought: I need to understand the problem first.
Action: analyze
Action Input: {"text": "problem statement"}"""
            ),
            # Second iteration - search for information
            Mock(
                content="""Thought: Now I need to search for relevant information.
Action: search
Action Input: {"query": "relevant information", "limit": 5}"""
            ),
            # Third iteration - process results and provide answer
            Mock(
                content="""Thought: Based on the analysis and search results, I can now provide the answer.
Final Answer: The solution is X because of Y and Z."""
            ),
        ]

        mock_get_llm_response.side_effect = responses
        mock_create_prompt.return_value = []

        # Set up tool responses
        mock_interface.tool_responses = [
            ToolResponse(
                success=True, result="Analysis complete: problem identified", error=None
            ),
            ToolResponse(
                success=True, result="Search results: relevant data found", error=None
            ),
        ]

        result = agent.run(mock_interface, "complex_task")

        assert result == "The solution is X because of Y and Z."
        assert len(mock_interface.tool_calls) == 2
        assert mock_interface.tool_calls[0]["tool_name"] == "analyze"
        assert mock_interface.tool_calls[1]["tool_name"] == "search"

        # Verify message flow
        assert (
            len(agent.messages) >= 5
        )  # 2 tool calls + 2 observations + final answer (create_prompt mocked to return empty list)

    @patch("corral.agents.base_agent.BaseAgent.get_llm_response")
    @patch("corral.agents.react.create_prompt")
    def test_error_recovery_scenario(
        self, mock_create_prompt, mock_get_llm_response, mock_interface
    ):
        """Test ReActAgent handling tool errors and recovery."""
        agent = ReActAgent(model="test-model", max_iterations=5)

        # Mock responses with error recovery
        responses = [
            # First iteration - try a tool that fails
            Mock(
                content="""Thought: I'll try using this tool.
Action: failing_tool
Action Input: {"param": "value"}"""
            ),
            # Second iteration - recover from error
            Mock(
                content="""Thought: The tool failed, let me try a different approach.
Action: backup_tool
Action Input: {"alternative": "approach"}"""
            ),
            # Third iteration - provide answer
            Mock(
                content="""Thought: This approach worked.
Final Answer: Successfully recovered and found the answer."""
            ),
        ]

        mock_get_llm_response.side_effect = responses
        mock_create_prompt.return_value = []

        # Set up tool responses - first fails, second succeeds
        mock_interface.tool_responses = [
            ToolResponse(
                success=False, result=None, error="Tool failed with error message"
            ),
            ToolResponse(success=True, result="Backup approach successful", error=None),
        ]

        result = agent.run(mock_interface, "error_prone_task")

        assert result == "Successfully recovered and found the answer."
        assert len(mock_interface.tool_calls) == 2
        assert mock_interface.tool_calls[0]["tool_name"] == "failing_tool"
        assert mock_interface.tool_calls[1]["tool_name"] == "backup_tool"

        # Check that error message was included in conversation
        error_message_found = any(
            msg.get("role") == "user"
            and "Error: Tool failed with error message" in msg.get("content", "")
            for msg in agent.messages
        )
        assert error_message_found
