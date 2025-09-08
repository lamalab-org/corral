"""Comprehensive tests for the ReActAgent class."""

import pytest

from corral.agents.react import Action, ReActAgent, Thought
from corral.agents.utils import LiteLLMMessage
from corral.report import ToolResponse

from .conftest import MockLLMResponse


def create_react_agent():
    """Create a ReActAgent instance for testing."""
    return ReActAgent(model="test-model", max_iterations=3, temperature=0.5)


@pytest.fixture()
def react_agent():
    """Create a ReActAgent instance for testing."""
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

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert thought.content == "I need to analyze this problem carefully."
        assert actions is None
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_thought_and_action(self, react_agent):
        """Test parsing response with thought and action."""
        response = """Thought: I need to search for information.
Action: search
Action Input: {"query": "test query", "limit": 10}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert thought.content == "I need to search for information."
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test query", "limit": 10}
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_multiple_actions(self, react_agent):
        """Test parsing response with multiple actions."""
        response = """Thought: I need to use multiple tools.
Action: search
Action Input: {"query": "test"}
Action: calculate
Action Input: {"expression": "2+2"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is not None
        assert len(actions) == 2
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test"}
        assert actions[1].tool_name == "calculate"
        assert actions[1].arguments == {"expression": "2+2"}
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_final_answer(self, react_agent):
        """Test parsing response with final answer."""
        response = """Thought: I have found the answer.
Final Answer: The result is 42."""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert thought.content == "I have found the answer."
        assert actions is None
        assert is_final is True
        assert parsing_error is None

    def test_parse_llm_response_invalid_json(self, react_agent):
        """Test parsing response with invalid JSON in action input."""
        response = """Thought: Testing invalid JSON.
Action: search
Action Input: {invalid json}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is None  # Should be None due to JSON parsing error
        assert is_final is False
        assert parsing_error is not None  # Should capture the parsing error
        assert "Invalid JSON in Action Input for 'search'" in parsing_error
        assert "json" in parsing_error.lower()  # Should mention JSON error

    def test_parse_llm_response_no_thought(self, react_agent):
        """Test parsing response with no thought."""
        response = """Action: search
Action Input: {"query": "test"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test"}
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_empty_response(self, react_agent):
        """Test parsing empty response."""
        response = ""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is None
        assert actions is None
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_multiline_thought(self, react_agent):
        """Test parsing response with multiline thought."""
        response = """Thought: This is a complex problem that requires
multiple lines of reasoning to solve properly.
Let me break it down step by step.
Action: search
Action Input: {"query": "complex problem"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert (
            "This is a complex problem that requires\nmultiple lines of reasoning"
            in thought.content
        )
        assert actions is not None
        assert len(actions) == 1
        assert is_final is False
        assert parsing_error is None

    def test_parse_llm_response_empty_thought_variations(self, react_agent):
        """Test parsing responses with various empty thought patterns."""
        test_cases = [
            ("Thought:", "Empty thought with colon only"),
            ("Thought: ", "Empty thought with space"),
            ("Thought:\n", "Empty thought with newline"),
            ("Thought: \n", "Empty thought with space and newline"),
            (
                "Thought:   \nAction: test\nAction Input: {}",
                "Whitespace-only thought with action",
            ),
        ]

        for response, description in test_cases:
            thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
                response
            )
            # All empty thought variations should return None for thought
            assert thought is None, f"Failed for case: {description}"
            assert is_final is False, f"Failed for case: {description}"
            assert parsing_error is None, f"Failed for case: {description}"

            # Check if actions are parsed correctly when present
            if "Action:" in response:
                assert actions is not None, f"Failed for case: {description}"
                assert len(actions) == 1, f"Failed for case: {description}"
            else:
                assert actions is None, f"Failed for case: {description}"

    def test_parse_llm_response_final_answer_only(self, react_agent):
        """Test parsing response with only final answer."""
        response = "Final Answer: Direct answer without thought"

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is None
        assert actions is None
        assert is_final is True
        assert parsing_error is None

    def test_parse_llm_response_all_components(self, react_agent):
        """Test parsing response with thought, actions, and final answer."""
        response = """Thought: I need to search and then provide an answer.
Action: search
Action Input: {"query": "test"}
Action: analyze
Action Input: {"data": "results"}
Final Answer: Based on my analysis, the answer is 42."""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert thought.content == "I need to search and then provide an answer."
        assert actions is not None
        assert len(actions) == 2
        assert actions[0].tool_name == "search"
        assert actions[1].tool_name == "analyze"
        assert is_final is True
        assert parsing_error is None


class TestReActAgentRun:
    """Test cases for ReActAgent.run method."""

    def test_run_with_final_answer(self, react_agent, mock_interface, monkeypatch):
        """Test run method that returns final answer immediately."""
        # Mock the LLM response
        mock_response = MockLLMResponse(
            content="Thought: I can answer this directly.\nFinal Answer: 42"
        )

        # Mock the methods using monkeypatch
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return mock_response

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "42"
        assert call_tracker["create_prompt"] == 1
        assert call_tracker["get_llm_response"] == 1

    def test_run_with_tool_execution(self, react_agent, mock_interface, monkeypatch):
        """Test run method that executes tools before finding answer."""
        # Mock the LLM responses
        response1 = MockLLMResponse(
            content="""Thought: I need to search for information.
Action: search
Action Input: {"query": "test"}"""
        )

        response2 = MockLLMResponse(
            content=("Thought: Based on the search results.\nFinal Answer: Found it!")
        )

        # Mock the methods using monkeypatch
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}
        responses = [response1, response2]

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return responses[call_tracker["get_llm_response"] - 1]

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        # Set up tool response
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Search results", error=None)
        ]

        result = react_agent.run(mock_interface, "test_task_id")

        assert result == "Found it!"
        assert len(mock_interface.tool_calls) == 1
        assert mock_interface.tool_calls[0]["tool_name"] == "search"
        assert mock_interface.tool_calls[0]["arguments"] == {"query": "test"}

    def test_run_with_tool_error(self, react_agent, mock_interface, monkeypatch):
        """Test run method with tool execution error."""
        # Mock create_prompt
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        # Set up tool error response
        mock_interface.tool_responses = [
            ToolResponse(success=False, result=None, error="Tool failed")
        ]

        # Define test scenarios: (response_content, expected_final_result, expected_tool_calls_count, iteration_description)
        test_scenarios = [
            (
                """Thought: I need to use a tool.
Action: failing_tool
Action Input: {"param": "value"}""",
                None,  # No final result yet, should continue iterating
                1,  # Should have made 1 tool call
                "First iteration: tool action with error",
            ),
            (
                "Thought: The tool failed.\nFinal Answer: Handled error",
                "Handled error",  # Should return this as final result
                1,  # Still just 1 tool call total
                "Second iteration: final answer after tool error",
            ),
        ]

        responses = []

        def mock_llm_response(*args, **kwargs):
            if call_tracker["get_llm_response"] < len(test_scenarios):
                response = MockLLMResponse(
                    content=test_scenarios[call_tracker["get_llm_response"]][0]
                )
                responses.append(response)
                call_tracker["get_llm_response"] += 1
                return response
            else:
                # Fallback to prevent infinite loops in testing
                return MockLLMResponse(content="Final Answer: Fallback result")

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_llm_response
        )

        # Run the agent
        result = react_agent.run(mock_interface, "test_task_id")

        # Verify final result
        assert result == "Handled error"
        assert len(mock_interface.tool_calls) == 1
        assert mock_interface.tool_calls[0]["tool_name"] == "failing_tool"
        assert mock_interface.tool_calls[0]["arguments"] == {"param": "value"}

        # Verify that we made the expected number of LLM calls
        assert call_tracker["get_llm_response"] == 2
        assert len(responses) == 2

        # Additional verification: check that the agent's message history reflects the error handling
        messages = react_agent.messages

        # Should contain the tool action
        assert any(
            msg.get("role") == "assistant"
            and "Action: failing_tool" in msg.get("content", "")
            for msg in messages
        )

        # Should contain the error observation
        assert any(
            msg.get("role") == "user" and "Error: Tool failed" in msg.get("content", "")
            for msg in messages
        )

        # Should contain the final answer
        assert any(
            msg.get("role") == "assistant"
            and "Final Answer: Handled error" in msg.get("content", "")
            for msg in messages
        )

    def test_run_max_iterations_exceeded(
        self, react_agent, mock_interface, monkeypatch
    ):
        """Test run method when max iterations is exceeded."""
        # Mock the LLM response that never returns final answer
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return MockLLMResponse(
                content="""Thought: I'm thinking about this problem.
Action: search
Action Input: {"query": "test"}"""
            )

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

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

    def test_run_with_multiple_tools_in_one_response(
        self, react_agent, mock_interface, monkeypatch
    ):
        """Test run method with multiple tools in one response."""
        # Mock the LLM responses
        response1 = MockLLMResponse(
            content="""Thought: I need to use multiple tools.
Action: search
Action Input: {"query": "test"}
Action: calculate
Action Input: {"expression": "2+2"}"""
        )

        response2 = MockLLMResponse(
            content="Thought: Got all results.\nFinal Answer: Complete"
        )

        call_tracker = {"get_llm_response": 0, "create_prompt": 0}
        responses = [response1, response2]

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return responses[call_tracker["get_llm_response"] - 1]

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

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

    def test_run_with_custom_task_prompt(
        self, react_agent, mock_interface, monkeypatch
    ):
        """Test run method with custom task prompt."""
        # Mock the LLM response
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}
        create_prompt_calls = []

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return MockLLMResponse(content="Thought: Custom task.\nFinal Answer: Done")

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            create_prompt_calls.append((args, kwargs))
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        custom_task_prompt = "Custom task description"
        result = react_agent.run(
            mock_interface, "test_task_id", task_prompt=custom_task_prompt
        )

        assert result == "Done"

        # Verify create_prompt was called with custom task prompt
        assert len(create_prompt_calls) == 1
        assert create_prompt_calls[0][1]["task_guide"] == custom_task_prompt

    def test_run_with_history_and_examples(
        self, react_agent, mock_interface, monkeypatch
    ):
        """Test run method with history and examples."""
        # Mock the LLM response
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}
        create_prompt_calls = []

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return MockLLMResponse(
                content=("Thought: Using history and examples.\nFinal Answer: Success")
            )

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            create_prompt_calls.append((args, kwargs))
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        history = [LiteLLMMessage(role="user", content="Previous message")]
        examples = ["Example 1", "Example 2"]

        result = react_agent.run(
            mock_interface, "test_task_id", history=history, examples=examples
        )

        assert result == "Success"

        # Verify create_prompt was called with history and examples
        assert len(create_prompt_calls) == 1
        call_args = create_prompt_calls[0]
        assert call_args[1]["history"] == history
        assert call_args[1]["examples"] == examples

    def test_run_message_construction(self, react_agent, mock_interface, monkeypatch):
        """Test that messages are constructed correctly during run."""
        # Mock the LLM responses
        response1 = MockLLMResponse(
            content="""Thought: I need to search.
Action: search
Action Input: {"query": "test"}"""
        )

        response2 = MockLLMResponse(content="Thought: Found it.\nFinal Answer: Result")

        call_tracker = {"get_llm_response": 0, "create_prompt": 0}
        responses = [response1, response2]

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return responses[call_tracker["get_llm_response"] - 1]

        # Mock create_prompt to return a list we can modify
        initial_messages = [LiteLLMMessage(role="system", content="System prompt")]

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return initial_messages

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        # Set up tool response
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="Search result", error=None)
        ]

        react_agent.run(mock_interface, "test_task_id")

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

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is None
        assert is_final is False
        assert parsing_error is not None  # Should capture the parsing error
        assert "Invalid JSON in Action Input for 'search'" in parsing_error

    def test_parse_llm_response_parsing_error_feedback(self, react_agent):
        """Test that parsing errors provide detailed feedback."""
        response = """Thought: I'll try using a tool with malformed JSON.
Action: test_tool
Action Input: {malformed: "json", missing_quotes: value}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert thought.content == "I'll try using a tool with malformed JSON."
        assert actions is None  # Should be None due to JSON parsing error
        assert is_final is False
        assert parsing_error is not None
        assert "Invalid JSON in Action Input for 'test_tool'" in parsing_error
        assert "json" in parsing_error.lower()  # Should mention JSON error

    def test_parse_llm_response_multiple_parsing_errors(self, react_agent):
        """Test that only the first parsing error is captured."""
        response = """Thought: Testing multiple malformed actions.
Action: first_tool
Action Input: {invalid: json}
Action: second_tool
Action Input: {also invalid}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is None  # Should be None due to JSON parsing errors
        assert is_final is False
        assert parsing_error is not None
        # Should capture only the first error
        assert "first_tool" in parsing_error
        assert "second_tool" not in parsing_error

    def test_parse_response_with_nested_final_answer(self, react_agent):
        """Test parsing response with nested final answer pattern."""
        response = """Thought: The answer mentions "Final Answer: not really" in the text.
Final Answer: The actual final answer is 42."""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is None
        assert is_final is True
        assert parsing_error is None

    def test_parse_response_with_special_characters(self, react_agent):
        """Test parsing response with special characters in JSON."""
        response = """Thought: Testing special characters.
Action: search
Action Input: {"query": "test with \\"quotes\\" and \\n newlines", "special": "chars: !@#$%^&*()"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert "quotes" in actions[0].arguments["query"]
        assert actions[0].arguments["special"] == "chars: !@#$%^&*()"
        assert is_final is False
        assert parsing_error is None

    def test_run_with_empty_llm_response(
        self, react_agent, mock_interface, monkeypatch
    ):
        """Test run method with empty LLM response."""
        # Mock empty LLM response
        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return MockLLMResponse(content="")

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

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

    def test_complete_react_cycle(self, mock_interface, monkeypatch):
        """Test a complete ReAct cycle with realistic interaction."""
        agent = ReActAgent(model="test-model", max_iterations=5)

        # Mock realistic LLM responses
        responses = [
            # First iteration - analyze problem
            MockLLMResponse(
                content="""Thought: I need to understand the problem first.
Action: analyze
Action Input: {"text": "problem statement"}"""
            ),
            # Second iteration - search for information
            MockLLMResponse(
                content="""Thought: Now I need to search for relevant information.
Action: search
Action Input: {"query": "relevant information", "limit": 5}"""
            ),
            # Third iteration - process results and provide answer
            MockLLMResponse(
                content="""Thought: Based on the analysis and search results, I can now provide the answer.
Final Answer: The solution is X because of Y and Z."""
            ),
        ]

        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return responses[call_tracker["get_llm_response"] - 1]

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

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

    def test_error_recovery_scenario(self, mock_interface, monkeypatch):
        """Test ReActAgent handling tool errors and recovery."""
        agent = ReActAgent(model="test-model", max_iterations=5)

        # Mock responses with error recovery
        responses = [
            # First iteration - try a tool that fails
            MockLLMResponse(
                content="""Thought: I'll try using this tool.
Action: failing_tool
Action Input: {"param": "value"}"""
            ),
            # Second iteration - recover from error
            MockLLMResponse(
                content="""Thought: The tool failed, let me try a different approach.
Action: backup_tool
Action Input: {"alternative": "approach"}"""
            ),
            # Third iteration - provide answer
            MockLLMResponse(
                content="""Thought: This approach worked.
Final Answer: Successfully recovered and found the answer."""
            ),
        ]

        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return responses[call_tracker["get_llm_response"] - 1]

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

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
