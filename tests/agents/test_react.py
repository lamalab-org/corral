"""Comprehensive tests for the ReActAgent class."""

import pytest

from corral.agents.react import Action, ReActAgent, Thought
from corral.agents.utils import LiteLLMMessage
from corral.types import ToolResponse

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
            "list_of_strings": ["Cu2O", "TiO2", "Si"],
            "complex_dict": {
                "materials": [
                    {
                        "material_id": "mp-1234",
                        "energy_above_hull": 0.1,
                        "band_gap": 1.5,
                    },
                    {
                        "material_id": "mp-5678",
                        "energy_above_hull": 0.05,
                        "band_gap": 2.0,
                    },
                ],
                "query_metadata": {
                    "source": "database_search",
                    "date": "2025-09-08",
                },
            },
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

    def test_parse_response_batch_retrieve_polymorphs(self, react_agent):
        """Test parsing response with batch_retrieve_polymorphs action containing large arrays."""
        response = """Thought: I'll use batch_retrieve_polymorphs with common nitride compositions. I'll include binary and ternary nitrides to ensure diversity.
Action: batch_retrieve_polymorphs
Action Input: {"compositions": ["AlN", "GaN", "InN", "TiN", "ZrN", "HfN", "VN", "NbN", "TaN", "CrN", "MoN", "WN", "ScN", "YN", "LaN", "Si3N4", "Ge3N4", "Sn3N4", "Li3N", "Na3N", "K3N", "Be3N2", "Mg3N2", "Ca3N2", "Sr3N2", "Ba3N2", "BN", "GaN", "InN", "TlN", "PN", "AsN", "SbN", "BiN", "ZnN", "CdN", "HgN", "MnN", "FeN", "CoN"], "max_energy_above_hull": 0.3, "max_per_composition": 3, "save_directory": "polymorph_data"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert "nitride compositions" in thought.content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "batch_retrieve_polymorphs"
        assert len(actions[0].arguments["compositions"]) == 40
        assert "AlN" in actions[0].arguments["compositions"]
        assert "CoN" in actions[0].arguments["compositions"]
        assert actions[0].arguments["max_energy_above_hull"] == 0.3
        assert actions[0].arguments["max_per_composition"] == 3
        assert actions[0].arguments["save_directory"] == "polymorph_data"
        assert is_final is False
        assert parsing_error is None

    def test_parse_response_xgboost_evaluation(self, react_agent):
        """Test parsing response with XGBoost model evaluation action."""
        response = """Thought: Let me break this down into steps:
1. First, I need to evaluate the XGBoost model using the test set
2. Then perform cross-validation
3. Finally combine the results and save them in the required JSON format

Let's start by evaluating the model performance on the test set.
Action: evaluate_xgboost_model
Action Input: {"model_path": "/Users/n0w0f/nitride_ml_claude_react/train_xgboost_formation_energy_model_trial_1/trained_xgboost_model.pkl", "test_data_path": "/Users/n0w0f/nitride_ml_claude_react/prepare_ml_ready_dataset_trial_1/nitride_ml_dataset/metadata.json", "target_column": "formation_energy_per_atom", "detailed_analysis": true}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert "break this down into steps" in thought.content
        assert "XGBoost model" in thought.content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "evaluate_xgboost_model"
        assert (
            "/Users/n0w0f/nitride_ml_claude_react/"
            in actions[0].arguments["model_path"]
        )
        assert "trained_xgboost_model.pkl" in actions[0].arguments["model_path"]
        assert "metadata.json" in actions[0].arguments["test_data_path"]
        assert actions[0].arguments["target_column"] == "formation_energy_per_atom"
        assert actions[0].arguments["detailed_analysis"] is True
        assert is_final is False
        assert parsing_error is None

    def test_parse_response_prepare_tabular_dataset(self, react_agent):
        """Test parsing response with prepare_tabular_dataset action."""
        response = """Thought: Let me break down the task and create a plan:
1. First, I need to prepare the ML-ready dataset using the provided input data
2. The requirements specify advanced feature engineering, 0.2 test split, and normalization
3. After preparation, I need to verify the output and provide the path to the metadata json file

Let's start by preparing the dataset using the prepare_tabular_dataset tool with the specified parameters.
Action: prepare_tabular_dataset
Action Input: {"polymorphs_json_path": "/Users/n0w0f/nitride_ml_claude_react/batch_retrieve_nitride_polymorphs_trial_8/nitride_polymorphs_dataset.json", "output_path": "nitride_ml_dataset", "target_property": "formation_energy_per_atom", "feature_engineering": "advanced", "test_split": 0.2, "normalize": true}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert "break down the task" in thought.content
        assert "ML-ready dataset" in thought.content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "prepare_tabular_dataset"
        assert (
            "nitride_polymorphs_dataset.json"
            in actions[0].arguments["polymorphs_json_path"]
        )
        assert actions[0].arguments["output_path"] == "nitride_ml_dataset"
        assert actions[0].arguments["target_property"] == "formation_energy_per_atom"
        assert actions[0].arguments["feature_engineering"] == "advanced"
        assert actions[0].arguments["test_split"] == 0.2
        assert actions[0].arguments["normalize"] is True
        assert is_final is False
        assert parsing_error is None

    def test_parse_response_final_answer_with_file_reference(self, react_agent):
        """Test parsing response with final answer referencing a file."""
        response = """Thought: The task has been completed successfully. I have:
1. Read the bulk structure CIF file
2. Generated possible slabs using the specified parameters (miller_index=[1,1,1], min_slab_size=12, min_vacuum_size=5)
3. Saved the enumerated slabs to slabs.json
Final Answer: slabs.json"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert "task has been completed successfully" in thought.content
        assert "slabs.json" in thought.content
        assert actions is None
        assert is_final is True
        assert parsing_error is None

    def test_parse_response_write_file_with_large_content(self, react_agent):
        """Test parsing response with write_file action containing large CIF content."""
        response = """Thought: Now that I have the combined structure, I need to save it as a CIF file. I'll write the result to a new file called "slab_with_co2.cif".
Action: write_file
Action Input: {"path": "slab_with_co2.cif", "content": "# generated using pymatgen\\ndata_Cu10CO7\\n_symmetry_space_group_name_H-M   'P 1'\\n_cell_length_a   6.00573977\\n_cell_length_b   6.00573977\\n_cell_length_c   33.97359456\\n_cell_angle_alpha   90.00000000\\n_cell_angle_beta   45.00000000\\n_cell_angle_gamma   60.00000000\\n_symmetry_Int_Tables_number   1\\n_chemical_formula_structural   Cu10CO7\\n_chemical_formula_sum   'Cu20 C2 O14'\\n_cell_volume   612.69526542\\n_cell_formula_units_Z   2\\nloop_\\n _symmetry_equiv_pos_site_id\\n _symmetry_equiv_pos_as_xyz\\n  1  'x, y, z'\\nloop_\\n _atom_site_type_symbol\\n _atom_site_label\\n _atom_site_symmetry_multiplicity\\n _atom_site_fract_x\\n _atom_site_fract_y\\n _atom_site_fract_z\\n _atom_site_occupancy\\n  Cu  Cu0  1  0.50000000  0.00000000  0.10937500  1.0\\n  Cu  Cu1  1  0.50000000  0.50000000  0.10937500  1.0\\n  Cu  Cu2  1  0.00000000  0.50000000  0.10937500  1.0\\n  Cu  Cu3  1  0.00000000  0.00000000  0.10937500  1.0\\n  Cu  Cu4  1  0.50000000  0.00000000  0.23437500  1.0\\n  Cu  Cu5  1  0.50000000  0.50000000  0.23437500  1.0\\n  Cu  Cu6  1  0.00000000  0.50000000  0.23437500  1.0\\n  Cu  Cu7  1  0.00000000  0.00000000  0.23437500  1.0\\n  Cu  Cu8  1  0.50000000  0.00000000  0.35937500  1.0\\n  Cu  Cu9  1  0.50000000  0.50000000  0.35937500  1.0\\n  Cu  Cu10  1  0.00000000  0.50000000  0.35937500  1.0\\n  Cu  Cu11  1  0.00000000  0.00000000  0.35937500  1.0\\n  Cu  Cu12  1  0.50000000  0.00000000  0.48437500  1.0\\n  Cu  Cu13  1  0.50000000  0.50000000  0.48437500  1.0\\n  Cu  Cu14  1  0.00000000  0.50000000  0.48437500  1.0\\n  Cu  Cu15  1  0.00000000  0.00000000  0.48437500  1.0\\n  Cu  Cu16  1  0.50000000  0.00000000  0.60937500  1.0\\n  Cu  Cu17  1  0.50000000  0.50000000  0.60937500  1.0\\n  Cu  Cu18  1  0.00000000  0.50000000  0.60937500  1.0\\n  Cu  Cu19  1  0.00000000  0.00000000  0.60937500  1.0\\n  O  O20  1  0.50000000  0.75000000  0.01562500  1.0\\n  O  O21  1  0.50000000  0.25000000  0.07812500  1.0\\n  O  O22  1  0.50000000  0.75000000  0.14062500  1.0\\n  O  O23  1  0.50000000  0.25000000  0.20312500  1.0\\n  O  O24  1  0.50000000  0.75000000  0.26562500  1.0\\n  O  O25  1  0.50000000  0.25000000  0.32812500  1.0\\n  O  O26  1  0.50000000  0.75000000  0.39062500  1.0\\n  O  O27  1  0.50000000  0.25000000  0.45312500  1.0\\n  O  O28  1  0.50000000  0.75000000  0.51562500  1.0\\n  O  O29  1  0.50000000  0.25000000  0.57812500  1.0\\n  C  C30  1  0.48007745  1.87973315  24.15907524  1\\n  C  C31  1  0.96864912  1.15470054  24.03957863  1\\n  O  O32  1  0.26338899  2.03836011  24.15907524  1\\n  O  O33  1  0.69676592  1.72110618  24.15907524  1\\n  O  O34  1  1.02671062  1.31332750  24.03957863  1\\n  O  O35  1  1.20807476  1.80882355  24.03957863  1\\n"}"""

        thought, actions, is_final, parsing_error = react_agent.parse_llm_response(
            response
        )

        assert thought is not None
        assert "combined structure" in thought.content
        assert "slab_with_co2.cif" in thought.content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        assert actions[0].arguments["path"] == "slab_with_co2.cif"
        assert "# generated using pymatgen" in actions[0].arguments["content"]
        assert "data_Cu10CO7" in actions[0].arguments["content"]
        assert "Cu20 C2 O14" in actions[0].arguments["content"]
        assert "_atom_site_type_symbol" in actions[0].arguments["content"]
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
