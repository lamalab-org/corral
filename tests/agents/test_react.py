"""Comprehensive tests for the ReActAgent class."""

import pytest

from corral.agents.react import Action, ReActAgent, Thought
from corral.agents.utils import LiteLLMMessage
from corral.types import ToolResponse

from .conftest import MockLLMResponse, MockPrompt


def create_react_agent():
    """Create a ReActAgent instance for testing."""
    return ReActAgent(
        model="test-model",
        max_iterations=3,
        temperature=0.5,
        system_prompt="You are a helpful assistant.",
        user_prompt="Task: {{task_guide}}",
        extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        surrender_prompt=MockPrompt("You may give up if the task is impossible."),
    )


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
        agent = ReActAgent(
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )

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
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )

        assert agent.model == "custom-model"
        assert agent.max_iterations == 5
        assert agent.temperature == 0.3
        assert agent.api_endpoint == "http://custom-endpoint"

    def test_initialization_with_custom_prompts(self, mock_promptstore_module):
        """Test ReActAgent initialization with custom prompts."""
        system_prompt = "Custom system prompt"
        user_prompt = "Custom user prompt: {{task_guide}}"

        agent = ReActAgent(system_prompt=system_prompt, user_prompt=user_prompt)

        # system_prompt is converted to string via fill({})
        assert agent.system_prompt == system_prompt
        # user_prompt is wrapped in StringPrompt
        assert hasattr(agent.user_prompt, "fill")

    def test_initialization_with_prompt_store(self):
        """Test ReActAgent initialization with PromptStore."""
        # Note: prompt_store is not directly passed to ReActAgent
        # The agent creates its own store internally
        agent = ReActAgent(
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        assert agent.store is not None

    def test_initialization_with_kwargs(self):
        """Test ReActAgent initialization with additional kwargs."""
        agent = ReActAgent(
            model="test-model",
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
            custom_param="custom_value",
            another_param=42,
        )

        assert agent.model == "test-model"
        # Additional kwargs should be passed to parent class


class TestReActAgentParsing:
    """Test cases for ReActAgent response parsing."""

    def test_parse_llm_response_thought_only(self, react_agent):
        """Test parsing response with only a thought."""
        response = (
            "Thought: <thought>I need to analyze this problem carefully.</thought>"
        )

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert thoughts[0].content == "I need to analyze this problem carefully."
        assert actions is None

    def test_parse_llm_response_thought_and_action(self, react_agent):
        """Test parsing response with thought and action."""
        response = """Thought: <thought>I need to search for information.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test query", "limit": 10}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert thoughts[0].content == "I need to search for information."
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test query", "limit": 10}

    def test_parse_llm_response_multiple_actions(self, react_agent):
        """Test parsing response with multiple actions."""
        response = """Thought: <thought>I need to use multiple tools.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>
Action: <action>calculate</action>
Action Input: <action_input>{"expression": "2+2"}</action_input>"""

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
        response = """Thought: <thought>I have found the answer.</thought>
Final Answer: <final_answer>42.</final_answer>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert thoughts[0].content == "I have found the answer."
        assert actions is None

    def test_parse_llm_response_invalid_json(self, react_agent):
        """Test parsing response with invalid JSON in action input."""
        response = """Thought: <thought>Testing invalid JSON.</thought>
Action: <action>search</action>
Action Input: <action_input>{invalid json}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None  # Should be None due to JSON parsing error

    def test_parse_llm_response_no_thought(self, react_agent):
        """Test parsing response with no thought."""
        response = """Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert actions[0].arguments == {"query": "test"}

    def test_parse_llm_response_empty_response(self, react_agent):
        """Test parsing empty response."""
        response = ""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is None
        assert actions is None

    def test_parse_llm_response_multiline_thought(self, react_agent):
        """Test parsing response with multiline thought."""
        response = """Thought: <thought>This is a complex problem that requires
multiple lines of reasoning to solve properly.
Let me break it down step by step.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "complex problem"}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert (
            "This is a complex problem that requires\nmultiple lines of reasoning"
            in thoughts[0].content
        )
        assert actions is not None
        assert len(actions) == 1

    def test_parse_llm_response_empty_thought_variations(self, react_agent):
        """Test parsing responses with various empty thought patterns."""
        test_cases = [
            ("Thought:", "Empty thought with colon only"),
            ("Thought: ", "Empty thought with space"),
            ("Thought:\n", "Empty thought with newline"),
            ("Thought: \n", "Empty thought with space and newline"),
            (
                "Thought:   \nAction: <action>test</action>\nAction Input: <action_input>{}</action_input>",
                "Whitespace-only thought with action",
            ),
        ]

        for response, description in test_cases:
            thought, actions = react_agent.parse_llm_response(response)
            # All empty thought variations should return None for thought
            assert thought is None, f"Failed for case: {description}"

            # Check if actions are parsed correctly when present
            if "Action:" in response:
                assert actions is not None, f"Failed for case: {description}"
                assert len(actions) == 1, f"Failed for case: {description}"
            else:
                assert actions is None, f"Failed for case: {description}"

    def test_parse_llm_response_final_answer_only(self, react_agent):
        """Test parsing response with only final answer."""
        response = (
            "Final Answer: <final_answer>Direct answer without thought</final_answer>"
        )

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is None
        assert actions is None

    def test_parse_llm_response_all_components(self, react_agent):
        """Test parsing response with thought, actions, and final answer."""
        response = """Thought: <thought>I need to search and then provide an answer.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>
Action: <action>analyze</action>
Action Input: <action_input>{"data": "results"}</action_input>
Final Answer: <final_answer>Based on my analysis, the answer is 42.</final_answer>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert thoughts[0].content == "I need to search and then provide an answer."
        assert actions is not None
        assert len(actions) == 2
        assert actions[0].tool_name == "search"
        assert actions[1].tool_name == "analyze"

    def test_parse_llm_response_unescaped_triple_quotes_simple(self, react_agent):
        """Test parsing response with simple unescaped triple-quoted string."""
        response = """<thought>Testing simple case</thought>
<action>write_file</action>
<action_input>{"path": "test.py", "content": \"\"\"print('hello')\"\"\"}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        assert actions[0].arguments["path"] == "test.py"
        assert actions[0].arguments["content"] == "print('hello')"

    def test_parse_llm_response_escaped_triple_quotes(self, react_agent):
        """Test parsing response with properly escaped triple quotes in JSON."""
        response = """<thought>Testing escaped case</thought>
<action>write_file</action>
<action_input>{"path": "test.py", "content": "x = \\"\\"\\"\\nMultiline\\nString\\n\\"\\"\\""}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        assert actions[0].arguments["path"] == "test.py"
        # Should preserve the triple quotes as actual string content
        assert '"""' in actions[0].arguments["content"]
        assert "Multiline" in actions[0].arguments["content"]

    def test_parse_llm_response_unescaped_multiline_code(self, react_agent):
        """Test parsing response with multi-line unescaped triple-quoted code."""
        response = """<thought>Need to write complex script</thought>
<action>write_file</action>
<action_input>{
  "path": "script.py",
  "content": \"\"\"import numpy as np
import pandas as pd

def analyze_data():
    data = pd.DataFrame({'x': [1, 2, 3]})
    print(f"Results: {data.mean()}")

analyze_data()\"\"\"
}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        assert actions[0].arguments["path"] == "script.py"
        content = actions[0].arguments["content"]
        # Verify multi-line content is preserved
        assert "import numpy" in content
        assert "import pandas" in content
        assert "def analyze_data" in content
        assert content.count("\n") >= 7  # Multiple lines

    def test_parse_llm_response_standard_json_with_newlines(self, react_agent):
        """Test parsing response with standard JSON using escaped newlines."""
        response = """<thought>Standard JSON format</thought>
<action>write_file</action>
<action_input>{"path": "test.py", "content": "line1\\nline2\\nline3"}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        content = actions[0].arguments["content"]
        assert content == "line1\nline2\nline3"
        assert content.count("\n") == 2

    def test_parse_llm_response_boolean_conversion(self, react_agent):
        """Test parsing response with Python boolean values."""
        response = """<thought>Testing boolean conversion</thought>
<action>configure</action>
<action_input>{"enabled": True, "debug": False, "count": 42}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "configure"
        assert actions[0].arguments["enabled"] is True
        assert actions[0].arguments["debug"] is False
        assert actions[0].arguments["count"] == 42


class TestReActAgentRun:
    """Test cases for ReActAgent.run method."""

    def test_run_with_final_answer(self, react_agent, mock_interface, monkeypatch):
        """Test run method that returns final answer immediately."""
        # Mock the LLM response
        mock_response = MockLLMResponse(
            content="Thought: <thought>I can answer this directly.</thought>\nFinal Answer: <final_answer>42</final_answer>"
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
            content="""Thought: <thought>I need to search for information.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>"""
        )

        response2 = MockLLMResponse(
            content=(
                "Thought: <thought>Based on the search results.</thought>\nFinal Answer: <final_answer>Found it!</final_answer>"
            )
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
                """Thought: <thought>I need to use a tool.</thought>
Action: <action>failing_tool</action>
Action Input: <action_input>{"param": "value"}</action_input>""",
                None,  # No final result yet, should continue iterating
                1,  # Should have made 1 tool call
                "First iteration: tool action with error",
            ),
            (
                "Thought: <thought>The tool failed.</thought>\nFinal Answer: <final_answer>Handled error</final_answer>",
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
            and "<action>failing_tool</action>" in msg.get("content", "")
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
            and "<final_answer>Handled error</final_answer>" in msg.get("content", "")
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
                content="""Thought: <thought>I'm thinking about this problem.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>"""
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
            content="""Thought: <thought>I need to use multiple tools.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>
Action: <action>calculate</action>
Action Input: <action_input>{"expression": "2+2"}</action_input>"""
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
                content=(
                    "Thought: <thought>Using history and examples.</thought>\nFinal Answer: <final_answer>Success</final_answer>"
                )
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
            content="""Thought: <thought>I need to search.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test"}</action_input>"""
        )

        response2 = MockLLMResponse(
            content="Thought: <thought>Found it.</thought>\nFinal Answer: <final_answer>Result</final_answer>"
        )

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
            len(react_agent.messages) >= 2
        )  # Initial + assistant action + user observation + final answer

        # Check message structure
        messages = react_agent.messages
        assert any(
            msg.get("role") == "assistant"
            and "<action>search</action>" in msg.get("content", "")
            for msg in messages
        )
        assert any(
            msg.get("role") == "user"
            and "Observation: Search result" in msg.get("content", "")
            for msg in messages
        )
        assert any(
            msg.get("role") == "assistant"
            and "<final_answer>Result</final_answer>" in msg.get("content", "")
            for msg in messages
        )


class TestReActAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_response_with_malformed_action(self, react_agent):
        """Test parsing response with malformed action structure."""
        response = """Thought: <thought>Testing malformed action.</thought>
Action: <action>search</action>
Action Input: <action_input>not valid json at all</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None

    def test_parse_llm_response_parsing_error_feedback(self, react_agent):
        """Test that parsing errors provide detailed feedback."""
        response = """Thought: <thought>I'll try using a tool with malformed JSON.</thought>
Action: <action>test_tool</action>
Action Input: <action_input>{malformed: "json", missing_quotes: value}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert thoughts[0].content == "I'll try using a tool with malformed JSON."
        assert actions is None  # Should be None due to JSON parsing error

    def test_parse_llm_response_multiple_parsing_errors(self, react_agent):
        """Test that only the first parsing error is captured."""
        response = """Thought: <thought>Testing multiple malformed actions.</thought>
Action: <action>first_tool</action>
Action Input: <action_input>{invalid: json}</action_input>
Action: <action>second_tool</action>
Action Input: <action_input>{also invalid}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None  # Should be None due to JSON parsing errors

    def test_parse_response_with_nested_final_answer(self, react_agent):
        """Test parsing response with nested final answer pattern."""
        response = """Thought: <thought>The answer mentions "Final Answer: not really" in the text.</thought>
Final Answer: <final_answer>The actual final answer is 42.</final_answer>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is None

    def test_parse_response_with_special_characters(self, react_agent):
        """Test parsing response with special characters in JSON."""
        response = """Thought: <thought>Testing special characters.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "test with \\"quotes\\" and \\n newlines", "special": "chars: !@#$%^&*()"}</action_input>"""

        thought, actions = react_agent.parse_llm_response(response)

        assert thought is not None
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "search"
        assert "quotes" in actions[0].arguments["query"]
        assert actions[0].arguments["special"] == "chars: !@#$%^&*()"

    def test_parse_response_batch_retrieve_polymorphs(self, react_agent):
        """Test parsing response with batch_retrieve_polymorphs action containing large arrays."""
        response = """Thought: <thought>I'll use batch_retrieve_polymorphs with common nitride compositions. I'll include binary and ternary nitrides to ensure diversity.</thought>
Action: <action>batch_retrieve_polymorphs</action>
Action Input: <action_input>{"compositions": ["AlN", "GaN", "InN", "TiN", "ZrN", "HfN", "VN", "NbN", "TaN", "CrN", "MoN", "WN", "ScN", "YN", "LaN", "Si3N4", "Ge3N4", "Sn3N4", "Li3N", "Na3N", "K3N", "Be3N2", "Mg3N2", "Ca3N2", "Sr3N2", "Ba3N2", "BN", "GaN", "InN", "TlN", "PN", "AsN", "SbN", "BiN", "ZnN", "CdN", "HgN", "MnN", "FeN", "CoN"], "max_energy_above_hull": 0.3, "max_per_composition": 3, "save_directory": "polymorph_data"}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert "nitride compositions" in thoughts[0].content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "batch_retrieve_polymorphs"
        assert len(actions[0].arguments["compositions"]) == 40
        assert "AlN" in actions[0].arguments["compositions"]
        assert "CoN" in actions[0].arguments["compositions"]
        assert actions[0].arguments["max_energy_above_hull"] == 0.3
        assert actions[0].arguments["max_per_composition"] == 3
        assert actions[0].arguments["save_directory"] == "polymorph_data"

    def test_parse_response_xgboost_evaluation(self, react_agent):
        """Test parsing response with XGBoost model evaluation action."""
        response = """Thought: <thought>Let me break this down into steps:
1. First, I need to evaluate the XGBoost model using the test set
2. Then perform cross-validation
3. Finally combine the results and save them in the required JSON format

Let's start by evaluating the model performance on the test set.
</thought>
Action: <action>evaluate_xgboost_model</action>
Action Input: <action_input>{"model_path": "/Users/n0w0f/nitride_ml_claude_react/train_xgboost_formation_energy_model_trial_1/trained_xgboost_model.pkl", "test_data_path": "/Users/n0w0f/nitride_ml_claude_react/prepare_ml_ready_dataset_trial_1/nitride_ml_dataset/metadata.json", "target_column": "formation_energy_per_atom", "detailed_analysis": true}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert "break this down into steps" in thoughts[0].content
        assert "XGBoost model" in thoughts[0].content
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

    def test_parse_response_prepare_tabular_dataset(self, react_agent):
        """Test parsing response with prepare_tabular_dataset action."""
        response = """Thought: <thought>Let me break down the task and create a plan:
1. First, I need to prepare the ML-ready dataset using the provided input data
2. The requirements specify advanced feature engineering, 0.2 test split, and normalization
3. After preparation, I need to verify the output and provide the path to the metadata json file

Let's start by preparing the dataset using the prepare_tabular_dataset tool with the specified parameters.
</thought>
Action: <action>prepare_tabular_dataset</action>
Action Input: <action_input>{"polymorphs_json_path": "/Users/n0w0f/nitride_ml_claude_react/batch_retrieve_nitride_polymorphs_trial_8/nitride_polymorphs_dataset.json", "output_path": "nitride_ml_dataset", "target_property": "formation_energy_per_atom", "feature_engineering": "advanced", "test_split": 0.2, "normalize": true}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert "break down the task" in thoughts[0].content
        assert "ML-ready dataset" in thoughts[0].content
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

    def test_parse_response_final_answer_with_file_reference(self, react_agent):
        """Test parsing response with final answer referencing a file."""
        response = """Thought: <thought>The task has been completed successfully. I have:
1. Read the bulk structure CIF file
2. Generated possible slabs using the specified parameters (miller_index=[1,1,1], min_slab_size=12, min_vacuum_size=5)
3. Saved the enumerated slabs to slabs.json</thought>
Final Answer: <final_answer>slabs.json</final_answer>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert "task has been completed successfully" in thoughts[0].content
        assert "slabs.json" in thoughts[0].content
        assert actions is None

    def test_parse_response_write_file_with_large_content(self, react_agent):
        """Test parsing response with write_file action containing large CIF content."""
        response = """Thought: <thought>Now that I have the combined structure, I need to save it as a CIF file. I'll write the result to a new file called "slab_with_co2.cif"</thought>.
Action: <action>write_file</action>
Action Input: <action_input>{"path": "slab_with_co2.cif", "content": "# generated using pymatgen\\ndata_Cu10CO7\\n_symmetry_space_group_name_H-M   'P 1'\\n_cell_length_a   6.00573977\\n_cell_length_b   6.00573977\\n_cell_length_c   33.97359456\\n_cell_angle_alpha   90.00000000\\n_cell_angle_beta   45.00000000\\n_cell_angle_gamma   60.00000000\\n_symmetry_Int_Tables_number   1\\n_chemical_formula_structural   Cu10CO7\\n_chemical_formula_sum   'Cu20 C2 O14'\\n_cell_volume   612.69526542\\n_cell_formula_units_Z   2\\nloop_\\n _symmetry_equiv_pos_site_id\\n _symmetry_equiv_pos_as_xyz\\n  1  'x, y, z'\\nloop_\\n _atom_site_type_symbol\\n _atom_site_label\\n _atom_site_symmetry_multiplicity\\n _atom_site_fract_x\\n _atom_site_fract_y\\n _atom_site_fract_z\\n _atom_site_occupancy\\n  Cu  Cu0  1  0.50000000  0.00000000  0.10937500  1.0\\n  Cu  Cu1  1  0.50000000  0.50000000  0.10937500  1.0\\n  Cu  Cu2  1  0.00000000  0.50000000  0.10937500  1.0\\n  Cu  Cu3  1  0.00000000  0.00000000  0.10937500  1.0\\n  Cu  Cu4  1  0.50000000  0.00000000  0.23437500  1.0\\n  Cu  Cu5  1  0.50000000  0.50000000  0.23437500  1.0\\n  Cu  Cu6  1  0.00000000  0.50000000  0.23437500  1.0\\n  Cu  Cu7  1  0.00000000  0.00000000  0.23437500  1.0\\n  Cu  Cu8  1  0.50000000  0.00000000  0.35937500  1.0\\n  Cu  Cu9  1  0.50000000  0.50000000  0.35937500  1.0\\n  Cu  Cu10  1  0.00000000  0.50000000  0.35937500  1.0\\n  Cu  Cu11  1  0.00000000  0.00000000  0.35937500  1.0\\n  Cu  Cu12  1  0.50000000  0.00000000  0.48437500  1.0\\n  Cu  Cu13  1  0.50000000  0.50000000  0.48437500  1.0\\n  Cu  Cu14  1  0.00000000  0.50000000  0.48437500  1.0\\n  Cu  Cu15  1  0.00000000  0.00000000  0.48437500  1.0\\n  Cu  Cu16  1  0.50000000  0.00000000  0.60937500  1.0\\n  Cu  Cu17  1  0.50000000  0.50000000  0.60937500  1.0\\n  Cu  Cu18  1  0.00000000  0.50000000  0.60937500  1.0\\n  Cu  Cu19  1  0.00000000  0.00000000  0.60937500  1.0\\n  O  O20  1  0.50000000  0.75000000  0.01562500  1.0\\n  O  O21  1  0.50000000  0.25000000  0.07812500  1.0\\n  O  O22  1  0.50000000  0.75000000  0.14062500  1.0\\n  O  O23  1  0.50000000  0.25000000  0.20312500  1.0\\n  O  O24  1  0.50000000  0.75000000  0.26562500  1.0\\n  O  O25  1  0.50000000  0.25000000  0.32812500  1.0\\n  O  O26  1  0.50000000  0.75000000  0.39062500  1.0\\n  O  O27  1  0.50000000  0.25000000  0.45312500  1.0\\n  O  O28  1  0.50000000  0.75000000  0.51562500  1.0\\n  O  O29  1  0.50000000  0.25000000  0.57812500  1.0\\n  C  C30  1  0.48007745  1.87973315  24.15907524  1\\n  C  C31  1  0.96864912  1.15470054  24.03957863  1\\n  O  O32  1  0.26338899  2.03836011  24.15907524  1\\n  O  O33  1  0.69676592  1.72110618  24.15907524  1\\n  O  O34  1  1.02671062  1.31332750  24.03957863  1\\n  O  O35  1  1.20807476  1.80882355  24.03957863  1\\n"}</action_input>"""

        thoughts, actions = react_agent.parse_llm_response(response)

        assert thoughts is not None
        assert len(thoughts) == 1
        assert "combined structure" in thoughts[0].content
        assert "slab_with_co2.cif" in thoughts[0].content
        assert actions is not None
        assert len(actions) == 1
        assert actions[0].tool_name == "write_file"
        assert actions[0].arguments["path"] == "slab_with_co2.cif"
        assert "# generated using pymatgen" in actions[0].arguments["content"]
        assert "data_Cu10CO7" in actions[0].arguments["content"]
        assert "Cu20 C2 O14" in actions[0].arguments["content"]
        assert "_atom_site_type_symbol" in actions[0].arguments["content"]

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

    def test_initialization_with_none_values(self, mock_promptstore_module):
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
        agent = ReActAgent(
            model="test-model",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            user_prompt="Task: {{task_guide}}",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
            surrender_prompt=MockPrompt("You may give up if the task is impossible."),
        )

        # Mock realistic LLM responses
        responses = [
            # First iteration - analyze problem
            MockLLMResponse(
                content="""Thought: <thought>I need to understand the problem first.</thought>
Action: <action>analyze</action>
Action Input: <action_input>{"text": "problem statement"}</action_input>"""
            ),
            # Second iteration - search for information
            MockLLMResponse(
                content="""Thought: <thought>Now I need to search for relevant information.</thought>
Action: <action>search</action>
Action Input: <action_input>{"query": "relevant information", "limit": 5}</action_input>"""
            ),
            # Third iteration - process results and provide answer
            MockLLMResponse(
                content="""Thought: <thought>Based on the analysis and search results, I can now provide the answer.</thought>
Final Answer: <final_answer>The solution is X because of Y and Z.</final_answer>"""
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
        agent = ReActAgent(
            model="test-model",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            user_prompt="Task: {{task_guide}}",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
            surrender_prompt=MockPrompt("You may give up if the task is impossible."),
        )

        # Mock responses with error recovery
        responses = [
            # First iteration - try a tool that fails
            MockLLMResponse(
                content="""Thought: <thought>I'll try using this tool.</thought>
Action: <action>failing_tool</action>
Action Input: <action_input>{"param": "value"}</action_input>"""
            ),
            # Second iteration - recover from error
            MockLLMResponse(
                content="""Thought: <thought>The tool failed, let me try a different approach.</thought>
Action: <action>backup_tool</action>
Action Input: <action_input>{"alternative": "approach"}</action_input>"""
            ),
            # Third iteration - provide answer
            MockLLMResponse(
                content="""Thought: <thought>This approach worked.</thought>
Final Answer: <final_answer>Successfully recovered and found the answer.</final_answer>"""
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

    def test_actions_before_final_answer(self, mock_interface, monkeypatch):
        """Test that actions are executed before final answer is returned in same message."""
        agent = ReActAgent(
            model="test-model",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            user_prompt="Task: {{task_guide}}",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
            surrender_prompt=MockPrompt("You may give up if the task is impossible."),
        )

        # Single response containing multiple thoughts, actions with code, and final answer
        # Note: LLMs typically output triple quotes directly without escaping, including nested docstrings
        # Using triple single quotes for the outer string so we can have unescaped triple double quotes inside
        response = MockLLMResponse(
            content='''Thought: <thought>I need to search for data and then analyze it before providing the answer.</thought>
Action: <action>write_script</action>
Action Input: <action_input>{
  "filename": "search.py",
  "content": """import pandas as pd
import numpy as np

def search_data(query):
    """Search for data based on query.

    Args:
        query: Search query string

    Returns:
        DataFrame with search results
    """
    results = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    filtered = results.query(query)
    return filtered

if __name__ == '__main__':
    data = search_data('id > 1')
    print(data)
    print(f'Found {len(data)} results')"""
}</action_input>
Thought: <thought>Now I'll create an analysis script to process the search results.</thought>
Action: <action>write_script</action>
Action Input: <action_input>{
  "filename": "analyze.py",
  "content": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_results(data):
    """Analyze the search results.

    Args:
        data: Input DataFrame

    Returns:
        Analysis summary dictionary
    """
    summary = {
        'mean': data['value'].mean(),
        'std': data['value'].std(),
        'count': len(data),
        'max': data['value'].max(),
        'min': data['value'].min()
    }

    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='id', y='value', data=data)
    plt.title('Search Results Analysis')
    plt.xlabel('ID')
    plt.ylabel('Value')
    plt.savefig('results.png')
    plt.close()

    return summary

if __name__ == '__main__':
    # Example usage
    test_data = pd.DataFrame({'id': [1, 2], 'value': [20, 30]})
    result = analyze_results(test_data)
    print(result)"""
}</action_input>
Final Answer: <final_answer>Based on the search and analysis, the answer is 42.</final_answer>'''
        )

        call_tracker = {"get_llm_response": 0, "create_prompt": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_tracker["get_llm_response"] += 1
            return response

        def mock_create_prompt(*args, **kwargs):
            call_tracker["create_prompt"] += 1
            return []

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr("corral.agents.react.create_prompt", mock_create_prompt)

        # Set up tool responses for both actions
        mock_interface.tool_responses = [
            ToolResponse(
                success=True, result="Script created successfully", error=None
            ),
            ToolResponse(success=True, result="Analysis script created", error=None),
        ]

        result = agent.run(mock_interface, "test_task")

        # Verify final answer is returned
        assert result == "Based on the search and analysis, the answer is 42."

        # Verify that both thoughts were captured in the parsing
        thoughts, actions = agent.parse_llm_response(response.content)
        assert thoughts is not None
        assert len(thoughts) == 2
        assert (
            thoughts[0].content
            == "I need to search for data and then analyze it before providing the answer."
        )
        assert (
            thoughts[1].content
            == "Now I'll create an analysis script to process the search results."
        )

        # Verify that both actions were executed before returning
        assert len(mock_interface.tool_calls) == 2
        assert mock_interface.tool_calls[0]["tool_name"] == "write_script"
        assert mock_interface.tool_calls[0]["arguments"]["filename"] == "search.py"
        # Verify the content contains multi-line code with docstrings (triple quotes)
        search_content = mock_interface.tool_calls[0]["arguments"]["content"]
        assert "import pandas as pd" in search_content
        assert "import numpy as np" in search_content
        assert "Search for data based on query" in search_content
        assert "def search_data(query):" in search_content
        assert "if __name__ == '__main__':" in search_content
        assert "print(f'Found {len(data)} results')" in search_content
        # Verify it's multi-line
        assert search_content.count("\n") >= 10

        assert mock_interface.tool_calls[1]["tool_name"] == "write_script"
        assert mock_interface.tool_calls[1]["arguments"]["filename"] == "analyze.py"
        # Verify the content contains multi-line code with docstrings (triple quotes)
        analyze_content = mock_interface.tool_calls[1]["arguments"]["content"]
        assert "import matplotlib.pyplot as plt" in analyze_content
        assert "import seaborn as sns" in analyze_content
        assert "Analyze the search results" in analyze_content
        assert "def analyze_results(data):" in analyze_content
        assert "plt.savefig('results.png')" in analyze_content
        assert "'max': data['value'].max()" in analyze_content
        # Verify it's multi-line
        assert analyze_content.count("\n") >= 25

        # Verify only one LLM call was made (single iteration)
        assert call_tracker["get_llm_response"] == 1

        # Verify message history contains the action executions and observations
        messages = agent.messages

        # Should contain the assistant's response with actions and final answer
        assert any(
            msg.get("role") == "assistant"
            and "<action>write_script</action>" in msg.get("content", "")
            and "<final_answer>" in msg.get("content", "")
            for msg in messages
        )

        # Should contain observations from both tool executions
        first_observation_found = any(
            msg.get("role") == "user"
            and "Observation: Script created successfully" in msg.get("content", "")
            for msg in messages
        )
        assert (
            first_observation_found
        ), "First script creation observation should be in message history"

        second_observation_found = any(
            msg.get("role") == "user"
            and "Observation: Analysis script created" in msg.get("content", "")
            for msg in messages
        )
        assert (
            second_observation_found
        ), "Second script creation observation should be in message history"
