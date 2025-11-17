"""Integration tests for ReflexionAgent."""

import pytest

from corral.agents.react import ReActAgent
from corral.agents.reflection import ReflectionModule
from corral.agents.reflexion_agent import ReflexionAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.agents.utils import LiteLLMMessage
from corral.types import ToolResponse

from .conftest import MockLLMResponse


class TestReflexionAgentInitialization:
    """Test cases for ReflexionAgent initialization."""

    def test_init_with_react_agent(self, mock_promptstore_module):
        """Test initializing ReflexionAgent with ReActAgent."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert reflexion_agent.actor == base_agent
        assert reflexion_agent.model == base_agent.model
        assert reflexion_agent.max_iterations == base_agent.max_iterations

    def test_init_with_tool_calling_agent(self, mock_promptstore_module):
        """Test initializing ReflexionAgent with ToolCallingAgent."""
        base_agent = ToolCallingAgent(
            model="test-model",
            max_iterations=10,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert reflexion_agent.actor == base_agent

    def test_init_custom_reflection_model(self, mock_promptstore_module):
        """Test initializing with custom reflection model."""
        base_agent = ReActAgent(
            model="gpt-4",
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(actor=base_agent, reflection_model="gpt-4o")

        assert reflexion_agent.reflection_module.model == "gpt-4o"

    def test_memory_initialization(self, mock_promptstore_module):
        """Test that memory is properly initialized."""
        base_agent = ReActAgent(
            model="test-model",
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert len(reflexion_agent.memory.reflections) == 0
        assert reflexion_agent.memory.max_size == 5


class TestReflexionAgentRun:
    """Test cases for ReflexionAgent.run method."""

    def test_success_on_first_attempt(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that reflexion agent succeeds on first attempt."""
        # Create base agent
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # Mock successful response
        response = MockLLMResponse(
            content="Thought: <thought>I can answer this.</thought>\nFinal Answer: <final_answer>42</final_answer>"
        )

        call_count = {"llm": 0, "reflection": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_count["llm"] += 1
            return response

        def mock_reflection_generate(*args, **kwargs):
            call_count["reflection"] += 1
            return "This should not be called"

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Run agent
        result = reflexion_agent.run(mock_interface, "test_task")

        # Should succeed on first attempt
        assert result == "42"
        assert call_count["llm"] == 1
        assert call_count["reflection"] == 0  # No reflection needed
        assert len(reflexion_agent.memory.reflections) == 0

    def test_retry_with_reflection(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that agent retries with reflection after failure."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First trial fails (needs 3 responses to exhaust iterations), second trial succeeds
        responses = [
            # First trial - iteration 1
            MockLLMResponse(
                content="Thought: <thought>Let me try something.</thought>\nAction: <action>tool1</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # First trial - iteration 2
            MockLLMResponse(
                content="Thought: <thought>That didn't work.</thought>\nAction: <action>tool2</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # First trial - iteration 3
            MockLLMResponse(
                content="Thought: <thought>Still not working.</thought>\nAction: <action>tool3</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # Second trial - succeeds immediately
            MockLLMResponse(
                content="Thought: <thought>Using the reflection, I now understand.</thought>\nFinal Answer: <final_answer>Success</final_answer>"
            ),
        ]

        call_count = {"llm": 0, "reflection": 0}

        def mock_get_llm_response(*args, **kwargs):
            response = responses[call_count["llm"]]
            call_count["llm"] += 1
            return response

        def mock_reflection_generate(*args, **kwargs):
            call_count["reflection"] += 1
            return "I should try a different approach next time.", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Set up tool responses for the failed tools
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="result1", error=None),
            ToolResponse(success=True, result="result2", error=None),
            ToolResponse(success=True, result="result3", error=None),
        ]

        # Run first trial (should fail by exhausting iterations)
        result1 = reflexion_agent.run(mock_interface, "test_task")

        # Verify first trial failed
        assert "Error" in result1
        assert call_count["llm"] == 3  # 3 iterations exhausted

        # Mock get_last_score to simulate framework providing previous score
        def mock_get_last_score(task_id):
            return {"score": 0.0, "trial_id": "trial_1"}

        monkeypatch.setattr(mock_interface, "get_last_score", mock_get_last_score)

        # Run second trial (should succeed with reflection)
        result2 = reflexion_agent.run(mock_interface, "test_task")

        # Should succeed on second trial
        assert result2 == "Success"
        assert call_count["llm"] == 4  # 3 for first trial + 1 for second
        assert call_count["reflection"] == 1
        assert len(reflexion_agent.memory.reflections) == 1

    def test_exhausts_all_attempts(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that agent can be called multiple times and generates reflections."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # All trials fail - each needs 3 responses to exhaust iterations
        call_count = {"llm": 0, "reflection": 0, "trial": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_count["llm"] += 1
            # Each iteration returns a failed action
            return MockLLMResponse(
                content=f"Thought: <thought>Trial {call_count['trial']}, iteration {call_count['llm']}.</thought>\nAction: <action>failing_tool</action>\nAction Input: <action_input>{{}}</action_input>"
            )

        def mock_reflection_generate(*args, **kwargs):
            call_count["reflection"] += 1
            return f"Reflection {call_count['reflection']}", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        def mock_get_last_score(task_id):
            # Return previous score if not first trial
            if call_count["trial"] > 0:
                return {"score": 0.0, "trial_id": f"trial_{call_count['trial']}"}
            raise AttributeError("No previous score")

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )
        monkeypatch.setattr(mock_interface, "get_last_score", mock_get_last_score)

        # Set up tool responses that always succeed (but agent never finds final answer)
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="result", error=None)
        ] * 20  # More than enough for all trials

        # Run agent 3 times (3 trials)
        for i in range(3):
            call_count["trial"] = i
            result = reflexion_agent.run(mock_interface, "test_task")
            assert "Error" in result  # Each trial fails

        # Should have tried all 3 trials
        assert call_count["llm"] == 9  # 3 trials x 3 iterations each
        assert (
            call_count["reflection"] == 2
        )  # Generate reflection after first 2 failures
        assert len(reflexion_agent.memory.reflections) == 2

    def test_memory_injection_into_history(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that reflections are injected into actor's history."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First trial fails, second trial succeeds
        responses = [
            MockLLMResponse(content="Error solving the task: failed"),
            MockLLMResponse(
                content="Thought: <thought>Got it.</thought>\nFinal Answer: <final_answer>Success</final_answer>"
            ),
        ]

        captured_history = []

        def mock_actor_run(self, interface, task_id, history=None, **kwargs):
            # Capture the history passed to actor
            captured_history.append(history)

            # Return responses
            if len(captured_history) == 1:
                self.messages = [
                    LiteLLMMessage(role="assistant", content=responses[0].content)
                ]
                return responses[0].content
            else:
                self.messages = [
                    LiteLLMMessage(role="assistant", content=responses[1].content)
                ]
                return "Success"

        def mock_reflection_generate(*args, **kwargs):
            return "Important lesson learned", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        def mock_get_last_score(task_id):
            # Only return score after first trial
            if len(captured_history) > 0:
                return {"score": 0.0, "trial_id": "trial_1"}
            raise AttributeError("No previous score")

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )
        monkeypatch.setattr(mock_interface, "get_last_score", mock_get_last_score)

        # Run first trial
        reflexion_agent.run(mock_interface, "test_task")

        # Run second trial (with reflection)
        reflexion_agent.run(mock_interface, "test_task")

        # Check that history was injected on second trial
        assert len(captured_history) == 2
        assert (
            captured_history[0] is None or len(captured_history[0]) == 0
        )  # First trial: no history
        assert captured_history[1] is not None  # Second trial: history with reflection
        assert len(captured_history[1]) > 0
        assert captured_history[1][0]["role"] == "system"
        assert "LESSONS FROM PREVIOUS ATTEMPTS" in captured_history[1][0]["content"]

    def test_memory_clears_for_new_task(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that memory accumulates reflections across tasks."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        trial_count = {"count": 0}

        def mock_actor_run(self, interface, task_id, history=None, **kwargs):
            self.messages = [LiteLLMMessage(role="assistant", content="Error")]
            return "Error solving the task"

        def mock_reflection_generate(*args, **kwargs):
            return f"Reflection for {kwargs.get('task_id', 'unknown')}", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        def mock_get_last_score(task_id):
            # Only return score if not first trial for a task
            if trial_count["count"] > 0:
                return {"score": 0.0, "trial_id": f"trial_{trial_count['count']}"}
            raise AttributeError("No previous score")

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )
        monkeypatch.setattr(mock_interface, "get_last_score", mock_get_last_score)

        # Run first trial for task_1
        trial_count["count"] = 0
        reflexion_agent.run(mock_interface, "task_1")

        # Run second trial for task_1 (should generate reflection)
        trial_count["count"] = 1
        reflexion_agent.run(mock_interface, "task_1")
        assert len(reflexion_agent.memory.reflections) == 1

        # Run first trial for task_2 (should generate another reflection from task_1's second trial)
        trial_count["count"] = 2
        reflexion_agent.run(mock_interface, "task_2")

        # Memory continues to accumulate (FIFO with max_size=3)
        assert len(reflexion_agent.memory.reflections) >= 1

    def test_with_tool_execution(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test reflexion agent with tool execution across trials."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=5,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First trial uses wrong tool and exhausts iterations, second trial uses right tool and succeeds
        responses = [
            # First trial - iteration 1
            MockLLMResponse(
                content="""Thought: <thought>I'll use tool A.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First trial - iteration 2
            MockLLMResponse(
                content="""Thought: <thought>Let me try again.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First trial - iteration 3
            MockLLMResponse(
                content="""Thought: <thought>Still trying.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First trial - iteration 4
            MockLLMResponse(
                content="""Thought: <thought>One more time.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First trial - iteration 5
            MockLLMResponse(
                content="""Thought: <thought>Last try.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # Second trial - iteration 1 (with reflection)
            MockLLMResponse(
                content="""Thought: <thought>Based on reflection, I'll use tool B.</thought>
Action: <action>correct_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # Second trial - iteration 2
            MockLLMResponse(
                content="Thought: <thought>Got result.</thought>\nFinal Answer: <final_answer>Success</final_answer>"
            ),
        ]

        call_count = {"llm": 0}

        def mock_get_llm_response(*args, **kwargs):
            response = responses[call_count["llm"]]
            call_count["llm"] += 1
            return response

        def mock_reflection_generate(*args, **kwargs):
            return "Use correct_tool instead of wrong_tool", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        def mock_get_last_score(task_id):
            # Return score after first trial
            if call_count["llm"] >= 5:
                return {"score": 0.0, "trial_id": "trial_1"}
            raise AttributeError("No previous score")

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )
        monkeypatch.setattr(mock_interface, "get_last_score", mock_get_last_score)

        # Set up tool responses
        mock_interface.tool_responses = [
            ToolResponse(
                success=False, result=None, error="Wrong tool"
            ),  # First trial calls
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(
                success=True, result="Correct result", error=None
            ),  # Second trial call
        ]

        # Run first trial (should fail by exhausting iterations)
        result1 = reflexion_agent.run(mock_interface, "test_task")
        assert "Error" in result1

        # Run second trial (should succeed with reflection)
        result2 = reflexion_agent.run(mock_interface, "test_task")
        assert result2 == "Success"
        assert len(reflexion_agent.memory.reflections) == 1


class TestReflexionAgentEdgeCases:
    """Test edge cases for ReflexionAgent."""

    def test_exception_handling(
        self, mock_interface, monkeypatch, mock_promptstore_module
    ):
        """Test that exceptions from the actor are propagated."""
        base_agent = ReActAgent(
            model="test-model",
            max_iterations=3,
            system_prompt="You are a helpful assistant.",
            extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        )
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        call_count = {"attempt": 0}

        def mock_actor_run(self, interface, task_id, history=None, **kwargs):
            call_count["attempt"] += 1
            if call_count["attempt"] == 1:
                raise ValueError("Simulated error")

            self.messages = [LiteLLMMessage(role="assistant", content="Success")]
            return "Success"

        def mock_reflection_generate(*args, **kwargs):
            return "Handle the error better", {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Run - should propagate exception from actor
        with pytest.raises(ValueError, match="Simulated error"):
            reflexion_agent.run(mock_interface, "test_task")


class TestReflectionModuleToolFormatting:
    """Test cases for tool output detection and formatting in ReflectionModule."""

    def test_is_tool_output_message_with_tool_role(self):
        """Test detection of messages with role='tool'."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        msg = LiteLLMMessage(role="tool", content="some content", tool_call_id="123")

        assert module._is_tool_output_message(msg, "some content") is True

    def test_is_tool_output_message_with_observation(self):
        """Test detection of ReAct-style Observation messages."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        msg = LiteLLMMessage(role="user", content="Observation: tool result")

        assert module._is_tool_output_message(msg, msg["content"]) is True

    def test_is_tool_output_message_with_tool_structure(self):
        """Test detection of messages with tool output structure."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        content = "{'tool_name': 'test', 'status': 'success', 'result': 'data'}"
        msg = LiteLLMMessage(role="user", content=content)

        assert module._is_tool_output_message(msg, content) is True

    def test_is_tool_output_message_regular_message(self):
        """Test that regular messages are not detected as tool outputs."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        msg = LiteLLMMessage(role="user", content="What is the answer?")

        assert module._is_tool_output_message(msg, msg["content"]) is False

    def test_summarize_tool_message_with_dict_string(self):
        """Test summarization of tool message with Python dict string format."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        content = "{'tool_name': 'list_files', 'arguments': {'path': '/test'}, 'result': 'file1\\nfile2\\nfile3', 'status': 'success', 'error_message': None, 'duration': 1.234, 'timestamp': '2025-08-03T10:16:37'}"
        msg = LiteLLMMessage(role="tool", content=content, name="list_files")

        summary = module._summarize_tool_message(msg, content)

        assert "Tool: list_files" in summary
        assert "Status: success" in summary
        assert "Duration: 1.23s" in summary
        assert "Arguments:" in summary
        assert "Result: <returned" in summary

    def test_summarize_tool_message_with_observation_format(self):
        """Test summarization of ReAct-style observation format."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        content = "Observation: {'tool_name': 'search', 'status': 'success', 'result': 'found data'}"
        msg = LiteLLMMessage(role="user", content=content, name="search")

        summary = module._summarize_tool_message(msg, content)

        assert "Tool: search" in summary
        assert "Status: success" in summary

    def test_summarize_tool_message_with_error(self):
        """Test summarization of tool message with error."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        content = "{'tool_name': 'execute_python', 'arguments': {}, 'result': None, 'status': 'execution_error', 'error_message': 'Syntax error in code', 'duration': 0.5}"
        msg = LiteLLMMessage(role="tool", content=content, name="execute_python")

        summary = module._summarize_tool_message(msg, content)

        assert "Tool: execute_python" in summary
        assert "Status: execution_error" in summary
        assert "Error: Syntax error in code" in summary

    def test_format_messages_preserves_non_tool_messages(self):
        """Test that non-tool messages are preserved as-is."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        messages = [
            LiteLLMMessage(role="system", content="You are a helpful assistant"),
            LiteLLMMessage(role="user", content="What is 2+2?"),
            LiteLLMMessage(role="assistant", content="The answer is 4"),
        ]

        formatted = module._format_messages(messages)

        assert "SYSTEM: You are a helpful assistant" in formatted
        assert "USER: What is 2+2?" in formatted
        assert "ASSISTANT: The answer is 4" in formatted

    def test_format_messages_summarizes_tool_outputs(self):
        """Test that tool output messages are summarized."""
        module = ReflectionModule(model="test-model", reflection_prompt="test prompt")
        messages = [
            LiteLLMMessage(role="user", content="List files"),
            LiteLLMMessage(
                role="tool",
                content="{'tool_name': 'list_files', 'status': 'success', 'result': 'very long file list...', 'arguments': {}, 'duration': 0.5}",
                name="list_files",
            ),
        ]

        formatted = module._format_messages(messages)

        # Tool output should be summarized
        assert "USER: List files" in formatted
        assert "TOOL:" in formatted
        assert "Tool: list_files" in formatted
        assert "Status: success" in formatted
        # Should not contain the full result
        assert "very long file list" not in formatted
