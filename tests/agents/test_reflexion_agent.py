"""Integration tests for ReflexionAgent."""

from corral.agents.react import ReActAgent
from corral.agents.reflexion_agent import ReflexionAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.agents.utils import LiteLLMMessage
from corral.types import ToolResponse

from .conftest import MockLLMResponse


class TestReflexionAgentInitialization:
    """Test cases for ReflexionAgent initialization."""

    def test_init_with_react_agent(self):
        """Test initializing ReflexionAgent with ReActAgent."""
        base_agent = ReActAgent(model="test-model", max_iterations=5)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert reflexion_agent.actor == base_agent
        assert reflexion_agent.model == base_agent.model
        assert reflexion_agent.max_iterations == base_agent.max_iterations

    def test_init_with_tool_calling_agent(self):
        """Test initializing ReflexionAgent with ToolCallingAgent."""
        base_agent = ToolCallingAgent(model="test-model", max_iterations=10)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert reflexion_agent.actor == base_agent

    def test_init_custom_reflection_model(self):
        """Test initializing with custom reflection model."""
        base_agent = ReActAgent(model="gpt-4")
        reflexion_agent = ReflexionAgent(actor=base_agent, reflection_model="gpt-4o")

        assert reflexion_agent.reflection_module.model == "gpt-4o"

    def test_memory_initialization(self):
        """Test that memory is properly initialized."""
        base_agent = ReActAgent(model="test-model")
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        assert len(reflexion_agent.memory.reflections) == 0
        assert reflexion_agent.memory.max_size == 3


class TestReflexionAgentRun:
    """Test cases for ReflexionAgent.run method."""

    def test_success_on_first_attempt(self, mock_interface, monkeypatch):
        """Test that reflexion agent succeeds on first attempt."""
        # Create base agent
        base_agent = ReActAgent(model="test-model", max_iterations=3)
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

    def test_retry_with_reflection(self, mock_interface, monkeypatch):
        """Test that agent retries with reflection after failure."""
        base_agent = ReActAgent(model="test-model", max_iterations=3)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First attempt fails (needs 3 responses to exhaust iterations), second succeeds
        responses = [
            # First attempt - iteration 1
            MockLLMResponse(
                content="Thought: <thought>Let me try something.</thought>\nAction: <action>tool1</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # First attempt - iteration 2
            MockLLMResponse(
                content="Thought: <thought>That didn't work.</thought>\nAction: <action>tool2</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # First attempt - iteration 3
            MockLLMResponse(
                content="Thought: <thought>Still not working.</thought>\nAction: <action>tool3</action>\nAction Input: <action_input>{}</action_input>"
            ),
            # Second attempt - succeeds immediately
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
            return "I should try a different approach next time."

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

        # Run agent
        result = reflexion_agent.run(mock_interface, "test_task")

        # Should succeed on second attempt
        assert result == "Success"
        assert call_count["llm"] == 4  # 3 for first attempt + 1 for second
        assert call_count["reflection"] == 1
        assert len(reflexion_agent.memory.reflections) == 1

    def test_exhausts_all_attempts(self, mock_interface, monkeypatch):
        """Test that agent exhausts all attempts before giving up."""
        base_agent = ReActAgent(model="test-model", max_iterations=3)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # All attempts fail - each needs 3 responses to exhaust iterations
        call_count = {"llm": 0, "reflection": 0}

        def mock_get_llm_response(*args, **kwargs):
            call_count["llm"] += 1
            # Each iteration returns a failed action
            return MockLLMResponse(
                content=f"Thought: <thought>Attempt {call_count['llm']}.</thought>\nAction: <action>failing_tool</action>\nAction Input: <action_input>{{}}</action_input>"
            )

        def mock_reflection_generate(*args, **kwargs):
            call_count["reflection"] += 1
            return f"Reflection {call_count['reflection']}"

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Set up tool responses that always succeed (but agent never finds final answer)
        mock_interface.tool_responses = [
            ToolResponse(success=True, result="result", error=None)
        ] * 20  # More than enough for all attempts

        # Run agent
        result = reflexion_agent.run(mock_interface, "test_task")

        # Should try all 3 attempts
        assert "Error" in result
        assert call_count["llm"] == 9  # 3 attempts x 3 iterations each
        assert (
            call_count["reflection"] == 2
        )  # Only generate reflection for first 2 failures
        assert len(reflexion_agent.memory.reflections) == 2

    def test_memory_injection_into_history(self, mock_interface, monkeypatch):
        """Test that reflections are injected into actor's history."""
        base_agent = ReActAgent(model="test-model", max_iterations=3)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First fails, second succeeds
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
            return "Important lesson learned"

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Run
        reflexion_agent.run(mock_interface, "test_task")

        # Check that history was injected on second attempt
        assert len(captured_history) == 2
        assert (
            captured_history[0] is None or len(captured_history[0]) == 0
        )  # First attempt: no history
        assert (
            captured_history[1] is not None
        )  # Second attempt: history with reflection
        assert len(captured_history[1]) > 0
        assert captured_history[1][0]["role"] == "system"
        assert "LESSONS FROM PREVIOUS ATTEMPTS" in captured_history[1][0]["content"]

    def test_memory_clears_for_new_task(self, mock_interface, monkeypatch):
        """Test that memory clears when switching tasks."""
        base_agent = ReActAgent(model="test-model", max_iterations=3)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        def mock_actor_run(self, interface, task_id, history=None, **kwargs):
            self.messages = [LiteLLMMessage(role="assistant", content="Error")]
            return "Error solving the task"

        def mock_reflection_generate(*args, **kwargs):
            return "Reflection"

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Run first task
        reflexion_agent.run(mock_interface, "task_1")
        assert len(reflexion_agent.memory.reflections) == 1

        # Run second task
        reflexion_agent.run(mock_interface, "task_2")

        # Memory should be cleared for new task
        # (Will have new reflections from task_2)
        assert all(r.task_id == "task_2" for r in reflexion_agent.memory.reflections)

    def test_with_tool_execution(self, mock_interface, monkeypatch):
        """Test reflexion agent with tool execution."""
        base_agent = ReActAgent(model="test-model", max_iterations=5)
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # First attempt uses wrong tool and exhausts iterations, second uses right tool and succeeds
        responses = [
            # First attempt - iteration 1
            MockLLMResponse(
                content="""Thought: <thought>I'll use tool A.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First attempt - iteration 2
            MockLLMResponse(
                content="""Thought: <thought>Let me try again.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First attempt - iteration 3
            MockLLMResponse(
                content="""Thought: <thought>Still trying.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First attempt - iteration 4
            MockLLMResponse(
                content="""Thought: <thought>One more time.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # First attempt - iteration 5
            MockLLMResponse(
                content="""Thought: <thought>Last try.</thought>
Action: <action>wrong_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # Second attempt - iteration 1 (with reflection)
            MockLLMResponse(
                content="""Thought: <thought>Based on reflection, I'll use tool B.</thought>
Action: <action>correct_tool</action>
Action Input: <action_input>{}</action_input>"""
            ),
            # Second attempt - iteration 2
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
            return "Use correct_tool instead of wrong_tool"

        monkeypatch.setattr(
            "corral.agents.base_agent.BaseAgent.get_llm_response", mock_get_llm_response
        )
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Set up tool responses
        mock_interface.tool_responses = [
            ToolResponse(
                success=False, result=None, error="Wrong tool"
            ),  # First attempt calls
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(success=False, result=None, error="Wrong tool"),
            ToolResponse(
                success=True, result="Correct result", error=None
            ),  # Second attempt call
        ]

        # Run
        result = reflexion_agent.run(mock_interface, "test_task")

        assert result == "Success"
        assert len(reflexion_agent.memory.reflections) == 1

    def test_get_reflection_summary(self):
        """Test getting reflection summary."""
        base_agent = ReActAgent(model="test-model")
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # Add some reflections manually
        from corral.agents.reflection import Reflection

        reflection = Reflection(
            trial_index=0,
            task_id="task_1",
            error_signal="Error",
            trajectory=[LiteLLMMessage(role="user", content="Test")],
            reflection_text="Test reflection",
            score=0.3,
        )
        reflexion_agent.memory.add_reflection(reflection)

        summary = reflexion_agent.get_reflection_summary()

        assert summary["num_reflections"] == 1
        assert len(summary["reflections"]) == 1
        assert summary["reflections"][0]["attempt"] == 1
        assert summary["reflections"][0]["score"] == 0.3
        assert summary["reflections"][0]["reflection"] == "Test reflection"


class TestReflexionAgentEdgeCases:
    """Test edge cases for ReflexionAgent."""

    def test_exception_handling(self, mock_interface, monkeypatch):
        """Test that exceptions are handled gracefully."""
        base_agent = ReActAgent(model="test-model", max_iterations=3)
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
            return "Handle the error better"

        monkeypatch.setattr("corral.agents.react.ReActAgent.run", mock_actor_run)
        monkeypatch.setattr(
            "corral.agents.reflection.ReflectionModule.generate_reflection",
            mock_reflection_generate,
        )

        # Run - should recover from exception
        result = reflexion_agent.run(mock_interface, "test_task")

        assert result == "Success"
        assert len(reflexion_agent.memory.reflections) == 1

    def test_is_successful_answer(self):
        """Test the success detection heuristic."""
        base_agent = ReActAgent(model="test-model")
        reflexion_agent = ReflexionAgent(
            reflection_model="test-model", actor=base_agent
        )

        # These should be considered failures
        assert not reflexion_agent._is_successful_answer("Error solving the task")
        assert not reflexion_agent._is_successful_answer(
            "unable to complete it in the iteration limit"
        )
        assert not reflexion_agent._is_successful_answer("Maximum iterations reached")
        assert not reflexion_agent._is_successful_answer("Failed to execute tool")

        # These should be considered successes
        assert reflexion_agent._is_successful_answer("The answer is 42")
        assert reflexion_agent._is_successful_answer("Success! Task completed")
        assert reflexion_agent._is_successful_answer("Here is the result: ...")
