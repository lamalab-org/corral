"""Tests for the session-only shared BaseAgent."""

import pytest

import corral.agents as agents


def test_base_agent_is_the_public_session_only_base():
    assert "BaseAgent" in agents.__all__
    assert agents.BaseAgent.run_session.__isabstractmethod__ is True
    assert agents.BaseAgent.__abstractmethods__ == frozenset({"run_session"})


def test_registered_agents_use_one_execution_entrypoint():
    for agent_type in (
        agents.ReActAgent,
        agents.ToolCallingAgent,
        agents.TerminusAgent,
        agents.LLMPlanner,
        agents.AIScientistAgent,
        agents.ReflexionAgent,
    ):
        assert hasattr(agent_type, "run_session")


def test_agent_level_run_limits_are_removed():
    with pytest.raises(TypeError, match="agent-level max_iterations"):
        agents.ReActAgent(max_iterations=3)

    with pytest.raises(TypeError, match="unexpected keyword argument 'max_turns'"):
        agents.ClaudeCodeAgent(max_turns=3)
