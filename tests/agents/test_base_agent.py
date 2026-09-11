"""Tests for the session-only shared BaseAgent."""

import pytest

import corral.agents as agents
from corral.agents.base_agent import _UsageAccumulator
from corral.agents.schema import AgentUsage


def test_base_agent_is_the_public_session_only_base():
    assert "BaseAgent" in agents.__all__
    assert agents.BaseAgent.run_session.__isabstractmethod__ is True
    assert agents.BaseAgent.__abstractmethods__ == frozenset({"run_session"})


def test_base_agent_normalizes_canonical_and_litellm_usage_fields():
    agent = agents.ReActAgent(system_prompt="system")

    assert agent._usage(
        {
            "input_tokens": 11,
            "output_tokens": 4,
            "reasoning_tokens": 3,
        },
        llm_calls=2,
    ) == AgentUsage(
        input_tokens=11,
        output_tokens=4,
        reasoning_tokens=3,
        llm_calls=2,
    )
    assert agent._usage(
        {"prompt_tokens": 7, "completion_tokens": 2},
        llm_calls=1,
    ) == AgentUsage(input_tokens=7, output_tokens=2, llm_calls=1)


def test_usage_accumulators_keep_conversion_and_totals_per_run():
    def convert(raw_usage, *, llm_calls=0):
        raw_usage = raw_usage or {}
        return AgentUsage(
            input_tokens=int(raw_usage.get("provider_input", 0)),
            output_tokens=int(raw_usage.get("provider_output", 0)),
            llm_calls=llm_calls,
        )

    first = _UsageAccumulator(convert)
    second = _UsageAccumulator(convert)
    first.add({"provider_input": 5, "provider_output": 2})
    first.add(None)
    second.add({"provider_input": 1, "provider_output": 1})

    assert first.outcome() == AgentUsage(input_tokens=5, output_tokens=2, llm_calls=2)
    assert second.outcome() == AgentUsage(input_tokens=1, output_tokens=1, llm_calls=1)


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
