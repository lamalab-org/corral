"""Sampling defaults must describe execution without modifying requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import litellm
import pytest

from corral.cli import create_agent
from corral.orchestration.models import AgentRuntimeDefinition
from corral.orchestration.parameters import model_parameter_metadata


@pytest.mark.parametrize(
    ("name", "temperature"),
    [("tool-calling", 0.7), ("ai-scientist", 0.2), ("reflexion", 0.0)],
)
def test_constructor_temperature_and_catalog_effort_are_recorded(
    monkeypatch, name, temperature
):
    lookup = Mock(return_value={"default_reasoning_effort": "medium"})
    monkeypatch.setattr(litellm, "get_model_info", lookup)
    definition = AgentRuntimeDefinition(name=name, model="openai/test-model")
    agent = create_agent(name, model=definition.model)

    metadata = model_parameter_metadata(agent, model=agent.model, definition=definition)

    assert metadata == {
        "temperature": temperature,
        "reasoning_effort": "medium",
        "parameter_sources": {
            "temperature": "agent",
            "reasoning_effort": "litellm_default",
        },
    }
    lookup.assert_called_once_with("openai/test-model")
    assert "reasoning_effort" not in agent.kwargs
    assert definition.temperature is None
    assert definition.reasoning_effort is None


@pytest.mark.parametrize("use_options", [False, True])
def test_explicit_zero_and_none_effort_take_precedence(monkeypatch, use_options):
    lookup = Mock(side_effect=AssertionError("explicit settings need no lookup"))
    monkeypatch.setattr(litellm, "get_model_info", lookup)
    settings = {"temperature": 0.0, "reasoning_effort": "none"}
    definition = AgentRuntimeDefinition(
        name="tool-calling", **({"options": settings} if use_options else settings)
    )
    agent = create_agent("tool-calling", agent_kwargs=settings)

    metadata = model_parameter_metadata(agent, model=agent.model, definition=definition)

    assert metadata == {
        **settings,
        "parameter_sources": {
            "temperature": "explicit",
            "reasoning_effort": "explicit",
        },
    }
    lookup.assert_not_called()


@pytest.mark.parametrize(
    ("agent_effort", "expected", "source"),
    [(None, "high", "sdk_default"), ("low", "low", "agent")],
)
def test_agent_and_sdk_defaults_precede_catalog(
    monkeypatch, agent_effort, expected, source
):
    lookup = Mock(return_value={"default_reasoning_effort": "medium"})
    monkeypatch.setattr(litellm, "get_model_info", lookup)
    agent = SimpleNamespace(
        temperature=0.7,
        reasoning_effort=agent_effort,
        model_parameter_defaults=lambda: {"reasoning_effort": "high"},
    )

    metadata = model_parameter_metadata(agent, model="openai/test-model")

    assert metadata["reasoning_effort"] == expected
    assert metadata["parameter_sources"]["reasoning_effort"] == source
    lookup.assert_not_called()


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_missing_catalog_defaults_remain_unknown(monkeypatch, lookup_fails):
    lookup = Mock(return_value={"supports_reasoning": True})
    if lookup_fails:
        lookup.side_effect = ValueError("unknown model")
    monkeypatch.setattr(litellm, "get_model_info", lookup)
    agent = SimpleNamespace(temperature=None, kwargs={})

    assert model_parameter_metadata(agent, model="custom/model") == {
        "temperature": None,
        "reasoning_effort": None,
        "parameter_sources": {
            "temperature": "unknown",
            "reasoning_effort": "unknown",
        },
    }


def test_ignored_temperature_is_not_reported_as_applied():
    agent = SimpleNamespace(
        temperature=0.7,
        reasoning_effort="high",
        unsupported_model_parameters={"temperature"},
    )
    metadata = model_parameter_metadata(
        agent,
        model="test-model",
        definition=AgentRuntimeDefinition(name="codex", temperature=0.5),
    )

    assert metadata["temperature"] is None
    assert metadata["parameter_sources"]["temperature"] == "unsupported"
    assert metadata["reasoning_effort"] == "high"


def test_real_litellm_model_info_exposes_catalog_default(monkeypatch):
    model = "openai/corral-parameter-test"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "max_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "default_reasoning_effort": "medium",
        },
    )
    metadata = model_parameter_metadata(SimpleNamespace(temperature=0.7), model=model)

    assert metadata["reasoning_effort"] == "medium"
    assert metadata["parameter_sources"]["reasoning_effort"] == "litellm_default"
