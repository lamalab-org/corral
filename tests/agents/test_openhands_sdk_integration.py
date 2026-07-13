"""Real-SDK boundary tests for :class:`OpenHandsAgent`.

Unlike `test_openhands.py` — which fakes the SDK to exercise the agent's
control flow — these tests construct *real* `openhands-sdk` objects. They pin
down the exact SDK shapes the transcript/serialization code depends on
(`Observation.is_error`, `MCPToolAction.data`, `AgentContext` datetime,
the MCP executor timeout), which fakes cannot catch: a fake that mirrors a buggy
implementation would agree with it. If the SDK moves these, these tests fail
instead of the trace silently going wrong.

The `openhands` extra is gated to Python >= 3.12, so the whole module is
skipped when the SDK is not importable.
"""

import pytest
from pydantic import ValidationError

pytest.importorskip("openhands.sdk")

from openhands.sdk import AgentContext
from openhands.sdk.mcp.definition import (
    MCPToolAction,
    MCPToolObservation,
)
from openhands.sdk.mcp.tool import MCP_TOOL_TIMEOUT_SECONDS

from corral.agents.openhands import (
    _OPENHANDS_EXECUTOR_TIMEOUT_S,
    OpenHandsAgent,
)


def test_action_arguments_reads_real_mcp_tool_action_data():
    """The recorder surfaces `MCPToolAction.data`, the real per-call arguments."""

    class _Event:
        action = MCPToolAction(data={"query": "benzene", "n": 3})

    assert OpenHandsAgent._action_arguments(_Event()) == {"query": "benzene", "n": 3}


def test_real_mcp_observation_error_flag_is_is_error_not_error():
    """A failed MCP observation exposes `is_error` (and no `error` attribute)."""
    obs = MCPToolObservation.from_text(text="boom", is_error=True, tool_name="t")

    # This is exactly the attribute `_record_observation_event` reads.
    assert getattr(obs, "is_error", None) is True
    # The pre-fix code read `.error`, which does not exist on real observations,
    # so it silently recorded every failure as a success.
    assert not hasattr(obs, "error")

    ok = MCPToolObservation.from_text(text="fine", is_error=False, tool_name="t")
    assert ok.is_error is False


def test_agent_context_datetime_none_omits_datetime_block():
    """`current_datetime=None` removes the machine-local datetime from the prompt."""
    assert AgentContext(current_datetime=None).get_formatted_datetime() is None
    # The default captures wall-clock time, i.e. it is non-reproducible.
    assert AgentContext().get_formatted_datetime() is not None


def test_executor_timeout_constant_tracks_installed_sdk():
    """The recorded executor cap is read from the SDK, not hard-coded."""
    assert float(MCP_TOOL_TIMEOUT_SECONDS) == _OPENHANDS_EXECUTOR_TIMEOUT_S


def test_default_tool_timeout_is_enforceable_end_to_end():
    """The default per-call timeout equals the cap OpenHands can actually enforce."""
    agent = OpenHandsAgent(model="openai/gpt-5.6")
    assert agent.tool_timeout_s == float(MCP_TOOL_TIMEOUT_SECONDS)
    assert agent.effective_tool_timeout_s == float(MCP_TOOL_TIMEOUT_SECONDS)


def test_reasoning_effort_is_applied_to_real_llm():
    """A requested reasoning_effort reaches the real SDK `LLM`; unset -> default."""
    llm = OpenHandsAgent(
        model="openai/gpt-5.6", api_key="x", reasoning_effort="low"
    )._make_llm()
    assert llm.reasoning_effort == "low"

    # Unset defers to the SDK's own default rather than pinning a value here.
    default_llm = OpenHandsAgent(model="openai/gpt-5.6", api_key="x")._make_llm()
    assert default_llm.reasoning_effort == "high"


def test_invalid_reasoning_effort_is_rejected_by_sdk():
    """An unsupported reasoning_effort fails at LLM construction (SDK-validated)."""
    with pytest.raises(ValidationError):
        OpenHandsAgent(
            model="openai/gpt-5.6", api_key="x", reasoning_effort="bogus"
        )._make_llm()
