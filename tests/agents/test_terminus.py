"""Tests for the Corral-native TerminusAgent."""

import json

import pytest
from pydantic import ValidationError

from corral.agents import TerminusAgent
from corral.agents.terminus import (
    TerminusResponse,
    _extract_json_object,
    _strip_code_fence,
)
from corral.types import ToolResponse


class FakeLLMResponse:
    """Minimal stand-in for the LLMResponse returned by get_llm_response."""

    def __init__(self, content: str, response_id: str | None = None):
        self.content = content
        self.id = response_id


def _install_responses(agent, responses, record_calls=None):
    """Make ``agent.get_llm_response`` yield the given contents in order.

    ``record_calls`` (optional list) captures the keyword arguments each call
    received, so tests can assert that ``response_format`` is forwarded.
    """
    queue = [FakeLLMResponse(r) if isinstance(r, str) else r for r in responses]

    def fake_get_llm_response(tools=None, **kwargs):
        if record_calls is not None:
            record_calls.append(kwargs)
        assert queue, "get_llm_response called more times than expected"
        return queue.pop(0)

    agent.get_llm_response = fake_get_llm_response


def _resp(**kwargs) -> str:
    """Serialize a TerminusResponse-shaped dict to JSON."""
    return json.dumps(kwargs)


# --------------------------------------------------------------------------- #
# Schema and parsing
# --------------------------------------------------------------------------- #


def test_extract_json_object_strips_code_fence():
    text = '```json\n{"analysis": "a", "plan": "b", "final_answer": "42"}\n```'
    obj = _extract_json_object(text)
    assert obj["final_answer"] == "42"


def test_strip_code_fence_consumes_whole_triple_fence():
    # A standard three-backtick fence must be removed entirely; regressing to a
    # two-backtick match leaves a stray trailing backtick that breaks the strict
    # ``model_validate_json`` fast path in ``_parse``.
    text = '```json\n{"analysis": "a", "plan": "b", "final_answer": "42"}\n```'
    stripped = _strip_code_fence(text)
    assert not stripped.endswith("`")
    # The result must be valid JSON on its own (strict parse succeeds).
    TerminusResponse.model_validate_json(stripped)


def test_strip_code_fence_without_fence_is_unchanged():
    text = '{"analysis": "a", "plan": "b"}'
    assert _strip_code_fence(text) == text


def test_extract_json_object_ignores_leading_prose():
    text = 'Here is my response: {"analysis": "a", "plan": "b"} trailing'
    assert _extract_json_object(text)["analysis"] == "a"


def test_extract_json_object_without_object_raises():
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("no json here")


def test_response_requires_exactly_one_terminal_action():
    # Zero terminal actions is invalid.
    with pytest.raises(ValidationError):
        TerminusResponse(analysis="a", plan="b")

    # Two terminal actions at once is invalid.
    with pytest.raises(ValidationError):
        TerminusResponse(
            analysis="a",
            plan="b",
            final_answer="x",
            tool_calls=[{"name": "t", "arguments": {}}],
        )

    # Exactly one is fine.
    ok = TerminusResponse(analysis="a", plan="b", final_answer="x")
    assert ok.final_answer == "x"


# --------------------------------------------------------------------------- #
# run() behavior
#
# The merged agent defaults to Terminus-2 completion confirmation
# (``confirmations_required=1``). Core-loop tests that only care about tool
# dispatch or a single final answer disable it with ``confirmations_required=0``
# so one proposed answer is accepted immediately.
# --------------------------------------------------------------------------- #


def test_run_executes_tool_then_returns_final_answer(mock_interface):
    mock_interface.tool_responses = [
        ToolResponse(success=True, result="tool-ok", error=None)
    ]
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="inspect",
                plan="call the tool",
                tool_calls=[{"name": "test_tool", "arguments": {"query": "hi"}}],
            ),
            _resp(analysis="done", plan="answer", final_answer="42"),
        ],
    )

    answer = agent.run(mock_interface, "task-1")

    assert answer == "42"
    # The tool was dispatched through the router with the model's arguments.
    assert mock_interface.tool_calls[0]["tool_name"] == "test_tool"
    assert mock_interface.tool_calls[0]["arguments"] == {"query": "hi"}
    # A structured observation was fed back to the model.
    observation_msg = agent.messages[-2]
    assert observation_msg["role"] == "user"
    obs = json.loads(observation_msg["content"])["observations"][0]
    assert obs["success"] is True
    assert obs["result"] == "tool-ok"


def test_invalid_response_triggers_feedback_and_retry(mock_interface):
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(
        agent,
        [
            "this is not json at all",
            _resp(analysis="a", plan="b", final_answer="recovered"),
        ],
    )

    answer = agent.run(mock_interface, "task-1")

    assert answer == "recovered"
    # A correction message was inserted after the invalid response.
    assert any(
        m["role"] == "user" and "was invalid" in m["content"] for m in agent.messages
    )


def test_unknown_tool_is_reported_as_failed_observation(mock_interface):
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[{"name": "nonexistent", "arguments": {}}],
            ),
            _resp(analysis="a", plan="b", final_answer="done"),
        ],
    )

    answer = agent.run(mock_interface, "task-1")

    assert answer == "done"
    # The unknown tool never reached the router.
    assert mock_interface.tool_calls == []
    obs = json.loads(agent.messages[-2]["content"])["observations"][0]
    assert obs["success"] is False
    assert "Unknown" in obs["error"]


def test_surrender_returns_sentinel_when_enabled(mock_interface):
    agent = TerminusAgent()
    _install_responses(agent, [_resp(analysis="a", plan="b", surrender=True)])

    answer = agent.run(mock_interface, "task-1", enable_surrender=True)

    assert answer == "SURRENDER"


def test_surrender_is_rejected_when_disabled(mock_interface):
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(analysis="a", plan="b", surrender=True),
            _resp(analysis="a", plan="b", final_answer="kept going"),
        ],
    )

    answer = agent.run(mock_interface, "task-1", enable_surrender=False)

    assert answer == "kept going"
    assert any(
        m["role"] == "user" and "Surrender is disabled" in m["content"]
        for m in agent.messages
    )


def test_reaching_iteration_limit_returns_error(mock_interface):
    agent = TerminusAgent(max_iterations=2)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[{"name": "test_tool", "arguments": {"query": "x"}}],
            ),
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[{"name": "test_tool", "arguments": {"query": "x"}}],
            ),
        ],
    )

    answer = agent.run(mock_interface, "task-1")

    assert answer.startswith("Error solving the task")


def test_tools_are_stored_for_logging(mock_interface):
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="x")])
    agent.run(mock_interface, "task-1")
    assert agent._available_tools == mock_interface.available_tools["tools"]


# --------------------------------------------------------------------------- #
# Hardening: structured output, batching, truncation, arg validation
# --------------------------------------------------------------------------- #


def test_structured_output_is_requested(mock_interface):
    calls = []
    agent = TerminusAgent(use_structured_output=True, confirmations_required=0)
    _install_responses(
        agent,
        [_resp(analysis="a", plan="b", final_answer="x")],
        record_calls=calls,
    )
    agent.run(mock_interface, "task-1")
    # The schema is forwarded to the provider via response_format.
    assert calls[0].get("response_format").__name__ == "TerminusResponse"


def test_structured_output_can_be_disabled(mock_interface):
    calls = []
    agent = TerminusAgent(use_structured_output=False, confirmations_required=0)
    _install_responses(
        agent,
        [_resp(analysis="a", plan="b", final_answer="x")],
        record_calls=calls,
    )
    agent.run(mock_interface, "task-1")
    assert "response_format" not in calls[0]


def test_structured_output_falls_back_on_provider_error(mock_interface):
    agent = TerminusAgent(use_structured_output=True, confirmations_required=0)
    queue = [FakeLLMResponse(_resp(analysis="a", plan="b", final_answer="ok"))]

    def fake_get_llm_response(tools=None, **kwargs):
        # Emulate a provider that rejects response_format but works without it.
        if "response_format" in kwargs:
            raise ValueError("response_format not supported")
        return queue.pop(0)

    agent.get_llm_response = fake_get_llm_response
    answer = agent.run(mock_interface, "task-1")
    assert answer == "ok"
    # Structured output is disabled for the rest of the run after the failure.
    assert agent.use_structured_output is False


def test_excess_tool_calls_are_rejected_not_dropped(mock_interface):
    agent = TerminusAgent(max_actions_per_turn=1, confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[
                    {"name": "test_tool", "arguments": {"query": "1"}},
                    {"name": "test_tool", "arguments": {"query": "2"}},
                ],
            ),
            _resp(analysis="a", plan="b", final_answer="done"),
        ],
    )

    agent.run(mock_interface, "task-1")

    # No call was executed; the model was told to resubmit.
    assert mock_interface.tool_calls == []
    assert any(
        m["role"] == "user" and "at most 1" in m["content"] for m in agent.messages
    )


def test_batch_executes_all_when_within_budget(mock_interface):
    mock_interface.tool_responses = [
        ToolResponse(success=True, result="r1", error=None),
        ToolResponse(success=True, result="r2", error=None),
    ]
    agent = TerminusAgent(max_actions_per_turn=4, confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[
                    {"name": "test_tool", "arguments": {"query": "1"}},
                    {"name": "test_tool", "arguments": {"query": "2"}},
                ],
            ),
            _resp(analysis="a", plan="b", final_answer="done"),
        ],
    )

    agent.run(mock_interface, "task-1")

    # Both calls run when the batch fits within the per-turn budget.
    assert len(mock_interface.tool_calls) == 2


def test_large_observation_is_truncated(mock_interface):
    big = "x" * 5000
    mock_interface.tool_responses = [ToolResponse(success=True, result=big, error=None)]
    agent = TerminusAgent(max_observation_chars=100, confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[{"name": "test_tool", "arguments": {"query": "x"}}],
            ),
            _resp(analysis="a", plan="b", final_answer="done"),
        ],
    )

    agent.run(mock_interface, "task-1")

    obs = json.loads(agent.messages[-2]["content"])["observations"][0]
    assert "[truncated" in obs["result"]
    assert len(obs["result"]) < len(big)


def test_invalid_tool_arguments_are_caught_locally(mock_interface):
    # `query` is required by the mock tool schema; omit it.
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(
        agent,
        [
            _resp(
                analysis="a",
                plan="b",
                tool_calls=[{"name": "test_tool", "arguments": {}}],
            ),
            _resp(analysis="a", plan="b", final_answer="done"),
        ],
    )

    agent.run(mock_interface, "task-1")

    # The bad call never reached the router; the model got a validation error.
    assert mock_interface.tool_calls == []
    obs = json.loads(agent.messages[-2]["content"])["observations"][0]
    assert obs["success"] is False
    assert "Invalid arguments" in obs["error"]


def test_max_actions_per_turn_must_be_positive():
    with pytest.raises(ValueError, match="max_actions_per_turn"):
        TerminusAgent(max_actions_per_turn=0)


def test_repeated_invalid_responses_abort(mock_interface):
    agent = TerminusAgent(max_consecutive_parse_failures=3)
    _install_responses(agent, ["nope", "still nope", "nope again"])
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")
    assert "invalid responses" in answer


def test_surrender_prompt_is_used_when_provided(mock_interface):
    agent = TerminusAgent(surrender_prompt="CUSTOM SURRENDER GUIDANCE")
    prompt = agent._initial_prompt("task", [], enable_surrender=True)
    assert "CUSTOM SURRENDER GUIDANCE" in prompt


def test_examples_are_included_in_prompt(mock_interface):
    agent = TerminusAgent()
    prompt = agent._initial_prompt("task", [], examples=["EXAMPLE ONE"])
    assert "EXAMPLE ONE" in prompt


# --------------------------------------------------------------------------- #
# Completion confirmation and context summarization
# --------------------------------------------------------------------------- #


def test_requires_confirmation_before_completing(mock_interface):
    agent = TerminusAgent(confirmations_required=1)
    _install_responses(
        agent,
        [
            _resp(analysis="a", plan="b", final_answer="42"),
            _resp(analysis="a", plan="b", final_answer="42"),
        ],
    )

    answer = agent.run(mock_interface, "task-1")

    assert answer == "42"
    # The first proposal triggered a confirmation request.
    assert any(
        m["role"] == "user" and "resubmit the exact same final_answer" in m["content"]
        for m in agent.messages
    )


def test_first_answer_is_not_accepted_immediately(mock_interface):
    agent = TerminusAgent(confirmations_required=1, max_iterations=1)
    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="42")])

    # A single unconfirmed final answer is not accepted; with the turn budget
    # exhausted the run reports the iteration-limit error instead.
    answer = agent.run(mock_interface, "task-1")
    assert answer.startswith("Error solving the task")


def test_changed_answer_restarts_confirmation(mock_interface):
    agent = TerminusAgent(confirmations_required=1)
    _install_responses(
        agent,
        [
            _resp(analysis="a", plan="b", final_answer="first"),
            _resp(analysis="a", plan="b", final_answer="second"),
            _resp(analysis="a", plan="b", final_answer="second"),
        ],
    )

    answer = agent.run(mock_interface, "task-1")
    # "second" only accepted after it is itself re-affirmed.
    assert answer == "second"


def test_confirmation_state_does_not_leak_across_runs(mock_interface):
    # A pending, unconfirmed answer from one run must not count toward
    # confirmation in the next run of the same agent instance.
    agent = TerminusAgent(confirmations_required=1, max_iterations=1)
    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="42")])
    first = agent.run(mock_interface, "task-1")
    assert first.startswith("Error solving the task")

    # Second run: a single "42" must again require confirmation rather than
    # being accepted because the previous run already proposed "42".
    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="42")])
    second = agent.run(mock_interface, "task-1")
    assert second.startswith("Error solving the task")


def test_confirmations_disabled_accepts_immediately(mock_interface):
    agent = TerminusAgent(confirmations_required=0)
    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="42")])
    answer = agent.run(mock_interface, "task-1")
    assert answer == "42"


def test_summarizes_when_over_threshold(mock_interface, monkeypatch):
    agent = TerminusAgent(
        confirmations_required=0,
        summarize_after_tokens=10,
        keep_recent_messages=2,
    )

    # Force the token threshold to trip on the next compaction check.
    agent.token_usage = {"total_tokens": 1000}

    summarize_calls = []

    def fake_summarize(messages):
        summarize_calls.append(list(messages))
        return "SUMMARY"

    agent._summarize = fake_summarize

    # Seed a long history so there is a middle to summarize.
    agent._initial_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "old-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "recent"},
    ]

    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="done")])
    agent.run(mock_interface, "task-1")

    # Summarization ran and collapsed the middle into a single summary message.
    assert summarize_calls
    assert any(
        m["role"] == "user" and "Summary of earlier progress" in m["content"]
        for m in agent.messages
    )


def test_summarization_failure_leaves_history_intact(mock_interface):
    agent = TerminusAgent(
        confirmations_required=0,
        summarize_after_tokens=10,
        keep_recent_messages=2,
    )
    agent.token_usage = {"total_tokens": 1000}
    agent._summarize = lambda messages: None  # emulate a failed summary

    agent._initial_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old-1"},
        {"role": "user", "content": "obs-1"},
        {"role": "assistant", "content": "old-2"},
        {"role": "user", "content": "obs-2"},
        {"role": "assistant", "content": "recent"},
    ]

    _install_responses(agent, [_resp(analysis="a", plan="b", final_answer="done")])
    agent.run(mock_interface, "task-1")

    # No summary message was inserted; original history is preserved.
    assert not any(
        m.get("content", "").startswith("Summary of earlier progress")
        for m in agent.messages
    )
