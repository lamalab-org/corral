"""Native Denario transport, session semantics, and optional full-SDK checks."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

import httpx
import pytest
from tests.agents.commit_session import start_session

from corral import cli
from corral.agents import DenarioAgent
from corral.agents.denario._bridge import Bridge, completion_event
from corral.agents.denario._responses import (
    ResponsesAdapter,
    attributed_message,
    encode_completion,
)
from corral.agents.session import AgentSession
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sandbox_workspace(monkeypatch, tmp_path):
    # Only test code runs here. Production requires the real Docker capability.
    monkeypatch.setattr(
        AgentSession, "native_tool_workspace", property(lambda _self: str(tmp_path))
    )
    return tmp_path


async def make_session(agent, limit=50):
    measurements = []

    def measure(sample: str) -> str:
        """Measure the named sample."""
        measurements.append(sample)
        if sample == "missing":
            raise ValueError("Sample does not exist")
        return "measured:42"

    environment = Environment(
        "denario-task",
        TaskDefinition(
            name="denario-task",
            description="Measure sample a and submit the result.",
            tools=["measure"],
            scoring_fn=lambda _answer: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
        toolset=Toolset(pool={"measure": tool(measure)}, workspace_factory=None),
    )
    return await start_session(
        environment, max_iterations=limit, agent=agent
    ), measurements


def fake_worker(monkeypatch, agent, program):
    prelude = """
import json, sys
config = json.loads(sys.stdin.readline())
def rpc(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)
    return json.loads(sys.stdin.readline())["result"]
def model(i):
    rpc("model_start", id=str(i))
    rpc("model_end", id=str(i), agent="planner", input=[{"role":"user","content":"plan"}],
        message={"role":"assistant","content":"I will measure.","tool_calls":None},
        usage={"prompt_tokens":10,"completion_tokens":5,"reasoning_tokens":2})
"""
    monkeypatch.setattr(
        agent,
        "_worker_command",
        lambda: [sys.executable, "-u", "-c", prelude + program],
    )


def test_cli_is_lazy_and_uses_one_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "denario", None)
    agent = cli.create_agent(
        "DenarioAgent",
        model="openai/gpt-6.1-sol",
        agent_kwargs={"denario_iterations": 3, "reasoning_effort": "low"},
    )
    assert isinstance(agent, DenarioAgent)
    assert not isinstance(agent, ToolCallingAgent)
    assert agent.denario_model == "gpt-6.1-sol"
    assert agent.kwargs == {"reasoning_effort": "low"}
    assert agent.temperature is None
    with pytest.raises(ValueError, match="must match"):
        DenarioAgent(model="openai/gpt-6.1-sol", denario_model="gpt-4o")
    with pytest.raises(TypeError, match="Unsupported"):
        DenarioAgent(epistemic=True)


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5, "2"])
def test_invalid_planning_iterations(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        DenarioAgent(denario_iterations=iterations)


@pytest.mark.parametrize(
    ("model", "provider", "native"),
    [
        ("openai/gpt-6.1-sol", "openai", "gpt-6.1-sol"),
        ("anthropic/claude-opus-5-5", "anthropic", "claude-opus-5-5"),
        (
            "openrouter/moonshotai/kimi-k3.1-pro",
            "openrouter",
            "moonshotai/kimi-k3.1-pro",
        ),
    ],
)
def test_provider_selection(model, provider, native):
    agent = DenarioAgent(model=model)
    assert (agent.provider, agent.denario_model) == (provider, native)
    assert agent.kwargs == {}
    with pytest.raises(ValueError, match="OpenRouter"):
        DenarioAgent(model="openai/gpt-6.1-sol", openrouter_zdr=True)


@pytest.mark.parametrize("streaming", [False, True])
def test_openrouter_enforces_zdr_and_preserves_reasoning(monkeypatch, streaming):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use")
    bridge = Bridge(
        {
            "provider": "openrouter",
            "model": "moonshotai/kimi-k3.1-pro",
            "model_settings": {},
            "openrouter_zdr": True,
            "openrouter_provider": {
                "zdr": False,
                "data_collection": "allow",
                "only": ["fireworks"],
            },
        },
        StringIO(),
        StringIO(),
    )
    events, requests = [], []
    monkeypatch.setattr(
        bridge, "rpc", lambda kind, **fields: events.append((kind, fields))
    )
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "opaque", "index": 0}
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "measure", "arguments": "{}"},
            }
        ],
    }
    completion = {
        "id": "test",
        "created": 1,
        "model": "moonshotai/kimi-k3.1-pro",
        "provider": "Fireworks",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }

    def respond(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer router-test"
        data = json.loads(request.content)
        assert data["model"] == "moonshotai/kimi-k3.1-pro"
        assert data["provider"] == {
            "zdr": True,
            "data_collection": "deny",
            "only": ["fireworks"],
        }
        assert data["stream"] is False
        assert "temperature" not in data
        assert "reasoning_effort" not in data
        requests.append(data)
        return httpx.Response(200, json=completion)

    client_class = httpx.Client
    monkeypatch.setattr(
        "corral.agents.denario._bridge.httpx.Client",
        lambda **kw: client_class(transport=httpx.MockTransport(respond), **kw),
    )
    try:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "measure"}],
            "stream": streaming,
        }
        assert bridge.model(body)[0] == 200
        # Simulate an older SDK dropping provider-specific reasoning fields.
        body["messages"] += [
            {k: v for k, v in message.items() if k != "reasoning_details"},
            {"role": "tool", "tool_call_id": "call-1", "content": "42"},
        ]
        assert bridge.model(body)[0] == 200
    finally:
        bridge.server.server_close()
    assert requests[1]["messages"][1] == message
    assert events[1][1]["message"] == message
    assert events[1][1]["provider_response"] == completion


@pytest.mark.anyio
async def test_requires_sandbox():
    agent = DenarioAgent()
    session, _ = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "harness_failure"
    assert "Docker" in result.error


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-opus-5-5"),
        ("openrouter", "moonshotai/kimi-k3.1-pro"),
        ("openai", "gpt-4o"),
    ],
)
def test_forced_tool_compatibility(monkeypatch, provider, model):
    bridge = Bridge(
        {"provider": provider, "model": model, "model_settings": {}},
        StringIO(),
        StringIO(),
    )
    events = []
    monkeypatch.setattr(
        bridge, "rpc", lambda kind, **fields: events.append((kind, fields))
    )
    choice = {"type": "function", "function": {"name": "record_plan_constraints"}}
    body = {
        "messages": [{"role": "user", "content": "Set constraints."}],
        "tool_choice": choice,
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in ("record_plan_constraints", "unrelated")
        ],
    }

    def respond(request):
        data = json.loads(request.content)
        if provider != "anthropic":
            assert data["tool_choice"] == choice
            assert len(data["tools"]) == 2
        else:
            assert data["tool_choice"] == "auto"
            assert len(data["tools"]) == 2
            assert data["tools"][0]["function"]["name"] == "record_plan_constraints"
            assert (
                data["messages"][-1]["content"]
                == "Call the record_plan_constraints tool now."
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        "corral.agents.denario._bridge.httpx.Client",
        lambda **kw: client_class(transport=httpx.MockTransport(respond), **kw),
    )
    try:
        assert bridge.model(body)[0] == 200
    finally:
        bridge.server.server_close()
    assert body["tool_choice"] == choice
    assert len(body["messages"]) == 1
    if provider == "anthropic":
        assert events[0][1]["event"]["type"] == "tool_choice_compatibility"


@pytest.mark.parametrize(
    ("provider", "model", "parallel"),
    [
        ("openrouter", "moonshotai/kimi-k3.1-pro", False),
        ("openrouter", "moonshotai/kimi-k3.1-pro", True),
        ("openrouter", "moonshotai/kimi-k3.1-pro", None),
        ("openrouter", "another/model", False),
        ("openai", "gpt-4o", False),
        ("anthropic", "claude-opus-5-5", False),
    ],
)
def test_kimi_parallel_hint_with_strict_zdr_routing(
    monkeypatch, provider, model, parallel
):
    kimi = provider == "openrouter" and model == "moonshotai/kimi-k3.1-pro"
    routing = {
        "only": ["fireworks"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    bridge = Bridge(
        {
            "provider": provider,
            "model": model,
            "model_settings": {},
            "openrouter_zdr": True,
            "openrouter_provider": routing,
        },
        StringIO(),
        StringIO(),
    )
    events, requests = [], []
    monkeypatch.setattr(
        bridge, "rpc", lambda kind, **fields: events.append((kind, fields))
    )
    choice = {"type": "function", "function": {"name": "measure"}} if kimi else "auto"
    body = {
        "messages": [{"role": "user", "content": "Measure a."}],
        "tool_choice": choice,
        "parallel_tool_calls": parallel,
        "tools": [
            {
                "type": "function",
                "function": {"name": "measure", "parameters": {"type": "object"}},
            }
        ],
    }
    calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": "measure", "arguments": "{}"},
        }
        for i in range(2)
    ]

    def respond(request):
        data = json.loads(request.content)
        requests.append(data)
        # Reproduce the actual routing failure: strict matching must stay on.
        if kimi and "parallel_tool_calls" in data:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "No endpoints found that can handle the requested parameters."
                    }
                },
            )
        assert data["tool_choice"] == choice
        assert data["tools"] == body["tools"]
        if provider == "openrouter":
            assert data["provider"] == {
                **routing,
                "zdr": True,
                "data_collection": "deny",
            }
        assert "reasoning_effort" not in data
        assert "temperature" not in data
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": calls,
                        }
                    }
                ]
            },
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        "corral.agents.denario._bridge.httpx.Client",
        lambda **kw: client_class(transport=httpx.MockTransport(respond), **kw),
    )
    try:
        status, _, content = bridge.model(body)
        assert status == 200
    finally:
        bridge.server.server_close()
    assert json.loads(content)["choices"][0]["message"]["tool_calls"] == calls
    assert events[-1][1]["message"]["tool_calls"] == calls
    assert events[-1][1]["input"] == requests[0]["messages"]
    assert len(body["messages"]) == 1
    assert body["parallel_tool_calls"] is parallel
    if kimi:
        event = events[0][1]["event"]
        assert event["type"] == "parallel_tool_calls_compatibility"
        assert event["requested"] is parallel
        assert event["applied"] == "omitted"
        if parallel is False:
            assert "at most one tool call" in requests[0]["messages"][-1]["content"]
        else:
            assert requests[0]["messages"] == body["messages"]
    else:
        assert requests[0]["parallel_tool_calls"] is parallel
        assert requests[0]["messages"] == body["messages"]


@pytest.mark.anyio
async def test_worker_receives_task_text_without_corral_prompt_templates(
    monkeypatch, sandbox_workspace
):
    agent = DenarioAgent(
        system_prompt="CORRAL_SYSTEM_WRAPPER",
        user_prompt="CORRAL_USER_WRAPPER {{task_guide}}",
        surrender_prompt="CORRAL_SURRENDER_WRAPPER",
    )
    task = 'Measure \u03b1.\nKeep this literal input: {"sample": "a"}.\nReturn a SMILES string.'
    monkeypatch.setattr(AgentSession, "prompt", property(lambda _self: task))
    fake_worker(
        monkeypatch,
        agent,
        """
from pathlib import Path
assert isinstance(config["task"], str)
Path(config["project_dir"], "task.txt").write_text(config["task"])
rpc("done")
""",
    )
    session, _ = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "protocol_failure"  # probe deliberately does not submit
    assert (Path(result.metadata["project_dir"]) / "task.txt").read_text() == task
    assert result.usage.llm_calls == 0


@pytest.mark.anyio
async def test_non_text_task_is_rejected_without_serializing_or_calling_provider(
    monkeypatch, sandbox_workspace
):
    monkeypatch.setattr(
        AgentSession,
        "prompt",
        property(lambda _self: [{"type": "image_url", "image_url": {"url": "test"}}]),
    )
    agent = DenarioAgent()
    session, measurements = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "harness_failure"
    assert "text task description" in result.error
    assert result.usage.llm_calls == 0
    assert measurements == []


@pytest.mark.anyio
async def test_session_tool_bridge_and_total_usage(monkeypatch, sandbox_workspace):
    agent = DenarioAgent()
    fake_worker(
        monkeypatch,
        agent,
        """
model(1)
assert rpc("tool", name="nonexistent", arguments={}, id="unknown")["status"] == "invalid_tool"
assert rpc("tool", name="measure", arguments={}, id="invalid")["status"] == "invalid_args"
assert rpc("tool", name="measure", arguments={"sample":"missing"}, id="failed")["status"] == "execution_error"
model(2)
rpc("messages", messages=[{"role":"assistant", "content":None, "sdk_native":True,
    "tool_calls":[{"id":"code", "type":"function", "function":{"name":"denario.execute_code", "arguments":"{}"}}]}])
r = rpc("tool", name="measure", arguments={"sample":"a"}, id="measure")
assert r["result"] == "measured:42"
rpc("proposal", arguments={"answer":r["result"]}, id="proposal", role="researcher", step=2)
from pathlib import Path
report = Path(config["project_dir"], "input_files", "results.md")
report.parent.mkdir()
report.write_text("The measured result is 42.")
rpc("done", context={"current_status":"completed", "current_plan_step_number":2,
    "number_of_steps_in_plan":2, "n_attempts":0, "max_n_attempts":10})
raise RuntimeError("must stop at accepted submission")
""",
    )
    session, measurements = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "completed", result.error
    assert result.answer == "measured:42"
    assert measurements == ["missing", "a"]
    assert result.usage.llm_calls == 2
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10
    assert result.usage.reasoning_tokens == 4
    assert session.state.usage.llm_calls == 2
    assert result.metadata["usage_scope"] == "all_native_model_requests"
    tail = session.messages[-1]
    assert tail["name"] == "denario.execute_code"
    assert tail["role"] == "tool"
    assert "interrupted after accepted submission" in tail["content"]


@pytest.mark.anyio
async def test_budget_stops_native_loop_before_next_request(
    monkeypatch, sandbox_workspace
):
    agent = DenarioAgent()
    fake_worker(
        monkeypatch,
        agent,
        """
model(1)
model(2)
rpc("tool", name="measure", arguments={"sample":"a"}, id="should-not-run")
""",
    )
    session, measurements = await make_session(agent, limit=1)
    result = await agent.run_session(session)
    assert result.status == "iteration_limit"
    assert result.usage.llm_calls == session.state.usage.llm_calls == 1
    assert measurements == []
    resumed = await agent.run_session(session)
    assert resumed.status == "harness_failure"
    assert "replaying" in resumed.error


@pytest.mark.anyio
async def test_no_submission_is_not_success(monkeypatch, sandbox_workspace):
    agent = DenarioAgent()
    fake_worker(monkeypatch, agent, 'rpc("done")')
    session, _ = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "protocol_failure"
    assert session.submission is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "ending",
    [
        "direct_submit",
        "worker_error",
        "no_report",
        "empty_report",
        "unfinished",
        "exhausted",
        "early_candidate",
    ],
)
async def test_candidate_cannot_bypass_native_completion(
    monkeypatch, sandbox_workspace, ending
):
    agent = DenarioAgent()
    program = r"""
from pathlib import Path
invalid = rpc("proposal", arguments={"answer":42}, id="invalid", role="researcher", step=2)
assert invalid["status"] == "invalid_args"
rpc("proposal", arguments={"answer":"preliminary"}, id="early", role="engineer", step=1)
"""
    if ending == "direct_submit":
        program += 'rpc("tool", name="submit_answer", arguments={"answer":"early"}, id="submit")'
    elif ending == "worker_error":
        program += 'rpc("error", error="native result extraction failed")'
    else:
        program += r"""
context = {"current_status":"completed", "current_plan_step_number":2,
    "number_of_steps_in_plan":2, "n_attempts":0, "max_n_attempts":10}
"""
        if ending != "early_candidate":
            program += 'rpc("proposal", arguments={"answer":"final"}, id="final", role="researcher", step=2)\n'
        if ending != "no_report":
            program += 'report = Path(config["project_dir"], "input_files", "results.md")\nreport.parent.mkdir()\n'
            program += (
                f"report.write_text({'' if ending == 'empty_report' else REPORT!r})\n"
            )
        if ending == "unfinished":
            program += 'context["current_status"] = "in progress"\n'
        if ending == "exhausted":
            program += 'context["n_attempts"] = 10\n'
        program += 'rpc("done", context=context)'
    fake_worker(monkeypatch, agent, program)
    session, _ = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == (
        "agent_failure" if ending == "worker_error" else "protocol_failure"
    )
    assert session.submission is None
    assert not any(a.action.is_submission for a in session.state.actions.values())


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
@pytest.mark.parametrize(
    "content",
    [
        "Analysis",
        None,
        [
            {"type": "text", "text": "Inspect"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AA", "detail": "low"},
            },
        ],
    ],
)
def test_speaker_attribution_without_provider_name_field(role, content):
    source = {"role": role, "name": "researcher", "content": content}
    snapshot = json.dumps(source)
    chat = attributed_message(source)
    assert "name" not in chat
    assert "[Speaker: researcher]" in str(chat["content"])
    request = ResponsesAdapter().request({"model": "gpt-6.1-sol", "messages": [source]})
    assert len(request["input"]) == 1
    item = request["input"][0]
    assert item["role"] == role
    assert "name" not in item
    assert "[Speaker: researcher]" in str(item["content"])
    if isinstance(content, list):
        assert item["content"][-1] == {
            "type": "input_image",
            "image_url": content[-1]["image_url"]["url"],
            "detail": "low",
        }
    assert json.dumps(source) == snapshot


def test_speaker_attribution_preserves_cached_function_and_reasoning_items():
    adapter = ResponsesAdapter()
    original = [
        {
            "type": "reasoning",
            "id": "rs-1",
            "summary": [],
            "encrypted_content": "opaque",
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "measure",
            "arguments": "{}",
        },
    ]
    completion = adapter.completion(
        {"id": "response-1", "model": "gpt-6.1-sol", "output": original}
    )
    message = {**completion["choices"][0]["message"], "name": "engineer"}
    result = adapter.request(
        {
            "model": "gpt-6.1-sol",
            "messages": [
                message,
                {
                    "role": "tool",
                    "name": "measure",
                    "tool_call_id": "call-1",
                    "content": "42",
                },
            ],
        }
    )
    assert result["input"] == [
        {"role": "assistant", "content": "[Speaker: engineer]"},
        *original,
        {"type": "function_call_output", "call_id": "call-1", "output": "42"},
    ]
    # The uncached tool-only path also needs attribution and unchanged call IDs.
    uncached = ResponsesAdapter().request(
        {"model": "gpt-6.1-sol", "messages": [message]}
    )
    assert uncached["input"][0] == {
        "role": "assistant",
        "content": "[Speaker: engineer]",
    }
    assert uncached["input"][1]["call_id"] == "call-1"
    assert uncached["input"][1]["name"] == "measure"


@pytest.mark.anyio
async def test_quota_failure_stops_retries(monkeypatch, sandbox_workspace):
    agent = DenarioAgent()
    fake_worker(
        monkeypatch,
        agent,
        """
rpc("model_start", id="quota")
rpc("model_end", id="quota", agent="idea", input=[],
    message={"role":"assistant","content":"Quota exhausted"}, usage={}, error_code="insufficient_quota")
model(2)
""",
    )
    session, _ = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "budget_exhausted"
    assert result.usage.llm_calls == 1


def test_streamed_function_call_and_reasoning_usage():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Checking.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "record_plan",
                                    "arguments": '{"plan":',
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"measure"}'}}
                        ]
                    },
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ]
    body = (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    ).encode()
    message, usage = completion_event(body, streaming=True)
    assert message["content"] == "Checking."
    assert message["tool_calls"][0]["id"] == "call-1"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "plan": "measure"
    }
    assert usage["reasoning_tokens"] == 2


def test_gateway_preserves_schemas_and_explicit_model_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    bridge = Bridge(
        {
            "model": "gpt-6.1-sol",
            "api_endpoint": "https://provider.invalid/v1",
            "model_settings": {
                "temperature": None,
                "reasoning_effort": "low",
                "max_completion_tokens": 1234,
            },
        },
        StringIO(),
        StringIO(),
    )
    events = []
    monkeypatch.setattr(
        bridge, "rpc", lambda kind, **fields: events.append((kind, fields))
    )
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "test", "schema": {"type": "object"}},
    }
    tools = [
        {
            "type": "function",
            "function": {"name": "measure", "parameters": {"type": "object"}},
        }
    ]

    def respond(request):
        assert str(request.url) == "https://provider.invalid/v1/responses"
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "gpt-6.1-sol"
        assert body["text"]["format"] == {
            "type": "json_schema",
            **schema["json_schema"],
        }
        assert body["tools"] == [
            {"type": "function", "strict": False, **tools[0]["function"]}
        ]
        assert body["reasoning"] == {"effort": "low"}
        assert body["max_output_tokens"] == 1234
        assert "temperature" not in body
        assert "top_p" not in body
        return httpx.Response(
            200,
            json={
                "id": "resp-test",
                "model": "gpt-6.1-sol",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "result"}],
                    }
                ],
                "usage": {"input_tokens": 5},
            },
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        "corral.agents.denario._bridge.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    try:
        status, _, _ = bridge.model(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "task"}],
                "tools": tools,
                "response_format": schema,
                "temperature": 0.00001,
                "top_p": 0.05,
                "reasoning_effort": "medium",
            }
        )
    finally:
        bridge.server.server_close()
    assert status == 200
    assert [kind for kind, _ in events] == ["model_start", "model_end"]
    assert events[1][1]["input"] == [{"role": "user", "content": "task"}]
    assert events[0][1]["endpoint"] == "responses"
    assert events[0][1]["request"]["tools"][0]["name"] == "measure"
    assert events[1][1]["provider_response"]["id"] == "resp-test"
    assert events[0][1]["agent"] == "starting"
    assert "sk-test" not in json.dumps(events)


@pytest.mark.parametrize("streaming", [False, True])
def test_responses_preserves_text_tools_reasoning_and_defaults(streaming):
    adapter = ResponsesAdapter()
    reasoning = {
        "type": "reasoning",
        "id": "rs-test",
        "summary": [],
        "encrypted_content": "opaque",
    }
    raw = {
        "id": "resp-test",
        "model": "gpt-6.1-sol",
        "status": "completed",
        "output": [
            reasoning,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will measure."}],
            },
            {
                "type": "function_call",
                "call_id": "call-a",
                "name": "measure",
                "arguments": '{"sample":"a"}',
            },
            {
                "type": "function_call",
                "call_id": "call-b",
                "name": "measure",
                "arguments": '{"sample":"b"}',
            },
        ],
        "usage": {
            "input_tokens": 9,
            "output_tokens": 7,
            "total_tokens": 16,
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }
    completion = adapter.completion(raw)
    _, encoded = encode_completion(completion, streaming=streaming)
    message, usage = completion_event(encoded, streaming=streaming)
    assert message["content"] == "I will measure."
    assert [call["id"] for call in message["tool_calls"]] == ["call-a", "call-b"]
    assert usage["reasoning_tokens"] == 3
    request = adapter.request(
        {
            "model": "gpt-6.1-sol",
            "messages": [
                {"role": "system", "content": "You are the engineer."},
                message,
                {"role": "tool", "tool_call_id": "call-a", "content": "42"},
                {"role": "tool", "tool_call_id": "call-b", "content": "43"},
            ],
        }
    )
    assert "reasoning" not in request
    assert "temperature" not in request
    assert request["input"].count(reasoning) == 1
    assert request["input"][1:-2] == raw["output"]
    assert request["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-b",
        "output": "43",
    }
    assert [
        item["call_id"]
        for item in request["input"]
        if item.get("type") == "function_call"
    ] == ["call-a", "call-b"]


@pytest.mark.parametrize("failure", ["http", "transport", "invalid_json"])
def test_gateway_records_failed_requests(monkeypatch, failure):
    bridge = Bridge(
        {
            "model": "gpt-6.1-sol",
            "model_settings": {},
            "api_endpoint": "https://provider.invalid/v1",
        },
        StringIO(),
        StringIO(),
    )
    events = []
    monkeypatch.setattr(
        bridge, "rpc", lambda kind, **fields: events.append((kind, fields))
    )

    def respond(request):
        # The input was acknowledged by Corral before contacting the provider.
        assert events[0][0] == "model_start"
        if failure == "transport":
            raise httpx.ConnectError("unavailable", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, content=b"invalid provider response")
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    client_class = httpx.Client
    monkeypatch.setattr(
        "corral.agents.denario._bridge.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    try:
        bridge.model({"messages": [{"role": "user", "content": "task"}]})
    finally:
        bridge.server.server_close()
    assert [kind for kind, _ in events] == ["model_start", "model_end"]
    assert events[1][1]["id"] == events[0][1]["id"]
    assert events[1][1]["is_error"]
    assert events[1][1]["usage"] == {}


@pytest.mark.anyio
async def test_unexecuted_proposals_and_sdk_messages_are_persisted(
    monkeypatch, sandbox_workspace
):
    agent = DenarioAgent()
    fake_worker(
        monkeypatch,
        agent,
        """
calls = [{"id": "never-executed", "type": "function",
          "function": {"name": "measure", "arguments": '{"sample":"a"}'}}]
rpc("model_start", id="proposal", agent="engineer", request={"messages": []})
rpc("model_end", id="proposal", agent="engineer", input=[],
    message={"role":"assistant", "content":"Measure next.", "tool_calls":calls}, usage={})
rpc("event", event={"type":"message", "sender":"engineer", "recipient":"executor",
    "message":{"role":"assistant", "content":"SDK-generated handoff", "tool_calls":calls}})
rpc("error", error="Stopped before executing proposed tools")
""",
    )
    session, measurements = await make_session(agent)
    result = await agent.run_session(session)
    assert result.status == "agent_failure"
    assert not measurements
    state = await session._test_commit_store.materialize("main")
    messages = state.conversations[session.actor.run_id]
    output = next(m["model_output"] for m in messages if "model_output" in m)
    assert output["content"] == "Measure next."
    assert output["tool_calls"][0]["id"] == "never-executed"
    handoff = next(
        m["sdk_event"] for m in messages if m.get("name") == "denario.message"
    )
    assert handoff["sender"] == "engineer"
    assert handoff["recipient"] == "executor"
    assert handoff["message"]["content"] == "SDK-generated handoff"
    assert state.usage.llm_calls == 1
    assert state.usage.tool_calls == 0
    assert not state.actions


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
async def test_stopping_kills_worker_and_code_child(
    monkeypatch, sandbox_workspace, cancel
):
    agent = DenarioAgent(wall_clock_timeout_s=10 if cancel else 0.5)
    pid_path = sandbox_workspace / "pids.json"
    fake_worker(
        monkeypatch,
        agent,
        """
import os, subprocess, time
from pathlib import Path
rpc("model_start", id="in-flight", agent="researcher",
    request={"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "Measure sample a"}]})
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
Path("pids.json").write_text(json.dumps({"pids": [os.getpid(), child.pid], "runtime": os.environ["PYTHONPATH"]}))
time.sleep(120)
""",
    )
    session, _ = await make_session(agent)
    task = asyncio.create_task(agent.run_session(session))
    if cancel:
        try:
            for _ in range(100):
                if pid_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_path.exists()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert task.cancelled()
    else:
        result = await task
        assert result.status == "timeout"
    record = json.loads(pid_path.read_text())
    assert not Path(record["runtime"]).exists()
    state = await session._test_commit_store.materialize("main")
    events = [
        m["sdk_event"]
        for m in state.conversations[session.actor.run_id]
        if "sdk_event" in m
    ]
    assert [event["type"] for event in events] == ["model_request", "model_interrupted"]
    assert events[0]["request"]["messages"] == [
        {"role": "user", "content": "Measure sample a"}
    ]
    assert events[1]["request_id"] == events[0]["request_id"] == "in-flight"
    assert events[1]["is_error"]
    assert state.usage.llm_calls == 1
    assert state.usage.input_tokens == state.usage.output_tokens == 0
    # A killed child can briefly remain a zombie until the OS reaps it.
    for pid in record["pids"]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        assert "State:\tZ" in Path(f"/proc/{pid}/status").read_text()


PLAN = "\n- Step 1:\n\t* sub-task: Measure and analyze\n\t* agent in charge: engineer\n\t* instructions:\n\t\t- Measure sample a and analyze the observation.\n- Step 2:\n\t* sub-task: Interpret and report\n\t* agent in charge: researcher\n\t* instructions:\n\t\t- Check the result, propose the final answer, and write the results.\n"
REPORT = "# Results\nSample a was measured with the environment tool: measured:42. Python confirmed 6 * 7 = 42."
CODE = "import importlib.util\nassert importlib.util.find_spec('corral.core') is None\nfrom pathlib import Path\nPath('analysis.txt').write_text(str(6 * 7))\nprint(6 * 7)\n"


def test_native_sdk_records_functions_and_executor_failures(tmp_path):
    if (
        not os.getenv("CORRAL_TEST_DENARIO_NATIVE")
        or importlib.util.find_spec("denario") is None
    ):
        pytest.skip("Run with CORRAL_TEST_DENARIO_NATIVE=1 in the Denario Docker image")
    # Hooks are process-local. Exercise real AG2 methods without replacing the
    # provider or polluting the test runner's SDK classes with permanent hooks.
    program = r"""
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from autogen import ConversableAgent
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from corral.agents.denario._worker import install_native_hooks

events = []
bridge = SimpleNamespace(
    config={"model": "gpt-6.1-sol", "tools": []}, token="test", url="http://127.0.0.1:1", agent="",
    client_model="gpt-6.1-sol",
    rpc=lambda kind, **fields: events.append({"kind":kind, **fields}),
    event=lambda kind, **fields: events.append({"kind":kind, **fields}),
)
install_native_hooks(bridge)
# The actual caller holds an imported binding, not a lookup on vlm_utils.
import importlib
vlm = importlib.import_module("cmbagent.vlm_utils")
functions = importlib.import_module("cmbagent.functions")
assert functions.call_external_plot_debugger is vlm.call_external_plot_debugger
import openai
plot_requests = []
def parse(**kwargs):
    plot_requests.append(kwargs)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=SimpleNamespace(fixes=["fix labels"])))])
openai.OpenAI = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
# _configure_plot_models captures OpenAI on installation; reinstall with the stub.
from corral.agents.denario._worker import _configure_plot_models
_configure_plot_models(bridge)
assert functions.call_external_plot_debugger("task", "analysis", ["labels"], "plot()") == ["fix labels"]
assert plot_requests[0]["model"] == "gpt-6.1-sol"
assert plot_requests[0]["response_format"] is vlm.PlotDebuggerResponse
agent = ConversableAgent("executor", llm_config=False, human_input_mode="NEVER")
def fail():
    raise ValueError("native function failed")
async def succeed():
    return "async success"
agent.register_function({"fail":fail, "succeed":succeed})
assert agent.execute_function({"name":"fail", "arguments":"{}"})[0] is False
assert asyncio.run(agent.a_execute_function({"name":"succeed", "arguments":"{}"}))[0] is True
executor = LocalCommandLineCodeExecutor(work_dir=sys.argv[1])
assert executor.execute_code_blocks([CodeBlock(language="python", code="raise ValueError('code failed')")]).exit_code != 0
# A missing parent directory raises before the subprocess can start.
try:
    executor.execute_code_blocks([CodeBlock(language="python", code="# filename: missing/subdir/test.py\nprint(1)")])
except FileNotFoundError:
    pass
else:
    raise AssertionError("expected code executor exception")
Path(sys.argv[1], "events.json").write_text(json.dumps(events))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OPENAI_API_KEY": "sk-test"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    messages = [
        m
        for event in json.loads((tmp_path / "events.json").read_text())
        for m in event.get("messages", [])
    ]
    calls = [call for m in messages for call in m.get("tool_calls", [])]
    results = [m for m in messages if m.get("role") == "tool"]
    assert len(calls) == len(results) == 4
    assert {c["id"] for c in calls} == {m["tool_call_id"] for m in results}
    assert [m["is_error"] for m in results] == [True, False, True, True]
    assert "native function failed" in results[0]["content"]
    assert results[1]["content"] == "async success"
    assert "code failed" in results[2]["content"]
    assert "FileNotFoundError" in results[3]["content"]


class NativeResponses:
    """Script only provider outputs, retaining the real Denario/AG2 workflow."""

    def __init__(self, model="gpt-6.1-sol", scenario="success"):
        self.model = model
        self.scenario = scenario
        self.injected_roles = set()
        self.batch_tail = False
        self.requests = []
        self.outputs = []
        self.measured = False
        self.code_requested = False
        self.proposed_early = False
        self.proposed_final = False
        self.engineer_done = False
        self.researcher_done = False
        self.planning = (
            r"\begin{IDEA}Measure sample a.\end{IDEA}",
            r"\begin{CRITIC}Use real observations.\end{CRITIC}",
            r"\begin{IDEA}Measure sample a with the environment tool.\end{IDEA}",
            r"\begin{CRITIC}Check the measurement before submission.\end{CRITIC}",
            r"\begin{IDEA}Measure sample a and check the observation.\end{IDEA}",
            r"\begin{CRITIC}Submit the observed result.\end{CRITIC}",
            r"\begin{IDEA}Measure sample a and submit.\end{IDEA}",
            r"\begin{METHODS}Measure sample a and submit the observation.\end{METHODS}",
        )

    def respond(self, body):
        self.requests.append(body)
        if "input" in body:
            messages = []
            for item in body["input"]:
                if "role" in item:
                    messages.append(item)
                elif item["type"] == "function_call":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": item["call_id"],
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": item["arguments"],
                                    },
                                }
                            ],
                        }
                    )
                elif item["type"] == "function_call_output":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["call_id"],
                            "content": item["output"],
                        }
                    )
            fmt = body.get("text", {}).get("format", {})
            body = {
                **body,
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {k: v for k, v in tool.items() if k != "type"},
                    }
                    for tool in body.get("tools", [])
                ],
                "response_format": {"type": "json_schema", "json_schema": fmt}
                if fmt.get("type") == "json_schema"
                else fmt,
            }
        assert body["model"] == self.model
        assert not (
            {
                "temperature",
                "top_p",
                "reasoning_effort",
                "reasoning",
                "max_tokens",
                "max_completion_tokens",
            }
            & body.keys()
        )
        if body.get("stream"):
            assert body["stream_options"]["include_usage"]
        if len(self.requests) <= len(self.planning):
            return {
                "role": "assistant",
                "content": self.planning[len(self.requests) - 1],
            }
        names = {tool["function"]["name"] for tool in body.get("tools", [])}
        assert not names & {"call_tool", "denario.execute_code"}
        arguments = None
        if "record_plan" in names:
            name, arguments = (
                "record_plan",
                {"plan_suggestion": PLAN, "number_of_steps_in_plan": 2},
            )
        elif "record_review" in names:
            name, arguments = (
                "record_review",
                {"plan_review": "Proceed with real measurements."},
            )
        elif "record_status_starter" in names:
            name, arguments = "record_status_starter", {}
        elif "post_execution_transfer" in names:
            if self.engineer_done:
                self.researcher_done = True
            else:
                self.engineer_done = True
            name, arguments = (
                "post_execution_transfer",
                {
                    "next_agent_suggestion": "control",
                    "execution_status": "success",
                    "fix_suggestion": None,
                },
            )
        elif "record_status" in names:
            name, arguments = (
                "record_status",
                {
                    "current_status": "completed"
                    if self.researcher_done
                    else "in progress",
                    "current_plan_step_number": 2 if self.engineer_done else 1,
                    "current_sub_task": "Interpret and report"
                    if self.engineer_done
                    else "Measure and analyze",
                    "current_instructions": "Complete the assigned step.",
                    "agent_for_sub_task": "researcher"
                    if self.engineer_done
                    else "engineer",
                },
            )
        elif "measure" in names:
            assert "submit_answer" not in names
            assert "propose_answer" in names
            researcher = (
                "You are the researcher agent" in body["messages"][0]["content"]
            )
            role = "researcher" if researcher else "engineer"
            should_fail = self.scenario in {"retry_both", "exhaust_" + role} or (
                not researcher and self.scenario in {"invalid_args", "batch_exhaust"}
            )
            if should_fail and role not in self.injected_roles:
                self.injected_roles.add(role)
                name = "measure"
                arguments = (
                    {} if self.scenario == "invalid_args" else {"sample": "missing"}
                )
                self.batch_tail = self.scenario == "batch_exhaust"
            elif researcher:
                if not self.proposed_final:
                    self.proposed_final = True
                    name, arguments = "propose_answer", {"answer": "measured:42"}
                else:
                    return {"role": "assistant", "content": REPORT}
            elif not self.measured:
                name, arguments = "measure", {"sample": "a"}
                self.measured = True
            elif not self.proposed_early:
                self.proposed_early = True
                name, arguments = (
                    "propose_answer",
                    {"answer": "preliminary; still needs interpretation"},
                )
            elif not self.code_requested:
                self.code_requested = True
                return {
                    "role": "assistant",
                    "content": "Analyze the observation using Python.\n```python\n"
                    + CODE
                    + "```",
                }
            else:
                raise AssertionError(
                    "Engineer should have handed control to researcher"
                )
        if arguments is not None:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{len(self.requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            if self.batch_tail:
                self.batch_tail = False
                message["tool_calls"].append(
                    {
                        "id": f"call-{len(self.requests)}-second",
                        "type": "function",
                        "function": {
                            "name": "measure",
                            "arguments": json.dumps({"sample": "must-not-run"}),
                        },
                    }
                )
            return message
        schema = (
            body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        )
        props = schema.get("properties", {})
        if "sub_tasks" in props:
            content = json.dumps(
                {
                    "sub_tasks": [
                        {
                            "sub_task": "Measure and analyze",
                            "sub_task_agent": "engineer",
                            "bullet_points": [
                                "Measure sample a and analyze the observation."
                            ],
                        },
                        {
                            "sub_task": "Interpret and report",
                            "sub_task_agent": "researcher",
                            "bullet_points": [
                                "Check the result, propose the final answer, and write the results."
                            ],
                        },
                    ]
                }
            )
        elif "recommendations" in props:
            content = json.dumps(
                {"recommendations": ["Proceed with real measurements."]}
            )
        elif "python_code" in props:
            content = json.dumps(
                {
                    "filename": "measure.py",
                    "relative_path": None,
                    "code_explanation": "Measure and submit.",
                    "modification_summary": None,
                    "python_code": CODE,
                }
            )
        elif "markdown_block" in props:
            self.researcher_done = True
            content = json.dumps(
                {"filename": "research_results.md", "markdown_block": REPORT}
            )
        elif "You are the engineer agent." in body["messages"][0]["content"]:
            content = "Measure and submit using Python.\n```python\n" + CODE + "```"
        else:
            content = PLAN
        return {"role": "assistant", "content": content}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "scenario"),
    [
        ("openai/gpt-6.1-sol", "success"),
        ("anthropic/claude-opus-5-5", "success"),
        ("openrouter/moonshotai/kimi-k3.1-pro", "success"),
        ("openai/gpt-6.1-sol", "retry_both"),
        ("openai/gpt-6.1-sol", "exhaust_engineer"),
        ("openai/gpt-6.1-sol", "exhaust_researcher"),
        ("openai/gpt-6.1-sol", "batch_exhaust"),
        ("openai/gpt-6.1-sol", "invalid_args"),
    ],
)
async def test_full_native_sdk_with_mock_provider(
    monkeypatch, sandbox_workspace, model, scenario
):
    if (
        not os.getenv("CORRAL_TEST_DENARIO_NATIVE")
        or importlib.util.find_spec("denario") is None
    ):
        pytest.skip("Run with CORRAL_TEST_DENARIO_NATIVE=1 in the Denario Docker image")
    from corral.persistence import SQLiteCommitStore

    # Use on-disk commits so the final assertions also exercise persistence.
    monkeypatch.setattr(
        "tests.agents.commit_session.SQLiteCommitStore",
        lambda _path, execution_id: SQLiteCommitStore(
            str(sandbox_workspace / "commits.sqlite"), execution_id
        ),
    )
    provider, native_model = model.split("/", 1)
    responses = NativeResponses(native_model, scenario)
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                assert self.path == (
                    "/v1/responses" if "input" in body else "/v1/chat/completions"
                )
                assert (
                    provider != "openai" or "input" in body or not body.get("tools")
                ), "gpt-6.1-sol tools require Responses"
                assert self.headers["Authorization"] == f"Bearer {provider}-test"
                assert all(
                    "name" not in item
                    for item in body.get("input", body.get("messages", []))
                    if "role" in item
                ), "Provider message objects reject the SDK name field"
                if provider == "anthropic":
                    assert body.get("tool_choice") in (None, "auto", "none")
                if provider == "openrouter":
                    assert body["provider"] == {
                        "zdr": True,
                        "data_collection": "deny",
                        "only": ["fireworks"],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                    assert "parallel_tool_calls" not in body
                message = responses.respond(body)
                responses.outputs.append(message)
                usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
                common = {"id": "chatcmpl-test", "created": 1, "model": body["model"]}
                if "input" in body:
                    output = [
                        {
                            "type": "reasoning",
                            "id": f"rs-{len(responses.requests)}",
                            "summary": [],
                        }
                    ]
                    if message.get("content"):
                        output.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": message["content"]}
                                ],
                            }
                        )
                    output.extend(
                        {
                            "type": "function_call",
                            "call_id": call["id"],
                            **call["function"],
                        }
                        for call in message.get("tool_calls", [])
                    )
                    payload = json.dumps(
                        {
                            "id": f"resp-{len(responses.requests)}",
                            "model": body["model"],
                            "status": "completed",
                            "output": output,
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                                "output_tokens_details": {"reasoning_tokens": 2},
                            },
                        }
                    ).encode()
                    content_type = "application/json"
                elif body.get("stream"):
                    parts = [
                        {
                            **common,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {"index": 0, "delta": message, "finish_reason": None}
                            ],
                        },
                        {
                            **common,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                            "usage": usage,
                        },
                    ]
                    payload = (
                        "".join("data: " + json.dumps(part) + "\n\n" for part in parts)
                        + "data: [DONE]\n\n"
                    ).encode()
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(
                        {
                            **common,
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": message,
                                    "finish_reason": "tool_calls"
                                    if message.get("tool_calls")
                                    else "stop",
                                }
                            ],
                            "usage": usage,
                        }
                    ).encode()
                    content_type = "application/json"
                self.send_response(200)
            except Exception as exc:
                errors.append(str(exc))
                payload, content_type = (
                    b'{"error":{"message":"Mock provider failed"}}',
                    "application/json",
                )
                self.send_response(500)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in ("openai", "anthropic", "openrouter"):
        monkeypatch.setenv(key.upper() + "_API_KEY", key + "-test")
    agent = DenarioAgent(
        model=model,
        system_prompt="CORRAL_SYSTEM_WRAPPER",
        user_prompt="CORRAL_USER_WRAPPER {{task_guide}}",
        openrouter_zdr=provider == "openrouter",
        openrouter_provider={
            "only": ["fireworks"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if provider == "openrouter"
        else None,
        api_endpoint=f"http://127.0.0.1:{server.server_port}/v1",
        max_n_attempts=2
        if scenario == "retry_both"
        else 1
        if scenario != "success"
        else 10,
        wall_clock_timeout_s=180,
    )
    session, measurements = await make_session(agent)
    try:
        result = await agent.run_session(session)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    log = Path(result.metadata["project_dir"]) / "denario-worker.log"
    recovery = [
        m["sdk_event"]
        for m in session.messages
        if m.get("name") == "denario.environment_tool_recovery"
    ]
    if scenario.startswith("exhaust_") or scenario == "batch_exhaust":
        assert result.status == "agent_failure", (
            f"{result.error}\n{errors}\n{log.read_text()[-12000:]}"
        )
        assert session.submission is None
        assert measurements == (
            ["a", "missing"] if scenario == "exhaust_researcher" else ["missing"]
        )
        assert len(recovery) == 1
        assert recovery[0]["n_attempts"] == recovery[0]["max_n_attempts"] == 1
        assert recovery[0]["target"] == "terminator"
        assert not any(a.action.is_submission for a in session.state.actions.values())
        if scenario == "batch_exhaust":
            blocked = [m for m in session.messages if m.get("not_executed")]
            assert len(blocked) == 1
            assert "failure budget" in blocked[0]["content"]
        assert not errors
        return
    assert result.status == "completed", (
        f"{result.error}\n{errors}\n{log.read_text()[-12000:]}"
    )
    assert not errors
    description = (
        Path(result.metadata["project_dir"]) / "input_files" / "data_description.md"
    ).read_text()
    task_text = description.split("\nTask:\n", 1)[1].split(
        "\nAvailable environment tools (JSON schemas):\n", 1
    )[0]
    assert task_text == session.prompt
    assert "Measure sample a and submit the result." in task_text
    assert "CORRAL_SYSTEM_WRAPPER" not in description
    assert "CORRAL_USER_WRAPPER" not in description
    assert "CORRAL_SYSTEM_WRAPPER" not in json.dumps(responses.requests)
    assert "CORRAL_USER_WRAPPER" not in json.dumps(responses.requests)
    assert measurements == (
        ["missing", "a", "missing"] if scenario == "retry_both" else ["a"]
    )
    if scenario == "retry_both":
        assert [event["step"] for event in recovery] == [1, 2]
        assert [event["n_attempts"] for event in recovery] == [1, 1]
        assert all(event["target"] == "control" for event in recovery)
        assert all(event["status"] == "execution_error" for event in recovery)
    else:
        assert recovery == []
    assert result.answer == "measured:42"
    assert any(
        "[Speaker: engineer]" in json.dumps(request) for request in responses.requests
    )
    assert any(
        "[Speaker: researcher]" in json.dumps(request) for request in responses.requests
    )
    report = Path(result.metadata["project_dir"]) / "input_files" / "results.md"
    assert report.read_text() == REPORT
    submissions = [
        a.action for a in session.state.actions.values() if a.action.is_submission
    ]
    assert len(submissions) == 1
    assert submissions[0].metadata["denario_results"] == str(report)
    finished = next(
        i for i, m in enumerate(session.messages) if m.get("name") == "denario.finished"
    )
    submitted = next(
        i
        for i, m in enumerate(session.messages)
        if any(
            c["function"]["name"] == "submit_answer" for c in m.get("tool_calls", [])
        )
    )
    assert finished < submitted
    steps = [
        m["sdk_event"]["context"]
        for m in session.messages
        if m.get("name") == "denario.native_step_finished"
    ]
    assert steps[-1]["current_status"] == "completed"
    assert steps[-1]["current_plan_step_number"] == 2
    assert "Step 1" in steps[-1]["previous_steps_execution_summary"]
    assert result.usage.llm_calls == len(responses.requests)
    assert result.usage.input_tokens == 10 * len(responses.requests)
    assert result.usage.reasoning_tokens == 2 * len(responses.requests)
    assert any(message.get("name") == "denario.planner" for message in session.messages)
    assert any(
        message.get("role") == "tool" and message.get("name") == "record_plan"
        for message in session.messages
    )
    assert any(
        call["function"]["name"] == "denario.execute_code"
        for message in session.messages
        for call in message.get("tool_calls", [])
    )
    analysis = list(Path(result.metadata["project_dir"]).rglob("analysis.txt"))
    assert len(analysis) == 1
    assert analysis[0].read_text() == "42"
    assert any(
        message.get("role") == "tool"
        and message.get("name") == "denario.execute_code"
        and message.get("is_error") is False
        for message in session.messages
    )
    # Reconstruct from commits, not the live session or worker's log. Match
    # every request/response at the provider boundary, including all tool IDs.
    await session._test_commit_store.aclose()
    async with SQLiteCommitStore(
        str(sandbox_workspace / "commits.sqlite"), session.execution_id
    ) as reopened:
        state = await reopened.materialize("main")
    messages = state.conversations[session.actor.run_id]
    requests = [
        m["sdk_event"] for m in messages if m.get("name") == "denario.model_request"
    ]
    outputs = [m for m in messages if "model_output" in m]
    assert result.metadata["denario_iterations"] == 4
    assert sum(m["name"] == "denario.idea" for m in outputs) == 7
    assert sum(m["name"] == "denario.method" for m in outputs) == 1
    assert [event["request"] for event in requests] == responses.requests
    assert [m["model_output"] for m in outputs] == responses.outputs
    assert [m["model_input"] for m in outputs] == [
        body.get("input", body.get("messages")) for body in responses.requests
    ]
    # The short user handoff is not the whole context: upstream Denario must
    # carry its generated artifacts and recorded plan into execution prompts.
    for name in ("denario.planner", "denario.engineer"):
        role_outputs = [m for m in outputs if m.get("name") == name]
        assert role_outputs
        for message in role_outputs:
            system = json.dumps(
                [m for m in message["model_input"] if m.get("role") == "system"]
            )
            assert "Measure sample a and submit." in system  # final idea
            assert "Measure sample a and submit the observation." in system  # method
            if name == "denario.engineer":
                assert "Measure and analyze" in system  # recorded plan
                assert "Current sub-task" in system
    if provider == "openai":
        assert any(event["endpoint"] == "responses" for event in requests)
        assert any(m.get("provider_response", {}).get("output") for m in outputs)
    assert [m["request_id"] for m in outputs] == [
        event["request_id"] for event in requests
    ]
    calls = [call for m in messages for call in m.get("tool_calls", [])]
    results = [m for m in messages if m.get("role") == "tool"]
    assert len(calls) == len({call["id"] for call in calls})
    assert len(results) == len({m["tool_call_id"] for m in results})
    assert {call["id"] for call in calls} == {m["tool_call_id"] for m in results}
    proposed = {
        call["id"] for m in responses.outputs for call in m.get("tool_calls", [])
    }
    assert proposed <= {call["id"] for call in calls}
    assert state.usage.tool_calls == len(calls)
    assert state.usage.llm_calls == len(responses.requests)
    sdk_messages = [
        m["sdk_event"] for m in messages if m.get("name") == "denario.message"
    ]
    assert sdk_messages
    assert all(m["sender"] and m["recipient"] for m in sdk_messages)
    assert any(m.get("recipients") for m in sdk_messages)
    # Native formatter output is transformed by the SDK before execution.
    assert any(
        "# filename: measure.py" in str(m["message"].get("content"))
        for m in sdk_messages
    )
