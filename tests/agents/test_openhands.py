"""Tests for the OpenHands native session adapter."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

pytest.importorskip("openhands.sdk")

import corral.agents.openhands as openhands_module
from corral.agents import OpenHandsAgent
from corral.agents.openhands import HarnessRunResult
from corral.core.action import submit_answer_tool
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.persistence import SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class Session:
    prompt = "solve"
    tools = (
        {
            "type": "function",
            "function": {
                "name": "measure",
                "description": "measure",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        submit_answer_tool(),
    )
    surrender_allowed = False
    execution_id = "execution-1"
    iteration_limit = 1
    initial_state = SimpleNamespace(metadata=SimpleNamespace(task={"id": "task-1"}))

    def __init__(self, submission=None, submission_status=None):
        self.messages = []
        self.submission = submission
        self.submission_status = submission_status

    @asynccontextmanager
    async def open_mcp(self):
        yield SimpleNamespace(url="http://127.0.0.1:1234/mcp")

    async def record_message(self, message):
        self.messages.append(message)


@pytest.mark.anyio()
async def test_openhands_uses_only_the_iteration_run_limit(monkeypatch, tmp_path):
    captured = {"arun_calls": 0}

    class FakeConversation:
        def __init__(
            self,
            *,
            agent,
            callbacks,
            token_callbacks,
            workspace,
            max_iteration_per_run,
            stuck_detection,
            persistence_dir,
            visualizer,
        ):
            captured.update(
                {
                    "agent": agent,
                    "callbacks": callbacks,
                    "token_callbacks": token_callbacks,
                    "workspace": workspace,
                    "max_iteration_per_run": max_iteration_per_run,
                    "stuck_detection": stuck_detection,
                    "persistence_dir": persistence_dir,
                    "visualizer": visualizer,
                }
            )
            self.state = SimpleNamespace(
                execution_status=openhands_module.ConversationExecutionStatus.FINISHED,
                agent=SimpleNamespace(tools_map={}),
            )

        def send_message(self, prompt):
            captured["prompt"] = prompt

        async def arun(self):
            captured["arun_calls"] += 1

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(openhands_module, "Conversation", FakeConversation)
    monkeypatch.setattr(
        OpenHandsAgent,
        "_make_llm",
        lambda self: SimpleNamespace(metrics=None),
    )
    monkeypatch.setattr(OpenHandsAgent, "_build_agent", lambda self, *args, **kw: {})

    agent = OpenHandsAgent(
        model="openai/test",
        api_key="key",
        system_prompt="system",
    )
    run = openhands_module._RunState()
    await agent._execute_openhands(
        prompt="solve",
        mcp_url="http://127.0.0.1:1234/mcp",
        cwd=str(tmp_path),
        enable_surrender=False,
        iteration_limit=7,
        run=run,
    )

    assert captured["max_iteration_per_run"] == 7
    assert captured["stuck_detection"] is False
    assert len(captured["token_callbacks"]) == 1
    assert captured["token_callbacks"][0](object()) is None
    assert captured["arun_calls"] == 1
    assert captured["closed"] is True
    assert not hasattr(agent, "wall_clock_timeout_s")
    assert not hasattr(agent, "interrupt_grace_s")


@pytest.mark.parametrize("name", ["wall_clock_timeout_s", "interrupt_grace_s"])
def test_openhands_rejects_removed_time_based_run_limits(name):
    with pytest.raises(TypeError, match="configure limits in BenchmarkTaskMetadata"):
        OpenHandsAgent(system_prompt="system", **{name: 1.0})


def test_openhands_counts_each_sdk_completion_as_one_llm_call():
    agent = OpenHandsAgent(api_key="key", system_prompt="system")
    run = openhands_module._RunState(metadata={"sdk_turns": 1})
    metrics = SimpleNamespace(
        token_usages=[object(), object(), object()],
        accumulated_token_usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
        ),
        accumulated_cost=0.25,
    )

    agent._record_usage(SimpleNamespace(metrics=metrics), run)

    assert run.metadata["sdk_turns"] == 3
    assert run.usage == {
        "input_tokens": 12,
        "output_tokens": 4,
        "reasoning_tokens": 0,
        "total_tokens": 16,
    }
    assert agent._usage(metrics, llm_calls=3).llm_calls == 3


@pytest.mark.anyio()
async def test_openhands_uses_session_mcp_and_returns_typed_outcome(monkeypatch):
    captured = {}

    async def fake_run(self, *, run, **kwargs):
        captured.update(kwargs)
        run.messages.append({"role": "assistant", "content": "42"})
        run.usage = {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }
        run.result = HarnessRunResult(
            status="success", answer="42", metadata={"session_id": "oh-1"}
        )
        return "42"

    monkeypatch.setattr(OpenHandsAgent, "_execute_harness", fake_run)
    agent = OpenHandsAgent(model="openai/test", api_key="key", system_prompt="system")
    outcome = await agent.run_session(Session("42", "submitted"))

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.output_tokens == 2
    assert outcome.usage.llm_calls == 1
    assert outcome.metadata["session_id"] == "oh-1"
    assert captured["mcp_url"] == "http://127.0.0.1:1234/mcp"
    assert captured["iteration_limit"] == 1
    assert agent.__dict__.keys().isdisjoint(
        {
            "messages",
            "token_usage",
            "cumulative_token_usage",
            "harness_result",
            "_run_meta",
            "_available_tools",
        }
    )
    assert not hasattr(agent, "run")
    assert not hasattr(agent, "arun")
    assert not hasattr(agent, "step")
    assert not hasattr(agent, "arun_agent")


@pytest.mark.anyio()
async def test_openhands_plain_text_answer_is_not_a_submission(monkeypatch):
    async def fake_run(self, *, run, **_kwargs):
        run.result = HarnessRunResult(status="success", answer="42")
        return "42"

    monkeypatch.setattr(OpenHandsAgent, "_execute_harness", fake_run)
    outcome = await OpenHandsAgent(api_key="key", system_prompt="system").run_session(
        Session()
    )

    assert outcome.status == "protocol_failure"
    assert "without calling submit_answer" in outcome.error


@pytest.mark.anyio()
async def test_openhands_maps_tool_failure(monkeypatch):
    async def fake_run(self, *, run, **_kwargs):
        run.result = HarnessRunResult(status="tool_failure", error="MCP unavailable")
        return "Error solving the task: MCP unavailable"

    monkeypatch.setattr(OpenHandsAgent, "_execute_harness", fake_run)
    outcome = await OpenHandsAgent(api_key="key", system_prompt="system").run_session(
        Session()
    )

    assert outcome.status == "tool_failure"
    assert outcome.error == "MCP unavailable"


@pytest.mark.anyio()
async def test_openhands_run_data_is_folded_into_final_state(monkeypatch, tmp_path):
    async def fake_run(self, *, mcp_url, run, **_kwargs):
        async with (
            streamablehttp_client(mcp_url) as streams,
            ClientSession(streams[0], streams[1]) as mcp_session,
        ):
            await mcp_session.initialize()
            result = await mcp_session.call_tool("submit_answer", {"answer": "42"})
            assert result.isError is False

        run.messages.append(
            {"role": "assistant", "content": "OpenHands completed the task"}
        )
        run.usage = {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }
        run.metadata.update({"num_events": 4, "total_cost_usd": 0.01})
        run.result = self._result(run, "success", answer="42")
        return "42"

    monkeypatch.setattr(OpenHandsAgent, "_execute_harness", fake_run)
    task = TaskDefinition(
        name="task",
        description="answer 42",
        tools=[],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={}, workspace_factory=None),
    )
    agent = OpenHandsAgent(model="openai/test", api_key="key", system_prompt="system")

    with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id="openhands-state-fold",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.submission == "42"
    assert final.usage.input_tokens == 8
    assert final.usage.output_tokens == 2
    assert final.tool_statistics == {"submit_answer": 1}
    assert any(
        message.get("content") == "OpenHands completed the task"
        for conversation in final.conversations.values()
        for message in conversation
    )
    assert final.runtime.metadata["agent_status"] == "completed"
    session_metadata = next(iter(final.agent_runs.values())).metadata
    assert session_metadata["harness_status"] == "success"
    assert session_metadata["num_events"] == 4
    assert session_metadata["total_cost_usd"] == 0.01
    assert not hasattr(agent, "harness_result")
    assert not hasattr(agent, "messages")
