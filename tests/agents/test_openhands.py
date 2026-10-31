"""Tests for the OpenHands native session adapter."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

pytest.importorskip("openhands.sdk")

from openhands.sdk.event import ActionEvent, ObservationEvent, SystemPromptEvent
from openhands.sdk.llm import MessageToolCall, TextContent
from openhands.sdk.mcp.definition import MCPToolAction

import corral.agents.openhands as openhands_module
from corral.agents import OpenHandsAgent
from corral.agents.openhands import HarnessRunResult
from corral.core.action import submit_answer_tool
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import ToolConnection, tool
from corral.persistence import SQLiteCommitStore
from corral.report.projection import _tool_statistics
from corral.runtime import TaskRuntime


def test_native_tool_durations_follow_call_ids_and_preserve_unknowns(monkeypatch):
    times = iter([10.0, 12.0, 13.0, 17.0])
    monkeypatch.setattr(openhands_module, "perf_counter", lambda: next(times))
    monkeypatch.setattr(OpenHandsAgent, "_event_text", lambda self, event: "output")
    agent = OpenHandsAgent(system_prompt="system")
    run = openhands_module._RunState()
    for call_id in ("a", "b"):
        agent._record_action_event(
            SimpleNamespace(tool_call_id=call_id, tool_name="terminal"), run
        )
    for call_id in ("b", "a", "unknown"):
        agent._record_observation_event(
            SimpleNamespace(
                tool_call_id=call_id,
                tool_name="terminal",
                observation=SimpleNamespace(is_error=call_id == "b"),
            ),
            run,
        )
    results = [m for m in run.messages if m["role"] == "tool"]
    assert [r["duration_ms"] for r in results] == [1000, 7000, None]
    assert results[0]["is_error"] is True
    assert not run.tool_started_at


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_openhands_uses_and_preserves_corral_workspace(monkeypatch, tmp_path):
    async def fake_conversation(self, *, cwd, **kwargs):
        assert cwd == str(tmp_path)
        (tmp_path / "native.txt").write_text("native output")
        return "done"

    monkeypatch.setattr(OpenHandsAgent, "_execute_openhands", fake_conversation)
    session = Session("done", "submitted")
    session.native_tool_workspace = str(tmp_path)
    outcome = await OpenHandsAgent(system_prompt="system").run_session(session)
    assert outcome.status == "completed"
    assert (tmp_path / "native.txt").read_text() == "native output"
    assert outcome.metadata["workspace_access"] == "corral_sandbox"


class Session:
    tool_connection = ToolConnection("mcp", "http://127.0.0.1:1234/mcp")
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

    async def record_message(self, message):
        self.messages.append(message)


def test_openhands_distinguishes_shell_isolation_from_real_task_tools():
    agent = OpenHandsAgent(api_key="test-key", system_prompt="system")
    instructions = agent._system_message_suffix(False)
    assert "does not imply those actions are simulated" in instructions
    assert "Follow the task-specific operating instructions" in instructions
    assert "sandboxed evaluation environment" not in instructions


def test_openhands_passes_retry_settings_without_changing_provider_or_reasoning():
    provider = {
        "only": ["together"],
        "allow_fallbacks": False,
        "zdr": True,
        "data_collection": "deny",
    }
    agent = OpenHandsAgent(
        model="openrouter/moonshotai/kimi-k3",
        api_key="test-key",
        system_prompt="system",
        litellm_extra_body={"provider": provider},
        num_retries=12,
        retry_min_wait=15,
        retry_max_wait=300,
        retry_multiplier=15.0,
    )
    llm = agent._make_llm()
    assert (
        llm.num_retries,
        llm.retry_min_wait,
        llm.retry_max_wait,
        llm.retry_multiplier,
    ) == (12, 15, 300, 15.0)
    assert llm.litellm_extra_body == {"provider": provider}
    assert (
        llm.reasoning_effort
        == openhands_module.LLM.model_fields["reasoning_effort"].default
    )


@pytest.mark.anyio
async def test_openhands_uses_only_the_iteration_run_limit(monkeypatch, tmp_path):
    captured = {"arun_calls": 0}

    class FakeConversation:
        def __init__(
            self,
            *,
            agent,
            mcp_tool_provider,
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
                    "mcp_tool_provider": mcp_tool_provider,
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

    monkeypatch.setattr(openhands_module, "LocalConversation", FakeConversation)
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
    assert captured["mcp_tool_provider"].tool_timeout_s == agent.tool_timeout_s
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


@pytest.mark.anyio
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


@pytest.mark.anyio
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


@pytest.mark.anyio
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


@pytest.mark.anyio
@pytest.mark.parametrize("native_error", [None, False, True])
async def test_openhands_run_data_is_folded_into_final_state(
    monkeypatch, tmp_path, native_error
):
    from openhands.sdk.event import ActionEvent, ObservationEvent
    from openhands.sdk.llm import MessageToolCall, TextContent
    from openhands.tools.terminal.definition import TerminalAction, TerminalObservation

    async def fake_run(self, *, mcp_url, run, **_kwargs):
        async with (
            streamablehttp_client(mcp_url) as streams,
            ClientSession(streams[0], streams[1]) as mcp_session,
        ):
            await mcp_session.initialize()
            result = await mcp_session.call_tool("submit_answer", {"answer": "42"})
            assert result.isError is False

        if native_error is not None:
            action = ActionEvent(
                thought=[],
                tool_name="terminal",
                tool_call_id="native-terminal",
                action=TerminalAction(command="pwd"),
                tool_call=MessageToolCall(
                    id="native-terminal",
                    name="terminal",
                    arguments='{"command":"pwd"}',
                    origin="completion",
                ),
                llm_response_id="response-1",
            )
            observation = ObservationEvent(
                tool_name="terminal",
                tool_call_id="native-terminal",
                action_id=action.id,
                observation=TerminalObservation(
                    command="pwd",
                    exit_code=1 if native_error else 0,
                    is_error=native_error,
                    content=[TextContent(text="shell output")],
                ),
            )
            self._record_event(action, run)
            self._record_event(observation, run)
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

    async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
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
    assert final.tool_statistics == (
        {"submit_answer": 1}
        if native_error is None
        else {"submit_answer": 1, "terminal": 1}
    )
    assert final.usage.tool_calls == (1 if native_error is None else 2)
    if native_error is not None:
        native = next(
            call
            for call in _tool_statistics(final)["tool_calls"]
            if call["tool_name"] == "terminal"
        )
        assert native["arguments"]["command"] == "pwd"
        assert "shell output" in native["result"]
        assert native["status"] == ("failed" if native_error else "completed")
        assert any(
            message.get("tool_call_id") == "native-terminal"
            for conversation in final.conversations.values()
            for message in conversation
        )
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


@pytest.mark.anyio
async def test_openhands_parallel_tools_flush_each_message_once(monkeypatch):
    session_loop = asyncio.get_running_loop()

    class SlowSession(Session):
        async def record_message(self, message):
            assert asyncio.get_running_loop() is session_loop
            await asyncio.sleep(0)
            await super().record_message(message)

    async def fake_run(self, *, run, **_kwargs):
        run.messages.insert(0, {"role": "system", "content": "system"})
        run.messages.append({"role": "assistant", "content": "Before tools"})
        # Two SDK MCP executors can both reach the persistence barrier while
        # its first commit is still awaiting I/O on the session loop.
        await asyncio.gather(
            asyncio.to_thread(lambda: asyncio.run(run.flush_messages())),
            asyncio.to_thread(lambda: asyncio.run(run.flush_messages())),
        )
        run.messages.append({"role": "assistant", "content": "Finished"})
        run.result = HarnessRunResult(status="success", answer="42")
        return "42"

    monkeypatch.setattr(OpenHandsAgent, "_execute_harness", fake_run)
    session = SlowSession("42", "submitted")
    outcome = await OpenHandsAgent(api_key="key", system_prompt="system").run_session(
        session
    )

    assert outcome.status == "completed"
    assert [message["content"] for message in session.messages] == [
        "system",
        "solve",
        "Before tools",
        "Finished",
    ]


@pytest.mark.anyio
async def test_openhands_persists_sdk_messages_between_runtime_tool_calls(
    monkeypatch, tmp_path
):
    """Exercise the real SDK MCP executor, session ledger, and cross-loop flush."""

    @tool
    def measure() -> str:
        """Return the measured answer."""
        return "42"

    class FakeConversation:
        def __init__(self, *, agent, mcp_tool_provider, callbacks, **_kwargs):
            self.agent = agent
            self.provider = mcp_tool_provider
            self.on_event = callbacks[0]
            self.client = None
            self.state = SimpleNamespace(
                execution_status=openhands_module.ConversationExecutionStatus.FINISHED
            )

        def send_message(self, _prompt):
            self.on_event(
                SystemPromptEvent(system_prompt=TextContent(text="system"), tools=[])
            )

        async def arun(self):
            self.client = await asyncio.to_thread(
                self.provider.create_tools, self.agent.mcp_config
            )
            tools = {item.name: item for item in self.client.tools}
            for name, arguments, thought in (
                ("measure", {}, "Measure before answering."),
                ("submit_answer", {"answer": "42"}, "The measured answer is 42."),
            ):
                action = MCPToolAction(data=arguments)
                event = ActionEvent(
                    thought=[TextContent(text=thought)],
                    action=action,
                    tool_name=name,
                    tool_call_id=f"sdk-{name}",
                    tool_call=MessageToolCall(
                        id=f"sdk-{name}",
                        name=name,
                        arguments=json.dumps(arguments),
                        origin="completion",
                    ),
                    llm_response_id=f"response-{name}",
                )
                self.on_event(event)
                observation = await asyncio.to_thread(tools[name], action)
                assert not observation.is_error
                self.on_event(
                    ObservationEvent(
                        action_id=event.id,
                        tool_name=name,
                        tool_call_id=event.tool_call_id,
                        observation=observation,
                    )
                )

        def close(self):
            if self.client is not None:
                self.client.sync_close()

    monkeypatch.setattr(openhands_module, "LocalConversation", FakeConversation)
    monkeypatch.setattr(
        OpenHandsAgent, "_make_llm", lambda self: SimpleNamespace(metrics=None)
    )
    monkeypatch.setattr(
        OpenHandsAgent,
        "_build_agent",
        lambda self, llm, mcp_url, **_kwargs: SimpleNamespace(
            mcp_config={
                "corral": openhands_module.MCPServer(
                    url=mcp_url, transport="streamable-http"
                )
            }
        ),
    )
    monkeypatch.setattr(
        OpenHandsAgent, "_extract_finish_message", lambda self, events: "Finished."
    )
    task = TaskDefinition(
        name="task",
        description="answer 42",
        tools=["measure"],
        scoring_fn=lambda answer: float(answer == "42"),
        submission_format={"answer": "string"},
        resolve_answer=False,
    )
    environment = Environment(
        "task",
        task,
        toolset=Toolset(pool={"measure": measure}, workspace_factory=None),
    )
    agent = OpenHandsAgent(model="openai/test", api_key="key", system_prompt="system")
    async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
        final = await asyncio.wait_for(
            TaskRuntime(store).run(
                agent,
                environment,
                execution_id="openhands-ordered-transcript",
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                max_iterations=3,
            ),
            timeout=20,
        )

    messages = next(iter(final.conversations.values()))
    assert final.submission == "42"
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["content"] == "Measure before answering."
    assert messages[3]["tool_calls"][0]["function"]["name"] == "measure"
    assert messages[5]["content"] == "The measured answer is 42."
    assert messages[6]["tool_calls"][0]["function"]["name"] == "submit_answer"
    assert messages[8]["content"] == "Finished."
    assert final.tool_statistics == {"measure": 1, "submit_answer": 1}
