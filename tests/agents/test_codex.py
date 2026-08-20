"""Tests for the Codex native session adapter."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

pytest.importorskip("openai_codex")

from corral.agents import CodexAgent
from corral.agents.codex import HarnessRunResult
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
    execution_workspace = None
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
async def test_codex_uses_session_mcp_and_returns_typed_outcome(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = {}

    def fake_run(self, *, run, **kwargs):
        captured.update(kwargs)
        captured["max_sdk_turns"] = run.metadata["max_sdk_turns"]
        run.messages.append({"role": "assistant", "content": "42"})
        run.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
        run.result = HarnessRunResult(
            status="success", answer="42", metadata={"session_id": "codex-1"}
        )
        return "42"

    monkeypatch.setattr(CodexAgent, "_execute_harness", fake_run)
    agent = CodexAgent(model="gpt-test", system_prompt="system")
    outcome = await agent.run_session(Session("42", "submitted"))

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert outcome.usage.input_tokens == 10
    assert outcome.usage.llm_calls == 1
    assert outcome.metadata["session_id"] == "codex-1"
    assert captured["mcp_url"] == "http://127.0.0.1:1234/mcp"
    assert captured["max_sdk_turns"] == 1
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


def test_codex_extracts_usage_from_nested_total():
    agent = CodexAgent(model="gpt-test", system_prompt="system")

    usage = agent._usage(
        SimpleNamespace(
            total=SimpleNamespace(
                input_tokens=13,
                output_tokens=5,
                reasoning_output_tokens=2,
            )
        ),
        llm_calls=1,
    )

    assert usage.input_tokens == 13
    assert usage.output_tokens == 5
    assert usage.reasoning_tokens == 2
    assert usage.llm_calls == 1


@pytest.mark.anyio()
async def test_codex_plain_text_answer_is_not_a_submission(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run(self, *, run, **_kwargs):
        run.result = HarnessRunResult(status="success", answer="42")
        return "42"

    monkeypatch.setattr(CodexAgent, "_execute_harness", fake_run)
    outcome = await CodexAgent(system_prompt="system").run_session(Session())

    assert outcome.status == "protocol_failure"
    assert "without calling submit_answer" in outcome.error


@pytest.mark.anyio()
async def test_codex_maps_native_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run(self, *, run, **_kwargs):
        run.result = HarnessRunResult(status="timeout", error="deadline")
        return "Error solving the task: deadline"

    monkeypatch.setattr(CodexAgent, "_execute_harness", fake_run)
    outcome = await CodexAgent(system_prompt="system").run_session(Session())

    assert outcome.status == "timeout"
    assert outcome.error == "deadline"


def test_codex_configuration_exposes_only_task_mcp_tools():
    config = CodexAgent(system_prompt="system")._render_config_toml(
        "http://127.0.0.1/capability/mcp", ["read_file", "submit_answer"]
    )

    assert 'web_search = "disabled"' in config
    assert "shell_tool = false" in config
    assert "unified_exec = false" in config
    assert "apps = false" in config
    assert "multi_agent = false" in config
    assert "view_image = false" in config
    assert 'enabled_tools = ["read_file", "submit_answer"]' in config


@pytest.mark.anyio()
async def test_codex_never_receives_the_physical_task_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    task_workspace = tmp_path / "task-a"
    sibling_workspace = tmp_path / "task-b"
    task_workspace.mkdir()
    sibling_workspace.mkdir()
    (sibling_workspace / "secret.txt").write_text("sibling", encoding="utf-8")
    captured = {}

    def fake_turn(self, *, workspace, **_kwargs):
        captured["cwd"] = workspace
        captured["exists_during_run"] = workspace.is_dir()
        return "42"

    monkeypatch.setattr(CodexAgent, "_execute_codex_turn", fake_turn)
    session = Session()
    session.execution_workspace = str(task_workspace)
    outcome = await CodexAgent(system_prompt="system").run_session(session)

    cwd = Path(captured["cwd"])
    assert captured["exists_during_run"] is True
    assert cwd != task_workspace
    assert tmp_path not in cwd.parents
    assert not cwd.exists()
    assert outcome.metadata["workspace_access"] == "mcp_only"
    assert outcome.metadata["codex_cwd_is_execution_workspace"] is False


@pytest.mark.anyio()
async def test_codex_run_data_is_folded_into_final_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run(self, *, mcp_url, run, **_kwargs):
        async def submit() -> None:
            async with (
                streamablehttp_client(mcp_url) as streams,
                ClientSession(streams[0], streams[1]) as mcp_session,
            ):
                await mcp_session.initialize()
                result = await mcp_session.call_tool("submit_answer", {"answer": "42"})
                assert result.isError is False

        anyio.run(submit)
        run.messages.append(
            {"role": "assistant", "content": "Codex completed the task"}
        )
        run.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cached_input_tokens": 3,
        }
        run.metadata.update({"num_tool_calls": 1, "duration_ms": 25})
        run.result = self._result(run, "success", answer="42")
        return "42"

    monkeypatch.setattr(CodexAgent, "_execute_harness", fake_run)
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
    agent = CodexAgent(model="gpt-test", system_prompt="system")

    with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
        final = await TaskRuntime(store).run(
            agent,
            environment,
            execution_id="codex-state-fold",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_iterations=1,
        )

    assert final.submission == "42"
    assert final.usage.input_tokens == 10
    assert final.usage.output_tokens == 2
    assert final.usage.reasoning_tokens == 0
    assert final.tool_statistics == {"submit_answer": 1}
    assert any(
        message.get("content") == "Codex completed the task"
        for conversation in final.conversations.values()
        for message in conversation
    )
    assert final.runtime.metadata["agent_status"] == "completed"
    session_metadata = next(iter(final.agent_runs.values())).metadata
    assert session_metadata["harness_status"] == "success"
    assert session_metadata["num_tool_calls"] == 1
    assert not hasattr(agent, "harness_result")
    assert not hasattr(agent, "messages")
