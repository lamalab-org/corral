"""Exercise direct Denario planning and real Corral session execution."""

import asyncio
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from tests.agents.commit_session import start_session

from corral import cli
from corral.agents import DenarioAgent
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.core.tool import tool


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake_sdk(monkeypatch):
    instances = []

    class FakeDenario:
        def __init__(self, *, project_dir):
            self.project = Path(project_dir)
            self.inputs = self.project / "input_files"
            self.inputs.mkdir()
            self.thread_id = threading.get_ident()
            instances.append(self)

        def set_data_description(self, description):
            self.description = description

        def get_idea_fast(self, *, llm, iterations):
            self.model = llm
            self.iterations = iterations
            (self.inputs / "idea.md").write_text(
                "Measure café sample.", encoding="utf-8"
            )

        def get_method_fast(self, *, llm):
            assert llm == self.model
            assert (self.inputs / "idea.md").exists()
            (self.inputs / "methods.md").write_text(
                "Measure, then submit.", encoding="utf-8"
            )

    monkeypatch.setitem(sys.modules, "denario", SimpleNamespace(Denario=FakeDenario))
    return SimpleNamespace(instances=instances, agent_class=FakeDenario)


class Session:
    tools = ()

    def __init__(self, state=None):
        self.prompt = [{"role": "user", "content": "Solve café task"}]
        self.state = state or {}
        self.messages = []

    def get_agent_state(self, namespace):
        assert namespace == "denario"
        return self.state

    async def set_agent_state(self, namespace, value):
        assert namespace == "denario"
        self.state = value

    async def record_message(self, message, *, llm_calls):
        assert llm_calls == 0
        self.messages.append(message)


def test_cli_preserves_executor_options_without_importing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "denario", None)
    agent = cli.create_agent(
        "DenarioAgent",
        model="openai/executor",
        api_endpoint="http://localhost:9000",
        temperature=0.5,
        agent_kwargs={
            "denario_model": "gpt-4o-mini",
            "denario_iterations": 3,
            "reasoning_effort": "low",
        },
    )
    assert isinstance(agent, DenarioAgent)
    assert agent.model == "openai/executor"
    assert agent.api_endpoint == "http://localhost:9000"
    assert agent.temperature == 0.5
    assert agent.kwargs == {"reasoning_effort": "low"}
    assert agent.denario_model == "gpt-4o-mini"


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5, "2"])
def test_invalid_planning_iterations(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        DenarioAgent(denario_iterations=iterations)


def test_planning_passes_task_and_tool_schemas_and_cleans_project(fake_sdk):
    original_cwd = Path.cwd()
    agent = DenarioAgent(denario_model="gpt-4o-mini", denario_iterations=3)
    tools = [{"type": "function", "function": {"name": "measure", "parameters": {}}}]
    idea, method = agent._generate_plan("Solve café task", tools)
    planner = fake_sdk.instances[0]
    assert (idea, method) == ("Measure café sample.", "Measure, then submit.")
    assert "Solve café task" in planner.description
    assert json.dumps(tools) in planner.description
    assert planner.model == "gpt-4o-mini"
    assert planner.iterations == 3
    assert not planner.project.exists()
    assert Path.cwd() == original_cwd


@pytest.mark.anyio
async def test_real_denario_graph_with_mocked_http(monkeypatch, tmp_path):
    if importlib.util.find_spec("denario") is None:
        pytest.skip("Install the denario extra to test the real SDK")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    import denario.langgraph_agents.reader as reader
    from langchain_openai import ChatOpenAI

    # Exercise the real graph, LangChain and OpenAI SDK, including streaming.
    # Mock only HTTP transport so this also detects SDK compatibility failures.
    requests = []
    responses = iter(
        [
            r"\begin{IDEA}Initial measurement idea.\end{IDEA}",
            r"\begin{CRITIC}Use actual measurements.\end{CRITIC}",
            r"\begin{IDEA}Measure a sample.\end{IDEA}",
            r"\begin{METHODS}Call measure, then submit_answer.\end{METHODS}",
        ]
    )

    def respond(request):
        assert request.url.host == "denario.invalid"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-denario-test"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"].startswith("gpt-4o")
        requests.append(body)
        content = next(responses)
        common = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
        }
        parts = [
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content[:20]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content[20:]},
                        "finish_reason": None,
                    }
                ],
            },
            {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                **common,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]
        data = "".join("data: " + json.dumps(part) + "\n\n" for part in parts)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(data + "data: [DONE]\n\n").encode(),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as async_client:

            def chat_model(**kwargs):
                kwargs["openai_api_key"] = "sk-denario-test"
                return ChatOpenAI(
                    **kwargs,
                    base_url="https://denario.invalid/v1",
                    max_retries=0,
                    http_client=client,
                    http_async_client=async_client,
                )

            monkeypatch.setattr(reader, "ChatOpenAI", chat_model)
            agent = DenarioAgent()
            idea, method = await asyncio.to_thread(
                agent._generate_plan, "Measure a sample and answer.", []
            )
    assert idea == "Measure a sample."
    assert method == "Call measure, then submit_answer."
    assert len(requests) == 4


@pytest.mark.anyio
@pytest.mark.usefixtures("session_stores")
async def test_plan_drives_real_tools_and_resumes_without_replanning(
    monkeypatch, fake_sdk
):
    measurements = []

    def measure(sample: str) -> str:
        """Measure a named sample."""
        measurements.append(sample)
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
    agent = DenarioAgent()
    session = await start_session(environment, max_iterations=1, agent=agent)

    def response(name, arguments):
        return SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id=name,
                    function=SimpleNamespace(
                        name=name, arguments=json.dumps(arguments)
                    ),
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            id=name,
        )

    model = AsyncMock(
        side_effect=[
            response("measure", {"sample": "a"}),
            response("submit_answer", {"answer": "42"}),
        ]
    )
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model)

    first = await agent.run_session(session)
    assert first.status == "iteration_limit"
    assert measurements == ["a"]
    assert fake_sdk.instances[0].thread_id != threading.get_ident()
    assert session.get_agent_state("denario")["idea"] == "Measure café sample."
    handoffs = [m for m in session.messages if m.get("role") == "user"]
    assert len(handoffs) == 1
    assert "not verified results" in handoffs[0]["content"]
    assert any(
        message.get("content") == handoffs[0]["content"]
        for message in model.await_args_list[0].kwargs["messages"]
    )

    # A fresh adapter must use persisted state, even without the SDK installed.
    monkeypatch.setitem(sys.modules, "denario", None)
    final = await DenarioAgent().run_session(session)
    assert final.status == "completed"
    assert final.answer == session.submission == "42"
    assert final.usage == AgentUsage(input_tokens=10, output_tokens=5, llm_calls=1)
    assert final.metadata["usage_scope"] == "executor_only"
    assert measurements == ["a"]
    assert len(fake_sdk.instances) == 1
    assert [m for m in session.messages if m.get("role") == "user"] == handoffs
    assert any(m.get("role") == "tool" for m in model.await_args.kwargs["messages"])


@pytest.mark.anyio
async def test_structured_prompt_and_incomplete_cached_plan(monkeypatch):
    agent = DenarioAgent()
    generate = Mock(return_value=("idea", "method"))
    monkeypatch.setattr(agent, "_generate_plan", generate)
    outcome = AgentOutcome(status="surrendered", metadata={"executor": "preserved"})
    monkeypatch.setattr(
        ToolCallingAgent, "run_session", AsyncMock(return_value=outcome)
    )
    session = Session({"idea": "incomplete"})
    result = await agent.run_session(session)
    assert json.loads(generate.call_args.args[0]) == session.prompt
    assert session.state == {"idea": "idea", "method": "method"}
    assert result.status == "surrendered"
    assert result.metadata["executor"] == "preserved"


@pytest.mark.anyio
async def test_missing_sdk_fails_before_execution(monkeypatch):
    monkeypatch.setitem(sys.modules, "denario", None)
    execute = AsyncMock()
    monkeypatch.setattr(ToolCallingAgent, "run_session", execute)
    session = Session()
    result = await DenarioAgent().run_session(session)
    assert result.status == "agent_failure"
    assert "uv sync --extra denario" in result.error
    assert not session.state
    assert not session.messages
    execute.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["exception", "missing", "empty"])
async def test_planner_errors_never_become_answers(monkeypatch, fake_sdk, failure):
    def fail(planner, **_kwargs):
        if failure == "exception":
            raise RuntimeError("planning unavailable")
        if failure == "empty":
            (planner.inputs / "methods.md").write_text(" \n", encoding="utf-8")

    monkeypatch.setattr(fake_sdk.agent_class, "get_method_fast", fail)
    execute = AsyncMock()
    monkeypatch.setattr(ToolCallingAgent, "run_session", execute)
    session = Session()
    result = await DenarioAgent().run_session(session)
    assert result.status == "agent_failure"
    assert result.answer is None
    assert "Denario planning failed" in result.error
    assert not session.state
    assert not fake_sdk.instances[0].project.exists()
    execute.assert_not_awaited()


@pytest.mark.anyio
async def test_cancellation_never_executes_or_commits_a_plan(monkeypatch):
    started = asyncio.Event()

    async def wait_for_plan(*_args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "to_thread", wait_for_plan)
    execute = AsyncMock()
    monkeypatch.setattr(ToolCallingAgent, "run_session", execute)
    session = Session()
    pending = asyncio.create_task(DenarioAgent().run_session(session))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not session.state
    assert not session.messages
    execute.assert_not_awaited()
