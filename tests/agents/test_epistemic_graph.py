"""Graph feedback across native and ReAct actor loops and resumed sessions."""

import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest
from tests.agents.test_epistemic_agent import reply, response
from tests.agents.test_tool_calling_agent import Session, call
from tests.epistemic_analysis.test_knowline_graph import config

from corral.agents.base_agent import provider_messages
from corral.agents.epistemic import EpistemicAgent
from corral.agents.react import ReActAgent
from corral.agents.tool_calling import ToolCallingAgent
from corral.epistemic_analysis.annotators.graph import AnnotationBackend, GraphExtractor
from corral.epistemic_analysis.annotators.local import (
    GraphAnnotator,
    KnowLineClient,
)
from corral.epistemic_analysis.schema import REPORT_MESSAGE_NAME


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("agent_type", [EpistemicAgent, ToolCallingAgent, ReActAgent])
def test_default_epistemic_backend_is_local_and_settings_stay_off_actor(agent_type):
    agent = agent_type(
        **({"epistemic": True} if agent_type is not EpistemicAgent else {}),
        epistemic_provider_config={**asdict(config()), "concurrency": 1},
        epistemic_graph_config={"max_request_tokens": 10000},
    )
    assert isinstance(agent.annotator, GraphAnnotator)
    assert agent.annotator.provider_config.concurrency == 1
    assert agent.annotator.config.max_request_tokens == 10000
    assert not any(k.startswith("epistemic_") for k in agent._call_kwargs())


@pytest.mark.anyio
@pytest.mark.parametrize("agent_type", [EpistemicAgent, ToolCallingAgent, ReActAgent])
async def test_graph_feedback_precedes_actor_and_repaired_submission_is_accepted(
    monkeypatch, agent_type, tmp_path
):
    events = []
    observed = []

    async def nodes(**kwargs):
        events.append("nodes")
        observed.append(kwargs["messages"])
        i = next(
            i for i, m in enumerate(kwargs["messages"]) if m["role"] == "assistant"
        )
        text = kwargs["messages"][i]["content"]
        if "REPAIRED" in json.dumps(kwargs["messages"]):
            return [], []
        return [
            {
                "node_id": "H",
                "type": "H",
                "text": text,
                "time": i,
                "support": [{"msg_idx": i, "quote": text}],
            }
        ], []

    async def edges(**kwargs):
        events.append("edges")
        return [], []

    async def explain(motifs, graph, trajectory, task):
        events.append("explanation")
        return {m.name: "The annotated hypothesis has no test." for m in motifs}, {
            "response": {"usage": {}}
        }

    annotator = GraphAnnotator(provider_config=asdict(config()), artifact_dir=tmp_path)
    monkeypatch.setattr(
        GraphExtractor,
        "backend",
        lambda self, **_: AnnotationBackend(nodes, edges, {"annotator": "test"}),
    )
    monkeypatch.setattr(KnowLineClient, "preflight", AsyncMock())
    monkeypatch.setattr(annotator, "_explain_candidates", explain)
    sent = []

    async def actor(**kwargs):
        events.append("actor")
        sent.append(kwargs["messages"][:])
        content = (
            "A resistance claim." if len(sent) == 1 else "REPAIRED with task evidence."
        )
        return reply(
            agent_type,
            call("submit_answer", '{"answer":"100"}', f"s{len(sent)}"),
            content=content,
        )

    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    agent = agent_type(
        annotator=annotator,
        **({"epistemic": True} if agent_type is not EpistemicAgent else {}),
    )
    session = Session()
    outcome = await agent.run_session(session)
    assert outcome.status == "completed", outcome.error
    assert len(sent) == 2
    assert len(session.calls) == 1
    assert events[0] == "actor"  # No prior behavior on the first turn.
    assert events[1:4] == ["nodes", "edges", "explanation"]
    assert any(m.get("name") == REPORT_MESSAGE_NAME for m in sent[1])
    assert all("epistemic_report" not in json.dumps(prefix) for prefix in observed)
    assert all("REPAIRED" not in json.dumps(prefix) for prefix in observed[:-1])
    reports = [m for m in session.messages if m.get("name") == REPORT_MESSAGE_NAME]
    assert reports[-1]["metadata"]["agent_visible"] is False
    assert reports[-1]["metadata"]["epistemic_report"]["timings_s"]["total"] > 0
    assert reports[-1]["content"] not in [
        m.get("content") for m in provider_messages(session.messages)
    ]
    assert all(
        m.get("content") != reports[-1]["content"]
        for m in agent._initial_messages(session)
    )


@pytest.mark.anyio
async def test_graph_failure_stops_before_actor(monkeypatch):
    annotate = AsyncMock(side_effect=ValueError("graph inference failed"))
    actor = AsyncMock(return_value=response())
    monkeypatch.setattr("corral.agents.base_agent.llm_call", actor)
    session = Session()
    session.messages.append({"role": "assistant", "content": "Existing hypothesis."})
    annotator = GraphAnnotator()
    monkeypatch.setattr(annotator, "annotate", annotate)
    result = await EpistemicAgent(annotator=annotator).run_session(session)
    assert result.status == "agent_failure"
    actor.assert_not_awaited()
    assert not session.calls
