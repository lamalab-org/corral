"""Resumed searches use saved evidence instead of internal worker transcripts."""

import json

import pytest
from tests.agents.ai_scientist.test_agent import make_session
from tests.agents.ai_scientist.test_prompt_views import SECRET, node_with_trajectory
from tests.agents.conftest import MockLLMResponse

from corral.agents.ai_scientist import agent as agent_module
from corral.agents.ai_scientist.agent import AIScientistAgent, _GatewayOwner
from corral.agents.ai_scientist.search.nodes import ExperimentDecision, ResearchStage
from corral.agents.ai_scientist.state import ScientistState, TaskFormulation
from corral.agents.ai_scientist.workers.base import LiteLLMStructuredModel

pytestmark = pytest.mark.usefixtures("session_stores")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_transcript", [False, True])
async def test_resume_keeps_evidence_and_external_context_without_worker_replay(
    monkeypatch, legacy_transcript
):
    owner = _GatewayOwner()
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return MockLLMResponse(
            content=json.dumps(
                {"decision": "finish", "rationale": SECRET, "conclusion": "done"}
            ),
            usage={"total_tokens": 1},
        )

    previous = LiteLLMStructuredModel(
        owner=owner,
        default_model="test-model",
        evaluator_model="test-model",
        system_prompt="old worker system",
        temperature=0,
        api_endpoint=None,
        max_calls=1,
        use_structured_output=True,
        completion_runner=complete,
    )
    previous.generate(
        f"Old worker prompt containing {SECRET}",
        ExperimentDecision,
        purpose="experiment_node_0001_step_1" if legacy_transcript else "custom_worker",
    )
    assert all("metadata" not in message for message in requests[0]["messages"])
    if legacy_transcript:
        for message in owner.messages:
            message.pop("metadata", None)

    agent = AIScientistAgent(model="test-model")
    calls = []
    session = await make_session(calls, agent=agent)
    await session.record_message(
        {"role": "user", "name": "researcher", "content": "Keep external instruction"}
    )
    await session.record_messages(owner.messages, usage={"total_tokens": 1})
    state = ScientistState(
        task_prompt="measure",
        tools=[],
        formulation=TaskFormulation(objective="measure", required_answer="a number"),
    )
    node = node_with_trajectory()
    state.tree.add(node)
    state.journal.integrate(node)
    state.current_stage = ResearchStage.COMPLETE
    await session.set_agent_state(
        "ai_scientist", agent._state_payload(state, (), status="partial")
    )

    prompts = []

    async def synthesize(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return MockLLMResponse(
            content='{"final_answer":"42"}', usage={"total_tokens": 1}
        )

    monkeypatch.setattr(agent_module, "llm_call", synthesize)
    outcome = await agent.run_session(session)

    assert outcome.status == "completed"
    assert outcome.answer == "42"
    assert calls == []
    (prompt,) = prompts
    assert SECRET not in prompt
    assert "old worker system" not in prompt
    assert "Keep external instruction" in prompt
    assert "design rationale stays" in prompt
    assert "measured:42" in prompt
    assert "the value is 42" in prompt
    # Neither canonical conversation nor the durable tree loses its rationale.
    assert any(SECRET in str(message.get("content")) for message in session.messages)
    saved = session.get_agent_state("ai_scientist")["search"]
    assert saved["tree"]["nodes"][0]["trajectory"][0]["decision"]["rationale"] == SECRET
