"""Tests for Reflexion's session-agent composition."""

from types import SimpleNamespace

import pytest

from corral.agents.reflection import ReflectionModule
from corral.agents.reflexion_agent import ReflexionAgent
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.session import ToolResponse
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class Actor:
    api_endpoint = None

    def __init__(self):
        self.calls = 0

    async def run_session(self, session):
        self.calls += 1
        await session.record_message(
            {"role": "assistant", "content": f"actor attempt {self.calls}"}
        )
        result = await session.execute(
            Action(name=SUBMIT_ANSWER_TOOL_NAME, arguments={"answer": "42"})
        )
        assert result.success is True
        return AgentOutcome(
            status="completed",
            answer="42",
            usage=AgentUsage(input_tokens=2, output_tokens=1, llm_calls=1),
        )


class FakeSession:
    prompt = "solve"
    execution_id = "execution-1"

    def __init__(
        self,
        previous_evaluation=None,
        previous_state=None,
        model="test-model",
        iteration_limit=10,
    ):
        self.initial_state = SimpleNamespace(
            metadata=SimpleNamespace(
                task={"id": "task-1"},
                model=({"name": model} if model is not None else {}),
            )
        )
        self.previous_evaluation = previous_evaluation
        self.previous_state = previous_state
        self.iteration_limit = iteration_limit
        self.messages = []
        self.submission = None
        self.agent_state = {}
        self.delegate_budgets = []

    def get_agent_state(self, namespace, *, previous=False):
        if previous:
            if self.previous_state is None:
                return None
            namespaces = self.previous_state.runtime.metadata.get("agent_state", {})
            return namespaces.get(namespace)
        return self.agent_state.get(namespace)

    def set_agent_state(self, namespace, value):
        self.agent_state[namespace] = dict(value)

    async def record_message(self, message):
        self.messages.append(dict(message))

    async def execute(self, action):
        assert action.name == SUBMIT_ANSWER_TOOL_NAME
        self.submission = str(action.arguments["answer"])
        return ToolResponse(success=True, result="answer accepted", error=None)

    async def run_delegate(self, agent, *, max_iterations=None):
        assert max_iterations is not None
        assert max_iterations >= 1
        self.delegate_budgets.append(max_iterations)
        return await agent.run_session(self)

    def final_messages(self):
        return tuple(self.messages)


def test_reflexion_requires_a_session_agent():
    with pytest.raises(TypeError, match="run_session"):
        ReflexionAgent(actor=object())


@pytest.mark.anyio()
async def test_reflexion_requires_model_metadata_without_requiring_actor_model():
    actor = Actor()
    agent = ReflexionAgent(actor=actor)

    outcome = await agent.run_session(FakeSession(model=None))

    assert outcome.status == "agent_failure"
    assert "State.metadata.model.name" in str(outcome.error)
    assert actor.calls == 0


@pytest.mark.anyio()
async def test_reflexion_generates_memory_before_next_actor_attempt(monkeypatch):
    actor = Actor()
    agent = ReflexionAgent(
        actor=actor,
        reflection_prompt=(
            "{{task_id}} {{trial_id}} {{score}} {{trajectory_summary}} "
            "{{task_description}}"
        ),
    )
    first = FakeSession()
    first_outcome = await agent.run_session(first)

    reflection_models = []

    async def generate_reflection(reflection_module, **_kwargs):
        reflection_models.append(reflection_module.model)
        return (
            "Check the measured value before answering.",
            {"prompt_tokens": 5, "completion_tokens": 2},
        )

    monkeypatch.setattr(
        ReflectionModule,
        "generate_reflection",
        generate_reflection,
    )
    previous_state = SimpleNamespace(
        messages=tuple(first.messages),
        state_hash="previous-state-hash",
        runtime=SimpleNamespace(metadata={"agent_state": first.agent_state}),
    )
    second = FakeSession(
        previous_evaluation={
            "trial_id": "trial-1",
            "score": 0.2,
            "state_hash": "previous-state-hash",
        },
        previous_state=previous_state,
    )
    second_outcome = await agent.run_session(second)

    assert first_outcome.status == "completed"
    assert second_outcome.status == "completed"
    assert second_outcome.usage.llm_calls == 2
    assert first.delegate_budgets == [10]
    assert second.delegate_budgets == [9]
    assert second_outcome.metadata["reflection_model"] == "test-model"
    assert reflection_models == ["test-model"]
    assert not hasattr(agent, "memory")
    assert not hasattr(agent, "_previous_messages")
    memory = second.agent_state["reflexion"]["memory"]
    assert second.agent_state["reflexion"]["reflection_model"] == "test-model"
    assert len(memory["reflections"]) == 1
    assert second.agent_state["reflexion"]["source_state_hash"] == (
        "previous-state-hash"
    )
    assert any(
        "Check the measured value" in str(message.get("content"))
        for message in second.messages
    )


@pytest.mark.anyio()
async def test_reflexion_reserves_a_single_available_call_for_the_actor(monkeypatch):
    actor = Actor()
    agent = ReflexionAgent(actor=actor)
    previous_state = SimpleNamespace(
        messages=({"role": "assistant", "content": "previous attempt"},),
        state_hash="previous-state-hash",
        runtime=SimpleNamespace(metadata={}),
    )
    session = FakeSession(
        previous_evaluation={"trial_id": "trial-1", "score": 0.0},
        previous_state=previous_state,
        iteration_limit=1,
    )

    generate = pytest.fail
    monkeypatch.setattr(ReflectionModule, "generate_reflection", generate)

    outcome = await agent.run_session(session)

    assert outcome.status == "completed"
    assert outcome.usage.llm_calls == 1
    assert session.delegate_budgets == [1]
