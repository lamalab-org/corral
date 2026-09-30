"""Blind benchmark policies at direct runtime and evaluation entry points."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.evaluation import TaskScorer
from corral.persistence import SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture
def anyio_backend():
    return "asyncio"


class ProbeAgent:
    async def run_session(self, session):
        assert session.previous_evaluation is None
        assert session.previous_state is None
        assert session.previous_messages == ()
        assert session.state.task.scaffold.get("previous_evaluation") is None
        await session.execute(Action(name="submit_answer", arguments={"answer": "42"}))
        return AgentOutcome(status="completed", answer="42")


@pytest.mark.anyio
async def test_runtime_drops_prior_feedback_for_direct_callers(tmp_path):
    task = TaskDefinition(
        name="blind",
        description="blind",
        tools=[],
        scoring_fn=lambda a: 1.0,
        submission_format={},
        resolve_answer=False,
        allow_previous_attempt_context=False,
        execution_version="blind-v1",
        scorer_version="score-v2:bank-a",
    )
    env = Environment("blind", task, toolset=Toolset(pool={}, workspace_factory=None))
    async with SQLiteCommitStore(tmp_path / "blind.sqlite3") as store:
        state = await TaskRuntime(store).run(
            ProbeAgent(),
            env,
            execution_id="one",
            started_at=datetime.now(timezone.utc),
            max_iterations=1,
            last_score={"score": 1, "feedback": "secret"},
            previous_state=None,
        )
        second = await TaskRuntime(store).run(
            ProbeAgent(),
            env,
            execution_id="two",
            started_at=datetime.now(timezone.utc),
            max_iterations=1,
            last_score={"score": 1, "feedback": "secret"},
            previous_state=state,
        )
        assert TaskScorer(task).evaluate(second).scorer_version == "score-v2:bank-a"
        with pytest.raises(ValueError, match="protocol or task bank"):
            TaskScorer(replace(task, execution_version="blind-v2")).evaluate(second)
        with pytest.raises(ValueError, match="Incompatible"):
            changed = Environment(
                "blind",
                replace(task, execution_version="blind-v2"),
                toolset=Toolset(pool={}, workspace_factory=None),
            )
            await TaskRuntime(store).run(
                ProbeAgent(),
                changed,
                execution_id="two",
                started_at=datetime.now(timezone.utc),
                max_iterations=1,
            )
