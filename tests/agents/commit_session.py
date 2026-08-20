from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from corral.agents.session import AgentSession, agent_session_capabilities
from corral.core import ActorRef, AgentStarted, CommitRequest
from corral.persistence import SQLiteCommitStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.core.environment import Environment
    from corral.core.state import ExecutionState


async def start_session(
    environment: Environment,
    *,
    max_iterations: int = 10,
    actor_id: str = "test-agent",
    previous_state: ExecutionState | None = None,
    last_score: Mapping[str, Any] | None = None,
    agent: Any = None,
) -> AgentSession:
    execution_id = f"test-{uuid4()}"
    store = SQLiteCommitStore(":memory:", execution_id)
    runtime = ActorRef(
        kind="runtime", actor_id="corral", run_id=f"runtime:{execution_id}"
    )
    actor = ActorRef(kind="agent", actor_id=actor_id, run_id=f"agent:{execution_id}")
    root = await store.append(
        CommitRequest(
            request_id="execution:started",
            branch_id="main",
            based_on_hash=None,
            author=runtime,
            event=environment.initial_event(
                execution_id=execution_id,
                actor_id=actor_id,
                started_at=datetime.now(timezone.utc),
                scaffold_metadata={"max_iterations": max_iterations},
            ),
        )
    )
    configured = await store.append(
        CommitRequest(
            request_id="task:configured",
            branch_id="main",
            based_on_hash=root.hash,
            author=runtime,
            event=environment.configure(await store.materialize("main")),
        )
    )
    await store.append(
        CommitRequest(
            request_id="agent:started",
            branch_id="main",
            based_on_hash=configured.hash,
            author=runtime,
            event=AgentStarted(
                agent_run_id=actor.run_id,
                agent_id=actor.actor_id,
                metadata={
                    "session_capabilities": agent_session_capabilities(
                        agent
                    ).to_metadata()
                },
            ),
        )
    )
    state = await store.materialize("main")
    session = AgentSession(
        environment,
        state,
        actor=actor,
        runtime_actor=runtime,
        state_store=store,
        max_iterations=max_iterations,
        previous_state=previous_state,
        last_score=last_score,
        hooks=getattr(agent, "hooks", None),
        agent=agent,
    )
    session._test_commit_store = store
    return session
