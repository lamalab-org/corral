"""Task runtime driven entirely by replayable authored event commits."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from corral.agents.schema import BudgetExhaustedError
from corral.agents.session import (
    Agent,
    AgentSession,
    agent_session_capabilities,
    run_agent_session,
)
from corral.backend.mcp import open_mcp_host
from corral.core.actors import ActorRef
from corral.core.commit import Commit, CommitRequest
from corral.core.errors import concise_error_message
from corral.core.events import (
    AgentCompleted,
    AgentStarted,
    ExecutionCompleted,
    ExecutionFailed,
)
from corral.observability import (
    LoggingObserver,
    Observation,
    ObservationContext,
    Observer,
    observe_safely,
    record_commit_safely,
    update_safely,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from pydantic import JsonValue

    from corral.core.environment import Environment
    from corral.core.state import ExecutionState, TaskOutput
    from corral.persistence import CommitStore


class TaskRuntime:
    """Append lifecycle events, resume durable actions, and materialize results."""

    def __init__(
        self, state_store: CommitStore, observer: Observer | None = None
    ) -> None:
        self.state_store = state_store
        self.observer = observer or LoggingObserver()

    @staticmethod
    def _context(
        execution_id: str,
        task_id: str,
        supplied: ObservationContext | None,
    ) -> ObservationContext:
        return supplied or ObservationContext(
            execution_id=execution_id, task_id=task_id
        )

    def _execution_store(self, execution_id: str) -> CommitStore:
        factory = getattr(self.state_store, "for_execution", None)
        if factory is not None:
            return factory(execution_id)
        bound = getattr(self.state_store, "execution_id", None)
        if bound not in {None, execution_id}:
            raise ValueError("commit store is bound to another execution")
        return self.state_store

    async def _append(
        self,
        store: CommitStore,
        request: CommitRequest,
        context: ObservationContext,
    ) -> Commit:
        commit = await store.bind(
            request.author,
            branch_id=request.branch_id,
            execution_id=request.execution_id,
        ).append(request)
        record_commit_safely(self.observer, commit, context=context)
        return commit

    async def _complete_accepted_submission(
        self,
        store: CommitStore,
        current: ExecutionState,
        *,
        runtime_actor: ActorRef,
        agent_actor: ActorRef,
        context: ObservationContext,
        recovery_error: Exception | None = None,
    ) -> tuple[ExecutionState, Commit | None]:
        """Close lifecycle records after an already durable submission."""
        if current.runtime.status not in {"submitted", "surrendered"}:
            return current, None
        run = current.agent_runs.get(agent_actor.run_id)
        if run is not None and run.status in {"created", "running"}:
            metadata: dict[str, JsonValue] = {"recovered_after_submission": True}
            if recovery_error is not None:
                metadata.update(
                    cleanup_error=concise_error_message(recovery_error),
                    cleanup_error_type=type(recovery_error).__name__,
                )
            agent_completed = await self._append(
                store,
                CommitRequest(
                    request_id=f"agent:{agent_actor.run_id}:completed",
                    execution_id=current.execution_id,
                    branch_id=current.branch_id,
                    based_on_hash=current.through_commit_hash,
                    author=runtime_actor,
                    event=AgentCompleted(
                        agent_run_id=agent_actor.run_id,
                        status=(
                            "surrendered"
                            if current.runtime.status == "surrendered"
                            else "completed"
                        ),
                        result_summary={"answer": current.submission},
                        trace_head=current.through_commit_hash,
                        metadata=metadata,
                    ),
                ),
                context,
            )
            current = await store.materialize(current.branch_id, agent_completed.hash)
        if current.runtime.metadata.get("execution_completed"):
            return current, None
        terminal = await self._append(
            store,
            CommitRequest(
                request_id="execution:completed",
                execution_id=current.execution_id,
                branch_id=current.branch_id,
                based_on_hash=current.through_commit_hash,
                author=runtime_actor,
                event=ExecutionCompleted(status=current.runtime.status),  # type: ignore[arg-type]
            ),
            context,
        )
        return await store.materialize(current.branch_id, terminal.hash), terminal

    async def run(
        self,
        agent: Agent,
        environment: Environment,
        *,
        execution_id: str,
        started_at: datetime,
        max_iterations: int,
        dependency_outputs: Mapping[str, TaskOutput | Mapping[str, Any]] | None = None,
        model_metadata: Mapping[str, JsonValue] | None = None,
        scaffold_metadata: Mapping[str, JsonValue] | None = None,
        last_score: Mapping[str, Any] | None = None,
        previous_state: ExecutionState | None = None,
        observation_context: ObservationContext | None = None,
    ) -> ExecutionState:
        if not isinstance(agent, Agent):
            raise TypeError("an agent must implement run_session(AgentSession)")
        store = self._execution_store(execution_id)
        branch_id = "main"
        context = self._context(execution_id, environment.task_id, observation_context)
        runtime_actor = ActorRef(
            kind="runtime",
            actor_id="corral",
            run_id=f"runtime:{execution_id}",
        )
        agent_actor = ActorRef(
            kind="agent",
            actor_id=type(agent).__name__,
            run_id=str(uuid5(NAMESPACE_URL, f"corral:agent:{execution_id}:main")),
        )
        session_capabilities = agent_session_capabilities(agent)

        head = await store.head(branch_id)
        if head is None:
            initial_event = await asyncio.to_thread(
                environment.initial_event,
                execution_id=execution_id,
                dependency_outputs=dependency_outputs,
                started_at=started_at,
                model_metadata=model_metadata,
                scaffold_metadata={
                    **dict(scaffold_metadata or {}),
                    "max_iterations": max_iterations,
                    "previous_evaluation": last_score,
                },
                actor_id=agent_actor.actor_id,
            )
            head = await self._append(
                store,
                CommitRequest(
                    request_id="execution:started",
                    execution_id=execution_id,
                    branch_id=branch_id,
                    based_on_hash=None,
                    author=runtime_actor,
                    event=initial_event,
                    occurred_at=started_at,
                ),
                context,
            )
        current = await store.materialize(branch_id)
        environment.validate_state_tool_catalog(current)
        if current.is_terminal:
            current, _ = await self._complete_accepted_submission(
                store,
                current,
                runtime_actor=runtime_actor,
                agent_actor=agent_actor,
                context=context,
            )
            return current

        with observe_safely(
            self.observer,
            Observation(
                name="task.run",
                context=context,
                based_on_hash=current.through_commit_hash,
                actor=runtime_actor,
                metadata={"resumed": head.sequence > 0},
            ),
        ) as task_span:
            try:
                event_types = [
                    commit.event.type async for commit in store.iter_commits(branch_id)
                ]
                if "task.configured" not in event_types:
                    configured_event = await asyncio.to_thread(
                        environment.configure, current
                    )
                    head = await self._append(
                        store,
                        CommitRequest(
                            request_id="task:configured",
                            execution_id=execution_id,
                            branch_id=branch_id,
                            based_on_hash=current.through_commit_hash,
                            author=runtime_actor,
                            event=configured_event,
                        ),
                        context,
                    )
                    current = await store.materialize(branch_id)

                async with open_mcp_host() as mcp_host:
                    if agent_actor.run_id not in current.agent_runs:
                        head = await self._append(
                            store,
                            CommitRequest(
                                request_id=f"agent:{agent_actor.run_id}:started",
                                execution_id=execution_id,
                                branch_id=branch_id,
                                based_on_hash=current.through_commit_hash,
                                author=runtime_actor,
                                event=AgentStarted(
                                    agent_run_id=agent_actor.run_id,
                                    agent_id=agent_actor.actor_id,
                                    metadata={
                                        "model": dict(model_metadata or {}),
                                        "session_capabilities": (
                                            session_capabilities.to_metadata()
                                        ),
                                    },
                                ),
                            ),
                            context,
                        )
                        current = await store.materialize(branch_id)

                    pending = tuple(
                        action_state
                        for action_state in current.actions.values()
                        if action_state.requested_by_run_id == agent_actor.run_id
                        and action_state.status in {"pending", "running"}
                    )
                    if pending:
                        session = AgentSession(
                            environment,
                            current,
                            actor=agent_actor,
                            runtime_actor=runtime_actor,
                            state_store=store,
                            branch_id=branch_id,
                            max_iterations=max_iterations,
                            observer=self.observer,
                            observation_context=context,
                            agent=agent,
                            hooks=getattr(agent, "hooks", None),
                            mcp_host=mcp_host,
                        )
                        await session.resume_pending_actions()
                        current = session.state
                    outcome = None
                    if not current.is_terminal:
                        outcome = await run_agent_session(
                            agent,
                            environment,
                            current,
                            actor=agent_actor,
                            runtime_actor=runtime_actor,
                            state_store=store,
                            branch_id=branch_id,
                            last_score=last_score,
                            previous_state=previous_state,
                            max_iterations=max_iterations,
                            observer=self.observer,
                            observation_context=context,
                            mcp_host=mcp_host,
                        )

                if outcome is None:
                    current, terminal = await self._complete_accepted_submission(
                        store,
                        current,
                        runtime_actor=runtime_actor,
                        agent_actor=agent_actor,
                        context=context,
                    )
                    if terminal is not None:
                        update_safely(task_span, commit=terminal)
                    return current
                result = outcome.outcome
                if result.status == "budget_exhausted":
                    raise BudgetExhaustedError(
                        result.error or "provider budget exhausted"
                    )
                if not result.is_submit_worthy:
                    terminal_event = ExecutionFailed(
                        error=result.error or "session agent failed",
                        error_type="AgentOutcomeError",
                        metadata={"agent_status": result.status},
                    )
                    request_id = "execution:failed"
                else:
                    assert outcome.state.submission is not None
                    terminal_event = ExecutionCompleted(
                        status=outcome.state.runtime.status,  # type: ignore[arg-type]
                        metadata={"agent_status": result.status},
                    )
                    request_id = "execution:completed"
                terminal = await self._append(
                    store,
                    CommitRequest(
                        request_id=request_id,
                        execution_id=execution_id,
                        branch_id=branch_id,
                        based_on_hash=outcome.final_commit.hash,
                        author=runtime_actor,
                        event=terminal_event,
                    ),
                    context,
                )
                current = await store.materialize(branch_id, terminal.hash)
                update_safely(task_span, commit=terminal)
                return current
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                current = await store.materialize(branch_id)
                if current.runtime.status in {"submitted", "surrendered"}:
                    current, terminal = await self._complete_accepted_submission(
                        store,
                        current,
                        runtime_actor=runtime_actor,
                        agent_actor=agent_actor,
                        context=context,
                        recovery_error=exc,
                    )
                    if terminal is not None:
                        update_safely(task_span, commit=terminal)
                    return current
                # Running invocations are intentionally resumable on retry.
                if any(
                    action.status in {"pending", "running"}
                    for action in current.actions.values()
                ):
                    raise
                run = current.agent_runs.get(agent_actor.run_id)
                if run is not None and run.status in {"created", "running"}:
                    agent_failed = await self._append(
                        store,
                        CommitRequest(
                            request_id=f"agent:{agent_actor.run_id}:failed",
                            execution_id=execution_id,
                            branch_id=branch_id,
                            based_on_hash=current.through_commit_hash,
                            author=runtime_actor,
                            event=AgentCompleted(
                                agent_run_id=agent_actor.run_id,
                                status="failed",
                                result_summary={"error": concise_error_message(exc)},
                                trace_head=current.through_commit_hash,
                            ),
                        ),
                        context,
                    )
                    current = await store.materialize(branch_id, agent_failed.hash)
                failed = await self._append(
                    store,
                    CommitRequest(
                        request_id="execution:failed",
                        execution_id=execution_id,
                        branch_id=branch_id,
                        based_on_hash=current.through_commit_hash,
                        author=runtime_actor,
                        event=ExecutionFailed(
                            error=concise_error_message(exc),
                            error_type=type(exc).__name__,
                        ),
                    ),
                    context,
                )
                update_safely(task_span, commit=failed)
                return await store.materialize(branch_id, failed.hash)


__all__ = ["TaskRuntime"]
