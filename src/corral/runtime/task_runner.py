"""Task runner with durable, Git-like State checkpoints.

The canonical execution head is persisted at task setup, immediately before
every tool call, immediately after every tool observation, and at task end.
Each checkpoint is a complete immutable State with a content hash and parent
hash, so a Temporal retry can resume an exact pending action instead of asking
the agent to decide again.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from corral.agents.schema import BudgetExhaustedError
from corral.agents.session import Agent, AgentSession, run_agent_session
from corral.core.state import RuntimeState, State, TaskOutput, checkpoint_state
from corral.observability import (
    NoOpObserver,
    Observation,
    ObservationContext,
    Observer,
    observe_safely,
    update_safely,
)
from corral.persistence import StateTransitionConflictError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from corral.core.environment import Environment
    from corral.persistence import StateStore


_TASK_CONFIGURED_TRANSITION_ID = "task:configured"
_TASK_COMPLETED_TRANSITION_ID = "task:completed"
_TASK_FAILED_TRANSITION_ID = "task:failed"


class TaskRuntime:
    """Execute a task while advancing its durable State head step by step."""

    def __init__(
        self,
        state_store: StateStore,
        observer: Observer | None = None,
    ) -> None:
        self.state_store = state_store
        self.observer = observer or NoOpObserver()

    @staticmethod
    def _context(
        state: State,
        observation_context: ObservationContext | None,
    ) -> ObservationContext:
        if observation_context is not None:
            return observation_context
        task_id = state.metadata.task.get("id")
        return ObservationContext(
            execution_id=state.id,
            task_id=str(task_id) if task_id is not None else None,
        )

    async def _save_initial(
        self,
        state: State,
        context: ObservationContext,
    ) -> State:
        with observe_safely(
            self.observer,
            Observation(
                name="state.commit",
                context=context,
                metadata={"boundary": "task-start"},
            ),
        ) as span:
            committed = await self.state_store.save(state, advance_head=True)
            update_safely(span, state_after=committed)
            return committed

    async def _save_checkpoint(
        self,
        source: State,
        *,
        parent: State,
        transition_id: str,
        boundary: str,
        context: ObservationContext,
    ) -> State:
        state = checkpoint_state(parent, source)
        with observe_safely(
            self.observer,
            Observation(
                name="state.commit",
                context=context,
                state_before=parent,
                metadata={"boundary": boundary},
            ),
        ) as span:
            reused = False
            try:
                committed = await self.state_store.save(
                    state,
                    transition_id=transition_id,
                    advance_head=True,
                )
            except StateTransitionConflictError:
                committed = await self.state_store.load_transition(
                    parent.state_hash,
                    transition_id,
                )
                if committed is None:
                    raise
                reused = True
            update_safely(
                span,
                state_after=committed,
                metadata={"reused_after_conflict": reused},
            )
            return committed

    async def _resume_pending_action(
        self,
        environment: Environment,
        state: State,
        *,
        context: ObservationContext,
    ) -> State:
        """Finish a before-tool checkpoint restored after an interrupted run."""
        session = AgentSession(
            environment,
            state,
            state_store=self.state_store,
            durable_state=state,
            observer=self.observer,
            observation_context=context,
        )
        try:
            await session.resume_pending_action()
            return session.state
        finally:
            session.close()

    async def _run_agent(
        self,
        agent: Agent,
        environment: Environment,
        state: State,
        *,
        last_score: Mapping[str, Any] | None,
        previous_state: State | None,
        max_iterations: int,
        context: ObservationContext,
    ) -> State:
        outcome = await run_agent_session(
            agent,
            environment,
            state,
            last_score=last_score,
            previous_state=previous_state,
            max_iterations=max_iterations,
            state_store=self.state_store,
            durable_state=state,
            observer=self.observer,
            observation_context=context,
        )
        result = outcome.outcome
        runtime_metadata = {
            **dict(outcome.runtime.metadata),
            "agent_status": result.status,
            "session_metadata": dict(result.metadata),
        }
        if result.error is not None:
            runtime_metadata["agent_error"] = result.error
        current = outcome.state.fork(
            runtime=RuntimeState(
                status=outcome.runtime.status,
                started_at=outcome.runtime.started_at,
                ended_at=outcome.runtime.ended_at,
                metadata=runtime_metadata,
            ),
        )
        if result.status == "budget_exhausted":
            raise BudgetExhaustedError(result.error or "provider budget exhausted")
        if not result.is_submit_worthy:
            current = current.fork(
                runtime=RuntimeState(
                    status="failed",
                    started_at=current.runtime.started_at,
                    ended_at=datetime.now(timezone.utc),
                    metadata={
                        **dict(current.runtime.metadata),
                        "error": result.error or "session agent failed",
                    },
                ),
            )
        else:
            # A submit-worthy outcome is valid only after the agent has executed
            # submit_answer inside its session. run_agent_session enforces that
            # invariant; the runtime must not manufacture a submission afterward.
            assert current.submission is not None

        return await self._save_checkpoint(
            current,
            parent=outcome.durable_state,
            transition_id=_TASK_COMPLETED_TRANSITION_ID,
            boundary="task-end",
            context=context,
        )

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
        previous_state: State | None = None,
        observation_context: ObservationContext | None = None,
    ) -> State:
        """Run or resume one task from its latest durable State checkpoint."""
        state_id = str(uuid5(NAMESPACE_URL, f"corral:execution:{execution_id}"))
        initial = await self.state_store.load_initial(state_id)
        if initial is None:
            # Environment State construction snapshots the task workspace via
            # a synchronous compatibility boundary. Keep that boundary off the
            # Temporal Activity event loop (and off every other async caller's
            # loop) so WorkspaceManager can safely run its async artifact I/O.
            initial = await asyncio.to_thread(
                environment.initial_state,
                dependency_outputs=dependency_outputs,
                state_id=state_id,
                started_at=started_at,
                model_metadata=model_metadata,
                scaffold_metadata=scaffold_metadata,
            )
        initial = await self._save_initial(
            initial, self._context(initial, observation_context)
        )
        context = self._context(initial, observation_context)
        current = await self.state_store.load_head(state_id) or initial
        # Every executable or completed chain belongs to the current State
        # protocol and therefore carries an authoritative tool catalog. There
        # is no legacy terminal-state exception.
        environment.validate_state_tool_catalog(current)
        if current != initial:
            with observe_safely(
                self.observer,
                Observation(
                    name="restore",
                    context=context,
                    input={"state_hash": current.state_hash},
                ),
            ) as restore:
                update_safely(restore, state_after=current)
        if current.is_terminal:
            return current

        # Workspace materialization has the same synchronous compatibility
        # boundary as capture above.
        await asyncio.to_thread(environment.prepare_workspace, current.workspace)

        with observe_safely(
            self.observer,
            Observation(
                name="task.run",
                context=context,
                state_before=initial,
            ),
        ) as task_span:
            try:
                if current == initial:
                    configured, configuration_status = await asyncio.to_thread(
                        environment.configure,
                        current,
                    )
                    configured = configured.fork(
                        runtime=RuntimeState(
                            status=configured.runtime.status,
                            started_at=configured.runtime.started_at,
                            ended_at=configured.runtime.ended_at,
                            metadata={
                                **dict(configured.runtime.metadata),
                                "configuration_status": configuration_status,
                            },
                        )
                    )
                    current = await self._save_checkpoint(
                        configured,
                        parent=initial,
                        transition_id=_TASK_CONFIGURED_TRANSITION_ID,
                        boundary="task-configured",
                        context=context,
                    )

                if current.pending_action is not None:
                    current = await self._resume_pending_action(
                        environment,
                        current,
                        context=context,
                    )
                    if current.is_terminal:
                        update_safely(task_span, state_after=current)
                        return current

                if not isinstance(agent, Agent):
                    raise TypeError("an agent must implement run_session(AgentSession)")
                current = await self._run_agent(
                    agent,
                    environment,
                    current,
                    last_score=last_score,
                    previous_state=previous_state,
                    max_iterations=max_iterations,
                    context=context,
                )
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                durable = await self.state_store.load_head(state_id) or current
                # The before-tool snapshot is intentionally recoverable. Do not
                # turn it into a terminal failure if the worker, persistence
                # backend, or process is interrupted before the observation is
                # committed; Temporal can retry and execute this exact Action ID.
                if durable.pending_action is not None:
                    raise
                failed = durable.fork(
                    runtime=RuntimeState(
                        status="failed",
                        started_at=durable.runtime.started_at,
                        ended_at=datetime.now(timezone.utc),
                        metadata={
                            **dict(durable.runtime.metadata),
                            "error": str(exc),
                        },
                    )
                )
                current = await self._save_checkpoint(
                    failed,
                    parent=durable,
                    transition_id=_TASK_FAILED_TRANSITION_ID,
                    boundary="task-failed",
                    context=context,
                )

            update_safely(task_span, state_after=current)
            return current


__all__ = ["TaskRuntime"]
