"""Reflexion wrapper for first-class session agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from corral.agents.base_agent import BaseAgent, _UsageAccumulator
from corral.agents.reflection import (
    Reflection,
    ReflectionMemory,
    ReflectionModule,
    create_reflexion_history,
)
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.session import Agent
from corral.core.errors import concise_error_message
from corral.logging import logger

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


_REFLEXION_STATE_NAMESPACE = "reflexion"


def _combined_usage(reflection: AgentUsage, actor: AgentUsage) -> AgentUsage:
    return AgentUsage(
        input_tokens=reflection.input_tokens + actor.input_tokens,
        output_tokens=reflection.output_tokens + actor.output_tokens,
        reasoning_tokens=reflection.reasoning_tokens + actor.reasoning_tokens,
        llm_calls=reflection.llm_calls + actor.llm_calls,
    )


class ReflexionAgent(BaseAgent):
    """Generate verbal memory from a prior evaluation, then run an actor.

    The task projection is authoritative for the reflection model.
    `reflection_model` remains a construction-time hint so older worker
    registrations can populate that metadata, but it is never read from the
    wrapped actor and never overrides the model recorded for a session.
    """

    def __init__(
        self,
        actor: Agent,
        reflection_model: str | None = None,
        api_endpoint: str | None = None,
        reflection_system_prompt: str | None = None,
        reflection_prompt: str | None = None,
        reflection_temperature: float | None = None,
        max_reflections: int = 5,
        **kwargs: Any,
    ) -> None:
        if not isinstance(actor, Agent):
            raise TypeError(
                "actor must implement run_session(AgentSession), got "
                f"{type(actor).__name__}"
            )
        model_hint = reflection_model or ""
        super().__init__(
            model=model_hint,
            api_endpoint=api_endpoint or getattr(actor, "api_endpoint", None),
            system_prompt=reflection_system_prompt,
            user_prompt=reflection_prompt or "reflexion/user_prompt",
            surrender_prompt=None,
            temperature=(
                0.0 if reflection_temperature is None else reflection_temperature
            ),
            **kwargs,
        )
        self.actor = actor
        self.max_reflections = max_reflections
        self.reflection_system_prompt = reflection_system_prompt

    def _reflection_module(self, session: AgentSession) -> ReflectionModule:
        """Build a run-local reflection client from projection metadata."""
        raw_model = session.state.task.model.get("name")
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise ValueError(
                "ReflexionAgent requires ExecutionState.task.model.name to be a "
                "non-empty string"
            )
        return ReflectionModule(
            model=raw_model,
            reflection_prompt=self.user_prompt,
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            reflection_system_prompt=self.reflection_system_prompt,
        )

    def _memory_from_state(self, session: AgentSession) -> ReflectionMemory:
        """Restore bounded verbal memory from agent-state events."""
        payload = session.get_agent_state(_REFLEXION_STATE_NAMESPACE)
        if payload is None:
            payload = session.get_agent_state(
                _REFLEXION_STATE_NAMESPACE,
                previous=True,
            )
        raw_memory = payload.get("memory") if payload is not None else None
        if isinstance(raw_memory, dict):
            try:
                memory = ReflectionMemory.from_dict(raw_memory)
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring invalid Reflexion memory in agent state")
            else:
                # Configuration is authoritative if it changed between runs.
                if memory.max_size == self.max_reflections:
                    return memory
                resized = ReflectionMemory(max_size=self.max_reflections)
                for reflection in list(memory.reflections)[-self.max_reflections :]:
                    resized.add_reflection(reflection)
                return resized
        return ReflectionMemory(max_size=self.max_reflections)

    @staticmethod
    def _trajectory_from_state(session: AgentSession) -> list[dict[str, Any]]:
        """Read the evaluated attempt's canonical projected transcript."""
        previous_state = session.previous_state
        if previous_state is None:
            return []
        run = next(
            (
                candidate
                for candidate in previous_state.agent_runs.values()
                if candidate.actor_id == session.actor.actor_id
            ),
            None,
        )
        if run is None:
            return []
        return [
            dict(message)
            for message in previous_state.conversations.get(run.run_id, ())
        ]

    @staticmethod
    async def _store_memory(
        session: AgentSession,
        memory: ReflectionMemory,
        *,
        reflection_model: str,
        actor_status: str | None = None,
    ) -> None:
        previous_state = session.previous_state
        await session.set_agent_state(
            _REFLEXION_STATE_NAMESPACE,
            {
                "schema_version": 1,
                "reflection_model": reflection_model,
                "memory": memory.to_dict(),
                "source_commit_hash": (
                    previous_state.through_commit_hash
                    if previous_state is not None
                    else None
                ),
                "actor_status": actor_status,
            },
        )

    async def _generate_reflection(
        self,
        *,
        reflection_module: ReflectionModule,
        memory: ReflectionMemory,
        trajectory: list[dict[str, Any]],
        task_id: str,
        trial_id: str,
        score: float,
        task_description: str,
    ) -> AgentUsage:
        if not trajectory:
            return AgentUsage()
        text, raw_usage = await reflection_module.generate_reflection(
            task_id=task_id,
            trial_id=trial_id,
            trajectory=trajectory,
            score=score,
            task_description=task_description,
        )
        memory.add_reflection(
            Reflection(
                trial_index=len(memory.reflections),
                task_id=task_id,
                trajectory=list(trajectory),
                reflection_text=text,
                score=score,
            )
        )
        usage = _UsageAccumulator(self._usage)
        usage.add(raw_usage)
        return usage.outcome()

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run the reflect-then-act lifecycle against one bound session."""
        try:
            reflection_module = self._reflection_module(session)
        except ValueError as exc:
            return AgentOutcome(
                status="agent_failure", error=concise_error_message(exc)
            )
        reflection_model = reflection_module.model
        task_id = str(getattr(session, "task_id", session.execution_id))
        memory = self._memory_from_state(session)
        previous_trajectory = self._trajectory_from_state(session)
        reflection_usage = AgentUsage()
        previous = session.previous_evaluation
        if previous is not None and previous_trajectory and session.iteration_limit > 1:
            prompt = session.prompt
            task_description = (
                prompt
                if isinstance(prompt, str)
                else json.dumps(prompt, ensure_ascii=False)
            )
            raw_score = previous.get("score", 0.0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            trial_id = str(previous.get("trial_id", "unknown"))
            try:
                reflection_usage = await self._generate_reflection(
                    reflection_module=reflection_module,
                    memory=memory,
                    trajectory=previous_trajectory,
                    task_id=task_id,
                    trial_id=trial_id,
                    score=score,
                    task_description=task_description,
                )
            except Exception as exc:
                logger.warning(f"Reflexion memory update failed: {exc}")

        await self._store_memory(
            session,
            memory,
            reflection_model=reflection_model,
        )
        for message in create_reflexion_history(memory) or []:
            await session.record_message(dict(message))

        remaining_iterations = session.iteration_limit - reflection_usage.llm_calls
        outcome = await session.run_delegate(
            self.actor,
            max_iterations=remaining_iterations,
        )
        await self._store_memory(
            session,
            memory,
            reflection_model=reflection_model,
            actor_status=outcome.status,
        )
        return AgentOutcome(
            status=outcome.status,
            answer=outcome.answer,
            error=outcome.error,
            usage=_combined_usage(reflection_usage, outcome.usage),
            metadata={
                "actor": type(self.actor).__name__,
                "reflection_model": reflection_model,
                "reflection_count": len(memory.reflections),
                **dict(outcome.metadata),
            },
        )


__all__ = ["ReflexionAgent"]
