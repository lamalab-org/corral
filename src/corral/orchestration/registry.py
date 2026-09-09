"""Live resources used by task and benchmark execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.agents.session import Agent
    from corral.core.environment import Environment


class RuntimeRegistry:
    """Agents and task-scoped environments shared by a run."""

    def __init__(
        self,
        *,
        agents: Mapping[str, Agent],
        environments: Mapping[str, Environment],
        workspace_manager_factory: Any | None = None,
    ) -> None:
        self._agents = dict(agents)
        self._environments = dict(environments)
        self._bound: dict[tuple[str, str], Environment] = {}
        self._last_evaluations: dict[str, dict[str, Any]] = {}
        self._workspace_manager_factory = workspace_manager_factory
        self._lock = threading.RLock()

    def agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"registry has no agent {agent_id!r}") from exc

    def environment(self, environment_id: str, execution_id: str) -> Environment:
        key = (environment_id, execution_id)
        with self._lock:
            bound = self._bound.get(key)
            if bound is not None:
                return bound
            try:
                template = self._environments[environment_id]
            except KeyError as exc:
                raise KeyError(
                    f"registry has no environment {environment_id!r}"
                ) from exc
            bound = template.for_task(execution_id)
            if self._workspace_manager_factory is not None:
                bound.workspace_manager = self._workspace_manager_factory(execution_id)
            self._bound[key] = bound
            return bound

    def record_evaluation(
        self,
        task_id: str,
        execution_id: str,
        result: Mapping[str, Any],
    ) -> None:
        """Expose an evaluation and its commit hash to the next task attempt."""
        with self._lock:
            self._last_evaluations[task_id] = {
                "trial_id": execution_id,
                **dict(result),
            }

    def last_evaluation(self, task_id: str | None) -> dict[str, Any] | None:
        if task_id is None:
            return None
        with self._lock:
            result = self._last_evaluations.get(task_id)
            return dict(result) if result is not None else None

    def close(self) -> None:
        with self._lock:
            environments = tuple(self._bound.values())
            self._bound.clear()
            self._last_evaluations.clear()
        for environment in environments:
            environment.shutdown_jobs()
