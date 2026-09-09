"""Schema-validating execution through canonical Corral actions."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from jsonschema import Draft202012Validator

from corral.agents.ai_scientist.search.nodes import (
    ExecutedAction,
    Observation,
    PlannedAction,
)
from corral.agents.session import ToolResponse
from corral.core.action import Action


class ToolCallBudgetExceeded(RuntimeError):
    """Raised before a tool call would exceed the configured global budget."""


@dataclass
class ResearchBudget:
    """One thread-safe physical tool-call budget shared by every branch."""

    max_physical_tool_calls: int
    scientific_calls: int = 0
    replay_calls: int = 0
    _lock: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._lock = Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self.scientific_calls + self.replay_calls

    @property
    def remaining(self) -> int:
        return max(0, self.max_physical_tool_calls - self.used)

    def consume(self, *, replay: bool) -> bool:
        """Atomically reserve one call, returning false at the global limit."""
        with self._lock:
            if (
                self.scientific_calls + self.replay_calls
                >= self.max_physical_tool_calls
            ):
                return False
            if replay:
                self.replay_calls += 1
            else:
                self.scientific_calls += 1
            return True


def _tool_definitions(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict]:
    if isinstance(payload, list):
        return payload
    tools = payload.get("tools", [])
    if isinstance(tools, dict):
        tools = tools.get("tools", [])
    return tools if isinstance(tools, list) else []


def _normalise_tool(tool: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return `(name, input_schema)` for OpenAI or MCP tool descriptions."""
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"]), function.get("parameters", {})
    name = tool.get("name")
    if name:
        return str(name), tool.get("inputSchema", tool.get("input_schema", {}))
    return None


class CorralExecutor:
    """Execute only declared Corral tools, one action at a time."""

    def __init__(
        self,
        *,
        execute_action: Callable[[Action], ToolResponse],
        tools: dict[str, Any] | list[dict[str, Any]],
        max_tool_calls: int,
        max_observation_chars: int | None = 12_000,
        stop_on_error: bool = True,
        budget: ResearchBudget | None = None,
        on_executed: Callable[[ExecutedAction], None] | None = None,
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.execute_action = execute_action
        self.max_tool_calls = max_tool_calls
        self.max_observation_chars = max_observation_chars
        self.stop_on_error = stop_on_error
        self.budget = budget or ResearchBudget(max_tool_calls)
        self._on_executed = on_executed
        self._on_tool_call = on_tool_call
        self._schemas: dict[str, dict[str, Any]] = {}
        for tool in _tool_definitions(tools):
            normalised = _normalise_tool(tool)
            if normalised is not None:
                self._schemas[normalised[0]] = normalised[1]

    @property
    def tool_names(self) -> set[str]:
        return set(self._schemas)

    @property
    def remaining_calls(self) -> int:
        return self.budget.remaining

    @property
    def call_count(self) -> int:
        return self.budget.used

    def prepare_replication_plan(
        self,
        executed_actions: list[ExecutedAction],
        *,
        seed: int,
    ) -> tuple[list[PlannedAction], list[str]]:
        """Copy an exact realized plan, changing only exposed seed arguments.

        Tool schemas, rather than argument-name guesses, determine whether an
        action supports a seed. This means tools without an explicit `seed`
        or `random_state` property are replayed byte-for-byte at the logical
        action level and are reported as independent unseeded repetitions.
        """
        plan: list[PlannedAction] = []
        overrides: list[str] = []
        for index, executed in enumerate(executed_actions):
            action = executed.action
            schema = self._schemas.get(action.tool_name, {})
            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            arguments = dict(action.arguments)
            for field_name in ("seed", "random_state"):
                if field_name not in properties:
                    continue
                arguments[field_name] = seed
                overrides.append(f"action_{index}.{action.tool_name}.{field_name}")
            plan.append(action.model_copy(deep=True, update={"arguments": arguments}))
        return plan, overrides

    def execute_plan(
        self,
        plan: list[PlannedAction],
        *,
        replay: bool = False,
        start_index: int = 0,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for index, action in enumerate(plan, start=start_index):
            error = self._validate_action(action)
            if error is not None:
                observations.append(self._failure(index, action, error))
                if self.stop_on_error:
                    break
                continue

            if not self.budget.consume(replay=replay):
                observations.append(
                    self._failure(
                        index,
                        action,
                        f"Tool-call budget exhausted ({self.max_tool_calls}).",
                    )
                )
                break

            # This is intentionally the only environment action in the package.
            # It never calls submit_answer/get_last_score and it executes plans
            # sequentially because a Corral task workspace may be stateful.
            started_at = datetime.now(tz=timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                response = self.execute_action(
                    Action(name=action.tool_name, arguments=action.arguments)
                )
                success = bool(getattr(response, "success", False))
                result = getattr(response, "result", None)
                response_error = getattr(response, "error", None)

                if success:
                    observation = Observation(
                        action_index=index,
                        purpose=action.purpose,
                        tool_name=action.tool_name,
                        arguments=action.arguments,
                        success=True,
                        result=self._bounded(result),
                    )
                    observations.append(observation)
                else:
                    observations.append(
                        self._failure(
                            index,
                            action,
                            str(response_error or "Corral tool returned failure"),
                        )
                    )
            except Exception as exc:
                observations.append(
                    self._failure(index, action, f"Tool execution raised: {exc}")
                )

            if self._on_executed is not None:
                self._on_executed(
                    ExecutedAction(action=action, observation=observations[-1])
                )

            if self._on_tool_call is not None:
                observation = observations[-1]
                self._on_tool_call(
                    {
                        "tool_name": action.tool_name,
                        "arguments": dict(action.arguments),
                        "result": observation.result,
                        "status": "success"
                        if observation.success
                        else "execution_error",
                        "error_message": observation.error,
                        "duration": time.perf_counter() - started,
                        "timestamp": started_at,
                    }
                )

            if not observations[-1].success and self.stop_on_error:
                break
        return observations

    def _validate_action(self, action: PlannedAction) -> str | None:
        schema = self._schemas.get(action.tool_name)
        if schema is None:
            return f"Unknown Corral tool: {action.tool_name}"
        errors = sorted(
            Draft202012Validator(schema).iter_errors(action.arguments),
            key=lambda item: list(item.path),
        )
        if not errors:
            return None
        return "Invalid tool arguments: " + "; ".join(error.message for error in errors)

    def _bounded(self, value: Any) -> str:
        rendered = str(value)
        limit = self.max_observation_chars
        if limit is None or len(rendered) <= limit:
            return rendered
        keep = limit // 2
        return rendered[:keep] + "\n... observation truncated ...\n" + rendered[-keep:]

    @staticmethod
    def _failure(index: int, action: PlannedAction, error: str) -> Observation:
        return Observation(
            action_index=index,
            purpose=action.purpose,
            tool_name=action.tool_name,
            arguments=action.arguments,
            success=False,
            error=error,
        )
