"""Tool execution boundary producing typed event effects, never child States."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.events import (
    AgentTurnRecorded,
    EnvironmentOperation,
    RuntimeUpdate,
    UsageDelta,
    WorkspaceDelta,
)
from corral.core.tool import ToolCallStatus
from corral.runtime import permissions

if TYPE_CHECKING:
    from pydantic import JsonValue

    from corral.core.environment import Environment
    from corral.core.state import ExecutionState

HIDDEN_ARGUMENTS_NAMESPACE = "hidden_arguments"
CORRAL_ACTION_ID_ARGUMENT = "corral_action_id"


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """A tool observation and optional complete environment namespace."""

    content: Any
    environment: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolEffects:
    """Atomic effects returned by tool execution for one completion commit."""

    observation: JsonValue
    status: Literal["success", "invalid_tool", "invalid_args", "execution_error"]
    environment_operations: tuple[EnvironmentOperation, ...] = ()
    workspace_delta: WorkspaceDelta | None = None
    usage_delta: UsageDelta = field(default_factory=lambda: UsageDelta(tool_calls=1))
    runtime_update: RuntimeUpdate | None = None
    expected_environment_revision: int | None = None
    expected_workspace_revision: int | None = None
    duration_ms: float | None = None

    @property
    def success(self) -> bool:
        return self.status == ToolCallStatus.SUCCESS.value


def propose_action(
    action: Action,
    *,
    messages: tuple[Mapping[str, JsonValue], ...] = (),
    usage_delta: UsageDelta | None = None,
    parallel_group_id: str | None = None,
) -> AgentTurnRecorded:
    """Build the durable agent event that proposes one action."""

    assistant_message = action.to_message()
    return AgentTurnRecorded(
        messages=(*messages, assistant_message),
        actions=(action,),
        usage_delta=usage_delta or UsageDelta(),
        parallel_group_id=parallel_group_id,
    )


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def _hidden_arguments(state: ExecutionState) -> dict[str, Any]:
    value = state.environment.values.get(HIDDEN_ARGUMENTS_NAMESPACE, {})
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ExecutionState.environment.{HIDDEN_ARGUMENTS_NAMESPACE} must be an object"
        )
    return _json_copy(value)


def environment_operations(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[EnvironmentOperation, ...]:
    """Return deterministic leaf/replacement operations from complete namespaces."""

    operations: list[EnvironmentOperation] = []

    def visit(path: tuple[str, ...], previous: Any, current: Any) -> None:
        if previous == current:
            return
        if isinstance(previous, Mapping) and isinstance(current, Mapping):
            operations.extend(
                EnvironmentOperation(operation="delete", path=(*path, str(key)))
                for key in sorted(previous.keys() - current.keys())
            )
            for key in sorted(current):
                if key not in previous:
                    operations.append(
                        EnvironmentOperation(
                            operation="set",
                            path=(*path, str(key)),
                            value=_json_copy(current[key]),
                        )
                    )
                else:
                    visit((*path, str(key)), previous[key], current[key])
            return
        if not path:
            raise ValueError("the environment namespace must remain a JSON object")
        operations.append(
            EnvironmentOperation(operation="set", path=path, value=_json_copy(current))
        )

    visit((), before, after)
    return tuple(operations)


def _capture_workspace(
    environment: Environment,
    state: ExecutionState,
    action: Action,
) -> WorkspaceDelta | None:
    capture = getattr(environment, "capture_workspace", None)
    if capture is None:
        return None
    workspace = capture(state.workspace, created_by_action=action.id)
    if workspace is None or (
        workspace.files == state.workspace.files
        and workspace.artifacts == state.workspace.artifacts
    ):
        return None
    return WorkspaceDelta.from_workspace(workspace)


def execute_action(
    environment: Environment,
    state: ExecutionState,
    action: Action,
) -> ToolEffects:
    """Execute an already-committed action and return its atomic typed effects."""

    if state.is_terminal:
        raise ValueError("cannot execute an action from a terminal execution")
    action_state = state.actions.get(action.id)
    if action_state is None or action_state.action != action:
        raise ValueError("action must already be recorded in ExecutionState")
    if action_state.status not in {"pending", "running"}:
        raise ValueError("action has already reached a terminal status")

    prepare_workspace = getattr(environment, "prepare_workspace", None)
    if prepare_workspace is not None:
        prepare_workspace(state.workspace)

    started = time.perf_counter()
    before_environment = state.environment.values
    next_environment: Mapping[str, Any] = before_environment

    if action.name == SUBMIT_ANSWER_TOOL_NAME:
        answer = action.arguments.get("answer")
        if set(action.arguments) != {"answer"} or not isinstance(answer, str):
            status = ToolCallStatus.INVALID_ARGS
            content: Any = (
                "submit_answer requires exactly one string argument named 'answer'"
            )
        else:
            raw_submitters = state.runtime.metadata.get("submitters")
            allowed = raw_submitters is None or (
                isinstance(raw_submitters, list | tuple)
                and action.actor_id is not None
                and action.actor_id in raw_submitters
            )
            if raw_submitters is not None and not isinstance(
                raw_submitters, list | tuple
            ):
                raise ValueError("RuntimeState.metadata.submitters must be a list")
            if not allowed:
                status = ToolCallStatus.EXECUTION_ERROR
                content = f"actor {action.actor_id!r} is not allowed to submit"
            else:
                status = ToolCallStatus.SUCCESS
                content = "answer accepted"
    else:
        tool = environment.tools.get(action.name)
        if tool is None:
            status = ToolCallStatus.INVALID_TOOL
            content = f"Tool {action.name} not found"
        elif permissions.enabled() and (
            error := permissions.visible_argument_error(tool, dict(action.arguments))
        ):
            status = ToolCallStatus.INVALID_ARGS
            content = error
        else:
            visible_arguments = environment.preprocess_arguments(
                action.name, dict(action.arguments)
            )
            call_arguments = dict(visible_arguments)
            hidden_values = _hidden_arguments(state)
            missing: list[str] = []
            for hidden_name in tool.hidden_args:
                if hidden_name == CORRAL_ACTION_ID_ARGUMENT:
                    call_arguments[hidden_name] = action.id
                elif hidden_name not in hidden_values:
                    missing.append(hidden_name)
                else:
                    call_arguments[hidden_name] = hidden_values[hidden_name]
            if missing:
                status = ToolCallStatus.INVALID_ARGS
                content = (
                    f"Hidden argument(s) {', '.join(repr(name) for name in missing)} "
                    f"required by tool {action.name!r} are not configured."
                )
            else:
                valid, validation_error = tool.validate_arguments(call_arguments)
                if not valid:
                    status = ToolCallStatus.INVALID_ARGS
                    content = validation_error or "invalid tool arguments"
                else:
                    try:
                        guard = getattr(environment, "execution_guard", None)
                        context = guard(tool) if guard is not None else nullcontext()
                        with context:
                            raw_result = (
                                permissions.execute_tool(
                                    environment, state, tool, call_arguments
                                )
                                if permissions.enabled()
                                else environment.execute_tool(
                                    state, tool, call_arguments
                                )
                            )
                        if isinstance(raw_result, ToolExecutionResult):
                            content = raw_result.content
                            if raw_result.environment is not None:
                                next_environment = raw_result.environment
                        else:
                            content = raw_result
                        status = ToolCallStatus.SUCCESS
                    except Exception as exc:  # tool failures remain observations
                        status = ToolCallStatus.EXECUTION_ERROR
                        content = str(exc)

    capture_environment = getattr(environment, "capture_environment", None)
    if capture_environment is not None:
        next_environment = capture_environment(next_environment)
    operations = environment_operations(before_environment, next_environment)
    workspace_delta = _capture_workspace(environment, state, action)
    observation = _json_copy(content)
    return ToolEffects(
        observation=observation,
        status=status.value,
        environment_operations=operations,
        workspace_delta=workspace_delta,
        expected_environment_revision=(
            state.environment.revision if operations else None
        ),
        expected_workspace_revision=(
            state.workspace.revision if workspace_delta is not None else None
        ),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )


__all__ = [
    "CORRAL_ACTION_ID_ARGUMENT",
    "HIDDEN_ARGUMENTS_NAMESPACE",
    "ToolEffects",
    "ToolExecutionResult",
    "environment_operations",
    "execute_action",
    "propose_action",
]
