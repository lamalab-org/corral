"""Immutable environment transitions.

The functions in this module are the execution boundary introduced by PR 4.
An :class:`~corral.core.environment.Environment` is a definition and tool catalog;
all task-execution data is supplied in :class:`~corral.core.state.State` and a
tool call produces a complete child State.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.state import RuntimeState, State, UsageState
from corral.core.tool import ToolCallStatus

if TYPE_CHECKING:
    from corral.core.environment import Environment
    from corral.core.workspace import WorkspaceState

HIDDEN_ARGUMENTS_NAMESPACE = "hidden_arguments"
# A tool may declare this as a hidden argument to receive the stable Action ID
# as an API/instrument idempotency key. It is runtime-owned and never supplied
# by the agent or persisted in the general hidden-argument namespace.
CORRAL_ACTION_ID_ARGUMENT = "corral_action_id"


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """A tool observation plus an optional complete environment namespace.

    Most tools only return content. A stateful tool returns this value with a
    complete JSON-serializable ``environment`` replacement; it never receives
    the mutable State object and never patches State in place.
    """

    content: Any
    environment: Mapping[str, Any] | None = None


def propose_action(
    state: State,
    action: Action,
    *,
    usage: UsageState | Mapping[str, Any] | None = None,
) -> State:
    """Record an agent action in the in-memory immutable State chain."""
    if state.pending_action is not None:
        raise ValueError("cannot propose an action while another action is pending")
    if any(previous.id == action.id for previous in state.actions):
        raise ValueError(f"action id {action.id!r} is already present in State")
    return state.fork(
        messages=(*state.messages, action.to_message()),
        usage=usage,
    )


def _json_copy(value: Any) -> Any:
    """Return a mutable JSON copy without leaking references out of State."""
    return json.loads(json.dumps(value, allow_nan=False))


def _hidden_arguments(state: State) -> dict[str, Any]:
    value = state.environment.get(HIDDEN_ARGUMENTS_NAMESPACE, {})
    if not isinstance(value, Mapping):
        raise ValueError(
            f"State.environment.{HIDDEN_ARGUMENTS_NAMESPACE} must be an object"
        )
    return _json_copy(value)


def _tool_message(
    action: Action,
    *,
    content: str,
    status: ToolCallStatus,
    duration: float,
) -> dict[str, Any]:
    success = status is ToolCallStatus.SUCCESS
    return {
        "role": "tool",
        "tool_call_id": action.id,
        "name": action.name,
        "content": content,
        "metadata": {
            "status": status.value,
            "success": success,
            "duration_ms": round(duration * 1000, 3),
        },
    }


def _usage_after_tool(state: State) -> UsageState:
    return UsageState(
        input_tokens=state.usage.input_tokens,
        output_tokens=state.usage.output_tokens,
        llm_calls=state.usage.llm_calls,
        tool_calls=state.usage.tool_calls + 1,
        agent_steps=state.usage.agent_steps,
        metadata=state.usage.metadata,
    )


def _runtime_after_submission(state: State, answer: str) -> RuntimeState:
    surrendered = bool(
        state.runtime.metadata.get("surrender_sentinel")
    ) and answer == str(state.runtime.metadata["surrender_sentinel"])
    return RuntimeState(
        status="surrendered" if surrendered else "submitted",
        started_at=state.runtime.started_at,
        ended_at=datetime.now(timezone.utc),
        metadata=state.runtime.metadata,
    )


def _submission_allowed(state: State, action: Action) -> bool:
    raw_submitters = state.runtime.metadata.get("submitters")
    if raw_submitters is None:
        return True
    if not isinstance(raw_submitters, list | tuple):
        raise ValueError("State.runtime.metadata.submitters must be a list")
    return action.actor_id is not None and action.actor_id in raw_submitters


def _capture_workspace(
    environment: Environment,
    state: State,
    action: Action,
) -> WorkspaceState:
    capture = getattr(environment, "capture_workspace", None)
    if capture is None:
        return state.workspace
    workspace = capture(state.workspace, created_by_action=action.id)
    return state.workspace if workspace is None else workspace


def execute_action(environment: Environment, state: State, action: Action) -> State:
    """Execute one pending Action and return exactly one immutable child.

    ``state`` must already contain ``action`` as its sole pending action. The
    runtime durably commits that before-tool State, then this function records
    the observation and the runtime commits the returned after-tool child. The
    parent is never mutated, including its nested hidden-tool namespace.
    """
    if not isinstance(state, State):
        raise TypeError("execute_action() requires State v2")
    if not isinstance(action, Action):
        raise TypeError("execute_action() requires Action")
    if state.pending_action != action:
        raise ValueError("action must be the pending action in State")
    if state.is_terminal:
        raise ValueError("cannot execute an action from a terminal State")

    prepare_workspace = getattr(environment, "prepare_workspace", None)
    if prepare_workspace is not None:
        prepare_workspace(state.workspace)

    started = time.perf_counter()
    next_environment: Mapping[str, Any] = state.environment
    next_runtime = state.runtime

    if action.name == SUBMIT_ANSWER_TOOL_NAME:
        answer = action.arguments.get("answer")
        if set(action.arguments) != {"answer"} or not isinstance(answer, str):
            status = ToolCallStatus.INVALID_ARGS
            content = (
                "submit_answer requires exactly one string argument named 'answer'"
            )
        elif not _submission_allowed(state, action):
            status = ToolCallStatus.EXECUTION_ERROR
            content = f"actor {action.actor_id!r} is not allowed to submit"
        else:
            status = ToolCallStatus.SUCCESS
            content = "answer accepted"
            next_runtime = _runtime_after_submission(state, answer)
    else:
        tool = environment.tools.get(action.name)
        if tool is None:
            status = ToolCallStatus.INVALID_TOOL
            content = f"Tool {action.name} not found"
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
                            raw_result = environment.execute_tool(
                                state,
                                tool,
                                call_arguments,
                            )
                        if isinstance(raw_result, ToolExecutionResult):
                            content_value = raw_result.content
                            if raw_result.environment is not None:
                                next_environment = raw_result.environment
                        else:
                            content_value = raw_result
                        content = (
                            content_value
                            if isinstance(content_value, str)
                            else json.dumps(
                                content_value,
                                allow_nan=False,
                                ensure_ascii=False,
                                default=str,
                            )
                        )
                        status = ToolCallStatus.SUCCESS
                    except Exception as exc:  # tool failures are observations
                        status = ToolCallStatus.EXECUTION_ERROR
                        content = str(exc)

    capture_environment = getattr(environment, "capture_environment", None)
    if capture_environment is not None:
        next_environment = capture_environment(next_environment)

    duration = time.perf_counter() - started
    next_workspace = _capture_workspace(environment, state, action)
    return state.fork(
        messages=(
            *state.messages,
            _tool_message(
                action,
                content=content,
                status=status,
                duration=duration,
            ),
        ),
        environment=next_environment,
        workspace=next_workspace,
        usage=_usage_after_tool(state),
        runtime=next_runtime,
    )


__all__ = [
    "CORRAL_ACTION_ID_ARGUMENT",
    "HIDDEN_ARGUMENTS_NAMESPACE",
    "ToolExecutionResult",
    "execute_action",
    "propose_action",
]
