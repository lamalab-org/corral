"""The single authorization and execution boundary for Corral tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from corral.core.resources import RESOURCE_STATE_NAMESPACE, MaterializedResourcePath
from corral.core.transition import (
    CORRAL_ACTION_ID_ARGUMENT,
    HIDDEN_ARGUMENTS_NAMESPACE,
    ToolExecutionResult,
)
from corral.runtime import permissions
from corral.workspace import PUBLIC_WORKSPACE_ROOT, materialize_local_tool_arguments

if TYPE_CHECKING:
    import threading

    from corral.core.environment import Environment
    from corral.core.state import ExecutionState
    from corral.core.tool import Tool


ExecutionKind = Literal["controller", "restricted", "local"]


class ToolArgumentError(ValueError):
    """An invocation failed before any tool code was executed."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def _resource_json(value: Any) -> Any:
    """Detach resource state while rejecting non-JSON runtime objects."""
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """A validated invocation with authority frozen at preparation time.

    The live `Tool` remains only as executable code. Every property that can
    affect routing or OS capabilities is copied into this frozen value and is
    never re-read from the tool when the call eventually executes.
    """

    tool: Tool = field(repr=False, compare=False)
    tool_name: str
    action_id: str
    execution_kind: ExecutionKind
    trusted: bool
    controller_dispatch: bool
    workspace_access: str
    workspace_args: tuple[str, ...]
    hidden_arg_names: tuple[str, ...]
    resources: tuple[str, ...]
    worker_operation: str | None
    workspace: str | None = field(repr=False)
    executor: str | None
    concurrency_key: str | None
    _argument_items: tuple[tuple[str, Any], ...] = field(repr=False)
    _visible_argument_items: tuple[tuple[str, Any], ...]
    _resource_mount_items: tuple[tuple[str, tuple[str, str]], ...] = field(repr=False)
    _stateful_items: tuple[tuple[str, object], ...] = field(repr=False)

    @classmethod
    def capture(
        cls,
        tool: Tool,
        arguments: Mapping[str, Any],
        *,
        visible_arguments: Mapping[str, Any] | None = None,
        action_id: str = "",
        workspace: str | None = None,
        resource_mounts: Mapping[str, tuple[str, str]] | None = None,
        stateful: Mapping[str, object] | None = None,
        execution_kind: ExecutionKind | None = None,
    ) -> PreparedToolCall:
        """Detach an already-authorized call and snapshot its execution policy.

        `ToolExecutor.prepare` is the public authorization path. This lower
        level constructor is also useful to exercise job infrastructure with
        calls whose arguments have already been validated by the caller.
        """
        hidden_arg_names = tuple(sorted(getattr(tool, "hidden_args", ())))
        trusted = bool(getattr(tool, "trusted", False))
        controller_dispatch = bool(getattr(tool, "controller_dispatch", False))
        if execution_kind is None:
            if trusted or controller_dispatch:
                execution_kind = "controller"
            elif permissions.enabled():
                execution_kind = "restricted"
            else:
                execution_kind = "local"
        visible = _json_copy(
            dict(visible_arguments)
            if visible_arguments is not None
            else {
                name: value
                for name, value in arguments.items()
                if name not in hidden_arg_names
            }
        )
        stateful_values = dict(stateful or {})
        detached_arguments = {
            name: value if name in stateful_values else deepcopy(value)
            for name, value in arguments.items()
        }
        detached_arguments.update(visible)
        access = getattr(tool, "workspace_access", "none")
        access = access.value if hasattr(access, "value") else str(access)
        return cls(
            tool=tool,
            tool_name=tool.name,
            action_id=action_id,
            execution_kind=execution_kind,
            trusted=trusted,
            controller_dispatch=controller_dispatch,
            workspace_access=access,
            workspace_args=tuple(getattr(tool, "workspace_args", ())),
            hidden_arg_names=hidden_arg_names,
            resources=tuple(getattr(tool, "resources", ())),
            worker_operation=getattr(tool, "worker_operation", None),
            workspace=workspace,
            executor=getattr(tool, "executor", None),
            concurrency_key=getattr(tool, "concurrency_key", None),
            _argument_items=tuple(detached_arguments.items()),
            _visible_argument_items=tuple(visible.items()),
            _resource_mount_items=tuple(dict(resource_mounts or {}).items()),
            _stateful_items=tuple(stateful_values.items()),
        )

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a fresh top-level argument mapping for execution."""
        stateful = self.stateful
        return {
            name: value if name in stateful else deepcopy(value)
            for name, value in self._argument_items
        }

    @property
    def visible_arguments(self) -> dict[str, Any]:
        """Return the agent-supplied, report-safe arguments."""
        return _json_copy(dict(self._visible_argument_items))

    @property
    def resource_mounts(self) -> dict[str, tuple[str, str]]:
        return dict(self._resource_mount_items)

    @property
    def stateful(self) -> dict[str, object]:
        return dict(self._stateful_items)


def execute_prepared_call(
    prepared: PreparedToolCall,
    *,
    cancel: threading.Event | None = None,
) -> Any:
    """Execute a frozen call that does not require Environment dispatch."""
    arguments = prepared.arguments
    if prepared.execution_kind == "controller":
        if prepared.controller_dispatch:
            raise RuntimeError(
                f"Tool {prepared.tool_name!r} requires state-aware controller dispatch"
            )
        if not prepared.trusted:  # pragma: no cover - constructor invariant
            raise PermissionError(
                f"Tool {prepared.tool_name!r} is not trusted for controller execution"
            )
        return prepared.tool.execute(**arguments)
    if prepared.execution_kind == "restricted":
        return permissions.execute_restricted_tool(
            prepared.tool,
            arguments,
            prepared.workspace or "",
            resource_mounts=prepared.resource_mounts,
            cancel=cancel,
            prepared=prepared,
        )
    local_arguments = materialize_local_tool_arguments(
        prepared.tool,
        arguments,
        prepared.workspace,
        prepared=prepared,
    )
    return prepared.tool.execute(**local_arguments)


def execute_prepared_background_call(
    prepared: PreparedToolCall,
    *,
    cancel: threading.Event | None = None,
) -> Any:
    """Execute a prepared job while forbidding uncommittable state changes."""
    result = execute_prepared_call(prepared, cancel=cancel)
    if isinstance(result, ToolExecutionResult):
        if result.environment is not None:
            raise ValueError("stateful task tools must execute in the foreground")
        return result.content
    return result


@dataclass(slots=True)
class ToolExecutor:
    """Validate policy, prepare capabilities, execute, and capture state."""

    environment: Environment

    def _hidden_values(self, state: ExecutionState) -> dict[str, Any]:
        value = state.environment.values.get(HIDDEN_ARGUMENTS_NAMESPACE, {})
        if not isinstance(value, Mapping):
            raise ToolArgumentError(
                f"ExecutionState.environment.{HIDDEN_ARGUMENTS_NAMESPACE} "
                "must be an object"
            )
        return _json_copy(value)

    def _prepare_arguments(
        self,
        state: ExecutionState,
        tool: Tool,
        visible_arguments: dict[str, Any],
        *,
        action_id: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, object],
        dict[str, Any],
    ]:
        # Provider adapters may normalize JSON-looking strings first. The
        # normalized public value is then validated before any private value is
        # injected, and the exact same value is retained for job provenance.
        visible = self.environment.preprocess_arguments(
            tool.name, dict(visible_arguments)
        )
        if error := permissions.visible_argument_error(tool, visible):
            raise ToolArgumentError(error)
        arguments = dict(visible)
        hidden_values = self._hidden_values(state)
        stateful: dict[str, object] = {}
        file_handles = self.environment.materialize_file_resources(tool.resources)
        configured = set(self.environment.resource_adapters) | set(
            self.environment.file_resources
        )
        missing_resources = sorted(set(tool.resources) - configured)
        if missing_resources:
            raise ToolArgumentError(
                f"Tool {tool.name!r} declares unconfigured resource(s): "
                f"{missing_resources}"
            )

        resource_state = state.environment.values.get(RESOURCE_STATE_NAMESPACE, {})
        if not isinstance(resource_state, Mapping):
            raise ToolArgumentError(
                f"ExecutionState.environment.{RESOURCE_STATE_NAMESPACE} must be an object"
            )
        controller_side = tool.trusted or getattr(tool, "controller_dispatch", False)
        for name in tool.resources:
            adapter = self.environment.resource_adapters.get(name)
            if adapter is not None:
                if not tool.trusted:
                    raise PermissionError(
                        f"stateful resource {name!r} requires a trusted tool"
                    )
                if name not in resource_state:
                    raise ToolArgumentError(
                        f"Tool {tool.name!r} requires missing resource state {name!r}"
                    )
                runtime = adapter.restore(_resource_json(resource_state[name]))
                stateful[name] = runtime
                if name in tool.hidden_args:
                    arguments[name] = runtime
            elif name in file_handles and name in tool.hidden_args:
                handle = file_handles[name]
                arguments[name] = (
                    handle
                    if controller_side
                    else (
                        handle.public_path
                        if permissions.enabled()
                        else MaterializedResourcePath(str(handle.controller_path))
                    )
                )

        missing: list[str] = []
        for name in tool.hidden_args:
            if name in arguments:
                continue
            if name == CORRAL_ACTION_ID_ARGUMENT:
                arguments[name] = action_id
            elif name in tool.workspace_args:
                arguments[name] = (
                    self.environment.workspace_path
                    if controller_side or not permissions.enabled()
                    else PUBLIC_WORKSPACE_ROOT
                )
            elif name not in hidden_values:
                missing.append(name)
            else:
                arguments[name] = hidden_values[name]
        if missing:
            raise ToolArgumentError(
                f"Hidden argument(s) {', '.join(repr(name) for name in missing)} "
                f"required by tool {tool.name!r} are not configured."
            )
        valid, error = tool.validate_arguments(arguments)
        if not valid:
            raise ToolArgumentError(error or "invalid tool arguments")
        return arguments, visible, stateful, file_handles

    def prepare(
        self,
        state: ExecutionState,
        tool: Tool,
        visible_arguments: dict[str, Any],
        *,
        action_id: str,
    ) -> PreparedToolCall:
        """Authorize and freeze one invocation without executing it."""
        # Validate on every call, not only when the session is constructed: a
        # mutable Environment must not be able to widen permissions mid-run.
        self.environment.validate_state_tool_catalog(state)
        arguments, visible, stateful, file_handles = self._prepare_arguments(
            state, tool, visible_arguments, action_id=action_id
        )
        return PreparedToolCall.capture(
            tool,
            arguments,
            visible_arguments=visible,
            action_id=action_id,
            workspace=self.environment.workspace_path,
            resource_mounts={
                name: (str(handle.controller_path), handle.public_path)
                for name, handle in file_handles.items()
            },
            stateful=stateful,
        )

    def execute_prepared(
        self, state: ExecutionState, prepared: PreparedToolCall
    ) -> Any:
        """Execute one previously authorized invocation."""
        if prepared.execution_kind == "controller":
            raw = self.environment.execute_controller_tool(state, prepared)
        else:
            raw = execute_prepared_call(prepared)

        environment = (
            dict(raw.environment)
            if isinstance(raw, ToolExecutionResult) and raw.environment is not None
            else dict(state.environment.values)
        )
        stateful = prepared.stateful
        if stateful:
            captured = dict(environment.get(RESOURCE_STATE_NAMESPACE, {}))
            for name, runtime in stateful.items():
                captured[name] = _resource_json(
                    self.environment.resource_adapters[name].capture(runtime)
                )
            environment[RESOURCE_STATE_NAMESPACE] = captured

        content = raw.content if isinstance(raw, ToolExecutionResult) else raw
        content = self.environment.normalize_public_paths(content)
        if stateful or (
            isinstance(raw, ToolExecutionResult) and raw.environment is not None
        ):
            return ToolExecutionResult(content=content, environment=environment)
        return content

    def execute(
        self,
        state: ExecutionState,
        tool: Tool,
        visible_arguments: dict[str, Any],
        *,
        action_id: str,
    ) -> Any:
        """Prepare and immediately execute one invocation."""
        prepared = self.prepare(state, tool, visible_arguments, action_id=action_id)
        return self.execute_prepared(state, prepared)


__all__ = [
    "ExecutionKind",
    "PreparedToolCall",
    "ToolArgumentError",
    "ToolExecutor",
    "execute_prepared_background_call",
    "execute_prepared_call",
]
