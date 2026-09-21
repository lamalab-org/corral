"""The single authorization and execution boundary for Corral tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from corral.core.resources import RESOURCE_STATE_NAMESPACE, MaterializedResourcePath
from corral.core.transition import (
    CORRAL_ACTION_ID_ARGUMENT,
    HIDDEN_ARGUMENTS_NAMESPACE,
    ToolExecutionResult,
)
from corral.runtime import permissions
from corral.workspace import PUBLIC_WORKSPACE_ROOT, materialize_local_tool_arguments

if TYPE_CHECKING:
    from corral.core.environment import Environment
    from corral.core.state import ExecutionState
    from corral.core.tool import Tool


class ToolArgumentError(ValueError):
    """An invocation failed before any tool code was executed."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def _resource_json(value: Any) -> Any:
    """Detach resource state while rejecting non-JSON runtime objects."""
    return json.loads(json.dumps(value, allow_nan=False))


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
    ) -> tuple[dict[str, Any], dict[str, object], dict[str, Any]]:
        arguments = self.environment.preprocess_arguments(
            tool.name, dict(visible_arguments)
        )
        if error := permissions.visible_argument_error(tool, arguments):
            raise ToolArgumentError(error)
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
                    if tool.trusted
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
                    if tool.trusted or not permissions.enabled()
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
        return arguments, stateful, file_handles

    def execute(
        self,
        state: ExecutionState,
        tool: Tool,
        visible_arguments: dict[str, Any],
        *,
        action_id: str,
    ) -> Any:
        """Execute one invocation under its persisted private policy."""
        # Validate on every call, not only when the session is constructed: a
        # mutable Environment must not be able to widen permissions mid-run.
        self.environment.validate_state_tool_catalog(state)
        arguments, stateful, file_handles = self._prepare_arguments(
            state, tool, visible_arguments, action_id=action_id
        )
        resource_mounts = {
            name: (str(handle.controller_path), handle.public_path)
            for name, handle in file_handles.items()
        }
        if tool.trusted:
            raw = self.environment.execute_trusted_tool(state, tool, arguments)
        elif permissions.enabled():
            raw = permissions.execute_restricted_tool(
                tool,
                arguments,
                self.environment.workspace_path,
                resource_mounts=resource_mounts,
            )
        else:
            local_arguments = materialize_local_tool_arguments(
                tool,
                arguments,
                self.environment.workspace_path,
            )
            raw = tool.execute(**local_arguments)

        environment = (
            dict(raw.environment)
            if isinstance(raw, ToolExecutionResult) and raw.environment is not None
            else dict(state.environment.values)
        )
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


__all__ = ["ToolArgumentError", "ToolExecutor"]
