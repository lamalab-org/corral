"""Immutable snapshots of the exact tool catalog exposed to an agent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue, model_validator

from corral.core._immutable import FrozenModel, validate_sha256_hex

if TYPE_CHECKING:
    from corral.core.state import ExecutionState


TOOL_CATALOG_METADATA_KEY = "tool_catalog"
TOOL_CATALOG_SCHEMA_VERSION = 1
TOOL_POLICY_METADATA_KEY = "tool_policy"
TOOL_POLICY_SCHEMA_VERSION = 1


class ToolCatalogBindingError(RuntimeError):
    """A persisted projection cannot use the supplied Environment catalog."""


class MissingToolCatalogError(ToolCatalogBindingError):
    """A projection is missing its authoritative catalog snapshot."""


class InvalidToolCatalogError(ToolCatalogBindingError):
    """A persisted catalog snapshot is malformed or internally inconsistent."""


class ToolCatalogMismatchError(ToolCatalogBindingError):
    """The persisted and currently executable tool catalogs differ."""


class MissingToolPolicyError(ToolCatalogBindingError):
    """A projection is missing its authoritative private tool policy."""


class InvalidToolPolicyError(ToolCatalogBindingError):
    """A persisted private tool policy is malformed."""


class ToolPolicyMismatchError(ToolCatalogBindingError):
    """The persisted and currently executable private tool policies differ."""


def _tool_name(tool: Mapping[str, Any], *, index: int) -> str:
    if tool.get("type") != "function":
        raise ValueError(f"tool catalog entry {index} must have type='function'")
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError(f"tool catalog entry {index} has no function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"tool catalog entry {index} has no function name")
    return name


def canonical_tool_catalog_json(
    tools: Sequence[Mapping[str, Any]],
) -> str:
    """Return a stable, order-independent representation for fingerprinting.

    Tool order is intentionally excluded from compatibility: the persisted
    snapshot remains authoritative for presentation order, while Environment
    execution dispatches by unique tool name. Every schema value remains part
    of the fingerprint.
    """
    named: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"tool catalog entry {index} must be an object")
        name = _tool_name(tool, index=index)
        if name in seen:
            raise ValueError(f"tool catalog contains duplicate tool {name!r}")
        seen.add(name)
        named.append((name, tool))
    return json.dumps(
        [tool for _name, tool in sorted(named, key=lambda item: item[0])],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def tool_catalog_fingerprint(tools: Sequence[Mapping[str, Any]]) -> str:
    """Return the SHA-256 fingerprint of a complete exposed tool catalog."""
    payload = canonical_tool_catalog_json(tools)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolCatalogSnapshot(FrozenModel):
    """Versioned catalog value persisted by the execution-start commit."""

    schema_version: Literal[1] = TOOL_CATALOG_SCHEMA_VERSION
    tools: tuple[Mapping[str, JsonValue], ...]
    fingerprint: str

    @model_validator(mode="after")
    def _validate_fingerprint(self) -> ToolCatalogSnapshot:
        validate_sha256_hex(self.fingerprint, field_name="tool catalog fingerprint")
        expected = tool_catalog_fingerprint(self.tools)
        if self.fingerprint != expected:
            raise ValueError(
                "tool catalog fingerprint does not match the persisted tools"
            )
        return self

    @classmethod
    def capture(
        cls,
        tools: Sequence[Mapping[str, Any]],
    ) -> ToolCatalogSnapshot:
        """Detach and fingerprint one complete exposed catalog."""
        # JSON round-tripping rejects non-portable provider schemas and prevents
        # later mutations of Environment-owned dictionaries from changing the
        # persisted snapshot.
        detached = json.loads(
            json.dumps(
                list(tools),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if not isinstance(detached, list):  # pragma: no cover - defensive
            raise TypeError("tool catalog must serialize as a list")
        return cls(
            tools=tuple(detached),
            fingerprint=tool_catalog_fingerprint(detached),
        )

    def detached_tools(self) -> tuple[dict[str, Any], ...]:
        """Return mutable copies without consulting a live Environment."""
        detached = json.loads(
            json.dumps(
                self.tools,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return tuple(detached)

    def mcp_tools(self) -> tuple[dict[str, Any], ...]:
        """Describe this exact catalog in MCP format without changing schemas."""
        return tuple(
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "inputSchema": tool["function"].get("parameters", {}),
            }
            for tool in self.detached_tools()
        )


class ToolPolicy(FrozenModel):
    """Private execution policy for one tool.

    These fields are deliberately kept out of provider/MCP schemas: they are
    controller authorization data, not arguments the agent may influence.
    """

    trusted: bool = False
    workspace_access: Literal["none", "read", "read_write"] = "none"
    hidden_args: tuple[str, ...] = ()
    workspace_args: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_bindings(self) -> ToolPolicy:
        if len(set(self.hidden_args)) != len(self.hidden_args):
            raise ValueError("hidden_args cannot contain duplicates")
        if len(set(self.workspace_args)) != len(self.workspace_args):
            raise ValueError("workspace_args cannot contain duplicates")
        if not set(self.workspace_args) <= set(self.hidden_args):
            raise ValueError("workspace_args must name hidden_args")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("resources cannot contain duplicates")
        return self


def canonical_tool_policy_json(policies: Mapping[str, Any]) -> str:
    """Return a stable representation of controller-only tool policies."""
    return json.dumps(
        {name: policies[name] for name in sorted(policies)},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def tool_policy_fingerprint(policies: Mapping[str, Any]) -> str:
    payload = canonical_tool_policy_json(policies)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolPolicySnapshot(FrozenModel):
    """Versioned private policy persisted with an execution's tool catalog."""

    schema_version: Literal[1] = TOOL_POLICY_SCHEMA_VERSION
    policies: Mapping[str, ToolPolicy]
    fingerprint: str

    @model_validator(mode="after")
    def _validate_fingerprint(self) -> ToolPolicySnapshot:
        validate_sha256_hex(self.fingerprint, field_name="tool policy fingerprint")
        payload = {
            name: policy.model_dump(mode="json")
            for name, policy in self.policies.items()
        }
        expected = tool_policy_fingerprint(payload)
        if self.fingerprint != expected:
            raise ValueError(
                "tool policy fingerprint does not match the persisted policies"
            )
        return self

    @classmethod
    def capture(cls, tools: Mapping[str, Any]) -> ToolPolicySnapshot:
        policies: dict[str, dict[str, Any]] = {}
        for name, tool in tools.items():
            if name != getattr(tool, "name", None):
                raise ValueError(f"tool policy key {name!r} does not match tool name")
            policies[name] = ToolPolicy(
                trusted=bool(getattr(tool, "trusted", False)),
                workspace_access=str(
                    getattr(tool, "workspace_access", "none").value
                    if hasattr(getattr(tool, "workspace_access", "none"), "value")
                    else getattr(tool, "workspace_access", "none")
                ),
                hidden_args=tuple(sorted(getattr(tool, "hidden_args", {}))),
                workspace_args=tuple(sorted(getattr(tool, "workspace_args", ()))),
                resources=tuple(sorted(getattr(tool, "resources", ()))),
            ).model_dump(mode="json")
        return cls(policies=policies, fingerprint=tool_policy_fingerprint(policies))


def state_tool_catalog(state: ExecutionState) -> ToolCatalogSnapshot:
    """Read and validate the authoritative catalog persisted on `state`."""
    raw = state.task.environment.get(TOOL_CATALOG_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise MissingToolCatalogError(
            f"Execution {state.execution_id!r} at {state.through_commit_hash} has no "
            "persisted tool catalog snapshot. Corral will not derive one from "
            "the current Environment. This execution does not implement the "
            "current commit protocol and cannot be run."
        )
    try:
        return ToolCatalogSnapshot.model_validate(raw)
    except Exception as exc:
        raise InvalidToolCatalogError(
            f"Execution {state.execution_id!r} at {state.through_commit_hash} has an invalid "
            f"persisted tool catalog snapshot: {exc}"
        ) from exc


def state_tool_policy(state: ExecutionState) -> ToolPolicySnapshot:
    """Read and validate the authoritative private policy on `state`."""
    raw = state.task.environment.get(TOOL_POLICY_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise MissingToolPolicyError(
            f"Execution {state.execution_id!r} at {state.through_commit_hash} has no "
            "persisted tool policy snapshot. Refusing to derive controller "
            "authorization from the current Environment."
        )
    try:
        return ToolPolicySnapshot.model_validate(raw)
    except Exception as exc:
        raise InvalidToolPolicyError(
            f"Execution {state.execution_id!r} at {state.through_commit_hash} has an "
            f"invalid persisted tool policy snapshot: {exc}"
        ) from exc


def validate_tool_catalog_binding(
    state: ExecutionState,
    current: ToolCatalogSnapshot,
) -> ToolCatalogSnapshot:
    """Validate the commit-projected catalog for one Environment binding."""
    stored = state_tool_catalog(state)
    if stored.fingerprint != current.fingerprint:
        stored_by_name = {
            _tool_name(tool, index=index): tool
            for index, tool in enumerate(stored.tools)
        }
        current_by_name = {
            _tool_name(tool, index=index): tool
            for index, tool in enumerate(current.tools)
        }
        stored_only = sorted(stored_by_name.keys() - current_by_name.keys())
        current_only = sorted(current_by_name.keys() - stored_by_name.keys())
        changed = sorted(
            name
            for name in stored_by_name.keys() & current_by_name.keys()
            if json.dumps(
                stored_by_name[name],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            != json.dumps(
                current_by_name[name],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        raise ToolCatalogMismatchError(
            f"Tool catalog mismatch for execution {state.execution_id!r} at "
            f"{state.through_commit_hash}: stored fingerprint={stored.fingerprint}, "
            f"current Environment fingerprint={current.fingerprint}; "
            f"stored_only={stored_only}, current_only={current_only}, "
            f"changed={changed}. Refusing "
            "to bind the execution because the agent schema and executable "
            "tools could diverge."
        )
    return stored


def validate_tool_policy_binding(
    state: ExecutionState,
    current: ToolPolicySnapshot,
) -> ToolPolicySnapshot:
    """Refuse resumed execution if any controller-only permission changed."""
    stored = state_tool_policy(state)
    if stored.fingerprint != current.fingerprint:
        stored_names = set(stored.policies)
        current_names = set(current.policies)
        changed = sorted(
            name
            for name in stored_names & current_names
            if stored.policies[name] != current.policies[name]
        )
        raise ToolPolicyMismatchError(
            f"Tool policy mismatch for execution {state.execution_id!r} at "
            f"{state.through_commit_hash}: stored fingerprint={stored.fingerprint}, "
            f"current Environment fingerprint={current.fingerprint}; "
            f"stored_only={sorted(stored_names - current_names)}, "
            f"current_only={sorted(current_names - stored_names)}, "
            f"changed={changed}. Refusing to resume with different permissions."
        )
    return stored


__all__ = [
    "TOOL_CATALOG_METADATA_KEY",
    "TOOL_CATALOG_SCHEMA_VERSION",
    "TOOL_POLICY_METADATA_KEY",
    "TOOL_POLICY_SCHEMA_VERSION",
    "InvalidToolCatalogError",
    "InvalidToolPolicyError",
    "MissingToolCatalogError",
    "MissingToolPolicyError",
    "ToolCatalogBindingError",
    "ToolCatalogMismatchError",
    "ToolCatalogSnapshot",
    "ToolPolicy",
    "ToolPolicyMismatchError",
    "ToolPolicySnapshot",
    "canonical_tool_catalog_json",
    "canonical_tool_policy_json",
    "state_tool_catalog",
    "state_tool_policy",
    "tool_catalog_fingerprint",
    "tool_policy_fingerprint",
    "validate_tool_catalog_binding",
    "validate_tool_policy_binding",
]
