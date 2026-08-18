"""Immutable snapshots of the exact tool catalog exposed to an agent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue, model_validator

from corral.core._immutable import FrozenModel, validate_sha256_hex

if TYPE_CHECKING:
    from corral.core.state import State


TOOL_CATALOG_METADATA_KEY = "tool_catalog"
TOOL_CATALOG_SCHEMA_VERSION = 1


class ToolCatalogBindingError(RuntimeError):
    """A persisted State cannot safely use the supplied Environment catalog."""


class MissingToolCatalogError(ToolCatalogBindingError):
    """A State is missing its required authoritative catalog snapshot."""


class InvalidToolCatalogError(ToolCatalogBindingError):
    """A persisted catalog snapshot is malformed or internally inconsistent."""


class ToolCatalogMismatchError(ToolCatalogBindingError):
    """The persisted and currently executable tool catalogs differ."""


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
    """Versioned catalog value persisted in immutable State metadata."""

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


def state_tool_catalog(state: State) -> ToolCatalogSnapshot:
    """Read and validate the authoritative catalog persisted on ``state``."""
    raw = state.metadata.environment.get(TOOL_CATALOG_METADATA_KEY)
    if not isinstance(raw, Mapping):
        raise MissingToolCatalogError(
            f"State {state.id!r} revision {state.revision} has no "
            "persisted tool catalog snapshot. Corral will not derive one from "
            "the current Environment. This execution does not implement the "
            "current State protocol and cannot be run."
        )
    try:
        return ToolCatalogSnapshot.model_validate(raw)
    except Exception as exc:
        raise InvalidToolCatalogError(
            f"State {state.id!r} revision {state.revision} has an invalid "
            f"persisted tool catalog snapshot: {exc}"
        ) from exc


def validate_tool_catalog_binding(
    state: State,
    current: ToolCatalogSnapshot,
) -> ToolCatalogSnapshot:
    """Validate and return the State-owned catalog for one Environment binding."""
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
            f"Tool catalog mismatch for State {state.id!r} revision "
            f"{state.revision}: stored fingerprint={stored.fingerprint}, "
            f"current Environment fingerprint={current.fingerprint}; "
            f"stored_only={stored_only}, current_only={current_only}, "
            f"changed={changed}. Refusing "
            "to bind the execution because the agent schema and executable "
            "tools could diverge."
        )
    return stored


__all__ = [
    "TOOL_CATALOG_METADATA_KEY",
    "TOOL_CATALOG_SCHEMA_VERSION",
    "InvalidToolCatalogError",
    "MissingToolCatalogError",
    "ToolCatalogBindingError",
    "ToolCatalogMismatchError",
    "ToolCatalogSnapshot",
    "canonical_tool_catalog_json",
    "state_tool_catalog",
    "tool_catalog_fingerprint",
    "validate_tool_catalog_binding",
]
