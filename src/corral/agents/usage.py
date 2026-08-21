"""Shared conversion helpers for agent usage accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from corral.agents.schema import AgentUsage


def usage_field(raw_usage: Any, key: str, default: Any = 0) -> Any:
    """Read one usage field from either an SDK object or a mapping."""
    if isinstance(raw_usage, Mapping):
        return raw_usage.get(key, default)
    return getattr(raw_usage, key, default)


def usage_from_mapping(
    raw_usage: Mapping[str, Any] | None,
    *,
    llm_calls: int = 0,
) -> AgentUsage:
    """Convert canonical or LiteLLM token fields into `AgentUsage`."""
    raw_usage = raw_usage or {}
    return AgentUsage(
        input_tokens=int(
            raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0
        ),
        output_tokens=int(
            raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0
        ),
        reasoning_tokens=int(raw_usage.get("reasoning_tokens", 0) or 0),
        llm_calls=llm_calls,
    )


__all__ = ["usage_field", "usage_from_mapping"]
