"""Shared MCP deadlines for tools that declare their own execution limit."""

import math
from typing import Any

# Match the installed Claude Code MCP ceiling (100,000,000 ms, about 27.8 h).
# This is only a fallback; declared tool limits still take precedence.
DEFAULT_TOOL_TIMEOUT_S = 100_000.0
TOOL_TIMEOUT_GRACE_S = 10.0
TIMEOUT_ARGUMENTS = ("timeout", "timeout_seconds", "timeout_s")


def mcp_timeout_prompt(fallback_s: float) -> str:
    """Describe the configured timeout policy to the agent."""
    fallback = f"{fallback_s:g} seconds"
    if fallback_s >= 3600:
        fallback += f" (about {fallback_s / 3600:.1f} hours)"
    return (
        "MCP calls use the tool's timeout argument or declared default, "
        f"plus {TOOL_TIMEOUT_GRACE_S:g} seconds of response grace; "
        f"otherwise the fallback is {fallback}. "
        "Native tools use their own timeout limits. "
    )


def mcp_call_timeout_s(
    arguments: dict[str, Any],
    input_schema: dict[str, Any],
    fallback: float,
    *,
    grace_s: float = TOOL_TIMEOUT_GRACE_S,
) -> float:
    """Use a declared argument or schema default, retaining tool validation."""
    properties = input_schema.get("properties", {})
    for name in TIMEOUT_ARGUMENTS:
        if name not in properties:
            continue
        value = arguments.get(name, properties[name].get("default"))
        if type(value) not in (int, float):
            continue
        try:
            timeout = float(value) + grace_s
        except OverflowError:
            continue
        if value > 0 and math.isfinite(timeout):
            return timeout
    return fallback
