"""Shared SDK wait limit; tools enforce their own execution limits."""

DEFAULT_TOOL_TIMEOUT_S = 10_000.0


def mcp_timeout_prompt(timeout_s: float) -> str:
    """Describe the configured timeout policy to the agent."""
    limit = f"{timeout_s:g} seconds"
    if timeout_s >= 3600:
        limit += f" (about {timeout_s / 3600:.1f} hours)"
    return (
        f"The SDK waits up to {limit} for each MCP call.\n"
        "A tool's own timeout or default can stop it sooner; tool arguments do not extend the SDK wait limit.\n"
        "Native tools use their own timeout limits.\n"
    )
