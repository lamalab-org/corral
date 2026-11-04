"""Public conversation content, excluding private provider reasoning."""

from collections.abc import Mapping
from typing import Any

_PUBLIC_FIELDS = {
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "function_call",
}
_REASONING_BLOCKS = {"thinking", "redacted_thinking", "reasoning"}
EMPTY_RESPONSE_REMINDER = "empty_response_retry"


def public_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Copy public fields; omit empty assistant turns but retain tool-only calls."""
    result = {key: value for key, value in message.items() if key in _PUBLIC_FIELDS}
    if result.get("role") != "assistant":
        return result

    metadata = message.get("metadata") or {}
    content = result.get("content")
    if (
        metadata.get("content_source") == "reasoning_summary"
        or message.get("content_source") == "reasoning_summary"
        or result.get("name") == "thinking"
    ):
        content = None
    elif isinstance(content, list):
        content = [
            dict(block) if isinstance(block, Mapping) else block
            for block in content
            if not isinstance(block, Mapping)
            or (
                block.get("type") not in _REASONING_BLOCKS
                and (
                    block.get("type") not in {"text", "output_text"}
                    or bool(str(block.get("text") or "").strip())
                )
            )
        ]
    if isinstance(content, str) and not content.strip():
        content = None
    if not content and not result.get("tool_calls") and not result.get("function_call"):
        return None
    result["content"] = content
    return result
