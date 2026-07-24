"""Compatibility shim for Claude Opus 4.8 on Bedrock.

Claude Opus 4.8 (`bedrock/us.anthropic.claude-opus-4-8`, via litellm) is stricter
than DeepSeek v3.2 in several ways that break corral's agents out of the box.
This module monkeypatches `litellm.completion` to absorb those differences so the
shared `src/corral` source stays untouched and the report run scripts remain
drop-in compatible with the DeepSeek/Kimi run.py convention.

Each run.py does `import _opus_shim  # noqa: F401` BEFORE building the agent.

Fixes applied for Opus 4.8 calls (matched on model id containing "opus-4-8"):

1. TEMPERATURE / TOP_P  — Opus 4.8 rejects these ("temperature is deprecated for
   this model"). Corral always forwards `temperature`; `litellm.drop_params` does
   NOT help (rejection is server-side at Bedrock). We strip them.

2. NETWORK SAFETY NET   — inject `timeout` + `num_retries`. Without a timeout a
   dead Bedrock SSL socket can hang `_ssl__SSLSocket_read` forever (observed in
   smoke testing: a toolcalling trial stuck at 0% CPU for an hour).

3. TOOL JSON SCHEMA      — corral's class-based tools (e.g. the shared filesystem
   tools and resistor_network) declare parameter types with Python names
   ("str", "bool", "int", "list[str]", ...) instead of JSON Schema names. Opus
   4.8 validates tool schemas against JSON Schema draft 2020-12 and rejects them
   ("tools.0.custom.input_schema: JSON schema is invalid"). DeepSeek tolerated
   them. We normalize the `tools` payload in place.

4. ASSISTANT-MESSAGE PREFILL — Bedrock Anthropic requires the conversation to end
   with a user/tool message ("This model does not support assistant message
   prefill. The conversation must end with a user message."). Corral's
   ToolCallingAgent can leave a trailing assistant message after an iteration
   error. We append a minimal user nudge when the last message is an assistant
   turn so the request is well-formed.
"""

import socket

import litellm

MODEL = "bedrock/us.anthropic.claude-opus-4-8"

# Process-wide socket read timeout. The litellm `timeout` kwarg does NOT arm the
# underlying Bedrock SSL socket: when AWS credentials expire mid-flight (e.g. an
# SSO token lapsing), in-progress `_ssl__SSLSocket_read` calls block forever in
# the kernel `poll()` at 0% CPU. Setting the default socket timeout forces any
# blocking read to raise `socket.timeout`, which surfaces as a litellm error and
# is picked up by corral's tenacity retry wrapper. Observed hang: 16 jobs wedged
# for ~40 min after an SSO token expired. 600s is comfortably longer than any
# legitimate Opus completion while still bounding a dead connection.
_SOCKET_TIMEOUT = 600

_DEPRECATED_FOR_OPUS48 = ("temperature", "top_p")

# Per-call network safety net for Opus 4.8 on Bedrock.
_REQUEST_TIMEOUT = 600  # seconds for a single completion before it's aborted
_NUM_RETRIES = 4        # litellm-level retries on timeout / transient errors

# Python type-name -> JSON Schema type-name.
_PY_TO_JSON_TYPE = {
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
    "none": "string",
    "nonetype": "string",
}


def _normalize_type(value):
    """Map a Python-style type string to a JSON Schema type. Handle list[str]."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if raw.startswith("list[") or raw.startswith("tuple["):
        return "array"
    if raw.startswith("dict["):
        return "object"
    return _PY_TO_JSON_TYPE.get(raw)


def _fix_schema(node):
    """Recursively rewrite invalid JSON-schema `type` values in place."""
    if isinstance(node, dict):
        if "type" in node:
            mapped = _normalize_type(node["type"])
            if mapped is not None:
                node["type"] = mapped
        for key in ("properties", "$defs", "definitions"):
            if isinstance(node.get(key), dict):
                for sub in node[key].values():
                    _fix_schema(sub)
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                _fix_schema(node[key])
        for key in ("anyOf", "allOf", "oneOf"):
            if isinstance(node.get(key), list):
                for sub in node[key]:
                    _fix_schema(sub)
    elif isinstance(node, list):
        for sub in node:
            _fix_schema(sub)


def _fix_tools(tools):
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        params = fn.get("parameters") or fn.get("input_schema")
        if isinstance(params, dict):
            _fix_schema(params)


def _fix_messages(messages):
    """Ensure the conversation does not end on an assistant message (Bedrock
    Anthropic forbids assistant-message prefill)."""
    if not isinstance(messages, list) or not messages:
        return messages
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        # Return a copy so we don't mutate the agent's persistent history.
        return list(messages) + [
            {
                "role": "user",
                "content": "Continue. Provide your next tool call or your Final Answer.",
            }
        ]
    return messages


def _install() -> None:
    if getattr(litellm, "_opus48_shim_installed", False):
        return

    # Force litellm's Bedrock path onto the httpx transport instead of aiohttp.
    # litellm defaults to an aiohttp transport for Bedrock, and aiohttp does NOT
    # honor the per-request `timeout` kwarg (nor socket.setdefaulttimeout) for an
    # in-flight read — so a stalled Bedrock connection hangs forever in
    # `_ssl__SSLSocket_read`. The httpx transport DOES honor `timeout`, turning a
    # stall into a retryable error. This is the actual fix for the recurring hang;
    # the socket-default below is a belt-and-suspenders backstop.
    litellm.disable_aiohttp_transport = True

    # Arm a global socket timeout so dead Bedrock SSL reads can't hang forever.
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(_SOCKET_TIMEOUT)

    _orig_completion = litellm.completion

    def _patched_completion(*args, **kwargs):
        model = kwargs.get("model", "")
        if "claude-opus-4-8" in model or "opus-4-8" in model:
            for key in _DEPRECATED_FOR_OPUS48:
                kwargs.pop(key, None)
            kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
            kwargs.setdefault("num_retries", _NUM_RETRIES)
            if "tools" in kwargs:
                _fix_tools(kwargs["tools"])
            if "messages" in kwargs:
                kwargs["messages"] = _fix_messages(kwargs["messages"])
        return _orig_completion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm._opus48_shim_installed = True


_install()
