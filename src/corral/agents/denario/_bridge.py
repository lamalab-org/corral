"""Worker-local transport for Denario's unmodified model and tool clients.

The model gateway preserves tool/structured-output payloads and buffers SSE so
the session durably records each response before native agents can act on it.
It is not an agent loop: Denario still chooses every request and next action.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

import httpx

from corral.agents.denario._responses import (
    ResponsesAdapter,
    attributed_message,
    encode_completion,
)


class EnvironmentToolError(RuntimeError):
    def __init__(self, response):
        super().__init__(response["error"])
        self.status = response["status"]


def completion_event(body: bytes, *, streaming: bool) -> tuple[dict, dict]:
    """Extract public assistant output and usage from Chat Completions."""
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    usage = {}
    calls: dict[int, dict] = {}
    if not streaming:
        data = json.loads(body)
        choices = data.get("choices") or []
        if choices:
            raw = choices[0]["message"]
            message.update(
                {
                    key: raw[key]
                    for key in (
                        "content",
                        "tool_calls",
                        "function_call",
                        "refusal",
                        "audio",
                        "annotations",
                        "reasoning",
                        "reasoning_content",
                        "reasoning_details",
                    )
                    if key in raw
                }
            )
        usage = data.get("usage") or {}
    else:
        for line in body.decode().splitlines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            data = json.loads(line[5:])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                message["content"] += delta.get("content") or ""
                if delta.get("refusal"):
                    message["refusal"] = message.get("refusal", "") + delta["refusal"]
                if delta.get("function_call"):
                    function = message.setdefault(
                        "function_call", {"name": "", "arguments": ""}
                    )
                    for key in ("name", "arguments"):
                        function[key] += delta["function_call"].get(key) or ""
                for piece in delta.get("tool_calls") or []:
                    call = calls.setdefault(
                        piece["index"],
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    call["id"] += piece.get("id") or ""
                    for key in ("name", "arguments"):
                        call["function"][key] += (
                            piece.get("function", {}).get(key) or ""
                        )
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
    usage = {
        **usage,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ),
    }
    return message, usage


class Bridge:
    """One authenticated loopback endpoint and serialized parent RPC channel."""

    def __init__(self, config, reader, writer):
        self.config = config
        self.reader, self.writer = reader, writer
        self.lock = threading.Lock()
        self.phase = "starting"
        self.agent = ""
        self.responses = ResponsesAdapter()
        self.chat_messages: dict[str, dict] = {}
        self.provider = config.get("provider", "openai")
        self.client_model = config["model"] if self.provider == "openai" else "gpt-4o"
        self.token = secrets.token_urlsafe(32)
        defaults = {
            "openai": (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ),
            "anthropic": (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "https://api.anthropic.com/v1",
            ),
            "openrouter": (
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
        }
        key_var, url_var, default_url = defaults[self.provider]
        self.upstream = (
            config.get("api_endpoint") or os.environ.get(url_var) or default_url
        ).rstrip("/")
        self.api_key = os.environ.get(key_var, "")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def rpc(self, kind, **fields):
        with self.lock:
            self.writer.write(
                json.dumps({"kind": kind, **fields}, ensure_ascii=False, default=str)
                + "\n"
            )
            self.writer.flush()
            line = self.reader.readline()
            if not line:
                raise EOFError("Corral session closed")
            return json.loads(line)["result"]

    def set_phase(self, phase):
        self.phase = phase
        self.rpc("phase", phase=phase)

    def event(self, event_type, **fields):
        self.rpc("event", event={"type": event_type, **fields})

    def tool(self, name, arguments, call_id=None):
        response = self.rpc(
            "tool", name=name, arguments=arguments, id=call_id or str(uuid4())
        )
        if not response["success"]:
            raise EnvironmentToolError(response)
        return response["result"]

    def model(self, body):
        body = dict(body)
        body["model"] = self.config["model"]
        # Old CMBagent injects sampling/reasoning defaults for older models.
        # An omitted Corral option must remain omitted at the provider boundary.
        for key in (
            "temperature",
            "top_p",
            "reasoning_effort",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "frequency_penalty",
            "presence_penalty",
        ):
            body.pop(key, None)
        body.update(
            {
                key: value
                for key, value in self.config["model_settings"].items()
                if value is not None
            }
        )
        choice = body.get("tool_choice")
        no_forced_tools = self.provider == "anthropic" and body["model"].startswith(
            "claude-opus-5-5"
        )
        if no_forced_tools and (isinstance(choice, dict) or choice == "required"):
            # This model rejects forced tools, including the plan setter's
            # single-tool request. Keep its tool and instruction, using the
            # supported automatic selection mode without disabling reasoning.
            name = (
                choice.get("function", {}).get("name")
                if isinstance(choice, dict)
                else None
            )
            if name and not any(
                t.get("function", {}).get("name") == name for t in body.get("tools", [])
            ):
                raise ValueError(f"Forced tool is not present in request: {name}")
            instruction = (
                f"Call the {name} tool now."
                if name
                else "Call at least one of the supplied tools now."
            )
            body["messages"] = [
                *body["messages"],
                {"role": "user", "content": instruction},
            ]
            body["tool_choice"] = "auto"
            self.event(
                "tool_choice_compatibility",
                model=body["model"],
                requested=choice,
                applied="auto",
                instruction=instruction,
            )
        if (
            self.provider == "openrouter"
            and body["model"] == "moonshotai/kimi-k3.1-pro"
            and "parallel_tool_calls" in body
        ):
            # CMBagent injects this flag, but Kimi's OpenRouter endpoints do
            # not advertise it. With require_parameters it prevents routing.
            # Preserve the SDK's serial-call preference in the prompt, while
            # retaining every returned call and the original tool_choice.
            parallel = body.pop("parallel_tool_calls")
            instruction = None
            if parallel is False:
                instruction = (
                    "Make at most one tool call in this response. "
                    "Wait for its result before choosing the next tool."
                )
                body["messages"] = [
                    *body["messages"],
                    {"role": "user", "content": instruction},
                ]
            self.event(
                "parallel_tool_calls_compatibility",
                model=body["model"],
                requested=parallel,
                applied="omitted",
                instruction=instruction,
            )
        if body.get("stream"):
            body["stream_options"] = {
                **body.get("stream_options", {}),
                "include_usage": True,
            }
        call_id = str(uuid4())
        agent = self.agent or self.phase
        responses = (
            self.provider == "openai"
            and body["model"].startswith("gpt-6")
            and (
                bool(body.get("tools"))
                or any(
                    m.get("tool_calls") or m.get("role") == "tool"
                    for m in body["messages"]
                )
            )
        )
        endpoint = "responses" if responses else "chat/completions"
        request = self.responses.request(body) if responses else body
        streaming = bool(body.get("stream"))
        if self.provider == "openrouter":
            # Buffer the upstream response so reasoning continuation fields and
            # provider metadata survive older SDKs that discard unknown fields.
            request = {**request, "stream": False}
            request.pop("stream_options", None)
            request["messages"] = [
                self._restore_chat_message(m) for m in body["messages"]
            ]
            request["provider"] = dict(self.config.get("openrouter_provider") or {})
            if self.config.get("openrouter_zdr"):
                # Apply last: neither the SDK nor caller routing preferences
                # may weaken a requested zero-retention policy.
                request["provider"].update(zdr=True, data_collection="deny")
        if not responses:
            request = {
                **request,
                "messages": [attributed_message(m) for m in request["messages"]],
            }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "anthropic" and os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
        self.rpc(
            "model_start", id=call_id, agent=agent, request=request, endpoint=endpoint
        )
        try:
            with httpx.Client(timeout=1200) as client:
                response = client.post(
                    f"{self.upstream}/{endpoint}",
                    json=request,
                    headers=headers,
                )
            error_code = None
            provider_response = None
            content_type = response.headers.get("content-type", "application/json")
            content = response.content
            failed = not response.is_success
            if response.is_success:
                try:
                    if responses:
                        provider_response = response.json()
                        if provider_response.get("error"):
                            raise ValueError(str(provider_response["error"]))
                        completion = self.responses.completion(provider_response)
                        content_type, content = encode_completion(
                            completion, streaming=bool(body.get("stream"))
                        )
                    if self.provider == "openrouter":
                        provider_response = response.json()
                        if provider_response.get("error"):
                            raise ValueError(str(provider_response["error"]))
                        message, usage = completion_event(content, streaming=False)
                        for call in message.get("tool_calls") or []:
                            self.chat_messages[call["id"]] = message
                        content_type, content = encode_completion(
                            provider_response, streaming=streaming
                        )
                    else:
                        message, usage = completion_event(content, streaming=streaming)
                except (ValueError, KeyError, TypeError) as exc:
                    failed = True
                    message = {
                        "role": "assistant",
                        "content": f"Invalid provider response: {type(exc).__name__}: {exc}",
                    }
                    usage = {}
            else:
                usage = {}
                message = {
                    "role": "assistant",
                    "content": f"Provider HTTP {response.status_code}: {response.text}",
                }
                with contextlib.suppress(ValueError, AttributeError):
                    error_code = response.json().get("error", {}).get("code")
            self.rpc(
                "model_end",
                id=call_id,
                agent=agent,
                input=request.get("input", request.get("messages", body["messages"])),
                message=message,
                usage=usage,
                error_code=error_code,
                is_error=failed,
                status_code=response.status_code,
                provider_response=provider_response,
            )
            return (
                response.status_code,
                content_type,
                content,
            )
        except httpx.HTTPError as exc:
            self.rpc(
                "model_end",
                id=call_id,
                agent=agent,
                input=request.get("input", request.get("messages", body["messages"])),
                message={
                    "role": "assistant",
                    "content": f"Provider transport failed: {type(exc).__name__}",
                },
                usage={},
                is_error=True,
            )
            return (
                502,
                "application/json",
                json.dumps(
                    {
                        "error": {
                            "message": "Upstream transport failed",
                            "type": "api_error",
                        }
                    }
                ).encode(),
            )

    def _restore_chat_message(self, message):
        """Replay provider reasoning alongside the unchanged tool-call turn."""
        for call in message.get("tool_calls") or []:
            original = self.chat_messages.get(call["id"])
            if (
                original
                and original.get("tool_calls") == message["tool_calls"]
                and (original.get("content") or "") == (message.get("content") or "")
            ):
                return {**message, **original}
        return message

    def _handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                    self.send_error(403)
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    if self.path == "/v1/chat/completions":
                        status, content_type, result = bridge.model(body)
                    else:
                        self.send_error(404)
                        return
                except Exception as exc:
                    status, content_type = 500, "application/json"
                    result = json.dumps({"error": str(exc)}).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

        return Handler

    def start(self):
        self.thread.start()
        os.environ["OPENAI_BASE_URL"] = self.url + "/v1"
        os.environ["OPENAI_API_KEY"] = self.token

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
