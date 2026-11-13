"""Responses transport compatibility for Denario's Chat Completions clients.

Keep native SDK messages and execution unchanged. Function tools use Responses
because current reasoning models cannot run them through Chat Completions.
"""

from __future__ import annotations

import json


def attributed_message(message):
    """Represent SDK speaker names in content accepted by every provider."""
    message = dict(message)
    name = message.pop("name", None)
    if name and message["role"] != "tool":
        label = f"[Speaker: {name}]"
        content = message.get("content") or message.get("refusal") or ""
        message["content"] = (
            [{"type": "text", "text": label}, *content]
            if isinstance(content, list)
            else label + ("\n" + content if content else "")
        )
    return message


class ResponsesAdapter:
    def __init__(self):
        # AG2 drops extra message fields. Retain opaque reasoning items beside
        # their call IDs so the next tool result can continue that reasoning.
        self.reasoning_by_call: dict[str, list[dict]] = {}
        self.output_by_call: dict[str, tuple[dict, list[dict]]] = {}

    def request(self, body):
        if body.get("n", 1) != 1:
            raise ValueError("Denario Responses transport requires n=1")
        for key in (
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "logprobs",
            "top_logprobs",
        ):
            if body.get(key) is not None:
                raise ValueError(f"Responses transport does not support {key}")
        items = []
        seen_reasoning = set()
        for message in body["messages"]:
            if message["role"] == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message.get("content") or "",
                    }
                )
                continue
            calls = message.get("tool_calls") or []
            saved = self.output_by_call.get(calls[0]["id"]) if calls else None
            if (
                saved
                and saved[0].get("tool_calls") == calls
                and (saved[0].get("content") or "") == (message.get("content") or "")
            ):
                # Preserve the original item IDs/order as well as opaque
                # reasoning when returning a function result to Responses.
                if message.get("name"):
                    items.append(
                        {
                            "role": message["role"],
                            "content": f"[Speaker: {message['name']}]",
                        }
                    )
                items.extend(saved[1])
                seen_reasoning.update(
                    item["id"] for item in saved[1] if item["type"] == "reasoning"
                )
                continue
            for call in calls:
                for item in self.reasoning_by_call.get(call["id"], []):
                    if item["id"] not in seen_reasoning:
                        items.append(item)
                        seen_reasoning.add(item["id"])
            attributed = attributed_message(message)
            content = attributed.get("content") or attributed.get("refusal")
            if content:
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part["type"] == "text":
                            parts.append({"type": "input_text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": part["image_url"]["url"],
                                    "detail": part["image_url"].get("detail", "auto"),
                                }
                            )
                        else:
                            raise ValueError(
                                f"Unsupported Responses content type: {part['type']}"
                            )
                    content = parts
                items.append({"role": message["role"], "content": content})
            items.extend(
                {"type": "function_call", "call_id": call["id"], **call["function"]}
                for call in calls
            )
        request = {"model": body["model"], "input": items}
        for key in (
            "temperature",
            "top_p",
            "parallel_tool_calls",
            "store",
            "metadata",
            "user",
        ):
            if key in body:
                request[key] = body[key]
        if "reasoning_effort" in body:
            request["reasoning"] = {"effort": body["reasoning_effort"]}
        for key in ("max_tokens", "max_completion_tokens"):
            if key in body:
                request["max_output_tokens"] = body[key]
        if body.get("tools"):
            request["tools"] = [
                {"type": "function", "strict": False, **tool["function"]}
                for tool in body["tools"]
            ]
        choice = body.get("tool_choice")
        if choice is not None:
            request["tool_choice"] = (
                {"type": "function", "name": choice["function"]["name"]}
                if isinstance(choice, dict)
                else choice
            )
        if body.get("response_format"):
            fmt = body["response_format"]
            request["text"] = {
                "format": (
                    {"type": "json_schema", **fmt["json_schema"]}
                    if fmt["type"] == "json_schema"
                    else fmt
                )
            }
        return request

    def completion(self, data):
        """Merge public output into the single choice the native SDK expects."""
        message = {"role": "assistant", "content": ""}
        calls = []
        reasoning = []
        for item in data.get("output", []):
            if item["type"] == "reasoning":
                reasoning.append(item)
            elif item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "output_text":
                        message["content"] += part["text"]
                    elif part["type"] == "refusal":
                        message["refusal"] = (
                            message.get("refusal", "") + part["refusal"]
                        )
            elif item["type"] == "function_call":
                calls.append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }
                )
        if calls:
            message["tool_calls"] = calls
            message["content"] = message["content"] or None
            for call in calls:
                self.reasoning_by_call[call["id"]] = reasoning
                self.output_by_call[call["id"]] = (message, data["output"])
        usage = data.get("usage") or {}
        chat_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
            "completion_tokens_details": usage.get("output_tokens_details") or {},
        }
        finish = (
            "length"
            if data.get("status") == "incomplete"
            else "tool_calls"
            if calls
            else "stop"
        )
        return {
            "id": data["id"],
            "object": "chat.completion",
            "created": int(data.get("created_at", 0)),
            "model": data["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": chat_usage,
        }


def encode_completion(completion, *, streaming):
    if not streaming:
        return "application/json", json.dumps(completion).encode()
    choice = completion["choices"][0]
    delta = dict(choice["message"])
    if delta.get("tool_calls"):
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(delta["tool_calls"])
        ]
    chunk = {
        **completion,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": choice["finish_reason"],
            }
        ],
    }
    return "text/event-stream", (
        "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
    ).encode()
