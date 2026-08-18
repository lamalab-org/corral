"""Durable actions used by the State v2 agent interface."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from pydantic import Field, JsonValue

from corral.core._immutable import FrozenModel

SUBMIT_ANSWER_TOOL_NAME = "submit_answer"


def new_action_id() -> str:
    """Mint a globally unique action identifier."""
    return str(uuid4())


class Action(FrozenModel):
    """One immutable, JSON-serializable tool call proposed by an agent."""

    name: str = Field(min_length=1)
    arguments: Mapping[str, JsonValue] = Field(default_factory=dict)
    id: str = Field(default_factory=new_action_id, min_length=1)
    actor_id: str | None = Field(default=None, min_length=1)
    content: str | None = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)

    @property
    def is_submission(self) -> bool:
        """Whether this action terminates the task execution with an answer."""
        return self.name == SUBMIT_ANSWER_TOOL_NAME

    def to_tool_call(self) -> dict[str, JsonValue]:
        """Render this action as an OpenAI-compatible tool call."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    dict(self.arguments),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        }

    def to_message(self) -> dict[str, JsonValue]:
        """Render the canonical assistant message that records this action."""
        message: dict[str, JsonValue] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [self.to_tool_call()],
        }
        if self.actor_id is not None:
            message["actor_id"] = self.actor_id
        if self.metadata:
            message["metadata"] = dict(self.metadata)
        return message

    @classmethod
    def from_tool_call(
        cls,
        value: Mapping[str, Any],
        *,
        actor_id: str | None = None,
        content: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Action:
        """Validate a provider-native or flat tool call as an Action."""
        function = value.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = value.get("name")
            arguments = value.get("arguments", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        if not isinstance(arguments, Mapping):
            raise ValueError("action arguments must be a JSON object")
        if not isinstance(name, str) or not name:
            raise ValueError("action tool name must be a non-empty string")

        action_id = value.get("id") or value.get("tool_call_id") or new_action_id()
        return cls(
            id=str(action_id),
            name=name,
            arguments=dict(arguments),
            actor_id=actor_id,
            content=content,
            metadata=dict(metadata or {}),
        )


def submit_answer_action(
    answer: str,
    *,
    actor_id: str | None = None,
    content: str | None = None,
    action_id: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> Action:
    """Build the only action that can complete a task execution."""
    values: dict[str, Any] = {
        "name": SUBMIT_ANSWER_TOOL_NAME,
        "arguments": {"answer": answer},
        "actor_id": actor_id,
        "content": content,
        "metadata": dict(metadata or {}),
    }
    if action_id is not None:
        values["id"] = action_id
    return Action(**values)


def submit_answer_tool() -> dict[str, Any]:
    """Return the strict provider schema for terminal answer submission."""
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_ANSWER_TOOL_NAME,
            "description": (
                "Submit the final answer and end the task. Call this only when "
                "the answer is complete; never return a final answer as plain text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer to submit.",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def with_submit_answer_tool(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy a catalog and install the canonical submission schema exactly once."""
    copied: list[dict[str, Any]] = []
    installed = False
    for raw in tools:
        tool = dict(raw)
        function = tool.get("function")
        name = (
            function.get("name") if isinstance(function, Mapping) else tool.get("name")
        )
        if name == SUBMIT_ANSWER_TOOL_NAME:
            if not installed:
                copied.append(submit_answer_tool())
                installed = True
            continue
        copied.append(tool)
    if not installed:
        copied.append(submit_answer_tool())
    return copied


__all__ = [
    "SUBMIT_ANSWER_TOOL_NAME",
    "Action",
    "new_action_id",
    "submit_answer_action",
    "submit_answer_tool",
    "with_submit_answer_tool",
]
