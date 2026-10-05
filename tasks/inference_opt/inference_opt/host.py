"""Runs a submitted policy in the jail; everything goes through the service channel.

Imports what a policy may use at load time: the package is unreadable once the jail is entered.
Locally: ``python -m inference_opt.host <request> <result> <fd>``.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import inference_opt.api  # noqa: F401 - policies import it after the jail closes
from corral.core.tool import Tool, WorkspaceAccess
from corral.runtime import service_channel
from corral.runtime.service_channel import ServiceError
from inference_opt.answers import completion_for
from inference_opt.api import (
    BudgetExhausted,
    LabeledExample,
    Question,
    RunContext,
)
from inference_opt.policy import PolicyError, discover_policy

__all__ = ["PolicyHostTool", "main", "run_policy"]


def _call(body: dict[str, Any]) -> Any:
    try:
        return service_channel.call(body)
    except ServiceError as exc:
        if exc.kind == "BudgetExhausted":
            raise BudgetExhausted(str(exc)) from None
        raise RuntimeError(str(exc)) from None


def _messages(prompt: Any, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        for message in prompt or []:
            if isinstance(message, Mapping):
                messages.append(
                    {
                        "role": str(message.get("role", "user")).lower(),
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                messages.append({"role": "user", "content": str(message)})
    # A request of only system messages is rejected by some servers.
    if not messages or all(message["role"] == "system" for message in messages):
        messages.append({"role": "user", "content": ""})
    return messages


class _Meter:
    """Calls used and left, as the controller last reported them."""

    def __init__(self, budget: int) -> None:
        self.used = 0
        self.remaining = budget
        self._lock = threading.Lock()

    def update(self, count: int, remaining: int) -> None:
        with self._lock:
            self.used += count
            self.remaining = remaining


#: Settings a batch entry may give itself, besides its ``prompt``.
_ENTRY_SETTINGS = (
    "system",
    "temperature",
    "max_tokens",
    "stop",
    "seed",
    "question_id",
    "component",
)


def _is_batch(prompt: Any) -> bool:
    """A list is a batch, unless it is one conversation of role/content messages."""
    if isinstance(prompt, str) or not isinstance(prompt, Sequence):
        return False
    return not prompt or not all(
        isinstance(item, Mapping) and "role" in item for item in prompt
    )


class StudentClient:
    """The metered student: one prompt per call, or a list sent side by side."""

    def __init__(self, meter: _Meter, default_max_tokens: int) -> None:
        self._meter = meter
        self._default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: Any,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
    ) -> str | list[str]:
        defaults = {
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed,
            "question_id": question_id,
            "component": component,
        }
        if not _is_batch(prompt):
            completions, errors = self._send([_request(prompt, defaults)])
            if errors[0]:
                raise RuntimeError(f"student request failed: {errors[0]}")
            return completions[0]
        entries = list(prompt)
        if not entries:
            return []
        completions, _ = self._send([_entry(item, defaults) for item in entries])
        return completions

    def _send(self, requests: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        reply = _call(
            {
                "op": "generate",
                "requests": requests,
                "default_max_tokens": self._default_max_tokens,
            }
        )
        self._meter.update(int(reply["sent"]), int(reply["remaining"]))
        return list(reply["completions"]), list(reply["errors"])

    @property
    def calls_used(self) -> int:
        return self._meter.used

    @property
    def calls_remaining(self) -> int:
        return self._meter.remaining


def _entry(item: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """One batch entry: a prompt, or a dict with ``prompt`` and its own settings."""
    if not isinstance(item, Mapping):
        return _request(item, defaults)
    unknown = set(item) - {"prompt", *_ENTRY_SETTINGS}
    if unknown or "prompt" not in item:
        raise TypeError(
            "a batch entry given as a dict needs a 'prompt' and may set only "
            f"{', '.join(_ENTRY_SETTINGS)}; got {', '.join(sorted(map(str, item)))}"
        )
    settings = {
        **defaults,
        **{key: item[key] for key in _ENTRY_SETTINGS if key in item},
    }
    return _request(item["prompt"], settings)


def _request(prompt: Any, settings: dict[str, Any]) -> dict[str, Any]:
    stop = settings["stop"]
    return {
        "messages": _messages(prompt, settings["system"]),
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "stop": list(stop) if stop else None,
        "seed": settings["seed"],
        "question_id": settings["question_id"],
        "component": settings["component"],
    }


def _question(record: dict[str, Any], index: int, total: int) -> Question:
    answer_format = str(record.get("answer_format", "text"))
    options = record.get("options") or None
    return Question(
        id=str(record["item_id"]),
        text=str(record["question"]),
        benchmark=str(record.get("benchmark", "")),
        answer_type=(
            "mcq"
            if answer_format.startswith("mcq")
            else "numeric"
            if answer_format == "numeric"
            else "text"
        ),
        choices=tuple(str(option) for option in options) if options else None,
        choice_labels=(
            tuple(chr(ord("A") + i) for i in range(len(options))) if options else None
        ),
        topic=record.get("category"),
        index=index,
        total=total,
    )


def _optional(value: Any, kind: type) -> Any:
    return None if value is None else kind(value)


def _example(record: dict[str, Any]) -> LabeledExample:
    return LabeledExample(
        question=_question(record, 0, 1),
        answer=str(record.get("target", "")),
        baseline_completion=_optional(record.get("baseline_completion"), str),
        baseline_correct=_optional(record.get("baseline_correct"), bool),
    )


def run_policy(request: dict[str, Any]) -> dict[str, Any]:
    """Load the policy, run it over every question, and say how it went."""
    try:
        policy = discover_policy(Path(request["policy_dir"]))
    except PolicyError as exc:
        return {"load_error": str(exc)}

    manifest = policy.manifest
    meter = _Meter(int(request["budget"]))
    client = StudentClient(meter, manifest.max_tokens_per_call)

    records = list(request["questions"])
    questions = [
        _question(record, index, len(records)) for index, record in enumerate(records)
    ]
    examples = tuple(_example(record) for record in request.get("examples", []))
    benchmark = str(request.get("benchmark", ""))

    def record(question_id: Any, text: Any) -> None:
        _call(
            {
                "op": "answer",
                "question_id": str(question_id),
                "answer": completion_for(text, benchmark),
            }
        )

    def answer(question_id: Any, text: Any = None) -> None:
        if isinstance(question_id, Mapping):
            if text is not None:
                raise TypeError("pass either answer(question_id, text) or answer(dict)")
            for key, value in question_id.items():
                record(key, value)
        else:
            record(question_id, text)

    def log(text: Any, question_id: str | None = None) -> None:
        _call({"op": "log", "text": str(text)[:500], "question_id": question_id})

    ctx = RunContext(student=client, examples=examples, answer=answer, log=log)
    error = ""
    try:
        returned = policy.run(questions, ctx)
        if returned is not None:
            raise TypeError(
                f"run() returned a {type(returned).__name__}; record answers with "
                "ctx.answer(question_id, text) or ctx.answer({question_id: text})"
            )
    except BudgetExhausted as exc:
        error = f"budget_exhausted: {exc}"
    except Exception:
        error = traceback.format_exc(limit=8)
    return {
        "manifest": {
            "name": manifest.name,
            "version": manifest.version,
            "max_tokens_per_call": manifest.max_tokens_per_call,
        },
        "error": error[-4000:],
    }


class PolicyHostTool(Tool):
    """Internal tool: the jailed side of one policy run."""

    #: The policy reaches the student only through the controller.
    network_access = "none"

    def __init__(self) -> None:
        super().__init__(
            name="_inference_opt_policy_host",
            description="Internal: run a submitted policy inside the jail.",
            params_json_schema={
                "type": "object",
                "properties": {"request": {"type": "string"}},
                "required": ["request"],
            },
            workspace_access=WorkspaceAccess.READ_WRITE,
        )

    def execute(self, **kwargs: Any) -> str:
        return json.dumps(run_policy(json.loads(kwargs["request"])))


def main(argv: list[str] | None = None) -> int:
    request_path, result_path, descriptor = list(sys.argv[1:] if argv is None else argv)
    service_channel.set_channel(int(descriptor))
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    result = run_policy(request)
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
