"""Runs a submitted policy in the jail; everything goes through the service channel.

Imports what a policy may use at load time: the package is unreadable once the
jail is entered. Locally: ``python -m inference_opt.host <request> <result> <fd>``.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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

#: Questions answered side by side when a policy defines only ``solve()``.
SOLVE_CONCURRENCY = 16


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


class EnhancedStudentClient:
    """The student, for enhanced tasks: one call, several samples, or a batch."""

    def __init__(
        self, meter: _Meter, default_max_tokens: int, question_id: str | None = None
    ) -> None:
        self._meter = meter
        self._default_max_tokens = default_max_tokens
        self._question_id = question_id

    def for_question(self, question_id: str) -> EnhancedStudentClient:
        return type(self)(self._meter, self._default_max_tokens, question_id)

    def _send(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int | None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
    ) -> list[str]:
        reply = _call(
            {
                "op": "generate",
                "conversations": conversations,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "default_max_tokens": self._default_max_tokens,
                "stop": list(stop) if stop else None,
                "seed": seed,
                "question_id": question_id or self._question_id,
                "component": component,
            }
        )
        self._meter.update(len(conversations), int(reply["remaining"]))
        return list(reply["completions"])

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
    ) -> str:
        return self._send(
            [_messages(prompt, system)],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            question_id=question_id,
            component=component,
        )[0]

    def sample(
        self,
        prompt: Any,
        *,
        n: int,
        temperature: float = 0.8,
        system: str | None = None,
        max_tokens: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
        **_ignored: Any,
    ) -> list[str]:
        messages = _messages(prompt, system)
        return self._send(
            [messages] * max(1, int(n)),
            temperature=temperature,
            max_tokens=max_tokens,
            question_id=question_id,
            component=component,
        )

    def batch(
        self,
        prompts: Sequence[Any],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        question_id: str | None = None,
        component: str | None = None,
        **_ignored: Any,
    ) -> list[str]:
        items = list(prompts)
        if not items:
            return []
        return self._send(
            [_messages(prompt, system) for prompt in items],
            temperature=temperature,
            max_tokens=max_tokens,
            question_id=question_id,
            component=component,
        )

    @property
    def calls_used(self) -> int:
        return self._meter.used

    @property
    def calls_remaining(self) -> int:
        return self._meter.remaining


class PrimitiveStudentClient:
    """The student, for primitive tasks: one completion per call."""

    def __init__(self, inner: EnhancedStudentClient) -> None:
        self._inner = inner

    def for_question(self, question_id: str) -> PrimitiveStudentClient:
        return type(self)(self._inner.for_question(question_id))

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        return self._inner.generate(prompt, **kwargs)

    @property
    def calls_used(self) -> int:
        return self._inner.calls_used

    @property
    def calls_remaining(self) -> int:
        return self._inner.calls_remaining


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


def _example(record: dict[str, Any]) -> LabeledExample:
    return LabeledExample(
        question=_question(record, 0, 1),
        answer=str(record.get("target", "")),
        baseline_completion=str(record.get("baseline_completion", "")),
        baseline_correct=bool(record.get("baseline_correct", False)),
    )


def _report_error(question_id: str, text: str) -> None:
    # A closed channel already ends the run on the controller side.
    with suppress(Exception):
        _call({"op": "error", "question_id": question_id, "text": text})


def run_policy(request: dict[str, Any]) -> dict[str, Any]:
    """Load the policy, run it over every question, and say how it went."""
    try:
        policy = discover_policy(Path(request["policy_dir"]))
    except PolicyError as exc:
        return {"load_error": str(exc)}

    manifest = policy.manifest
    meter = _Meter(int(request["budget"]))
    client: Any = EnhancedStudentClient(meter, manifest.max_tokens_per_call)
    if request.get("policy_api", "primitive") == "primitive":
        client = PrimitiveStudentClient(client)

    records = list(request["questions"])
    questions = [
        _question(record, index, len(records)) for index, record in enumerate(records)
    ]
    examples = tuple(_example(record) for record in request.get("examples", []))
    benchmark = str(request.get("benchmark", ""))

    def submit(question_id: str, answer: Any) -> None:
        _call(
            {
                "op": "submit",
                "question_id": str(question_id),
                "answer": completion_for(answer, benchmark),
            }
        )

    def log(text: Any, question_id: str | None = None) -> None:
        _call({"op": "log", "text": str(text)[:500], "question_id": question_id})

    ctx = RunContext(student=client, examples=examples, submit=submit, log=log)
    error = ""
    try:
        if policy.runs_whole_set:
            returned = policy.run(questions, ctx)
            if isinstance(returned, Mapping):
                for question_id, answer in returned.items():
                    submit(str(question_id), answer)
        else:
            _solve_each(policy, questions, ctx, client)
    except BudgetExhausted as exc:
        error = f"budget_exhausted: {exc}"
    except Exception:
        error = traceback.format_exc(limit=8)
    return {
        "manifest": {
            "name": manifest.name,
            "version": manifest.version,
            "max_tokens_per_call": manifest.max_tokens_per_call,
            "runs_whole_set": policy.runs_whole_set,
        },
        "error": error[-4000:],
    }


def _solve_each(policy: Any, questions: list[Question], ctx: RunContext, client: Any) -> None:
    """The default ``run()``: ``solve()`` per question, side by side."""

    def one(question: Question) -> None:
        def log(text: Any, question_id: str | None = None) -> None:
            ctx.log(text, question_id or question.id)

        question_ctx = RunContext(
            student=client.for_question(question.id),
            examples=ctx.examples,
            submit=ctx.submit,
            log=log,
        )
        try:
            ctx.submit(question.id, policy.solve(question, question_ctx))
        except BudgetExhausted as exc:
            _report_error(question.id, f"budget_exhausted: {exc}")
        except Exception as exc:  # a policy fault costs this question, not the run
            _report_error(
                question.id,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            )

    with ThreadPoolExecutor(max_workers=SOLVE_CONCURRENCY) as pool:
        list(pool.map(one, questions))


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
