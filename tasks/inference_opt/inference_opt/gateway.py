"""Controller-side handler for a policy run: meters student calls, records answers.

Requests come from agent code, so every field is validated here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from inference_opt.answers import MAX_COMPLETION_CHARS
from inference_opt.api import BudgetExhausted
from inference_opt.client import StudentCompletion, StudentEndpoint

__all__ = ["Backend", "MockBackend", "StudentGateway", "backend_for"]

_ROLES = frozenset({"system", "user", "assistant"})
_MAX_LOG_LINES = 50
_MAX_RUN_LOG_LINES = 200
_MAX_PROMPT_CHARS = 200_000
#: Completions of one request sent side by side; the rest wait their turn.
_MAX_PARALLEL = 16

#: ``backend(messages, temperature, max_tokens, stop, seed)`` -> one completion.
Backend = Callable[..., StudentCompletion]


@dataclass(frozen=True, slots=True)
class MockBackend:
    """A student that needs no server, for tests and offline dry runs."""

    reply: str = "Default output from mockllm"

    def __call__(self, messages: list[dict[str, str]], **_: Any) -> StudentCompletion:
        del messages
        return StudentCompletion(text=self.reply, finish_reason="stop", output_tokens=1)


def backend_for(
    model_spec: str, base_url: str | None, api_key: str | None, timeout: float | None
) -> Backend:
    """``mockllm/...`` answers locally; anything else is a served model name."""
    if model_spec.startswith("mockllm/"):
        return MockBackend()
    if not base_url:
        raise ValueError(f"no endpoint configured for student {model_spec!r}")
    endpoint = StudentEndpoint(
        base_url=base_url,
        # "vllm/Qwen/Qwen3-8B" is served as "Qwen/Qwen3-8B".
        model=model_spec.partition("/")[2] or model_spec,
        api_key=api_key,
        timeout=timeout,
    )
    return endpoint.chat


class StudentGateway:
    """Meter, forward and record every request of one policy run."""

    def __init__(
        self,
        *,
        backend: Backend,
        budget: int,
        max_tokens_cap: int,
        question_ids: list[str],
        policy_api: str = "primitive",
        retries: int = 2,
    ) -> None:
        self._backend = backend
        self.budget = max(0, int(budget))
        self.max_tokens_cap = max(1, int(max_tokens_cap))
        self.question_ids = list(question_ids)
        self._known = frozenset(self.question_ids)
        self.policy_api = policy_api
        self._retries = max(0, retries)
        self._lock = threading.Lock()

        self.used = 0
        self.output_tokens = 0
        self.backend_failures = 0
        self.backend_successes = 0
        self.exhausted = False
        self.answers: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.logs: dict[str, list[str]] = {}
        self.calls: list[dict[str, Any]] = []
        self.per_question: dict[str, dict[str, int]] = {}

    # -- the service handler ----------------------------------------------

    def handle(self, body: Any) -> Any:
        if not isinstance(body, dict):
            raise ValueError("a request must be a JSON object")
        op = body.get("op")
        if op == "generate":
            return self._generate(body)
        if op == "submit":
            return self._submit(body)
        if op == "log":
            return self._log(body)
        if op == "error":
            return self._error(body)
        raise ValueError(f"unknown request {op!r}")

    def _question(self, value: Any) -> str | None:
        if value is None:
            return None
        question_id = str(value)
        if question_id not in self._known:
            raise ValueError(f"{question_id!r} is not a question in this run")
        return question_id

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        conversations = body.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("generate needs a non-empty list of conversations")
        if self.policy_api == "primitive" and len(conversations) != 1:
            raise ValueError("this task allows one completion per call")
        cleaned = [_conversation(messages) for messages in conversations]
        question_id = self._question(body.get("question_id"))
        component = body.get("component")
        component = str(component)[:64] if component else None
        temperature = min(2.0, max(0.0, float(body.get("temperature") or 0.0)))
        requested = body.get("max_tokens") or body.get("default_max_tokens")
        max_tokens = max(1, min(int(requested or self.max_tokens_cap), self.max_tokens_cap))
        stop = body.get("stop") or None
        if stop is not None:
            if not isinstance(stop, list) or len(stop) > 4:
                raise ValueError("stop must be a list of at most 4 strings")
            stop = [str(item)[:100] for item in stop]
        seed = body.get("seed")
        seed = int(seed) if seed is not None else None

        count = len(cleaned)
        with self._lock:
            if self.used + count > self.budget:
                self.exhausted = True
                raise BudgetExhausted(
                    f"run call budget exhausted ({self.used}/{self.budget} used, "
                    f"{count} more requested)"
                )
            self.used += count
            if question_id is not None:
                bucket = self.per_question.setdefault(
                    question_id, {"calls": 0, "output_tokens": 0}
                )
                bucket["calls"] += count

        def one(messages: list[dict[str, str]]) -> StudentCompletion:
            return self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed,
                question_id=question_id,
                component=component,
            )

        if count == 1:
            completions = [one(cleaned[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(count, _MAX_PARALLEL)) as pool:
                completions = list(pool.map(one, cleaned))
        with self._lock:
            remaining = self.budget - self.used
        return {
            "completions": [completion.text for completion in completions],
            "finish_reasons": [completion.finish_reason for completion in completions],
            "remaining": remaining,
        }

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        question_id: str | None,
        component: str | None,
        **params: Any,
    ) -> StudentCompletion:
        started = time.monotonic()
        error = ""
        completion = StudentCompletion("")
        for attempt in range(self._retries + 1):
            try:
                completion = self._backend(messages, **params)
                error = ""
                break
            except Exception as exc:  # recorded, then reported to the policy
                error = f"{type(exc).__name__}: {exc}"[:500]
                if attempt < self._retries:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
        with self._lock:
            if error:
                self.backend_failures += 1
            else:
                self.backend_successes += 1
                self.output_tokens += completion.output_tokens
                if question_id is not None:
                    self.per_question[question_id]["output_tokens"] += (
                        completion.output_tokens
                    )
            self.calls.append(
                {
                    "question_id": question_id,
                    "component": component,
                    "messages": messages,
                    "params": params,
                    "completion": completion.text,
                    "finish_reason": completion.finish_reason,
                    "output_tokens": completion.output_tokens,
                    "seconds": round(time.monotonic() - started, 3),
                    "error": error,
                }
            )
        if error:
            raise RuntimeError(f"student request failed: {error}")
        return completion

    def _submit(self, body: dict[str, Any]) -> dict[str, Any]:
        question_id = self._question(body.get("question_id"))
        if question_id is None:
            raise ValueError("submit needs a question_id")
        answer = str(body.get("answer", ""))[:MAX_COMPLETION_CHARS]
        with self._lock:
            self.answers[question_id] = answer
            self.errors.pop(question_id, None)
        return {"accepted": True}

    def _log(self, body: dict[str, Any]) -> dict[str, Any]:
        question_id = self._question(body.get("question_id"))
        key = question_id or "_run"
        limit = _MAX_RUN_LOG_LINES if question_id is None else _MAX_LOG_LINES
        with self._lock:
            lines = self.logs.setdefault(key, [])
            if len(lines) < limit:
                lines.append(str(body.get("text", ""))[:500])
        return {"accepted": True}

    def _error(self, body: dict[str, Any]) -> dict[str, Any]:
        question_id = self._question(body.get("question_id"))
        if question_id is None:
            raise ValueError("error needs a question_id")
        with self._lock:
            if question_id not in self.answers:
                self.errors[question_id] = str(body.get("text", ""))[:2000]
        return {"accepted": True}


def _conversation(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("a conversation must be a non-empty list of messages")
    cleaned = []
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object with role and content")
        role = str(message.get("role", "user")).lower()
        if role not in _ROLES:
            raise ValueError(f"unsupported message role {role!r}")
        content = str(message.get("content", ""))
        total += len(content)
        cleaned.append({"role": role, "content": content})
    if total > _MAX_PROMPT_CHARS:
        raise ValueError(f"a prompt may hold at most {_MAX_PROMPT_CHARS:,} characters")
    return cleaned
