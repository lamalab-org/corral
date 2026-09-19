"""Provide per-run memory, metering, and student-client access."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import anyio

from inference_opt.api import (
    LabeledExample,
    Memory,
    SetupContext,
    SolveContext,
)
from inference_opt.budget import QuestionAllocator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from inspect_ai.model import Model

    from inference_opt.api import Question

__all__ = [
    "PrimitiveStudentClient",
    "QuestionMeter",
    "RunRuntime",
    "SharedMemory",
    "StudentClientImpl",
]


class SharedMemory(Memory):
    """Cross-question state. Mutating it requires ``memory="shared"``.

    The lock keeps the public memory object safe for policy code that manages its
    own threads, while question execution itself is sequential.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _require(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                """this policy declared memory='none', so cross-question memory is read-only. Set MANIFEST['memory'] = 'shared' to enable it."""
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._require()
        with self._lock:
            self._data[key] = value

    def append(self, key: str, value: Any) -> None:
        self._require()
        with self._lock:
            self._data.setdefault(key, []).append(value)

    def items(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


class QuestionMeter:
    """Charges one question's calls against the run allocator."""

    def __init__(self, allocator: QuestionAllocator, question_id: str) -> None:
        self._allocator = allocator
        self._question_id = question_id
        self._used = 0
        self._tokens = 0
        self._by_component: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def reserve(self, count: int) -> None:
        with self._lock:
            self._allocator.charge(self._question_id, count)
            self._used += count

    def record(self, output_tokens: int, component: str | None) -> None:
        with self._lock:
            self._tokens += max(0, output_tokens)
            bucket = self._by_component.setdefault(
                component or "default", {"calls": 0, "output_tokens": 0}
            )
            bucket["calls"] += 1
            bucket["output_tokens"] += max(0, output_tokens)

    @property
    def used(self) -> int:
        return self._used

    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def by_component(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {name: dict(counts) for name, counts in self._by_component.items()}

    def allowance(self) -> int:
        return self._allocator.allowance(self._question_id)


def _as_messages(prompt: Any, system: str | None) -> list[Any]:
    """Convert policy prompts to Inspect chat messages."""
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
    )

    by_role = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    messages: list[Any] = []
    if system:
        messages.append(ChatMessageSystem(content=str(system)))
    if isinstance(prompt, str):
        messages.append(ChatMessageUser(content=prompt))
    else:
        for message in prompt or []:
            if not isinstance(message, dict):
                messages.append(ChatMessageUser(content=str(message)))
                continue
            role = str(message.get("role", "user")).lower()
            builder = by_role.get(role, ChatMessageUser)
            messages.append(builder(content=str(message.get("content", ""))))
    # A request consisting only of a system message is rejected by some servers.
    if not messages or all(
        isinstance(message, ChatMessageSystem) for message in messages
    ):
        messages.append(ChatMessageUser(content=""))
    return messages


class StudentClientImpl:
    """Expose the metered student model to a policy."""

    def __init__(
        self,
        model: Model,
        meter: QuestionMeter,
        *,
        max_tokens_cap: int,
        in_thread: bool = True,
    ) -> None:
        self._model = model
        self._meter = meter
        self._max_tokens_cap = max_tokens_cap
        self._in_thread = in_thread

    def _run(self, fn: Any, *args: Any) -> Any:
        if self._in_thread:
            return anyio.from_thread.run(fn, *args)
        return anyio.run(fn, *args)

    def _config(self, **overrides: Any) -> Any:
        from inspect_ai.model import GenerateConfig

        overrides["max_tokens"] = min(
            int(overrides.get("max_tokens") or self._max_tokens_cap),
            self._max_tokens_cap,
        )
        return GenerateConfig(**{k: v for k, v in overrides.items() if v is not None})

    def _generate(self, messages: list[dict[str, str]], config: Any) -> Any:
        async def call() -> Any:
            return await self._model.generate(messages, config=config)

        return self._run(call)

    # -- public API -------------------------------------------------------

    def generate(
        self,
        prompt: Any,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        component: str | None = None,
    ) -> str:
        self._meter.reserve(1)
        config = self._config(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_seqs=list(stop) if stop else None,
            seed=seed,
        )
        output = self._generate(_as_messages(prompt, system), config)
        self._meter.record(_output_tokens(output), component)
        return output.completion or ""

    def sample(
        self,
        prompt: Any,
        *,
        n: int,
        temperature: float = 0.8,
        system: str | None = None,
        max_tokens: int = 1024,
        component: str | None = None,
        **_ignored: Any,
    ) -> list[str]:
        """Draw ``n`` completions. Charged as ``n`` calls, not one. ``num_choices=n`` is one HTTP request but ``n`` generations of compute, so billing it as a single call would make self-consistency free."""
        count = max(1, int(n))
        self._meter.reserve(count)
        config = self._config(
            temperature=temperature, max_tokens=max_tokens, num_choices=count
        )
        output = self._generate(_as_messages(prompt, system), config)
        self._meter.record(_output_tokens(output), component)
        completions = [(choice.message.text or "") for choice in (output.choices or [])]
        if not completions:
            completions = [output.completion or ""]
        # Do not issue unmetered fallback requests when a server ignores
        # ``num_choices``. Empty slots preserve the hard call budget.
        completions.extend([""] * max(0, count - len(completions)))
        return completions[:count]

    def batch(
        self,
        prompts: Sequence[Any],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        component: str | None = None,
        **_ignored: Any,
    ) -> list[str]:
        """Complete several prompts sequentially. Charges ``len(prompts)`` calls."""
        items = list(prompts)
        if not items:
            return []
        self._meter.reserve(len(items))
        config = self._config(temperature=temperature, max_tokens=max_tokens)
        results: list[str] = []
        for prompt in items:
            output = self._run(
                self._model.generate, _as_messages(prompt, system), config=config
            )
            results.append(output.completion or "")
            self._meter.record(_output_tokens(output), component)
        return results

    @property
    def calls_used(self) -> int:
        return self._meter.used

    @property
    def calls_remaining(self) -> int:
        return self._meter.allowance()


def _output_tokens(output: Any) -> int:
    usage = getattr(output, "usage", None)
    return int(getattr(usage, "output_tokens", 0) or 0)


class PrimitiveStudentClient:
    """Expose one-call generation and budget access."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def generate(self, prompt: Any, **kwargs: Any) -> str:
        return self._inner.generate(prompt, **kwargs)

    @property
    def calls_used(self) -> int:
        return self._inner.calls_used

    @property
    def calls_remaining(self) -> int:
        return self._inner.calls_remaining


class RunRuntime:
    """Owns the model handle, the allocator, and the artifacts of one run."""

    def __init__(
        self,
        *,
        model: Model,
        questions: int,
        total_calls: int,
        max_calls_per_question: int,
        max_tokens_per_call: int,
        memory_enabled: bool,
        predictions_path: Path,
        benchmark: str = "",
        split: str = "train",
        policy_api: str = "enhanced",
    ) -> None:
        if policy_api not in {"primitive", "enhanced"}:
            raise ValueError(f"unknown policy API mode: {policy_api}")
        self.model = model
        self.benchmark = benchmark
        self.split = split
        self.policy_api = policy_api
        self.max_tokens_per_call = max_tokens_per_call
        self.allocator = QuestionAllocator(
            total_calls=total_calls,
            questions=max(1, questions),
            per_question_cap=max_calls_per_question,
        )
        self.memory = SharedMemory(
            enabled=memory_enabled and policy_api == "enhanced"
        )
        self.predictions_path = predictions_path
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self._predictions: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.exhausted_at: dict[str, Any] | None = None
        self.questions_after_exhaustion = 0
        self.per_component: dict[str, dict[str, int]] = {}
        self.started = time.monotonic()

    # -- contexts ---------------------------------------------------------

    def meter_for(self, question_id: str) -> QuestionMeter:
        return QuestionMeter(self.allocator, question_id)

    def _component_factory(self, holder: dict[str, str]) -> Any:
        @contextmanager
        def component(name: str) -> Iterator[None]:
            previous = holder.get("current")
            holder["current"] = name
            try:
                yield
            finally:
                if previous is None:
                    holder.pop("current", None)
                else:
                    holder["current"] = previous

        return component

    def make_solve_context(
        self, meter: QuestionMeter, artifacts: Path
    ) -> tuple[Any, list[str]]:
        log_lines: list[str] = []
        holder: dict[str, str] = {}
        client: Any = _TaggingClient(
            StudentClientImpl(
                self.model, meter, max_tokens_cap=self.max_tokens_per_call
            ),
            holder,
        )
        if self.policy_api == "primitive":
            client = PrimitiveStudentClient(client)
            return (
                SimpleNamespace(
                    student=client,
                    log=lambda message: log_lines.append(str(message)[:500]),
                    scratch={},
                ),
                log_lines,
            )
        context = SolveContext(
            student=client,
            memory=self.memory if self.policy_api == "enhanced" else None,
            artifacts=artifacts,
            budget_remaining=self.allocator.pool + 0,
            question_budget=meter.allowance(),
            log=lambda message: log_lines.append(str(message)[:500]),
            scratch={},
            component=(
                self._component_factory(holder)
                if self.policy_api == "enhanced"
                else None
            ),
        )
        return context, log_lines

    def make_setup_context(
        self,
        meter: QuestionMeter,
        artifacts: Path,
        examples: tuple[LabeledExample, ...],
        config: dict[str, Any],
    ) -> tuple[Any, list[str]]:
        log_lines: list[str] = []
        client: Any = StudentClientImpl(
            self.model, meter, max_tokens_cap=self.max_tokens_per_call
        )
        if self.policy_api == "primitive":
            client = PrimitiveStudentClient(client)
            return (
                SimpleNamespace(
                    student=client,
                    train_examples=examples,
                    log=lambda message: log_lines.append(str(message)[:500]),
                ),
                log_lines,
            )
        context = SetupContext(
            student=client,
            train_examples=examples,
            memory=self.memory if self.policy_api == "enhanced" else None,
            artifacts=artifacts,
            budget_remaining=meter.allowance(),
            log=lambda message: log_lines.append(str(message)[:500]),
            config=config,
        )
        return context, log_lines

    # -- results ----------------------------------------------------------

    def note_exhaustion(self, question: Question) -> None:
        with self._lock:
            if self.exhausted_at is None:
                self.exhausted_at = {
                    "index": question.index,
                    "question_id": question.id,
                }
            self.questions_after_exhaustion += 1

    def emit_prediction(
        self,
        question: Question,
        answer: str,
        meter: QuestionMeter,
        error: str,
        log_lines: list[str],
        seconds: float,
    ) -> None:
        record = {
            "item_id": question.id,
            "index": question.index,
            "answer": answer,
            "calls": meter.used,
            "output_tokens": meter.tokens,
            "seconds": round(seconds, 3),
            "error": error,
            "log": log_lines[:50],
            "components": meter.by_component,
        }
        with self._lock:
            self._predictions.append(record)
            for name, counts in meter.by_component.items():
                bucket = self.per_component.setdefault(
                    name, {"calls": 0, "output_tokens": 0}
                )
                bucket["calls"] += counts["calls"]
                bucket["output_tokens"] += counts["output_tokens"]

    def flush_predictions(self) -> int:
        with self._lock:
            ordered = sorted(self._predictions, key=lambda row: row["index"])
        with self.predictions_path.open("w", encoding="utf-8") as stream:
            for record in ordered:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        return len(ordered)

    @property
    def predictions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._predictions)

    @property
    def calls_used(self) -> int:
        return self.allocator.used_total

    @property
    def output_tokens(self) -> int:
        with self._lock:
            return sum(row["output_tokens"] for row in self._predictions)


class _TaggingClient:
    """Forwards to the real client, defaulting ``component`` to the active tag."""

    def __init__(self, inner: StudentClientImpl, holder: dict[str, str]) -> None:
        self._inner = inner
        self._holder = holder

    def _tag(self, component: str | None) -> str | None:
        return component or self._holder.get("current")

    def generate(
        self, prompt: Any, *, component: str | None = None, **kwargs: Any
    ) -> str:
        return self._inner.generate(prompt, component=self._tag(component), **kwargs)

    def sample(
        self, prompt: Any, *, component: str | None = None, **kwargs: Any
    ) -> list[str]:
        return self._inner.sample(prompt, component=self._tag(component), **kwargs)

    def batch(
        self, prompts: Any, *, component: str | None = None, **kwargs: Any
    ) -> list[str]:
        return self._inner.batch(prompts, component=self._tag(component), **kwargs)

    @property
    def calls_used(self) -> int:
        return self._inner.calls_used

    @property
    def calls_remaining(self) -> int:
        return self._inner.calls_remaining
