"""Model-provider boundary for schema-constrained worker calls."""

import base64
import json
import mimetypes
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from corral.agents.schema import BudgetExhaustedError
from corral.agents.utils import LiteLLMMessage
from corral.logging import logger

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class LLMBudgetExceeded(RuntimeError):
    """Raised before a physical request exceeds the local LLM-call budget."""


class StructuredModel(Protocol):
    call_count: int

    def generate(
        self,
        prompt: str,
        response_model: type[ResponseT],
        *,
        model: str | None = None,
        purpose: str = "scientific_worker",
    ) -> ResponseT: ...


def _is_unsupported_structured_output_error(exc: Exception) -> bool:
    """Return whether a provider explicitly rejected structured output.

    Provider SDKs do not expose one stable exception hierarchy across all
    supported LiteLLM versions, so use the dedicated exception name when
    available and tightly classify known 4xx/value errors by parameter text.
    Timeouts, connection failures, 5xx responses, and malformed completions are
    deliberately not treated as capability failures.
    """
    exception_names = {item.__name__ for item in type(exc).__mro__}
    if "UnsupportedParamsError" in exception_names:
        return True
    if not exception_names.intersection(
        {"BadRequestError", "InvalidRequestError", "ValueError"}
    ):
        return False
    status = getattr(exc, "status_code", None)
    if status is not None and not 400 <= status < 500:
        return False
    message = str(exc).casefold()
    parameter_markers = (
        "response_format",
        "response format",
        "structured output",
        "json schema",
    )
    rejection_markers = (
        "not supported",
        "unsupported",
        "does not support",
        "unknown parameter",
        "unrecognized",
        "invalid parameter",
    )
    return any(item in message for item in parameter_markers) and any(
        item in message for item in rejection_markers
    )


def _extract_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^`+[^\n]*\n?", "", content)
        content = re.sub(r"\n?`+\s*$", "", content).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        if start < 0:
            raise ValueError("The model response contained no JSON object.") from None
        value, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(value, dict):
        raise ValueError("The model response must be one JSON object.")
    return value


class LiteLLMStructuredModel:
    """Isolated structured calls with transcript and token accounting."""

    def __init__(
        self,
        *,
        owner: Any,
        default_model: str,
        evaluator_model: str,
        system_prompt: str,
        temperature: float,
        api_endpoint: str | None,
        max_calls: int,
        use_structured_output: bool,
        completion_runner: Callable[..., Any],
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.owner = owner
        self.default_model = default_model
        self.evaluator_model = evaluator_model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.api_endpoint = api_endpoint
        self.max_calls = max_calls
        self.use_structured_output = use_structured_output
        self._completion_runner = completion_runner
        self.llm_kwargs = llm_kwargs or {}
        self.call_count = 0
        self.token_count = 0
        self._state_lock = threading.Lock()

    @property
    def remaining_calls(self) -> int:
        with self._state_lock:
            return max(0, self.max_calls - self.call_count)

    def generate(
        self,
        prompt: str,
        response_model: type[ResponseT],
        *,
        model: str | None = None,
        purpose: str = "scientific_worker",
    ) -> ResponseT:
        messages: list[LiteLLMMessage] = [
            LiteLLMMessage(role="system", content=self.system_prompt),
            LiteLLMMessage(role="user", content=prompt),
        ]
        return self._generate_messages(
            messages,
            response_model,
            model=model,
            purpose=purpose,
        )

    def generate_multimodal(
        self,
        prompt: str,
        response_model: type[ResponseT],
        *,
        image_paths: list[str],
        model: str | None = None,
        purpose: str = "multimodal_scientific_worker",
    ) -> ResponseT:
        """Generate a structured evaluation with local images attached."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for raw_path in image_paths:
            path = Path(raw_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{encoded}",
                        "detail": "high",
                    },
                }
            )
        messages: list[LiteLLMMessage] = [
            LiteLLMMessage(role="system", content=self.system_prompt),
            LiteLLMMessage(role="user", content=content),
        ]
        return self._generate_messages(
            messages,
            response_model,
            model=model,
            purpose=purpose,
        )

    def _generate_messages(
        self,
        messages: list[LiteLLMMessage],
        response_model: type[ResponseT],
        *,
        model: str | None,
        purpose: str,
    ) -> ResponseT:
        self._reserve_request(purpose)
        with self._state_lock:
            use_structured_output = self.use_structured_output
        selected_model = model or self.default_model

        response = None
        if use_structured_output:
            try:
                response = self._completion_runner(
                    model=selected_model,
                    messages=messages,
                    temperature=self.temperature,
                    api_endpoint=self.api_endpoint,
                    response_format=response_model,
                    **self.llm_kwargs,
                )
            except BudgetExhaustedError:
                # Authentication/quota failures are benchmark stop conditions,
                # not evidence that a provider lacks structured output.
                raise
            except Exception as exc:
                if not _is_unsupported_structured_output_error(exc):
                    raise
                logger.warning(
                    "Structured output is unsupported for {} ({}); retrying as JSON text.",
                    purpose,
                    exc,
                )
                with self._state_lock:
                    self.use_structured_output = False

        if response is None:
            if use_structured_output:
                # The fallback is a second physical provider request, so
                # reserve it independently from the rejected structured call.
                self._reserve_request(f"{purpose}_json_fallback")
            response = self._completion_runner(
                model=selected_model,
                messages=messages,
                temperature=self.temperature,
                api_endpoint=self.api_endpoint,
                **self.llm_kwargs,
            )

        content = response.content or ""
        try:
            parsed = response_model.model_validate_json(content.strip())
        except ValidationError:
            parsed = response_model.model_validate(_extract_object(content))

        # The worker contexts are intentionally isolated, but the full sequence
        # remains visible to Corral's normal verbose transcript machinery. Add
        # the same legal `name` to every role in the recorded call so node
        # turns can be grouped without inference from adjacency. These are
        # copies made *after* the provider request: the API-bound `messages`
        # above remain untouched and contain no trace-only fields.
        recorded_messages: list[LiteLLMMessage] = []
        for message in messages:
            recorded_message = message.copy()
            recorded_message["name"] = purpose
            recorded_messages.append(recorded_message)
        with self._state_lock:
            self.owner.messages.extend(
                [
                    *recorded_messages,
                    LiteLLMMessage(
                        role="assistant",
                        content=content,
                        id=getattr(response, "id", None),
                        name=purpose,
                    ),
                ]
            )
            if response.usage:
                self.token_count += int(response.usage.get("total_tokens", 0))
                self.owner.token_usage = response.usage
                self.owner._accumulate_token_usage(response.usage)
            record_usage = getattr(self.owner, "_record_turn_usage", None)
            if callable(record_usage):
                record_usage(dict(response.usage or {}))
        return parsed

    def _reserve_request(self, purpose: str) -> None:
        """Atomically reserve one physical provider request."""
        with self._state_lock:
            if self.call_count >= self.max_calls:
                raise LLMBudgetExceeded(
                    f"LLM-call budget exhausted ({self.max_calls}) before {purpose}."
                )
            self.call_count += 1
