"""Query student models from trusted environment tools."""

from __future__ import annotations

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

__all__ = [
    "StudentCompletion",
    "StudentEndpoint",
    "endpoint_for",
    "probe_student",
    "student_endpoint",
]


def _check_url(url: str, api_key: str | None) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and not api_key:
        return
    if parsed.scheme == "http" and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        return
    if api_key:
        raise ValueError("refusing to send an API key over non-local HTTP")
    raise ValueError(f"unsupported student endpoint scheme: {parsed.scheme or '<none>'}")


def endpoint_for(model: str) -> str:
    """Resolve the base URL for one student model.

    ``CORRAL_VLLM_URL_<MODEL>`` wins over ``CORRAL_VLLM_URL``, so several students
    can be served side by side.
    """
    key = "CORRAL_VLLM_URL_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
    return os.environ.get(
        key, os.environ.get("CORRAL_VLLM_URL", "http://127.0.0.1:8000")
    )


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def student_endpoint(
    model_spec: str,
    base_url: str | None = None,
    timeout: float | None = None,
    options: dict[str, Any] | None = None,
) -> StudentEndpoint:
    """The endpoint that serves one student.

    ``vllm/Qwen/Qwen3-8B`` is requested as ``Qwen/Qwen3-8B`` from ``base_url``,
    with ``VLLM_API_KEY``. ``openrouter/mistralai/ministral-3b-2512@mistral`` is
    sent to OpenRouter with ``OPENROUTER_API_KEY``, and OpenRouter may route it
    only to the provider after ``@``: without fallbacks, a student is the same
    deployment in every run, or the request fails.

    ``options`` are a task's ``student_options`` for this student. Only
    ``{"reasoning": False}`` is defined: it turns a reasoning model's thinking
    off on every call, so a policy cannot turn it back on.
    """
    reasoning = (options or {}).get("reasoning")
    name = model_spec.partition("/")[2] or model_spec
    name, _, provider = name.partition("@")
    if model_spec.startswith("openrouter/"):
        # Tasks bind every student to the vLLM default URL; an OpenRouter
        # student, and its key, only ever go to OpenRouter.
        return StudentEndpoint(
            base_url=OPENROUTER_URL,
            model=name,
            timeout=timeout,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            provider=provider or None,
            reasoning=reasoning,
        )
    if not base_url:
        raise ValueError(f"no endpoint configured for student {model_spec!r}")
    return StudentEndpoint(
        base_url=base_url, model=name, timeout=timeout,
        api_key=os.environ.get("VLLM_API_KEY"), reasoning=reasoning,
    )


@dataclass(frozen=True, slots=True)
class StudentCompletion:
    """One student answer, with how it ended."""

    text: str
    #: ``"length"`` means ``max_tokens`` ran out, possibly while still reasoning.
    finish_reason: str | None = None
    #: Reasoning the server returned separately from the answer, if any.
    reasoning: str = ""
    #: Tokens generated for this completion, reasoning included.
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StudentEndpoint:
    """One OpenAI-compatible chat-completions endpoint."""

    base_url: str
    model: str = "student"
    #: Seconds per request; ``None`` scales with ``max_tokens`` so long answers
    #: from a loaded server are not cut off.
    timeout: float | None = None
    api_key: str | None = None
    #: OpenRouter only: the one provider allowed to serve the request.
    provider: str | None = None
    #: ``False`` turns a reasoning model's thinking off; ``None`` leaves the
    #: model's default.
    reasoning: bool | None = None

    @property
    def completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> list[StudentCompletion]:
        """Return ``n`` completions, each from its own request.

        Some servers reject ``n > 1`` in one request, so several samples are
        separate requests sent side by side.
        """
        _check_url(self.completions_url, self.api_key)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        count = max(1, n)
        if count == 1:
            return self._request(body)
        with ThreadPoolExecutor(max_workers=count) as pool:
            batches = list(pool.map(lambda _: self._request(body), range(count)))
        return [completion for batch in batches for completion in batch]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> StudentCompletion:
        """Complete one conversation in one request."""
        _check_url(self.completions_url, self.api_key)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = seed
        return self._request(body)[0]

    def _request(self, body: dict[str, Any]) -> list[StudentCompletion]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.provider:
            body = {**body, "provider": {"order": [self.provider], "allow_fallbacks": False}}
        if self.reasoning is False:
            # OpenRouter's switch, or vLLM's chat-template switch for Qwen-style models.
            if self.base_url == OPENROUTER_URL:
                body = {**body, "reasoning": {"enabled": False}}
            else:
                body = {**body, "chat_template_kwargs": {"enable_thinking": False}}
        # Assume a slow floor of 5 tokens/s: a busy server can drop well below 20.
        timeout = self.timeout or min(1800.0, 120.0 + int(body["max_tokens"]) / 5)
        payload = self._post_with_retries(body, headers, timeout)
        choices = payload.get("choices") or []
        completions = [_completion(choice) for choice in choices]
        if len(completions) == 1:
            usage = payload.get("usage") or {}
            only = completions[0]
            completions = [
                StudentCompletion(
                    text=only.text,
                    finish_reason=only.finish_reason,
                    reasoning=only.reasoning,
                    output_tokens=int(usage.get("completion_tokens") or 0),
                )
            ]
        return completions or [StudentCompletion("")]

    def _post_with_retries(
        self, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        """POST, retrying rate limits, server errors and dropped connections.

        A hosted provider answers bursts with 429, and OpenRouter reports a
        provider failure as an ``error`` body with no choices, or as a partial
        reply with finish_reason ``error``. Without retries each of these would
        become an empty or cut-off answer, graded wrong.
        """
        for attempt in range(RETRIES + 1):
            last = attempt == RETRIES
            try:
                response = requests.post(
                    self.completions_url, json=body, timeout=timeout, headers=headers
                )
                if response.status_code in RETRY_STATUSES and not last:
                    time.sleep(_backoff(attempt))
                    continue
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"the student at {self.completions_url} rejected the request "
                    f"({exc.response.status_code}): {_server_message(exc.response)}"
                ) from exc
            except requests.RequestException as exc:
                if not last:
                    time.sleep(_backoff(attempt))
                    continue
                raise RuntimeError(
                    f"could not reach the student at {self.completions_url}: {exc}"
                ) from exc
            payload: dict[str, Any] = response.json()
            choices = payload.get("choices") or []
            # OpenRouter reports a provider failing part-way through a reply as
            # finish_reason "error", with the partial text: retry it too.
            failed_midway = bool(choices) and choices[0].get("finish_reason") == "error"
            if (choices and not failed_midway) or (not choices and not payload.get("error")):
                return payload
            if failed_midway and last:
                return payload
            if last:
                raise RuntimeError(
                    f"the student at {self.completions_url} failed: "
                    f"{str(payload['error'])[:300]}"
                )
            time.sleep(_backoff(attempt))
        raise AssertionError("unreachable")


#: Retries after the first attempt; the backoff adds up to about four minutes.
RETRIES = 6
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0 * 2**attempt) * random.uniform(0.75, 1.25)


def _completion(choice: dict[str, Any]) -> StudentCompletion:
    message = choice.get("message") or {}
    # A thinking model that runs out of tokens mid-reasoning returns null content.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return StudentCompletion(
        text=message.get("content") or "",
        finish_reason=choice.get("finish_reason"),
        reasoning=str(reasoning),
    )


def _server_message(response: Any) -> str:
    try:
        error = response.json().get("error")
    except ValueError:
        return response.text[:300]
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    return str(error or response.text)[:300]


def probe_student(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int,
    n: int = 1,
    model_spec: str | None = None,
    options: dict[str, Any] | None = None,
) -> list[StudentCompletion]:
    """One ad-hoc query against a student model; the caller applies the token cap."""
    spec = model_spec or f"vllm/{model}"
    endpoint = student_endpoint(
        spec, None if spec.startswith("openrouter/") else endpoint_for(model),
        options=options,
    )
    return endpoint.complete(
        prompt, system=system, temperature=temperature, max_tokens=max_tokens, n=n
    )
