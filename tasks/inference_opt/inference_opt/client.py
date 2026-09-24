"""Query student models from trusted environment tools."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

__all__ = ["StudentCompletion", "StudentEndpoint", "endpoint_for", "probe_student"]


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


@dataclass(frozen=True, slots=True)
class StudentCompletion:
    """One student answer, with how it ended."""

    text: str
    #: ``"length"`` means ``max_tokens`` ran out, possibly while still reasoning.
    finish_reason: str | None = None
    #: Reasoning the server returned separately from the answer, if any.
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class StudentEndpoint:
    """One OpenAI-compatible chat-completions endpoint."""

    base_url: str
    model: str = "student"
    #: Seconds per request; ``None`` scales with ``max_tokens`` so long answers
    #: from a loaded server are not cut off.
    timeout: float | None = None
    api_key: str | None = None

    @property
    def completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        n: int = 1,
    ) -> list[str]:
        """Return the text of ``n`` completions, or raise with a message worth reading."""
        return [
            completion.text
            for completion in self.complete(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens, n=n
            )
        ]

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

    def _request(self, body: dict[str, Any]) -> list[StudentCompletion]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        timeout = self.timeout or 120.0 + int(body["max_tokens"]) / 20
        try:
            response = requests.post(
                self.completions_url, json=body, timeout=timeout, headers=headers
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"the student at {self.completions_url} rejected the request "
                f"({exc.response.status_code}): {_server_message(exc.response)}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"could not reach the student at {self.completions_url}: {exc}"
            ) from exc

        payload: dict[str, Any] = response.json()
        choices = payload.get("choices") or []
        return [_completion(choice) for choice in choices] or [StudentCompletion("")]


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
    max_tokens: int = 8192,
    n: int = 1,
    served_name: str | None = None,
) -> list[StudentCompletion]:
    """One ad-hoc query against a student model."""
    endpoint = StudentEndpoint(
        base_url=endpoint_for(model),
        model=served_name or model,
        api_key=os.environ.get("VLLM_API_KEY"),
    )
    return endpoint.complete(
        prompt, system=system, temperature=temperature, max_tokens=max_tokens, n=n
    )
