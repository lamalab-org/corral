"""Query the student directly from the controller-side probe tool."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests

__all__ = ["StudentEndpoint", "endpoint_for", "probe_student"]


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
class StudentEndpoint:
    """One OpenAI-compatible chat-completions endpoint."""

    base_url: str
    model: str = "student"
    timeout: float = 120.0
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
        """Return ``n`` completions, or raise with a message worth reading."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.post(
                self.completions_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "n": max(1, n),
                },
                timeout=self.timeout,
                headers=headers,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"could not reach the student at {self.completions_url}: {exc}"
            ) from exc

        payload: dict[str, Any] = response.json()
        choices = payload.get("choices") or []
        return [str(choice["message"]["content"]) for choice in choices] or [""]


def probe_student(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    n: int = 1,
    served_name: str | None = None,
) -> list[str]:
    """One ad-hoc query against a student model."""
    endpoint = StudentEndpoint(
        base_url=endpoint_for(model),
        model=served_name or model,
        api_key=os.environ.get("VLLM_API_KEY"),
    )
    return endpoint.generate(
        prompt, system=system, temperature=temperature, max_tokens=max_tokens, n=n
    )
