"""Backend request estimates and sanitized size diagnostics.

Structured annotation requests use a general text BPE estimate with 25%
headroom and 256 tokens for provider formatting. UTF-8 bytes are a conservative
fallback when the encoding cannot be loaded; neither is a provider token count.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .context import current_context

BUDGET_VERSION = "request-budget-v1"


class RequestBudgetExceeded(RuntimeError):
    """A different view is required before any further provider requests."""

    def __init__(self, diagnostic: dict[str, Any]):
        self.diagnostic = diagnostic
        self.attempts: list[dict[str, Any]] = []
        self.annotation_view: dict[str, Any] | None = None
        self.diagnostics: list[dict[str, Any]] = []
        super().__init__(
            f"{diagnostic['backend']} request exceeds its context budget "
            f"({diagnostic.get('estimated_tokens', 'unknown')} estimated tokens; "
            f"budget {diagnostic.get('budget_tokens', 0)}; "
            f"reason {diagnostic.get('reason', 'request_budget')})"
        )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def estimator_configuration() -> dict[str, Any]:
    return {
        "version": BUDGET_VERSION,
        "estimator": "cl100k_base-bpe-1.25-plus-256-v1",
        "tiktoken_version": _package_version("tiktoken"),
        "fallback": "utf8-bytes-plus-256-v1",
    }


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_request(body: Any) -> tuple[int, str]:
    text = json.dumps(body, ensure_ascii=False)
    encoding = _encoding()
    if encoding is None:
        return len(text.encode("utf-8")) + 256, "utf8-bytes-plus-256-v1"
    return math.ceil(
        len(encoding.encode(text, disallowed_special=())) * 1.25
    ) + 256, "cl100k_base-bpe-1.25-plus-256-v1"


def record_request(diagnostic: dict[str, Any]) -> None:
    context = current_context()
    if context is not None:
        context.diagnostics.append(
            {"effective_cap": context.view.effective_cap, **diagnostic}
        )
