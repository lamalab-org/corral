"""Small, serialization-safe exception projections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _provider_message(exc: BaseException) -> str | None:
    """Return a provider's structured error message when one is available."""
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return None

    error: Any = body.get("error", body)
    if isinstance(error, Mapping):
        message = error.get("message")
    elif isinstance(error, str):
        message = error
    else:
        message = body.get("message")
    return message.strip() if isinstance(message, str) and message.strip() else None


def concise_error_message(exc: BaseException) -> str:
    """Format the useful final exception without traceback or wrapper layers.

    Explicit causes (and unsuppressed implicit contexts) are followed to the
    leaf exception. Provider exceptions are a special case: their structured
    response body usually contains a clearer message than either the SDK
    wrapper or its low-level HTTP cause.
    """
    chain = [exc]
    seen = {id(exc)}
    current = exc
    while True:
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        chain.append(cause)
        current = cause

    selected = chain[-1]
    message = None
    for candidate in reversed(chain):
        provider_message = _provider_message(candidate)
        if provider_message is not None:
            selected = candidate
            message = provider_message
            break

    if message is None:
        message = str(selected).strip()
    # Projection metadata is intended for reports and JSON, not traceback rendering.
    # Keep embedded multi-line SDK text readable as one exception-only message.
    message = " ".join(message.split())
    return message or type(selected).__name__


__all__ = ["concise_error_message"]
