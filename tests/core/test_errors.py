"""Verify useful messages are preserved across nested exception chains."""

import httpx
import pytest

from corral.core.errors import concise_error_message


class ProviderError(RuntimeError):
    """Represent a provider error with optional structured details."""

    def __init__(self, message: str, *, body: dict | None = None) -> None:
        """Attach the response body to the exception."""
        super().__init__(message)
        self.body = body


class CustomHTTPStatusError(httpx.HTTPStatusError):
    """Represent an application subclass of the public HTTPX exception."""


def test_concise_error_message_uses_leaf_exception_only():
    """Keep the leaf message for ordinary exception chains."""
    try:
        raise ValueError("response schema is invalid")
    except ValueError as cause:
        error = RuntimeError("model request failed")
        error.__cause__ = cause

    assert concise_error_message(error) == "response schema is invalid"


def test_concise_error_message_prefers_structured_provider_detail():
    """Prefer structured provider details over HTTP status text."""
    error = ProviderError(
        "litellm.BadRequestError: request failed through provider adapter",
        body={"error": {"message": "Required field 'answer' is missing."}},
    )
    request = httpx.Request("GET", "https://example.test")
    error.__cause__ = httpx.HTTPStatusError(
        "HTTP 400", request=request, response=httpx.Response(400, request=request)
    )

    assert concise_error_message(error) == "Required field 'answer' is missing."


@pytest.mark.parametrize("error_type", [httpx.HTTPStatusError, CustomHTTPStatusError])
def test_concise_error_message_uses_outer_message_for_httpx_status_error(error_type):
    """Preserve context-limit details above real HTTPX errors and subclasses."""
    request = httpx.Request("GET", "https://example.test")
    error = RuntimeError("maximum context length is 65536 tokens")
    error.__cause__ = RuntimeError("request failed")
    error.__cause__.__cause__ = error_type(
        "Client error '400 Bad Request' for url 'https://example.test'",
        request=request,
        response=httpx.Response(400, request=request),
    )

    assert concise_error_message(error) == "maximum context length is 65536 tokens"


def test_concise_error_message_collapses_multiline_text():
    """Render multiline messages on one line for reports."""
    error = RuntimeError("first line\n  final detail")

    assert concise_error_message(error) == "first line final detail"


def test_concise_error_message_reads_http_provider_response():
    import httpx

    message = "You have reached your specified API usage limits."
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://provider.test/v1/messages"),
        json={"error": {"type": "invalid_request_error", "message": message}},
    )
    error = httpx.HTTPStatusError(
        "HTTP 400", request=response.request, response=response
    )
    assert concise_error_message(error) == message


def test_concise_error_message_keeps_non_json_http_error():
    import httpx

    response = httpx.Response(
        502,
        request=httpx.Request("POST", "https://provider.test/v1/messages"),
        text="gateway failure",
    )
    error = httpx.HTTPStatusError(
        "HTTP 502", request=response.request, response=response
    )
    assert concise_error_message(error) == "HTTP 502"
